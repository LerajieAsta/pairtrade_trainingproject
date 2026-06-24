import os
import sys
import json
import time
import re
import traceback
import threading
import multiprocessing
import concurrent.futures
import sqlite3
import importlib
import gc
import copy
import inspect
from datetime import datetime
import numpy as np
import pandas as pd

from strategies.preprocess_equity import DataProcessor
from strategies.db_utils import init_formation_db, get_db_connection
from strategies.config import (
    FORCE_RERUN, CPU_LIMIT_PCT, DB_PROFILES, DB_PATH, TABLE_NAME, INFO_TABLE,
    TICKER_COL, SECTOR_COL, BACKTEST_START, BACKTEST_END,
    FORMATION_WINDOW, FORWARD_DAYS, rolling_step,
    base_params, hdbscan_common, strategies_raw,
    ProgressAwareStdout, draw_dashboard,
    start_dashboard_thread, interleave_strategies, print_summary_report,
    _DASHBOARD_FIXED_LINES
)

# ════════════════════════════════════════════════════════════════════════════
# 斷點續傳 (Smart Resume) 工具函數
# ════════════════════════════════════════════════════════════════════════════

def check_formation_completed(config: dict, db_path: str, log_dir: str, formation_db_path: str) -> bool:
    """
    智慧判定策略形成期是否已完整跑完（斷點續傳）。
    """
    if FORCE_RERUN:
        return False
        
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in config["name"])
    mark_path = os.path.join(log_dir, f"{safe_name}_completed.json")

    if not os.path.exists(mark_path):
        return False

    try:
        with open(mark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. 價格資料庫路徑與 mtime
        if data.get("db_path") != db_path:
            return False
        current_db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        if abs(data.get("db_mtime", -1.0) - current_db_mtime) > 1.0:
            return False

        # 2. 策略參數
        if data.get("strategy_params") != config["params"]:
            return False

        # 3. 檢查 SQLite 中確實存有該策略的數據
        if not os.path.exists(formation_db_path):
            return False

        with get_db_connection(formation_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='formation_pairs';"
            )
            if cursor.fetchone()[0] == 0:
                return False

            cursor.execute(
                "SELECT count(*) FROM formation_pairs WHERE strategy_id = ?;",
                (config["name"],),
            )
            if cursor.fetchone()[0] == 0:
                return False

        return True

    except Exception:
        return False


def write_formation_completion_mark(config: dict, db_path: str, log_dir: str) -> None:
    """
    配對形成成功後寫入標記檔。
    """
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in config["name"])
    mark_path = os.path.join(log_dir, f"{safe_name}_completed.json")
    try:
        db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        info = {
            "db_path":         db_path,
            "db_mtime":        db_mtime,
            "strategy_params": config["params"],
            "completed_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(log_dir, exist_ok=True)
        with open(mark_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
    except Exception as e:
        sys.stderr.write(f"\n⚠️ 寫入完成標記檔失敗 ({config['name']}): {e}\n")




# ════════════════════════════════════════════════════════════════════════════
# 子行程工作單元 (Worker Task)
# ════════════════════════════════════════════════════════════════════════════

def worker_task(
    strategy_config: dict,
    price_pivot,
    all_dates,
    total_days,
    local_first_trade_idx,
    sector_mapping,
    progress_dict,
    df_memberships=None,
) -> dict:
    """
    子行程工作單元：執行單一策略形成期的滾動視窗配對篩選，並將結果暫存至獨立 SQLite 中。
    """
    progress_stream = None
    name          = strategy_config["name"]
    module_name   = strategy_config["formation_module"]
    params        = strategy_config["params"]
    log_path      = strategy_config["log_path"]
    temp_db_path  = strategy_config["temp_db_path"]
    db_path       = strategy_config["db_path"]
    log_dir       = strategy_config["log_dir"]

    # 參數解構
    FORWARD_DAYS     = params.get("trading_window", 126)
    rolling_step     = params.get("rolling_step", 21)
    FORMATION_WINDOW = params.get("formation_window", 252)

    # 預先計算滾動期數
    roll_start_indices = list(range(local_first_trade_idx, total_days - FORWARD_DAYS + 1, rolling_step))
    last = total_days - FORWARD_DAYS
    if roll_start_indices[-1] != last:
        roll_start_indices.append(last)
    total_windows = len(roll_start_indices)

    progress_dict[name] = {
        "status":   "RUNNING",
        "progress": f"0/{total_windows}",
        "pct":      0,
        "msg":      "正在初始化策略模組...",
        "elapsed":  0.0,
    }

    start_time  = time.time()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # 攔截 Stdout 輸出至日誌，避免洗版
    progress_stream = ProgressAwareStdout(log_path, progress_dict, name, total_windows, pattern_keyword="Window")
    sys.stdout = progress_stream
    sys.stderr = progress_stream

    try:
        # 1. 初始化該行程的暫存資料庫
        from strategies.db_utils import init_formation_db
        os.makedirs(os.path.dirname(temp_db_path), exist_ok=True)
        conn = init_formation_db(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM formation_pairs")
        conn.commit()

        # 2. 載入形成期模組
        strat_module = importlib.import_module(module_name)
        FormationClass = strat_module.Formation

        sig = inspect.signature(FormationClass.__init__)
        valid_param_names = set(sig.parameters.keys())

        # 3. 滾動視窗計算
        # 預先查詢已完成的期數 (實現期數級別的斷點續傳)
        completed_periods = set()
        main_db = strategy_config.get("formation_db_path")
        if main_db and os.path.exists(main_db) and not FORCE_RERUN:
            try:
                main_conn = get_db_connection(main_db)
                df_exists = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name='formation_pairs'", main_conn)
                if not df_exists.empty:
                    df_completed = pd.read_sql_query(
                        "SELECT DISTINCT Period_Start FROM formation_pairs WHERE strategy_id = ?",
                        main_conn, params=(name,)
                    )
                    completed_periods = set(df_completed["Period_Start"].tolist())
                main_conn.close()
            except Exception:
                pass

        for i, idx in enumerate(roll_start_indices):
            form_start_idx = idx - FORMATION_WINDOW
            form_end_idx   = idx - 1
            trade_end_idx  = min(idx + FORWARD_DAYS - 1, total_days - 1)

            form_start_dt  = all_dates[form_start_idx].strftime('%Y-%m-%d')
            form_end_dt    = all_dates[form_end_idx].strftime('%Y-%m-%d')
            trade_start_dt = all_dates[idx].strftime('%Y-%m-%d')
            trade_end_dt   = all_dates[trade_end_idx].strftime('%Y-%m-%d')

            # 輸出進度，供攔截器分析
            if form_start_dt in completed_periods and not FORCE_RERUN:
                print(f"Window {i+1}/{total_windows}: {form_start_dt} to {form_end_dt} -> \033[90m已完成，跳過\033[0m")
                continue
            
            print(f"Window {i+1}/{total_windows}: {form_start_dt} to {form_end_dt}")

            # 整理形成期數據與過濾停牌股票
            form_prices = price_pivot.iloc[form_start_idx:idx]

            # ── 動態成分股過濾 (Dynamic S&P 500 Constituents Filtering) ─────────────────
            # 目的：僅保留在當前滾動形成期結束日 (form_end_dt) 真實處於標普 500 的成分股。
            # 防止使用已退市股票（例如 TIE 在 2012 年退市後的脏數據）或尚未上市/入選成分股的標的。
            if df_memberships is not None and not df_memberships.empty:
                active_df = df_memberships[
                    (df_memberships['start_date'] <= form_end_dt) & 
                    ((df_memberships['end_date'].isna()) | (df_memberships['end_date'] >= form_end_dt))
                ]
                active_symbols = set(active_df['Symbol'].unique())
                valid_cols = [c for c in form_prices.columns if c in active_symbols]
                form_prices = form_prices[valid_cols]

            form_prices = form_prices.dropna(axis=1)

            # 動態匹配建構子參數
            kwargs = {
                "price_df":       form_prices,
                "form_start":     form_start_dt,
                "form_end":       form_end_dt,
                "top_n":          params.get("top_n", 10),
                "sector_mapping": sector_mapping
            }
            for k, v in params.items():
                if k not in kwargs:
                    kwargs[k] = v

            # 依據 Init 簽章篩選合法參數
            valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_param_names}

            formation_instance = FormationClass(**valid_kwargs)
            pairs_df = formation_instance.run()

            if pairs_df.empty:
                continue

            records = []
            for rank, row in pairs_df.iterrows():
                ticker_a = row["Ticker_A"]
                ticker_b = row["Ticker_B"]
                sector_a = sector_mapping.get(ticker_a, "Unknown")
                sector_b = sector_mapping.get(ticker_b, "Unknown")

                param_dict = {}
                for col in pairs_df.columns:
                    if col not in ["Ticker_A", "Ticker_B", "Sector", "Rank"]:
                        val = row[col]
                        if pd.isna(val):
                            val = None
                        elif isinstance(val, (np.integer, np.floating)):
                            val = val.item()
                        param_dict[col] = val

                records.append((
                    name,
                    form_start_dt,
                    trade_start_dt,
                    trade_end_dt,
                    ticker_a,
                    ticker_b,
                    sector_a,
                    sector_b,
                    int(row.get("Rank", rank)),
                    json.dumps(param_dict)
                ))

            cursor.executemany("""
                INSERT OR REPLACE INTO formation_pairs 
                (strategy_id, Period_Start, Trade_Start, Trade_End, Ticker_A, Ticker_B, Sector_A, Sector_B, Pair_Rank, Formation_Params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

        conn.close()

        # 4. 寫入該策略的完成標記檔
        write_formation_completion_mark(strategy_config, db_path, log_dir)

        elapsed = time.time() - start_time
        progress_dict[name] = {
            "status":   "SUCCESS",
            "progress": "完成",
            "pct":      100,
            "msg":      f"配對形成計算成功！共跑完 {total_windows} 期",
            "elapsed":  elapsed,
        }
        return {"name": name, "status": "SUCCESS", "skipped": False, "elapsed": elapsed, "error": None}

    except Exception as e:
        err_msg = traceback.format_exc()
        sys.stderr.write(f"\n❌ [ERROR] 策略: {name} 執行失敗！\n{err_msg}\n")

        elapsed = time.time() - start_time
        progress_dict[name] = {
            "status":   "FAILED",
            "progress": "失敗",
            "pct":      100,
            "msg":      f"❌ 失敗: {str(e)}",
            "elapsed":  elapsed,
        }
        return {"name": name, "status": "FAILED", "skipped": False, "elapsed": elapsed, "error": str(e)}

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if progress_stream is not None:
            try:
                progress_stream.close()
            except Exception:
                pass
        import gc
        gc.collect()





# ════════════════════════════════════════════════════════════════════════════
# 暫存資料庫合併
# ════════════════════════════════════════════════════════════════════════════

def merge_databases(main_db_path: str, temp_db_paths: list):
    """
    將各行程獨立產出的配對暫存 SQLite 數據合併至最終主資料庫，並刪除暫存檔。
    """
    print("\n[DB Merge] 正在整合所有並行行程的配對數據至主資料庫...", flush=True)
    from strategies.db_utils import init_formation_db
    conn = init_formation_db(main_db_path)
    cursor = conn.cursor()

    for temp_db in temp_db_paths:
        if not os.path.exists(temp_db):
            continue
        try:
            temp_conn = get_db_connection(temp_db)
            temp_cursor = temp_conn.cursor()
            temp_cursor.execute("SELECT strategy_id, Period_Start, Trade_Start, Trade_End, Ticker_A, Ticker_B, Sector_A, Sector_B, Pair_Rank, Formation_Params FROM formation_pairs")
            rows = temp_cursor.fetchall()
            temp_conn.close()

            # 直接插入，OR REPLACE 會處理重複
            cursor.executemany("""
                INSERT OR REPLACE INTO formation_pairs 
                (strategy_id, Period_Start, Trade_Start, Trade_End, Ticker_A, Ticker_B, Sector_A, Sector_B, Pair_Rank, Formation_Params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
        except Exception as e:
            print(f"⚠️ [DB Merge] 整合暫存資料庫 {temp_db} 失敗: {e}", flush=True)

        # 清除暫存檔
        try:
            os.remove(temp_db)
        except Exception:
            pass

    conn.close()
    print("✅ [DB Merge] 所有配對數據整合與清理完畢！", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# 主控引擎
# ════════════════════════════════════════════════════════════════════════════

def run_all_formations():
    print("=" * 80, flush=True)
    print("      🚀 形成期平行化控制主程式 (High-Performance Parallel Engine) 🚀", flush=True)
    print("=" * 80, flush=True)

    db_basename = os.path.splitext(os.path.basename(DB_PATH))[0]
    formation_db_path = f"formation_data/formation_pairs_{db_basename}.db"

    # 搜尋是否有吻合的 profile 來決定日誌目錄
    matched_profile = None
    for name, prof in DB_PROFILES.items():
        if os.path.normcase(os.path.abspath(prof["db_path"])) == os.path.normcase(os.path.abspath(DB_PATH)):
            matched_profile = prof
            break

    if matched_profile:
        output_root = matched_profile["output_root"]
    else:
        output_root = f"./results/{db_basename}"

    # 日誌目錄
    log_dir = f"{output_root}/logs/run_formation"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(formation_db_path), exist_ok=True)

    # 自動清理舊有殘留的暫存資料庫檔案 (防範先前因中斷、強退而殘留)
    import glob
    old_temps = glob.glob(f"formation_data/formation_pairs_{db_basename}_*.db*")
    for old_temp in old_temps:
        try:
            os.remove(old_temp)
        except Exception:
            pass

    # 1. 載入歷史價格 (只在主行程載入一次，並傳遞給各子行程)
    print(f"📊 正在從資料庫載入與前處理價格數據 '{DB_PATH}'...", flush=True)
    start_io  = time.time()
    processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    sector_mapping = processor.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)

    try:
        price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
            BACKTEST_START, BACKTEST_END, FORMATION_WINDOW
        )
        io_elapsed = time.time() - start_io
        print(
            f"✅ 數據載入成功！"
            f"歷史天數: {total_days} 天 | "
            f"標的數: {price_pivot.shape[1]} 檔 | "
            f"耗時: {io_elapsed:.2f}s\n",
            flush=True,
        )
    except Exception as e:
        print(f"❌ [嚴重錯誤] 無法加載配對形成所需數據：{e}", flush=True)
        sys.exit(1)

    # 1.5 載入標普 500 歷史成員變動紀錄 (index_memberships)
    print("⏳ 正在從資料庫載入標普 500 歷史成員變動紀錄 (index_memberships)...", flush=True)
    try:
        with get_db_connection(DB_PATH) as conn:
            df_memberships = pd.read_sql_query("SELECT Symbol, start_date, end_date FROM index_memberships", conn)
        print(f"✅ 成功載入成員紀錄，共 {len(df_memberships)} 筆數據。\n", flush=True)
    except Exception as e:
        print(f"⚠️ 無法載入 index_memberships 表，回退至全歷史標的池。錯誤: {e}\n", flush=True)
        df_memberships = None

    # 2. 斷點續傳篩選
    print(f"🔍 正在進行參數網格展開與任務篩選（斷點續傳檢查）...", flush=True)
    
    # ── 展開策略網格（形成期配對數固定為 20，不受 top_n_list 影響，僅依據 max_sector_ratio_list 展開） ──
    expanded_strategies_raw = []
    for raw in strategies_raw:
        params = raw["params"]
        msr_list = params.get("max_sector_ratio_list")
        if not msr_list:
            msr_list = [params.get("max_sector_ratio", 0.0)]
        elif not isinstance(msr_list, list):
            msr_list = [msr_list]
            
        for msr in msr_list:
            new_raw = copy.deepcopy(raw)
            new_params = new_raw["params"]
            
            # 形成期固定篩選 20 對配對
            new_params["top_n"] = 20
            new_params["max_sector_ratio"] = msr
            
            # 清理 list 參數以防混淆
            new_params.pop("top_n_list", None)
            new_params.pop("stop_loss_list", None)
            new_params.pop("stop_loss_pct_list", None)
            new_params.pop("max_sector_ratio_list", None)
            
            # 加後綴以區分 MSR 參數組合
            msr_pct = int(msr * 100)
            new_raw["name"] = f"{raw['name']}_MSR{msr_pct}"
            expanded_strategies_raw.append(new_raw)

    # 建立未展開前的原始策略配置列表，用於儀表板統合呈現
    original_strategies_config = []
    for raw in strategies_raw:
        original_strategies_config.append({
            "name":             raw["name"],
            "formation_module": raw["formation_module"],
            "params":           raw["params"],
        })

    manager       = multiprocessing.Manager()
    progress_dict = manager.dict()
    results       = []
    strategies_to_run = []
    strategies_config = []

    for raw in expanded_strategies_raw:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw["name"])
        config = {
            "name":             raw["name"],
            "formation_module": raw["formation_module"],
            "params":           raw["params"],
            "log_path":         f"{log_dir}/{safe_name}.log",
            "temp_db_path":     f"formation_data/formation_pairs_{db_basename}_{safe_name}.db",
            "db_path":          DB_PATH,
            "log_dir":          log_dir,
            "formation_db_path": formation_db_path,
        }
        strategies_config.append(config)

        if check_formation_completed(config, DB_PATH, log_dir, formation_db_path):
            progress_dict[config["name"]] = {
                "status":   "SUCCESS",
                "progress": "完成",
                "pct":      100,
                "msg":      "✨ 已跳過 (偵測到已有完整形成結果)",
                "elapsed":  0.0,
            }
            results.append({
                "name":    config["name"],
                "status":  "SUCCESS",
                "skipped": True,
                "elapsed": 0.0,
                "error":   None,
            })
            print(f"  ⟳ 跳過（已完成）：{config['name']}", flush=True)
        else:
            progress_dict[config["name"]] = {
                "status":   "PENDING",
                "progress": "0/0",
                "pct":      0,
                "msg":      "排隊等待中...",
                "elapsed":  0.0,
            }
            strategies_to_run.append(config)
            print(f"  ● 排入執行：{config['name']}", flush=True)

    # ── 交錯打散任務順序 (Interleave) ──────────────────────────────────────────
    # 將相同原始策略的子任務交錯排列，以確保多核心並行時，不同策略能同時啟動並呈現於儀表板
    if strategies_to_run:
        strategies_to_run = interleave_strategies(strategies_to_run, original_strategies_config)

    # 3. 決定並行行程數 (根據 CPU_LIMIT_PCT 限制 CPU 使用率)
    max_cores = max(1, int((os.cpu_count() or 4) * CPU_LIMIT_PCT))
    max_workers = min(len(strategies_to_run), max_cores)


    if not strategies_to_run:
        print("\n✨ 所有策略形成期配對數據皆已存在且最新，無須重新計算！", flush=True)
        return

    # ── 啟動儀表板 ───────────────────────────────────────────────────────
    os.system("")  # Windows console 支持
    placeholder_lines = len(original_strategies_config) + _DASHBOARD_FIXED_LINES
    sys.stdout.write("\n" * placeholder_lines)
    sys.stdout.flush()
    time.sleep(0.05)

    main_start_time = time.time()
    stop_event      = threading.Event()

    dashboard_thread = start_dashboard_thread(progress_dict, original_strategies_config, main_start_time, log_dir, stop_event, stage_title="形成")

    # 4. 多行程並行運算
    try:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for config in strategies_to_run:
                    f = executor.submit(
                        worker_task,
                        config,
                        price_pivot,
                        all_dates,
                        total_days,
                        local_first_trade_idx,
                        sector_mapping,
                        progress_dict,
                        df_memberships,
                    )
                    futures[f] = config

                for f in concurrent.futures.as_completed(futures):
                    config = futures[f]
                    try:
                        res = f.result()
                        results.append(res)
                    except Exception as exc:
                        results.append({
                            "name":    config["name"],
                            "status":  "FAILED",
                            "skipped": False,
                            "elapsed": 0.0,
                            "error":   str(exc),
                        })
        finally:
            # 停止儀表板並重繪最終狀態
            stop_event.set()
            dashboard_thread.join(timeout=1.0)
            draw_dashboard(progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir, stage_title="形成")
    finally:
        # 5. 合併與清理暫存資料庫 (即使中斷執行，也會將已完成策略合併並清除暫存檔)
        temp_db_paths = [cfg["temp_db_path"] for cfg in strategies_to_run]
        merge_databases(formation_db_path, temp_db_paths)

    # 6. 終端總結報告
    total_elapsed = time.time() - main_start_time
    print_summary_report(results, original_strategies_config, total_elapsed, show_equity=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_all_formations()
