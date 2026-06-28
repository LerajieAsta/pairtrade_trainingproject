import os
import sys
import json
import sqlite3
import inspect
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
import itertools
import copy
import importlib
import time
import concurrent.futures
import threading
import re
import traceback

from strategies.preprocess_equity import DataProcessor
from strategies.portfolio_manager import PortfolioManager
from strategies.db_utils import get_db_connection
from strategies.config import (
    INITIAL_CAPITAL,
    FORCE_RERUN, CPU_LIMIT_PCT, DB_PROFILES, DB_PATH, TABLE_NAME, INFO_TABLE,
    TICKER_COL, SECTOR_COL, BACKTEST_START, BACKTEST_END,
    FORMATION_WINDOW, FORWARD_DAYS, rolling_step,
    base_params, hdbscan_common, strategies_raw,
    ProgressAwareStdout, draw_dashboard,
    start_dashboard_thread, interleave_strategies, print_summary_report,
    _DASHBOARD_FIXED_LINES
)


# ════════════════════════════════════════════════════════════════════════════
# 斷點續傳檢查
# ════════════════════════════════════════════════════════════════════════════

def _build_filename(params: dict) -> str:
    top_n = params.get("top_n", 10)
    sl    = int(params.get("stop_loss_pct", 0.0) * 100)
    zwin  = params.get("zscore_window", 0)
    msr   = int(params.get("max_sector_ratio", 0.0) * 100)
    return f"TradeLogs_Top{top_n}_SL{sl}_ZWin{zwin}_MSR{msr}.csv"

def check_trading_completed(strategy_config: dict, output_root: str, results_db_path: str = "", dataset_name: str = "") -> bool:
    if FORCE_RERUN:
        return False

    sub_dir  = strategy_config["sub_dir"]
    params   = strategy_config["params"]
    filename = _build_filename(params)
    csv_path = os.path.join(output_root, sub_dir, filename)

    if not os.path.exists(csv_path):
        return False

    # Also verify the result is registered in result.db to catch stale CSV
    if results_db_path and os.path.exists(results_db_path) and dataset_name:
        try:
            path_key = f"{dataset_name.lower()}/{sub_dir}/{filename}"
            with sqlite3.connect(results_db_path, timeout=5.0) as _conn:
                row = _conn.execute(
                    "SELECT 1 FROM strategy_summaries WHERE _path = ?", (path_key,)
                ).fetchone()
                return row is not None
        except Exception:
            pass

    return True


# ════════════════════════════════════════════════════════════════════════════
# 子行程工作單元 (Worker Task)
# ════════════════════════════════════════════════════════════════════════════

def worker_task(
    strategy_config: dict,
    price_pivot,
    all_dates,
    total_days: int,
    local_first_trade_idx: int,
    sector_mapping: dict,
    formation_db_path: str,
    output_root: str,
    db_path: str,
    dataset_name: str,
    progress_dict,
) -> dict:
    name          = strategy_config["name"]
    module_name   = strategy_config["trading_module"]
    params        = strategy_config["params"]
    sub_dir       = strategy_config["sub_dir"]
    
    # 建立日誌與 CSV 輸出路徑
    safe_name     = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    log_dir       = f"{output_root}/logs/run_trading"
    log_path      = f"{log_dir}/{safe_name}.log"

    # 進度檔初始化
    progress_dict[name] = {
        "status":   "RUNNING",
        "progress": "0/0",
        "pct":      0,
        "msg":      "正在初始化交易期模組...",
        "elapsed":  0.0,
    }

    start_time  = time.time()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    progress_stream = None # 函數開頭初始化，防範 finally 內 NameError

    try:
        # 提前建立並攔截 stdout / stderr (total_periods 先傳入預設值 0)
        progress_stream = ProgressAwareStdout(log_path, progress_dict, name, 0, pattern_keyword="Period")
        sys.stdout = progress_stream
        sys.stderr = progress_stream

        # 預先建立日期到索引的 map，優化 O(n) 線性搜尋為 O(1) 查找
        date_to_idx = {pd.to_datetime(d): i for i, d in enumerate(all_dates)}

        # 載入交易策略模組
        strat_module = importlib.import_module(module_name)
        if hasattr(strat_module, 'Trading'):
            TradingClass = strat_module.Trading
        else:
            raise AttributeError(f"Strategy {module_name} missing Trading class.")

        conn = get_db_connection(formation_db_path)
        
        # 讀取該策略的所有交易週期 (使用 formation_strategy_id)
        formation_strategy_id = strategy_config.get("formation_strategy_id", name)
        df_periods = pd.read_sql_query(
            "SELECT DISTINCT Period_Start, Trade_Start, Trade_End FROM formation_pairs WHERE strategy_id = ? ORDER BY Period_Start", 
            conn, 
            params=(formation_strategy_id,)
        )
        total_periods = len(df_periods)
        
        if total_periods == 0:
            conn.close()
            raise ValueError(f"No periods found in formation_pairs for strategy: {formation_strategy_id}")

        # 更新實際的 total_periods
        progress_stream.total_steps = total_periods

        pm = PortfolioManager(strategy_id=name, initial_capital=INITIAL_CAPITAL, max_pairs=params.get("top_n", 10))
        all_trade_logs = []

        # Cache signature once — valid for all periods and pairs of the same Trading class
        _sig = inspect.signature(TradingClass.__init__)
        _valid_kwargs_keys = set(_sig.parameters.keys())

        for i, (_, p_row) in enumerate(df_periods.iterrows()):
            period_start = p_row["Period_Start"]
            trade_start = p_row["Trade_Start"]
            trade_end = p_row["Trade_End"]

            # 列印期數，觸發 ProgressAwareStdout 解析
            print(f"Period {i+1}/{total_periods}: {period_start} to {trade_end}")

            # 讀取該週期在形成期篩選出的配對 (使用 formation_strategy_id)
            df_pairs = pd.read_sql_query("""
                SELECT Ticker_A, Ticker_B, Sector_A, Sector_B, Pair_Rank, Formation_Params
                FROM formation_pairs
                WHERE strategy_id = ? AND Period_Start = ?
                ORDER BY Pair_Rank ASC
            """, conn, params=(formation_strategy_id, period_start))

            # 根據交易期設定只篩選前 top_n 對進行回測模擬
            top_n = params.get("top_n", 10)
            df_pairs = df_pairs.head(top_n)

            candidates = []
            param_map = {}
            for _, pair_row in df_pairs.iterrows():
                pair_tuple = (pair_row["Ticker_A"], pair_row["Ticker_B"])
                candidates.append(pair_tuple)
                param_map[pair_tuple] = {
                    "Sector_A": pair_row["Sector_A"],
                    "Sector_B": pair_row["Sector_B"],
                    "Rank": pair_row["Pair_Rank"],
                    "Params": json.loads(pair_row["Formation_Params"])
                }

            if params.get("vol_target_allocation", False) and param_map and pm.current_equity > 0:
                slots = pm.get_available_slots()
                selected = candidates[:slots]
                vol_inv = {
                    p: 1.0 / max(float(param_map[p]["Params"].get("Spread_Std", 1.0) or 1.0), 1e-6)
                    for p in selected
                }
                total_w = sum(vol_inv.values()) or 1.0
                base = pm.current_equity / max(pm.max_pairs, 1)
                n = len(selected)
                allocations = {p: min(base * n * vol_inv[p] / total_w, base * 2.0) for p in selected}
                for p, cap in allocations.items():
                    pm.active_pairs[p] = cap
            else:
                allocations = pm.allocate_capital(candidates)

            # 取得該週期的價格數據切片
            trade_start_idx = date_to_idx.get(pd.to_datetime(trade_start))
            trade_end_idx = date_to_idx.get(pd.to_datetime(trade_end))
            if trade_start_idx is None or trade_end_idx is None:
                print(f"  [Warning] Period dates ({trade_start} to {trade_end}) not found in database price index. Skipping period.")
                continue
            trade_dates = all_dates[trade_start_idx : trade_end_idx + 1]

            # 延伸價格數據，提供 zscore_window 前期的價格以避免 trading 前期 Z-Score 為 NaN
            zwin = params.get("zscore_window", 0)
            extended_start_idx = max(0, trade_start_idx - zwin)
            trade_prices_extended = price_pivot.iloc[extended_start_idx : trade_end_idx + 1]

            for pair, capital in allocations.items():
                ticker_a, ticker_b = pair
                pair_data = param_map[pair]
                form_params = pair_data["Params"]

                kwargs = {
                    "price_df": trade_prices_extended,  # 傳入延伸價格數據，修復前期 NaN 交易缺失問題
                    "trade_dates": trade_dates,
                    "selected_pairs": pd.DataFrame(),
                    "capital_per_pair": capital,
                    "full_price_df": price_pivot,
                    "formation_start": period_start,
                    "formation_end": trade_start
                }
                for k, v in params.items():
                    kwargs[k] = v

                valid_kwargs = {k: v for k, v in kwargs.items() if k in _valid_kwargs_keys}

                trading_instance = TradingClass(**valid_kwargs)

                if hasattr(trading_instance, '_simulate_pair'):
                    try:
                        # OLS_Alpha 存在時（HDBSCAN）走 log-price 路徑；不存在時（SSD/DTW）走向下相容路徑
                        raw_alpha = form_params.get("OLS_Alpha", None)
                        try:
                            ols_alpha_val = float(raw_alpha) if raw_alpha is not None and not pd.isna(float(raw_alpha)) else None
                        except (TypeError, ValueError):
                            ols_alpha_val = None
                        df_log = trading_instance._simulate_pair(
                            period_start=trade_start,
                            period_end=trade_end,
                            sector=pair_data["Sector_A"] if pair_data["Sector_A"] == pair_data["Sector_B"] else "CrossSector",
                            ticker_a=ticker_a,
                            ticker_b=ticker_b,
                            pair_rank=pair_data["Rank"],
                            hedge_ratio=form_params.get("Hedge_Ratio", 1.0),
                            form_spread_mean=form_params.get("Spread_Mean", 0.0),
                            form_spread_std=form_params.get("Spread_Std", 1.0),
                            log_mean_a=form_params.get("Log_Mean_A"),
                            log_std_a=form_params.get("Log_Std_A", 1.0),
                            log_mean_b=form_params.get("Log_Mean_B"),
                            log_std_b=form_params.get("Log_Std_B", 1.0),
                            first_price_a=float(form_params.get("First_Price_A") or 0.0),
                            first_price_b=float(form_params.get("First_Price_B") or 0.0),
                            ols_alpha=ols_alpha_val
                        )

                        if not df_log.empty:
                            # 下市/停牌處理：若模擬實際跑到的最後日期 < trade_end，
                            # 代表模擬中途因為 NaN（下市或停牌）而提早結束；
                            # 直接標記最後一筆為 FORCED_CLOSE_DELISTED，P&L 已由
                            # 模擬器的 PERIOD_END_EXIT 邏輯正確結算，無需再截斷。
                            last_sim_date = pd.to_datetime(df_log['Date'].iloc[-1])
                            expected_end   = pd.to_datetime(trade_end)
                            if last_sim_date < expected_end:
                                last_idx = df_log.index[-1]
                                df_log.loc[last_idx, 'Status'] = 'FORCED_CLOSE_DELISTED'

                            all_trade_logs.append(df_log)
                            final_realized_pnl = df_log['Realized_PnL'].iloc[-1]
                            pm.process_closed_trade(pair, final_realized_pnl)


                    except Exception as e:
                        print(f"Error simulating pair {pair}: {e}")

            # 清理 PortfolioManager 中的 active_pairs 以防累積
            pm.active_pairs.clear()

        conn.close()

        # 合併與儲存交易結果
        if all_trade_logs:
            df_all = pd.concat(all_trade_logs, ignore_index=True)
            df_all = df_all.sort_values("Date").reset_index(drop=True)

            # 依據命名規範建立 CSV
            filename = _build_filename(params)
            
            # 建立子目錄並輸出 CSV
            strat_output_dir = os.path.join(output_root, sub_dir)
            os.makedirs(strat_output_dir, exist_ok=True)
            csv_path = os.path.join(strat_output_dir, filename)
            df_all.to_csv(csv_path, index=False)

            # 寫入 SQLite 資料庫 (result.db)
            dataset_subdir = dataset_name.lower()
            path_key = f"{dataset_subdir}/{sub_dir}/{filename}"
            
            from strategies.db_utils import export_df_to_db
            export_df_to_db(
                df=df_all,
                strategy_name=strategy_config.get("db_method", name),
                params=params,
                dataset_name=dataset_name,
                path_key=path_key,
                db_path=db_path,
                overwrite=True,
                trade_method=strategy_config.get("trade_method", "Z-Score")
            )

        elapsed = time.time() - start_time
        progress_dict[name] = {
            "status":   "SUCCESS",
            "progress": "完成",
            "pct":      100,
            "msg":      f"交易期回測成功！最終淨值: ${pm.current_equity:.2f}",
            "elapsed":  elapsed,
        }
        return {"name": name, "status": "SUCCESS", "skipped": False, "elapsed": elapsed, "error": None, "final_equity": pm.current_equity}

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
        return {"name": name, "status": "FAILED", "skipped": False, "elapsed": elapsed, "error": str(e), "final_equity": 0.0}

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
# 主控引擎
# ════════════════════════════════════════════════════════════════════════════

def run_all_trading():
    print("=" * 80, flush=True)
    print("      🚀 交易期平行化控制主程式 (High-Performance Parallel Engine) 🚀", flush=True)
    print("=" * 80, flush=True)

    db_basename = os.path.splitext(os.path.basename(DB_PATH))[0]
    formation_db_path = f"formation_data/formation_pairs_{db_basename}.db"

    if not os.path.exists(formation_db_path):
        print(f"❌ 找不到配對資料庫 {formation_db_path}，請先執行 run_formation.py！", flush=True)
        return

    # 搜尋是否有吻合的 profile 來決定輸出與日誌目錄
    matched_profile = None
    for name, prof in DB_PROFILES.items():
        if os.path.normcase(os.path.abspath(prof["db_path"])) == os.path.normcase(os.path.abspath(DB_PATH)):
            matched_profile = prof
            break

    if matched_profile:
        OUTPUT_ROOT = matched_profile["output_root"]
        if "current" in matched_profile["output_root"].lower():
            dataset_name = "Current"
        elif "tiingo" in matched_profile["output_root"].lower():
            dataset_name = "Tiingo"
        else:
            dataset_name = "yF"
    else:
        OUTPUT_ROOT = f"./results/{db_basename}"
        if "current" in db_basename.lower():
            dataset_name = "Current"
        elif "tiingo" in db_basename.lower():
            dataset_name = "Tiingo"
        else:
            dataset_name = "yF"

    log_dir = f"{OUTPUT_ROOT}/logs/run_trading"
    os.makedirs(log_dir, exist_ok=True)

    # 1. 載入歷史價格 (只在主行程載入一次，並傳遞給各子行程)
    print(f"📊 正在從資料庫載入與前處理價格數據 '{DB_PATH}'...", flush=True)
    start_io = time.time()
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
        print(f"❌ [嚴重錯誤] 無法加載回測所需數據：{e}", flush=True)
        sys.exit(1)

    # 2. 斷點續傳篩選
    print(f"🔍 正在進行參數網格展開與任務篩選（斷點續傳檢查）...", flush=True)
    
    # ── 展開策略網格（依據 top_n_list, stop_loss_list/stop_loss_pct_list, max_sector_ratio_list 展開） ──
    expanded_strategies_raw = []
    for raw in strategies_raw:
        params = raw["params"]
        
        # 1. top_n_list
        top_n_list = params.get("top_n_list")
        if not top_n_list:
            top_n_list = [params.get("top_n", 10)]
        elif not isinstance(top_n_list, list):
            top_n_list = [top_n_list]
            
        # 2. stop_loss_list (相容 stop_loss_pct_list)
        sl_list = params.get("stop_loss_list") or params.get("stop_loss_pct_list")
        if not sl_list:
            sl_list = [params.get("stop_loss_pct", 0.0)]
        elif not isinstance(sl_list, list):
            sl_list = [sl_list]
            
        # 3. max_sector_ratio_list
        msr_list = params.get("max_sector_ratio_list")
        if not msr_list:
            msr_list = [params.get("max_sector_ratio", 0.0)]
        elif not isinstance(msr_list, list):
            msr_list = [msr_list]
            
        # (itertools and copy imports moved to top of file)
        for top_n, sl, msr in itertools.product(top_n_list, sl_list, msr_list):
            new_raw = copy.deepcopy(raw)
            new_params = new_raw["params"]
            
            # 設定具體數值
            new_params["top_n"] = top_n
            new_params["stop_loss_pct"] = sl
            new_params["max_sector_ratio"] = msr
            
            # 清理 list 參數以防混淆
            new_params.pop("top_n_list", None)
            new_params.pop("stop_loss_list", None)
            new_params.pop("stop_loss_pct_list", None)
            new_params.pop("max_sector_ratio_list", None)
            
            # 對接 Formation 配對資料庫的 strategy_id (Formation 階段只受 max_sector_ratio 影響，固定 top_n=20)
            msr_pct = int(msr * 100)
            new_raw["formation_strategy_id"] = f"{raw['name']}_MSR{msr_pct}"
            
            # 回測唯一的任務名稱字串 (加上後綴)
            sl_pct = int(sl * 100)
            new_raw["name"] = f"{raw['name']}_Top{top_n}_SL{sl_pct}_MSR{msr_pct}"
            
            expanded_strategies_raw.append(new_raw)

    # 建立未展開前的原始策略配置列表，用於儀表板統合呈現
    original_strategies_config = []
    for raw in strategies_raw:
        original_strategies_config.append({
            "name": raw["name"],
            "trading_module": raw["trading_module"],
            "sub_dir": raw["sub_dir"],
            "db_method": raw.get("db_method", raw["name"]),
            "trade_method": raw.get("trade_method", "Z-Score"),
            "params": raw["params"],
        })

    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    results = []
    strategies_to_run = []
    strategies_config = []

    # 全域結果資料庫路徑
    results_db_path = "results/result.db"
    os.makedirs(os.path.dirname(results_db_path), exist_ok=True)

    for raw in expanded_strategies_raw:
        config = {
            "name": raw["name"],
            "trading_module": raw["trading_module"],
            "sub_dir": raw["sub_dir"],
            "db_method": raw.get("db_method", raw["name"]),
            "trade_method": raw.get("trade_method", "Z-Score"),
            "params": raw["params"],
            "formation_strategy_id": raw["formation_strategy_id"],
        }
        strategies_config.append(config)

        if check_trading_completed(config, OUTPUT_ROOT, results_db_path, dataset_name):
            progress_dict[config["name"]] = {
                "status": "SUCCESS",
                "progress": "完成",
                "pct": 100,
                "msg": "✨ 已跳過 (偵測到已有完整回測結果)",
                "elapsed": 0.0,
            }
            # 嘗試從結果資料庫中讀取真實的 Final_Equity
            final_eq = INITIAL_CAPITAL
            filename = _build_filename(config["params"])
            dataset_subdir = dataset_name.lower()
            path_key = f"{dataset_subdir}/{config['sub_dir']}/{filename}"
            if os.path.exists(results_db_path):
                try:
                    with sqlite3.connect(results_db_path, timeout=10.0) as temp_conn:
                        cursor = temp_conn.cursor()
                        cursor.execute("SELECT Final_Equity FROM strategy_summaries WHERE _path = ?", (path_key,))
                        row = cursor.fetchone()
                        if row:
                            final_eq = float(row[0])
                except Exception:
                    pass
            results.append({
                "name": config["name"],
                "status": "SUCCESS",
                "skipped": True,
                "elapsed": 0.0,
                "error": None,
                "final_equity": final_eq,
            })
            print(f"  ⟳ 跳過（已完成）：{config['name']}", flush=True)
        else:
            progress_dict[config["name"]] = {
                "status": "PENDING",
                "progress": "0/0",
                "pct": 0,
                "msg": "排隊等待中...",
                "elapsed": 0.0,
            }
            strategies_to_run.append(config)
            print(f"  ● 排入執行：{config['name']}", flush=True)

    # ── 將策略分組 (依原始策略) 以便循序處理 ────────────────────────────────
    groups = []
    if strategies_to_run:
        from collections import defaultdict
        group_dict = defaultdict(list)
        for cfg in strategies_to_run:
            orig_name = None
            for orig in original_strategies_config:
                if cfg["name"].startswith(orig["name"]):
                    orig_name = orig["name"]
                    break
            if orig_name is None:
                orig_name = cfg["name"]
            group_dict[orig_name].append(cfg)
            
        # 保持原始策略定義的順序
        for orig in original_strategies_config:
            if orig["name"] in group_dict:
                groups.append(group_dict[orig["name"]])

    # 根據 CPU_LIMIT_PCT 限制 CPU 使用率
    max_cores = max(1, int((os.cpu_count() or 4) * CPU_LIMIT_PCT))
    max_workers = min(len(strategies_to_run), max_cores)


    if not strategies_to_run:
        print("\n✨ 所有策略交易期回測數據皆已存在且最新，無須重新計算！", flush=True)
        return

    # ── 啟動儀表板 ───────────────────────────────────────────────────────
    os.system("")
    placeholder_lines = len(original_strategies_config) + _DASHBOARD_FIXED_LINES
    sys.stdout.write("\n" * placeholder_lines)
    sys.stdout.flush()
    time.sleep(0.05)

    main_start_time = time.time()
    stop_event = threading.Event()

    dashboard_thread = start_dashboard_thread(
        progress_dict, original_strategies_config, main_start_time, log_dir, stop_event, stage_title="交易"
    )

    # 4. 多行程並行運算
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for group_cfgs in groups:
                futures = {}
                for config in group_cfgs:
                    f = executor.submit(
                        worker_task,
                        config,
                        price_pivot,
                        all_dates,
                        total_days,
                        local_first_trade_idx,
                        sector_mapping,
                        formation_db_path,
                        OUTPUT_ROOT,
                        results_db_path,
                        dataset_name,
                        progress_dict,
                    )
                    futures[f] = config

                for f in concurrent.futures.as_completed(futures):
                    config = futures[f]
                    try:
                        res = f.result()
                        results.append(res)
                    except Exception as exc:
                        results.append({
                            "name": config["name"],
                            "status": "FAILED",
                            "skipped": False,
                            "elapsed": 0.0,
                            "error": str(exc),
                            "final_equity": 0.0,
                        })
    finally:
        # 停止儀表板並重繪最終狀態
        stop_event.set()
        dashboard_thread.join(timeout=1.0)
        draw_dashboard(
            progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir, stage_title="交易"
        )

    # 6. 終端總結報告
    total_elapsed = time.time() - main_start_time
    print_summary_report(results, original_strategies_config, total_elapsed, show_equity=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_all_trading()
