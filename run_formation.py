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

# ── CPU 限制與 Python 3.14 資源追蹤器相容性補丁 ──────────────────────────────
# 1. 可調控的 CPU 使用率上限 (80%)
CPU_LIMIT_PCT = 0.8

# 2. 限制底層矩陣運算庫的執行緒數，防止單個子行程佔滿所有 CPU 核心
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NCORES"] = "1"

# 3. 預先註冊資源類型，解決 Python 3.14 resource_tracker 收到 JSON 時的解析錯誤
try:
    import multiprocessing.resource_tracker
    if 'folder' not in multiprocessing.resource_tracker._CLEANUP_FUNCS:
        multiprocessing.resource_tracker._CLEANUP_FUNCS['folder'] = lambda x: None
    if 'file' not in multiprocessing.resource_tracker._CLEANUP_FUNCS:
        multiprocessing.resource_tracker._CLEANUP_FUNCS['file'] = lambda x: None
except Exception:
    pass

# 4. 對 joblib.Parallel 進行 Monkey Patch，若在子行程中執行則強制單執行緒 (n_jobs=1)
try:
    import joblib
    original_parallel = joblib.Parallel
    class PatchedParallel(original_parallel):
        def __init__(self, n_jobs=None, *args, **kwargs):
            if multiprocessing.current_process().name != 'MainProcess':
                n_jobs = 1
            super().__init__(n_jobs=n_jobs, *args, **kwargs)
    joblib.Parallel = PatchedParallel
    joblib.parallel.Parallel = PatchedParallel
except ImportError:
    pass

# ── 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering ──────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# ── Configuration Settings ──────────────────────────────────────────────
FORCE_RERUN = True  # 設定為 True 可以強制重新執行，無視斷點續傳紀錄
DB_PROFILES = {
    "sp500_Current": {
        "db_path":     "./dataset/sp500_Current.db",
        "output_root": "./results/current",
        "label":       "S&P 500 現行成分股 (Current)",
    },
    "sp500_yF": {
        "db_path":     "./dataset/sp500_yF.db",
        "output_root": "./results/full",
        "label":       "S&P 500 完整歷史成分股 (yFinance)",
    },
    "sp500_Tiingo": {
        "db_path":     "./dataset/sp500_Tiingo.db",
        "output_root": "./results/tiingo",
        "label":       "S&P 500 完整歷史成分股 (Tiingo)",
    },
}

DB_PATH = "./dataset/sp500_yF.db"
TABLE_NAME = "Daily_Prices"
INFO_TABLE = "Constituents"
TICKER_COL = "Symbol"
SECTOR_COL = "GICS_Sector"

BACKTEST_START   = "2000-01"
BACKTEST_END     = "2025-12"
FORMATION_WINDOW = 252
FORWARD_DAYS     = 126
rolling_step     = 21

use_vol_adjust = False

base_params = {
    "entry_z":                      2.0,
    "exit_z":                       0.0,
    "formation_window":             FORMATION_WINDOW,
    "trading_window":               FORWARD_DAYS,
    "rolling_step":                 rolling_step,
    "fee_rate":                     0.001,
    "slippage_rate":                0.001,
    "initial_capital":              10000,
    "allow_reentry":                False,
    "zscore_clip":                  10.0,
    "min_spread_std":               1e-6,
    "min_tickers_for_pairing":      2,
    "use_vol_adjust":               use_vol_adjust,
    "max_holding_days":             30,
    # 支援多數值比對的網格搜尋參數清單
    "top_n_list":                   [1, 3, 5, 10, 20],
    "stop_loss_list":               [0.0, 0.05, 0.15],
    "max_sector_ratio_list":        [0.0, 0.30, 0.50],
    # 單一數值作為 fallback 預設值 (在形成期，配對數固定為 20 對)
    "top_n":                        20,
    "stop_loss_pct":                0.0,
    "zscore_window":                0,
    "portfolio_stop_loss_pct":      0.0,
    "max_sector_ratio":             0.0,
    "dynamic_stop_z":               0.0,
}

hdbscan_common = {
    "use_dynamic_stop":         True,
    "hdbscan_min_cluster_size": 30,
    "hdbscan_min_samples":      10,
    "hdbscan_metric":           "euclidean",
    "adf_max_lags":             1,
    "adf_pvalue_threshold":     0.01,
    "min_corr":                 0.50,
    "min_zero_crossings":       5,
}

# ════════════════════════════════════════════════════════════════════════════
# 💡 未來如何更換與切換交易/形成期策略說明指引：
# ────────────────────────────────────────────────────────────────────────────
# 1. 新增策略模組：
#    - 形成期策略：在 `strategies/formation/` 目錄下建立 Python 檔案，實作 `Formation` 類別及 `run()` 方法。
#    - 交易期策略：在 `strategies/trading/` 目錄下建立 Python 檔案，實作 `Trading` 類別及 `_simulate_pair()` 方法。
# 2. 註冊與切換策略：
#    - 調整下方 `strategies_raw` 列表中對應策略的 `formation_module`（在 run_formation.py）或 `trading_module`（在 run_trading.py）。
#    - 程式在執行時會使用 `importlib.import_module` 動態加載您指定的策略模組，無需更改主控邏輯。
# ════════════════════════════════════════════════════════════════════════════
strategies_raw = [
    # 1. SSD Basic
    # {
    #     "name":    "SSD Basic",
    #     "formation_module": "strategies.formation.ssd_basic",
    #     "sub_dir": "SSD_Basic",
    #     "db_method": "SSD (Basic)",
    #     "params":  {
    #         **base_params,
    #     },
    # },
    # 2. SSD Rolling
    # {
    #     "name":    "SSD Rolling",
    #     "formation_module": "strategies.formation.ssd",
    #     "sub_dir": "SSD_Rolling",
    #     "db_method": "SSD (Rolling)",
    #     "params":  {
    #         **base_params,
    #     },
    # },
    # 3. HDBSCAN SameSector UMAP
    # {
    #     "name":    "HDBSCAN SameSector UMAP",
    #     "formation_module": "strategies.formation.HDBSCAN",
    #     "sub_dir": "HDBSCAN_SS_UMAP",
    #     "db_method": "HDBSCAN (SS-UMAP)",
    #     "params":  {
    #         **base_params,
    #         **hdbscan_common,
    #         "umap_n_components":  5,
    #         "umap_n_neighbors":   40,
    #         "umap_min_dist":      0.01,
    #         "umap_random_state":  42,
    #         "reduce_method":      "umap",
    #         "feature_mode":       "stats13",
    #     },
    # },
    # 4. HDBSCAN SameSector PCA
    # {
    #     "name":    "HDBSCAN SameSector PCA",
    #     "formation_module": "strategies.formation.HDBSCAN",
    #     "sub_dir": "HDBSCAN_SS_PCA",
    #     "db_method": "HDBSCAN (SS-PCA)",
    #     "params":  {
    #         **base_params,
    #         **hdbscan_common,
    #         "umap_n_components":  3,
    #         "umap_n_neighbors":   40,
    #         "umap_min_dist":      0.01,
    #         "umap_random_state":  42,
    #         "reduce_method":      "pca",
    #         "feature_mode":       "stats13",
    #     },
    # },
    # 5. HDBSCAN MacroCluster UMAP
    {
        "name":    "HDBSCAN MacroCluster UMAP",
        "formation_module": "strategies.formation.HDBSCAN_MacroCluster_UMAP",
        "sub_dir": "HDBSCAN_Macro_UMAP",
        "db_method": "HDBSCAN (Macro-UMAP)",
        "params":  {
            **base_params,
            **hdbscan_common,
            "umap_n_components":  5,
            "umap_n_neighbors":   40,
            "umap_min_dist":      0.01,
            "umap_random_state":  42,
            "reduce_method":      "umap",
            "feature_mode":       "stats13",
        },
    },
    # 6. HDBSCAN CrossSector MF
    # {
    #     "name":    "HDBSCAN CrossSector MF",
    #     "formation_module": "strategies.formation.HDBSCAN_CrossSector_MultiFactor",
    #     "sub_dir": "HDBSCAN_CS_MF",
    #     "db_method": "HDBSCAN (CS-MF)",
    #     "params":  {
    #         **base_params,
    #         **hdbscan_common,
    #         "use_mom1_filter": True,
    #     },
    # },
    # 7. HDBSCAN CrossSector PCA
    {
        "name":    "HDBSCAN CrossSector PCA",
        "formation_module": "strategies.formation.HDBSCAN_CrossSector_PCA",
        "sub_dir": "HDBSCAN_CS_PCA",
        "db_method": "HDBSCAN (CS-PCA)",
        "params":  {
            **base_params,
            **hdbscan_common,
            "umap_n_components":  3,
            "umap_n_neighbors":   40,
            "umap_min_dist":      0.01,
            "umap_random_state":  42,
            "reduce_method":      "pca",
            "feature_mode":       "stats13",
            "use_mom1_filter":    True,
        },
    },
    # 8. HDBSCAN CrossSector UMAP
    {
        "name":    "HDBSCAN CrossSector UMAP",
        "formation_module": "strategies.formation.HDBSCAN_CrossSector_UMAP",
        "sub_dir": "HDBSCAN_CS_UMAP",
        "db_method": "HDBSCAN (CS-UMAP)",
        "params":  {
            **base_params,
            **hdbscan_common,
            "umap_n_components":  5,
            "umap_n_neighbors":   40,
            "umap_min_dist":      0.01,
            "umap_random_state":  42,
            "reduce_method":      "umap",
            "feature_mode":       "stats13",
            "use_mom1_filter":    True,
        },
    },
    # 9. Pure DTW (Notebook Ver)
    # {
    #     "name":    "Pure DTW (Notebook Ver)",
    #     "formation_module": "strategies.formation.DTW_Pure_Notebook",
    #     "sub_dir": "Pure_DTW",
    #     "db_method": "Pure_DTW",
    #     "params":  {
    #         **base_params,
    #     },
    # },
    # 10. DTW Cointegration Paper (DTW)
    # {
    #     "name":    "DTW Cointegration Paper DTW",
    #     "formation_module": "strategies.formation.DTW_Cointegration_Paper",
    #     "sub_dir": "DTW_Paper",
    #     "db_method": "DTW (Paper)",
    #     "params":  {
    #         **base_params,
    #         "method": "dtw",
    #     },
    # },
    # 11. DTW Cointegration Paper (SSD+DTW PCA)
    # {
    #     "name":    "DTW Cointegration Paper SSD-DTW-PCA",
    #     "formation_module": "strategies.formation.DTW_Cointegration_Paper",
    #     "sub_dir": "SSD_DTW_PCA_Paper",
    #     "db_method": "SSD-DTW-PCA (Paper)",
    #     "params":  {
    #         **base_params,
    #         "method": "ssd_dtw_pca",
    #     },
    # },
    # 12. DRL LSTM
    {
        "name":    "DRL LSTM",
        "formation_module": "strategies.formation.HDBSCAN_CrossSector_PCA",
        "sub_dir": "DRL_LSTM",
        "db_method": "DRL LSTM",
        "params":  {
            **base_params,
            "drl_episodes": 200,
            "drl_batch_size": 64,
            "drl_gamma": 0.99,
            "drl_epsilon_start": 1.0,
            "drl_epsilon_end": 0.05,
            "drl_epsilon_decay": 0.995,
            "drl_lr": 1e-3,
            "drl_hidden_size": 64,
            "drl_num_layers": 1,
        },
    },
]

from strategies.preprocess_equity import DataProcessor
from strategies.db_utils import init_formation_db

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

        with sqlite3.connect(formation_db_path) as conn:
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
# ProgressAwareStdout — 子行程 stdout 攔截器
# ════════════════════════════════════════════════════════════════════════════

class ProgressAwareStdout:
    """
    攔截子行程的 stdout，解析「Window X/Y」並即時更新跨行程進度字典。
    """
    def __init__(self, log_filepath: str, progress_dict, strategy_name: str, total_windows: int):
        self.log_file      = None
        self.progress_dict = progress_dict
        self.strategy_name = strategy_name
        self.total_windows = total_windows
        self.start_time    = time.time()
        self.pattern       = re.compile(r"Window\s*(\d+)/(\d+)")

        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> None:
        self.log_file.write(s)
        match = self.pattern.search(s)
        if match:
            try:
                curr_win = int(match.group(1))
                total    = int(match.group(2))
                pct      = min(100, int(curr_win / total * 100))

                current_info = dict(self.progress_dict.get(self.strategy_name, {}))
                current_info.update({
                    "status":   "RUNNING",
                    "progress": f"{curr_win}/{total}",
                    "pct":      pct,
                    "msg":      f"正在處理第 {curr_win:03d}/{total:03d} 期配對計算",
                    "elapsed":  time.time() - self.start_time,
                })
                self.progress_dict[self.strategy_name] = current_info
            except Exception:
                pass

    def flush(self) -> None:
        if self.log_file is not None:
            self.log_file.flush()

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.close()

    def __getattr__(self, name: str):
        log_file = self.__dict__.get("log_file")
        if log_file is None:
            raise AttributeError(name)
        return getattr(log_file, name)


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
    progress_stream = ProgressAwareStdout(log_path, progress_dict, name, total_windows)
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
                main_conn = sqlite3.connect(main_db)
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
# 終端進度監控儀表板
# ════════════════════════════════════════════════════════════════════════════

_DASHBOARD_FIXED_LINES = 8

def _visible_len(s: str) -> int:
    """計算去除 ANSI Escape Code 後的可見字元長度"""
    return len(re.sub(r"\033\[[^m]*m", "", s))


def _pad_visible(s: str, width: int, align: str = "left") -> str:
    """考慮 ANSI Escape Code 的可見長度進行填充對齊"""
    v_len = _visible_len(s)
    if v_len >= width:
        return s
    padding = " " * (width - v_len)
    if align == "right":
        return padding + s
    return s + padding


def draw_dashboard(
    progress_dict,
    strategies_config: list,  # 傳入原始 strategies_config
    main_start_time: float,
    log_dir_desc: str = "",
) -> None:
    """
    原地渲染終端進度儀表板。
    """
    n_strategies = len(strategies_config)
    total_lines  = n_strategies + _DASHBOARD_FIXED_LINES

    sys.stdout.write(f"\033[{total_lines}A")

    try:
        term_width = min(os.get_terminal_size().columns, 85)
    except OSError:
        term_width = 85

    def line(s: str) -> None:
        visible = re.sub(r"\033\[[^m]*m", "", s)
        if len(visible) > term_width:
            s = s[:term_width + (len(s) - len(visible))] + "\033[0m"
        print(f"{s}\033[K")

    line("\033[95m" + "═" * term_width + "\033[0m")
    line("        \033[93;1m🚀 配對交易形成期平行化即時監控儀表板 (Formation Stage Core) 🚀\033[0m")
    line("\033[95m" + "═" * term_width + "\033[0m")

    # 統合計算各原始策略的進度資訊
    aggregated_info = {}
    for config in strategies_config:
        orig_name = config["name"]
        
        sub_total = 0
        sub_success = 0
        sub_skipped = 0
        sub_failed = 0
        sub_running = 0
        sub_pending = 0
        sum_pct = 0.0
        all_elapsed = []

        # 遍歷 progress_dict 尋找所有以 orig_name 開頭的子組合任務
        for k, info in progress_dict.items():
            if k == orig_name or k.startswith(orig_name + "_"):
                sub_total += 1
                status = info.get("status", "PENDING")
                pct = info.get("pct", 0)
                sum_pct += pct
                all_elapsed.append(info.get("elapsed", 0.0))
                msg = info.get("msg", "")
                
                if status == "PENDING":
                    sub_pending += 1
                elif status == "RUNNING":
                    sub_running += 1
                elif status == "SUCCESS":
                    if "跳過" in msg or "已跳過" in msg:
                        sub_skipped += 1
                    else:
                        sub_success += 1
                elif status == "FAILED":
                    sub_failed += 1

        if sub_total == 0:
            # 預設狀態
            aggregated_info[orig_name] = {
                "status": "PENDING",
                "progress": "0/0",
                "pct": 0,
                "msg": "等待中...",
                "elapsed": 0.0,
            }
        else:
            avg_pct = int(sum_pct / sub_total)
            max_elapsed = max(all_elapsed) if all_elapsed else 0.0
            completed = sub_success + sub_skipped + sub_failed
            
            # 狀態統合
            if completed == sub_total:
                if sub_failed > 0:
                    status = "FAILED"
                    msg = f"完成 ({sub_success}組成功, {sub_failed}組失敗)"
                elif sub_skipped == sub_total:
                    status = "SUCCESS"
                    msg = "已跳過 (已有完整結果)"
                else:
                    status = "SUCCESS"
                    msg = f"完成 ({sub_success}組成功)"
            elif sub_pending == sub_total:
                status = "PENDING"
                msg = "排隊等待中..."
            else:
                status = "RUNNING"
                msg = f"執行中 ({sub_running}組運行, {completed}/{sub_total}完成)"

            aggregated_info[orig_name] = {
                "status": status,
                "progress": f"{completed}/{sub_total}",
                "pct": avg_pct,
                "msg": msg,
                "elapsed": max_elapsed,
            }

    # 統計總狀態
    counts = {"PENDING": 0, "RUNNING": 0, "SUCCESS": 0, "SKIPPED": 0, "FAILED": 0}
    for config in strategies_config:
        info = aggregated_info[config["name"]]
        status = info["status"]
        if status == "SUCCESS" and "跳過" in info["msg"]:
            counts["SKIPPED"] += 1
        else:
            counts[status] = counts.get(status, 0) + 1

    elapsed = time.time() - main_start_time
    line(
        f"  📊 \033[1m形成期進度\033[0m | 總任務: {n_strategies:<2} | "
        f"運行: \033[96m{counts['RUNNING']:<2}\033[0m | "
        f"成功: \033[92m{counts['SUCCESS']:<2}\033[0m | "
        f"跳過: \033[33m{counts['SKIPPED']:<2}\033[0m | "
        f"失敗: \033[91m{counts['FAILED']:<2}\033[0m | "
        f"耗時: {elapsed:.1f}s"
    )
    line("\033[90m" + "─" * term_width + "\033[0m")

    for config in strategies_config:
        name = config["name"]
        info = aggregated_info[name]

        status       = info["status"]
        pct          = info["pct"]
        prog         = info["progress"]
        msg          = info["msg"]
        task_elapsed = info["elapsed"]

        if status == "PENDING":
            status_str = "\033[90m○ PENDING\033[0m"
        elif status == "RUNNING":
            status_str = "\033[96m● RUNNING\033[0m"
        elif status == "SUCCESS":
            is_skipped = "跳過" in msg
            status_str = "\033[33m⟳ SKIPPED\033[0m" if is_skipped else "\033[92m✓ SUCCESS\033[0m"
        elif status == "FAILED":
            status_str = "\033[91m❌ FAILED \033[0m"
        else:
            status_str = f"\033[37m{status}\033[0m"

        bar_width = 5
        completed = int(bar_width * pct / 100)
        bar       = "\033[94m" + "█" * completed + "\033[90m" + "░" * (bar_width - completed) + "\033[0m"

        if status == "RUNNING" and pct > 0:
            eta_str = f"ETA {task_elapsed / pct * (100 - pct):.0f}s"
        elif status in ("SUCCESS", "SKIPPED"):
            eta_str = "Done"
        elif status == "FAILED":
            eta_str = "Err"
        else:
            eta_str = "---"

        # 簡化名稱長度以利儀表板排版
        display_name = name
        if display_name.endswith(" NoReEntry"):
            display_name = display_name[:-10]
        elif display_name.endswith(" ReEntry"):
            display_name = display_name[:-8]

        status_pad = _pad_visible(status_str, 10)
        name_pad   = _pad_visible(display_name[:15], 15)
        bar_pad    = _pad_visible(bar, 5)
        msg_pad    = _pad_visible(msg[:20], 20)

        line(
            f"  {status_pad} | {name_pad} | "
            f"{bar_pad} {pct:>3}% ({prog:<7}) | "
            f"{eta_str:<8} | \033[93m{task_elapsed:>5.1f}s\033[0m | "
            f"\033[37m{msg_pad}\033[0m"
        )

    line("\033[90m" + "─" * term_width + "\033[0m")
    line(f"  📁 詳細日誌重導向至: \033[36m{log_dir_desc}/策略名稱.log\033[0m")
    line("\033[95m" + "═" * term_width + "\033[0m")
    sys.stdout.flush()


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
            temp_conn = sqlite3.connect(temp_db)
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
    formation_db_path = f"data/formation_pairs_{db_basename}.db"

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
    old_temps = glob.glob(f"data/formation_pairs_{db_basename}_*.db*")
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
            "temp_db_path":     f"data/formation_pairs_{db_basename}_{safe_name}.db",
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
        from collections import defaultdict
        groups = defaultdict(list)
        for cfg in strategies_to_run:
            orig_name = None
            for orig in original_strategies_config:
                if cfg["name"].startswith(orig["name"]):
                    orig_name = orig["name"]
                    break
            if orig_name is None:
                orig_name = cfg["name"]
            groups[orig_name].append(cfg)
        
        interleaved_to_run = []
        max_len = max(len(v) for v in groups.values()) if groups else 0
        for i in range(max_len):
            for orig_name in groups:
                if i < len(groups[orig_name]):
                    interleaved_to_run.append(groups[orig_name][i])
        strategies_to_run = interleaved_to_run

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

    def dashboard_updater() -> None:
        while not stop_event.is_set():
            draw_dashboard(progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir)
            time.sleep(0.3)

    dashboard_thread = threading.Thread(target=dashboard_updater, daemon=True)
    dashboard_thread.start()

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
            draw_dashboard(progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir)
    finally:
        # 5. 合併與清理暫存資料庫 (即使中斷執行，也會將已完成策略合併並清除暫存檔)
        temp_db_paths = [cfg["temp_db_path"] for cfg in strategies_to_run]
        merge_databases(formation_db_path, temp_db_paths)

    # 6. 終端總結報告
    total_elapsed = time.time() - main_start_time
    print("\n" + "=" * 80, flush=True)
    print("                     📊 形成期配對計算執行績效總結報告 (Summary) 📊", flush=True)
    print("=" * 80, flush=True)
    print(f" 總耗時: {total_elapsed:.2f} 秒（約 {total_elapsed / 60:.2f} 分鐘）", flush=True)
    print(
        f"\n{'策略名稱':<45} | {'狀態':<10} | {'跳過':<4} | {'耗時(秒)':<10} | 錯誤訊息",
        flush=True,
    )
    print("-" * 90, flush=True)
    
    # 按照策略定義順序排序
    name_order = {c["name"]: i for i, c in enumerate(strategies_config)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["name"], 999))
    
    for res in results_sorted:
        err     = res.get("error") or "無"
        skipped = "是" if res.get("skipped") else "否"
        print(
            f"{res['name']:<45} | {res['status']:<10} | {skipped:<4} | {res['elapsed']:<10.2f} | {err}",
            flush=True,
        )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_all_formations()
