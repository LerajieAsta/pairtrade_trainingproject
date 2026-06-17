import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
import itertools
import copy

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
        "db_path":     "./data/sp500_Current.db",
        "output_root": "./results/current",
        "label":       "S&P 500 現行成分股 (Current)",
    },
    "sp500_yF": {
        "db_path":     "./data/sp500_yF.db",
        "output_root": "./results/full",
        "label":       "S&P 500 完整歷史成分股 (yFinance)",
    },
    "sp500_Tiingo": {
        "db_path":     "./data/sp500_Tiingo.db",
        "output_root": "./results/tiingo",
        "label":       "S&P 500 完整歷史成分股 (Tiingo)",
    },
}

DB_PATH = "./data/sp500_Tiingo.db"
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
    # 單一數值作為 fallback 預設值
    "top_n":                        10,
    "stop_loss_pct":                0.0,
    "zscore_window":                0,
    "portfolio_stop_loss_pct":      0.0,
    "max_sector_ratio":             0.0,
    "dynamic_stop_z":               0.0,
}

hdbscan_common = {
    "use_dynamic_stop":         False,
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
    {
        "name":    "SSD Basic",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "SSD_Basic",
        "db_method": "SSD (Basic)",
        "trade_method": "Z-Score",
        "params":  {
            **base_params,
        },
    },
    # 2. SSD Rolling
    {
        "name":    "SSD Rolling",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "SSD_Rolling",
        "db_method": "SSD (Rolling)",
        "trade_method": "Z-Score",
        "params":  {
            **base_params,
        },
    },
    # 3. HDBSCAN SameSector UMAP
    {
        "name":    "HDBSCAN SameSector UMAP",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "HDBSCAN_SS_UMAP",
        "db_method": "HDBSCAN (SS-UMAP)",
        "trade_method": "Z-Score",
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
    # 4. HDBSCAN SameSector PCA
    {
        "name":    "HDBSCAN SameSector PCA",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "HDBSCAN_SS_PCA",
        "db_method": "HDBSCAN (SS-PCA)",
        "trade_method": "Z-Score",
        "params":  {
            **base_params,
            **hdbscan_common,
            "umap_n_components":  3,
            "umap_n_neighbors":   40,
            "umap_min_dist":      0.01,
            "umap_random_state":  42,
            "reduce_method":      "pca",
            "feature_mode":       "stats13",
        },
    },
    # 5. HDBSCAN MacroCluster UMAP
    {
        "name":    "HDBSCAN MacroCluster UMAP",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "HDBSCAN_Macro_UMAP",
        "db_method": "HDBSCAN (Macro-UMAP)",
        "trade_method": "Z-Score",
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
    {
        "name":    "HDBSCAN CrossSector MF",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "HDBSCAN_CS_MF",
        "db_method": "HDBSCAN (CS-MF)",
        "trade_method": "Z-Score",
        "params":  {
            **base_params,
            **hdbscan_common,
            "use_mom1_filter": True,
        },
    },
    # 7. HDBSCAN CrossSector PCA
    {
        "name":    "HDBSCAN CrossSector PCA",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "HDBSCAN_CS_PCA",
        "db_method": "HDBSCAN (CS-PCA)",
        "trade_method": "Z-Score",
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
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "HDBSCAN_CS_UMAP",
        "db_method": "HDBSCAN (CS-UMAP)",
        "trade_method": "Z-Score",
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
    {
        "name":    "Pure DTW (Notebook Ver)",
        "trading_module": "strategies.trading.pure_dtw_trading",
        "sub_dir": "Pure_DTW",
        "db_method": "Pure_DTW",
        "trade_method": "Pure DTW",
        "params":  {
            **base_params,
        },
    },
    # 10. DTW Cointegration Paper (DTW)
    {
        "name":    "DTW Cointegration Paper DTW",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "DTW_Paper",
        "db_method": "DTW (Paper)",
        "trade_method": "Z-Score",
        "params":  {
            **base_params,
            "method": "dtw",
        },
    },
    # 11. DTW Cointegration Paper (SSD+DTW PCA)
    {
        "name":    "DTW Cointegration Paper SSD-DTW-PCA",
        "trading_module": "strategies.trading.zscore_trading",
        "sub_dir": "SSD_DTW_PCA_Paper",
        "db_method": "SSD-DTW-PCA (Paper)",
        "trade_method": "Z-Score",
        "params":  {
            **base_params,
            "method": "ssd_dtw_pca",
        },
    },
    # 12. DRL LSTM
    {
        "name":    "DRL LSTM",
        "trading_module": "strategies.trading.drl_lstm_trading",
        "sub_dir": "DRL_LSTM",
        "db_method": "DRL LSTM",
        "trade_method": "DRL LSTM",
        "params":  {
            **base_params,
            "drl_episodes": 200,          # number of episodes for fast training
            "drl_batch_size": 64,         # batch size for experience replay
            "drl_gamma": 0.99,            # discount factor
            "drl_epsilon_start": 1.0,     # exploration rate
            "drl_epsilon_end": 0.05,
            "drl_epsilon_decay": 0.995,
            "drl_lr": 1e-3,               # learning rate
            "drl_hidden_size": 64,        # LSTM hidden size
            "drl_num_layers": 1,          # LSTM layers
        },
    },
]
from strategies.preprocess_equity import DataProcessor
from strategies.portfolio_manager import PortfolioManager
import importlib
import time
import multiprocessing
import concurrent.futures
import threading
import re
import traceback

# ProgressAwareStdout — 子行程 stdout 攔截器
# ════════════════════════════════════════════════════════════════════════════

class ProgressAwareStdout:
    """
    攔截子行程的 stdout，解析「Period X/Y」並即時更新跨行程進度字典。
    """
    def __init__(self, log_filepath: str, progress_dict, strategy_name: str, total_periods: int):
        self.log_file      = None
        self.progress_dict = progress_dict
        self.strategy_name = strategy_name
        self.total_periods = total_periods
        self.start_time    = time.time()
        self.pattern       = re.compile(r"Period\s*(\d+)/(\d+)")

        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> None:
        self.log_file.write(s)
        match = self.pattern.search(s)
        if match:
            try:
                curr_period = int(match.group(1))
                total    = int(match.group(2))
                pct      = min(100, int(curr_period / total * 100))

                current_info = dict(self.progress_dict.get(self.strategy_name, {}))
                current_info.update({
                    "status":   "RUNNING",
                    "progress": f"{curr_period}/{total}",
                    "pct":      pct,
                    "msg":      f"正在處理第 {curr_period:02d}/{total:02d} 期交易期模擬",
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
# 斷點續傳檢查
# ════════════════════════════════════════════════════════════════════════════

def _build_filename(params: dict) -> str:
    top_n = params.get("top_n", 10)
    sl    = int(params.get("stop_loss_pct", 0.0) * 100)
    zwin  = params.get("zscore_window", 0)
    msr   = int(params.get("max_sector_ratio", 0.0) * 100)
    return f"TradeLogs_Top{top_n}_SL{sl}_ZWin{zwin}_MSR{msr}.csv"

def check_trading_completed(strategy_config: dict, output_root: str) -> bool:
    if FORCE_RERUN:
        return False
        
    sub_dir = strategy_config["sub_dir"]
    params  = strategy_config["params"]
    
    filename = _build_filename(params)
    csv_path = os.path.join(output_root, sub_dir, filename)
    return os.path.exists(csv_path)


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
        progress_stream = ProgressAwareStdout(log_path, progress_dict, name, 0)
        sys.stdout = progress_stream
        sys.stderr = progress_stream

        # 載入交易策略模組
        strat_module = importlib.import_module(module_name)
        if hasattr(strat_module, 'Trading'):
            TradingClass = strat_module.Trading
        else:
            raise AttributeError(f"Strategy {module_name} missing Trading class.")

        conn = sqlite3.connect(formation_db_path)
        
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
        progress_stream.total_periods = total_periods

        pm = PortfolioManager(strategy_id=name, initial_capital=10000.0, max_pairs=params.get("top_n", 10))
        all_trade_logs = []

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

            allocations = pm.allocate_capital(candidates)

            # 取得該週期的價格數據切片
            trade_start_idx = all_dates.index(pd.to_datetime(trade_start))
            trade_end_idx = all_dates.index(pd.to_datetime(trade_end))
            trade_prices = price_pivot.iloc[trade_start_idx : trade_end_idx + 1]
            trade_dates = all_dates[trade_start_idx : trade_end_idx + 1]

            # 延伸價格數據，提供 zscore_window 前期的價格以避免 trading 前期 Z-Score 為 NaN
            zwin = params.get("zscore_window", 0)
            extended_start_idx = max(0, trade_start_idx - zwin)
            trade_prices_extended = price_pivot.iloc[extended_start_idx : trade_end_idx + 1]

            for pair, capital in allocations.items():
                ticker_a, ticker_b = pair
                pair_data = param_map[pair]
                form_params = pair_data["Params"]

                # 排除 NaN
                pa_series = trade_prices[ticker_a]
                pb_series = trade_prices[ticker_b]

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

                import inspect
                sig = inspect.signature(TradingClass.__init__)
                valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

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
                            log_mean_a=form_params.get("Log_Mean_A", 0.0),
                            log_std_a=form_params.get("Log_Std_A", 1.0),
                            log_mean_b=form_params.get("Log_Mean_B", 0.0),
                            log_std_b=form_params.get("Log_Std_B", 1.0),
                            ols_alpha=ols_alpha_val
                        )

                        if not df_log.empty:
                            # 停牌下市處理
                            nan_dates = pa_series[pa_series.isna()].index.union(pb_series[pb_series.isna()].index)
                            if not nan_dates.empty:
                                first_nan_date = nan_dates[0]
                                valid_log = df_log[pd.to_datetime(df_log['Date']) < first_nan_date].copy()
                                if not valid_log.empty:
                                    last_idx = valid_log.index[-1]
                                    valid_log.loc[last_idx, 'Status'] = 'FORCED_CLOSE_DELISTED'
                                    df_log = valid_log

                            all_trade_logs.append(df_log)
                            final_realized_pnl = df_log['Realized_PnL'].iloc[-1]
                            pm.process_closed_trade(pair, final_realized_pnl)

                    except Exception as e:
                        print(f"Error simulating pair {pair}: {e}")

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

            # 匯入至結果資料庫 (result.db)
            dataset_subdir = "current" if dataset_name.lower() == "current" else "full"
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
# 終端進度監控儀表板
# ════════════════════════════════════════════════════════════════════════════

_DASHBOARD_FIXED_LINES = 8

def _visible_len(s: str) -> int:
    return len(re.sub(r"\033\[[^m]*m", "", s))

def _pad_visible(s: str, width: int, align: str = "left") -> str:
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
    line("        \033[93;1m🚀 配對交易交易期平行化即時監控儀表板 (Trading Stage Core) 🚀\033[0m")
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
        f"  📊 \033[1m回測進度\033[0m | 總任務: {n_strategies:<2} | "
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
# 主控引擎
# ════════════════════════════════════════════════════════════════════════════

def run_all_trading():
    print("=" * 80, flush=True)
    print("      🚀 交易期平行化控制主程式 (High-Performance Parallel Engine) 🚀", flush=True)
    print("=" * 80, flush=True)

    db_basename = os.path.splitext(os.path.basename(DB_PATH))[0]
    formation_db_path = f"data/formation_pairs_{db_basename}.db"

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

        if check_trading_completed(config, OUTPUT_ROOT):
            progress_dict[config["name"]] = {
                "status": "SUCCESS",
                "progress": "完成",
                "pct": 100,
                "msg": "✨ 已跳過 (偵測到已有完整回測結果)",
                "elapsed": 0.0,
            }
            results.append({
                "name": config["name"],
                "status": "SUCCESS",
                "skipped": True,
                "elapsed": 0.0,
                "error": None,
                "final_equity": 10000.0,  # 預設
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

    # ── 交錯打散任務順序 (Interleave) ──────────────────────────────────────────
    # 將相同原始策略的子網格任務交錯排列，以確保多核心並行時，不同策略能同時啟動並呈現於儀表板
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

    def dashboard_updater() -> None:
        while not stop_event.is_set():
            draw_dashboard(progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir)
            time.sleep(0.3)

    dashboard_thread = threading.Thread(target=dashboard_updater, daemon=True)
    dashboard_thread.start()

    # 4. 多行程並行運算
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
        draw_dashboard(progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir)

    # 6. 終端總結報告
    total_elapsed = time.time() - main_start_time
    print("\n" + "=" * 80, flush=True)
    print("                     📊 交易期回測計算執行績效總結報告 (Summary) 📊", flush=True)
    print("=" * 80, flush=True)
    print(f" 總耗時: {total_elapsed:.2f} 秒（約 {total_elapsed / 60:.2f} 分鐘）", flush=True)
    print(
        f"\n{'策略名稱':<45} | {'狀態':<10} | {'跳過':<4} | {'最終權益':<10} | {'耗時(秒)':<10} | 錯誤訊息",
        flush=True,
    )
    print("-" * 110, flush=True)
    
    name_order = {c["name"]: i for i, c in enumerate(strategies_config)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["name"], 999))
    
    for res in results_sorted:
        err = res.get("error") or "無"
        skipped = "是" if res.get("skipped") else "否"
        final_eq_str = f"${res['final_equity']:.2f}" if 'final_equity' in res else "N/A"
        print(
            f"{res['name']:<45} | {res['status']:<10} | {skipped:<4} | {final_eq_str:<10} | {res['elapsed']:<10.2f} | {err}",
            flush=True,
        )
    print("=" * 80, flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_all_trading()
