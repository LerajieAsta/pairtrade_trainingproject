import os
import sys
import time
import re
import threading
import json
import numpy as np
import pandas as pd

# ── CPU 限制與 Python 3.14 資源追蹤器相容性補丁 ──────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NCORES"] = "1"

try:
    import multiprocessing.resource_tracker
    if 'folder' not in multiprocessing.resource_tracker._CLEANUP_FUNCS:
        multiprocessing.resource_tracker._CLEANUP_FUNCS['folder'] = lambda x: None
    if 'file' not in multiprocessing.resource_tracker._CLEANUP_FUNCS:
        multiprocessing.resource_tracker._CLEANUP_FUNCS['file'] = lambda x: None
except Exception:
    pass

try:
    import joblib
    import multiprocessing
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# ── 共用常數 ─────────────────────────────────────────────────────────────
FORCE_RERUN = False
CPU_LIMIT_PCT = 0.8
DRL_MAX_WORKERS = 1  # DRL 一次只跑一個，避免多 process 搶 GPU (HAMI 會 kill)
DB_PROFILES = {
    "sp500_Current": {
        "db_path":     "./dataset/sp500_Current.db",
        "output_root": "./results/current",
        "label":       "S&P 500 現行成分股 (Current)",
    },
    "sp500_yF": {
        "db_path":     "./dataset/sp500_yF.db",
        "output_root": "./results/yFinance",
        "label":       "S&P 500 完整歷史成分股 (yFinance)",
    },
    "sp500_Tiingo": {
        "db_path":     "./dataset/sp500_Tiingo.db",
        "output_root": "./results/tiingo",
        "label":       "S&P 500 完整歷史成分股 (Tiingo)",
    },
}

DB_PATH = "./dataset/sp500_Tiingo.db"
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

INITIAL_CAPITAL = 10000.0
# 最大同時重疊交易期數（rolling_step 整除 FORWARD_DAYS）
CONCURRENT_PERIODS = max(1, FORWARD_DAYS // rolling_step)

base_params = {
    "entry_z":                      2.0,
    "exit_z":                       0.0,
    "formation_window":             FORMATION_WINDOW,
    "trading_window":               FORWARD_DAYS,
    "rolling_step":                 rolling_step,
    "fee_rate":                     0.001,
    "slippage_rate":                0.001,
    "initial_capital":              INITIAL_CAPITAL,
    "allow_reentry":                False,
    "zscore_clip":                  10.0,
    "min_spread_std":               1e-6,
    "min_tickers_for_pairing":      2,
    "use_vol_adjust":               use_vol_adjust,
    "max_holding_days":             30,
    "top_n_list":                   [1, 3, 5, 10, 20],
    "stop_loss_list":               [0.0, 0.05, 0.15],
    "max_sector_ratio_list":        [0.0, 0.30, 0.50],
    "top_n":                        20,
    "stop_loss_pct":                0.0,
    "zscore_window":                0,
    "portfolio_stop_loss_pct":      0.0,
    "max_sector_ratio":             0.0,
    "dynamic_stop_z":               0.0,
    "vol_regime_threshold":         0.0,
    "vol_target_allocation":        False,
}

# hdbscan_common — kept as empty dict for import compatibility;
# old HDBSCAN strategies (3-14) archived 2026-06-27 to archive/11506/formation/
hdbscan_common = {}

# ── 共用 HDBSCAN / UMAP 超參數區塊 ─────────────────────────────────────────
_HDBSCAN_UMAP_COMMON = {
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples":      2,
    "hdbscan_metric":           "euclidean",
    "umap_n_components":        5,
    "umap_n_neighbors":         40,
    "umap_min_dist":            0.01,
    "umap_random_state":        42,
}

# HDBSCAN_UMAP 的篩選門檻（用於策略 7、8、12 及 Ensemble 子策略）
_HDBSCAN_UMAP_FILTERS = {
    "adf_pvalue_threshold": 0.01,
    "min_corr":             0.50,
    "min_zero_crossings":   3,              # 放寬：5 → 3（允許訊號較少的 window）
    "hurst_threshold":      0.55,           # 放寬：0.5 → 0.55（允許稍弱的均值回歸）
    "halflife_min":         1.0,
    "halflife_max":         FORWARD_DAYS / 2,  # 放寬：/3(42d) → /2(63d)
    "roll_corr_window":     60,
    "max_beta_diff":        0.8,
    "max_vol_ratio":        3.0,
    "min_adv_ratio":        0.1,
    "use_mom1_filter":      True,
    "feature_mode":         "stats10",
}

# HDBSCAN_MultiScale 的篩選門檻（用於策略 5、6、13 及 Ensemble 子策略）
_HDBSCAN_MS_FILTERS = {
    "adf_pvalue_threshold": 0.05,
    "adf_sub_pvalue":       0.10,
    "min_corr_mean":        0.50,
    "min_corr_min":         0.10,
    "max_corr_std":         0.30,
    "min_coint_pass_rate":  0.40,
    "max_regime_diff":      0.50,
    "max_vol_ratio_std":    0.80,
    "use_mom1_filter":      True,
    "halflife_min":         1.0,
    "halflife_max":         FORWARD_DAYS / 2,  # 放寬：/3(42d) → /2(63d)
}

strategies_raw_all = [
    # ── SSD ──────────────────────────────────────────────────────────────────
    # 1. SSD Basic
    {
        "name":             "SSD Basic",
        "formation_module": "strategies.formation.ssd_basic",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_Basic",
        "db_method":        "SSD (Basic)",
        "trade_method":     "Z-Score",
        "params":  {
            **base_params,
        },
    },
    # 2. SSD Rolling
    {
        "name":             "SSD Rolling",
        "formation_module": "strategies.formation.ssd_rolling",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_Rolling",
        "db_method":        "SSD (Rolling)",
        "trade_method":     "Z-Score",
        "params":  {
            **base_params,
        },
    },
    # ── DTW ──────────────────────────────────────────────────────────────────
    # 3. DTW Paper (DTW)
    {
        "name":             "DTW Paper (DTW)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "DTW_Paper",
        "db_method":        "DTW (Paper)",
        "trade_method":     "Z-Score",
        "params":  {
            **base_params,
            "method": "dtw",
        },
    },
    # 4. DTW Paper (SSD-DTW-PCA)
    {
        "name":             "DTW Paper (SSD-DTW-PCA)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_DTW_PCA_Paper",
        "db_method":        "SSD-DTW-PCA (Paper)",
        "trade_method":     "Z-Score",
        "params":  {
            **base_params,
            "method": "ssd_dtw_pca",
        },
    },
    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    # 5. HDBSCAN MultiScale
    {
        "name":             "HDBSCAN MultiScale",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_MultiScale",
        "db_method":        "HDBSCAN (MultiScale)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "umap"},
    },
    # 6. HDBSCAN MultiScale PCA-UMAP
    {
        "name":             "HDBSCAN MultiScale PCA-UMAP",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_MultiScale_PCA_UMAP",
        "db_method":        "HDBSCAN (MultiScale-PCA-UMAP)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "pca_umap"},
    },
    # 7. HDBSCAN UMAP
    {
        "name":             "HDBSCAN UMAP",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_UMAP",
        "db_method":        "HDBSCAN (UMAP)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "umap"},
    },
    # 8. HDBSCAN UMAP PCA-UMAP
    {
        "name":             "HDBSCAN UMAP PCA-UMAP",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_UMAP_PCA_UMAP",
        "db_method":        "HDBSCAN (UMAP-PCA-UMAP)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "pca_umap"},
    },
    # ── Ensemble ─────────────────────────────────────────────────────────────
    # 9. Ensemble: HDBSCAN UMAP × HDBSCAN MultiScale
    {
        "name":             "Ensemble HDBSCAN",
        "formation_module": "strategies.formation.ensemble",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Ensemble_HDBSCAN",
        "db_method":        "Ensemble (HDBSCAN)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "sub_top_n_multiplier": 3,
            "sub_strategies": [
                {
                    "name":   "HDBSCAN UMAP",
                    "module": "strategies.formation.HDBSCAN_UMAP",
                    "params": {**_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "umap"},
                },
                {
                    "name":   "HDBSCAN MultiScale",
                    "module": "strategies.formation.HDBSCAN_MultiScale",
                    "params": {**_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "umap"},
                },
            ],
        },
    },
    # 10. Ensemble: SSD Rolling × DTW Paper
    {
        "name":             "Ensemble SSD-DTW",
        "formation_module": "strategies.formation.ensemble",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Ensemble_SSD_DTW",
        "db_method":        "Ensemble (SSD-DTW)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "sub_top_n_multiplier": 3,
            "sub_strategies": [
                {
                    "name":   "SSD Rolling",
                    "module": "strategies.formation.ssd_rolling",
                    "params": {},
                },
                {
                    "name":   "DTW Paper",
                    "module": "strategies.formation.DTW_Cointegration_Paper",
                    "params": {"method": "dtw"},
                },
            ],
        },
    },
    # ── DRL ──────────────────────────────────────────────────────────────────
    # 11. SSD Rolling DRL
    {
        "name":             "SSD Rolling DRL",
        "formation_module": "strategies.formation.ssd_rolling",
        "trading_module":   "strategies.trading.drl_lstm_trading",
        "sub_dir":          "SSD_Rolling_DRL",
        "db_method":        "SSD (Rolling-DRL)",
        "trade_method":     "DRL",
        "params":  {
            **base_params,
            "drl_episodes": 40,
        },
    },
    # 12. HDBSCAN UMAP DRL
    {
        "name":             "HDBSCAN UMAP DRL",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.drl_lstm_trading",
        "sub_dir":          "HDBSCAN_UMAP_DRL",
        "db_method":        "HDBSCAN (UMAP-DRL)",
        "trade_method":     "DRL",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "umap", "drl_episodes": 40},
    },
    # 13. HDBSCAN MultiScale DRL
    {
        "name":             "HDBSCAN MultiScale DRL",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "strategies.trading.drl_lstm_trading",
        "sub_dir":          "HDBSCAN_MultiScale_DRL",
        "db_method":        "HDBSCAN (MultiScale-DRL)",
        "trade_method":     "DRL",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "umap", "drl_episodes": 40},
    },
    # ── Kalman Filter ─────────────────────────────────────────────────────────
    # 14. SSD Rolling Kalman
    {
        "name":             "SSD Rolling Kalman",
        "formation_module": "strategies.formation.ssd_rolling",
        "trading_module":   "strategies.trading.kalman_trading",
        "sub_dir":          "SSD_Rolling_Kalman",
        "db_method":        "SSD (Rolling-Kalman)",
        "trade_method":     "Kalman",
        "params":  {
            **base_params,
            "kalman_delta": 1e-4,
            "kalman_R":     1e-2,
        },
    },
    # 15. HDBSCAN UMAP Kalman
    {
        "name":             "HDBSCAN UMAP Kalman",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.kalman_trading",
        "sub_dir":          "HDBSCAN_UMAP_Kalman",
        "db_method":        "HDBSCAN (UMAP-Kalman)",
        "trade_method":     "Kalman",
        "params": {
            **base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS,
            "reduce_method": "umap",
            "kalman_delta":  1e-4,
            "kalman_R":      1e-2,
        },
    },
]
strategies_raw = strategies_raw_all[:]


# ── 儀表板與 ProgressAwareStdout 類別與函數 ───────────────────────────────
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
    strategies_config: list,
    main_start_time: float,
    log_dir_desc: str = "",
    stage_title: str = "形成",
) -> None:
    """
    原地渲染終端進度儀表板。
    """
    n_strategies = len(strategies_config)
    total_lines  = n_strategies + _DASHBOARD_FIXED_LINES

    sys.stdout.write(f"\033[{total_lines}A")

    try:
        term_width = min(os.get_terminal_size().columns, 130)
    except OSError:
        term_width = 130

    def line(s: str) -> None:
        visible = re.sub(r"\033\[[^m]*m", "", s)
        if len(visible) > term_width:
            s = s[:term_width + (len(s) - len(visible))] + "\033[0m"
        print(f"{s}\033[K")

    stage_eng = "Formation" if stage_title == "形成" else "Trading"

    line("\033[95m" + "═" * term_width + "\033[0m")
    line(f"        \033[93;1m🚀 配對交易{stage_title}期平行化即時監控儀表板 ({stage_eng} Stage Core) 🚀\033[0m")
    line("\033[95m" + "═" * term_width + "\033[0m")

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

        display_name = name

        status_pad = _pad_visible(status_str, 10)
        name_pad   = _pad_visible(display_name[:30], 30)
        bar_pad    = _pad_visible(bar, 5)
        msg_pad    = _pad_visible(msg[:35], 35)

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

class ProgressAwareStdout:
    """
    攔截子行程的 stdout，解析指定關鍵字 (Window 或 Period) 並即時更新跨行程進度字典。
    """
    def __init__(self, log_filepath: str, progress_dict, strategy_name: str, total_steps: int, pattern_keyword: str = "Window"):
        self.log_file      = None
        self.progress_dict = progress_dict
        self.strategy_name = strategy_name
        self.total_steps   = total_steps
        self.start_time    = time.time()
        self.pattern       = re.compile(rf"{pattern_keyword}\s*(\d+)/(\d+)")

        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> None:
        self.log_file.write(s)
        match = self.pattern.search(s)
        if match:
            try:
                curr_step = int(match.group(1))
                total     = int(match.group(2))
                pct       = min(100, int(curr_step / total * 100))

                current_info = dict(self.progress_dict.get(self.strategy_name, {}))
                current_info.update({
                    "status":   "RUNNING",
                    "progress": f"{curr_step}/{total}",
                    "pct":      pct,
                    "msg":      f"正在處理第 {curr_step:03d}/{total:03d} 期配對計算",
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

# ── 儀表板執行緒管理 helper ───────────────────────────────────────────────
def start_dashboard_thread(progress_dict, original_strategies_config, main_start_time, log_dir, stop_event, stage_title="形成"):
    """啟動儀表板背景執行緒，回傳 thread 物件"""
    def updater():
        while not stop_event.is_set():
            draw_dashboard(progress_dict, original_strategies_config, main_start_time, log_dir_desc=log_dir, stage_title=stage_title)
            time.sleep(0.3)
    t = threading.Thread(target=updater, daemon=True)
    t.start()
    return t

# ── Interleave 打散工具函數 ───────────────────────────────────────────────
def interleave_strategies(strategies_to_run, original_strategies_config):
    """將相同原始策略的子任務交錯排列"""
    from collections import defaultdict
    groups = defaultdict(list)
    for cfg in strategies_to_run:
        orig_name = next(
            (orig["name"] for orig in original_strategies_config if cfg["name"].startswith(orig["name"])),
            cfg["name"]
        )
        groups[orig_name].append(cfg)
    
    interleaved = []
    max_len = max(len(v) for v in groups.values()) if groups else 0
    for i in range(max_len):
        for orig_name in groups:
            if i < len(groups[orig_name]):
                interleaved.append(groups[orig_name][i])
    return interleaved

# ── 終端總結報告函數 ───────────────────────────────────────────────────────
def print_summary_report(results, strategies_config, total_elapsed, show_equity=False):
    """
    show_equity=False 對應 run_formation（無 final_equity 欄位）
    show_equity=True  對應 run_trading（有 final_equity 欄位）
    """
    print("\n" + "=" * 80, flush=True)
    stage_name = "交易期回測" if show_equity else "形成期配對"
    print(f"                     📊 {stage_name}計算執行績效總結報告 (Summary) 📊", flush=True)
    print("=" * 80, flush=True)
    print(f" 總耗時: {total_elapsed:.2f} 秒（約 {total_elapsed / 60:.2f} 分鐘）", flush=True)
    
    if show_equity:
        print(
            f"\n{'策略名稱':<45} | {'狀態':<10} | {'跳過':<4} | {'最終權益':<10} | {'耗時(秒)':<10} | 錯誤訊息",
            flush=True,
        )
        print("-" * 110, flush=True)
    else:
        print(
            f"\n{'策略名稱':<45} | {'狀態':<10} | {'跳過':<4} | {'耗時(秒)':<10} | 錯誤訊息",
            flush=True,
        )
        print("-" * 90, flush=True)
        
    name_order = {c["name"]: i for i, c in enumerate(strategies_config)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["name"], 999))
    
    for res in results_sorted:
        err = res.get("error") or "無"
        skipped = "是" if res.get("skipped") else "否"
        if show_equity:
            final_eq = res.get("final_equity", 0.0)
            final_eq_str = f"${final_eq:.2f}"
            print(
                f"{res['name']:<45} | {res['status']:<10} | {skipped:<4} | {final_eq_str:<10} | {res['elapsed']:<10.2f} | {err}",
                flush=True,
            )
        else:
            print(
                f"{res['name']:<45} | {res['status']:<10} | {skipped:<4} | {res['elapsed']:<10.2f} | {err}",
                flush=True,
            )
    print("=" * 80, flush=True)
