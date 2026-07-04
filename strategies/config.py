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
    if 'folder' not in multiprocessing.resource_tracker._CLEANUP_FUNCS:  # type: ignore
        multiprocessing.resource_tracker._CLEANUP_FUNCS['folder'] = lambda x: None  # type: ignore
    if 'file' not in multiprocessing.resource_tracker._CLEANUP_FUNCS:  # type: ignore
        multiprocessing.resource_tracker._CLEANUP_FUNCS['file'] = lambda x: None  # type: ignore
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
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore

# ── 共用常數 ─────────────────────────────────────────────────────────────
FORCE_RERUN = False
CPU_LIMIT_PCT = 0.8
DRL_MAX_WORKERS = 45  # H200 143GB VRAM，每個 worker ~856MiB，45組同時跑綽綽有餘
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

# 無風險利率年化假設（Excess_Ret_RF 口徑用；市場中性策略的閒置現金與保證金收 rf）
# 2000–2025 美國 3M T-bill 平均約 1.8–2.0%；可日後換成實際序列
RF_ANNUAL = 0.02

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
    "max_sector_ratio_list":        [0.0],
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

# 2026-07-03 精簡：16 個已淘汰/驗證無效的策略移至 archive/config_archived_strategies.py
# （SSD Basic、DTW 原版×2〔座標 artifact，形成配對仍被 Fixed 版借用〕、
#   HDBSCAN stats10 系×4、Ensemble×2、DRLv1×3、Kalman×2、CONV×2）。
# 歷史回測結果保留於 results/result.db；復活方式見封存檔 docstring。
strategies_raw_all = [
    # ── 基準 ────────────────────────────────────────────────────────────────
    # 1. SSD Rolling（SSD 家族代表基準；亦為 #7 DRL 對照的形成來源）
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
    # ── HDBSCAN 形成法（命題 1 主線） ────────────────────────────────────────
    # 2. HDBSCAN PCA-Loadings（第一層特徵 = 報酬 PCA 因子載荷，
    #     reduce_method="none" 跳過 UMAP，直接以 loadings 餵 HDBSCAN）
    {
        "name":             "HDBSCAN PCA-Loadings",
        "formation_module": "strategies.formation.HDBSCAN_PCA_Loadings",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_PCA_Loadings",
        "db_method":        "HDBSCAN (PCA-Loadings)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS,
            "reduce_method":     "none",
            "pca_n_components":  15,   # 報酬 PCA 因子數（Avellaneda & Lee 2010 用 ~15）
            "feature_mode":      "pca_loadings",
            # T2 實驗：進場門檻 × 發散停損聯合掃描（SL 固定 0 以控制網格大小）
            "stop_loss_list":       [0.0],
            "entry_z_list":         [1.5, 2.0, 2.5],
            "dynamic_stop_z_list":  [0.0, 3.0, 4.0],
        },
    },
    # ── DTW 基準（座標修正版；原版為 artifact 已封存） ───────────────────────
    # DTW Paper 原版的 OLS 在標準化空間擬合但輸出 OLS_Alpha → 交易端路徑 A
    # 以原始 log-price 空間重建 spread → 常數 Z 偏移（詳見封存檔說明）。
    # #3/#4 借用原版的形成期配對（formation_strategy_id_base），
    # 以 ignore_ols_alpha 強制路徑 B（標準化空間），為誠實的 Z-Score 基準。
    # 3. DTW Paper Fixed（座標修正版）
    {
        "name":             "DTW Paper Fixed (DTW)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "formation_strategy_id_base": "DTW Paper (DTW)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "DTW_Paper_Fixed",
        "db_method":        "DTW (Paper-Fixed)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method": "dtw",
            "ignore_ols_alpha": True,   # 強制路徑 B：與形成期一致的標準化空間
        },
    },
    # 4. SSD-DTW-PCA Paper Fixed（座標修正版，目前最佳誠實基準：Top3 Sharpe 0.49）
    {
        "name":             "SSD-DTW-PCA Paper Fixed",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "formation_strategy_id_base": "DTW Paper (SSD-DTW-PCA)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_DTW_PCA_Paper_Fixed",
        "db_method":        "SSD-DTW-PCA (Paper-Fixed)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method": "ssd_dtw_pca",
            "ignore_ols_alpha": True,
            # T2 實驗：進場門檻 × 發散停損聯合掃描（SL 固定 0 以控制網格大小）
            "stop_loss_list":       [0.0],
            "entry_z_list":         [1.5, 2.0, 2.5],
            "dynamic_stop_z_list":  [0.0, 3.0, 4.0],
        },
    },
    # 5. HDBSCAN Cluster SSD-DTW-PCA（分組消融：對照組 = #4 GICS 分組，
    #    排序與交易端完全相同。295 期結果：報酬統計上等效（月配對 t 檢定不顯著），
    #    但不依賴 GICS 元資料、可救回 Unknown 股、形成期快 2 倍）
    {
        "name":             "HDBSCAN Cluster SSD-DTW-PCA",
        "formation_module": "strategies.formation.HDBSCAN_Cluster_SSD_DTW",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_Cluster_SSD_DTW",
        "db_method":        "HDBSCAN (Cluster-SSD-DTW-PCA)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method":               "ssd_dtw_pca",
            "pca_n_components":     15,
            "hdbscan_min_cluster_size": 5,
            "hdbscan_min_samples":  2,
            "umap_random_state":    42,
            "adf_pvalue_threshold": 0.01,
            "ignore_ols_alpha":     True,   # 路徑 B：與對照組交易端一致
        },
    },
    # ── DRL v4：門檻選擇式（Kim & Kim 2019）──────────────────────────────
    #   v1–v3 逐日定位動作空間已系統性證偽（OOS 過度交易，Sharpe −1.1~−2.3，
    #   FQI 系列 2026-07-04 封存至 archive/config_archived_strategies.py）。
    #   v4 每配對每期只選一個動作：SKIP + 8 組 (entry_z, exit_z) 門檻，
    #   選單包含基準 (2.0, 0.0) → 策略空間 ⊇ Z-Score；訓練樣本不足時自動用基準。
    #   反事實全資訊標籤 + walk-forward 監督回歸（無探索問題）。
    #   #6 vs #2（同 HDBSCAN PCA-Loadings 配對，DRL vs Z-Score）→ 命題 2
    # 6. HDBSCAN PCA-Loadings DRL THR
    {
        "name":             "HDBSCAN PCA-Loadings DRL THR",
        "formation_module": "strategies.formation.HDBSCAN_PCA_Loadings",
        "formation_strategy_id_base": "HDBSCAN PCA-Loadings",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "HDBSCAN_PCA_Loadings_DRL_THR",
        "db_method":        "HDBSCAN (PCA-Loadings-DRL-THR)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
    # 7. SSD Rolling DRL THR
    {
        "name":             "SSD Rolling DRL THR",
        "formation_module": "strategies.formation.ssd_rolling",
        "formation_strategy_id_base": "SSD Rolling",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "SSD_Rolling_DRL_THR",
        "db_method":        "SSD (Rolling-DRL-THR)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
    # 8. HDBSCAN Cluster SSD-DTW-PCA DRL THR —— 命題 2 的第二個 ML 配對底。
    #    #6（PCA-Loadings 分群 + DRL）已證實：即使修好 _shared 跨變體污染的 bug，
    #    15 組全負 Sharpe（中位數 −0.41）維持不變，問題在配對訊號本身太弱，不是
    #    交易端。#5 的 Z-Score 基準（Sharpe 0.15、年化 0.82%）是 #2 的 2–3 倍，
    #    換成這個更穩的 ML 分群底再疊 DRL-THR，看能不能重現 #7（SSD+DRL）
    #    那種「疊加後全網格轉正、且優於自身 Z-Score 基準」的效果。
    {
        "name":             "HDBSCAN Cluster SSD-DTW-PCA DRL THR",
        "formation_module": "strategies.formation.HDBSCAN_Cluster_SSD_DTW",
        "formation_strategy_id_base": "HDBSCAN Cluster SSD-DTW-PCA",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "HDBSCAN_Cluster_SSD_DTW_DRL_THR",
        "db_method":        "HDBSCAN (Cluster-SSD-DTW-PCA-DRL-THR)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
    # 命題 2 方案 C：#4 的 Z-Score 基準（Sharpe 0.15~0.22）平均每期有 30.9%
    # （中位數 35.8%）標的被 HDBSCAN 判為雜訊（label=-1）排除，群數在 2~25
    # 之間劇烈震盪；15 維 PCA loadings 平均僅解釋 58.6% 報酬變異，後段主成分
    # 訊號弱、卻仍等權參與歐氏距離計算，可能是維度詛咒稀釋了密度訊號的元兇。
    # 診斷驗證有效：雜訊比例中位數 35.8%→8.1%、群數標準差 6.4→4.3；
    # Z-Score 基準正 Sharpe 比例 60%→73%、中位數 Sharpe 0.07→0.13、
    # 最佳 Sharpe 0.22→0.29、最佳年化 0.82%→1.08%，全面優於 15 維版本。
    # 接下來疊 DRL-THR 驗證方案 C 是否能真正跨過 2% 門檻。
    {
        "name":             "HDBSCAN Cluster SSD-DTW-PCA PCA5",
        "formation_module": "strategies.formation.HDBSCAN_Cluster_SSD_DTW",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_Cluster_SSD_DTW_PCA5",
        "db_method":        "HDBSCAN (Cluster-SSD-DTW-PCA-PCA5)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method":               "ssd_dtw_pca",
            "pca_n_components":     5,
            "hdbscan_min_cluster_size": 5,
            "hdbscan_min_samples":  2,
            "umap_random_state":    42,
            "adf_pvalue_threshold": 0.01,
            "ignore_ols_alpha":     True,
        },
    },
    # 9. HDBSCAN Cluster SSD-DTW-PCA PCA5 DRL THR —— 命題 2 方案 C 的完整驗證：
    #    在降維後更穩的 ML 配對底上疊 DRL-THR，看正 Sharpe 比例是否再被推到
    #    100%（重現 #7/#9 的模式），年化報酬是否真的跨過 2% 且同時贏過 SSD。
    {
        "name":             "HDBSCAN Cluster SSD-DTW-PCA PCA5 DRL THR",
        "formation_module": "strategies.formation.HDBSCAN_Cluster_SSD_DTW",
        "formation_strategy_id_base": "HDBSCAN Cluster SSD-DTW-PCA PCA5",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "HDBSCAN_Cluster_SSD_DTW_PCA5_DRL_THR",
        "db_method":        "HDBSCAN (Cluster-SSD-DTW-PCA-PCA5-DRL-THR)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
]
# 2026-07-04 清理：FQI 系列×3（逐日定位動作空間已證偽）、重建任務完成的
# 拉回策略×4（DTW 原版×2、CONV×2）歸位 archive/config_archived_strategies.py。
# 歷史結果均在 results/result.db；復活方式見封存檔 docstring。

# 共 8 個策略（#1–#5 Z-Score 完整；#6–#8 DRL THR = STRATEGIES_SLICE="5:8"）
# ⚠️ FORCE_RERUN=True 時會忽略斷點續傳全部重算
strategies_raw = strategies_raw_all[:]

# ── 跨機器免改檔覆寫：環境變數 STRATEGIES_SLICE ───────────────────────────
# 本機 PowerShell：  $env:STRATEGIES_SLICE="5:7"; python run_trading.py
# H200 Linux：       STRATEGIES_SLICE="-1:" python run_trading.py
# 語法同 Python 切片，支援逗號複合："5:7"、"-1:"、":"、"2"、"0:5,8:12"
_env_slice = os.environ.get("STRATEGIES_SLICE", "").strip()
if _env_slice:
    try:
        _picked = []
        for _part in _env_slice.split(","):
            _part = _part.strip()
            if ":" in _part:
                _lo, _hi = _part.split(":", 1)
                _picked += strategies_raw_all[
                    int(_lo) if _lo else None : int(_hi) if _hi else None
                ]
            else:
                _picked.append(strategies_raw_all[int(_part)])
        strategies_raw = _picked
        print(f"[config] STRATEGIES_SLICE={_env_slice} → 執行 {[s['name'] for s in strategies_raw]}")
    except (ValueError, IndexError):
        print(f"⚠️ [config] STRATEGIES_SLICE='{_env_slice}' 無法解析，使用預設範圍")


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
        self.log_file.write(s)  # type: ignore
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
