import os
import sys
import time
import re
import copy
import threading
import json
import unicodedata
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
# 交易期是否另存人類可讀的 Trade Log CSV。續傳與儀表板均以 result.db 為準，
# 故 CSV 為可選產物；False 省下 ~14GB（大型網格）且不影響回測/續傳/儀表板。
# 2026-07-08 起關閉：改採純 result.db 工作流。需人類可讀 CSV 時再設 True。
WRITE_TRADE_CSV = False
CPU_LIMIT_PCT = 0.95
DRL_MAX_WORKERS = 10  # 本機 CPU-only（無 CUDA），torch.set_num_threads(1)，10 並行安全
DB_PROFILES = {
    "sp500_Current": {
        "db_path":     "./dataset/price/sp500_Current.db",
        "output_root": "./results/current",
        "label":       "S&P 500 現行成分股 (Current)",
    },
    "sp500_yF": {
        "db_path":     "./dataset/price/sp500_yF.db",
        "output_root": "./results/yFinance",
        "label":       "S&P 500 完整歷史成分股 (yFinance)",
    },
    "sp500_Tiingo": {
        "db_path":     "./dataset/price/sp500_Tiingo.db",
        "output_root": "./results/tiingo",
        "label":       "S&P 500 完整歷史成分股 (Tiingo)",
    },
}

DB_PATH = "./dataset/price/sp500_Tiingo.db"
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
    # 單邊交易成本 0.29%（= 29 bps），依 Do & Faff (2012) 對美股 pairs trading
    # 單邊成本約 30 bps（佣金 + 市場衝擊）之估計。friction = fee_rate + slippage_rate
    # 於每次進場、出場各按部位名目額扣一次（故每配對一完整往返 ≈ 0.58% 名目額）。
    # Do, B. H., & Faff, R. (2012). Are pairs trading profits robust to trading
    #   costs? Journal of Financial Research, 35(2), 261–287.
    "fee_rate":                     0.0029,
    "slippage_rate":                0.0,
    "initial_capital":              INITIAL_CAPITAL,
    "allow_reentry":                False,
    "zscore_clip":                  10.0,
    "min_spread_std":               1e-6,
    "min_tickers_for_pairing":      2,
    "use_vol_adjust":               use_vol_adjust,
    # P1 時間停損：2026-07-19 起 zscore_trading 已實際接線此參數（TIME_STOP）。
    # 預設 0 = 停用，維持全部既有策略行為不變；啟用見 FMP TS 系列 A/B 條目。
    # （舊值 30 從未被引擎使用，改為 0 以免接線後意外改變所有策略。）
    "max_holding_days":             0,
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
# 2026-07-04 二次精簡：再封存 3 個確認無效的策略（HDBSCAN PCA-Loadings 及其
# DRL THR 變體、ML Pair Quality）——原因與數據見 archive/config_archived_strategies.py
# docstring「HDBSCAN PCA-Loadings 系列」「ML Pair Quality」兩節。
# 歷史回測結果保留於 results/result.db；復活方式見封存檔 docstring。
strategies_raw_all = [
    # ── 基準 ────────────────────────────────────────────────────────────────
    # 0. SSD Basic（Gatev et al. 2006 原型：累積回報指數 + 固定 β=1；
    #    一切策略的基礎原型，2026-07-06 自封存拉回現役）
    {
        "name":             "SSD Basic",
        "formation_module": "strategies.formation.ssd_basic",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_Basic",
        "db_method":        "SSD (Basic)",
        "trade_method":     "Z-Score",
        "params":  {**base_params},
    },
    # 1. SSD Rolling（SSD 家族代表基準；亦為 DRL / 距離對照的形成來源）
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
    # 1b. SSD Rolling Distance（距離基準交易，Gatev et al. 2006 GGR 距離法）
    #     借用 SSD Rolling 的形成期配對，唯一變因 = 交易端 spread 空間：
    #       回歸基準（#1 zscore_trading）：spread = OLS 共整合殘差
    #       距離基準（本策略 distance_trading）：spread = 正規化價格距離（等權、hedge=1）
    #     構成「回歸 vs 距離」的乾淨單變因對照（指導教授指定）。
    {
        "name":             "SSD Rolling Distance",
        "formation_module": "strategies.formation.ssd_rolling",
        "formation_strategy_id_base": "SSD Rolling",
        "trading_module":   "strategies.trading.distance_trading",
        "sub_dir":          "SSD_Rolling_Distance",
        "db_method":        "SSD (Distance)",
        "trade_method":     "Distance",
        "params":  {
            **base_params,
        },
    },
    # ── DTW 基準（座標修正版；原版為 artifact 已封存） ───────────────────────
    # DTW Paper 原版的 OLS 在標準化空間擬合但輸出 OLS_Alpha → 交易端路徑 A
    # 以原始 log-price 空間重建 spread → 常數 Z 偏移（詳見封存檔說明）。
    # #2/#3 借用原版的形成期配對（formation_strategy_id_base），
    # 以 ignore_ols_alpha 強制路徑 B（標準化空間），為誠實的 Z-Score 基準。
    # 2. DTW Paper Fixed（座標修正版）
    {
        "name":             "DTW Paper Fixed (DTW)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "formation_strategy_id_base": "DTW Paper (DTW)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "DTW_Paper_Fixed",
        "db_method":        "DTW",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method": "dtw",
            "ignore_ols_alpha": True,   # 強制路徑 B：與形成期一致的標準化空間
        },
    },
    # 3. SSD-DTW-PCA Paper Fixed（座標修正版，目前最佳誠實基準：Top3 Sharpe 0.56）
    {
        "name":             "SSD-DTW-PCA Paper Fixed",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "formation_strategy_id_base": "DTW Paper (SSD-DTW-PCA)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_DTW_PCA_Paper_Fixed",
        "db_method":        "SSD-DTW-PCA",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method": "ssd_dtw_pca",
            "ignore_ols_alpha": True,
            # T2 實驗（entry_z × dynamic_stop_z 網格）已於 2026-07-06 拆除：
            # 結論 = DSZ 全面有害（停掉 67% 交易、勝率 61%→33%）、EZ 2.5 微幅較優。
            # 歷史結果保留於 result.db（ENTRY Z / DYN Z NUM 欄位可篩）。
        },
    },
    # （HDBSCAN Cluster SSD-DTW-PCA 15 維版與其 DRL THR、PCA5 DRL THR 已於
    #   2026-07-06 封存——HDBSCAN 系列僅保留 PCA5 Z-Score 版作為分組消融對照組，
    #   見 archive/config_archived_strategies.py）
    # ── DRL v4：門檻選擇式（Kim & Kim 2019）──────────────────────────────
    #   v1–v3 逐日定位動作空間已系統性證偽（OOS 過度交易，Sharpe −1.1~−2.3，
    #   FQI 系列 2026-07-04 封存至 archive/config_archived_strategies.py）。
    #   v4 每配對每期只選一個動作：SKIP + 8 組 (entry_z, exit_z) 門檻，
    #   選單包含基準 (2.0, 0.0) → 策略空間 ⊇ Z-Score；訓練樣本不足時自動用基準。
    #   反事實全資訊標籤 + walk-forward 監督回歸（無探索問題）。
    #   HDBSCAN PCA-Loadings DRL THR（命題 2 的第一次嘗試，vs HDBSCAN PCA-Loadings
    #   Z-Score）已封存：修好 _shared 跨變體污染的 bug 後結果不變（15 組全負
    #   Sharpe，中位數 −0.41），確認是配對訊號本身太弱，非 DRL 交易端問題。
    #   見 archive/config_archived_strategies.py。目前活躍對照改為 #1 vs #5、
    #   #4 vs #6、#7 vs #8。
    # 5. SSD Rolling DRL THR
    {
        "name":             "SSD Rolling DRL THR",
        "formation_module": "strategies.formation.ssd_rolling",
        "formation_strategy_id_base": "SSD Rolling",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "SSD_Rolling_DRL_THR",
        "db_method":        "SSD (DRL)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
    # 6. HDBSCAN Cluster SSD-DTW-PCA PCA5 —— HDBSCAN 系列唯一保留的分組消融
    #    對照組（5 維報酬 PCA 因子載荷聚類 + SSD-DTW-PCA 排序 + 路徑 B 交易）
    {
        "name":             "HDBSCAN Cluster SSD-DTW-PCA PCA5",
        "formation_module": "strategies.formation.HDBSCAN_Cluster_SSD_DTW",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_Cluster_SSD_DTW_PCA5",
        "db_method":        "HDBSCAN",
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
    # 6b. 同上 + 因子殘差化（研究框架 #1 消融）：聚類前移除市場+產業因子，
    #     PCA 建於特殊性報酬，理論上降低偽相關、提升 regime 穩健性。
    #     與 #6 唯一差異 = factor_residual=True → 直接對照殘差化的增量貢獻。
    {
        "name":             "HDBSCAN Cluster SSD-DTW-PCA PCA5 Resid",
        "formation_module": "strategies.formation.HDBSCAN_Cluster_SSD_DTW",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_Cluster_SSD_DTW_PCA5_Resid",
        "db_method":        "HDBSCAN (殘差)",
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
            "factor_residual":      True,
        },
    },
    # 7. Agglomerative Fundamentals —— 分組消融第三支：以「報酬 PCA loadings
    #     ⊕ GICS one-hot ⊕ log(市值) ⊕ 盈餘殖利率(1/PE)」混合特徵空間做
    #     Agglomerative（average-linkage、依合併距離分位數校準
    #     distance_threshold，非固定 n_clusters/ward，避免重現 HDBSCAN 的
    #     群組不平衡問題）分群，取代 GICS 靜態分組；分群結果轉為
    #     sector_mapping 餵給既有 SSD Rolling 排序/共整合流程完全不變
    #     （min-SSD + Engle-Granger ADF + Hurst）。交易端沿用既有 Z-Score /
    #     Beta-動態配重（EG Hedge Ratio）/ 停損引擎，未做任何修改。
    #     基本面資料為單一時點靜態快照（yfinance，見
    #     fetch/fundamentals_yfinance.py）——已知前視偏誤限制。
    {
        "name":             "Agglomerative Fundamentals (yF)",
        "formation_module": "strategies.formation.agglomerative_yF",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_yF",
        "db_method":        "Agglomerative (yF)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "pca_n_components":            5,
            "fundamentals_db_path":        "./dataset/fundamental/fundamentals_sp500.db",
            "price_feature_weight":        1.0,
            "fundamentals_feature_weight": 1.0,
            "sector_onehot_weight":        1.0,
            "agg_linkage":                 "average",
            "agg_threshold_percentile":    75.0,
            "min_cluster_size":            5,
            "adf_pvalue_threshold":        0.05,
        },
    },
    # （DynCap 動態槽位配置：2026-07-19 驗證為負面結果——集中配置放大波動拖累，
    #   最佳年化 1.80%→1.62%、Top10/20 MDD 惡化至 -58%。已封存至
    #   archive/config_archived_strategies.py「DynCap」節；PortfolioManager 的
    #   dynamic_slots 機制保留供復活。結論：閒置現金是低波動的代價，
    #   配置端集中無法免費兌現「動用資本年化」。）
    # 8. Agglomerative Fundamentals DRL THR —— DRL-THR 疊加在 Agglomerative
    #    Fundamentals 配對底上（借用 #7 的形成期配對，不需重跑形成期）。
    {
        "name":             "Agglomerative Fundamentals DRL THR (yF)",
        "formation_module": "strategies.formation.agglomerative_yF",
        "formation_strategy_id_base": "Agglomerative Fundamentals (yF)",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "Agglomerative_Fundamentals_DRL_THR_yF",
        "db_method":        "Agglomerative (yF·DRL)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
    # 9. Agglomerative Fundamentals FMP —— 升級為 FMP 長週期 Point-in-Time 數據，無前視偏誤
    {
        "name":             "Agglomerative Fundamentals (FMP)",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP",
        "db_method":        "Agglomerative (FMP)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "pca_n_components":            5,
            "fundamentals_parquet_path":   "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
            "price_feature_weight":        1.0,
            "fundamentals_feature_weight": 1.0,
            "sector_onehot_weight":        1.0,
            "agg_linkage":                 "average",
            "agg_threshold_percentile":    75.0,
            "min_cluster_size":            5,
            "adf_pvalue_threshold":        0.05,
        },
    },
    # 9b. Agglomerative Fundamentals (FMP) DTW —— 教授要求：分群相同、群內排序端
    #     由 min-SSD 換成 DTW（雙向 OLS + ADF + Sakoe-Chiba DTW 距離升序）。
    #     經中性排序層 _ranking(ranking_backend="dtw") 組裝，與 FMP 基準單一變因對照。
    #     DTW 排序端輸出 OLS_Alpha → 交易端需 ignore_ols_alpha=True 走標準化空間（路徑 B）。
    {
        "name":             "Agglomerative Fundamentals (FMP) DTW",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_DTW",
        "db_method":        "Agglomerative (FMP-DTW)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "pca_n_components":            5,
            "fundamentals_parquet_path":   "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
            "price_feature_weight":        1.0,
            "fundamentals_feature_weight": 1.0,
            "sector_onehot_weight":        1.0,
            "agg_linkage":                 "average",
            "agg_threshold_percentile":    75.0,
            "min_cluster_size":            5,
            "adf_pvalue_threshold":        0.05,
            "ranking_backend":             "dtw",
            "dtw_window":                  15,
            "ignore_ols_alpha":            True,
        },
    },
    # 9c. Agglomerative Fundamentals (FMP) SSD-DTW-PCA —— 第三種群內排序：
    #     SSD 與 DTW 距離標準化後 PCA 融合、取第一主成分升序（許鈞翔 2025 實驗組）。
    #     同樣經中性排序層 _ranking(ranking_backend="ssd_dtw_pca") 組裝——展示中性化
    #     後新增排序準則零改碼。與 FMP(SSD)、FMP-DTW 構成三種排序的單變因對照。
    {
        "name":             "Agglomerative Fundamentals (FMP) SSD-DTW-PCA",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_SDP",
        "db_method":        "Agglomerative (FMP-SSD-DTW-PCA)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "pca_n_components":            5,
            "fundamentals_parquet_path":   "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
            "price_feature_weight":        1.0,
            "fundamentals_feature_weight": 1.0,
            "sector_onehot_weight":        1.0,
            "agg_linkage":                 "average",
            "agg_threshold_percentile":    75.0,
            "min_cluster_size":            5,
            "adf_pvalue_threshold":        0.05,
            "ranking_backend":             "ssd_dtw_pca",
            "dtw_window":                  15,
            "ignore_ols_alpha":            True,
        },
    },
    # （方案 C「Agglo-FMP × 因子殘差」：2026-07-19 驗證為負面結果——15 網格
    #   0 個正 Sharpe（中位 -0.69），最佳年化 -0.90% vs 基準 2.28%。根因：殘差化
    #   表徵與下游原始價格空間 SSD 排序錯位，且與 GICS one-hot 特徵互相抵銷。
    #   已封存至 archive/config_archived_strategies.py「FMP-Resid」節；
    #   agglomerative_FMP.factor_residual 參數保留供復活。）
    # （P1/P2/P4 交易端優化戰役：2026-07-19 全數封存為負面/中性結果——
    #   TS63/TS42 時間停損（後見之明偏誤：63d 為損益低谷，砍在谷底）、
    #   XZ05 提早出場（省的風險 < 放棄的收斂利潤）、DRL 選單 v5 時間維度
    #   （重訓噪音範圍內，agent 學會不用它）。完整診斷與數據見
    #   archive/config_archived_strategies.py「交易端微觀規則戰役」節。
    #   結論：交易端微觀規則已系統性掃過，2.25-2.28% 為現有訊號組天花板；
    #   僅存的實證槓桿為 regime 條件化進場（見下方 DG 系列 A/B）。）
    # ── P3 A/B：regime 條件化進場（低分散度閘門，2026-07-19）───────────────
    # 依據：FMP Top1 日損益依橫斷面分散度四分位分層——Q4 貢獻 +8,674、
    # Q1+Q2 合計 -4,920（全部淨利來自分散度最高的 25% 交易日）。
    # 機制：walk-forward 30 日分散度的歷史分位 < pctl 時暫停「新開倉」
    # （持倉/出場不受影響；expanding 分位 min 252 日暖身，無前視）。
    {
        "name":             "Agglomerative Fundamentals (FMP) DG50",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_DG50",
        "db_method":        "Agglomerative (FMP-DG50)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "adf_pvalue_threshold": 0.05,
                   "disp_gate_pctl": 50.0},
    },
    {
        "name":             "Agglomerative Fundamentals (FMP) DG25",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_DG25",
        "db_method":        "Agglomerative (FMP-DG25)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "adf_pvalue_threshold": 0.05,
                   "disp_gate_pctl": 25.0},
    },
    # P3 組合格：DG25 × 高進場門檻（兩個獨立正效應疊加，朝 3% 目標）
    # 基準 EZ 掃描最佳：EZ3.0 Top1 SL0 = 2.28%、EZ2.5 = 2.10%；DG25 全格 +0.2~0.5pp
    {
        "name":             "Agglomerative Fundamentals (FMP) DG25 EZ",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_DG25_EZ",
        "db_method":        "Agglomerative (FMP-DG25-EZ)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "adf_pvalue_threshold": 0.05,
                   "disp_gate_pctl": 25.0,
                   "stop_loss_list": [0.0],
                   "entry_z_list":   [2.5, 3.0]},
    },
    # P3 組合格：DG25 × DRL 疊加（閘門同時作用於反事實標籤與正式模擬，
    # 訓練標籤 = 閘門下可實現的報酬，無標籤/執行不一致）
    {
        "name":             "Agglomerative Fundamentals DRL THR (FMP) DG25",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_DRL_DG25",
        "db_method":        "Agglomerative (FMP·DRL-DG25)",
        "trade_method":     "DRL",
        "params": {**base_params, "adf_pvalue_threshold": 0.05,
                   "drl_hidden_size": 64, "thr_train_epochs": 40,
                   "thr_min_train_samples": 200,
                   "disp_gate_pctl": 25.0},
    },
    # 10. Agglomerative Fundamentals DRL THR FMP —— DRL-THR 疊加在 FMP 版本上
    {
        "name":             "Agglomerative Fundamentals DRL THR (FMP)",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "Agglomerative_Fundamentals_DRL_THR_FMP",
        "db_method":        "Agglomerative (FMP·DRL)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_hidden_size": 64,
            "thr_train_epochs": 40,
            "thr_min_train_samples": 200,
        },
    },
    # ── DTW Paper 原版（formation-only：僅產生形成期配對供 #2/#3 借用） ──────
    # 2026-07-05：repo LFS 額度用罄，formation_pairs DB 無法下載，原版配對
    # 需本地重算。原版「交易端」為座標 artifact 已封存（見 archive/
    # config_archived_strategies.py），故以 formation_only 旗標讓 run_trading
    # 跳過回測，只由 run_formation 產生 DTW Paper (DTW)/(SSD-DTW-PCA) 配對。
    # 置於清單尾端以保持 #1–#10 的 STRATEGIES_SLICE 索引穩定
    # （注意："-1:" 之類的尾端切片現在會切到 formation-only 條目）。
    {
        "name":             "DTW Paper (DTW)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "trading_module":   "strategies.trading.zscore_trading",
        "formation_only":   True,
        "sub_dir":          "DTW_Paper",
        "db_method":        "DTW (Paper)",
        "trade_method":     "Z-Score",
        "params":  {**base_params, "method": "dtw"},
    },
    {
        "name":             "DTW Paper (SSD-DTW-PCA)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "trading_module":   "strategies.trading.zscore_trading",
        "formation_only":   True,
        "sub_dir":          "SSD_DTW_PCA_Paper",
        "db_method":        "SSD-DTW-PCA (Paper)",
        "trade_method":     "Z-Score",
        "params":  {**base_params, "method": "ssd_dtw_pca"},
    },
]
# 2026-07-04 清理：FQI 系列×3、CONV×2 等歸位 archive/config_archived_strategies.py。
# 2026-07-06 調整：SSD Basic（基礎原型）自封存拉回現役；HDBSCAN 系列僅保留
# PCA5 Z-Score 版作為分組消融對照組（15 維版、15 維 DRL THR、PCA5 DRL THR 封存）。
# 歷史結果均在 results/result.db；復活方式見封存檔 docstring。

# 共 8 個交易策略（#1 SSD Basic、#2 SSD Rolling、#3 DTW Paper Fixed、
# #4 SSD-DTW-PCA Paper Fixed、#6 HDBSCAN PCA5、#7 Agglomerative Z-Score；
# #5 SSD Rolling DRL THR、#8 Agglomerative DRL THR）
# ＋ 2 個 formation-only 條目（DTW Paper 原版 ×2，只產生配對不回測）
# ⚠️ FORCE_RERUN=True 時會忽略斷點續傳全部重算

# ── 3×3 分群 × 排序 消融矩陣（2026-07 中性化重構後宣告式展開）──────────────
# 固定特徵 = 混合特徵（報酬 PCA ⊕ FMP PIT 基本面 ⊕ GICS one-hot），三種分群共用；
# 唯一變因 = 分群方法 × 群內排序準則。全部指向中性組裝器 cluster_formation。
# K-means 群數對齊同期 Agglomerative（見 cluster_formation._cluster）。
# 註：AGG-SSD 格經回歸測試證實 bit-identical 復現現役 Agglomerative (FMP)。
_GRID_COMMON = {
    "pca_n_components":            5,
    "fundamentals_parquet_path":   "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
    "price_feature_weight":        1.0,
    "fundamentals_feature_weight": 1.0,
    "sector_onehot_weight":        1.0,
    "agg_linkage":                 "average",
    "agg_threshold_percentile":    75.0,
    "min_cluster_size":            5,
    "hdbscan_min_cluster_size":    5,
    "hdbscan_min_samples":         2,
    "adf_pvalue_threshold":        0.05,
    "dtw_window":                  15,
}
_GRID_CLUSTERS = {"hdbscan": "HDB", "agglomerative": "AGG", "kmeans": "KM"}
_GRID_RANKINGS = {"ssd": "SSD", "dtw": "DTW", "ssd_dtw_pca": "SDP"}
_grid_entries = []
for _cm, _cs in _GRID_CLUSTERS.items():
    for _rb, _rs in _GRID_RANKINGS.items():
        _p = {**base_params, **_GRID_COMMON,
              "feature_mode": "fundamentals_mix",
              "cluster_method": _cm, "ranking_backend": _rb}
        if _rb != "ssd":     # DTW 排序端輸出 OLS_Alpha → 交易端走標準化空間
            _p["ignore_ols_alpha"] = True
        _grid_entries.append({
            "name":             f"Grid {_cs}-{_rs}",
            "formation_module": "strategies.formation.cluster_formation",
            "trading_module":   "strategies.trading.zscore_trading",
            "sub_dir":          f"Grid_{_cs}_{_rs}",
            "db_method":        f"Grid ({_cs}-{_rs})",
            "trade_method":     "Z-Score",
            "params":           _p,
        })
# 插在 formation-only 條目之前（保持 formation-only 於清單尾端）
_fo_idx = next((i for i, s in enumerate(strategies_raw_all) if s.get("formation_only")),
               len(strategies_raw_all))
strategies_raw_all[_fo_idx:_fo_idx] = _grid_entries

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


# ── 敏感性分析（OFAT，口試委員要求）─────────────────────────────────────────
# 對各策略使用參數做 One-Factor-At-A-Time 敏感性分析：每次只變動一個參數、
# 其餘固定基準值，量測 Sharpe/強平率/PF/break-even/DSR 隨參數的變化曲線，
# 展示每個參數的邊際敏感度與穩健區間（全網格組合爆炸，OFAT 為學位論文標準做法）。
#
# formation 參數（adf_pvalue_threshold、pca_n_components）每個值都要重跑 formation，
# 故每值 = 一個獨立策略條目（自有 name/sub_dir/db_method → formation 配對 DB 以
# name 為 strategy_id 鍵，天然不與基準碰撞）。交易參數（entry_z、top_n、stop_loss）
# 已由 run_trading 的 _list 網格涵蓋，不需在此建 formation 變體。
#
# 用法（PowerShell）：
#   $env:SENSITIVITY_PARAM="adf_pvalue_threshold"; python run_formation.py; python run_trading.py
#   可選 $env:SENSITIVITY_BASE="Agglomerative Fundamentals (FMP)"（預設跑三支論文主力）
#   可選 $env:SENSITIVITY_VALUES="0.01,0.05,0.1"（覆寫預設值清單）
#   評估：python -m analysis.sensitivity_report

# formation 參數的預設掃描範圍（基準值標於註解）
SENSITIVITY_TIER1_FORMATION = {
    "adf_pvalue_threshold": [0.01, 0.05, 0.10],   # 基準 0.01(HDBSCAN)/0.05(Agg)
    "pca_n_components":     [3, 5, 10, 15],        # 基準 5
}
# 論文主力策略（口試敏感性分析聚焦對象）
SENSITIVITY_BASES = [
    "Agglomerative Fundamentals (FMP)",
    "Agglomerative Fundamentals (yF)",
    "HDBSCAN Cluster SSD-DTW-PCA PCA5 Resid",
]
# 特定參數僅對特定策略有意義（beta_feature_weight 只有 SEC-PIT+β 有 β 區塊；
# 若套到無此參數的策略，會被 run_formation 的 inspect.signature 過濾成無效重複）。
SENSITIVITY_PARAM_BASES = {}  # （beta_feature_weight → SEC-PIT Beta 已封存）
_SENSITIVITY_INT_PARAMS = {"pca_n_components"}


def _sens_slug(v) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def make_sensitivity_variants(base_name: str, param: str, values: list) -> list:
    """把基準策略沿單一參數的值清單複製成獨立策略條目（OFAT）。"""
    base = next((s for s in strategies_raw_all if s["name"] == base_name), None)
    if base is None:
        print(f"⚠️ [sensitivity] 找不到基準策略 '{base_name}'")
        return []
    variants = []
    for v in values:
        var = copy.deepcopy(base)
        var["params"] = {**base["params"], param: v}
        tag = f"SENS-{param}-{_sens_slug(v)}"
        var["name"]      = f"{base_name} [{param}={v}]"
        var["sub_dir"]   = f"{base.get('sub_dir', base_name)}__{tag}"
        var["db_method"] = f"{base['db_method']} [{param}={v}]"
        var.pop("formation_strategy_id_base", None)   # 自行產生 formation 配對
        variants.append(var)
    return variants


# 交易端 _list 網格參數 → SENSITIVITY_PARAM 命中時改設對應 _list（沿用既有
# formation 配對、只重跑交易，成本低）。top_n/stop_loss 已預設掃描，此處補 entry_z。
_SENSITIVITY_TRADING_LIST = {
    "entry_z":   ("entry_z_list",   [1.5, 2.0, 2.5, 3.0]),
    "top_n":     ("top_n_list",     [1, 3, 5, 10, 20]),
    "stop_loss": ("stop_loss_list", [0.0, 0.05, 0.10, 0.15]),
}

def _build_all_sensitivity():
    """全掃：所有 formation 敏感性變體（adf×pca）＋ 三主力 entry_z 交易掃描。
    回傳 (formation_變體清單, entry_z_基準清單)。單次 run_formation+run_trading 覆蓋 Tier-1。"""
    form_variants = []
    for _p, _vals in SENSITIVITY_TIER1_FORMATION.items():
        for _b in SENSITIVITY_PARAM_BASES.get(_p, SENSITIVITY_BASES):
            form_variants += make_sensitivity_variants(_b, _p, _vals)
    ez_bases = []
    _ez_key, _ez_vals = _SENSITIVITY_TRADING_LIST["entry_z"]
    for _b in SENSITIVITY_BASES:
        _base = next((s for s in strategies_raw_all if s["name"] == _b), None)
        if _base is not None:
            _v = copy.deepcopy(_base)
            _v["params"][_ez_key] = _ez_vals      # 沿用既有 formation，交易端展開 entry_z
            ez_bases.append(_v)
    return form_variants, ez_bases


_sens_all = os.environ.get("SENSITIVITY_ALL", "").strip().lower() not in ("", "0", "false", "no")
_sens_param = os.environ.get("SENSITIVITY_PARAM", "").strip()

if _sens_all:
    # 一次跑完所有敏感性分析：run_formation 產生全部 formation 變體的配對，
    # run_trading 回測全部變體 + 三主力 entry_z 網格。搭配 FORCE_RERUN=True 即「全部重測」。
    _form_variants, _ez_bases = _build_all_sensitivity()
    strategies_raw_all = strategies_raw_all + _form_variants   # 附加尾端（不動既有索引）
    strategies_raw = _form_variants + _ez_bases
    print(f"[config] 敏感性【全掃】模式：{len(_form_variants)} 個 formation 變體"
          f"（adf×pca）+ {len(_ez_bases)} 支主力 entry_z 掃描 = {len(strategies_raw)} 條。"
          f"單次 run_formation + run_trading 涵蓋 Tier-1 全部。")
elif _sens_param:
    _venv = os.environ.get("SENSITIVITY_VALUES", "").strip()
    _base_env = os.environ.get("SENSITIVITY_BASE", "").strip()
    if _base_env:
        _bases = [_base_env]
    elif _sens_param in SENSITIVITY_PARAM_BASES:      # 參數專屬策略（如 beta → SEC-PIT+β）
        _bases = SENSITIVITY_PARAM_BASES[_sens_param]
    else:
        _bases = SENSITIVITY_BASES

    if _sens_param in _SENSITIVITY_TRADING_LIST:
        # 交易端參數：對每個基準策略設對應 _list，run_trading 網格自動展開（不建 formation 變體）
        _list_key, _default = _SENSITIVITY_TRADING_LIST[_sens_param]
        _cast = int if _sens_param == "top_n" else float
        _vals = [_cast(x) for x in _venv.split(",")] if _venv else _default
        _sel = []
        for _b in _bases:
            _base = next((s for s in strategies_raw_all if s["name"] == _b), None)
            if _base is None:
                print(f"⚠️ [sensitivity] 找不到基準策略 '{_b}'"); continue
            _v = copy.deepcopy(_base)
            _v["params"][_list_key] = _vals
            _sel.append(_v)
        strategies_raw = _sel
        print(f"[config] 敏感性分析（交易端）：{_sens_param} ∈ {_vals} × {len(_sel)} 策略"
              f"（沿用既有 formation，只重跑交易）")
    else:
        # formation 參數：每值一個獨立變體（需重跑 formation）
        _cast = int if _sens_param in _SENSITIVITY_INT_PARAMS else float
        _vals = [_cast(x) for x in _venv.split(",") if x.strip()] if _venv \
                else SENSITIVITY_TIER1_FORMATION.get(_sens_param, [])
        if not _vals:
            print(f"⚠️ [sensitivity] 參數 '{_sens_param}' 無值清單（提供 SENSITIVITY_VALUES 或用已知參數）")
        else:
            _variants = []
            for _b in _bases:
                _variants += make_sensitivity_variants(_b, _sens_param, _vals)
            strategies_raw_all = strategies_raw_all + _variants   # 附加尾端（不影響既有索引）
            strategies_raw = _variants                            # 敏感性模式：只跑變體
            print(f"[config] 敏感性分析（formation）：{_sens_param} ∈ {_vals} × {len(_bases)} 策略 → {len(_variants)} 變體")


# ── 儀表板與 ProgressAwareStdout 類別與函數 ───────────────────────────────
_DASHBOARD_FIXED_LINES = 8
_ANSI_RE = re.compile(r"\033\[[^m]*m")

def _char_width(ch: str) -> int:
    """終端顯示欄寬：CJK/全形/寬版 emoji = 2；組合字元 = 0；其餘 = 1。
    （東亞 Ambiguous 視為 1，符合 VS Code / Windows Terminal 非 CJK 語系預設。）"""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

def _visible_len(s: str) -> int:
    """去除 ANSI 後的「顯示欄寬」（非字元數）——CJK/emoji 佔 2 欄。"""
    return sum(_char_width(c) for c in _ANSI_RE.sub("", s))

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
        term_width = min(os.get_terminal_size().columns, 200)
    except OSError:
        term_width = 200

    def line(s: str) -> None:
        # 依「終端顯示欄寬」截斷至 term_width-1（保留 ANSI 碼、不填滿最後一欄）。
        # 關鍵：CJK/emoji 佔 2 欄，若以字元數計算，補齊後的欄位顯示寬度會超出預期，
        # 使整行顯示寬度 > 終端欄寬 → 觸發右邊界 auto-wrap → 每行後多一空行。
        # 逐字元累加「顯示欄寬」即根治。
        limit = max(1, term_width - 1)
        out, vis, i = [], 0, 0
        while i < len(s):
            if s[i] == "\033":
                m = _ANSI_RE.match(s, i)
                if m:
                    out.append(m.group(0))
                    i = m.end()
                    continue
            w = _char_width(s[i])
            if vis + w > limit:
                break
            out.append(s[i])
            vis += w
            i += 1
        print("".join(out) + "\033[0m\033[K")

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

        bar_width = 20
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

        # 策略名稱欄放寬至 45 字元，讓完整策略名可見（含 MSR/Top_n 後綴）
        display_name = name if len(name) <= 45 else name[:44] + "…"

        status_pad = _pad_visible(status_str, 10)
        name_pad   = _pad_visible(display_name, 45)
        bar_pad    = _pad_visible(bar, bar_width)
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
    
    # 以 _pad_visible（顯示欄寬）補齊，讓 CJK 標題/儲存格與資料欄對齊（native :<N 以字元數
    # 計算，中文佔 2 欄會錯位）。
    if show_equity:
        print(
            "\n" + " | ".join([_pad_visible("策略名稱", 45), _pad_visible("狀態", 10),
                               _pad_visible("跳過", 4), _pad_visible("最終權益", 10),
                               _pad_visible("耗時(秒)", 10), "錯誤訊息"]),
            flush=True,
        )
        print("-" * 110, flush=True)
    else:
        print(
            "\n" + " | ".join([_pad_visible("策略名稱", 45), _pad_visible("狀態", 10),
                               _pad_visible("跳過", 4), _pad_visible("耗時(秒)", 10), "錯誤訊息"]),
            flush=True,
        )
        print("-" * 90, flush=True)

    name_order = {c["name"]: i for i, c in enumerate(strategies_config)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["name"], 999))

    for res in results_sorted:
        err = res.get("error") or "無"
        skipped = "是" if res.get("skipped") else "否"
        cells = [_pad_visible(str(res["name"]), 45), _pad_visible(str(res["status"]), 10),
                 _pad_visible(skipped, 4)]
        if show_equity:
            final_eq = res.get("final_equity", 0.0)
            cells.append(_pad_visible(f"${final_eq:.2f}", 10))
        cells += [_pad_visible(f"{res['elapsed']:.2f}", 10), err]
        print(" | ".join(cells), flush=True)
    print("=" * 80, flush=True)
