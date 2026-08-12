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
# 預設 False（斷點續傳）。上游資料變動時需要重算特定幾格，改檔再改回來容易忘記，
# 故比照 STRATEGIES_SLICE 開放環境變數覆寫——兩者搭配即「只強制重跑這幾格」：
#   $env:FORCE_RERUN="1"; $env:STRATEGIES_SLICE="25:27"; python run_formation.py
FORCE_RERUN = os.environ.get("FORCE_RERUN", "").strip().lower() in ("1", "true", "yes")
# 交易期是否另存人類可讀的 Trade Log CSV。續傳與儀表板均以 result.db 為準，
# 故 CSV 為可選產物；False 省下 ~14GB（大型網格）且不影響回測/續傳/儀表板。
# 2026-07-08 起關閉：改採純 result.db 工作流。需人類可讀 CSV 時再設 True。
WRITE_TRADE_CSV = False
# 併發度愈高，寫 result.db 的競爭愈兇。2026-08-03 的重跑就有一格因
# 「database is locked」丟掉全部 trade_logs 列，而摘要列仍寫成功——執行器照樣
# 回報 SUCCESS，缺漏只在事後稽核才被發現。補跑時調低本值可換取寫入穩定。
CPU_LIMIT_PCT = float(os.environ.get("CPU_LIMIT_PCT", "0.95"))
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

# 回測期間（可用環境變數覆寫，免改檔做分期實驗）
#   例：SEC XBRL 結構性財報自 2009 起才有資料 → 該系列實驗以
#       $env:BACKTEST_START="2009-01" 執行，與全期結果並存於 result.db
BACKTEST_START   = os.environ.get("BACKTEST_START", "2000-01").strip()
BACKTEST_END     = os.environ.get("BACKTEST_END", "2025-12").strip()
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
strategies_raw_all = [    # （2026-07-25 主軸收斂：原生傳統基準 4 條（SSD Basic/Rolling、DTW、SSD-DTW-PCA）
    #   與其 formation-only 配對來源 2 條，一併移至 archive/config_archived_strategies.py。
    #   傳統基準改由 Grid (GICS-*) 三格擔任——ADF 門檻統一 0.05，與 ML 分群列可比，
    #   構成乾淨的 4 分組 × 3 排序矩陣。原生條目為「文獻原始設定復現」，
    #   其中 Grid (GICS-SSD) 已驗證與 SSD Rolling 數值完全相同。）
]
# 2026-07-04 清理：FQI 系列×3、CONV×2 等歸位 archive/config_archived_strategies.py。
# 2026-07-06 調整：SSD Basic（基礎原型）自封存拉回現役；HDBSCAN 系列僅保留
# PCA5 Z-Score 版作為分組消融對照組（15 維版、15 維 DRL THR、PCA5 DRL THR 封存）。
# 歷史結果均在 results/result.db；復活方式見封存檔 docstring。
#
# 2026-08-05 更正：上一段的「僅保留 PCA5 Z-Score 版」已不成立。現役 36 條全部走
# cluster_formation，HDBSCAN 臂為 Grid (HDB-SSD/DTW/SDP) 且一律使用
# feature_mode="fundamentals_mix"（19 維：5 報酬 PCA ⊕ 2 基本面 ⊕ 12 GICS one-hot）——
# 三種分群法必須共用同一份特徵，分組才是命題 1 的唯一變因。PCA5（5 維純報酬載荷）
# 已無現役條目，結果僅存於 result.db。
# notebooks/formation/hdbscan_cluster_pca5.ipynb 曾依上一段描述 PCA5，已一併改寫。

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
# ── DRL 疊加：對矩陣 top-2 贏家格（AGG-SSD、HDB-SDP）疊 DRL 門檻選擇式 ──────
# 借用該格已算好的形成期配對（formation_strategy_id_base），零重跑 formation；
# 交易端換成 drl_threshold_trading（走標準化空間 z_of，不讀 OLS_Alpha）。
# 驗證「配對品質矩陣（Z-Score）× DRL 交易端增益」的正交第二層。
# 命題 2：三種 ML 配對底（命題 1 的全部分群方法）各自疊 DRL 交易端，
# 證明 DRL 增益不依賴特定分群方法。各底取其在 3×3 矩陣中的最佳排序。
for _cl_m, _cl_s, _rk_m, _rk_s in [("agglomerative", "AGG", "ssd", "SSD"),
                                    ("hdbscan", "HDB", "ssd_dtw_pca", "SDP"),
                                    ("kmeans", "KM", "ssd", "SSD")]:
    _pd = {**base_params, **_GRID_COMMON,
           "feature_mode": "fundamentals_mix",
           "cluster_method": _cl_m, "ranking_backend": _rk_m,
           "drl_hidden_size": 64, "thr_train_epochs": 40, "thr_min_train_samples": 200}
    _grid_entries.append({
        "name":             f"Grid {_cl_s}-{_rk_s} DRL",
        "formation_module": "strategies.formation.cluster_formation",
        "formation_strategy_id_base": f"Grid {_cl_s}-{_rk_s}",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          f"Grid_{_cl_s}_{_rk_s}_DRL",
        "db_method":        f"Grid ({_cl_s}-{_rk_s}-DRL)",
        "trade_method":     "DRL",
        "params":           _pd,
    })
# 插在 formation-only 條目之前（保持 formation-only 於清單尾端）
_fo_idx = next((i for i, s in enumerate(strategies_raw_all) if s.get("formation_only")),
               len(strategies_raw_all))

# ── 同產業（GICS）分組 × 排序 × 篩選 消融（形成期第三維度：篩選開關）──────
# 對照組設計：與 3×3 Grid 共用同一組排序準則與交易端，唯二差異＝
#   (a) 分組改用真實 GICS 產業（不跑分群、不需特徵矩陣）
#   (b) filter_mode 可關閉三道統計過濾（ADF/半衰期/Hurst）
# 三項實驗：① GICS+排序（NF，無篩選）② GICS+排序+篩選 ③ 分群+排序+篩選（= 3×3 Grid）
for _rb, _rs in _GRID_RANKINGS.items():
    for _fm, _fs_tag in (("coint", ""),):   # NF（無篩選）消融已移附錄
        _pg = {**base_params, **_GRID_COMMON,
               "cluster_method": "gics", "ranking_backend": _rb,
               "filter_mode": _fm}
        if _rb != "ssd":
            _pg["ignore_ols_alpha"] = True
        _grid_entries.append({
            "name":             f"Grid GICS-{_rs}{_fs_tag}",
            "formation_module": "strategies.formation.cluster_formation",
            "trading_module":   "strategies.trading.zscore_trading",
            "sub_dir":          f"Grid_GICS_{_rs}{_fs_tag.replace('-','_')}",
            "db_method":        f"Grid (GICS-{_rs}{_fs_tag})",
            "trade_method":     "Z-Score",
            "params":           _pg,
        })

# ── 傳統分組底 × DRL：命題 1 與命題 2 的交叉對照 ────────────────────────────
# 命題 1 未獲支持：9 組直接對照經逐日 HAC + BH-FDR 校正後無一顯著（方向 8/9 偏 GICS）。
# 註：舊註解曾寫「5 組顯著較差」，係基於已被否定的 n=15 偽重複檢定，勿引用。
# 本組回答後續問題：ML 分群的價值是否須透過 DL 交易端才顯現？
#   設計＝同排序、同交易端，唯一變因為分組（GICS vs ML 分群）：
#     Grid (GICS-SSD-DRL) ↔ Grid (AGG-SSD-DRL)
#     Grid (GICS-SDP-DRL) ↔ Grid (HDB-SDP-DRL)
#   結果（2026-07-29 以逐日差分 Newey-West HAC 重做；舊版 n=15 參數格檢定因
#   偽重複已降為描述性附錄，見 analysis/proposition2_daily_hac.py docstring）：
#     GICS 底 DRL 的增益最大（年化 +1.066%，NW p=0.0002），五種配對底全部顯著
#     （p = 0.0002–0.0177），並經 block bootstrap 無母數對照確認。
#   → ML 分群未提供 DL 端可利用的額外結構；命題 1 未獲支持，
#     命題 2 因涵蓋傳統配對底而普適性更強。
#   三項替代解釋已排除（prop2_exposure_control / prop2_skip_permutation）：
#     非拉高門檻、非篩掉爛配對、非減少曝險（DRL 進場次數反為 1.6–1.9 倍）。
#   絕對績效仍不成立：DSR 於 N=104 的試驗宇宙下 0/92 個策略族通過 0.95，
#   最高者 Grid (GICS-SDP) 為 0.609。注意論證方向——門檻 SR0=0.384 低於最高 SR 0.437，
#   故不可寫成「門檻高於最高 SR」，結論由 DSR 檢定本身承擔。
#   （舊註記的 N=87 / SR0=0.433 / 最高 SR 0.343 三個數字皆已失效，見
#    analysis/regime_cost_dsr_eval.py 的 TRIAL_CENSUS 清點沿革。）
# 借用 Grid GICS-{SSD,SDP} 已算好的形成期配對，零重跑 formation。
for _rk_m, _rk_s in (("ssd", "SSD"), ("ssd_dtw_pca", "SDP")):
    _pgd = {**base_params, **_GRID_COMMON,
            "cluster_method": "gics", "ranking_backend": _rk_m, "filter_mode": "coint",
            "drl_hidden_size": 64, "thr_train_epochs": 40, "thr_min_train_samples": 200}
    _grid_entries.append({
        "name":             f"Grid GICS-{_rk_s} DRL",
        "formation_module": "strategies.formation.cluster_formation",
        "formation_strategy_id_base": f"Grid GICS-{_rk_s}",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          f"Grid_GICS_{_rk_s}_DRL",
        "db_method":        f"Grid (GICS-{_rk_s}-DRL)",
        "trade_method":     "DRL",
        "params":           _pgd,
    })

# ── RL-THR：部分回饋對照組（命題 2 的第四項受控對照，2026-08-05）────────────
# 現役的 DL-THR（drl_threshold_trading）名為「DRL」，實際是全資訊監督回歸：
# 每期把 9 個動作的報酬全部反事實回算後餵給網路，無探索問題。本組把它換成
# 真正的部分回饋——只觀測選中動作的報酬 + ε-greedy——其餘（動作選單、12 維
# 狀態、網路、walk-forward 切分）逐位元相同，構成單一變因對照：
#     反事實標籤值多少錢？
# 借用 Grid AGG-SSD 已算好的形成期配對，零重跑 formation，直接對接
# Grid (AGG-SSD-DRL)。
#
# ε 掃三組並以「對 bandit 最有利者」與 DL-THR 對比——若在最有利條件下仍輸，
# 「反事實標籤有價值」的結論才保守且站得住。三組各自持有網路與經驗
# （見 rl_threshold_trading._get_shared 的 scope key），不互相餵食樣本。
#
# 形式歸屬（勿含糊）：這是 contextual bandit，沒有 γ、沒有序列信用分配——
# 每期一次決策且狀態全由形成期視窗算出，選哪個門檻不改變下期狀態。
# 逐日定位動作空間的真 RL（v1 DQN / v2 / v3 FQI，γ=0.99 + bootstrapped
# target）已系統性證偽，見 archive/trading/ 與 archive/config_archived_strategies.py。
for _eps0, _epsf, _etag in ((0.05, None, "E05"),
                            (0.10, None, "E10"),
                            (0.20, 0.02, "E20D")):
    _prl = {**base_params, **_GRID_COMMON,
            "feature_mode": "fundamentals_mix",
            "cluster_method": "agglomerative", "ranking_backend": "ssd",
            "drl_hidden_size": 64, "thr_train_epochs": 40, "thr_min_train_samples": 200,
            "rl_epsilon": _eps0, "rl_epsilon_final": _epsf,
            "rl_epsilon_decay_steps": 2000}
    _grid_entries.append({
        "name":             f"Grid AGG-SSD RLTHR {_etag}",
        "formation_module": "strategies.formation.cluster_formation",
        "formation_strategy_id_base": "Grid AGG-SSD",
        "trading_module":   "strategies.trading.rl_threshold_trading",
        "sub_dir":          f"Grid_AGG_SSD_RLTHR_{_etag}",
        "db_method":        f"Grid (AGG-SSD-RLTHR-{_etag})",
        "trade_method":     "RLTHR",
        "params":           _prl,
    })


# ── 許鈞翔 (2025) 設定復現：對照組差異的定位實驗（2026-08-06）──────────────
# 指導教授質疑本研究的回測結果與前一屆學生（許鈞翔 2025，ref/ 內）差異過大。
# 逐項比對其論文第三章後，兩份研究只有「單邊交易成本 0.29%」是相同的，
# 其餘五項全不同。本組以本引擎重現他的設定，用來判定差異是「口徑」還是「方法」：
#
#   項目        許鈞翔 (2025)                    本研究主線
#   ─────────────────────────────────────────────────────────────
#   樣本期間    2008-01 ~ 2024-12（16 年）        2000-01 ~ 2025-12（25 年）
#   分組        無（全市場 112,101 對）           GICS 或 ML 分群，群內配對
#   篩選        只有 ADF，p < 0.01                ADF p<0.05 + 半衰期 + Hurst
#   排序        SSD / DTW / SSD⊕DTW 的 PCA PC1    同（但 SSD 後端先排序後檢定）
#   取幾對      前 5 對                           網格 1/3/5/10/20，主口徑全網格等權
#   報酬口徑    R^EC 動用資本                      承諾資本為主
#   交易期      6 或 12 個月                       126 交易日 ≈ 6 個月
#
# 期間以環境變數控制（BACKTEST_START=2008-01 BACKTEST_END=2024-12），
# 不寫死於此，使同一組條目也能在全期上跑作為對照。
# 交易期 12 個月需另設 trading_window=252，屬第二階段實驗，此處先做 6 個月。
for _hs_rb, _hs_tag in (("ssd", "SSD"), ("dtw", "DTW"), ("ssd_dtw_pca", "SDP")):
    _p_hsu = {**base_params, **_GRID_COMMON,
              "cluster_method": "none",          # 不分組：全市場配對
              "ranking_backend": _hs_rb,
              "filter_mode": "adf_only",         # 只做 ADF，無半衰期、無 Hurst
              "adf_pvalue_threshold": 0.01,      # 他用 0.01（本研究主線用 0.05）
              "top_n_list": [5],                 # 他只取前 5 對
              "stop_loss_list": [0.0, 0.05, 0.10, 0.15],   # 他的四檔止損
              "top_n": 5}
    if _hs_rb != "ssd":
        _p_hsu["ignore_ols_alpha"] = True
    _grid_entries.append({
        "name":             f"HSU25 {_hs_tag}",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          f"HSU25_{_hs_tag}",
        "db_method":        f"HSU25 ({_hs_tag})",
        "trade_method":     "Z-Score",
        "params":           _p_hsu,
    })
    # 差異第五項：進場時點（2026-08-11）。上面三條沿用本研究的突破式進場
    # （|z| > entry_z 即進場）；許鈞翔的程式是「價差先發散到帶外、再收斂回帶內」
    # 才進場。借用上面三條已算好的形成期配對，只換交易端，故差異可完全歸因於
    # 進場時點。成本、止損網格、篩選、取幾對一律不動。
    _grid_entries.append({
        "name":             f"HSU25 {_hs_tag} REV",
        "formation_module": "strategies.formation.cluster_formation",
        "formation_strategy_id_base": f"HSU25 {_hs_tag}",
        "trading_module":   "strategies.trading.zscore_reversion_entry_trading",
        "sub_dir":          f"HSU25_{_hs_tag}_REV",
        "db_method":        f"HSU25 ({_hs_tag}-REV)",
        "trade_method":     "Z-Score",
        "params":           {**_p_hsu},
    })


# （特徵消融「多尺度動量」：2026-07-24 驗證為負面結果——三種分群全面劣化
#   （HDB −0.94pp、AGG −2.43pp 至 −0.65%、KM −1.40pp）。根因：動量度量「過去
#   漲跌幅」而非「走勢同步性」，且橫斷面變異大於 PCA 載荷，在歐氏距離中主導
#   分群、稀釋原本有效的因子暴露訊號（與 SEC-PIT-Beta 同一失敗模式）。
#   註：Han, He & Toh (2021) 的動量特徵用於「識別高估/低估股票」而非分群依據，
#   其分群依據為 78 個公司特徵——正確的擴充方向是結構性基本面特徵。
#   已封存至 archive/config_archived_strategies.py；
#   _features.build_momentum_features 與 feature_mode="momentum"/"momentum_mix"
#   機制保留供復活。）


# （特徵消融 F09 結構性財報特徵：2026-07-24 驗證為負面結果（三分群 Δ 皆在 ±0.22pp
#   噪音範圍內），已移至 archive/config_archived_strategies.py；
#   structural_features 機制保留於 _features / cluster_formation。）

# ── 分組 × 篩選 × 產業先驗：命題 1 的機制因子設計（2026-07-29）─────────────
# 命題 1 的動機來自 Han, He & Toh (2021) *Pairs Trading via Unsupervised Learning*
# （程式碼舊註解誤植為 "Sanders (2021)"）。該文以 CRSP 全美股、48 動量因子 +
# 78 公司特徵分群，群內「做多低估、做空高估」，**不施加共整合篩選**，並明文
# 指出跨產業發散亦為利潤來源。
#
# 本研究的實作與其有三處結構性差異，皆可能單獨壓抑該假說，且**從未被消融**：
#   (a) 特徵含 12 維 GICS one-hot（權重 1.0）。實測：在分群真正會抓的近鄰中，
#       跨產業配對距離被推遠 +77.9%，同產業 +0.0% —— 特徵設計主動懲罰
#       「跨產業隱藏配對」，正是假說要找的東西。
#   (b) 共整合篩選（ADF+半衰期+Hurst）與分群目標衝突：特徵相似的跨產業股票
#       最不可能通過價差平穩性檢定。ML×無篩選這格從未測過。
#   (c) 消融矩陣的「分組」維度只有四種分組法，缺「不分組」零點——分組層在本
#       管線中只負責限制候選池，不比較「限制 vs 不限制」就無法評價其價值。
#       （舊有 SSD (Basic) 雖不分組，但同時無篩選、β=1，三變因混淆。）
#
# 設計：排序固定 ssd（主軸且最省算力），2×2×2 中缺的 5 格。GICS 兩格與
# AGG 基準格已存在，故不重複建立。
for _cm_g, _fm_g, _ohw, _tag in (
    ("agglomerative", "coint", 0.0, "AGG-SSD-NOSEC"),      # 拿掉產業先驗
    ("agglomerative", "none",  1.0, "AGG-SSD-NF"),         # 拿掉共整合篩選
    ("agglomerative", "none",  0.0, "AGG-SSD-NF-NOSEC"),   # 兩者都拿掉（最接近 Han et al.）
    ("none",          "coint", 1.0, "NOGRP-SSD"),          # 不分組 + 篩選
    ("none",          "none",  1.0, "NOGRP-SSD-NF"),       # 不分組 + 無篩選
):
    _pf = {**base_params, **_GRID_COMMON,
           "feature_mode": "fundamentals_mix",
           "cluster_method": _cm_g, "ranking_backend": "ssd",
           "filter_mode": _fm_g, "sector_onehot_weight": _ohw}
    _grid_entries.append({
        "name":             f"Grid {_tag}",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          f"Grid_{_tag.replace('-', '_')}",
        "db_method":        f"Grid ({_tag})",
        "trade_method":     "Z-Score",
        "params":           _pf,
    })

# ── Han et al. (2021) 交易機制的逐步歸因鏈（2026-07-29）────────────────────
# 前一組因子設計證實：三處**形成期**實作差異（產業 one-hot、共整合篩選、
# 缺不分組零點）不足以解釋命題 1 的否定。剩餘兩個殘差為母體範圍（無資料，
# 不可測）與**交易機制**——本鏈檢驗後者。
#
# Han, He & Toh (2021) 的機制與本研究的四項差異，逐步施加以維持單變因：
#   起點 = Grid (AGG-SSD-NF)：AGG 分群 + SSD 距離 + 無篩選 + OLS-β + z>2/126 日
#   ②  β 改 1 等金額            → distance_trading（原文："buy one stock and
#                                 sell the other for the same amount"）
#   ③  選對準則改月報酬發散      → ranking_backend="reversal"
#   ④  進出場改月頻固定持有      → 21 日窗、entry_z=0（發散即建倉）、
#                                 hold_to_period_end（不做收斂出場）
# ④ 完成即為 Han et al. 的交易端全貌。
#
# ⚠ 仍無法關閉的缺口（即使 ④ 完美復刻）：分群依據仍是 7 維連續特徵，
#   而原文為 48 動量因子 + 78 公司特徵；母體仍為 S&P 500 而非 CRSP 全市場。
#   故本鏈只能回答「交易機制解釋了多少差距」，不能預期複製 24.8% 的績效。
_han_base = {**base_params, **_GRID_COMMON,
             "feature_mode": "fundamentals_mix",
             "cluster_method": "agglomerative",
             "filter_mode": "none"}          # 起點已無篩選，全鏈維持
for _tag, _rb, _extra, _borrow in (
    # ② 形成期參數與 Grid AGG-SSD-NF 完全相同（唯一變因在交易端），故借用其
    #    已算好的配對——既省一次 formation，也保證兩臂配對逐筆一致。
    ("HAN2-B1",      "ssd",      {}, "Grid AGG-SSD-NF"),
    ("HAN3-REV",     "reversal", {}, None),
    ("HAN4-MONTHLY", "reversal", {"trading_window": 21, "rolling_step": 21,
                                  "entry_z": 0.0, "hold_to_period_end": True}, None),
):
    _e = {
        "name":             f"Grid {_tag}",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.distance_trading",   # β = 1 等金額
        "sub_dir":          f"Grid_{_tag.replace('-', '_')}",
        "db_method":        f"Grid ({_tag})",
        "trade_method":     "Distance",
        "params":           {**_han_base, "ranking_backend": _rb, **_extra},
    }
    if _borrow:
        _e["formation_strategy_id_base"] = _borrow
    _grid_entries.append(_e)

# ── 特徵維度：Han et al. 的「78 公司特徵」可得子集（2026-07-30）──────────────
# 前兩組實驗已排除形成期實作差異與交易機制為命題 1 失敗的原因。剩餘殘差之一是
# **特徵維度**：本研究原為 7 維連續（5 報酬 PCA + 2 基本面），原文為 48 動量因子
# + 78 公司特徵。本組把可得的公司特徵補上，檢驗維度是否為關鍵。
#
# 資料現實（見 fetch/fetch_sec_characteristics.py docstring）：
#   - SEC XBRL 強制申報自 ~2009 起，2000–2008 無財報特徵
#   - 40 個建出的特徵中，僅 10 個在「PIT 成分股身分」分母下覆蓋率 >70%
#   - 估值比率（bm/ep/cfp/sp/dy/lev）因未調整股價快取被 bug 毀損而僅 15–22%，
#     補抓中（Tiingo 免費方案 ~50 檔/小時）
#
# 兩項防污染措施（否則會把 GICS 資訊從後門送進特徵向量）：
#   - sector_onehot_weight=0：不直接編碼產業
#   - impute_scope="global"：缺失值用全域中位數，非產業中位數
#
# 執行全期以與基準共用期間定義；分析時限制在 2012+（特徵實際存在的期間），
# 故基準 Grid (GICS-SSD) / Grid (AGG-SSD) 不需重跑。
# 兩格構成單變因對照——唯一差異為「是否納入那 12 個公司特徵」：
#   NOSEC-GI : 7 維連續（5 PCA + 市值 + 盈餘殖利率），one-hot=0，全域插補
#   CHARS    : 19 維連續（同上 + 12 個公司特徵），其餘完全相同
# 若直接拿 CHARS 對比既有的 AGG-SSD-NOSEC，會同時改動特徵數與插補方式，
# 無法歸因（與舊 SSD (Basic) 對照的三變因混淆同型）。
#
# 名單＝characteristics_coverage.csv 中覆蓋率 >70% 者（分母為 2012+ 真實成分股
# 身分，非面板交叉積）。2026-08-03 補齊 Tiingo 原始價快取後由 10 個增為 12 個：
# 新增的 bm、ep 是評價類特徵，先前因 market_cap 覆蓋率僅 10.5% 而卡在 22%／21%，
# 補抓後升至 83%／80%。這使本組首度含有帶價格資訊的特徵，不再是純會計面。
_CHARS = ("agr", "egr", "bm", "chtx", "cash_ratio", "roa", "roe", "ep",
          "capital_intensity", "tb", "lgr", "currat")
_chars_common = {**base_params, **_GRID_COMMON,
                 "feature_mode": "fundamentals_mix",
                 "cluster_method": "agglomerative", "ranking_backend": "ssd",
                 "filter_mode": "coint",
                 "fundamentals_parquet_path":
                     "dataset/fundamental/sp500_pit_characteristics_monthly.parquet",
                 "sector_onehot_weight": 0.0,
                 "impute_scope": "global"}
for _tag, _feats in (("AGG-SSD-NOSEC-GI", ()), ("AGG-SSD-CHARS", _CHARS)):
    _grid_entries.append({
        "name":             f"Grid {_tag}",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          f"Grid_{_tag.replace('-', '_')}",
        "db_method":        f"Grid ({_tag})",
        "trade_method":     "Z-Score",
        "params": {**_chars_common, "structural_features": _feats,
                   "structural_weight": 1.0},
    })

# ── F09 結構性財報特徵消融的重驗（2026-07-30）──────────────────────────────
# 原始 F09（2026-07-24，已封存）結論：「10 維 SEC XBRL 財報比率對分群品質無顯著
# 貢獻，三分群 Δ 皆在 ±0.22pp 噪音範圍內」。該實驗在 `impute_scope="group"`
# （產業中位數插補）下執行。
#
# 為何值得重驗：STRUCT 臂的結構性特徵有 30–50% 缺失，被填成**產業中位數**，
# 而產業資訊**已由 sector_onehot 編碼**。那些插補值因此是冗餘資訊，
# 使結構性區塊的邊際貢獻被系統性低估。
# 註：one-hot 在 BASE/STRUCT 兩臂皆為 1.0，於差分中對消，本身不造成偏誤——
#     問題出在插補值的冗餘，而非 one-hot。
#
# 設計：唯一相對原始 F09 的改動是 impute_scope="global"。3 分群 × {BASE, STRUCT}
# 共 6 格，以完整重驗「三分群皆為噪音」的原宣稱。分析限制在 2012+
# （XBRL 覆蓋率穩定期），與 prop1_feature_dimension 同口徑。
_F09_FEATS = ("book_to_market", "roe", "roa", "gross_margin", "op_margin",
              "leverage", "cash_ratio", "capital_intensity",
              "asset_turnover", "accruals")
for _cm_f, _cs_f in (("hdbscan", "HDB"), ("agglomerative", "AGG"), ("kmeans", "KM")):
    for _arm, _feats in (("BASE", ()), ("STRUCT", _F09_FEATS)):
        _grid_entries.append({
            "name":             f"F09GI {_cs_f}-{_arm}",
            "formation_module": "strategies.formation.cluster_formation",
            "trading_module":   "strategies.trading.zscore_trading",
            "sub_dir":          f"F09GI_{_cs_f}_{_arm}",
            "db_method":        f"F09GI ({_cs_f}-{_arm})",
            "trade_method":     "Z-Score",
            "params": {**base_params, **_GRID_COMMON,
                       "feature_mode": "fundamentals_mix",
                       "cluster_method": _cm_f, "ranking_backend": "ssd",
                       "filter_mode": "coint",
                       "structural_features": _feats,
                       "structural_weight": 1.0,
                       # 唯一相對原始 F09 的改動
                       "impute_scope": "global"},
        })

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
# 2026-07-29 修正：原列的三個名字（Agglomerative Fundamentals FMP／yF、
# HDBSCAN Cluster SSD-DTW-PCA PCA5 Resid）已於 2026-07-24 隨舊架構封存，
# 不在現役 17 條之列 → make_sensitivity_variants 一律找不到基準而靜默跳過，
# SENSITIVITY_ALL 與省略 SENSITIVITY_BASE 的用法形同失效。
# 改為現行論文主軸的三個 ML 配對底（與 analysis/proposition2_*.py 的 PAIRS 對齊）。
# 需要傳統分組對照時另加 "Grid GICS-SSD" / "Grid GICS-SDP"，或以
# SENSITIVITY_BASE 環境變數逐一指定。
SENSITIVITY_BASES = [
    "Grid AGG-SSD",
    "Grid HDB-SDP",
    "Grid KM-SSD",
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
