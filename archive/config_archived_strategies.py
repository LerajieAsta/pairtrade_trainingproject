"""
已封存策略定義（2026-07-03 自 strategies/config.py 移出）
======================================================================

封存原因總覽（完整診斷見對話紀錄與 results/result.db 歷史數據）：

【座標系 artifact — 已被修正版取代】
  DTW Paper (DTW) / (SSD-DTW-PCA)：OLS 在標準化空間擬合但輸出 OLS_Alpha，
    交易端路徑 A 以原始 log-price 空間重建 spread → 常數 Z 偏移
    （進場中位 |Z|=3.24、85% 期初 3 天進場、99% 期末強平）。
    中位 Sharpe 0.45–0.46 為 bug 副產物；修正後（#Fixed 版）跌至 ≈0。
    ⚠️ 其形成期配對仍被現役 DTW Paper Fixed / SSD-DTW-PCA Paper Fixed 借用
    （formation_strategy_id_base），formation DB 中的資料列不可刪除；
    若需在全新資料庫重建形成配對，暫時將本檔對應項目加回 strategies_raw_all。

【HDBSCAN 舊特徵系 — 已被 PCA-Loadings 取代】
  HDBSCAN MultiScale / MultiScale PCA-UMAP / UMAP / UMAP PCA-UMAP：
    stats10 特徵與共整合無因果關聯（與 DTW 池僅 9% 重疊）、
    Quality Score 排序不顯著。中位 Sharpe −0.06 ~ +0.10。

【Ensemble — 放大劣質配對】
  Ensemble HDBSCAN：中位 Sharpe −0.24（比子策略更差）。
  Ensemble SSD-DTW：子策略含 artifact 版 DTW，結果不可解讀。

【DRL v1 — 架構缺陷已診斷，被 FQI (v3) 取代】
  SSD Rolling DRL：假共享（每配對過擬合單軌跡）、獎勵重複計算、
    epsilon 未退完、觀測未標準化 → 中位 Sharpe −0.71（6.5 天持有、3000+ 進出）。
    結果保留於 result.db（METHOD='SSD (Rolling-DRL)'）作為 v1 對照基準。
  HDBSCAN UMAP DRL / MultiScale DRL：未完成回測，形成法亦已淘汰。

【Kalman — 與論文兩命題無關】
  SSD Rolling Kalman（中位 Sharpe 0.31 但年化僅 0.3%）、HDBSCAN UMAP Kalman（0.03）。

【CONV 收斂持有 — 假設已檢驗並否決】
  DTW Paper Fixed CONV（−0.11）、HDBSCAN PCA-Loadings CONV（−0.33）：
    長持收斂無法誠實重現 artifact 版獲利。負面結果已記錄，可寫入論文。

復活方式：from archive.config_archived_strategies import strategies_raw_archived
          strategies_raw_all += strategies_raw_archived（或挑選單一項目加回）
"""

from strategies.config import (
    base_params, _HDBSCAN_UMAP_COMMON, _HDBSCAN_UMAP_FILTERS, _HDBSCAN_MS_FILTERS,
)

strategies_raw_archived = [
    # ── 舊 #1：SSD Basic（被 SSD Rolling 取代的簡化基準） ─────────────────────
    {
        "name":             "SSD Basic",
        "formation_module": "strategies.formation.ssd_basic",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_Basic",
        "db_method":        "SSD (Basic)",
        "trade_method":     "Z-Score",
        "params":  {**base_params},
    },
    # ── 舊 #3：DTW Paper (DTW)（座標 artifact；形成配對仍被 Fixed 版借用） ──
    {
        "name":             "DTW Paper (DTW)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "DTW_Paper",
        "db_method":        "DTW (Paper)",
        "trade_method":     "Z-Score",
        "params":  {**base_params, "method": "dtw"},
    },
    # ── 舊 #4：DTW Paper (SSD-DTW-PCA)（座標 artifact；形成配對仍被借用） ───
    {
        "name":             "DTW Paper (SSD-DTW-PCA)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "SSD_DTW_PCA_Paper",
        "db_method":        "SSD-DTW-PCA (Paper)",
        "trade_method":     "Z-Score",
        "params":  {**base_params, "method": "ssd_dtw_pca"},
    },
    # ── 舊 #5–#8：HDBSCAN stats10 特徵系（被 PCA-Loadings 取代） ─────────────
    {
        "name":             "HDBSCAN MultiScale",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_MultiScale",
        "db_method":        "HDBSCAN (MultiScale)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "umap"},
    },
    {
        "name":             "HDBSCAN MultiScale PCA-UMAP",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_MultiScale_PCA_UMAP",
        "db_method":        "HDBSCAN (MultiScale-PCA-UMAP)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "pca_umap"},
    },
    {
        "name":             "HDBSCAN UMAP",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_UMAP",
        "db_method":        "HDBSCAN (UMAP)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "umap"},
    },
    {
        "name":             "HDBSCAN UMAP PCA-UMAP",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_UMAP_PCA_UMAP",
        "db_method":        "HDBSCAN (UMAP-PCA-UMAP)",
        "trade_method":     "Z-Score",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "pca_umap"},
    },
    # ── 舊 #9–#10：Ensemble ──────────────────────────────────────────────────
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
    # ── 舊 #11–#13：DRL v1（drl_lstm_trading.py，架構缺陷版） ────────────────
    {
        "name":             "SSD Rolling DRL",
        "formation_module": "strategies.formation.ssd_rolling",
        "trading_module":   "strategies.trading.drl_lstm_trading",
        "sub_dir":          "SSD_Rolling_DRL",
        "db_method":        "SSD (Rolling-DRL)",
        "trade_method":     "DRL",
        "params":  {
            **base_params,
            "drl_episodes": 150, "drl_hidden_size": 256,
            "drl_num_layers": 2, "drl_batch_size": 512,
        },
    },
    {
        "name":             "HDBSCAN UMAP DRL",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "strategies.trading.drl_lstm_trading",
        "sub_dir":          "HDBSCAN_UMAP_DRL",
        "db_method":        "HDBSCAN (UMAP-DRL)",
        "trade_method":     "DRL",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "umap", "drl_episodes": 150, "drl_hidden_size": 256, "drl_num_layers": 2, "drl_batch_size": 512},
    },
    {
        "name":             "HDBSCAN MultiScale DRL",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "strategies.trading.drl_lstm_trading",
        "sub_dir":          "HDBSCAN_MultiScale_DRL",
        "db_method":        "HDBSCAN (MultiScale-DRL)",
        "trade_method":     "DRL",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "umap", "drl_episodes": 150, "drl_hidden_size": 256, "drl_num_layers": 2, "drl_batch_size": 512},
    },
    # ── 舊 #14–#15：Kalman ───────────────────────────────────────────────────
    {
        "name":             "SSD Rolling Kalman",
        "formation_module": "strategies.formation.ssd_rolling",
        "trading_module":   "strategies.trading.kalman_trading",
        "sub_dir":          "SSD_Rolling_Kalman",
        "db_method":        "SSD (Rolling-Kalman)",
        "trade_method":     "Kalman",
        "params":  {**base_params, "kalman_delta": 1e-4, "kalman_R": 1e-2},
    },
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
    # ── 舊 #18 / #20：CONV 收斂持有（假設已檢驗並否決的負面結果） ────────────
    {
        "name":             "DTW Paper Fixed CONV (DTW)",
        "formation_module": "strategies.formation.DTW_Cointegration_Paper",
        "formation_strategy_id_base": "DTW Paper (DTW)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "DTW_Paper_Fixed_CONV",
        "db_method":        "DTW (Paper-Fixed-CONV)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "method": "dtw",
            "ignore_ols_alpha":   True,
            "hold_to_period_end": True,
        },
    },
    {
        "name":             "HDBSCAN PCA-Loadings CONV",
        "formation_module": "strategies.formation.HDBSCAN_PCA_Loadings",
        "formation_strategy_id_base": "HDBSCAN PCA-Loadings",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "HDBSCAN_PCA_Loadings_CONV",
        "db_method":        "HDBSCAN (PCA-Loadings-CONV)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS,
            "reduce_method":      "none",
            "pca_n_components":   15,
            "feature_mode":       "pca_loadings",
            "hold_to_period_end": True,
        },
    },
]
