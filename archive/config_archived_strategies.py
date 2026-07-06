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
    （formation_strategy_id_base），formation DB 中的資料列不可刪除。
    【2026-07-05 更新】repo LFS 額度用罄致 formation DB 遺失、需本地重算，
    兩個原版條目已以 formation_only 旗標回歸 strategies/config.py 現役清單
    （run_formation 產生配對、run_trading 跳過回測），故自本檔
    strategies_raw_archived 移除，避免日後整批復活時重複定義。

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

【DRL v3 FQI — 逐日定位動作空間已系統性證偽（2026-07-04 封存）】
  HDBSCAN PCA-Loadings DRL FQI（中位 Sharpe −2.30）、SSD Rolling DRL FQI（−1.10）：
    FQI 修復了 v1 的訓練計算與統計缺陷（持有 0.5→5.7 天、PF 0.63→0.82），
    但「每日自由決定持倉」的動作空間讓模型對日級噪音計時，OOS 換手
    3700+ 次（基準 617）被費用磨死。v1→v2→v3 構成完整架構消融鏈：
    失敗根因 = 動作空間設計，非訓練方法。由 v4 門檻選擇式（現役）取代。
  DRL FQI XL：H200 壓測 + 容量縮放用，隨 FQI 架構一併封存。

【HDBSCAN PCA-Loadings 系列 — 命題 1/2 已被 Cluster-SSD-DTW-PCA 系列取代
  （2026-07-04 二次封存）】
  HDBSCAN PCA-Loadings（Z-Score 基準）：報酬 PCA 因子載荷 15 維直接餵 HDBSCAN
    （reduce_method="none"），在 45 組 entry_z×dynamic_stop_z 網格中僅 4.4%
    Sharpe 為正、中位數 −0.62，全組最不穩定的 Z-Score 基準。診斷：15 維空間
    平均每期僅解釋 58.6% 報酬變異、後段主成分訊號弱卻等權參與歐氏距離計算，
    疑似維度詛咒稀釋 HDBSCAN 密度估計（平均 30.9% 標的被判雜訊排除、群數在
    2~25 間震盪）。已被 HDBSCAN Cluster SSD-DTW-PCA（改用 HDBSCAN 只做分組、
    排序沿用 SSD-DTW-PCA 距離法）及其 PCA5 降維版取代，兩者 Z-Score 基準
    Sharpe 分別為 0.22、0.29，全面優於本策略。
  HDBSCAN PCA-Loadings DRL THR：疊加 DRL-THR 後 15 組全負 Sharpe（中位數
    −0.41、最差 −0.67），是全部策略中表現最差者。曾懷疑是
    strategies/trading/drl_threshold_trading.py 的 _shared 全域 key 跨策略
    變體污染訓練資料所致（該 bug 已修復，見 commit 9ae56ec）；修復後重跑
    30 組變體驗證：結果幾乎不變（中位數 Sharpe −0.41 → −0.41），排除訓練污染
    假說，確認是配對訊號本身太弱、DRL 交易端救不回來。HDBSCAN Cluster
    SSD-DTW-PCA（PCA5）+ DRL THR 的等效實驗則成功：Sharpe 0.42 → 0.54。
  ⚠️ 兩者的 formation_strategy_id_base 互相借用（DRL THR 借用 Z-Score 基準的
    形成期配對），一併封存、一併復活。"HDBSCAN PCA-Loadings_MSR0" 這個
    strategy_id 在 formation_data 中的配對列保留不刪（比照 DTW 原版的慣例，
    供未來復活時直接沿用，不需重新計算）。

【ML Pair Quality — 監督式學習排序，兩次嘗試皆未驗證成功（2026-07-04 封存）】
  假說：不模仿 SSD Rolling 的 SSD 距離排序（天花板頂多打平），改用
  drl_threshold_trading.py（v4）已驗證有效的 walk-forward 反事實監督回歸
  範式往形成期搬一格——用候選配對「下一期實際已實現報酬」（透過
  zscore_trading.Trading._simulate_pair 算出）當標籤，訓練 MLP 學習配對
  品質評分，取代 SSD 距離排序；候選池仍沿用 SSD Rolling 的同 GICS 產業分組，
  只換排序函式，乾淨對照「學出來的排序 vs SSD 距離排序」。
    第一次嘗試（原始報酬 % 當標籤，MSE 回歸）：15 組中 4 組正 Sharpe，
      中位數 −0.155，最佳 Sharpe 0.203（Top1，跟 SSD 同量級），但 Top10/
      Top20 排名快速惡化（Sharpe 至 −0.66）——模型抓得到「最強訊號」但
      排序信心校準差。診斷：單一歷史實例的已實現報酬肥尾嚴重（標準差
      28%、曾見 +314% 極端值），MSE 對這種分布容易被少數極端值主導梯度。
    第二次嘗試（log 壓縮標籤 sign(x)*log1p(|x|)，其餘不變）：結果不僅沒有
      改善，反而更差——15 組全負 Sharpe，中位數 −0.245。顯示問題可能不是
      「MSE 被極端值主導」這個表面現象，而是這組特徵（SSD、DTW 距離、ADF
      p-value/統計量、半衰期、Hurst、hedge ratio、spread 標準差、波動率比、
      產業內 SSD 排名）對「這個配對下一期會不會賺錢」本身就缺乏可學習的
      訊號——不同次訓練結果因隨機初始化而有落差，但都圍繞在「沒有真實優勢」
      這條線附近震盪，不是單純的損失函數/標籤轉換問題。
    若要復活，較有機會的方向：(a) 把回歸問題換成 pairwise ranking loss
      （只要求排序正確，不要求預測數值，對極端值天然不敏感，更貼近實際
      需求）；(b) 重新設計特徵（目前這組主要是共整合統計量，可能需要
      加入能真正預測「持續性」而非「單次實現報酬」的訊號，例如形成期內
      分段檢驗共整合強度的穩定度）。
  過程中意外發現並修好一個真正的資料庫 bug（見 commit 01b6da4）：
  run_formation.py 的 merge_databases 用 INSERT OR REPLACE 假設同一
  strategy_id 重跑必然選出完全相同的配對（對 SSD/DTW 等決定性方法成立），
  但對會迭代訓練、重跑會選出不同配對的模組（如本策略）不成立，導致新舊
  兩批配對一起留在資料庫裡（同一期配對數超過 top_n）。此 bug 修復已保留
  在現役程式碼中，不隨本策略封存而回退。

復活方式：from archive.config_archived_strategies import strategies_raw_archived
          strategies_raw_all += strategies_raw_archived（或挑選單一項目加回）

【非現役交易模組檔案歸位 — 2026-07-05】
  已封存/孤兒的 5 個交易模組（drl_lstm_trading.py、drl_lstm_v2_trading.py、
  drl_fqi_trading.py、kalman_trading.py、pure_dtw_trading.py）自
  strategies/trading/ 移至 archive/trading/，本檔對應條目的 trading_module
  已同步改為 archive.trading.*，復活時無需搬回檔案即可直接執行
  （run_trading.py 以 importlib 依字串載入，archive/ 為 namespace package）。
  strategies/trading/ 現只保留現役的 zscore_trading.py 與 drl_threshold_trading.py。

【從未進入策略清單的孤兒模組 — 2026-07-04 盤點記錄】
  以下兩個檔案（現位於 `archive/trading/`）不對應任何 `strategies_raw_all` 或本檔
  `strategies_raw_archived` 條目（現役與封存皆無），亦即從未被 `run_trading.py`
  正式跑過一次完整回測，`results/result.db` 中不存在對應的 METHOD/TRADE_METHOD
  組合。盤點結論：兩者皆為架構演進過程中的中繼草稿，在被賦予正式策略身分
  （config 條目）之前就已被下一代設計取代，程式碼保留供架構脈絡參考，
  不建議直接啟用（缺乏基準回測數據佐證）：

  - `pure_dtw_trading.py`：早期「Z-Score 交叉回穿波帶內才進場」構想的獨立
    交易類別，繼承 `zscore_trading.Trading` 並覆寫進出場條件。設計目的是
    搭配已封存的 DTW Paper 原版（座標 artifact 版）；DTW 系列改用誠實座標
    的 Fixed 版後，此交易邏輯未被重新接上，也從未取得自己的 config 條目。
  - `drl_lstm_v2_trading.py`：DRL v1（`drl_lstm_trading.py`，已封存）→ v2
    （本檔，修復假共享/獎勵重複計算/epsilon 排程三個缺陷）→ v3 FQI
    （`drl_fqi_trading.py`，已封存）→ v4 門檻選擇式（`drl_threshold_trading.py`，
    現役）演進鏈中的中間產物。v2 的修復在 v3 FQI 的 docstring 開頭有摘要
    引用，但 v2 本身在被 v3 取代前未被賦予獨立的策略 config 條目，因此
    `result.db` 沒有 v2 的基準回測數據。
"""

from strategies.config import (
    base_params, _HDBSCAN_UMAP_COMMON, _HDBSCAN_UMAP_FILTERS, _HDBSCAN_MS_FILTERS,
)

strategies_raw_archived = [
    # ── SSD Basic：2026-07-06 已自本清單拉回 strategies/config.py 現役
    #    （一切策略的基礎原型，Gatev 2006 累積回報指數 + 固定 β=1） ─────────
    # ── HDBSCAN Cluster 系列精簡（2026-07-06）：僅保留 PCA5 Z-Score 版於現役
    #    作為分組消融對照組，以下三條目封存（歷史結果在 results/result.db）──
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
            "ignore_ols_alpha":     True,
        },
    },
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
    # ── 舊 #3/#4：DTW Paper (DTW)/(SSD-DTW-PCA) ──────────────────────────────
    # 2026-07-05 以 formation_only 旗標回歸 strategies/config.py 現役清單
    # （見 docstring【座標系 artifact】節），自本清單移除。
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
        "trading_module":   "archive.trading.drl_lstm_trading",
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
        "trading_module":   "archive.trading.drl_lstm_trading",
        "sub_dir":          "HDBSCAN_UMAP_DRL",
        "db_method":        "HDBSCAN (UMAP-DRL)",
        "trade_method":     "DRL",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_UMAP_FILTERS, "reduce_method": "umap", "drl_episodes": 150, "drl_hidden_size": 256, "drl_num_layers": 2, "drl_batch_size": 512},
    },
    {
        "name":             "HDBSCAN MultiScale DRL",
        "formation_module": "strategies.formation.HDBSCAN_MultiScale",
        "trading_module":   "archive.trading.drl_lstm_trading",
        "sub_dir":          "HDBSCAN_MultiScale_DRL",
        "db_method":        "HDBSCAN (MultiScale-DRL)",
        "trade_method":     "DRL",
        "params": {**base_params, **_HDBSCAN_UMAP_COMMON, **_HDBSCAN_MS_FILTERS, "reduce_method": "umap", "drl_episodes": 150, "drl_hidden_size": 256, "drl_num_layers": 2, "drl_batch_size": 512},
    },
    # ── 舊 #14–#15：Kalman ───────────────────────────────────────────────────
    {
        "name":             "SSD Rolling Kalman",
        "formation_module": "strategies.formation.ssd_rolling",
        "trading_module":   "archive.trading.kalman_trading",
        "sub_dir":          "SSD_Rolling_Kalman",
        "db_method":        "SSD (Rolling-Kalman)",
        "trade_method":     "Kalman",
        "params":  {**base_params, "kalman_delta": 1e-4, "kalman_R": 1e-2},
    },
    {
        "name":             "HDBSCAN UMAP Kalman",
        "formation_module": "strategies.formation.HDBSCAN_UMAP",
        "trading_module":   "archive.trading.kalman_trading",
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
    # ── DRL v3 FQI（逐日定位動作空間已證偽；2026-07-04 封存） ────────────────
    {
        "name":             "HDBSCAN PCA-Loadings DRL FQI",
        "formation_module": "strategies.formation.HDBSCAN_PCA_Loadings",
        "formation_strategy_id_base": "HDBSCAN PCA-Loadings",
        "trading_module":   "archive.trading.drl_fqi_trading",
        "sub_dir":          "HDBSCAN_PCA_Loadings_DRL_FQI",
        "db_method":        "HDBSCAN (PCA-Loadings-DRL-FQI)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_episodes": 150, "drl_hidden_size": 128,
            "drl_num_layers": 1,
            "drl_scope": "global", "drl_buffer_periods": 24,
        },
    },
    {
        "name":             "SSD Rolling DRL FQI",
        "formation_module": "strategies.formation.ssd_rolling",
        "formation_strategy_id_base": "SSD Rolling",
        "trading_module":   "archive.trading.drl_fqi_trading",
        "sub_dir":          "SSD_Rolling_DRL_FQI",
        "db_method":        "SSD (Rolling-DRL-FQI)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "drl_episodes": 150, "drl_hidden_size": 128,
            "drl_num_layers": 1,
            "drl_scope": "global", "drl_buffer_periods": 24,
        },
    },
    {
        "name":             "HDBSCAN PCA-Loadings DRL FQI XL",
        "formation_module": "strategies.formation.HDBSCAN_PCA_Loadings",
        "formation_strategy_id_base": "HDBSCAN PCA-Loadings",
        "trading_module":   "archive.trading.drl_fqi_trading",
        "sub_dir":          "HDBSCAN_PCA_Loadings_DRL_FQI_XL",
        "db_method":        "HDBSCAN (PCA-Loadings-DRL-FQI-XL)",
        "trade_method":     "DRL",
        "params": {
            **base_params,
            "top_n_list":        [20],
            "stop_loss_list":    [0.0],
            "drl_episodes":      400,
            "drl_finetune_episodes": 120,
            "drl_hidden_size":   512,
            "drl_num_layers":    2,
            "drl_scope":         "global",
            "drl_buffer_periods": 36,
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
    # ── HDBSCAN PCA-Loadings 系列（2026-07-04 二次封存；被 Cluster-SSD-DTW-PCA
    #    系列取代，見上方 docstring「HDBSCAN PCA-Loadings 系列」節） ──────────
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
    # ── ML Pair Quality（2026-07-04 封存；兩次嘗試皆未驗證成功，
    #    見上方 docstring「ML Pair Quality」節） ─────────────────────────────
    {
        "name":             "ML Pair Quality",
        "formation_module": "strategies.formation.ml_pair_quality",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "ML_Pair_Quality",
        "db_method":        "ML (Pair-Quality)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
        },
    },
]
