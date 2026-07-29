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

【研究框架消融 — 負面結果（2026-07-15 封存）】
  三個「研究框架次步」策略經全時段回測皆為負面/劣於現役骨幹，退出現役清單：
  - HDBSCAN (PCA-Loadings-ResidFDR)：研究框架 #1+#2+#3（因子殘差化＋BH-FDR＋成本
    過濾）合併示範。中位 Sharpe −0.402（全組最差）；BH-FDR 確實把強平率由 ~58%
    降到 39%，但存活配對仍靠弱的 PCA-Loadings 品質分數排序 → 少而不精。證明
    「激進統計修剪救不了弱排序器」；#1 殘差化的正面效果保留在現役 #7
    HDBSCAN Cluster PCA5 Resid（配強 SSD-DTW 排序，最佳年化 +1.24%）。
  - MST (PartialCorr-SSD-DTW-PCA)：研究框架 #4，偏相關網路圖（Ledoit-Wolf 精確
    矩陣）候選生成。中位 Sharpe −1.474（三方候選對照最差），且與獲利的聚類配對
    幾乎零重疊——圖稀疏骨幹按相關強度選邊，是 tradeable 共整合的差代理。可寫的
    反證：候選生成的關鍵是「豐富且含可交易對的候選池」，非稀疏優雅的圖拓撲。
  - Agglomerative (SEC-PIT-Beta)：研究框架 #5，Agglomerative 加 Beta 風險先驗。
    等權(1.0)加入 Beta 全面變差（中位 Sharpe −0.500 vs FMP −0.227）；高變異的
    Beta 主導分群距離、稀釋既有良好表徵。敏感性掃描證實低權重(0.25)才中性偏正。
  模組檔案保留於 strategies/formation/（HDBSCAN_PCA_Loadings.py 仍被 #6/#7 使用；
  MST_PartialCorr_Cointegration.py、agglomerative_sec_pit.py 保留供復活）；
  對應 notebook 移至 archive/notebooks/negative_results/。研究框架 #6 評估層
  （analysis/regime_cost_dsr_eval.py、sensitivity_report.py）為分析工具，續留現役。
  母題：好的基礎表徵已捕捉結構，額外堆疊特徵/激進修剪常是稀釋而非增益。

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
    # ══ 2026-07-25 主軸收斂：現役精簡至 15 條（4 分組 × 3 排序 + 3 DRL）══════
    # 論文主軸＝「機器學習分群（命題 1）+ 深度學習交易（命題 2）」，現役清單
    # 只保留直接支撐兩命題的策略；下列為移出主軸者，成果全在 results/result.db。
    #
    # 【A. 原生傳統基準 4 條 + formation-only 2 條】（下方 dict 條目）
    #   SSD (Basic)/(Rolling)、DTW、SSD-DTW-PCA 及其配對來源 DTW Paper ×2。
    #   移出理由：傳統基準改由 Grid (GICS-SSD/DTW/SDP) 擔任——ADF 門檻統一 0.05，
    #   與 ML 分群列可比，構成乾淨的 4×3 矩陣；原生條目 ADF 預設不一致
    #   （ssd_rolling 0.05 vs DTW 模組 0.01），不宜混在同一張表。
    #   ⚠️ Grid (GICS-SSD) 已驗證與 SSD Rolling 數值完全相同（1.66%/Sh0.20/PF1.19），
    #   兩者方法論等價，故主軸保留前者不損失任何實證內容。
    #   論文定位：附錄「文獻原始設定復現」（Gatev 2006 原型、許鈞翔 2025 ADF 0.01）。
    #
    # 【B. 篩選消融 3 條】Grid (GICS-SSD-NF/DTW-NF/SDP-NF)（config 迴圈已改為只產生
    #   coint 分支）。成果：三道統計過濾貢獻 +0.25~0.87pp。
    #   論文定位：附錄「方法論設計依據——為何加入共整合篩選」。
    #
    # 【C. 結構性財報特徵消融 6 條】F09 (HDB/AGG/KM)-(BASE/STRUCT)（2009+ 期間）。
    #   成果：10 維 SEC XBRL 財報比率對分群品質無顯著貢獻（三分群 Δ 皆在 ±0.22pp
    #   噪音範圍內）。與動量特徵消融同為「特徵不是瓶頸」的證據。
    #   論文定位：附錄「特徵工程消融」。
    #
    # 【D. regime 閘門三層疊加 3 條】Grid (AGG-SSD/HDB-SDP/KM-SSD)-DRL-DG25。
    #   成果：AGG 版五輪中位 2.69% [2.65, 2.71]、Sharpe 0.40、最差輪 14/15 正 Sharpe，
    #   為全專案最穩健配置；但 regime 條件化進場不屬本論文兩命題，
    #   論文定位：附錄「延伸探索——regime 條件化進場」。
    #   機制保留：disp_gate_pctl 參數於 zscore_trading / drl_threshold_trading 皆在。
    #
    # 復活：把下方條目貼回 strategies/config.py；B/C/D 三組為 config 迴圈展開，
    #   復活方式見本檔案末的註記與 git 歷史（commit 「主軸收斂」之前的版本）。

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

    # ══ 2026-07-24 封存：多尺度動量特徵消融（負面結果，3 條）══════════════
    # 假說：ref 內 ML 配對文獻（Han, He & Toh 2021）用 48 動量因子 + 78 公司特徵分群，
    #   本研究僅 19 維（5 報酬 PCA ⊕ 2 基本面 ⊕ 12 產業 one-hot）；3×3 矩陣顯示
    #   ML 分群僅小勝 GICS（1.77% vs 1.66%），疑似受限於特徵而非演算法。
    #   → 加入 8 維多尺度動量/波動（mom 1/3/6/9/12 月 + vol20d/vol60d/downvol60d）。
    # 驗證（固定 SSD 排序與篩選，唯一變因＝特徵集，各 15 網格）：
    #   HDBSCAN       1.29% → 0.35%（−0.94pp）
    #   Agglomerative 1.77% → −0.65%（−2.43pp，15 格 0 個正 Sharpe）
    #   K-means       1.31% → −0.09%（−1.40pp）
    #   三種分群全面劣化，非參數噪音。
    # 根因：(1) 動量度量「過去漲跌幅」而非「走勢同步性」——漲幅相近 ≠ 價差會
    #   回歸，與配對交易的目標不一致；(2) 動量的橫斷面變異大於 PCA 載荷，在
    #   歐氏距離中主導分群、稀釋原本有效的因子暴露訊號（與 SEC-PIT-Beta
    #   「高變異特徵主導距離」同一失敗模式）。
    # 對文獻的正確理解：Han, He & Toh (2021) 的動量特徵用於「識別高估/低估股票」
    #   （交易訊號層），分群依據是 78 個公司特徵，且作者強調公司特徵「更具
    #   前瞻性」。正確的擴充方向是結構性基本面特徵，非價格動量。
    # 學術價值：排除一條看似合理的特徵擴充路徑，成本僅 45 網格。
    # 復活：_features.build_momentum_features 與 cluster_formation 的
    #   feature_mode="momentum"/"momentum_mix"、momentum_* 參數皆保留。
    {
        "name":             "Grid HDB-SSD MOM",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Grid_HDB_SSD_MOM",
        "db_method":        "Grid (HDB-SSD-MOM)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "feature_mode": "momentum_mix",
                   "cluster_method": "hdbscan", "ranking_backend": "ssd",
                   "pca_n_components": 5, "adf_pvalue_threshold": 0.05,
                   "momentum_horizons": (1, 3, 6, 9, 12),
                   "momentum_include_vol": True, "momentum_weight": 1.0},
    },
    {
        "name":             "Grid AGG-SSD MOM",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Grid_AGG_SSD_MOM",
        "db_method":        "Grid (AGG-SSD-MOM)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "feature_mode": "momentum_mix",
                   "cluster_method": "agglomerative", "ranking_backend": "ssd",
                   "pca_n_components": 5, "adf_pvalue_threshold": 0.05,
                   "momentum_horizons": (1, 3, 6, 9, 12),
                   "momentum_include_vol": True, "momentum_weight": 1.0},
    },
    {
        "name":             "Grid KM-SSD MOM",
        "formation_module": "strategies.formation.cluster_formation",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Grid_KM_SSD_MOM",
        "db_method":        "Grid (KM-SSD-MOM)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "feature_mode": "momentum_mix",
                   "cluster_method": "kmeans", "ranking_backend": "ssd",
                   "pca_n_components": 5, "adf_pvalue_threshold": 0.05,
                   "momentum_horizons": (1, 3, 6, 9, 12),
                   "momentum_include_vol": True, "momentum_weight": 1.0},
    },

    # ══ 2026-07-23 封存：被 3×3 Grid 消融矩陣取代的舊分群/疊加系列（14 條）══
    # 背景：形成期中性化重構後（_features / _clustering / _ranking / cluster_formation），
    #   「分群 × 排序」的所有組合改由宣告式 3×3 Grid 展開，舊的每組合一模組寫法退場。
    # 對應關係（封存 → 取代者）：
    #   Agglomerative (FMP)              → Grid (AGG-SSD)   ※ 經回歸測試 bit-identical，
    #                                       同窗口選出完全相同的 20 對配對（雜湊一致）
    #   Agglomerative (FMP-DTW)          → Grid (AGG-DTW)
    #   Agglomerative (FMP-SSD-DTW-PCA)  → Grid (AGG-SDP)
    #   Agglomerative (FMP·DRL)          → Grid (AGG-SSD-DRL)
    #   Agglomerative (FMP·DRL-DG25)     → Grid (AGG-SSD-DRL-DG25)
    #   HDBSCAN / HDBSCAN (殘差)          → Grid (HDB-*) 系列（同分群演算法，中性層組裝）
    #   Agglomerative (yF) / (yF·DRL)    → FMP PIT 版取代（yF 為單一時點快照，有前視偏誤）
    #   Agglomerative (FMP-DG25/DG50/DG25-EZ) → 閘門效應已在 Grid 層驗證並保留
    #   SSD (Distance) / SSD (DRL)       → 交易端對照，DRL 增益已於 Grid 層量化
    # 保留的實證價值（歷史數據全在 results/result.db，可直接查詢引用）：
    #   - DG 閘門系列：DG25 使全網格 Sharpe 轉正、MDD 下降、PF 上升的原始證據
    #   - FMP·DRL-DG25：五輪變異數 2.68% [2.64, 2.84]（旗艦策略的早期版本）
    #   - SSD (Distance)：GGR 距離基準 vs 回歸基準的交易端單變因對照
    # 復活方式：把下方條目貼回 strategies/config.py 的 strategies_raw_all；
    #   所需 formation 模組（agglomerative_yF/FMP、HDBSCAN_Cluster_SSD_DTW）皆未刪除。
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

    # ══ 交易端微觀規則戰役（2026-07-19 封存，P1/P2/P4 全數負面/中性）═══════
    # 背景：FMP Top1 交易解剖顯示期末強平桶（238 筆，勝率 21.4%）ΣPnL -19,530，
    #   >63d 未收斂勝率 26-46% → 提出時間停損/提早出場/DRL 時間維度三案。
    # P1 TS63/TS42（時間停損，zscore_trading.max_holding_days 已接線）：
    #   全網格一致劣化（TS63 Top1 Δ年化 -0.71pp；TS42 -1.23pp）。
    #   根因=後見之明偏誤：追蹤 174 筆 >63d 交易，第 63 天浮虧 -13,745、
    #   期末回升至 -11,166（平均每筆 +15 回血）——63d 是損益低谷，
    #   時間停損砍在谷底，且放棄 63-90d 桶近半數晚收斂贏家。
    #   教訓：「最終虧損者持倉長」≠「提早出場更好」；部分均值回歸仍在回歸。
    # P2 XZ05（exit_z 0→0.5 提早下車）：全網格 -0.14~-0.28pp。
    #   勝率微升但單筆獲利縮水更多；省下的尾部風險 < 放棄的收斂利潤。
    # P4 DRL-v5（動作選單加 max_hold∈{0,63}，17 動作）：
    #   最佳年化 2.01% vs v4 2.25%——重訓噪音範圍內的中性結果；
    #   agent 拿到反事實真值後大多學會不用時間停損，與 P1 結論互洽。
    #   thr_menu_version=5 機制保留於 drl_threshold_trading.py 供復活。
    # 總結論：交易端微觀規則已系統性掃過（含更早的 DynCap 配置端），
    #   2.25-2.28% 為現有訊號組天花板；僅存實證槓桿為 regime 條件化進場
    #   （獲利集中於橫斷面分散度 Q4：+8,674 vs Q1+Q2 -4,920）。
    {
        "name":             "Agglomerative Fundamentals (FMP) TS63",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_TS63",
        "db_method":        "Agglomerative (FMP-TS63)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "adf_pvalue_threshold": 0.05, "max_holding_days": 63},
    },
    {
        "name":             "Agglomerative Fundamentals (FMP) TS42",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_TS42",
        "db_method":        "Agglomerative (FMP-TS42)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "adf_pvalue_threshold": 0.05, "max_holding_days": 42},
    },
    {
        "name":             "Agglomerative Fundamentals (FMP) XZ05",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_XZ05",
        "db_method":        "Agglomerative (FMP-XZ05)",
        "trade_method":     "Z-Score",
        "params": {**base_params, "adf_pvalue_threshold": 0.05, "exit_z": 0.5},
    },
    {
        "name":             "Agglomerative Fundamentals DRL THR (FMP) v5",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "formation_strategy_id_base": "Agglomerative Fundamentals (FMP)",
        "trading_module":   "strategies.trading.drl_threshold_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_DRL_v5",
        "db_method":        "Agglomerative (FMP·DRL-v5)",
        "trade_method":     "DRL",
        "params": {**base_params, "adf_pvalue_threshold": 0.05,
                   "drl_hidden_size": 64, "thr_train_epochs": 40,
                   "thr_min_train_samples": 200, "thr_menu_version": 5},
    },
    # ── FMP-Resid：Agglo-FMP × 因子殘差（2026-07-19 封存，負面結果）────────
    # 假說：合體兩個各自驗證有效的成分——FMP PIT 基本面分群（最佳年化 2.28%）
    #   × 因子殘差化表徵（在 HDBSCAN Cluster 管線中有效）——應能再推高配對品質。
    # 驗證（vs Agglomerative (FMP) 基準，唯一差異 factor_residual=True）：
    #   15 網格 0 個正 Sharpe（中位 -0.69）；最佳年化 -0.90% vs 基準 2.28%；
    #   SL0% 各 TopN 一致劣化 ΔSharpe -0.46~-0.57（系統性失效，非參數噪音）。
    # 根因：(1) 表徵/交易空間錯位——殘差化群集聚「特殊性共動」，但下游
    #   ssd_rolling 的 min-SSD 排序與 spread 建構在原始標準化價格空間，
    #   β 差異使群內 SSD 最小配對在原始空間反而不穩（HDBSCAN Cluster 版因
    #   下游為嚴格雙向 OLS+ADF 0.01 共整合篩選而能擋掉錯配，Agglo 的
    #   ADF 0.05+SSD 排序擋不住）；(2) 特徵自相矛盾——殘差化移除產業共動，
    #   GICS one-hot 區塊（12 維）又把它加回，兩區塊互相抵銷。
    # 學術價值：「成分有效 ≠ 可移植」——殘差化的有效性依賴下游排序機制相容。
    # 復活：agglomerative_FMP.Formation(factor_residual=True) 參數保留；
    #   result.db 保留 METHOD='Agglomerative (FMP-Resid)' 15 筆網格結果；
    #   formation DB 保留 'Agglomerative Fundamentals (FMP) Resid_MSR0' 2767 列。
    {
        "name":             "Agglomerative Fundamentals (FMP) Resid",
        "formation_module": "strategies.formation.agglomerative_FMP",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_FMP_Resid",
        "db_method":        "Agglomerative (FMP-Resid)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "pca_n_components":            5,
            "factor_residual":             True,
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
    # ── DynCap：動態槽位資金配置（2026-07-19 封存，負面結果）──────────────
    # 假說：靜態等權切槽（equity ÷ top_n×6）使 40%+ 資金閒置，動用資本年化
    #   （~4.6-5.9%）被稀釋成帳面 ~2%；以歷史同時承諾數 75 百分位校準有效
    #   槽位數應可把帳面年化推上 3%。
    # 驗證（vs Agglomerative (yF) 靜態基準，同配對同交易端）：
    #   最佳年化 1.80% → 1.62%（不升反降）；Top1 -0.16pp、Top3 +0.67pp、
    #   Top10 -2.60pp（MDD -13%→-58.5%）、Top20 -2.96pp（MDD -7%→-58.3%）。
    # 根因：幾何複利下集中配置同倍放大報酬與波動，vol drag（≈σ²/2）把
    #   放大的算術報酬吃回並倒貼；閒置現金實為靜態等權的隱性波動控制。
    #   排隊護欄另使 Top20 利用率 35%→28%，失分散而未得集中收益。
    # 學術價值：實證「動用資本年化無法靠配置端集中免費兌現」。
    # 復活：PortfolioManager(dynamic_slots=True, slot_percentile, pair_cap_frac,
    #   slot_warmup_obs) 機制仍在 strategies/portfolio_manager.py；
    #   result.db 保留 METHOD='Agglomerative (yF-DynCap)' 15 筆網格結果。
    {
        "name":             "Agglomerative Fundamentals (yF) DynCap",
        "formation_module": "strategies.formation.agglomerative_yF",
        "formation_strategy_id_base": "Agglomerative Fundamentals (yF)",
        "trading_module":   "strategies.trading.zscore_trading",
        "sub_dir":          "Agglomerative_Fundamentals_yF_DynCap",
        "db_method":        "Agglomerative (yF-DynCap)",
        "trade_method":     "Z-Score",
        "params": {
            **base_params,
            "pca_n_components":            5,
            "fundamentals_db_path":        "./dataset/fundamental/fundamentals_sp500.db",
            "adf_pvalue_threshold":        0.05,
            "dynamic_slots":               True,
            "slot_percentile":             75.0,
            "pair_cap_frac":               0.15,
            "slot_warmup_obs":             8,
        },
    },
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
