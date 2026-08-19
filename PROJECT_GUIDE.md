# Pairs Trading 專案開發指南 (PROJECT_GUIDE)

---

## 1. 專案目錄結構

```text
pairtrade_trainingproject/
├── strategies/
│   ├── config.py                  # 5×3 分組×排序 Grid 宣告式展開（40 條現役策略）、網格參數、全域設定、敏感性 OFAT 產生器
│   ├── db_utils.py                # SQLite 合併、讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理（MSR 產業上限）
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── cluster_formation.py   # ★ 中性組裝器：feature_mode × cluster_method × ranking_backend 參數驅動，40 條現役策略共用
│   │   ├── _clustering.py         # 分群 dispatcher：hdbscan／agglomerative／kmeans backend（GICS 分組不經此層）
│   │   ├── _ranking.py            # 排序 dispatcher：委派 ssd_rolling.Formation（ssd）／DTW_Cointegration_Paper.Formation（dtw、ssd_dtw_pca）
│   │   ├── _features.py           # 報酬 PCA 因子載荷 + 基本面混合特徵萃取（供任何分群 backend 共用）
│   │   ├── _fundamentals.py       # FMP PIT 基本面讀取（從 agglomerative_FMP 抽出中性化）
│   │   ├── _cointegration.py      # 共整合篩選（現行僅 ADF 一道，EG 臨界值）；半衰期／Hurst 已停用，見論文附錄 B
│   │   ├── _utils.py              # 共用統計工具（_ols、_adf_stat、_compute_hurst、_residualize_returns、_bh_fdr_threshold、_cost_viable）
│   │   ├── ssd_rolling.py         # ⚠️ 非獨立策略入口，但 `_ranking.py` 仍動態 import 其 Formation 作為 "ssd" 排序引擎（現役、不可封存）
│   │   ├── DTW_Cointegration_Paper.py  # ⚠️ 同上，供 "dtw"／"ssd_dtw_pca" 排序引擎（現役、不可封存）
│   │   ├── HDBSCAN_PCA_Loadings.py / HDBSCAN_Cluster_SSD_DTW.py / agglomerative_yF.py / agglomerative_FMP.py
│   │   │     # 已封存的舊版一策略一模組寫法（2026-07-24 起讓位給 cluster_formation.py），程式碼保留供復活
│   │   ├── MST_PartialCorr_Cointegration.py / agglomerative_sec_pit.py / ssd_basic.py # 已封存策略模組（負面結果或已淘汰，保留供復活）
│   │   ├── ml_pair_quality.py     # 監督式學習排序 walk-forward 反事實回歸（已封存，程式碼保留）
│   │   ├── HDBSCAN_UMAP.py / HDBSCAN_MultiScale.py / ensemble.py  # 已封存
│   │   └── __init__.py
│   └── trading/
│       ├── zscore_trading.py      # Z-Score 狀態機（基礎類，三條 Spread 路徑；現役僅走路徑 B）
│       ├── drl_threshold_trading.py # DL-THR 門檻選擇模組（#9–11、#15–16 使用）
│       └── distance_trading.py    # ⚠️ GGR 2006 距離基準——config 端已封存（隨 #2 一併移除），檔案仍留在此目錄未搬移
├── analysis/                      # 評估層（讀 result.db，不重跑）
│   ├── regime_cost_dsr_eval.py    # regime 分層 Sharpe + break-even 成本表 + Deflated Sharpe
│   ├── proposition2_stats.py      # 交易層配對檢定（DL-THR vs 固定門檻，五種配對底）
│   ├── drl_behavior.py            # 從 trade_logs 還原 DL-THR 決策，解構增益來源（門檻選擇 vs SKIP）
│   ├── granularity_sweep.py       # （2026-07-28 新增，尚未整理進本指南）
│   └── sensitivity_report.py      # OFAT 參數敏感性報表（formation 變體曲線 + 交易端 top_n）
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   ├── sp500_yf_now.py            # yFinance 當日數據更新
│   ├── fundamentals_yfinance.py   # 公司基本面靜態快照（市值、本益比，供 agglomerative_yF）
│   ├── fetch_fmp_fundamentals.py  # FMP Point-in-Time 基本面 → parquet（供 agglomerative_FMP）
│   └── fetch_sec_fundamentals.py  # SEC EDGAR XBRL PIT 基本面 + Tiingo 原始股價 → parquet（供 agglomerative_sec_pit）
├── dataset/                       # 資料庫（大檔案透過 Git LFS 追蹤），分 price／fundamental 兩類
│   ├── price/                     # 價格類資料庫
│   │   ├── sp500_Tiingo.db        # 主要資料庫，`DB_PATH` 預設指向此
│   │   ├── sp500_yF.db            # yFinance 備用資料庫
│   │   └── sp500_Current.db       # 現行成分股查詢
│   ├── fundamental/               # 基本面類（DB 進 LFS；parquet／fmp_cache 本地不進版控）
│   │   ├── fundamentals_sp500.db  # 公司基本面快照（單一時點靜態資料，見下方限制說明）
│   │   ├── sp500_pit_2000_2025_monthly.parquet  # FMP Point-in-Time 基本面（本地產物）
│   │   └── fmp_cache/             # FMP API 快取（本地產物）
│   └── audit_report.csv           # 交易期資料品質審計報告
├── formation_data/
│   └── formation_pairs_sp500_Tiingo.db  # 形成期主合併資料庫（LFS 追蹤）
├── notebooks/                     # 策略筆記本（Quarto revealjs 投影片；2026-07-27 trim 至論文主軸，見 notebooks/README.md）
│   ├── formation/                 # 形成期 ×6：agglomerative_fundamentals、dtw_paper_fixed、hdbscan_cluster_pca5、kmeans_fundamentals、ssd_dtw_pca_paper_fixed、ssd_rolling
│   ├── trading/                   # 交易期 ×2：zscore_trading、drl_threshold_trading（distance_trading 已隨其策略封存移除）
│   ├── comparison.ipynb           # 現役策略績效總比較（讀 config.strategies_raw_all + result.db 動態產生）
│   ├── main_results.ipynb         # 主軸結果彙整（分組層／交易層／兩層組合）
│   ├── performance_guide.ipynb    # 績效指標說明
│   ├── _quarto.yml / slides.scss  # revealjs 投影片設定（大字型、Alt+點擊縮放、KaTeX）
├── docs/                          # GitHub Pages 輸出
│   ├── index.html                 # 入口頁（連結全部投影片）
│   └── slides/                    # quarto render 產出：comparison + main_results + performance_guide + formation×6 + trading×2
├── archive/                       # 歷史存檔（分類索引見 archive/README.md）
│   ├── config_archived_strategies.py  # 已封存策略 config（含孤兒模組盤點記錄）
│   ├── trading/                   # 非現役交易模組（drl_lstm×2、drl_fqi、kalman、pure_dtw）
│   ├── notebooks/                 # 舊版筆記本（114/11505/11506/11507 依時期）
│   ├── formation/                 # 已封存形成期模組（11506 CrossSector 系列等）
│   ├── scripts/                   # 一次性工具腳本
│   ├── docs/                      # 歷次學術 HTML 簡報（含舊版 formation/trading.html）
│   └── h200/                      # H200 GPU 伺服器相關（2026-07-06 起不再使用）
├── tools/                         # 輔助工具（皆從專案根執行）
│   ├── status.py                  #   pt status：資料/形成期/交易期/投影片 狀態總覽 + 建議動作
│   ├── snapshot_run.py            #   全量重跑前歸檔 result.db（原根目錄，2026-07 移入）
│   └── run_drl_variance.py        #   DL-THR 訓練變異數多輪評估（原根目錄，2026-07 移入）
├── dashboard.py                   # Streamlit 績效比對儀表板
├── run_formation.py               # 形成期主程式
├── run_trading.py                 # 交易期主程式
├── pt.bat                         # 統一指令入口：pt status/formation/trading/all/dashboard/slides/variance/snapshot/fetch-*
├── run.bat                        # 一鍵啟動 Dashboard（= pt dashboard，保留相容）
├── setup.bat                      # 環境初始化腳本（Windows；setup.sh 已封存至 archive/h200/）
├── requirements.txt               # Python 套件清單
└── pyproject.toml                 # 套件配置（Editable Install）
```

---

## 2. 執行工作流

```
setup.bat
    ↓
python run_formation.py     ← 形成期：篩選配對 → formation_data/formation_pairs_*.db
    ↓
python run_trading.py       ← 交易期：逐日模擬 → results/ + dataset/audit_report.csv
    ↓
run.bat                     ← Streamlit Dashboard（http://localhost:8501）
```

### run_formation.py 細節

- 讀取 `dataset/price/sp500_Tiingo.db`（或 config 指定資料庫，見 `DB_PROFILES`）
- 多行程平行執行每個策略的每個滾動期形成期計算（`ProcessPoolExecutor`，`spawn` context 避免 CUDA fork 污染）
- 結果寫入 `formation_data/formation_pairs_{db_basename}.db`
- 逐滾動期續傳：formation DB 的 `formation_progress` 表記錄每個「已嘗試」窗口（含空配對窗口）；策略完整性 = progress 期數 == 預期窗口數。中斷後重跑只補算缺漏窗口，不整策略重來。（舊 JSON 標記檔 fallback 已移除）
- `merge_databases()`：合併前先刪除同 `strategy_id` 的既有列再 `INSERT OR REPLACE`——避免非決定性模組（如已封存的 `ml_pair_quality.py`）重跑選出不同配對時，新舊兩批配對同時留在資料庫（同一期配對數超過 `top_n`）
- `FORCE_RERUN = False`（config.py）：正常模式，不強制重跑

### run_trading.py 細節

- 讀取 formation_data/ 的配對清單 + dataset/ 的價格資料
- 多行程平行執行交易期逐日模擬（同樣使用 `spawn` context）
- 輸出：`results/tiingo/` 下的 Trade Log CSV + `dataset/audit_report.csv` + `results/result.db` 的 `strategy_summaries`/`trade_logs`/`strategy_pairs`
- 網格搜尋：Top N / Stop Loss / MSR 等參數組合
- 所有交易全部失敗時拋出 `RuntimeError`（fail-loud），不會靜默回傳空結果
- 逐滾動期續傳（Z-Score 策略）：每期算完即以 pickle 落地至 `results/<dataset>/.ckpt/<策略>/<期>.pkl`；中斷重跑僅補算缺漏期，並自 checkpoint 重建 PortfolioManager 權益。summary 於全部期完成後定稿並清除 checkpoint。DL-THR 策略因 walk-forward 訓練狀態暫不套用（維持整策略重跑）
- **完成判定純以 result.db 為準**（`check_trading_completed` 查 `strategy_summaries` 有無該 config 列，不再要求 CSV 存在）
- **Trade Log CSV 為可選產物**：`config.WRITE_TRADE_CSV`（預設 True）。設 False 可省 ~14GB，不影響回測／續傳／儀表板（皆讀 result.db）

### results/ 目錄結構與重跑

```
results/
├── result.db            單一真相來源（儀表板 + 續傳；per-config 覆寫）
├── <dataset>/           Trade Log CSV（可選，WRITE_TRADE_CSV）+ pipeline logs
├── archive/             snapshot_run.py 歸檔的舊 result.db 與 summary CSV
└── analysis/            額外分析產物（如 drl_variance）
```

重跑三種模式：
1. **選擇性重跑**（改一個策略）：`STRATEGIES_SLICE="i:j" python run_trading.py` —— result.db 逐 config 覆寫，其餘不動。
2. **全量重跑前先歸檔舊版**：`python tools/snapshot_run.py <tag>`（或 `pt snapshot <tag>`）把 result.db 搬到 `archive/` 並匯出 summary CSV，再 `FORCE_RERUN=True` 重建。可新舊並存比較、回滾。
3. **輕量對照**：`python tools/snapshot_run.py <tag> --summary-only` 只匯出 summary CSV 不動 DB。

---

## 3. 策略清單（`config.py`）

**2026-08 主軸重整**：策略由單一組裝器 `strategies.formation.cluster_formation`
依 `cluster_method` × `ranking_backend` 宣告式展開。2026-08-17 起主軸改以
**分組層對候選池的限制強度**為有序維度（不再是「ML vs 傳統」的二分對照），
`_GRID_CLUSTERS` 擴為五級，主軸成為 **5 分組 × 3 排序 = 15 格**。

`strategies_raw_all` 為現役策略池（**40 條**，0-based 索引），
`strategies_raw = strategies_raw_all[:]` 決定實際執行範圍
（或用環境變數 `STRATEGIES_SLICE` 免改檔覆寫，支援逗號複合切片）：

```bash
STRATEGIES_SLICE="0:15" python run_trading.py   # 只跑主軸 15 格
```

⚠️ **`STRATEGIES_SLICE` 用的是索引，不是名稱。** 2026-08-17 曾因在
`run_drl_variance` 執行期間重整 config 造成索引位移，使第 4、5 輪跑到別的策略
且未報錯。該工具已改為每輪重新以 `db_method` 解析索引並檢查收成筆數；
但**任何長時間背景任務執行期間都不應修改 config 的策略清單**。

| # | 策略名稱 | `cluster_method` | 排除標的 | 角色 |
| :---: | :--- | :--- | ---: | :--- |
| 0–2 | Grid NOGRP-{SSD,DTW,SDP} | `none` | 0% | 主軸：限制強度零點 |
| 3–5 | Grid GICS-{SSD,DTW,SDP} | `gics` | 15.1% | 主軸：靜態產業先驗 |
| 6–8 | Grid HDB-{SSD,DTW,SDP} | `hdbscan` | 26.4% | 主軸：資料驅動分群 |
| 9–11 | Grid AGG-{SSD,DTW,SDP} | `agglomerative` | 35.3% | 主軸：資料驅動分群 |
| 12–14 | Grid KM-{SSD,DTW,SDP} | `kmeans` | 42.0% | 主軸：資料驅動分群 |
| 15–19 | Grid {AGG-SSD,HDB-SDP,KM-SSD,GICS-SSD,GICS-SDP} DRL | 借用對應格 | — | 交易層：DL-THR 疊加 |
| 20–22 | Grid AGG-SSD RLTHR E{05,10,20D} | 借用 #9 | — | 交易層：RL-THR 受控對照 |
| 23–28 | HSU25 {SSD,DTW,SDP}{,-REV} | `none` | 0% | 許鈞翔 (2025) 復現 |
| 29–33 | AGG-SSD-{NOSEC,NF,NF-NOSEC}、NOGRP-SSD-NF 等 | — | — | 分組×篩選×產業先驗消融 |
| 34–36 | Grid HAN{2-B1,3-REV,4-MONTHLY} | — | — | Han et al. (2021) 交易機制歸因鏈 |
| 37–39 | F09GI {HDB,AGG,KM}-STRUCT | — | — | 結構性財報特徵消融 |

實際清單以執行
`Project/Scripts/python.exe -c "import strategies.config as c; [print(i,s['name']) for i,s in enumerate(c.strategies_raw_all)]"`
為準。

**三項檢定**（論文第四章；不再使用「命題」的分類）：
**分組層**（#0–14）限制強度是否影響配對品質；
**交易層**（#15–19 對其配對底）學習式門檻選擇是否優於固定門檻；
**兩層組合**（資料驅動分群 + DL-THR vs GICS + 固定門檻）。

**2026-08-20 檢定結果**（`analysis/`，抽樣單位為時間、循環 block bootstrap L=126）：

- **三項檢定經 BH-FDR 校正後皆 0 顯著**，方向則一致：
  分組層 9/9 偏向 GICS、交易層 5/5 為正、兩層組合全期 3/3 為負。
- **分組層**：限制愈強績效愈差。逐格觀之，不分組三格全部優於同排序的 GICS，
  GICS 三格又全部優於任一資料驅動分群格。列平均 Sharpe 為
  不分組 +0.081、GICS +0.024、AGG −0.065、HDB −0.116、KM −0.249。
  ⚠️ 非嚴格單調：AGG 排除標的多於 HDB 卻表現較好。
- **機制**：期末強制平倉。強平率隨限制強度自 36.8% 單調升至 43.1%，
  與績效的相關在 15 個參數配置中一致為負（中位 r = −0.801）。
  逐筆勝率 0.578、獲利因子 0.969——多數交易收斂獲利但被少數大額虧損吃光。
  強平的配對再追 126 日僅 39.1% 回歸，其餘 |z| 自 4.88 擴大至 6.61。
  ⚠️ 全樣本混合的相關為 **+0.201**（符號相反，停損維度造成的 Simpson 悖論），
  **不可引用**。
- **交易層**：五底方向全正（+0.112 ~ +0.573pp），增益最大者為兩條 GICS 底。
  三項機械性替代解釋（SKIP 選股、門檻水準、總曝險）在 GICS 兩底皆排除；
  K-means 底的增益屬機械效應，Agglomerative 底無增益。
  五輪重訓的跨輪全距 0.010–0.047 個 Sharpe 單位。
- **絕對績效**：最佳為 GICS-SDP + DL-THR，25 年 10,000 → 12,548（年化 0.914%）；
  最佳固定門檻配置為不分組 × DTW（12,236）。
  同期無風險（2%）約 16,400——**任何配置皆未達此水準**。
  DSR 於 N=44 下無一通過 0.95（最高 0.769）。

**⚠️ 2026-08 管線修正**：形成期有六項實作與宣告不符（ADF 誤用 DF 臨界值、
Hurst 測錯對象、HDBSCAN 噪音被當成群、產業中位數插補、候選截斷、
交易統計計數器兩項錯誤），已全部修正。修正後主軸 13 格中 12 格績效改善，
ΔSharpe 中位 +0.236。**修正前的一切數字不可引用**；
完整記錄見 `thesis/07_附錄B_管線修正記錄.md`。

**已封存的負面結果**：舊版研究框架消融 ResidFDR、MST 偏相關圖候選、SEC-PIT Beta、
GICS 分組×排序×篩選 NF 消融、多尺度動量特徵消融，經回測皆為負面／劣於骨幹，已隨其宿主策略移至
`archive/config_archived_strategies.py`（程式碼與 result.db 數據保留，可復活）。`analysis/` 下的
regime 分層、break-even、Deflated Sharpe、`drl_behavior.py` 決策分解為現行評估層，續留現役。

**參數敏感性分析（OFAT，口試委員要求）**：`config.py` 內建 env 驅動變體產生器——
`SENSITIVITY_ALL=1` 一次產生 Tier-1 全部變體（`adf_pvalue_threshold`、`pca_n_components`、
`beta_feature_weight`、`entry_z`），或 `SENSITIVITY_PARAM=<param>` 單參數。formation 參數每值 =
一個獨立變體（自有 db_method，重跑 formation）；交易端參數改設 `_list` 沿用既有 formation。
評估：`python -m analysis.sensitivity_report`。

**已封存策略**：`archive/config_archived_strategies.py`
（HDBSCAN 舊特徵系×4、Ensemble×2、DRL v1×3、Kalman×2、CONV×2、
DRL FQI 系×3〔逐日定位動作空間已證偽〕、HDBSCAN PCA-Loadings DRL×2、ML Pair Quality×1 等；
此處 `DRL v1/FQI` 為**真正的**強化學習實作，與現役 DL-THR 不同），
封存理由與完整診斷數據見該檔 docstring，歷史回測結果保留於 `results/result.db`。
⚠️ HDBSCAN PCA-Loadings 的形成期配對仍被借用，formation DB 資料列不可刪。

**孤兒模組（2 個，從未進入策略清單）**：`pure_dtw_trading.py`（早期交叉回穿波帶構想）、
`drl_lstm_v2_trading.py`（DRL v1→v3 演進鏈中繼修復版）——程式碼保留供架構脈絡參考，
`result.db` 無對應回測數據，盤點記錄見 `archive/config_archived_strategies.py` docstring末段。

**非現役交易模組已歸位 `archive/trading/`（2026-07-05）**：上述孤兒模組×2 加上已封存的
`drl_lstm_trading.py`（v1）、`drl_fqi_trading.py`（v3）、`kalman_trading.py` 共 5 檔自
`strategies/trading/` 移入；封存 config 的 `trading_module` 已同步指向 `archive.trading.*`，
復活時不需搬回檔案。`strategies/trading/` 現含現役的 `zscore_trading.py`、`drl_threshold_trading.py`；
`distance_trading.py`（GGR 2006 距離基準，隨舊 #2 一併封存）仍在此目錄但已無現役策略引用，僅
`archive/config_archived_strategies.py` 保留其 `trading_module` 指標供復活。

---

## 4. 形成期策略說明（2026-07 中性組裝架構）

### 三層流水線：特徵 → 分群 → 排序（`cluster_formation.py` 組裝）

```
_features.py            _clustering.py              _ranking.py
（報酬 PCA 因子載荷    →  cluster(method, X, …)   →  rank_within_groups(backend, …)
 ⊕ 基本面 ⊕ GICS         "hdbscan"/"agglomerative"    "ssd"      → ssd_rolling.Formation
 one-hot 混合特徵）       /"kmeans"（GICS 分組         "dtw"／    → DTW_Cointegration_Paper
                          略過此層，直接用真實            "ssd_dtw_pca"   .Formation
                          GICS_Sector 當群標籤）
```

`cluster_formation.py` 依 `strategies/config.py` 傳入的 `feature_mode`／`cluster_method`／
`ranking_backend`／`filter_mode` 四個參數，把上述三層串成一條 formation pipeline；17 條現役
策略（見第 3 節）皆走同一份程式碼，唯一差異是這四個參數的組合。舊版「一策略一模組」的公式
與診斷細節（已封存但可復活）仍可在對應 notebook 找到，見 `notebooks/formation/`
（`hdbscan_cluster_pca5.ipynb`、`agglomerative_fundamentals.ipynb`、`kmeans_fundamentals.ipynb`、
`ssd_rolling.ipynb`、`dtw_paper_fixed.ipynb`、`ssd_dtw_pca_paper_fixed.ipynb`）。

### 統計過濾（`_cointegration.screen_pair`，`filter_mode` 控制開關）

**現行僅一道**（`filter_mode="adf_only"`，2026-08-14 起）：

| 指標 | 設定 |
| :--- | :--- |
| ADF（Engle-Granger 殘差） | $p <$ `adf_pvalue_threshold`（現役統一 0.05） |

⚠️ **p 值取自 MacKinnon 響應曲面（`regression="c"`, `N=2`），不是 `adfuller` 直接回傳的值。**
後者是 Dickey-Fuller 分布，用於估計殘差會系統性過度拒絕——實測通過率 77.4%，
改用 EG 臨界值後為 9.5%。見 `_utils._adf_stat` 的 docstring。

**半衰期與 Hurst 已停用**（原為三道過濾）。逐層拆解顯示三道實為一道：
ADF 淘汰 22.6%、半衰期 **0.2%**、Hurst **96.8%**；而承擔 96.8% 的 Hurst
兩個實作都對價差的**水準值**做 R/S，標準估計量定義在**增量**上，
標準版的 `H<0.5` 通過率為 0%。完整診斷見 `thesis/07_附錄B_管線修正記錄.md` B.1.2。

`filter_mode` 三值：`adf_only`（現役）／`coint`（三道，僅供復現舊結果）／
`none`（零道，消融用）。現役 40 條中 34 條為 `adf_only`、6 條為 `none`
（刻意的無篩選消融）。

### 「組合優於重寫」架構模式（已從模組層下沉到函式層）

舊版模式是「新策略 = import 既有策略模組的內部方法」（如 `HDBSCAN_PCA_Loadings._build_feature_matrix()`
被 `HDBSCAN_Cluster_SSD_DTW.py` 借用）；2026-07 重構後同一設計哲學延續，但耦合單位從
「策略模組互相 import」下沉為「中性函式庫供任何策略呼叫」——新增分群法只需在 `_clustering.py`
加一個 backend、新增排序準則只需在 `_ranking.py` 加一個 backend，不必碰任何既有策略程式碼，
也不會產生模組間的隱藏依賴（見 `archive/README.md` 的耦合說明）。

**分組層對照設計**（同排序準則、同篩選、同交易端，唯一變因 = 分組方式）：
不分組（#0–2）↔ GICS 產業（#3–5）↔ HDBSCAN／Agglomerative／K-means（#6–14）。
五級依對候選池的限制強度排列，見第 3 節的檢定結果。

---

## 5. 交易期策略說明

### Spread 重建（`zscore_trading.py`，現役策略統一走路徑 B）

$$P'_{i,t} = \frac{\ln P_{i,t} - \mu^{form}_{\ln P_i}}{\sigma^{form}_{\ln P_i}}, \qquad \text{Spread}_t = P'_{A,t} - \beta \cdot P'_{B,t}$$

路徑由 `run_trading.py` 依 `Formation_Params` 是否含 `OLS_Alpha` 及 `ignore_ols_alpha` 參數決定；
現役策略全數強制或原生走此路徑（路徑 A 原始 log-price OLS 殘差空間、路徑 B1 累積回報比值空間
僅供已封存策略使用，程式碼保留於 `zscore_trading.py._compute_spread()`）。

### DL-THR 門檻選擇式 v4（`drl_threshold_trading.py`，db id 仍為 `…-DRL`）

- **動作空間**：SKIP + 8 組 `(entry_z, exit_z)` 門檻 ∈ `{1.5,2.0,2.5,3.0} × {0.0,0.5}`，每配對每期只選 1 個
- **學習範式**：Walk-forward 反事實監督回歸（12 維形成期特徵 → 9 動作預期報酬），非探索型 RL——
  歷史配對期全部 9 個動作的報酬皆可精確反事實回算，全資訊監督問題
- **無前視保證**：`eligible = [(f,r) for f,r,te in buffer if te < trade_start_k]`——只用交易期已於本期開始前
  結束的樣本訓練
- **`_shared` 狀態隔離**：以 `variant_id`（策略名稱+Top_n+停損+MSR）為 key，避免同一 worker process
  依序處理不同策略變體時互相污染訓練資料（已修復的真實 bug，見 commit `9ae56ec`）
- 前代（v1 online DQN、v2 修復版、v3 FQI）逐日定位動作空間已系統性證偽，已封存，見
  `archive/config_archived_strategies.py`

### 六大風控機制

1. **SL**：個配對停損（`stop_loss_pct`，網格 `[0.0, 0.05, 0.15]`）
2. **DSZ**：Z-Score 發散停損（`dynamic_stop_z`）
3. **PSL**：全域組合停損（`portfolio_stop_loss_pct`，使用 `Unrealized_PnL` 計帳）
4. **MSR**：產業分散上限（`max_sector_ratio`）
5. **Cooldown**：方向性冷卻（等 Z 穿越 0 才解凍）
6. **VOL ADJ**：波動率自適應（`use_vol_adjust`，動態放大 σ）

兩個交易模組（`zscore_trading.py`、`drl_threshold_trading.py`）共用完全相同的部位配置公式與六大風控設計原則，
確保「固定門檻 Z-Score 基準 vs DL-THR 疊加」比較時唯一變因是交易決策邏輯。詳見 `notebooks/trading/drl_threshold_trading.ipynb`。

---

## 5.5 論文章節與產出對照

`thesis/` 的八份檔案與 `analysis/`／`tools/` 的對應關係：

| 論文 | 內容 | 主要來源 |
| :--- | :--- | :--- |
| 第三章 | 研究方法（四層架構、統計檢定、敏感性） | `strategies/`、`tools/thesis_numbers.py` |
| 4.1 | 分組層：限制強度與配對品質 | `proposition1_daily_hac.py` |
| 4.2 | 交易層：門檻選擇 | `proposition2_daily_hac.py`、`prop2_*.py`、`drl_behavior.py` |
| 4.3 | 兩層組合 | `prop3_combined_system.py` |
| 4.4 | 強制平倉機制 | `trade_logs` 直查、`forced_close_by_config.csv` |
| 4.5 | 檢定力不對等 | 自 4.1／4.2 的 CI 反推 |
| 4.6 | 風險評估 | `regime_cost_ew.py`、`regime_cost_dsr_eval.py` |
| 附錄 A | 前行研究差異定位 | `hsu25_entry_timing.py` |
| 附錄 B | 管線修正記錄 | `git 62554de`、`54c8856`；備份庫 |
| 附錄 C | 交易端三項否證 | `regime_cost_dsr_eval.build_market_regimes` + 逐筆反事實 |

**數字真相表**：`python -m tools.thesis_numbers --all` 自三個資料庫重算論文
引用的全部數字，輸出 `results/analysis/thesis_numbers.{json,md}`，
依章節分組並標註新鮮度（`formation_groups` 有列 = 已被最近一次重跑覆寫）。
論文中的任何數字都應與該表一致。

**交易統計欄位若有疑義**：`Entries`／`Exits`／`Stop_Losses`／`Forced_Closes`／
`Win_Rate`／`Profit_Factor` 曾有兩項計數錯誤（見附錄 B.2），
已於 2026-08-17 修正並以 `tools/recompute_trade_stats.py` 重算全部配置。
該工具以 SQL 視窗函式重算，不必重跑回測。

## 6. 資料庫規範

### dataset/ — 價格與基本面資料庫（Git LFS）

| 檔案 | 用途 |
| :--- | :--- |
| `price/sp500_Tiingo.db` | 主要資料庫，`DB_PATH` 預設指向此 |
| `price/sp500_yF.db` | yFinance 備用（config `DB_PROFILES` 可切換） |
| `price/sp500_Current.db` | 現行成分股查詢 |
| `fundamental/fundamentals_sp500.db` | 公司基本面快照（市值、本益比），供 `agglomerative_yF.py` 使用；⚠️ 單一時點靜態資料，非歷史逐日序列。以 `fetch/fundamentals_yfinance.py` 重抓（843 檔：633 有效、210 已下市）。逐點 PIT 版見 `fundamental/sp500_pit_2000_2025_monthly.parquet`（FMP／SEC，供 `agglomerative_FMP.py`／`agglomerative_sec_pit.py`） |
| `fundamental/sp500_pit_2000_2025_monthly.parquet`、`fundamental/fmp_cache/` | FMP Point-in-Time 基本面與 API 快取（本地產物，不進版控） |

資料表：`Daily_Prices`（Date, Symbol, Open, High, Low, Close, Volume）、`Constituents`（Symbol, GICS_Sector）

### formation_data/ — 形成期結果資料庫（Git LFS）

| 檔案 | 說明 |
| :--- | :--- |
| `formation_pairs_sp500_Tiingo.db` | 所有策略的形成期配對主資料庫 |
| `formation_pairs_sp500_Tiingo_*.db` | 各策略獨立暫存庫（測試用，不上傳 Git） |

---

## 7. 策略新增 SOP

**新增分群法或排序準則（現行主要路徑）**：不必新建 `Formation` 模組——在 `_clustering.py` 加一個
`cluster_*` backend（分群法）或在 `_ranking.py` 加一個 `rank_*` backend（排序準則），註冊進
`_BACKENDS` dict，再到 `strategies/config.py` 的 `_GRID_CLUSTERS`/`_GRID_RANKINGS` 加一行即可自動
展開進 3×3 矩陣。

**新增全新獨立形成邏輯（例外路徑，僅當中性組裝器的三層無法表達新想法時使用）**：

1. 在 `strategies/formation/` 建立新模組，實作 `class Formation` 與 `run()` 方法，回傳含
   `Ticker_A/B, Rank, Hedge_Ratio, Spread_Mean, Spread_Std, Log_Mean_A/B, Log_Std_A/B` 等欄位的 DataFrame
   （這 7 個是 `run_trading.py` 讀 `Formation_Params` 的硬性合約，缺一不可）
2. 優先考慮「組合」既有中性層（`_features`/`_clustering`/`_ranking`/`_cointegration`，見第 4 節）
   而非重寫統計邏輯
3. 在 `strategies/trading/` 確認交易期模組（通常直接使用 `zscore_trading.py` 或
   `drl_threshold_trading.py`，兩者接口一致）
4. 在 `strategies/config.py` 的 `strategies_raw_all` 新增策略字典，指定 `formation_module`、
   `trading_module` 及所有 params；若要借用其他策略已算好的配對，加上
   `formation_strategy_id_base` 指向該策略名稱
5. 依序執行 `run_formation.py` → `run_trading.py`
6. 用 `results/result.db` 的 `strategy_summaries` 與現有基準（Grid GICS-SSD／Grid GICS-SDP，
   即分組層的傳統對照）做同條件對照；驗證無效則移入 `archive/config_archived_strategies.py`
   並記錄診斷結論（不刪除歷史數據）

---

## 8. Git 規範

### 追蹤原則

| 類型 | 處理 |
| :--- | :--- |
| 核心 `.py` 程式碼 | 一律追蹤 |
| `dataset/*.db`、`formation_data/formation_pairs_sp500_Tiingo.db` | Git LFS 追蹤（`.gitattributes` 設定） |
| `formation_data/formation_pairs_sp500_Tiingo_*.db` | `.gitignore` 忽略（測試暫存） |
| `results/`（含 `results/result.db`）、`tmp/`、`scratch/`、`data/` | `.gitignore` 忽略（回測輸出，本機重算即可還原） |
| `*.db-shm`、`*.db-wal` | `.gitignore` 忽略（SQLite WAL 暫存） |

### 注意事項

- 執行 `git add formation_data/*.db` 前確認已安裝 Git LFS（`git lfs install`）
- 大型 DB 首次推送需要 LFS 儲存空間配額
- ⚠️ **LFS 額度已用罄（2026-07-05 確認）**：`git lfs fetch/pull/push` 皆會失敗
  （"This repository exceeded its LFS budget"）。在新環境 LFS 檔案只會是 pointer 檔
  （~130 bytes），遇到 `file is not a database` 或 run_trading「No periods found」即為此因。
  重建方式：`fetch/fundamentals_yfinance.py`（基本面）＋ `run_formation.py`（形成期配對，
  formation-only 條目會一併產生 DTW 原版配對）。額度恢復前 **不要 commit/push .db 檔案**。
- `results/` 整個目錄被忽略：回測結果只存在本機。績效比較改由 `notebooks/comparison.ipynb`
  動態讀取 result.db 產生——重大回測更新後重新執行該筆記本並 `quarto render`，
  `docs/slides/comparison.html` 即同步最新數據
