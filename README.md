# S&P 500 Pairs Trading 量化回測平台

本專案針對 S&P 500 成分股實作配對交易 (Pairs Trading) 滾動回測系統，支援多種形成期策略（含機器學習配對）、深度強化學習交易期模組與互動式績效視覺化儀表板。

---

本研究兩大命題：**命題1（形成期）** 機器學習分組能找到比傳統距離／共整合法更高品質的配對；
**命題2（交易期）** 深度學習（DRL）能比傳統 Z-Score 有更好的交易績效。

> **2026-07-29 命題檢定結果**（純分析層，未重跑回測；`entry_z` 對照組除外）
>
> **⚠ 統計基礎已更換。** 舊版所有 p 值以「15 個參數格」為抽樣單位
> （`top_n` 5 × `stop_loss` 3），但這 15 格共用同一份資料、同一段期間、同一批配對
> （`top_n=10` 與 `top_n=20` 共用 10 個配對；三種停損是同一批交易的不同出場規則），
> 觀測高度相關，`ttest_rel` 的獨立性假設不成立（pseudo-replication，有效樣本數 ≈ 1 條
> 回測路徑）。**現以「時間」為抽樣單位重做**：逐日報酬差 + Newey-West HAC
> （`analysis/proposition2_daily_hac.py`）。效果量與舊版幾乎相同（年化 +0.719% vs 舊
> 0.732pp）——偽重複影響的是**推論正當性**，不是效果量本身。
>
> **命題2 成立（相對宣稱）**——DRL vs Z-Score，五種配對底逐日差分 HAC 全部顯著
> （p = 0.0002–0.0177），對落後階 {auto, 63, 126, 252} 穩健，並經循環 block bootstrap
> （L ∈ {21, 126, 252}，5000 次）無母數對照確認（`prop2_block_bootstrap.py`）。
> 增益在**傳統 GICS 配對底上最大**（年化 +1.066%），證明 DRL 的增益與「配對怎麼找到」正交。
>
> 三項機制對照已排除常見的替代解釋（`prop2_exposure_control.py`、`prop2_skip_permutation.py`）：
> 非源自**拉高門檻**（同門檻 2.2 對照下五支仍顯著；門檻管道只複製 4.2–20.9% 且不顯著）、
> 非源自**篩掉爛配對**（SKIP 置換檢定 75 格僅 7 格顯著，隨機期望 3.75 格）、
> 非源自**減少曝險**（DRL 進場次數反而是 Z-Score 的 1.6–1.9 倍）。
> 殘差與「槽位週轉」一致，惟現有資料無法直接驗證，列為後續研究。
>
> **命題1 未獲支持**（`analysis/proposition1_daily_hac.py`，同一套逐日 HAC + block bootstrap）——
> 3×3 消融矩陣的 9 組直接對照（固定排序準則與交易端，唯一變因為分組方法），
> **BH-FDR 校正後無一組達統計顯著**（校正後最小 p = 0.4947；校正前最小 p = 0.0625）。
> 方向上 8/9 組偏向 GICS、1 組偏向 ML，但 9 組比較彼此不獨立（每三組共用同一個 GICS 臂），
> 故不對「方向一致性」施加正式檢定。
>
> ⚠️ **表述須精確**：命題 1 主張「ML 分群能找到更好的配對」，本研究**未能拒絕虛無假設**，
> 故命題 1 不成立。但這**不等於**「ML 顯著較差」——舊版「9 組中 5 組顯著劣於 GICS」的說法
> 建立在已被否定的 n=15 偽重複基礎上，經逐日 HAC 重做後不再成立，不應引用。
>
> 機制證據（獨立於顯著性）：ML 分群會把 11–25% 的股票跨產業配對，且會壓縮候選池
> （K-means 每期 20 個名額只填滿 8.7 個）。粒度掃描進一步顯示決定配對品質的是
> **候選池充足度**而非分群演算法（`analysis/granularity_sweep.py`）。
>
> **方法論觀察**：命題 1 的效果量（年化 −0.11% ~ −0.76%）遠小於命題 2（+0.59% ~ +1.07%），
> 且命題 1 的兩臂為**不同的配對集合**（特異變異大），命題 2 的兩臂**共用同一批配對**
> （差分消掉共同噪音）。形成期的比較天生比交易期的比較更難達到顯著性。
>
> **絕對績效（限制節）**：所有策略的 Deflated Sharpe 皆不通過。以 87 個相異 METHOD 為試驗
> 宇宙（`result.db` 實有 1,659 個回測配置，含試過後封存的負面結果），純靠運氣的期望最高
> Sharpe 門檻 SR0 = 0.433，而全部策略中最高的 SR 僅 0.343。Newey-West 絕對檢定亦無一顯著
> （最佳 p = 0.088）。**命題2 是「A 優於 B」的相對宣稱，不主張策略本身可獲利。**
>
> 詳見 `strategies/config.py` 第 291–300 行註解與 `analysis/` 各模組 docstring。

## 策略清單（`strategies_raw_all`，17 條，皆由 `cluster_formation.py` 中性組裝器動態產生，0-based 索引）

2026-07 中性化重構後，形成期不再是「一策略一支獨立模組」，改由**分群方法 × 群內排序準則**的
3×3 消融矩陣宣告式展開（`strategies/config.py` 第 204 行起）。舊版一策略一模組的獨立策略入口
（`HDBSCAN_Cluster_SSD_DTW.py`、`agglomerative_yF.py`、`agglomerative_FMP.py`、`ssd_basic.py`）
已封存，程式碼保留供復活（見 [archive/README.md](archive/README.md)）。
⚠️ 例外：`ssd_rolling.py`／`DTW_Cointegration_Paper.py` 雖不再是獨立策略入口，但兩者的
`Formation` 類別被新版 `_ranking.py` 動態 import 作為現役排序引擎（每次跑 formation 都會用到），
**並未真正封存**。

| # | 策略 | 分群方法 | 排序準則 | 交易期 | 角色 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 0–8 | `Grid {HDB,AGG,KM}-{SSD,DTW,SDP}` | HDBSCAN／Agglomerative／K-means | SSD／DTW／SSD-DTW-PCA | Z-Score | **命題1** 3×3 分群×排序消融矩陣（ML 分組主力，結果：不支持） |
| 9–11 | `Grid {AGG-SSD,HDB-SDP,KM-SSD} DRL` | 借用 3×3 矩陣內對應格已算好的配對 | 同上 | DRL-THR | **命題2** DRL vs Z-Score（ML 配對底 ×3） |
| 12–14 | `Grid GICS-{SSD,DTW,SDP}` | 真實 GICS 產業（不跑分群） | SSD／DTW／SSD-DTW-PCA | Z-Score | **命題1 對照組**（傳統分組基準） |
| 15–16 | `Grid GICS-{SSD,SDP} DRL` | 借用 GICS 格已算好的配對 | 同上 | DRL-THR | **命題2 對照組**：傳統配對底 + DRL（結果：GICS 底增益最大） |

全部 17 條策略共用單一形成期模組 `strategies.formation.cluster_formation`（`formation_module`），
唯一變因由 `cluster_method` / `ranking_backend` / `filter_mode` 三個參數決定；交易端固定二選一
（`zscore_trading` 或 `drl_threshold_trading`）。實際清單以
`Project/Scripts/python.exe -c "import strategies.config as c; [print(s['name']) for s in c.strategies_raw_all]"`
為準（會隨敏感性分析等環境變數變動）。

切換執行範圍：用環境變數免改檔覆寫（0-based Python 切片，支援逗號複合），如 `STRATEGIES_SLICE="0:9" python run_trading.py` 只跑 3×3 矩陣。

**研究框架沿革**：舊版「#1–#6 研究框架」（因子殘差化、BH-FDR、成本過濾、MST 圖候選、Beta 先驗）
隨其宿主策略（HDBSCAN Cluster、Agglomerative 等）一併封存於 2026-07-24
（`archive/config_archived_strategies.py`，commit `2fb47b6`），程式碼與歷史結果保留可復活。
現行評估層（`analysis/`：regime 分層、break-even 成本表、Deflated Sharpe、`drl_behavior.py` 決策
分解）持續適用於新版 17 策略。

**參數敏感性分析（口試委員要求）**：`config.py` 內建 OFAT 變體產生器。`$env:SENSITIVITY_ALL="1"; python run_formation.py; python run_trading.py` 一次產生 Tier-1 全部變體（`adf_pvalue_threshold`、`pca_n_components`、`beta_feature_weight`、`entry_z`），再 `python -m analysis.sensitivity_report` 看敏感度曲線。

**已封存策略**：完整清單、失敗根因診斷與復活方式見 `archive/config_archived_strategies.py` docstring；歷史回測結果保留於 `results/result.db`。封存分類索引見 [archive/README.md](archive/README.md)。

---

## 專案目錄結構

```text
pairtrade_trainingproject/
├── strategies/
│   ├── config.py                  # 全域參數、3×3 分群×排序 Grid 宣告式展開（17 條現役策略）、敏感性 OFAT 產生器
│   ├── db_utils.py                # SQLite 讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── cluster_formation.py       # ★ 中性組裝器：feature_mode × cluster_method × ranking_backend，17 條現役策略共用
│   │   ├── _clustering.py             # 分群 dispatcher（hdbscan／agglomerative／kmeans；GICS 分組不經此層）
│   │   ├── _ranking.py                # 排序 dispatcher，委派 ssd_rolling／DTW_Cointegration_Paper 的 Formation 類別
│   │   ├── _features.py / _fundamentals.py / _cointegration.py  # 特徵萃取／基本面讀取／ADF+半衰期+Hurst 篩選（各自中性、單向依賴）
│   │   ├── _utils.py                  # 共用統計工具（OLS、ADF、Hurst、_residualize_returns、_bh_fdr_threshold、_cost_viable）
│   │   ├── ssd_rolling.py / DTW_Cointegration_Paper.py  # ⚠️ 非獨立策略入口，但被 `_ranking.py` 動態 import 為現役排序引擎（不可封存）
│   │   ├── HDBSCAN_PCA_Loadings.py / HDBSCAN_Cluster_SSD_DTW.py / agglomerative_yF.py / agglomerative_FMP.py
│   │   │     # 已封存的舊版一策略一模組寫法（2026-07-24 讓位給 cluster_formation.py），程式碼保留供復活
│   │   ├── MST_PartialCorr_Cointegration.py / agglomerative_sec_pit.py / ssd_basic.py  # 已封存策略模組（負面結果，保留供復活）
│   │   ├── ml_pair_quality.py / HDBSCAN_MultiScale.py / HDBSCAN_UMAP.py / ensemble.py  # 已封存（程式碼保留）
│   │   └── __init__.py
│   └── trading/
│       ├── zscore_trading.py          # Z-Score 狀態機（基礎類，三條 Spread 路徑；現役走路徑 B）
│       ├── drl_threshold_trading.py   # DRL 門檻選擇模組（#9–11、#15–16）
│       └── distance_trading.py        # ⚠️ GGR 2006 距離基準——config 端已封存，檔案未搬移
├── analysis/                      # 評估層（讀 result.db，不重跑）
│   ├── regime_cost_dsr_eval.py    # regime 分層 + break-even 成本表 + Deflated Sharpe
│   ├── proposition2_stats.py      # 命題2 配對檢定（DRL vs Z-Score，五種配對底）
│   ├── drl_behavior.py            # 還原 DRL agent 決策，解構增益來源
│   └── sensitivity_report.py      # OFAT 參數敏感性報表
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   ├── sp500_yf_now.py            # yFinance 當日數據更新
│   ├── fundamentals_yfinance.py   # 公司基本面靜態快照（市值、本益比）
│   ├── fetch_fmp_fundamentals.py  # FMP Point-in-Time 基本面 → parquet
│   └── fetch_sec_fundamentals.py  # SEC EDGAR XBRL PIT 基本面（+ Tiingo 原始股價）→ parquet
├── dataset/                       # 資料庫（大檔透過 Git LFS），分 price／fundamental 兩類
│   ├── price/                     # sp500_Tiingo.db（主，DB_PATH 預設）、sp500_yF.db、sp500_Current.db
│   ├── fundamental/               # fundamentals_sp500.db（快照）、sp500_pit_2000_2025_monthly.parquet（PIT，本地產物）、fmp_cache/
│   └── audit_report.csv           # 資料品質審計報告
├── formation_data/
│   └── formation_pairs_sp500_Tiingo.db  # 形成期主合併資料庫（LFS；可用 run_formation.py 完整重建）
├── notebooks/                     # 策略筆記本（Quarto revealjs 投影片；2026-07-27 trim 至論文主軸，見 notebooks/README.md）
│   ├── formation/                 # 形成期 ×6（agglomerative_fundamentals、dtw_paper_fixed、hdbscan_cluster_pca5、kmeans_fundamentals、ssd_dtw_pca_paper_fixed、ssd_rolling）
│   ├── trading/                   # 交易期 ×2（zscore、drl_threshold；distance 已隨其策略封存移除）
│   ├── comparison.ipynb           # 現役策略績效總比較（讀 config + result.db 動態產生）
│   └── main_results.ipynb         # 命題1/2 主軸結果彙整（新增）
├── docs/                          # GitHub Pages：index.html 入口 + slides/（quarto render 產出：comparison + main_results + performance_guide + formation×6 + trading×2）
├── archive/                       # 歷史存檔（分類索引見 archive/README.md）
│   ├── notebooks/ formation/ trading/ scripts/ docs/ h200/
│   └── config_archived_strategies.py  # 已封存策略 config 與完整診斷
├── tools/                         # 輔助工具（從專案根執行）
│   ├── status.py                  #   pt status：資料/形成期/交易期/投影片 狀態總覽 + 建議動作
│   ├── snapshot_run.py            #   全量重跑前歸檔 result.db
│   └── run_drl_variance.py        #   DRL 訓練變異數多輪評估
├── dashboard.py                   # Streamlit 績效比對儀表板
├── run_formation.py               # 形成期主程式（多行程平行）
├── run_trading.py                 # 交易期主程式（多行程平行）
├── pt.bat                         # ★ 統一指令入口（pt status / formation / trading / dashboard / slides…）
├── run.bat / setup.bat            # 一鍵啟動 Dashboard／環境初始化（保留相容）
└── requirements.txt               # Python 套件清單
```

---

## 快速啟動

所有日常操作都走統一入口 `pt.bat`（不帶參數顯示完整指令表）：

```bat
pt setup        # 1. 環境初始化（建立 Project/ 虛擬環境 + 安裝套件）
pt status       # 2. 專案狀態總覽——哪些策略缺形成期/交易期數據、附建議動作
pt all          # 3. 形成期 + 交易期一鍵連跑（或分開 pt formation / pt trading）
pt dashboard    # 4. Streamlit 績效儀表板
pt slides       # 5. 渲染全部 Quarto 投影片 → docs/slides/
```

其他：`pt variance N`（DRL 變異數 N 輪）、`pt snapshot tag`（重跑前歸檔 result.db）、
`pt fetch-price / fetch-fund / fetch-fmp`（資料下載）。

### 傳統呼叫方式（保留相容）

```bash
python run_formation.py    # 形成期：篩選配對 → formation_data/
python run_trading.py      # 交易期：逐日模擬 → results/result.db
```

兩個主程式均支援：
- **智慧續傳**：完成的策略/期數自動跳過（以 SQLite 資料庫為唯一真相來源；`FORCE_RERUN=True` 可強制全部重算）
- **多行程平行**：每個策略的滾動期獨立平行計算（`spawn` context，避免 CUDA fork 污染）
- **網格搜尋**：自動搜尋 Top N / Stop Loss / MSR 等參數組合
- **免改檔範圍覆寫**：環境變數 `STRATEGIES_SLICE`（如 `"5:7"`、`"0:5,8:12"`）

### 3. 查看結果

```bat
run.bat
```

啟動 Streamlit Dashboard（預設 http://localhost:8501），提供多維篩選、權益曲線對比與逐期 Trade Visualizer。

---

## 核心回測參數（`config.py`）

| 參數 | 值 | 說明 |
| :--- | :--- | :--- |
| `FORMATION_WINDOW` | 252 天 | 形成期長度 |
| `FORWARD_DAYS` | 126 天 | 交易期長度 |
| `rolling_step` | 21 天 | 滾動步長 |
| `entry_z` | 2.0 | Z-Score 進場閾值 |
| `exit_z` | 0.0 | Z-Score 出場閾值（回歸均值） |
| `max_holding_days` | 0（停用） | 時間停損；預設 0 維持既有策略行為（舊文件誤植 30，引擎從未使用該值） |
| `fee_rate` | 0.0029 | 手續費率（單邊；Do & Faff 2012 美股 pairs ~30 bps/邊） |
| `slippage_rate` | 0.0 | 滑點率（已併入 fee_rate；往返成本 = 0.29%×2 = 0.58%） |
| `RF_ANNUAL` | 0.02 | 無風險利率（超額報酬計息） |
| `INITIAL_CAPITAL` | 10,000 | 每配對初始資金 |

---

## 資料範圍與已知限制

**回測期間** 2000-01 至 2025-12（`BACKTEST_START` / `BACKTEST_END` 可用環境變數覆寫）。
成分股採**時點（point-in-time）**認定：`run_formation.py` 每期依 `index_memberships`
的 `start_date` / `end_date`，只保留該形成期結束日**當下真實在 S&P 500 內**的標的，
並納入已下市股，以降低存活者偏誤。

**存活者偏誤未完全消除（量化揭露）**：成分股名單 843 檔，但價格表僅 747 檔，
114 檔完全無價格資料，且缺失與下市狀態相關：

| | 有價格資料 | 無價格資料 |
| :--- | ---: | ---: |
| 未下市 | 608 | 65 |
| **已下市** | **121** | **49** |

即 170 檔已下市成分股中有 **49 檔（29%）無價格資料**；無價格者的已下市比率為 **43%**，
有價格者僅 **17%**——缺失並非隨機，死亡的公司缺資料的機率約為存活者的 2.5 倍。
（另：2000–2026 相異 S&P 500 成分股一般認定在 1,000–1,100 檔，843 本身亦偏低。）

**偏誤方向（重要）**：
- 對**絕對績效**：樣本偏向倖存者 → 報酬被**高估**。本研究的絕對結論為「績效不顯著、
  DSR 全數不通過」，在被高估的樣本上仍不顯著，故該結論**偏保守**。
- 對**相對宣稱**（命題 1／命題 2）：各比較臂共用同一個殘缺標的池，偏誤同向抵銷，
  相對比較**基本不受影響**。

**成本假設的年代**：`fee_rate` 0.29%/邊取自 Do & Faff (2012)，其估計期為 1962–2009，
本研究套用於 2000–2025。美股摩擦成本長期下降，故此假設偏保守。
Break-even 分析顯示各策略往返成本餘裕僅 6.5–12.8 bps（`analysis/regime_cost_dsr_eval.py`），
對成本假設高度敏感。

**Sharpe 口徑**：`Sharpe_Raw` 未扣無風險利率（`dashboard.py` 之定義）；rf 超額另見
`Excess_Ret_RF` 欄（扣 `RF_ANNUAL × 平均利用率`）。命題 2 的逐日差分檢定中 rf 於兩臂
相減時對消，不受此口徑影響。

---

## 研究框架 #1–#6 主要結論（已由 2026-07-28 命題檢定取代，見上方提要框）

> ⚠️ 本節為舊版 12+2 策略（Agglomerative 基本面、HDBSCAN Cluster+Resid 等，已於 2026-07-24
> 全數封存）時期的結論，與現行 17 策略 Grid 架構的正式命題檢定結果不一致，僅保留作歷史記錄。
> **目前有效的命題1／命題2結論見本文件開頭的提要框**及 `strategies/config.py` 第 291–300 行。

- **（歷史）最強誠實策略 = Agglomerative 基本面（yF/FMP）**：Top1 年化（動用資金）約 **+3.3~3.4%、Profit Factor ~1.2**，且在多空／高低波動各 regime 皆為正、空頭略優（逆週期分散報酬源）。
- **（歷史）#1 因子殘差化**：配強距離排序器（SSD-DTW）有效——HDBSCAN Cluster+Resid 最佳年化較原版近乎翻倍；#2 BH-FDR／#3 成本過濾能降強制平倉率，但在弱排序器路徑上救不了 Sharpe。
- **（歷史）#4 MST 偏相關圖候選** 與 **#5 Beta 風險先驗**：皆為**負面結果**——過度稀疏或加權主導反而稀釋既有的良好表徵。
- **（歷史）#6 穩健性評估**：所有策略往返 break-even 成本僅 ~0.6–0.67%（餘裕薄，符合 Do & Faff）；校正 best-of-15 選擇偏誤後 **Deflated Sharpe 全部 < 0.95**。

> 具體數值以 `results/result.db` 為準（每次重跑會更新）。完整比較見 [notebooks/comparison.ipynb](notebooks/comparison.ipynb)（讀 config + result.db 動態產生，投影片版 `docs/slides/comparison.html`）；穩健性三表見 `python -m analysis.regime_cost_dsr_eval`。

---

## 策略說明文件

- 形成期排序引擎（公式、參數、文獻標註，6 本）：[notebooks/formation/](notebooks/formation/)
- 交易期策略（Z-Score 狀態機、DRL-THR v4，2 本）：[notebooks/trading/](notebooks/trading/)
- 命題1/2 主軸結果彙整：[notebooks/main_results.ipynb](notebooks/main_results.ipynb)
- 績效總比較：[notebooks/comparison.ipynb](notebooks/comparison.ipynb)
- 投影片入口（quarto render 產出）：[docs/index.html](docs/index.html) → `docs/slides/`
- 詳細開發指南：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- 已封存策略完整診斷：[archive/config_archived_strategies.py](archive/config_archived_strategies.py)
