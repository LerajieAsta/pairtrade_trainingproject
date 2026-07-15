# S&P 500 Pairs Trading 量化回測平台

本專案針對 S&P 500 成分股實作配對交易 (Pairs Trading) 滾動回測系統，支援多種形成期策略（含機器學習配對）、深度強化學習交易期模組與互動式績效視覺化儀表板。

---

本研究兩大命題：**命題1（形成期）** 機器學習分組能找到比傳統距離／共整合法更高品質的配對；
**命題2（交易期）** 深度學習（DRL）能比傳統 Z-Score 有更好的交易績效。

## 策略清單（`strategies_raw_all`，12 個交易策略 + 2 個 formation-only 條目，0-based 索引）

| # | 策略 | 形成期模組 | 交易期 | 角色 |
| :---: | :--- | :--- | :--- | :--- |
| 0 | SSD Basic | `ssd_basic.py` | Z-Score | 傳統距離法基礎原型（命題1 基準） |
| 1 | SSD Rolling | `ssd_rolling.py` | Z-Score | SSD 距離法基準（命題1 基準） |
| 2 | SSD (Distance) | `ssd_rolling.py`（借用 #1） | Distance | 回歸 vs 距離交易對照（GGR 2006） |
| 3 | DTW | 借用 DTW Paper 配對 | Z-Score（路徑 B） | DTW／共整合基準（命題1 基準） |
| 4 | SSD-DTW-PCA | 借用配對 | Z-Score（路徑 B） | 距離排序基準（命題1 基準） |
| 5 | SSD (DRL) | 借用 #1 配對 | DRL-THR | **命題2**：DRL vs Z-Score（SSD 對照） |
| 6 | HDBSCAN | `HDBSCAN_Cluster_SSD_DTW.py` | Z-Score | **命題1**：機器學習分組（HDBSCAN vs GICS） |
| 7 | HDBSCAN (殘差) | `HDBSCAN_Cluster_SSD_DTW.py` | Z-Score | **命題1**：+因子殘差化（命題1 主力） |
| 8 | Agglomerative (yF) | `agglomerative_yF.py` | Z-Score | **命題1**：ML 基本面分組（yF 快照，主力） |
| 9 | Agglomerative (yF·DRL) | 借用 #8 配對 | DRL-THR | **命題2**：DRL vs Z-Score（Agg-yF 對照） |
| 10 | Agglomerative (FMP) | `agglomerative_FMP.py` | Z-Score | **命題1**：ML 基本面分組（FMP PIT，主力） |
| 11 | Agglomerative (FMP·DRL) | 借用 #10 配對 | DRL-THR | **命題2**：DRL vs Z-Score（Agg-FMP 對照） |
| 12 | DTW Paper (DTW) | `DTW_Cointegration_Paper.py` | formation-only | 產生 #3 借用的原版配對 |
| 13 | DTW Paper (SSD-DTW-PCA) | `DTW_Cointegration_Paper.py` | formation-only | 產生 #4 借用的原版配對 |

> **已封存的負面結果（2026-07-15）**：研究框架消融策略 ResidFDR（#1+#2+#3）、MST 偏相關圖（#4）、
> SEC-PIT Beta（#5）經全時段回測皆為負面／劣於骨幹，已移至 `archive/config_archived_strategies.py`
> 與 `archive/notebooks/negative_results/`（程式碼與 result.db 歷史數據保留，可復活）。這些負面結果
> 仍有學術價值：證明「框架的增益是特定的——好的基礎表徵已捕捉結構，激進修剪／堆疊特徵常是稀釋」。

> **formation-only 條目**：`DTW Paper (DTW)` / `(SSD-DTW-PCA)` 以 `formation_only: True` 旗標僅產生配對供 #3/#4 借用，`run_trading.py` 跳過回測（原版交易端為座標 artifact，維持封存）。全部形成期配對因此可在任何機器以 `run_formation.py` 本地重算，不依賴 repo LFS。

切換執行範圍：用環境變數免改檔覆寫（0-based Python 切片，支援逗號複合），如 `STRATEGIES_SLICE="6:9" python run_trading.py` 只跑 #6–#8。

**研究框架 #1–#6（畢業論文次步）**：#1 因子殘差化（**有效**，保留於現役 #7 HDBSCAN Cluster Resid）；#2 BH-FDR、#3 成本過濾、#4 MST 圖候選、#5 Beta 先驗（**負面**，已封存）；#6 評估層（`analysis/`：regime 分層、break-even 成本表、Deflated Sharpe，續留現役）。關鍵結論見下方績效摘要與 `analysis/`。

**參數敏感性分析（口試委員要求）**：`config.py` 內建 OFAT 變體產生器。`$env:SENSITIVITY_ALL="1"; python run_formation.py; python run_trading.py` 一次產生 Tier-1 全部變體（`adf_pvalue_threshold`、`pca_n_components`、`beta_feature_weight`、`entry_z`），再 `python -m analysis.sensitivity_report` 看敏感度曲線。

**已封存策略**：完整清單、失敗根因診斷與復活方式見 `archive/config_archived_strategies.py` docstring；歷史回測結果保留於 `results/result.db`。封存分類索引見 [archive/README.md](archive/README.md)。

---

## 專案目錄結構

```text
pairtrade_trainingproject/
├── strategies/
│   ├── config.py                  # 全域參數、現役策略清單（12 交易 + 2 formation-only）、敏感性 OFAT 產生器
│   ├── db_utils.py                # SQLite 讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── _utils.py                  # 共用統計工具（OLS、ADF、Hurst、_residualize_returns、_bh_fdr_threshold、_cost_viable）
│   │   ├── ssd_basic.py / ssd_rolling.py  # SSD 家族（#0/#1；ssd_rolling 亦被 #2/#5 複用）
│   │   ├── DTW_Cointegration_Paper.py # DTW 形成期（#3/#4 借用配對；#6/#7 複用排序邏輯）
│   │   ├── HDBSCAN_PCA_Loadings.py    # 報酬 PCA 因子載荷特徵萃取（被 #6/#7 複用）
│   │   ├── HDBSCAN_Cluster_SSD_DTW.py # HDBSCAN 聚類 + SSD-DTW-PCA 排序組合（#6/#7，命題1 主力）
│   │   ├── agglomerative_yF.py        # Agglomerative（價格 PCA ⊕ 基本面靜態快照）+ SSD 排序（#8/#9）
│   │   ├── agglomerative_FMP.py       # 同上，改用 FMP Point-in-Time parquet（#10/#11，命題1 主力）
│   │   ├── MST_PartialCorr_Cointegration.py / agglomerative_sec_pit.py  # 已封存策略模組（負面結果，保留供復活）
│   │   ├── ml_pair_quality.py / HDBSCAN_MultiScale.py / HDBSCAN_UMAP.py / ensemble.py  # 已封存（程式碼保留）
│   │   └── __init__.py
│   └── trading/
│       ├── zscore_trading.py          # Z-Score 狀態機（基礎類，三條 Spread 路徑；現役走路徑 B）
│       ├── distance_trading.py        # 距離基準交易（GGR 2006，#2）
│       └── drl_threshold_trading.py   # DRL 門檻選擇模組（#5、#9、#11）
├── analysis/                      # 評估層（讀 result.db，不重跑）
│   ├── regime_cost_dsr_eval.py    # 研究框架 #6：regime 分層 + break-even 成本表 + Deflated Sharpe
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
├── notebooks/                     # 策略筆記本（一策略一本；Quarto revealjs 投影片，見 notebooks/README.md）
│   ├── formation/                 # 形成期策略 ×7（ssd_basic/rolling、dtw、ssd-dtw-pca、hdbscan×2、agglomerative）
│   ├── trading/                   # 交易期策略 ×3（zscore、distance、drl_threshold）
│   └── comparison.ipynb           # 現役策略績效總比較（讀 config + result.db 動態產生）
├── docs/                          # GitHub Pages：index.html 入口 + slides/（quarto render 產出投影片）
├── archive/                       # 歷史存檔（分類索引見 archive/README.md）
│   ├── notebooks/ formation/ trading/ scripts/ docs/ h200/
│   └── config_archived_strategies.py  # 已封存策略 config 與完整診斷
├── dashboard.py                   # Streamlit 績效比對儀表板
├── snapshot_run.py                # 全量重跑前歸檔 result.db 工具
├── run_formation.py               # 形成期主程式（多行程平行）
├── run_trading.py                 # 交易期主程式（多行程平行）
├── run.bat / setup.bat            # 一鍵啟動 Dashboard／環境初始化（Windows）
└── requirements.txt               # Python 套件清單
```

---

## 快速啟動

### 1. 環境初始化

```bat
setup.bat
```

自動建立 `Project/` 虛擬環境並安裝 `requirements.txt` 所有套件。

### 2. 執行回測

```bash
# 步驟一：形成期（篩選配對，寫入 formation_data/）
python run_formation.py

# 步驟二：交易期（逐日模擬，輸出 results/ 與 dataset/audit_report.csv）
python run_trading.py
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
| `max_holding_days` | 30 天 | 最長持倉天數 |
| `fee_rate` | 0.0029 | 手續費率（單邊；Do & Faff 2012 美股 pairs ~30 bps/邊） |
| `slippage_rate` | 0.0 | 滑點率（已併入 fee_rate；往返成本 = 0.29%×2 = 0.58%） |
| `RF_ANNUAL` | 0.02 | 無風險利率（超額報酬計息） |
| `INITIAL_CAPITAL` | 10,000 | 每配對初始資金 |

---

## 研究框架 #1–#6 主要結論（誠實會計，扣 0.58% 往返成本）

- **最強誠實策略 = Agglomerative 基本面（yF/FMP）**：Top1 年化（動用資金）約 **+3.3~3.4%、Profit Factor ~1.2**，且在多空／高低波動各 regime 皆為正、空頭略優（逆週期分散報酬源）。
- **#1 因子殘差化**：配強距離排序器（SSD-DTW）有效——HDBSCAN Cluster+Resid 最佳年化較原版近乎翻倍；#2 BH-FDR／#3 成本過濾能降強制平倉率，但在弱排序器路徑上救不了 Sharpe。
- **#4 MST 偏相關圖候選** 與 **#5 Beta 風險先驗**：皆為**負面結果**——過度稀疏或加權主導反而稀釋既有的良好表徵（可寫的消融證據，強化「聚類優勢是特定的」論點）。
- **#6 穩健性評估**：所有策略往返 break-even 成本僅 ~0.6–0.67%（餘裕薄，符合 Do & Faff）；校正 best-of-15 選擇偏誤後 **Deflated Sharpe 全部 < 0.95**（無單一配置可宣稱統計顯著為正）——這是誠實研究應有的結論。

> 具體數值以 `results/result.db` 為準（每次重跑會更新）。完整比較見 [notebooks/comparison.ipynb](notebooks/comparison.ipynb)（讀 config + result.db 動態產生，投影片版 `docs/slides/comparison.html`）；穩健性三表見 `python -m analysis.regime_cost_dsr_eval`。

---

## 策略說明文件

- 形成期策略（一策略一本：公式、參數、文獻標註）：[notebooks/formation/](notebooks/formation/)
- 交易期策略（Z-Score 狀態機、DRL-THR v4）：[notebooks/trading/](notebooks/trading/)
- 績效總比較：[notebooks/comparison.ipynb](notebooks/comparison.ipynb)
- 投影片入口（quarto render 產出）：[docs/index.html](docs/index.html) → `docs/slides/`
- 詳細開發指南：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- 已封存策略完整診斷：[archive/config_archived_strategies.py](archive/config_archived_strategies.py)
