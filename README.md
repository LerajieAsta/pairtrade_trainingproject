# S&P 500 Pairs Trading 量化回測平台

本專案針對 S&P 500 成分股實作配對交易 (Pairs Trading) 滾動回測系統，支援多種形成期策略（含機器學習配對）、深度強化學習交易期模組與互動式績效視覺化儀表板。

---

## 策略清單（`strategies_raw_all`，10 個交易策略 + 2 個 formation-only 條目）

| # | 策略 | 形成期模組 | 交易期 | 角色 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | SSD Rolling | `ssd_rolling.py` | Z-Score | SSD 家族基準 |
| 2 | DTW Paper Fixed (DTW) | 借用配對 | Z-Score | 誠實 DTW 基準 |
| 3 | SSD-DTW-PCA Paper Fixed | 借用配對 | Z-Score | 全組最佳誠實基準（Sharpe 0.56） |
| 4 | HDBSCAN Cluster SSD-DTW-PCA | `HDBSCAN_Cluster_SSD_DTW.py` | Z-Score | 分組消融：HDBSCAN vs GICS |
| 5 | SSD Rolling DRL THR | 借用 #1 配對 | DRL | DRL 疊加對照組 |
| 6 | HDBSCAN Cluster SSD-DTW-PCA DRL THR | 借用 #4 配對 | DRL | DRL 疊加實驗組 |
| 7 | HDBSCAN Cluster SSD-DTW-PCA PCA5 | `HDBSCAN_Cluster_SSD_DTW.py`（5 維 PCA） | Z-Score | 維度詛咒修復版 |
| 8 | HDBSCAN Cluster SSD-DTW-PCA PCA5 DRL THR | 借用 #7 配對 | DRL | 全組 DRL 疊加最高 Sharpe（0.46） |
| 9 | Agglomerative Fundamentals | `agglomerative_fundamentals.py` | Z-Score | 分組消融：價格 PCA ⊕ 公司基本面 |
| 10 | Agglomerative Fundamentals DRL THR | 借用 #9 配對 | DRL | 全組最佳年化報酬（2.89%） |

> **formation-only 條目（2026-07-05 新增）**：`DTW Paper (DTW)` 與 `DTW Paper (SSD-DTW-PCA)` 原版以 `formation_only: True` 旗標回歸現役清單——`run_formation.py` 計算它們的配對供 #2/#3 借用，`run_trading.py` 跳過不回測（原版交易端為座標 artifact，維持封存）。全部形成期配對因此可在任何機器本地重算，不依賴 repo LFS。

切換執行範圍：修改 `strategies/config.py` 中的 `strategies_raw`，或用環境變數免改檔覆寫（0-based Python 切片，如 `STRATEGIES_SLICE="4:6" python run_trading.py` 只跑 #5–#6）。

**已封存策略（20 個 config 條目 + 2 個從未賦予策略身分的孤兒模組）**：完整清單、失敗根因診斷與復活方式見 `archive/config_archived_strategies.py` docstring；歷史回測結果保留於 `results/result.db`。策略邏輯與封存摘要見 [notebooks/formation.ipynb](notebooks/formation.ipynb) 第五節。

---

## 專案目錄結構

```text
pairtrade_trainingproject/
├── strategies/
│   ├── config.py                  # 全域回測參數與現役策略清單（10 交易 + 2 formation-only）
│   ├── db_utils.py                # SQLite 讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── ssd_rolling.py             # SSD Rolling 形成期（#1；亦被 #9 複用排序邏輯）
│   │   ├── DTW_Cointegration_Paper.py # DTW 形成期（#2/#3 借用配對；#4/#7 複用排序邏輯）
│   │   ├── HDBSCAN_PCA_Loadings.py    # 報酬 PCA 因子載荷特徵萃取（被 #4/#7/#9 組合複用）
│   │   ├── HDBSCAN_Cluster_SSD_DTW.py # HDBSCAN 聚類 + SSD-DTW-PCA 排序組合模組（#4/#7）
│   │   ├── agglomerative_fundamentals.py # Agglomerative 聚類（價格⊕基本面）+ SSD 排序（#9）
│   │   ├── ml_pair_quality.py         # 監督式學習排序（已封存，程式碼保留）
│   │   ├── HDBSCAN_MultiScale.py / HDBSCAN_UMAP.py / ensemble.py / ssd_basic.py  # 已封存
│   │   └── _utils.py                  # 共用統計工具（OLS、ADF、Hurst）
│   └── trading/
│       ├── zscore_trading.py          # Z-Score 狀態機（基礎類，#1–#4、#7、#9 使用）
│       └── drl_threshold_trading.py   # DRL 門檻選擇式 v4（#5、#6、#8、#10 使用）
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   ├── sp500_yf_now.py            # yFinance 當日數據更新
│   └── fundamentals_yfinance.py   # 公司基本面快照下載（市值、本益比）
├── dataset/
│   ├── sp500_Tiingo.db             # 主要資料庫（LFS 追蹤，`DB_PATH` 預設指向此）
│   ├── sp500_yF.db / sp500_Current.db  # 備用資料庫（LFS 追蹤）
│   ├── fundamentals_sp500.db      # 公司基本面快照（LFS 追蹤；可用 fetch/fundamentals_yfinance.py 重建）
│   └── audit_report.csv           # 資料品質審計報告
├── formation_data/
│   └── formation_pairs_sp500_Tiingo.db  # 形成期主合併資料庫（LFS 追蹤；可用 run_formation.py 完整重建）
├── notebooks/
│   ├── formation.ipynb            # 現役 10 策略形成期邏輯完整說明 + 論文引述
│   └── trading.ipynb              # 交易期邏輯說明 + DRL-THR 架構 + 績效總比較
├── archive/                       # 歷史存檔（已封存策略 config、非現役交易模組、舊版 notebook、docs 簡報）
│   └── trading/                   # 已封存/孤兒交易模組（drl_lstm×2、drl_fqi、kalman、pure_dtw）
├── dashboard.py                   # Streamlit 績效比對儀表板
├── run_formation.py               # 形成期主程式（多行程平行）
├── run_trading.py                 # 交易期主程式（多行程平行）
├── run.bat                        # 一鍵啟動 Dashboard
├── setup.bat / setup.sh           # 環境初始化
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
| `fee_rate` | 0.001 | 手續費率（單程） |
| `slippage_rate` | 0.001 | 滑點率（單程） |
| `INITIAL_CAPITAL` | 10,000 | 每配對初始資金 |

---

## 現役策略績效摘要（`results/result.db`，2026-07-05）

| 指標 | 策略 | 數值 |
| :--- | :--- | :---: |
| 最佳 Sharpe | SSD-DTW-PCA (Paper-Fixed) | 0.558 |
| 最佳年化報酬 | Agglomerative (Fundamentals-DRL-THR) | 2.89% |
| 唯一同時優於 SSD 基準（Sharpe + 年化）的 ML 配對法 | Agglomerative (Fundamentals) | Sharpe 0.35 / 年化 2.70%（⚠️ 基本面前視偏誤限制） |

> 2026-07-05 因 repo LFS 額度用罄，形成期配對與基本面快照已於本機完整重建：Z-Score 系列結果與前版一致（確定性計算復現）；DRL-THR 系列因重新訓練數值有波動（如最佳年化 3.03% → 2.89%），排名結構與定性結論不變。

完整比較表與逐項解讀見 [notebooks/trading.ipynb](notebooks/trading.ipynb) 第五節「現役策略績效總比較」。

---

## 策略說明文件

- 形成期邏輯與論文引述：[notebooks/formation.ipynb](notebooks/formation.ipynb)
- 交易期邏輯、DRL-THR 架構與績效總比較：[notebooks/trading.ipynb](notebooks/trading.ipynb)
- 詳細開發指南：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- 已封存策略完整診斷：[archive/config_archived_strategies.py](archive/config_archived_strategies.py)
