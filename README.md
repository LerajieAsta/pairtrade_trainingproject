# S&P 500 Pairs Trading 量化回測平台

本專案針對 S&P 500 成分股實作配對交易 (Pairs Trading) 滾動回測系統，支援多種形成期策略、深度強化學習交易期模組與互動式績效視覺化儀表板。

---

## 策略清單（`strategies_raw_all`，共 13 個）

| # | 策略 | 形成期模組 | 交易期 |
| :---: | :--- | :--- | :--- |
| 1 | SSD Basic | `ssd_basic.py` | Z-Score |
| 2 | SSD Rolling | `ssd_rolling.py` | Z-Score |
| 3 | DTW Paper (DTW) | `DTW_Cointegration_Paper.py` | Z-Score |
| 4 | DTW Paper (SSD-DTW-PCA) | `DTW_Cointegration_Paper.py` | Z-Score |
| 5 | HDBSCAN MultiScale | `HDBSCAN_MultiScale.py` | Z-Score |
| 6 | HDBSCAN MultiScale PCA-UMAP | `HDBSCAN_MultiScale.py` | Z-Score |
| 7 | HDBSCAN UMAP | `HDBSCAN_UMAP.py` | Z-Score |
| 8 | HDBSCAN UMAP PCA-UMAP | `HDBSCAN_UMAP.py` | Z-Score |
| 9 | Ensemble HDBSCAN | `ensemble.py` | Z-Score |
| 10 | Ensemble SSD-DTW | `ensemble.py` | Z-Score |
| 11 | SSD Rolling DRL | `ssd_rolling.py` | DRL |
| 12 | HDBSCAN UMAP DRL | `HDBSCAN_UMAP.py` | DRL |
| 13 | HDBSCAN MultiScale DRL | `HDBSCAN_MultiScale.py` | DRL |

切換執行範圍：修改 `strategies/config.py` 中的 `strategies_raw`，例如 `strategies_raw_all[-3:]` 只跑後 3 個，`strategies_raw_all` 跑全部 13 個。

---

## 專案目錄結構

```text
Papper/
├── strategies/
│   ├── config.py                  # 全域回測參數與策略清單
│   ├── db_utils.py                # SQLite 讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── ssd_basic.py           # SSD Basic 形成期
│   │   ├── ssd_rolling.py         # SSD Rolling 形成期
│   │   ├── DTW_Cointegration_Paper.py  # DTW 形成期
│   │   ├── HDBSCAN_UMAP.py        # HDBSCAN UMAP 形成期
│   │   ├── HDBSCAN_MultiScale.py  # HDBSCAN MultiScale 形成期（自適應子期間）
│   │   ├── ensemble.py            # Ensemble 形成期（取交集/聯集）
│   │   └── _utils.py              # 共用統計工具（OLS、ADF、Hurst）
│   └── trading/
│       ├── zscore_trading.py      # Z-Score 狀態機（基礎類）
│       ├── drl_lstm_trading.py    # DRL LSTM-DQN 交易（目前啟用）
│       └── pure_dtw_trading.py    # 純 DTW 交叉進場（保留備用）
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   └── sp500_yf_now.py            # yFinance 當日數據更新
├── dataset/
│   ├── sp500_yF.db                # 主要資料庫（yFinance，LFS 追蹤）
│   ├── sp500_Tiingo.db            # Tiingo 資料庫（LFS 追蹤）
│   ├── sp500_Current.db           # 現行成分股資料庫
│   └── audit_report.csv           # 資料品質審計報告
├── formation_data/
│   └── formation_pairs_sp500_yF.db  # 形成期主合併資料庫（LFS 追蹤）
├── notebooks/
│   ├── formation.ipynb            # 所有形成期策略邏輯完整說明
│   └── trading.ipynb              # 所有交易期策略邏輯完整說明
├── archive/                       # 歷史存檔（舊版 notebook、docs 簡報等）
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
- **智慧續傳**：完成的期數自動跳過（JSON 完成標記 + SQLite 期數雙重驗證）
- **多行程平行**：每個策略的滾動期獨立平行計算
- **網格搜尋**：自動搜尋 Top N / Stop Loss / MSR 等參數組合

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

## 策略說明文件

- 形成期邏輯：[notebooks/formation.ipynb](notebooks/formation.ipynb)
- 交易期邏輯：[notebooks/trading.ipynb](notebooks/trading.ipynb)
- 詳細開發指南：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)
