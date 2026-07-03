# Pairs Trading 專案開發指南 (PROJECT_GUIDE)

---

## 1. 專案目錄結構

```text
Papper/
├── strategies/
│   ├── config.py                  # 策略清單、網格參數、全域回測設定
│   ├── db_utils.py                # SQLite 合併、讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理（MSR 產業上限）
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── _utils.py              # 共用統計工具（_ols、_adf_stat、_compute_hurst）
│   │   ├── ssd_basic.py           # SSD Basic：累積回報比值距離
│   │   ├── ssd_rolling.py         # SSD Rolling：Z-Score 標準化 log-price 空間
│   │   ├── DTW_Cointegration_Paper.py  # DTW + ADF 雙重篩選
│   │   ├── HDBSCAN_UMAP.py        # HDBSCAN UMAP：10 維特徵 Quality Score
│   │   ├── HDBSCAN_MultiScale.py  # HDBSCAN MultiScale：形成期內部 n_splits 等分子期間
│   │   ├── HDBSCAN_PCA_Loadings.py # HDBSCAN PCA-Loadings：報酬 PCA 因子載荷特徵（消融實驗）
│   │   └── ensemble.py            # Ensemble：Tier-1 交集 + Tier-2 聯集補足
│   └── trading/
│       ├── zscore_trading.py      # Z-Score 狀態機（基礎類，三條 Spread 路徑）
│       ├── drl_lstm_trading.py    # DRL LSTM-DQN（目前三個策略使用）
│       └── pure_dtw_trading.py    # 純 DTW 交叉進場（程式碼保留，目前停用）
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   └── sp500_yf_now.py            # yFinance 當日數據更新
├── dataset/                       # 資料庫（大檔案透過 Git LFS 追蹤）
│   ├── sp500_yF.db                # 主要資料庫（目前 DB_PATH 指向此）
│   ├── sp500_Tiingo.db            # Tiingo 備用資料庫
│   ├── sp500_Current.db           # 現行成分股資料庫
│   └── audit_report.csv           # 交易期資料品質審計報告
├── formation_data/
│   └── formation_pairs_sp500_yF.db  # 形成期主合併資料庫（LFS 追蹤）
├── notebooks/
│   ├── formation.ipynb            # 五大形成期策略完整邏輯說明
│   └── trading.ipynb              # 三大交易期模組完整邏輯說明
├── archive/
│   ├── 114/                       # 114 學年早期 notebook
│   ├── 11505/                     # 115 年 5 月 notebook
│   ├── 11506/                     # 115 年 6 月 notebook（含早期 formation 模組）
│   ├── docs/                      # 歷次學術 HTML 簡報（1150325–1150527）
│   └── Ref_CODE/                  # 原始研究 notebook 與 dashboard 截圖
├── dashboard.py                   # Streamlit 績效比對儀表板
├── run_formation.py               # 形成期主程式
├── run_trading.py                 # 交易期主程式
├── run.bat                        # 一鍵啟動 Dashboard
├── setup.bat / setup.sh           # 環境初始化腳本
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

- 讀取 `dataset/sp500_yF.db`（或 config 指定資料庫）
- 多行程平行執行每個策略的每個滾動期形成期計算
- 結果寫入 `formation_data/formation_pairs_{db_basename}.db`
- 智慧續傳：JSON 完成標記 + SQLite 期數計數雙重驗證，已完成的期數自動跳過
- `FORCE_RERUN = False`（config.py）：正常模式，不強制重跑

### run_trading.py 細節

- 讀取 formation_data/ 的配對清單 + dataset/ 的價格資料
- 多行程平行執行交易期逐日模擬
- 輸出：`results/yFinance/` 下的 Trade Log CSV + `dataset/audit_report.csv`
- 網格搜尋：Top N / Stop Loss / MSR 等參數組合

---

## 3. 策略清單（`config.py`）

`strategies_raw_all` 為現役策略池（共 7 個），`strategies_raw` 決定實際執行範圍：

```python
strategies_raw = strategies_raw_all[-2:]   # 只跑 DRL-FQI 兩策略
# strategies_raw = strategies_raw_all      # 跑全部 7 個
```

| # | 策略名稱 | 形成期 | 交易期 | 角色 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | SSD Rolling | `ssd_rolling.py` | `zscore_trading.py` | SSD 家族基準 |
| 2 | HDBSCAN PCA-Loadings | `HDBSCAN_PCA_Loadings.py` | `zscore_trading.py` | 命題 1 主線形成法 |
| 3 | DTW Paper Fixed (DTW) | 借用 DTW Paper 配對 | `zscore_trading.py`（路徑 B） | 誠實 DTW 基準 |
| 4 | SSD-DTW-PCA Paper Fixed | 借用配對 | `zscore_trading.py`（路徑 B） | 最佳誠實基準 |
| 5 | HDBSCAN Cluster SSD-DTW-PCA | `HDBSCAN_Cluster_SSD_DTW.py` | `zscore_trading.py` | 分組消融（vs #4） |
| 6 | HDBSCAN PCA-Loadings DRL FQI | 借用 #2 配對 | `drl_fqi_trading.py` | 命題 2 主線 |
| 7 | SSD Rolling DRL FQI | 借用 #1 配對 | `drl_fqi_trading.py` | 命題 2 對照 |

**已封存策略（16 個）**：2026-07-03 移至 `archive/config_archived_strategies.py`
（SSD Basic、DTW 原版×2〔座標 artifact〕、HDBSCAN stats10 系×4、Ensemble×2、
DRLv1×3、Kalman×2、CONV×2），封存理由見該檔 docstring，歷史回測結果保留於
`results/result.db`。⚠️ DTW 原版的形成期配對仍被 #3/#4 借用，formation DB 資料列不可刪。

---

## 4. 形成期策略說明

### Spread 空間與 Formation_Params

| 策略 | Spread 空間 | 關鍵 Formation_Params |
| :--- | :--- | :--- |
| SSD Basic | 累積回報比值 $P_A/P_{A0} - P_B/P_{B0}$ | `First_Price_A/B`, `Hedge_Ratio=1.0` |
| SSD Rolling | Z-Score 標準化 log-price | `Log_Mean_A/B`, `Log_Std_A/B`, `Spread_Mean/Std` |
| DTW | OLS 殘差 log-price | `OLS_Alpha`, `Hedge_Ratio`, `Spread_Mean/Std` |
| HDBSCAN UMAP | OLS 殘差 log-price | `OLS_Alpha`, `Hedge_Ratio`, `Quality_Score` |
| HDBSCAN MultiScale | OLS 殘差 log-price | `OLS_Alpha`, `Hedge_Ratio`, `Corr_Mean`, `Coint_Pass_Rate` |

### 過濾門檻（全策略共用）

| 指標 | 門檻 |
| :--- | :--- |
| OU 半衰期 | 1 ≤ halflife ≤ 60 天 |
| Hurst 指數 | < 0.50（均值回歸） |
| ADF p-value | ≤ 0.05（DTW/HDBSCAN）或 ≤ 0.01（SSD） |
| 零穿越次數 | ≥ 5 次 |

### HDBSCAN MultiScale 設計說明

- **子期間**：`_make_relative_sub_periods(form_start, form_end, n_splits=4)` 將形成期等分為 4 段（Q1–Q4），確保每個 252 天滾動窗口都有 4 段可計算，不依賴固定日曆邊界
- **熊市判斷**：`_build_dynamic_bear_mask()` 以等權 log-price 指數低於 60 日 MA 動態標記熊市日，無需硬編碼歷史日期
- **Coverage 評分**：`s_coverage = n_valid_periods / n_splits`，分母與實際子期間數一致，滿分可達

---

## 5. 交易期策略說明

### Spread 重建三條路徑（`zscore_trading.py`）

| 路徑 | 觸發條件 | 公式 |
| :--- | :--- | :--- |
| A | `OLS_Alpha` 不為 None | $\ln P_A - \alpha - \beta \ln P_B$ |
| B1 | `OLS_Alpha` 為 None + `First_Price_A/B > 0` | $P_A/P_{A0} - P_B/P_{B0}$ |
| B2 | `OLS_Alpha` 為 None，無 `First_Price` | $P'_A - \beta P'_B$（Z-Score 標準化 log） |

### DRL LSTM-DQN（`drl_lstm_trading.py`）

- **訓練**：每配對每期獨立訓練，40 episodes，agent 快取於 `Trading._shared_agents`
- **8 維觀測**：`[ZScore, Rel_Return, MA_Dist, TTM, Spread_Std, position, days_held_norm, Spread_Trend]`
- **動作**：`0=Flat, 1=Long_Spread, 2=Short_Spread`
- **網路**：LSTM(seq_len=10, hidden=64) → FC → 3 Q 值

### 六大風控機制

1. **SL**：個配對停損（`stop_loss_pct`，預設停用）
2. **DSZ**：Z-Score 發散停損（`dynamic_stop_z`）
3. **PSL**：全域組合停損（`portfolio_stop_loss_pct`，使用 `Unrealized_PnL` 計帳）
4. **MSR**：產業分散上限（`max_sector_ratio`）
5. **Cooldown**：方向性冷卻（等 Z 穿越 0 才解凍）
6. **VOL ADJ**：波動率自適應（`use_vol_adjust`，動態放大 σ）

---

## 6. 資料庫規範

### dataset/ — 價格資料庫（Git LFS）

| 檔案 | 用途 |
| :--- | :--- |
| `sp500_yF.db` | 主要資料庫，`DB_PATH` 預設指向此 |
| `sp500_Tiingo.db` | Tiingo 備用（config 可切換） |
| `sp500_Current.db` | 現行成分股查詢 |

資料表：`Prices`（Date, Symbol, Open, High, Low, Close, Volume）、`Constituents`（Symbol, GICS_Sector）

### formation_data/ — 形成期結果資料庫（Git LFS）

| 檔案 | 說明 |
| :--- | :--- |
| `formation_pairs_sp500_yF.db` | 所有策略的形成期配對主資料庫 |
| `formation_pairs_sp500_yF_*.db` | 各策略獨立暫存庫（測試用，不上傳 Git） |

---

## 7. 策略新增 SOP

1. 在 `strategies/formation/` 建立新模組，實作 `class Formation` 與 `run()` 方法，回傳含 `Ticker_A/B, Rank, Hedge_Ratio` 等欄位的 DataFrame
2. 在 `strategies/trading/` 確認交易期模組（通常直接使用 `drl_lstm_trading.py`）
3. 在 `strategies/config.py` 的 `strategies_raw_all` 新增策略字典，指定 `formation_module`、`trading_module` 及所有 params
4. 調整 `strategies_raw = strategies_raw_all[...]` 以啟用新策略
5. 依序執行 `run_formation.py` → `run_trading.py`

---

## 8. Git 規範

### 追蹤原則

| 類型 | 處理 |
| :--- | :--- |
| 核心 `.py` 程式碼 | 一律追蹤 |
| `dataset/*.db`、`formation_data/formation_pairs_sp500_yF.db` | Git LFS 追蹤（`.gitattributes` 設定） |
| `formation_data/formation_pairs_sp500_yF_*.db` | `.gitignore` 忽略（測試暫存） |
| `results/`、`tmp/`、`scratch/`、`data/`、`Ref_CODE/` | `.gitignore` 忽略 |
| `*.db-shm`、`*.db-wal` | `.gitignore` 忽略（SQLite WAL 暫存） |

### 注意事項

- 執行 `git add formation_data/*.db` 前確認已安裝 Git LFS（`git lfs install`）
- 大型 DB 首次推送需要 LFS 儲存空間配額
