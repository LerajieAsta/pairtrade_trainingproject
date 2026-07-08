# Pairs Trading 專案開發指南 (PROJECT_GUIDE)

---

## 1. 專案目錄結構

```text
pairtrade_trainingproject/
├── strategies/
│   ├── config.py                  # 策略清單（10 交易策略 + 2 formation-only）、網格參數、全域回測設定
│   ├── db_utils.py                # SQLite 合併、讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理（MSR 產業上限）
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── _utils.py              # 共用統計工具（_ols、_adf_stat、_compute_hurst）
│   │   ├── ssd_rolling.py         # SSD Rolling：Z-Score 標準化 log-price 空間（#1；被 #9 複用排序）
│   │   ├── DTW_Cointegration_Paper.py  # DTW + ADF 雙重篩選 + Sakoe-Chiba DTW + PCA 融合（#2/#3 借用；#4/#7 複用）
│   │   ├── HDBSCAN_PCA_Loadings.py # 報酬 PCA 因子載荷特徵萃取（被 #4/#7/#9 組合複用）
│   │   ├── HDBSCAN_Cluster_SSD_DTW.py # 組合：HDBSCAN 聚類 + DTW 排序（#4/#7）
│   │   ├── agglomerative_fundamentals.py # 組合：Agglomerative 聚類（價格 PCA⊕基本面）+ SSD 排序（#9）
│   │   ├── ml_pair_quality.py     # 監督式學習排序 walk-forward 反事實回歸（已封存，程式碼保留）
│   │   ├── ssd_basic.py / HDBSCAN_UMAP.py / HDBSCAN_MultiScale.py / ensemble.py  # 已封存
│   │   └── __init__.py
│   └── trading/
│       ├── zscore_trading.py      # Z-Score 狀態機（基礎類，三條 Spread 路徑；現役僅走路徑 B）
│       └── drl_threshold_trading.py # DRL 門檻選擇式 v4（#5、#6、#8、#10 使用）
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   ├── sp500_yf_now.py            # yFinance 當日數據更新
│   └── fundamentals_yfinance.py   # 公司基本面快照下載（市值、本益比，供 Agglomerative Fundamentals 使用）
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
├── notebooks/                     # 策略筆記本（一策略一本，Quarto revealjs 投影片；見 notebooks/README.md）
│   ├── formation/                 # 形成期策略 ×6：ssd_rolling ~ agglomerative_fundamentals（以策略方法命名）
│   ├── trading/                   # 交易期策略 ×2：zscore_trading、drl_threshold_trading
│   ├── comparison.ipynb           # 現役 10 策略績效總比較（讀 results/result.db 動態產生）
│   ├── _quarto.yml / slides.scss  # revealjs 投影片設定（大字型、Alt+點擊縮放、KaTeX）
├── docs/                          # GitHub Pages 輸出
│   ├── index.html                 # 入口頁（連結全部投影片）
│   └── slides/                    # quarto render 產出的 9 份投影片
├── archive/                       # 歷史存檔（分類索引見 archive/README.md）
│   ├── config_archived_strategies.py  # 已封存策略 config（含孤兒模組盤點記錄）
│   ├── trading/                   # 非現役交易模組（drl_lstm×2、drl_fqi、kalman、pure_dtw）
│   ├── notebooks/                 # 舊版筆記本（114/11505/11506/11507 依時期）
│   ├── formation/                 # 已封存形成期模組（11506 CrossSector 系列等）
│   ├── scripts/                   # 一次性工具腳本
│   ├── docs/                      # 歷次學術 HTML 簡報（含舊版 formation/trading.html）
│   └── h200/                      # H200 GPU 伺服器相關（2026-07-06 起不再使用）
├── dashboard.py                   # Streamlit 績效比對儀表板
├── run_formation.py               # 形成期主程式
├── run_trading.py                 # 交易期主程式
├── run.bat                        # 一鍵啟動 Dashboard
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
- 逐滾動期續傳（Z-Score 策略）：每期算完即以 pickle 落地至 `results/<dataset>/.ckpt/<策略>/<期>.pkl`；中斷重跑僅補算缺漏期，並自 checkpoint 重建 PortfolioManager 權益。summary 於全部期完成後定稿並清除 checkpoint。DRL 策略因 walk-forward 訓練狀態暫不套用（維持整策略重跑）
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
2. **全量重跑前先歸檔舊版**：`python snapshot_run.py <tag>` 把 result.db 搬到 `archive/` 並匯出 summary CSV，再 `FORCE_RERUN=True` 重建。可新舊並存比較、回滾。
3. **輕量對照**：`python snapshot_run.py <tag> --summary-only` 只匯出 summary CSV 不動 DB。

---

## 3. 策略清單（`config.py`）

`strategies_raw_all` 為現役策略池（10 個交易策略 + 尾端 2 個 formation-only 條目），
`strategies_raw = strategies_raw_all[:]` 決定實際執行範圍
（或用環境變數 `STRATEGIES_SLICE` 免改檔覆寫，支援逗號複合切片；0-based Python 切片語意）：

```bash
STRATEGIES_SLICE="4:6" python run_trading.py   # 只跑 #5 SSD Rolling DRL THR、#6 HDBSCAN Cluster ... DRL THR
```

| # | 策略名稱 | 形成期 | 交易期 | 角色 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | SSD Rolling | `ssd_rolling.py` | `zscore_trading.py` | SSD 家族基準 |
| 2 | DTW Paper Fixed (DTW) | 借用 DTW Paper 原版配對 | `zscore_trading.py`（路徑 B） | DTW 基準 |
| 3 | SSD-DTW-PCA Paper Fixed | 借用配對 | `zscore_trading.py`（路徑 B） | 全組最佳 Sharpe 基準：Top3 0.56 |
| 4 | HDBSCAN Cluster SSD-DTW-PCA | `HDBSCAN_Cluster_SSD_DTW.py` | `zscore_trading.py` | 分組消融（vs #3，15 維 PCA） |
| 5 | SSD Rolling DRL THR | 借用 #1 配對 | `drl_threshold_trading.py` | DRL 疊加對照組 |
| 6 | HDBSCAN Cluster SSD-DTW-PCA DRL THR | 借用 #4 配對 | `drl_threshold_trading.py` | DRL 疊加實驗組 |
| 7 | HDBSCAN Cluster SSD-DTW-PCA PCA5 | `HDBSCAN_Cluster_SSD_DTW.py`（5 維 PCA） | `zscore_trading.py` | 5 維 PCA 聚類穩定性修復版（vs #4） |
| 8 | HDBSCAN Cluster SSD-DTW-PCA PCA5 DRL THR | 借用 #7 配對 | `drl_threshold_trading.py` | 全組 DRL 疊加最高 Sharpe 0.46 |
| 9 | Agglomerative Fundamentals | `agglomerative_fundamentals.py` | `zscore_trading.py` | 分組消融第三支：價格 PCA⊕基本面 |
| 10 | Agglomerative Fundamentals DRL THR | 借用 #9 配對 | `drl_threshold_trading.py` | 全組最佳年化報酬 2.89% |
| — | DTW Paper (DTW) / (SSD-DTW-PCA) | `DTW_Cointegration_Paper.py` | formation-only（跳過回測） | 產生 #2/#3 借用的原版配對 |

**formation-only 條目（2026-07-05）**：repo LFS 額度用罄致 formation DB 無法下載後，DTW Paper 原版
兩條目以 `formation_only: True` 旗標回歸現役清單尾端——`run_formation.py` 計算其配對（供 #2/#3
借用），`run_trading.py` 跳過（原版交易端為座標 artifact，維持封存）。全部形成期配對因此可在任何
機器上以 `run_formation.py` 完整本地重算。

**已封存策略（20 個 config 條目）**：`archive/config_archived_strategies.py`
（SSD Basic、HDBSCAN 舊特徵系×4、Ensemble×2、DRL v1×3、Kalman×2、
CONV×2、DRL FQI 系×3〔逐日定位動作空間已證偽〕、HDBSCAN PCA-Loadings 系×2、ML Pair Quality×1），
封存理由與完整診斷數據見該檔 docstring，歷史回測結果保留於 `results/result.db`。
⚠️ HDBSCAN PCA-Loadings 的形成期配對仍被復活備用借用，formation DB 資料列不可刪。

**孤兒模組（2 個，從未進入策略清單）**：`pure_dtw_trading.py`（早期交叉回穿波帶構想）、
`drl_lstm_v2_trading.py`（DRL v1→v3 演進鏈中繼修復版）——程式碼保留供架構脈絡參考，
`result.db` 無對應回測數據，盤點記錄見 `archive/config_archived_strategies.py` docstring末段。

**非現役交易模組已歸位 `archive/trading/`（2026-07-05）**：上述孤兒模組×2 加上已封存的
`drl_lstm_trading.py`（v1）、`drl_fqi_trading.py`（v3）、`kalman_trading.py` 共 5 檔自
`strategies/trading/` 移入；封存 config 的 `trading_module` 已同步指向 `archive.trading.*`，
復活時不需搬回檔案。`strategies/trading/` 現只含現役的 `zscore_trading.py` 與
`drl_threshold_trading.py`。

---

## 4. 形成期策略說明

### Spread 空間與 Formation_Params（現役策略統一走路徑 B）

| 策略 | 分組依據 | 排序邏輯 | 關鍵 Formation_Params |
| :--- | :--- | :--- | :--- |
| SSD Rolling (#1) | 真實 GICS 產業 | min-SSD + 三道統計過濾 | `Log_Mean/Std_A/B`, `Spread_Mean/Std` |
| DTW Paper Fixed (#2/#3) | 真實 GICS 產業 | DTW / SSD-DTW-PCA 距離 | `OLS_Alpha`（交易端因 `ignore_ols_alpha=True` 而忽略）, `Log_Mean/Std_A/B` |
| HDBSCAN Cluster SSD-DTW-PCA (#4/#7) | HDBSCAN 聚類（報酬 PCA 因子載荷，15 維/5 維） | 沿用 DTW 模組的 SSD-DTW-PCA 排序 | 同上 + `Sector_A/B`（真實 GICS 回填） |
| Agglomerative Fundamentals (#9) | Agglomerative 聚類（價格 PCA⊕市值⊕PE⊕GICS one-hot） | 沿用 SSD Rolling 的 min-SSD 排序 | `Log_Mean/Std_A/B` + `Cluster_ID`、`MarketCap`、`TrailingPE` |

### 過濾門檻（全現役策略共用）

| 指標 | 門檻 |
| :--- | :--- |
| OU 半衰期 | 1 ≤ halflife ≤ `trading_window`/3 天 |
| Hurst 指數 | < 0.50（均值回歸） |
| ADF p-value | ≤ 0.05（SSD Rolling/Agglomerative）或 ≤ 0.01（DTW/HDBSCAN 系列） |

### 「組合優於重寫」架構模式

#4/#7（HDBSCAN Cluster SSD-DTW-PCA）與 #9（Agglomerative Fundamentals）都不是從零打造的獨立形成邏輯，
而是把既有模組像積木一樣組裝：

```
HDBSCAN_PCA_Loadings._build_feature_matrix()  ──┬──→ HDBSCAN_Cluster_SSD_DTW.py（#4/#7）
  （報酬 PCA 因子載荷特徵萃取）                    │       + DTW_Cointegration_Paper 排序
                                                 │
                                                 └──→ agglomerative_fundamentals.py（#9）
                                                         + 基本面特徵 + Agglomerative 分群
                                                         + ssd_rolling 排序
```

只替換「配對候選怎麼分組」這一環節，其餘（群內共整合篩選、距離排序、spread 定義、交易端）完全相同——
確保每個新策略都是乾淨的單變因消融實驗。詳細公式與診斷數據見 `notebooks/formation/`（一策略一本）。

---

## 5. 交易期策略說明

### Spread 重建（`zscore_trading.py`，現役策略統一走路徑 B）

$$P'_{i,t} = \frac{\ln P_{i,t} - \mu^{form}_{\ln P_i}}{\sigma^{form}_{\ln P_i}}, \qquad \text{Spread}_t = P'_{A,t} - \beta \cdot P'_{B,t}$$

路徑由 `run_trading.py` 依 `Formation_Params` 是否含 `OLS_Alpha` 及 `ignore_ols_alpha` 參數決定；
現役策略全數強制或原生走此路徑（路徑 A 原始 log-price OLS 殘差空間、路徑 B1 累積回報比值空間
僅供已封存策略使用，程式碼保留於 `zscore_trading.py._compute_spread()`）。

### DRL 門檻選擇式 v4（`drl_threshold_trading.py`）

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
確保「Z-Score 基準 vs DRL 疊加」比較時唯一變因是交易決策邏輯。詳見 `notebooks/trading/drl_threshold_trading.ipynb`。

---

## 6. 資料庫規範

### dataset/ — 價格與基本面資料庫（Git LFS）

| 檔案 | 用途 |
| :--- | :--- |
| `price/sp500_Tiingo.db` | 主要資料庫，`DB_PATH` 預設指向此 |
| `price/sp500_yF.db` | yFinance 備用（config `DB_PROFILES` 可切換） |
| `price/sp500_Current.db` | 現行成分股查詢 |
| `fundamental/fundamentals_sp500.db` | 公司基本面快照（市值、本益比），供 `agglomerative_fundamentals.py` 使用；⚠️ 單一時點靜態資料，非歷史逐日序列。2026-07-05 以 `fetch/fundamentals_yfinance.py` 重抓（843 檔：633 有效、210 已下市） |
| `fundamental/sp500_pit_2000_2025_monthly.parquet`、`fundamental/fmp_cache/` | FMP Point-in-Time 基本面與 API 快取（本地產物，不進版控） |

資料表：`Daily_Prices`（Date, Symbol, Open, High, Low, Close, Volume）、`Constituents`（Symbol, GICS_Sector）

### formation_data/ — 形成期結果資料庫（Git LFS）

| 檔案 | 說明 |
| :--- | :--- |
| `formation_pairs_sp500_Tiingo.db` | 所有策略的形成期配對主資料庫 |
| `formation_pairs_sp500_Tiingo_*.db` | 各策略獨立暫存庫（測試用，不上傳 Git） |

---

## 7. 策略新增 SOP

1. 在 `strategies/formation/` 建立新模組，實作 `class Formation` 與 `run()` 方法，回傳含
   `Ticker_A/B, Rank, Hedge_Ratio, Spread_Mean, Spread_Std, Log_Mean_A/B, Log_Std_A/B` 等欄位的 DataFrame
   （這 7 個是 `run_trading.py` 讀 `Formation_Params` 的硬性合約，缺一不可）
2. 若要複用既有分組/排序邏輯，優先考慮「組合」既有模組（見第 4 節架構模式）而非重寫
3. 在 `strategies/trading/` 確認交易期模組（通常直接使用 `zscore_trading.py` 或
   `drl_threshold_trading.py`，兩者接口一致）
4. 在 `strategies/config.py` 的 `strategies_raw_all` 新增策略字典，指定 `formation_module`、
   `trading_module` 及所有 params；若要借用其他策略已算好的配對，加上
   `formation_strategy_id_base` 指向該策略名稱
5. 依序執行 `run_formation.py` → `run_trading.py`
6. 用 `results/result.db` 的 `strategy_summaries` 與現有基準（SSD Rolling / SSD-DTW-PCA Paper Fixed）
   做同條件對照；驗證無效則移入 `archive/config_archived_strategies.py` 並記錄診斷結論（不刪除歷史數據）

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
