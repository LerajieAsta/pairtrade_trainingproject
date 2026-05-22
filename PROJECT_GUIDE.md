# Pairs Trading 專案開發與策略管理指南 (PROJECT_GUIDE)

歡迎使用 Pairs Trading 量化專案！本指南將說明專案的目錄結構、一鍵啟動方式，以及如何進行策略的**新增、修訂與刪除**作業，協助您建立高效的量化回測與績效比對流程。

---

## 1. 專案目錄結構導覽

```text
pairtrade_trainingproject/
├── .vscode/               # VS Code 開發環境配置（設定檔、除錯、自訂任務）
├── Project/               # Python venv 虛擬環境（已在 .gitignore 中排除）
├── src/                   # 共享核心模組目錄（預留給未來共用邏輯，已支援 Editable Install）
│   └── __init__.py
├── notebooks/             # Jupyter Notebooks 實驗與策略回測開發目錄
├── results/               # 存放所有策略回測結果 CSV 的目錄
├── app.py                 # Streamlit 績效比對 Dashboard 應用程式
├── requirements.txt       # 專案相依套件清單
├── pyproject.toml         # 專案套件配置檔
├── setup.bat              # [一鍵工具] 專案環境初始化與套件安裝
└── run.bat                # [一鍵工具] 一鍵啟動 Streamlit Dashboard
```

---

## 2. 快速啟動與開發環境

### 🚀 一鍵初始化與啟動
1. **首次執行**：按兩下 `setup.bat`。它會自動為您建立 `Project` 虛擬環境、升級 pip、安裝 `requirements.txt` 所有依賴，並以開發者模式（Editable Mode）安裝 `src` 本地模組。
2. **啟動儀表板**：按兩下 `run.bat`，即可一鍵載入虛擬環境並啟動 Streamlit 績效比對儀表板（Dashboard）。瀏覽器會自動開啟至 `http://localhost:8501`。

### 💻 VS Code 進階功能
* **自動直譯器切換**：開啟專案後，VS Code 會自動將直譯器切換為 `Project` 虛擬環境。
* **一鍵偵錯 (F5)**：
  - 選擇 **"Streamlit: Pairs Trading Dashboard"**：可直接在 `app.py` 中設置中斷點並進行除錯。
  - 選擇 **"Python: Current File"**：除錯當前開啟的 Python 腳本。

---

## 3. 策略 CSV 資料對接規範 (Data Contract)

Dashboard (`app.py`) 是透過掃描 `results/` 目錄下的 CSV 檔案來進行績效分析的。為了讓您新產出的策略回測結果能順利在 Dashboard 上顯示並被正確篩選，回測 CSV 必須符合以下規範。

### 3.1 檔名特徵解析規則
Dashboard 會掃描 `results/` 目錄中檔名包含 **`TradeLogs`** 或 **`detailed_trade_logs`** 的 CSV 檔案。
它會透過檔名的關鍵字自動解析出策略特徵並提供多維度 Filter。解析邏輯如下：

| 欄位 | 規則 (不分大小寫) | 範例 / 輸出結果 |
| :--- | :--- | :--- |
| **DATASET** | 包含 `full` $\rightarrow$ `Full`<br>包含 `current` $\rightarrow$ `Current`<br>其他 $\rightarrow$ `Unknown` | `strategy_full_TradeLogs.csv` $\rightarrow$ **Full** |
| **RE-ENTRY** | 包含 `noreentry` $\rightarrow$ `NoReEntry`<br>包含 `reentry` $\rightarrow$ `ReEntry`<br>其他 $\rightarrow$ `Unknown` | `SSD_noreentry_TradeLogs.csv` $\rightarrow$ **NoReEntry** |
| **METHOD** | 包含 `ssd_basic` $\rightarrow$ `SSD (Basic)`<br>包含 `ssd` $\rightarrow$ `SSD`<br>包含 `eg` $\rightarrow$ `EG`<br>包含 `hdbscan_ae_pca` 或包含 `_ae_` 且 `_pca_` $\rightarrow$ `HDBSCAN (AE PCA)`<br>包含 `hdbscan_ae` 或單純有 `_ae_` $\rightarrow$ `HDBSCAN (AE UMAP)`<br>包含 `hdbscan_pca` 或單純有 `_pca_` $\rightarrow$ `HDBSCAN (PCA)`<br>其他 `hdbscan` 相關 $\rightarrow$ `HDBSCAN (UMAP)` | `eg_reentry_TradeLogs.csv` $\rightarrow$ **EG**<br>`hdbscan_ae_pca_TradeLogs.csv` $\rightarrow$ **HDBSCAN (AE PCA)** |
| **TOP N** | 正則匹配 `top(\d+)` | `top20` $\rightarrow$ **Top 20**（預設為 Top 20） |
| **STOP LOSS %** | 正則匹配 `sl(\d+)` | `sl2` $\rightarrow$ **2%**（預設為 0%） |
| **Z-WINDOW** | 正則匹配 `zwin(\d+)` | `zwin60` $\rightarrow$ **60**（預設為 0） |

> 💡 **最佳命名格式建議**：
> `results/strategy_[DATASET]_[RE-ENTRY]_[METHOD]_top[N]_sl[SL]_zwin[Z]_detailed_trade_logs.csv`
> * 實例：`results/strategy_full_reentry_ssd_top20_sl2_zwin60_detailed_trade_logs.csv`

### 3.2 CSV 必備欄位結構
回測產出的 CSV 檔案中，請務必包含並精確命名以下欄位（Dashboard 載入時會自動進行欄位重命名與優化）：
* `Date` (格式: `YYYY-MM-DD`) - 交易日期
* `Position` (數值, 例如 `1`, `0`, `-1`) - 持倉狀態
* `Ticker_A`, `Ticker_B` - 交易配對標的代號
* `Daily_Delta` (數值) - 每日 PnL 損益變化量（用以計算累計權益曲線、年化報酬率與 Sharpe Ratio）
* `Status` (字串, 如 `Stop Loss`、`Normal Exit` 或 `停損`) - 交易結束狀態（若包含 stop/sl/停損，會被統計入 Stop Losses 次數）
* `Hedge_Ratio` - 配對避險比例
* `Price_A`, `Price_B` - 標的價格（用以進行單對 Trade Visualizer 繪圖）
* `Days_Held` - 持倉天數
* `Period_Start`, `Period_End` - 該配對所屬的交易週期起訖時間

---

## 4. 策略管理標準作業程序 (SOP)

### ➕ 4.1 新增策略作業
1. **策略開發與回測**：在 `notebooks/` 的 Jupyter Notebook 跑完策略回測，並產出符合上述欄位結構的 Trade Logs。
2. **匯出 CSV**：將該 Trade Logs 以符合 [3.1 檔名特徵解析規則](#31-檔名特徵解析規則) 的名稱格式匯出至 `results/` 資料夾中。
3. **Dashboard 載入**：
   - 重新整理 Streamlit 網頁。
   - Dashboard 會自動掃描 `results/` 下的新檔案，解析其特徵，並立刻在 **Filters** 面板的下拉選單以及 **Performance Table** 中呈現。

---

### 📝 4.2 修訂策略作業
如果您要調整現有策略的參數（例如將停損點 `sl2` 改為 `sl5`，或修改策略邏輯）：
1. **覆寫回測數據**：修改 Notebook 中的參數並重新運行，生成最新的 Trade Logs。
2. **更新檔案**：
   - **情況 A（參數相同，微調回測邏輯）**：以相同的檔名覆寫 `results/` 下的舊 CSV 檔案。
   - **情況 B（參數改變）**：將新 CSV 儲存為對應的新檔名（例如 `sl2` 變更為 `sl5`），並可自行決定是否刪除舊的 `sl2` 檔案。
3. **⚠️ 關鍵步驟：清除 Dashboard 快取**：
   - 由於 Dashboard 使用了 Streamlit 記憶體快取技術 (`@st.cache_data`) 來加速巨量數據的讀取，**單純重新整理網頁是不會讀入更新後的 CSV 資料的！**
   - **清除快取方法**：
     - 在 Streamlit 網頁右上角，點選三個點的選單 $\rightarrow$ 點擊 **"Clear cache"**（或直接在網頁畫面上按下鍵盤的 **`C`** 鍵）。
     - 點擊彈出確認視窗中的 "Clear cache"，網頁便會重新載入並強迫讀取最新的實體 CSV 檔案進行運算。

---

### ❌ 4.3 刪除策略作業
當某些舊策略不再需要進行績效比對時：
1. **移除檔案**：
   - 開啟 `results/` 目錄，直接刪除不需要的 CSV 檔案。
   - *（推薦做法）* 如果不想永久刪除，可以在專案下建立備份資料夾如 `archive/`，將舊 CSV 移入。因為 Dashboard 只會掃描 `results/` 根目錄與其子目錄，移出此目錄的檔案將不會被載入。
2. **⚠️ 關鍵步驟：清除 Dashboard 快取**：
   - 刪除實體檔案後，Streamlit 的記憶體快取中可能仍殘留該策略的運算結果。
   - 請務必在 Streamlit 介面上按下鍵盤的 **`C`** 鍵清除快取，如此一來，已刪除的策略就會完全從 Dashboard 的選單與表格中消失。
