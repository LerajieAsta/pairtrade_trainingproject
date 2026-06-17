# S&P 500 Pairs Trading 量化交易平台 (Pairs Trading Platform)

本專案是一個針對 S&P 500 成分股所建構的高效能、科學化**配對交易 (Pairs Trading) 滾動回測、指標編譯與互動式視覺化分析平台**。專案支援多行程平行計算、自編碼器 (Autoencoder) 特徵提取、無監督學習密度分群 (HDBSCAN)、以及自動化學術簡報發佈。

---

## 🚀 平台核心亮點

### 1. 高效能多行程並行回測引擎 (`main.py`)
* **智慧單次 I/O 設計**：主行程統一載入 SQLite 資料庫並完成 Pivot 矩陣轉換，子行程直接共用記憶體數據，根除硬碟讀寫瓶頸。
* **網格搜尋優化**：針對配對組數 (Top N)、停損門檻 (Stop Loss)、Z-Score 滾動天數 (Z-Window 固定為 0)、全域動態停損 (PSL)、配對產業上限 (MSR) 與部位動態停損 (DSZ) 等 7 組參數自動進行網格搜尋與科學績效比對，並引進高效的外部產業上限過濾技術。
* **智慧斷點續傳**：自動對比資料庫修改時間、策略代碼雜湊與網格參數，未變動之策略自動跳過，極速節省計算開銷。

### 2. 五大前沿配對交易策略 (`strategies/`)
* **經典 SSD (Basic)**：基於最小平方距離法進行傳統股票配對。
* **進階 SSD (OLS)**：基於滾動殘差回歸與統計參數優化的進階配對距離策略。
* **HDBSCAN Clustering (UMAP/PCA)**：利用非線性流形降維 (UMAP) 或主成分分析 (PCA) 後，進行密度分群與 Engle-Granger 共整合雙重過濾。
* **HDBSCAN Autoencoder (AE)**：利用 PyTorch 深度神經自編碼器壓縮高維特徵，再進行密度聚類與共整合配對篩選。
* **HDBSCAN MultiFactor**：結合股票基本面、動量與統計多因子特徵進行高維密度分群。

### 3. 互動式績效比對儀表板 (`dashboard.py`)
* **多維交叉篩選**：支援資料集、重入機制、波動調節、策略方法、配對數、停損率、Z-Window 的下拉即時過濾。
* **權益曲線對比**：動態載入與疊加多達 5 組最優策略的累計帳戶資產曲線。
* **Trade Visualizer**：可下鑽至特定交易期與配對股票，動態繪製含有買入（Buy Long）、賣空（Sell Short）、平倉（Close）與停損（Stop Loss）時點標記的 K 線與價差圖。

### 4. 自動化指標編譯與簡報發佈 (`tohtml/`)
* **性能自動注入**：`preprocess_equity.py` 會自動合併最優曲線至 `equity_curves.csv`，並將最新真實數據的 HTML 表格直接注入至分析 Notebook。
* **一鍵 RevealJS 渲染**：整合 Quarto 工具鏈，一鍵發佈高質感網頁簡報至 `docs/`（相容 GitHub Pages），實現學術進度實時展示。

---

## 📁 專案目錄結構摘要

```text
pairtrade_trainingproject/
├── strategies/            # 核心策略目錄 (SSD, HDBSCAN, Autoencoder 等)
├── fetch/                 # 數據下載與維護模組 (Yahoo Finance, Tiingo)
├── data/                  # 歷史價格與成分股 SQLite 資料庫
├── results/               # 回測日誌與 TradeLogs CSV 資料夾
├── notebooks/             # 核心分析 Notebook (analysis.ipynb) 與淨值曲線
├── tohtml/                # 量化指標編譯器、互動圖表生成與 Quarto 渲染腳本
├── docs/                  # 簡報 HTML 與網頁發佈目錄 (GitHub Pages 來源)
├── ref/                   # 學術文獻與經典論文庫 (共 23 篇 PDF)
├── archive/               # 歷史存檔與過期分析檔案目錄 (依日期/月份分類，如 114/, 11505/, 11506/)
├── Ref_CODE/              # 歷史參考程式碼與結果比對備份
├── tmp/                   # 臨時腳本與防禦性論證輔助工具目錄
├── dashboard.py           # Streamlit 視覺化 Dashboard 應用程式
├── main.py                # 多行程滾動回測與網格搜尋控制主程式
├── requirements.txt       # 專案相依 Python 套件清單
├── setup.bat              # [一鍵工具] 專案環境初始化與套件安裝
└── run.bat                # [一鍵工具] 績效比對 Dashboard 一鍵啟動器
```

> ⚙️ 更詳細的目錄與檔案說明，請參閱 [PROJECT_GUIDE.md](file:///c:/Clark/YZU/Papper/Code/PROJECT_GUIDE.md)。

---

## 🏁 快速啟動指南

### 1. 環境一鍵初始化
按兩下根目錄底下的 **`setup.bat`**。它會自動為您：
* 創建 `Project` Python 虛擬環境。
* 升級 pip 並自動安裝 `requirements.txt` 中所有的量化、機器學習與深度學習相依套件。
* 以開發者模式安裝本地 `src` 模組，確保跨策略核心引用暢通。

### 2. 啟動多行程回測與網格搜尋
在專案根目錄下執行：
```bash
# 啟動互動選單（手動點選資料集、重入機制與波動度調節）：
python main.py

# 或使用非互動式 CLI 參數在後台直接跑完：
python main.py --db sp500_Current --allow-reentry --workers 4
```

### 3. 指標編譯與 Notebook 表格注入
當回測結束並在 `results/` 產生 CSV 檔案後，執行編譯程式：
```bash
python preprocess_equity.py
```
這會自動提取六大策略的最優績效，合併淨值生成 `notebooks/equity_curves.csv`，並將最新績效 HTML 表格注入到 [notebooks/analysis.ipynb](file:///c:/Clark/YZU/Papper/Code/notebooks/analysis.ipynb) 中。

### 4. 一鍵啟動 Streamlit 績效比對儀表板
按兩下根目錄底下的 **`run.bat`**。
它會自動啟用虛擬環境並開啟瀏覽器展示儀表板（預設網址為 [http://localhost:8501](http://localhost:8501)）。

---

## 📖 核心參考文件

* **策略開發與檔案管理 SOP**：請務必詳閱 [PROJECT_GUIDE.md](file:///c:/Clark/YZU/Papper/Code/PROJECT_GUIDE.md)。其中包含了資料契約 (Data Contract)、命名規範以及「新增、修訂、刪除策略」的標準作業程序。
* **學術與文獻背景**：請參閱 [ref/](file:///c:/Clark/YZU/Papper/Code/ref) 目錄，其中收錄了 23 篇本專案架構（包含聚類、降維與自編碼器應用於配對交易）的奠基學術文獻。
