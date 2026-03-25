# S&P 500 無倖存者偏差歷史資料庫 (Survivorship-Bias-Free Database)

本專案建立了一個健全的本地 SQLite 資料庫，其中包含自 2000 年 1 月 1 日起標普 500 (S&P 500) 指數的歷史成分股與每日股價資料。本系統透過追蹤隨時間推移的指數納入與剔除紀錄，有效消除了股市回測中常見的倖存者偏差 (Survivorship bias)。

## 專案架構

*   `models.py`: SQLite 資料庫的 SQLAlchemy 模型定義 (`Tickers`, `IndexMembership`, `DailyPrices`)。
*   `history_scraper.py`: 負責爬取 Wikipedia 的標普 500 列表，以逆向推導還原 2000 年至今所有曾經存在的歷史成分股與其存續區間。
*   `price_downloader.py`: 透過 `yfinance` 自動下載資料庫內所有股票清單的歷史股價資料。若遇到已下市或無法取得資料的公司，程式具備自動容錯機制以確保流程不中斷。
*   `main.py`: 端對端 (End-to-End) 的自動化整合測試腳本。

## 安裝與執行步驟

1.  **安裝依賴套件**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **初始化資料庫並爬取歷史成分股清單**:
    ```bash
    python history_scraper.py
    ```
    此步驟將會生成 `sp500_data.db`，並寫入所有曾出現的股票代碼與所屬區間。
3.  **下載歷史股價資料**:
    ```bash
    python price_downloader.py
    ```

## 🔍 如何查詢特定日期的「真實」標普 500 成分股清單

若要查詢過去任意日期（例如 2008 年 9 月 15 日）當下真正的標普 500 成分股，您只需對 `index_memberships` 資料表執行 SQL 查詢。

一檔股票在「目標日期」屬於成分股的條件為：
1. `start_date` <= 目標日期 且
2. `end_date` 為空值 (NULL) 或是 `end_date` >= 目標日期

### Python SQLAlchemy 查詢範例

```python
from datetime import datetime
from models import get_engine, get_session, IndexMembership

engine = get_engine()
session = get_session(engine)

target_date = datetime(2008, 9, 15).date()

constituents = session.query(IndexMembership.ticker).filter(
    (IndexMembership.start_date <= target_date) & 
    ((IndexMembership.end_date == None) | (IndexMembership.end_date >= target_date))
).all()

tickers = [row[0] for row in constituents]
print(f"2008-09-15 標普 500 成分股數量: {len(tickers)}")
print(tickers)
```

### 原生 SQL 查詢範例

```sql
SELECT ticker 
FROM index_memberships 
WHERE start_date <= '2008-09-15'
AND (end_date IS NULL OR end_date >= '2008-09-15');
```

## 下市股票 (Delisted Stocks) 處理機制
因為 `yfinance` 的資料來源是當前的 Yahoo Finance，部分經歷重大重組、破產或是完全下市的股票可能會無法下載。`price_downloader.py` 腳本將會自動捕捉這些錯誤，並在 `Tickers` 資料表內的 `is_delisted` 欄位標記為 `True`。您可以利用此欄位在後續分析中過濾掉缺乏歷史資料的股票，或者作為清單去向專業且付費的資料庫（如 CRSP 或 Compustat）尋找這些下市公司的股價。
