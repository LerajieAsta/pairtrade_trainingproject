"""
Firm-fundamentals snapshot fetcher (market cap, trailing P/E) via yfinance
======================================================================
一次性快照抓取：非歷史逐日資料。yfinance/Tiingo 免費層皆不提供 2000-2025
逐點時點 (point-in-time) 基本面資料（需付費資料商如 WRDS/Compustat），
因此本腳本抓取「今日」市值與本益比，供 Agglomerative Fundamentals 形成期
模組在所有歷史形成窗口中重複使用（已知前視偏誤限制，見該模組 docstring）。

寫入獨立的 dataset/fundamentals_sp500.db（不動用共用的 sp500_Tiingo.db），
避免與價格資料庫的讀寫並發或 LFS diff 產生干擾。

用法：
    python fetch/fundamentals_yfinance.py            # 略過已存在的 ticker
    python fetch/fundamentals_yfinance.py --force     # 全部重抓
"""

import argparse
import os
import sqlite3
import time
from datetime import date

import yfinance as yf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

SOURCE_DB_PATH = os.path.join(_PROJECT_DIR, "dataset", "sp500_Tiingo.db")
SOURCE_TABLE = "Constituents"
SOURCE_TICKER_COL = "Symbol"

OUTPUT_DB_PATH = os.path.join(_PROJECT_DIR, "dataset", "fundamentals_sp500.db")

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0
REQUEST_DELAY_SECONDS = 0.4
COMMIT_EVERY = 25


def setup_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Fundamentals (
            Symbol      TEXT PRIMARY KEY,
            MarketCap   REAL,
            TrailingPE  REAL,
            Sector_YF   TEXT,
            Is_Missing  INTEGER,
            Fetch_Date  TEXT
        )
        """
    )
    conn.commit()
    return conn


def get_ticker_universe(source_db_path: str) -> list[str]:
    conn = sqlite3.connect(source_db_path)
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {SOURCE_TICKER_COL} FROM {SOURCE_TABLE} "
            f"WHERE {SOURCE_TICKER_COL} IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return sorted({r[0] for r in rows if r[0]})


def get_existing_symbols(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT Symbol FROM Fundamentals").fetchall()
    return {r[0] for r in rows}


def fetch_one(symbol: str) -> dict:
    """回傳 dict，抓取失敗或缺欄位時 MarketCap/TrailingPE 為 None、Is_Missing=1"""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(symbol).get_info()
            market_cap = info.get("marketCap")
            trailing_pe = info.get("trailingPE")
            sector_yf = info.get("sector")
            is_missing = 1 if (market_cap is None and trailing_pe is None) else 0
            return {
                "Symbol": symbol,
                "MarketCap": market_cap,
                "TrailingPE": trailing_pe,
                "Sector_YF": sector_yf,
                "Is_Missing": is_missing,
            }
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

    print(f"  跳過 {symbol}：連續 {MAX_RETRIES} 次失敗（{last_err}）")
    return {
        "Symbol": symbol,
        "MarketCap": None,
        "TrailingPE": None,
        "Sector_YF": None,
        "Is_Missing": 1,
    }


def run(force: bool = False) -> None:
    os.makedirs(os.path.dirname(OUTPUT_DB_PATH), exist_ok=True)
    conn = setup_database(OUTPUT_DB_PATH)

    tickers = get_ticker_universe(SOURCE_DB_PATH)
    print(f"從 {SOURCE_DB_PATH} 取得 {len(tickers)} 檔待抓取標的。")

    existing = set() if force else get_existing_symbols(conn)
    todo = [t for t in tickers if t not in existing]
    print(f"已存在 {len(existing)} 檔，待抓取 {len(todo)} 檔。")

    fetch_date = date.today().isoformat()
    n_missing = 0
    n_ok = 0

    for i, symbol in enumerate(todo):
        record = fetch_one(symbol)
        record["Fetch_Date"] = fetch_date

        conn.execute(
            """
            INSERT INTO Fundamentals (Symbol, MarketCap, TrailingPE, Sector_YF, Is_Missing, Fetch_Date)
            VALUES (:Symbol, :MarketCap, :TrailingPE, :Sector_YF, :Is_Missing, :Fetch_Date)
            ON CONFLICT(Symbol) DO UPDATE SET
                MarketCap=excluded.MarketCap, TrailingPE=excluded.TrailingPE,
                Sector_YF=excluded.Sector_YF, Is_Missing=excluded.Is_Missing,
                Fetch_Date=excluded.Fetch_Date
            """,
            record,
        )

        if record["Is_Missing"]:
            n_missing += 1
        else:
            n_ok += 1

        if (i + 1) % COMMIT_EVERY == 0:
            conn.commit()
            print(f"[{i + 1}/{len(todo)}] 進度儲存中... (成功 {n_ok} / 缺失 {n_missing})")

        time.sleep(REQUEST_DELAY_SECONDS)

    conn.commit()
    conn.close()
    print(f"\n完成。本次抓取 {len(todo)} 檔：成功 {n_ok}，缺失 {n_missing}。")
    print(f"資料庫：{OUTPUT_DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a static S&P 500 fundamentals snapshot via yfinance.")
    parser.add_argument("--force", action="store_true", help="重新抓取所有 ticker（忽略已存在資料）")
    args = parser.parse_args()
    run(force=args.force)
