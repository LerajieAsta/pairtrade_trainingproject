import pandas as pd
import yfinance as yf
import sqlite3
import time
import requests
from datetime import datetime

def get_sp500_tickers():
    """
    從 Wikipedia 抓取目前的 S&P 500 成份股以及歷史變動表。
    使用 requests 加入 User-Agent 以避免 403 Forbidden 錯誤。
    """
    print("正在從 Wikipedia 獲取 S&P 500 成份股清單...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        
        df_current = tables[0]
        current_tickers = df_current['Symbol'].tolist()
        
        # 歷史變動表
        df_adjustments = tables[1]
        added_tickers = []
        removed_tickers = []
        
        # 處理 Wikipedia 表格可能的層級結構
        if 'Added' in df_adjustments.columns:
            added_tickers = df_adjustments['Added']['Ticker'].dropna().tolist() if isinstance(df_adjustments['Added'], pd.DataFrame) else df_adjustments['Added'].dropna().tolist()
        if 'Removed' in df_adjustments.columns:
            removed_tickers = df_adjustments['Removed']['Ticker'].dropna().tolist() if isinstance(df_adjustments['Removed'], pd.DataFrame) else df_adjustments['Removed'].dropna().tolist()
        
        all_tickers = list(set(current_tickers + added_tickers + removed_tickers))
        all_tickers = [str(t).replace('.', '-') for t in all_tickers if pd.notna(t)]
        
        print(f"成功！共找到 {len(all_tickers)} 支歷年成分股。")
        return all_tickers, df_current

    except Exception as e:
        print(f"獲取清單失敗: {e}")
        return [], pd.DataFrame()

def setup_database(db_name="SP500.db"):
    """
    初始化 SQLite 資料庫結構
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Constituents (
            Symbol TEXT PRIMARY KEY,
            Security TEXT,
            GICS_Sector TEXT,
            GICS_Sub_Industry TEXT,
            Headquarters_Location TEXT,
            Date_Added TEXT,
            CIK TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Daily_Prices (
            Date TEXT,
            Symbol TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume INTEGER,
            PRIMARY KEY (Date, Symbol)
        )
    ''')
    
    conn.commit()
    return conn

def download_and_save_data(tickers, conn, start_date="2010-01-01"):
    """
    下載股價並存入資料庫，優化欄位缺失處理
    """
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{total}] 正在下載 {ticker} 的數據...")
        try:
            # 獲取數據，明確禁用 auto_adjust 以獲取 Adj Close
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            
            if df.empty:
                print(f"警告: {ticker} 沒有回傳數據。")
                continue
            
            # 1. 處理 yfinance 的 MultiIndex 欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            
            # 2. 標準化欄位名稱 (將空格替換為底線)
            df.columns = [str(col).replace(" ", "_") for col in df.columns]
            df['Symbol'] = ticker
            
            # 3. 容錯機制：如果缺少 Adj_Close，則使用 Close 代替
            if 'Adj_Close' not in df.columns:
                if 'Close' in df.columns:
                    print(f"提示: {ticker} 缺少 Adj_Close，將使用 Close 代替。")
                    df['Adj_Close'] = df['Close']
                else:
                    print(f"跳過 {ticker}: 同時缺少 Close 與 Adj_Close。")
                    continue
            
            # 4. 確保日期格式為字串
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # 5. 挑選並排序欄位
            required_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            
            # 再次檢查必要欄位
            existing_cols = [col for col in required_cols if col in df.columns]
            if len(existing_cols) < len(required_cols):
                missing = set(required_cols) - set(df.columns)
                print(f"跳過 {ticker}: 關鍵欄位缺失 {missing}")
                continue
                
            df_to_save = df[required_cols]
            
            # 6. 寫入資料庫
            df_to_save.to_sql('Daily_Prices', conn, if_exists='append', index=False)
            
            # 避免觸發頻率限制
            time.sleep(0.2)
            
        except Exception as e:
            print(f"錯誤: 無法處理 {ticker}。原因: {e}")

def save_constituents_info(df_current, conn):
    """
    將目前成分股的基本資訊存入資料庫
    """
    if df_current.empty: return
    
    df_current = df_current.copy()
    df_current.columns = [c.replace(' ', '_') for c in df_current.columns]
    
    mapping = {
        'Symbol': 'Symbol',
        'Security': 'Security',
        'GICS_Sector': 'GICS_Sector',
        'GICS_Sub-Industry': 'GICS_Sub_Industry',
        'Headquarters_Location': 'Headquarters_Location',
        'Date_added': 'Date_Added',
        'CIK': 'CIK'
    }
    df_current = df_current.rename(columns=mapping)
    df_current['Symbol'] = df_current['Symbol'].str.replace('.', '-')
    
    valid_cols = ['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry', 'Headquarters_Location', 'Date_Added', 'CIK']
    df_current = df_current[[c for c in valid_cols if c in df_current.columns]]
    
    df_current.to_sql('Constituents', conn, if_exists='replace', index=False)
    print("成分股基本資訊已更新。")

if __name__ == "__main__":
    db_file = "SP500.db"
    
    all_tickers, df_info = get_sp500_tickers()
    
    if not all_tickers:
        print("無法取得股票清單，程式終止。")
    else:
        connection = setup_database(db_file)
        try:
            save_constituents_info(df_info, connection)
            # 開始下載
            download_and_save_data(all_tickers, connection, start_date="2000-01-01")
            print(f"\n任務完成！所有數據已儲存至 {db_file}")
        finally:
            connection.close()