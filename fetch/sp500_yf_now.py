import pandas as pd
import yfinance as yf
import sqlite3
import time
import requests
from datetime import datetime
import os

def get_sp500_tickers():
    """
    從 Wikipedia 抓取目前的 S&P 500 成份股。
    使用 requests 加入 User-Agent 以避免 403 Forbidden 錯誤。
    （已修改為僅保留當前成分股，不包含歷史剔除的股票）
    """
    print("正在從 Wikipedia 獲取 S&P 500 當前成份股清單...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        
        # tables[0] 是目前的成分股列表
        df_current = tables[0]
        current_tickers = df_current['Symbol'].tolist()
        
        # 將 ticker 中的 '.' 替換為 '-' 以符合 yfinance 格式 (例如 BRK.B -> BRK-B)
        current_tickers = [str(t).replace('.', '-') for t in current_tickers if pd.notna(t)]
        
        # 移除重複項以防萬一
        current_tickers = list(set(current_tickers))
        
        print(f"成功！共找到 {len(current_tickers)} 支當前成分股。")
        return current_tickers, df_current

    except Exception as e:
        print(f"獲取清單失敗: {e}")
        return [], pd.DataFrame()

def setup_database(db_name="SP500_Current.db"):
    """
    初始化 SQLite 資料庫結構，包含成分股資訊表與日K線價格表
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 建立成分股基本資訊表
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
    
    # 建立歷史股價表
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

def download_and_save_data(tickers, conn, start_date="2000-01-01"):
    """
    使用 yfinance 下載股價並存入資料庫，並包含欄位缺失的容錯處理
    """
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{total}] 正在下載 {ticker} 的數據...")
        try:
            # 獲取數據，明確禁用 auto_adjust 以獲取原始的 Adj Close
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            
            if df.empty:
                print(f"警告: {ticker} 沒有回傳數據。")
                continue
            
            # 1. 處理 yfinance 可能產生的 MultiIndex 欄位結構
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
            
            # 4. 確保日期格式為 YYYY-MM-DD 字串
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # 5. 挑選並排序要寫入的欄位
            required_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            
            # 再次檢查所有必要欄位是否存在
            existing_cols = [col for col in required_cols if col in df.columns]
            if len(existing_cols) < len(required_cols):
                missing = set(required_cols) - set(df.columns)
                print(f"跳過 {ticker}: 關鍵欄位缺失 {missing}")
                continue
                
            df_to_save = df[required_cols]
            
            # 6. 寫入 SQLite 資料庫 (使用 append 附加資料)
            df_to_save.to_sql('Daily_Prices', conn, if_exists='append', index=False)
            
            # 加入短暫延遲，避免過度頻繁請求觸發 Yahoo Finance 阻擋機制
            time.sleep(0.2)
            
        except Exception as e:
            print(f"錯誤: 無法處理 {ticker}。原因: {e}")

def save_constituents_info(df_current, conn):
    """
    將目前成分股的基本資訊整理後存入 SQLite 資料庫
    """
    if df_current.empty: return
    
    df_current = df_current.copy()
    # 將欄位名稱中的空格替換為底線
    df_current.columns = [c.replace(' ', '_') for c in df_current.columns]
    
    # 定義欄位映射以確保與資料表定義一致
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
    
    # 一併處理 Symbol 內的 '.' 符號
    df_current['Symbol'] = df_current['Symbol'].str.replace('.', '-')
    
    # 只保留我們資料庫有定義的有效欄位
    valid_cols = ['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry', 'Headquarters_Location', 'Date_Added', 'CIK']
    df_current = df_current[[c for c in valid_cols if c in df_current.columns]]
    
    # 寫入資料庫
    df_current.to_sql('Constituents', conn, if_exists='replace', index=False)
    print("當前成分股基本資訊已更新至資料庫。")

if __name__ == "__main__":
        # ==========================================
    # 精準路徑定位：與 fetch/ 同層的 dataset/ 資料夾
    # ==========================================
    try:
        # 1. 取得這支 python 檔案所在的資料夾 (例如: .../MyProject/fetch)
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    # 2. 取得專案根目錄 (即 fetch 的上一層，例如: .../MyProject)
    project_dir = os.path.dirname(script_dir)
    
    # 3. 指定目標資料夾為專案根目錄下的 dataset/price 資料夾（價格類資料庫）
    data_dir = os.path.join(project_dir, "dataset", "price")

    # 4. 如果資料夾不存在，自動建立它
    os.makedirs(data_dir, exist_ok=True)

    # 5. 將資料庫檔案路徑指到該資料夾內
    db_file = os.path.join(data_dir, "SP500_Current.db")

    # db_file = "SP500_Current.db" # 修改了預設資料庫名稱以便區分
    
    # 1. 取得當前成分股清單與資訊表
    current_tickers, df_info = get_sp500_tickers()
    
    if not current_tickers:
        print("無法取得股票清單，程式終止。")
    else:
        # 2. 建立資料庫連線
        connection = setup_database(db_file)
        try:
            # 3. 儲存基本資訊
            save_constituents_info(df_info, connection)
            # 4. 開始下載股價數據
            download_and_save_data(current_tickers, connection, start_date="2000-01-01")
            print(f"\n任務完成！所有當前成分股的數據已儲存至 {db_file}")
        finally:
            # 5. 確保程式結束時關閉資料庫連線
            connection.close()