import pandas as pd
import sqlite3
import time
import requests
import os
from datetime import datetime, timedelta

# ==========================================
# 請在此填入您在 Tiingo 註冊取得的 API Key
# 註冊網址: https://api.tiingo.com/
# ==========================================
TIINGO_API_KEY = "432374758dbd3bdcad66bf3cf990393b2fd37579"

def get_sp500_tickers():
    """
    從 Wikipedia 抓取目前的 S&P 500 成份股以及歷史變動表。
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
        
        if 'Added' in df_adjustments.columns:
            added_tickers = df_adjustments['Added']['Ticker'].dropna().tolist() if isinstance(df_adjustments['Added'], pd.DataFrame) else df_adjustments['Added'].dropna().tolist()
        if 'Removed' in df_adjustments.columns:
            removed_tickers = df_adjustments['Removed']['Ticker'].dropna().tolist() if isinstance(df_adjustments['Removed'], pd.DataFrame) else df_adjustments['Removed'].dropna().tolist()
        
        all_tickers = list(set(current_tickers + added_tickers + removed_tickers))
        # Tiingo 也能接受 BRK-B 這種格式
        all_tickers = [str(t).replace('.', '-') for t in all_tickers if pd.notna(t)]
        
        print(f"成功！共找到 {len(all_tickers)} 支歷年成分股。")
        return all_tickers, df_current

    except Exception as e:
        print(f"獲取清單失敗: {e}")
        return [], pd.DataFrame()

def setup_database(db_name="SP500Full.db"):
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

def download_and_save_data(tickers, conn, start_date="2000-01-01", end_date=None):
    """
    使用 Tiingo API 下載股價並存入資料庫。
    同時支援「向上更新」(Forward-fill) 與「向下補償」(Backfill)。
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {TIINGO_API_KEY}'
    }
    
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        cursor = conn.cursor()
        
        # 取得目前資料庫中的日期區間
        cursor.execute("SELECT MIN(Date), MAX(Date) FROM Daily_Prices WHERE Symbol=?", (ticker,))
        min_date_str, max_date_str = cursor.fetchone()
        
        tasks = []
        
        # 邏輯 A: 回溯補償 (Backfill) - 如果現存最早日期比目標 start_date 還晚
        if min_date_str and min_date_str > start_date:
            backfill_end = (datetime.strptime(min_date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            tasks.append((start_date, backfill_end, "回溯歷史"))
            
        # 邏輯 B: 向上更新 (Forward-fill) - 如果現存最晚日期比 end_date 還早
        if max_date_str:
            if max_date_str < end_date:
                forward_start = (datetime.strptime(max_date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                if forward_start <= end_date:
                    tasks.append((forward_start, end_date, "同步新資料"))
        else:
            # 邏輯 C: 全新下載
            tasks.append((start_date, end_date, "首次下載"))

        for f_start, f_end, task_type in tasks:
            if f_start > f_end: continue
            
            print(f"[{i+1}/{total}] {ticker} {task_type}: {f_start} -> {f_end}")
            
            try:
                # Tiingo 允許同時指定 startDate 與 endDate
                url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={f_start}&endDate={f_end}"
                response = requests.get(url, headers=headers)
           
                if response.status_code == 404:
                    print(f"提示: Tiingo 找不到 {ticker} 的數據 (可能已下市或代碼變更)。")
                    continue
                elif response.status_code == 429 or response.status_code == 403:
                    print(f"\n警告: 似乎已經達到 Tiingo 的 API 請求限制 (狀態碼: {response.status_code})。")
                    print("您可以隨時終止程式，下個月再執行會自動從斷點續傳！")
                    # 遇到限制時提早中結，避免無限噴錯
                    break 
                elif response.status_code != 200:
                    print(f"警告: {ticker} 獲取失敗 (狀態碼: {response.status_code}) - {response.text}")
                    continue
                
                data = response.json()
                if not data: continue
                               
                df = pd.DataFrame(data)
                
                # 先轉換日期格式（必須在 rename 之前）
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # 將 Tiingo 的欄位名稱轉換為符合資料庫的 Schema
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'adjClose': 'Adj_Close',
                    'volume': 'Volume'
                })
                
                df['Symbol'] = ticker
                df_to_save = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']]
                
                df_to_save.to_sql('Daily_Prices', conn, if_exists='append', index=False)
                time.sleep(0.1) # 頻率限制緩衝
                
            except Exception as e:
                print(f"錯誤: 無法處理 {ticker} ({task_type})。原因: {e}")
            
        # 避免觸發 API Rate Limit
        time.sleep(0.2)

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
    # ==========================================
    # 精準路徑定位：與 fetch/ 同層的 data/ 資料夾
    # ==========================================
    try:
        # 1. 取得這支 python 檔案所在的資料夾 (例如: .../MyProject/fetch)
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    # 2. 取得專案根目錄 (即 fetch 的上一層，例如: .../MyProject)
    project_dir = os.path.dirname(script_dir)
    
    # 3. 指定目標資料夾為專案根目錄下的 data 資料夾 (例如: .../MyProject/data)
    data_dir = os.path.join(project_dir, "data")
    
    # 4. 如果資料夾不存在，自動建立它
    os.makedirs(data_dir, exist_ok=True)
    
    # 5. 將資料庫檔案路徑指到該資料夾內
    db_file = os.path.join(data_dir, "sp500Full.db")
    
    abs_db_path = os.path.abspath(db_file)
    print(f"\n=======================================================")
    print(f"📁 系統準備將資料庫儲存於:")
    print(f"👉 {abs_db_path}")
    print(f"=======================================================\n")
    
    all_tickers, df_info = get_sp500_tickers()
    
    if not all_tickers:
        print("無法取得股票清單，程式終止。")
    else:
        # 改用絕對路徑 abs_db_path 連線資料庫
        connection = setup_database(abs_db_path)
        try:
            save_constituents_info(df_info, connection)
            # 開始下載 (改為從 2000 年起抓取)
            download_and_save_data(all_tickers, connection, start_date="2000-01-01", end_date="2025-12-31")
            print(f"\n任務完成！所有數據已儲存至 {abs_db_path}")
        finally:
            connection.close()