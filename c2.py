def load_data_from_db(db_path, start_date, end_date):
    """Load price and sector data from SQLite."""
    conn = sqlite3.connect(db_path)
    price_queries = [
        (f"SELECT date, ticker, open, high, low, adj_close AS close, volume FROM daily_prices "
         f"WHERE date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date"),
        (f"SELECT date, ticker, open, high, low, close, volume FROM stock_prices "
         f"WHERE date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date"),
    ]
    prices_df = None
    for q in price_queries:
        try:
            prices_df = pd.read_sql_query(q, conn, parse_dates=['date'])
            if len(prices_df) > 0:
                print(f'Price rows: {len(prices_df):,}')
                break
        except Exception:
            continue
    if prices_df is None or len(prices_df) == 0:
        tbls = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        conn.close()
        raise RuntimeError(f"No price table. Tables: {tbls['name'].tolist()}")
    sector_queries = [
        "SELECT ticker, sector FROM tickers",
        "SELECT ticker, sector FROM sp500_components GROUP BY ticker",
    ]
    sector_df = None
    for q in sector_queries:
        try:
            sector_df = pd.read_sql_query(q, conn)
            if len(sector_df) > 0:
                print(f'Sector rows: {len(sector_df):,}')
                break
        except Exception:
            continue
    conn.close()
    if sector_df is None or len(sector_df) == 0:
        print('WARNING: No sector table, using Unknown.')
        sector_df = pd.DataFrame({'ticker': prices_df['ticker'].unique(), 'sector': 'Unknown'})
        
    # [Mod] Mitigate Survivorship Bias: Check if delisted components exist
    max_date = prices_df['date'].max()
    max_dates = prices_df.groupby('ticker')['date'].max()
    delisted_count = (max_dates < max_date - pd.Timedelta(days=30)).sum()
    if delisted_count < 10:
        import logging
        logging.warning("STRICT RESEARCH LIMITATION: Database lacks historically delisted S&P 500 components. Survivorship Bias is present.")
        
    return prices_df, sector_df



def fix_unknown_sectors(sector_df, use_dynamic=True, save_path=r'data\imputed_sectors.csv'):
    """具備本機快取與全域開關控制的產業補齊模組"""
    # 1. 開關判斷：如果不使用動態補齊，直接原封不動回傳
    if not use_dynamic:
        print("不使用動態產業補齊，維持原始 Unknown 分類作為對照組。")
        return sector_df

    # 確保儲存的目錄 (data\) 存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 2. 快取讀取：如果已經抓過並存檔，直接載入
    if os.path.exists(save_path):
        print(f"從本機快取載入已補齊的產業分類: {save_path}")
        cached_df = pd.read_csv(save_path)
        update_df = cached_df.set_index('ticker')
        sector_df = sector_df.set_index('ticker')
        sector_df.update(update_df)
        return sector_df.reset_index()

    # 3. API 抓取：如果沒有快取，執行連線作業
    unknown_mask = sector_df['sector'] == 'Unknown'
    unknown_tickers = sector_df[unknown_mask]['ticker'].tolist()
    
    if not unknown_tickers:
        return sector_df

    print(f"找不到本機快取，正在透過 API 補齊 {len(unknown_tickers)} 檔 Unknown 股票的產業分類...")
    
    yf_logger = logging.getLogger('yfinance')
    original_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL) 
    
    fixed_sectors = []
    
    for i, ticker in enumerate(unknown_tickers):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            fixed_sectors.append({'ticker': ticker, 'sector': sector})
            time.sleep(0.02) 
        except Exception:
            fixed_sectors.append({'ticker': ticker, 'sector': 'Unknown'})
            
        if (i + 1) % 50 == 0:
            print(f"已處理 {i + 1} / {len(unknown_tickers)}...")
            
    yf_logger.setLevel(original_level)
    
    # 4. 儲存快取：將剛抓下來的資料存成 CSV，下次就不用再抓了
    fetched_df = pd.DataFrame(fixed_sectors)
    fetched_df.to_csv(save_path, index=False)
    print(f"API 抓取完畢！已將動態產業分類永久儲存至: {save_path}")
    
    # 更新回原本的 DataFrame
    update_df = fetched_df.set_index('ticker')
    sector_df = sector_df.set_index('ticker')
    sector_df.update(update_df)
    sector_df = sector_df.reset_index()
    
    remaining = len(sector_df[sector_df['sector'] == 'Unknown'])
    print(f"補齊完成！剩餘真實無法識別(已下市)的 Unknown 股票數量: {remaining}")
    
    return sector_df

def preprocess_prices(prices_df, min_days=MIN_HISTORY_DAYS):
    """Pivot, forward-fill, drop sparse tickers."""
    pivot = prices_df.pivot_table(index='date', columns='ticker', values='close', aggfunc='last')
    pivot.index = pd.to_datetime(pivot.index)
    pivot.sort_index(inplace=True)
    pivot.ffill(limit=5, inplace=True)
    valid = pivot.columns[pivot.notna().sum() >= min_days]
    pivot = pivot[valid]
    print(f'Matrix: {len(pivot)} days x {len(pivot.columns)} tickers')

    # [Mod] Integrate Macroeconomic Indicator (VIX)
    import yfinance as yf
    print("Fetching VIX data...")
    try:
        vix_data = yf.download("^VIX", start=pivot.index.min(), end=pivot.index.max() + pd.Timedelta(days=1), progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_close = vix_data['Close'].squeeze()
        else:
            vix_close = vix_data['Close']
        vix_df = pd.DataFrame({'VIX': vix_close})
        vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
        # [Mod] Shift VIX by 1 day to completely prevent look-ahead bias
        vix_shifted = vix_df.shift(1)
        vix_aligned = vix_shifted.reindex(pivot.index).ffill()
        print("VIX feature matrix aligned.")
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        vix_aligned = pd.DataFrame(index=pivot.index, columns=['VIX'])

    return pivot, vix_aligned
