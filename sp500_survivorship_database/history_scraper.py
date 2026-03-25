import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from models import get_engine, get_session, init_db, Ticker, IndexMembership

def scrape_sp500_history():
    engine = get_engine()
    init_db(engine)
    session = get_session(engine)
    
    print("Fetching Wikipedia data...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    
    df_current = tables[0]
    df_changes = tables[1]
    
    print("Processing current components...")
    active_intervals = {}
    completed_intervals = []
    all_tickers_info = {}
    
    for _, row in df_current.iterrows():
        ticker = str(row['Symbol']).strip().replace('.', '-')
        name = str(row['Security']).strip()
        sector = str(row['GICS Sector']).strip()
        
        all_tickers_info[ticker] = {'company_name': name, 'sector': sector}
        active_intervals[ticker] = {'end_date': None}
        
    print("Reverse engineering historical changes...")
    df_changes.columns = ['Date', 'Added_Ticker', 'Added_Name', 'Removed_Ticker', 'Removed_Name', 'Reason']
    df_changes['Date'] = pd.to_datetime(df_changes['Date'], errors='coerce')
    df_changes = df_changes.dropna(subset=['Date'])
    df_changes = df_changes.sort_values(by='Date', ascending=False)
    
    for _, row in df_changes.iterrows():
        date_val = row['Date'].date()
        
        added_ticker = str(row['Added_Ticker']).strip().replace('.', '-') if pd.notna(row['Added_Ticker']) and str(row['Added_Ticker']).strip() not in ['', 'None', 'nan'] else None
        removed_ticker = str(row['Removed_Ticker']).strip().replace('.', '-') if pd.notna(row['Removed_Ticker']) and str(row['Removed_Ticker']).strip() not in ['', 'None', 'nan'] else None
        
        added_name = str(row['Added_Name']).strip() if pd.notna(row['Added_Name']) else "Unknown"
        removed_name = str(row['Removed_Name']).strip() if pd.notna(row['Removed_Name']) else "Unknown"
        
        if added_ticker and added_ticker in active_intervals:
            interval = active_intervals.pop(added_ticker)
            interval['start_date'] = date_val
            interval['ticker'] = added_ticker
            completed_intervals.append(interval)
            
            if added_ticker not in all_tickers_info:
                all_tickers_info[added_ticker] = {'company_name': added_name, 'sector': 'Unknown'}

        if removed_ticker:
            if removed_ticker not in active_intervals:
                active_intervals[removed_ticker] = {'end_date': date_val}
            
            if removed_ticker not in all_tickers_info:
                all_tickers_info[removed_ticker] = {'company_name': removed_name, 'sector': 'Unknown'}
                
    print("Closing remaining intervals with default start date (2000-01-01)...")
    default_start = datetime(2000, 1, 1).date()
    for ticker, interval in active_intervals.items():
        interval['start_date'] = default_start
        interval['ticker'] = ticker
        completed_intervals.append(interval)
        
    print("Writing to database...")
    for ticker, info in all_tickers_info.items():
        existing = session.query(Ticker).filter_by(ticker=ticker).first()
        if not existing:
            new_ticker = Ticker(
                ticker=ticker,
                company_name=info['company_name'],
                sector=info['sector'],
                is_delisted=False
            )
            session.add(new_ticker)
        else:
            if existing.sector == 'Unknown' and info['sector'] != 'Unknown':
                existing.sector = info['sector']
    
    session.commit()
    
    session.query(IndexMembership).delete()
    
    for interval in completed_intervals:
        if interval['end_date'] is not None and interval['start_date'] > interval['end_date']:
            interval['start_date'], interval['end_date'] = interval['end_date'], interval['start_date']
            
        mem = IndexMembership(
            ticker=interval['ticker'],
            start_date=interval['start_date'],
            end_date=interval['end_date']
        )
        session.add(mem)
        
    session.commit()
    session.close()
    
    print(f"Success! Inserted {len(all_tickers_info)} unique tickers and {len(completed_intervals)} membership intervals.")
    
if __name__ == '__main__':
    scrape_sp500_history()
