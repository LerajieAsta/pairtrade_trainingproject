import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from sqlalchemy.exc import IntegrityError

from models import get_engine, get_session, Ticker, DailyPrice

def download_prices(tickers_to_fetch=None, start_date='2000-01-01'):
    engine = get_engine()
    session = get_session(engine)
    
    if tickers_to_fetch is None:
        tickers = session.query(Ticker).all()
    else:
        tickers = session.query(Ticker).filter(Ticker.ticker.in_(tickers_to_fetch)).all()
        
    print(f"Starting download for {len(tickers)} tickers from {start_date}...")
    
    for t_obj in tickers:
        ticker_sym = t_obj.ticker
        
        # Yahoo finance sometimes uses different ticker format (e.g. replacing '-' with '')
        # but standard yfinance works well with '-'
        # Example: BRK.B on wiki might be BRK-B on yahoo
        yf_ticker = ticker_sym
        
        print(f"[{ticker_sym}] Fetching data...")
        try:
            # Download data
            stock = yf.Ticker(yf_ticker)
            df = stock.history(start=start_date, auto_adjust=False)
            
            if df.empty:
                print(f"[{ticker_sym}] No data found. Marking as potentially delisted.")
                t_obj.is_delisted = True
                session.commit()
                continue
                
            # Keep records to insert
            records = []
            
            df = df.reset_index()
            # Ensure timezone-naive dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            for _, row in df.iterrows():
                try:
                    price_record = DailyPrice(
                        ticker=ticker_sym,
                        date=row['Date'],
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        adj_close=float(row.get('Adj Close', row['Close'])),
                        volume=int(row['Volume'])
                    )
                    records.append(price_record)
                except Exception as row_error:
                    pass # Skip problematic rows
                    
            # Insert records
            try:
                # To handle duplicate insertions easily during testing, we'll try to merge/ignore duplicates
                # For simplicity in loop, add each and commit 
                for rec in records:
                    session.merge(rec)
                session.commit()
                print(f"[{ticker_sym}] Successfully saved {len(records)} daily records.")
            except IntegrityError:
                session.rollback()
                print(f"[{ticker_sym}] Integrity error when inserting. Partial data might exist.")
                
            # Always reset delisted status if data is successfully fetched
            if t_obj.is_delisted:
                t_obj.is_delisted = False
                session.commit()
                
            # Sleep slightly to avoid rate limiting
            time.sleep(0.5)
                
        except Exception as e:
            print(f"[{ticker_sym}] Exception during fetch: {e}. Marking as delisted.")
            t_obj.is_delisted = True
            session.commit()
            
    session.close()
    print("Download process completed.")

if __name__ == '__main__':
    download_prices()
