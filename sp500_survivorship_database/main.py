import random
from models import get_engine, get_session, init_db, Ticker, IndexMembership, DailyPrice
from history_scraper import scrape_sp500_history
from price_downloader import download_prices

def run_integration_test():
    print("=== S&P 500 Survivorship-Bias-Free Database Pipeline ===")
    
    # 1. Init DB
    print("\n[Step 1] Initializing Database Models...")
    engine = get_engine()
    init_db(engine)
    
    # 2. Run Scraper
    print("\n[Step 2] Running Wikipedia Scraper...")
    try:
        scrape_sp500_history()
    except Exception as e:
        print(f"Scraper failed: {e}")
        return

    # 3. Select test tickers
    print("\n[Step 3] Selecting sample tickers for price download...")
    session = get_session(engine)
    
    # Find current constituents (end_date is null or > today)
    current_members = session.query(IndexMembership.ticker).filter(IndexMembership.end_date == None).distinct().all()
    current_tickers = [m[0] for m in current_members]
    
    # Find past constituents (end_date is not null)
    past_members = session.query(IndexMembership.ticker).filter(IndexMembership.end_date != None).distinct().all()
    past_tickers = [m[0] for m in past_members if m[0] not in current_tickers]
    
    sample_current = random.sample(current_tickers, min(5, len(current_tickers)))
    sample_past = random.sample(past_tickers, min(2, len(past_tickers)))
    
    test_tickers = sample_current + sample_past
    print(f"Current Constituents Sample: {sample_current}")
    print(f"Removed Constituents Sample: {sample_past}")
    
    # 4. Download Prices
    print("\n[Step 4] Downloading historical prices for sample tickers...")
    download_prices(tickers_to_fetch=test_tickers, start_date='2000-01-01')
    
    # 5. Verification
    print("\n[Step 5] Verifying Downloaded Data & Delisting Status...")
    print("Verification Results:")
    for ticker in test_tickers:
        t_obj = session.query(Ticker).filter_by(ticker=ticker).first()
        price_count = session.query(DailyPrice).filter_by(ticker=ticker).count()
        print(f" - {ticker}: {price_count} price records found. Is_Delisted = {t_obj.is_delisted}")
        
    session.close()
    
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    run_integration_test()
