import sys
sys.path.append('.')
import sqlite3
import pandas as pd
from strategies.preprocess_equity import DataProcessor
from strategies.config import DB_PATH, TABLE_NAME, FORMATION_WINDOW, BACKTEST_START, BACKTEST_END

print("DB_PATH:", DB_PATH)
processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)

# 1. Load data
price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
    BACKTEST_START, BACKTEST_END, FORMATION_WINDOW
)

conn = sqlite3.connect(DB_PATH)
df_memberships = pd.read_sql_query("SELECT Symbol, start_date, end_date FROM index_memberships", conn)
conn.close()

# Let's find window ending in 2016-04-13 (just before 2016-04-14)
# We want form_end_dt to be around 2016-04-13
dates_series = pd.Series(all_dates)
idx_list = dates_series[dates_series >= '2016-04-14'].index
if len(idx_list) > 0:
    idx = idx_list[0]
    form_start_idx = idx - FORMATION_WINDOW
    form_end_idx = idx - 1
    
    form_start_dt = all_dates[form_start_idx].strftime('%Y-%m-%d')
    form_end_dt = all_dates[form_end_idx].strftime('%Y-%m-%d')
    
    print(f"Target window: {form_start_dt} to {form_end_dt}")
    
    # Extract form prices
    form_prices_raw = price_pivot.iloc[form_start_idx:idx]
    print(f"Before filter, 'TIE' in columns? {'TIE' in form_prices_raw.columns}")
    
    # Apply filtering
    active_df = df_memberships[
        (df_memberships['start_date'] <= form_end_dt) & 
        ((df_memberships['end_date'].isna()) | (df_memberships['end_date'] >= form_end_dt))
    ]
    active_symbols = set(active_df['Symbol'].unique())
    valid_cols = [c for c in form_prices_raw.columns if c in active_symbols]
    form_prices_filtered = form_prices_raw[valid_cols]
    
    print(f"After filter, 'TIE' in columns? {'TIE' in form_prices_filtered.columns}")
    
    # For comparison, let's do the same for 2010 window
    idx_2010_list = dates_series[dates_series >= '2010-04-14'].index
    if len(idx_2010_list) > 0:
        idx_2010 = idx_2010_list[0]
        form_start_idx_2010 = idx_2010 - FORMATION_WINDOW
        form_end_idx_2010 = idx_2010 - 1
        form_end_dt_2010 = all_dates[form_end_idx_2010].strftime('%Y-%m-%d')
        
        form_prices_2010 = price_pivot.iloc[form_start_idx_2010:idx_2010]
        
        active_df_2010 = df_memberships[
            (df_memberships['start_date'] <= form_end_dt_2010) & 
            ((df_memberships['end_date'].isna()) | (df_memberships['end_date'] >= form_end_dt_2010))
        ]
        active_symbols_2010 = set(active_df_2010['Symbol'].unique())
        valid_cols_2010 = [c for c in form_prices_2010.columns if c in active_symbols_2010]
        form_prices_filtered_2010 = form_prices_2010[valid_cols_2010]
        
        print(f"For 2010 window, 'TIE' in columns before filter? {'TIE' in form_prices_2010.columns}")
        print(f"For 2010 window, 'TIE' in columns after filter? {'TIE' in form_prices_filtered_2010.columns}")
