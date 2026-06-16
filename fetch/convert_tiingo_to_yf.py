import sqlite3
import pandas as pd
import os
import time

def convert_tiingo_database_pandas():
    project_dir = r"d:\Unknown\Papper\Code"
    data_dir = os.path.join(project_dir, "data")
    
    db_yf = os.path.join(data_dir, "sp500.db")
    db_tg = os.path.join(data_dir, "sp500Full.db")
    db_out = os.path.join(data_dir, "sp500_Tiingo_Converted.db")
    
    print("=== Lock-Free Tiingo to yFinance Schema Database Converter ===")
    print(f"yFinance (Source Structure): {db_yf}")
    print(f"Tiingo (Source Data):        {db_tg}")
    print(f"Target Output Database:      {db_out}")
    
    if not os.path.exists(db_yf):
        print(f"Error: Source yFinance database '{db_yf}' not found.")
        return
    if not os.path.exists(db_tg):
        print(f"Error: Source Tiingo database '{db_tg}' not found.")
        return
        
    if os.path.exists(db_out):
        print(f"Removing existing output database: {db_out}")
        try:
            os.remove(db_out)
        except Exception as e:
            print(f"Error removing output db: {e}")
            # If locked, try to rename to v2
            db_out = os.path.join(data_dir, "sp500_Tiingo_Converted_v2.db")
            print(f"Falling back to alternative database: {db_out}")
            if os.path.exists(db_out):
                try:
                    os.remove(db_out)
                except Exception as ex:
                    print(f"Error removing fallback db: {ex}")
                    return
            
    start_time = time.time()
    
    # 1. Read metadata tables from yFinance db and write them to output db
    print("Reading metadata from yFinance database...")
    try:
        conn_yf = sqlite3.connect(db_yf)
        df_const = pd.read_sql_query("SELECT * FROM Constituents;", conn_yf)
        df_im = pd.read_sql_query("SELECT * FROM index_memberships;", conn_yf)
        conn_yf.close()
        print(f"Metadata read successfully. Constituents: {len(df_const)}, Index Memberships: {len(df_im)}")
    except Exception as e:
        print(f"Error reading yFinance metadata: {e}")
        return
        
    print("Writing metadata to new database...")
    try:
        conn_out = sqlite3.connect(db_out)
        df_const.to_sql("Constituents", conn_out, if_exists="replace", index=False)
        df_im.to_sql("index_memberships", conn_out, if_exists="replace", index=False)
        
        # Create target table structure with Primary Key
        cursor_out = conn_out.cursor()
        cursor_out.execute('''
            CREATE TABLE Daily_Prices (
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
        conn_out.commit()
    except Exception as e:
        print(f"Error creating metadata tables in output database: {e}")
        if 'conn_out' in locals():
            conn_out.close()
        return

    # 2. Read Daily_Prices from Tiingo db in chunks, convert, and write to output db
    print("Starting chunked copy of price data from Tiingo db...")
    try:
        conn_tg = sqlite3.connect(db_tg)
        
        # Disable journal for target to maximize write speed
        cursor_out.execute("PRAGMA journal_mode=WAL;")
        cursor_out.execute("PRAGMA synchronous=OFF;")
        
        chunksize = 200000
        query = "SELECT Date, Symbol, Open, High, Low, Close, Adj_Close, Volume FROM Daily_Prices;"
        
        total_rows = 0
        chunk_idx = 0
        
        for chunk in pd.read_sql_query(query, conn_tg, chunksize=chunksize):
            chunk_idx += 1
            chunk_start = time.time()
            
            # Perform conversion in Python
            # Ratio = Adj_Close / Close
            # Handle zeros and NaNs safely
            ratio = chunk['Adj_Close'] / chunk['Close']
            ratio = ratio.fillna(1.0)
            ratio = ratio.replace([float('inf'), float('-inf')], 1.0)
            
            chunk['Open'] = chunk['Open'] * ratio
            chunk['High'] = chunk['High'] * ratio
            chunk['Low'] = chunk['Low'] * ratio
            chunk['Close'] = chunk['Adj_Close']
            
            # Inverse ratio for volume
            inv_ratio = 1.0 / ratio
            inv_ratio = inv_ratio.fillna(1.0)
            inv_ratio = inv_ratio.replace([float('inf'), float('-inf')], 1.0)
            chunk['Volume'] = (chunk['Volume'] * inv_ratio).fillna(0).astype('int64')
            
            # Write to output db
            chunk.to_sql("Daily_Prices", conn_out, if_exists="append", index=False)
            
            total_rows += len(chunk)
            print(f"  Processed chunk {chunk_idx}: {len(chunk)} rows. Cumulative: {total_rows} rows. (Time: {time.time() - chunk_start:.2f}s)")
            
        conn_tg.close()
        
        # 3. Create index for backtesting speed
        print("Creating index for Daily_Prices...")
        cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date ON Daily_Prices (Symbol, Date);")
        conn_out.commit()
        
    except Exception as e:
        print(f"Error during price data conversion: {e}")
    finally:
        conn_out.close()
        
    # Final check: rename to original name if we fell back to v2
    if "sp500_Tiingo_Converted_v2.db" in db_out:
        original_db = os.path.join(data_dir, "sp500_Tiingo_Converted.db")
        print(f"Attempting to rename fallback db back to original target: {original_db}")
        try:
            if os.path.exists(original_db):
                os.remove(original_db)
            os.rename(db_out, original_db)
            print("Successfully renamed back to original name.")
        except Exception as e:
            print(f"Could not rename back to original name due to file lock: {e}")
            print(f"Please use alternative database: {db_out}")
        
    elapsed = time.time() - start_time
    print(f"\nAll done! Converted {total_rows} rows. Total elapsed time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    convert_tiingo_database_pandas()
