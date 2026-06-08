import sqlite3
import os

def check_db_constituents(db_path):
    if not os.path.exists(db_path):
        print(f"File {db_path} does not exist!")
        return
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Constituents'")
        exists = cursor.fetchone()
        if exists:
            print(f"\nTable 'Constituents' exists in {db_path}.")
            cursor.execute("PRAGMA table_info(Constituents)")
            cols = cursor.fetchall()
            print("Columns in Constituents:")
            for col in cols:
                print(f"  {col[1]} ({col[2]})")
            
            cursor.execute("SELECT count(*) FROM Constituents")
            print(f"Total rows: {cursor.fetchone()[0]}")
            
            cursor.execute("SELECT Symbol, GICS_Sector FROM Constituents LIMIT 5")
            print("First 5 rows:")
            for row in cursor.fetchall():
                print(f"  {row}")
        else:
            print(f"\nTable 'Constituents' DOES NOT exist in {db_path}.")
        conn.close()
    except Exception as e:
        print(f"Error checking {db_path}: {e}")

if __name__ == "__main__":
    check_db_constituents("data/sp500.db")
    check_db_constituents("data/SP500_Current.db")
