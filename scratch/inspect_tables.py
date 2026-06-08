import sqlite3
import os

for db_path in ["data/sp500_Current.db", "data/sp500.db", "results/result.db"]:
    if os.path.exists(db_path):
        print(f"--- Database: {db_path} ---")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print("Tables:", tables)
            for t in tables:
                if t in ["strategy_summaries", "strategy_pairs"]:
                    cursor.execute(f"SELECT count(*) FROM {t}")
                    print(f"  Table {t} has {cursor.fetchone()[0]} rows")
            conn.close()
        except Exception as e:
            print("Error:", e)
    else:
        print(f"Database {db_path} does not exist")
