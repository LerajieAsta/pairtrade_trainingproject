import sqlite3
import os

db_path = "results/result.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DATASET, METHOD, COUNT(*) FROM strategy_summaries GROUP BY DATASET, METHOD")
        rows = cursor.fetchall()
        print("Dataset | Method | Count")
        print("-" * 40)
        for row in rows:
            print(f"{row[0]} | {row[1]} | {row[2]}")
            
        print("\nPaths containing SSD:")
        cursor.execute("SELECT DISTINCT _path FROM strategy_summaries WHERE _path LIKE '%SSD%'")
        ssd_paths = cursor.fetchall()
        for p in ssd_paths:
            print(p[0])
    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
else:
    print("DB does not exist")
