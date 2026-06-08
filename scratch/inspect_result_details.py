import sqlite3
import os

db_path = "results/result.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT _path, METHOD, DATASET FROM strategy_summaries")
    rows = c.fetchall()
    print("Unique paths, methods, datasets in database:")
    for path, method, dataset in sorted(rows):
        # Only print prefix of path to be clean
        prefix = path.split('/')[0] if '/' in path else path
        print(f"  Prefix: {prefix:<50} | Method: {method:<20} | Dataset: {dataset}")
    conn.close()
else:
    print("Database does not exist")
