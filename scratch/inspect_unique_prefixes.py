import sqlite3
import os

db_path = "results/result.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT _path FROM strategy_summaries")
    paths = [r[0] for r in c.fetchall()]
    prefixes = sorted(list(set(p.split('/')[0] for p in paths if '/' in p)))
    print("Unique prefixes in strategy_summaries:")
    for pref in prefixes:
        c.execute("SELECT DATASET, count(*) FROM strategy_summaries WHERE _path LIKE ? GROUP BY DATASET", (pref + "/%",))
        datasets = c.fetchall()
        print(f"  {pref}:")
        for ds, count in datasets:
            print(f"    Dataset: {ds} -> {count} rows")
    conn.close()
else:
    print("Database does not exist")
