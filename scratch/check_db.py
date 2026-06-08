import sqlite3
src = sqlite3.connect("data/sp500_Current.db")
c = src.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("Tables in source DB:", tables)
for t in ["strategy_summaries", "trade_logs", "strategy_pairs"]:
    if t in tables:
        c.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t}: {c.fetchone()[0]} rows")
src.close()

dest = sqlite3.connect("results/result.db")
d = dest.cursor()
d.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables_dest = [r[0] for r in d.fetchall()]
print("Tables in result.db:", tables_dest)
for t in ["strategy_summaries", "trade_logs", "strategy_pairs"]:
    if t in tables_dest:
        d.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t}: {d.fetchone()[0]} rows")
dest.close()
