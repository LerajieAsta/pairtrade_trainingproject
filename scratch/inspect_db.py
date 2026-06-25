import sqlite3
import pandas as pd
import numpy as np
import json

db = './results/result.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

# 1) List all tables
tabs = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("=== Tables in result.db ===")
for t in tabs:
    cnt = cur.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t[0]})").fetchall()]
    print(f"  {t[0]}: {cnt[0]} rows | cols: {cols}")

conn.close()
