import sqlite3, pandas as pd
conn = sqlite3.connect('results/result.db')
print(pd.read_sql_query('SELECT _path, DATASET FROM strategy_summaries WHERE DATASET="Tiingo" LIMIT 5', conn))
