import sqlite3
import pandas as pd

conn = sqlite3.connect('./dataset/sp500_yF.db')
df = pd.read_sql_query("SELECT Date, Open, High, Low, Close, Adj_Close, Volume FROM Daily_Prices WHERE Symbol='TIE' AND Date BETWEEN '2015-10-01' AND '2015-11-01'", conn)
print("sp500_yF.db TIE prices:")
print(df.to_string())

# Also check sp500_Tiingo.db (if it has TIE prices)
conn_tiingo = sqlite3.connect('./dataset/sp500_Tiingo.db')
df_tiingo = pd.read_sql_query("SELECT Date, Open, High, Low, Close, Adj_Close, Volume FROM Daily_Prices WHERE Symbol='TIE' AND Date BETWEEN '2015-10-01' AND '2015-11-01'", conn_tiingo)
print("\nsp500_Tiingo.db TIE prices:")
print(df_tiingo.to_string())

conn.close()
conn_tiingo.close()
