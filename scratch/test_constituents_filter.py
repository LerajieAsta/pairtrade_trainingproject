import sqlite3
import pandas as pd

# Load memberships
db_path = './dataset/sp500_yF.db'
conn = sqlite3.connect(db_path)
df_memberships = pd.read_sql_query("SELECT Symbol, start_date, end_date FROM index_memberships", conn)
conn.close()

def get_active_constituents(date_str):
    active_df = df_memberships[
        (df_memberships['start_date'] <= date_str) & 
        ((df_memberships['end_date'].isna()) | (df_memberships['end_date'] >= date_str))
    ]
    return set(active_df['Symbol'].unique())

# Test case 1: 2010-04-14
active_2010 = get_active_constituents('2010-04-14')
print(f"2010-04-14: TIE in active? {'TIE' in active_2010}")

# Test case 2: 2016-04-14 (should be False)
active_2016 = get_active_constituents('2016-04-14')
print(f"2016-04-14: TIE in active? {'TIE' in active_2016}")

# Also test compiling run_formation.py to make sure no syntax errors
try:
    import py_compile
    py_compile.compile('run_formation.py')
    print("run_formation.py compiled successfully!")
except Exception as e:
    print(f"Error compiling run_formation.py: {e}")
