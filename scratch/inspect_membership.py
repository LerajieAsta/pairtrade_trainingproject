import sqlite3

conn = sqlite3.connect('./dataset/sp500_yF.db')
cursor = conn.cursor()

# Get distinct symbols from Daily_Prices
cursor.execute("SELECT DISTINCT Symbol FROM Daily_Prices")
dp_symbols = {row[0] for row in cursor.fetchall()}

# Get distinct symbols from index_memberships
cursor.execute("SELECT DISTINCT Symbol FROM index_memberships")
im_symbols = {row[0] for row in cursor.fetchall()}

print(f"Number of distinct symbols in Daily_Prices: {len(dp_symbols)}")
print(f"Number of distinct symbols in index_memberships: {len(im_symbols)}")

mismatch = im_symbols - dp_symbols
print(f"Number of index_memberships symbols NOT in Daily_Prices: {len(mismatch)}")
if mismatch:
    print("Sample mismatches:", list(mismatch)[:10])

mismatch_2 = dp_symbols - im_symbols
print(f"Number of Daily_Prices symbols NOT in index_memberships: {len(mismatch_2)}")
if mismatch_2:
    print("Sample Daily_Prices symbols NOT in index_memberships:", list(mismatch_2)[:10])

conn.close()
