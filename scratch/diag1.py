import pandas as pd
import numpy as np

# === 診斷 1：HDBSCAN_CS_UMAP 的天文數字問題 ===
df = pd.read_csv('results/yFinance/HDBSCAN_CS_UMAP/TradeLogs_Top10_SL0_ZWin0_MSR0.csv', parse_dates=['Date'])

# 找 Trade_PnL 最大的那幾筆
huge = df[df['Trade_PnL'].abs() > 50000][['Date','Ticker_A','Ticker_B','Period_Start','Price_A','Price_B','Hedge_Ratio','Position','Trade_PnL','Realized_PnL','Status']].sort_values('Trade_PnL', key=abs, ascending=False)
print('=== HDBSCAN_CS_UMAP: 超大 Trade_PnL ===')
print(huge.head(10).to_string())
print()

# 顯示這些配對的進場前後狀況
checked = set()
for _, row in huge.head(5).iterrows():
    key = (row['Ticker_A'], row['Ticker_B'], row['Period_Start'])
    if key in checked:
        continue
    checked.add(key)
    
    mask = (df['Ticker_A'] == row['Ticker_A']) & (df['Ticker_B'] == row['Ticker_B']) & (df['Period_Start'] == row['Period_Start'])
    subset = df[mask].reset_index(drop=True)
    
    # Show first entry and some context
    enter_rows = subset[subset['Status'].str.startswith('ENTER')]
    if len(enter_rows) > 0:
        eidx = enter_rows.index[0]
        window = subset.loc[max(0, eidx-1):min(len(subset)-1, eidx+5)]
        ta, tb = row['Ticker_A'], row['Ticker_B']
        ps = row['Period_Start']
        print(f"--- {ta}/{tb} period={ps} ---")
        print(window[['Date','Price_A','Price_B','Hedge_Ratio','ZScore','Position','Unrealized_PnL','Realized_PnL','Status']].to_string())
        print()

# === 診斷 2：HDBSCAN_Macro_UMAP - 完全爆掉的策略 ===
print("\n=== HDBSCAN_Macro_UMAP: 最大 Trade_PnL 分佈 ===")
df2 = pd.read_csv('results/yFinance/HDBSCAN_Macro_UMAP/TradeLogs_Top10_SL0_ZWin0_MSR0.csv', parse_dates=['Date'])
huge2 = df2[df2['Trade_PnL'].abs() > 1e9][['Date','Ticker_A','Ticker_B','Period_Start','Price_A','Price_B','Hedge_Ratio','Trade_PnL','Status']]
print(f"Trade_PnL > 1e9: {len(huge2)} rows")
print(huge2.head(5).to_string())

# Check Hedge_Ratio distribution for Macro_UMAP
print("\nHedge_Ratio stats for Macro_UMAP:")
print(df2['Hedge_Ratio'].describe())
print("Extreme hedge ratios:")
print(df2[df2['Hedge_Ratio'].abs() > 100]['Hedge_Ratio'].describe())
