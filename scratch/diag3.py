"""
診斷 HDBSCAN_CS_MF 的單筆 170,379 Trade_PnL
以及 TIE 股票資料問題（UMAP 爆掉的根源）
"""
import pandas as pd
import numpy as np

# === CS_MF 異常單筆 ===
df = pd.read_csv('results/yFinance/HDBSCAN_CS_MF/TradeLogs_Top10_SL0_ZWin0_MSR0.csv', parse_dates=['Date'])
huge = df[df['Trade_PnL'].abs() > 10000].sort_values('Trade_PnL', key=abs, ascending=False)
print("=== HDBSCAN_CS_MF: 大 Trade_PnL ===")
print(huge[['Date','Ticker_A','Ticker_B','Period_Start','Price_A','Price_B','Hedge_Ratio','Trade_PnL','Status']].head(5).to_string())

# Show entry for biggest case
row = huge.iloc[0]
mask = (df['Ticker_A']==row['Ticker_A']) & (df['Ticker_B']==row['Ticker_B']) & (df['Period_Start']==row['Period_Start'])
pair_df = df[mask].reset_index(drop=True)
enter_idx = pair_df[pair_df['Status'].str.startswith('ENTER')].index
if len(enter_idx) > 0:
    i = enter_idx[0]
    print(f"\nEntry+close window for {row['Ticker_A']}/{row['Ticker_B']}:")
    print(pair_df.loc[max(0,i-1):i+5][['Date','Price_A','Price_B','Hedge_Ratio','ZScore','Position','Unrealized_PnL','Realized_PnL','Status']].to_string())

# === PortfolioManager 複利問題 ===
print("\n\n=== 檢查 capital_per_pair 的變化 ===")
# capital_per_pair 是 INITIAL_CAPITAL / top_n = 10000 / 10 = 1000
# 但 process_closed_trade 會更新 equity，下一期 allocate_capital 會給更多資金

# 找 NKTR/DYN 的 entry unrealized (-24460) - 這代表 capital_per_pair 很大
df_umap = pd.read_csv('results/yFinance/HDBSCAN_CS_UMAP/TradeLogs_Top10_SL0_ZWin0_MSR0.csv', parse_dates=['Date'])
nktr = df_umap[(df_umap['Ticker_A']=='NKTR') & (df_umap['Ticker_B']=='DYN')].reset_index(drop=True)
print("NKTR/DYN entry row:")
enter = nktr[nktr['Status'].str.startswith('ENTER')]
print(enter[['Date','Price_A','Unrealized_PnL','Realized_PnL','Status']].to_string())
# Compute capital from entry unrealized: unrealized = -entry_fee = -capital * friction_rate
# friction_rate = fee + slippage = 0.001 + 0.001 = 0.002 (guess)
# capital = -unrealized / friction_rate
entry_unreal = enter['Unrealized_PnL'].iloc[0]
print(f"\nImplied capital (if friction=0.002): {-entry_unreal / 0.002:.0f}")
print(f"Implied capital (if friction=0.001): {-entry_unreal / 0.001:.0f}")
