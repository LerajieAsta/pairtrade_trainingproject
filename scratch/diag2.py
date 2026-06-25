"""
診斷 SSD_Basic / Pure_DTW / SSD_Rolling / DTW_Paper 的
  sum(Daily_Delta) ≠ sum(final Realized_PnL) 問題

這些策略都是 zscore_trading.py 或 pure_dtw_trading.py 跑的，
且帶有大量 HOLDING 狀態，說明倉位橫跨整個交易期。

核心假設：Daily_Delta 的加總應等於最終 Realized_PnL 的加總。
若不等，代表某些天的 Daily_Delta 計算有誤。
"""
import pandas as pd
import numpy as np

for strat, fname in [
    ('SSD_Basic', 'results/yFinance/SSD_Basic/TradeLogs_Top10_SL0_ZWin0_MSR0.csv'),
    ('Pure_DTW',  'results/yFinance/Pure_DTW/TradeLogs_Top10_SL0_ZWin0_MSR0.csv'),
    ('SSD_Rolling','results/yFinance/SSD_Rolling/TradeLogs_Top10_SL0_ZWin0_MSR0.csv'),
]:
    df = pd.read_csv(fname, parse_dates=['Date'])
    
    pair_groups = df.groupby(['Period_Start','Ticker_A','Ticker_B'])
    
    # Per-pair: check if sum(Daily_Delta) == last(Realized_PnL)
    bad_pairs = []
    for key, g in pair_groups:
        g = g.sort_values('Date').reset_index(drop=True)
        sum_delta = g['Daily_Delta'].sum()
        last_realized = g['Realized_PnL'].iloc[-1]
        diff = sum_delta - last_realized
        if abs(diff) > 1.0:  # tolerance
            bad_pairs.append({
                'key': key,
                'sum_delta': sum_delta,
                'last_realized': last_realized,
                'diff': diff,
                'n_rows': len(g),
                'last_status': g['Status'].iloc[-1],
                'n_period_end': (g['Status'] == 'PERIOD_END_EXIT').sum(),
            })
    
    bad_df = pd.DataFrame(bad_pairs).sort_values('diff', key=abs, ascending=False)
    print(f"\n=== {strat}: {len(bad_pairs)}/{pair_groups.ngroups} pairs have discrepancy ===")
    print(bad_df.head(10).to_string())
    
    # Show worst pair in detail
    if len(bad_pairs) > 0:
        worst = bad_df.iloc[0]
        key = worst['key']
        g = pair_groups.get_group(key).sort_values('Date').reset_index(drop=True)
        # find PERIOD_END_EXIT rows
        pe_rows = g[g['Status'] == 'PERIOD_END_EXIT']
        if not pe_rows.empty:
            idx = pe_rows.index[0]
            window = g.loc[max(0, idx-3):idx+1]
            ps, ta, tb = key
            print(f"\nWorst discrepant pair: {ta}/{tb} period={ps}")
            print(window[['Date','Position','Unrealized_PnL','Realized_PnL','Cumulative_PnL','Daily_Delta','Status']].to_string())
