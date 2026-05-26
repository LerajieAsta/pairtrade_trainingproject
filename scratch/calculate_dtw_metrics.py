import pandas as pd
import numpy as np
import os
import sys

# Configure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Read Data
price_file_path = r"D:\Unknown\Papper\Code\Ref_CODE\二十年股價.xlsx"
df_prices = pd.read_excel(price_file_path, sheet_name="Sheet1")
df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_prices.set_index('Date', inplace=True)

dates_file_path = r"D:\Unknown\Papper\Code\Ref_CODE\日期.xlsx"
df_dates = pd.read_excel(dates_file_path, sheet_name="Sheet1")

# 2. Core Functions
def calculate_daily_spread(stock_pairs, data):
    spread = pd.DataFrame(index=data.index)
    for _, row in stock_pairs.iterrows():
        s1, s2 = row['Stock1'], row['Stock2']
        pair = f"{s1}-{s2}"
        if s1 in data.columns and s2 in data.columns:
            spread[pair] = data[s1] - data[s2]
    return spread

def generate_trade_signals(price_diff, mean_spread, std_spread):
    upper = mean_spread + 2 * std_spread
    lower = mean_spread - 2 * std_spread
    df = pd.DataFrame(index=price_diff.index)
    df['spread'] = price_diff
    df['spread_prev'] = df['spread'].shift(1)
    df['signal1'] = 0
    df['position1'] = 0

    for t in df.index[1:]:
        prev = df.at[t, 'spread_prev']
        curr = df.at[t, 'spread']
        pos_prev = df.at[df.index[df.index.get_loc(t) - 1], 'position1']
        if pos_prev == 0:
            if prev > upper and curr <= upper:
                df.at[t, 'signal1'] = -1
            elif prev < lower and curr >= lower:
                df.at[t, 'signal1'] = 1
        else:
            if pos_prev == -1 and curr >= mean_spread:
                df.at[t, 'signal1'] = 1
            elif pos_prev == 1 and curr <= mean_spread:
                df.at[t, 'signal1'] = -1
        df.at[t, 'position1'] = pos_prev + df.at[t, 'signal1']

    penult = df.index[-2]
    last = df.index[-1]
    if df.at[penult, 'position1'] != 0:
        df.at[last, 'signal1'] = -df.at[penult, 'position1']
        df.at[last, 'position1'] = 0

    df['signal2'] = -df['signal1']
    df['position2'] = -df['position1']
    df['trade1'] = df['signal1']
    df['trade2'] = df['signal2']
    return df[['spread', 'position1', 'position2', 'trade1', 'trade2']]

def generate_trade_records(signals, trade_data, initial_capital=100000, stop_loss_pct=0.10):
    half_cap = initial_capital / 2
    records = []
    for pair in signals['pair'].unique():
        sig = signals[signals['pair'] == pair]
        s1, s2 = pair.split('-')
        p1 = trade_data[s1]; p2 = trade_data[s2]
        rec = pd.DataFrame(index=sig.index)
        rec['trade1'] = sig['trade1']
        rec['trade2'] = sig['trade2']
        rec['price1'] = p1
        rec['price2'] = p2
        rec['cash1'] = np.nan; rec['cash2'] = np.nan
        rec['holdings1'] = np.nan; rec['holdings2'] = np.nan
        rec['total_value'] = np.nan
        rec['stop_loss'] = 0

        cash1 = cash2 = half_cap
        hold1 = hold2 = 0.0
        rec.iloc[0, rec.columns.get_loc('cash1')] = cash1
        rec.iloc[0, rec.columns.get_loc('cash2')] = cash2
        rec.iloc[0, rec.columns.get_loc('holdings1')] = hold1
        rec.iloc[0, rec.columns.get_loc('holdings2')] = hold2
        rec.iloc[0, rec.columns.get_loc('total_value')] = cash1 + cash2

        for i in range(1, len(rec)):
            t = rec.index[i]
            t1 = rec.at[t, 'trade1']; t2 = rec.at[t, 'trade2']
            pr1 = rec.at[t, 'price1']; pr2 = rec.at[t, 'price2']
            if t1 != 0:
                cash1 += -t1 * half_cap - abs(half_cap) * 0.001
                hold1 = t1 * (half_cap / pr1)
            if t2 != 0:
                cash2 += -t2 * half_cap - abs(half_cap) * 0.001
                hold2 = t2 * (half_cap / pr2)
            total = cash1 + cash2 + hold1 * pr1 + hold2 * pr2
            loss_pct = (initial_capital - total) / initial_capital
            if loss_pct >= stop_loss_pct and (hold1 != 0 or hold2 != 0):
                cash1 += hold1 * pr1; cash2 += hold2 * pr2
                hold1 = hold2 = 0.0
                rec.at[t, 'stop_loss'] = 1
                total = cash1 + cash2
            else:
                rec.at[t, 'stop_loss'] = 0
            rec.at[t, 'cash1'] = cash1; rec.at[t, 'cash2'] = cash2
            rec.at[t, 'holdings1'] = hold1; rec.at[t, 'holdings2'] = hold2
            rec.at[t, 'total_value'] = total
        rec['pair'] = pair
        records.append(rec)
    return pd.concat(records)

# 3. Main Loop
all_tr_records = []
all_round_meta = []
unique_pairs_traded = set()

print("Running backtest for DTW...", flush=True)
for idx, row in df_dates.iterrows():
    fs = pd.to_datetime(row["形成期開始"])
    fe = pd.to_datetime(row["形成期結束"])
    ts = pd.to_datetime(row["交易期開始"])
    te = pd.to_datetime(row["交易期結束"])

    formation = df_prices.loc[fs:fe].ffill().bfill()
    trade     = df_prices.loc[ts:te].ffill().bfill()

    pair_fp  = f"D:/Unknown/Papper/Code/Ref_CODE/DTW配對結果/DTW配對結果({fs.strftime('%Y-%m-%d')}).xlsx"
    df_pairs = pd.read_excel(pair_fp)
    stock_pairs = df_pairs[["Stock1", "Stock2"]].head(20)

    # Track unique pairs traded in this round
    for _, r in stock_pairs.iterrows():
        unique_pairs_traded.add(f"{r['Stock1']}-{r['Stock2']}")

    norm_f      = (1 + formation.pct_change().fillna(0)).cumprod()
    spread_f    = calculate_daily_spread(stock_pairs, norm_f)
    mean_spread = spread_f.mean()
    std_spread  = spread_f.std(ddof=0)

    norm_t   = trade.div(formation.iloc[0])
    spread_t = calculate_daily_spread(stock_pairs, norm_t)

    sigs = []
    for _, r in stock_pairs.iterrows():
        name   = f"{r['Stock1']}-{r['Stock2']}"
        sig_df = generate_trade_signals(spread_t[name], mean_spread[name], std_spread[name])
        sig_df["pair"] = name
        sigs.append(sig_df)
    all_signals = pd.concat(sigs)
    tr_records  = generate_trade_records(all_signals, trade)

    tr_records['round'] = idx + 1
    all_tr_records.append(tr_records)
    
    pairs20 = stock_pairs.apply(lambda x: f"{x['Stock1']}-{x['Stock2']}", axis=1).tolist()
    all_round_meta.append({'round': idx + 1, 'pairs': pairs20})

# Combine NAV series
INITIAL_CAPITAL = 100000
combined_tv = []
for rdf in all_tr_records:
    pairs20 = all_round_meta[rdf['round'].iloc[0] - 1]['pairs']
    sub = rdf[rdf['pair'].isin(pairs20)].copy()
    daily_avg = sub.groupby(sub.index)['total_value'].mean()
    combined_tv.append(daily_avg)

combined_tv = pd.concat(combined_tv).sort_index()
combined_tv = combined_tv[~combined_tv.index.duplicated(keep='last')]
combined_nav = combined_tv / INITIAL_CAPITAL

# Calculate Performance Metrics
combined_daily_ret = combined_nav.pct_change().dropna()
total_days = len(combined_nav)
years = total_days / 252
total_ret = combined_nav.iloc[-1] - 1
ann_ret   = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

# Max Drawdown
rolling_max = combined_nav.cummax()
drawdown = (combined_nav - rolling_max) / rolling_max
mdd_val = drawdown.min()

# Sharpe
excess = combined_daily_ret - 0.02 / 252
sharpe = np.sqrt(252) * excess.mean() / excess.std()

# RCC
final_pnl = combined_tv.iloc[-1] - INITIAL_CAPITAL
rcc = final_pnl / INITIAL_CAPITAL

# REC
# n_traded unique pairs
n_unique_traded = len(unique_pairs_traded)
c_pair = INITIAL_CAPITAL / 20 # 5000 per pair
engaged_capital = n_unique_traded * c_pair
rec = final_pnl / engaged_capital

# Annual Performance
annual_ret = combined_nav.resample('YE').last().pct_change().dropna()
annual_pnl = combined_tv.resample('YE').last().pct_change().dropna()

print("\n" + "="*50)
print("             EXACT DTW PERFORMANCE METRICS")
print("="*50)
print(f"Total Return      : {total_ret*100:.2f}%")
print(f"Annualized Return : {ann_ret*100:.2f}%")
print(f"Max Drawdown      : {mdd_val*100:.2f}%")
print(f"Sharpe Ratio      : {sharpe:.3f}")
print(f"RCC               : {rcc*100:.2f}%")
print(f"REC (Engaged Cap) : {rec*100:.2f}%")
print(f"Unique Pairs Traded: {n_unique_traded}")
print(f"Engaged Capital   : {engaged_capital:,.2f}")
print(f"Final PnL         : {final_pnl:,.2f}")

print("\nAnnual Returns:")
years_series = combined_nav.resample('YE').last()
prev_val = 1.0
for yr, val in years_series.items():
    yr_ret = (val / prev_val) - 1
    print(f"  {yr.year}: {yr_ret*100:.2f}%")
    prev_val = val
