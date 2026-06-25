import pandas as pd
import numpy as np
import os
import glob

base = "./results/yFinance"

# Collect one representative file from each strategy
files = {}
for strat in os.listdir(base):
    d = os.path.join(base, strat)
    if not os.path.isdir(d):
        continue
    csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
    if csvs:
        files[strat] = csvs[0]

print(f"Found {len(files)} strategies\n")

for strat, fpath in files.items():
    try:
        df = pd.read_csv(fpath, parse_dates=["Date"])
    except Exception as e:
        print(f"{strat}: ERROR reading - {e}")
        continue

    # Per-pair final realized PnL
    pair_final = df.groupby(["Period_Start","Ticker_A","Ticker_B"])["Realized_PnL"].last()
    total_realized = pair_final.sum()

    # Sum of daily_delta 
    daily = df.groupby("Date")["Daily_Delta"].sum()
    sum_delta = daily.sum()

    # Check for outsized single-day jumps
    max_day = daily.max()
    min_day = daily.min()
    n_pairs = df.groupby(["Period_Start","Ticker_A","Ticker_B"]).ngroups

    # Check if sum_delta ≈ total_realized (they SHOULD be equal)
    discrepancy = sum_delta - total_realized

    # Check for suspicious large individual Trade_PnL
    if "Trade_PnL" in df.columns:
        trade_pnl = df[df["Trade_PnL"] != 0]["Trade_PnL"]
        max_trade = trade_pnl.max() if len(trade_pnl) > 0 else 0
        min_trade = trade_pnl.min() if len(trade_pnl) > 0 else 0
        # flag pairs with outrageously large trade PnL
        capital_guess = 10000
        huge_wins = (trade_pnl > capital_guess * 5).sum()
        huge_loss = (trade_pnl < -capital_guess * 5).sum()
    else:
        max_trade = min_trade = huge_wins = huge_loss = "N/A"

    # Unrealized check - there should be no unrealized at period boundary
    if "Unrealized_PnL" in df.columns:
        period_ends = df[df["Status"].isin(["PERIOD_END_EXIT","FORCED_CLOSE_DELISTED"])]
        nonzero_unreal = (period_ends["Unrealized_PnL"] != 0).sum()
    else:
        nonzero_unreal = "N/A"

    # Show status distribution
    status_counts = df["Status"].value_counts().to_dict()

    print(f"=== {strat} ===")
    print(f"  File: {os.path.basename(fpath)}")
    print(f"  Pairs/periods: {n_pairs}")
    print(f"  Total realized PnL:  {total_realized:>12.2f}")
    print(f"  Sum daily_delta:     {sum_delta:>12.2f}  (discrepancy: {discrepancy:.2f})")
    print(f"  Day delta range:     [{min_day:.2f}, {max_day:.2f}]")
    print(f"  Trade PnL range:     [{min_trade}, {max_trade}]")
    print(f"  Huge wins/losses:    {huge_wins} / {huge_loss}  (> 5x capital)")
    print(f"  Period-end non-zero unrealized: {nonzero_unreal}")
    print(f"  Status dist: {status_counts}")
    print()
