# -*- coding: utf-8 -*-
"""A：訊號空間 ≠ 執行空間 —— 量測引擎持倉與「能複製訊號的持倉」的差距。

唯讀。用法：
    python dev/trading_arch/measure_hedge_mismatch.py

訊號（ssd_rolling 的標準化空間）：
    S = (ln P_A − μ_A)/σ_A − β·(ln P_B − μ_B)/σ_B
    ΔS = r_A/σ_A − β·r_B/σ_B
→ 要複製 ΔS，美元權重須為 (1/σ_A) : (β/σ_B)。

引擎（zscore_trading._execute_entry）：
    v_a : v_b = 1 : |β|
→ PnL ∝ r_A − β·r_B。B 腳錯乘 σ_B/σ_A。

本腳本對每筆完成交易，用相同的進出場日期與相同的總名目額 C=1，
分別算兩個組合的**毛**報酬（不含手續費，費用對兩者完全相同故不影響對照）。
"""
import argparse
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from strategies.config import DB_PATH, TABLE_NAME  # noqa: E402

RESULT_DB = "results/result.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"

# 主力臂：交易端 path_key 與其形成期 strategy_id
STRATEGY = "tiingo/Grid_AGG_SSD_NF/TradeLogs_Top20_SL0_ZWin0_MSR0.csv"
FORMATION_ID = "Grid AGG-SSD-NF_MSR0"

_CLOSE_STATUS = ("EXIT", "PERIOD_END_EXIT", "STOP_LOSS_TRIGGERED",
                 "TIME_STOP", "FORCED_CLOSE_DELISTED")


def load_prices() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        px = pd.read_sql(f"SELECT Date, Symbol, Adj_Close FROM {TABLE_NAME}", con)
    finally:
        con.close()
    px["Date"] = pd.to_datetime(px["Date"])
    return px.pivot(index="Date", columns="Symbol", values="Adj_Close")


def load_formation_params() -> pd.DataFrame:
    """每 (Trade_Start, A, B) 的 σ_A、σ_B、β。"""
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True)
    try:
        fp = pd.read_sql(
            "SELECT Trade_Start AS Period_Start, Ticker_A, Ticker_B, Formation_Params "
            "FROM formation_pairs WHERE strategy_id = ?", con, params=(FORMATION_ID,))
    finally:
        con.close()
    j = fp["Formation_Params"].map(json.loads)
    fp["sa"] = [float(x.get("Log_Std_A", 1.0) or 1.0) for x in j]
    fp["sb"] = [float(x.get("Log_Std_B", 1.0) or 1.0) for x in j]
    fp["beta"] = [float(x.get("Hedge_Ratio", 1.0) or 1.0) for x in j]
    fp["Period_Start"] = fp["Period_Start"].astype(str)
    return fp.set_index(["Period_Start", "Ticker_A", "Ticker_B"])[["sa", "sb", "beta"]]


def sigma_dispersion() -> None:
    """σ_A/σ_B 在全庫的分散程度 —— 決定這個錯誤有多大。"""
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True)
    try:
        df = pd.read_sql("SELECT Formation_Params FROM formation_pairs LIMIT 200000", con)
    finally:
        con.close()
    j = df["Formation_Params"].map(json.loads)
    a = np.array([float(x.get("Log_Std_A", np.nan) or np.nan) for x in j])
    b = np.array([float(x.get("Log_Std_B", np.nan) or np.nan) for x in j])
    r = pd.Series(a / b).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"σ_A/σ_B  n={len(r):,}  中位={r.median():.3f}  "
          f"p10={r.quantile(.10):.3f}  p90={r.quantile(.90):.3f}  "
          f"|ln ratio|>0.3 比例={np.mean(np.abs(np.log(r)) > 0.3):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=RESULT_DB,
                    help="要量測的 result.db（試跑時指向 results/pilot_hedge.db）")
    args = ap.parse_args()

    sigma_dispersion()

    prices = load_prices()
    key = load_formation_params()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        tl = pd.read_sql(
            "SELECT Period_Start, Ticker_A, Ticker_B, Date, ZScore, Status "
            "FROM trade_logs WHERE strategy_id = ? "
            "ORDER BY Ticker_A, Ticker_B, Period_Start, Date",
            con, params=(STRATEGY,))
    finally:
        con.close()
    tl["Date"] = pd.to_datetime(tl["Date"])
    tl["Period_Start"] = tl["Period_Start"].astype(str)

    rows = []
    for k, d in tl.groupby(["Period_Start", "Ticker_A", "Ticker_B"], sort=False):
        if k not in key.index:
            continue
        sa, sb, beta = key.loc[k]
        d = d.reset_index(drop=True)
        ent = d.index[d["Status"].str.startswith("ENTER")].tolist()
        if not ent:
            continue
        i = ent[0]
        ex = [x for x in d.index[d["Status"].isin(_CLOSE_STATUS)] if x > i]
        if not ex:
            continue
        j = ex[0]

        ta, tb = k[1], k[2]
        if ta not in prices.columns or tb not in prices.columns:
            continue
        pa0, pa1 = prices[ta].asof(d["Date"][i]), prices[ta].asof(d["Date"][j])
        pb0, pb1 = prices[tb].asof(d["Date"][i]), prices[tb].asof(d["Date"][j])
        if not np.all(np.isfinite([pa0, pa1, pb0, pb1])) or min(pa0, pb0) <= 0:
            continue

        r_a, r_b = np.log(pa1 / pa0), np.log(pb1 / pb0)
        sgn = -1.0 if d["ZScore"][i] > 0 else 1.0     # z>0 → 空 A 多 B

        # 引擎：v_a : v_b = 1 : |β|，總名目額 C = 1
        tw = 1.0 + abs(beta)
        pnl_exec = sgn * (r_a / tw - abs(beta) * r_b / tw)

        # 複製訊號：v_a : v_b = (1/σ_A) : (β/σ_B)，同樣總名目額 C = 1
        wa, wb = 1.0 / sa, beta / sb
        tot = abs(wa) + abs(wb)
        pnl_rep = sgn * (wa / tot * r_a - wb / tot * r_b)

        rows.append((abs(d["ZScore"][i]) - abs(d["ZScore"][j]), pnl_exec, pnl_rep))

    t = pd.DataFrame(rows, columns=["dz", "exec", "rep"])
    print(f"\n策略 {STRATEGY}")
    print(f"完成交易 n = {len(t):,}")
    print(f"  exec vs rep 相關          = {t['exec'].corr(t['rep']):.4f}")
    print(f"  符號相反比例              = {np.mean(np.sign(t['exec']) != np.sign(t['rep'])):.4f}")

    conv = t[t["dz"] > 1.0]
    print(f"\n收斂交易（Δ|z| > 1）n = {len(conv):,}")
    print(f"  複製組合 毛虧損率        = {np.mean(conv['rep'] < 0):.4f}   ← 理論上應 ≈ 0")
    print(f"  引擎組合 毛虧損率        = {np.mean(conv['exec'] < 0):.4f}")
    print(f"  平均毛報酬 複製 → 引擎   = {conv['rep'].mean():.5f} → {conv['exec'].mean():.5f}"
          f"   （差 {(conv['rep'].mean() - conv['exec'].mean()) * 100:.2f} pp）")

    dev = (t["exec"] - t["rep"]).abs() / t["rep"].abs().clip(lower=1e-9)
    print(f"\n|exec − rep| / |rep| 中位 = {dev.median():.3f}")


if __name__ == "__main__":
    main()
