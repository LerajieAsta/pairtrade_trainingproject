# -*- coding: utf-8 -*-
"""殘差報酬價差網格的評估（判準見同目錄 PREREGISTRATION.md §三）。

  主要：逐臂 ΔSharpe＝(15 格 Sharpe 平均)_RR − (15 格 Sharpe 平均)_現行，n = 15 臂；後半期同法
  另報：15 格等權逐日差分（4.3 同程序：循環區塊自助法 L=126、B=10,000、15 臂 BH）、進場次數、
        β_ret 與 β_level 的分布、β_ret ≤ 0 的比例、break-even

    python dev/residual_spread/evaluate.py
"""
import os
import re
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.block_bootstrap import bh_adjust, bootstrap_test  # noqa: E402
from strategies.config import INITIAL_CAPITAL  # noqa: E402
from strategies.metrics import breakeven_roundtrip, metrics  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DB, RR_DB = "results/result.db", "results/residual_spread.db"
OUT = "results/analysis/residual_spread_eval.csv"
HALF = pd.Timestamp("2014-01-02")
ARMS = [f"{g}_{r}" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM") for r in ("SSD", "DTW", "SDP")]
CELLS = [(t, s) for t in (1, 3, 5, 10, 20) for s in (0, 5, 15)]


def path(arm, tn, sl, rr):
    return f"tiingo/Grid_{arm}/TradeLogs_Top{tn}_SL{sl}_ZWin0_MSR0" + ("_RR" if rr else "") + ".csv"


def daily(db, p):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    d = pd.read_sql("SELECT Date, Daily_Delta FROM trade_logs WHERE strategy_id=?", con, params=(p,))
    con.close()
    d = d.groupby("Date").Daily_Delta.sum()
    d.index = pd.to_datetime(d.index)
    return d.sort_index()


def cell_stats(db, p, tn):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute("SELECT Sharpe_Raw, Entries, Final_Equity FROM strategy_summaries WHERE _path=?",
                    (p,)).fetchone()
    hr = pd.read_sql("SELECT Period_Start, Ticker_A, Ticker_B, AVG(Hedge_Ratio) h FROM trade_logs "
                     "WHERE strategy_id=? GROUP BY 1,2,3", con, params=(p,)).h
    con.close()
    if r is None:
        raise RuntimeError(f"缺列：{db} {p}")
    d = daily(db, p)
    h2 = float(metrics(p, dates=d.index[d.index >= HALF], result_db=db)["Sharpe_Raw"])
    return {"sharpe": r[0], "h2": h2, "entries": r[1],
            "be": breakeven_roundtrip(p, tn, r[2], result_db=db),
            "beta_med": float(hr.median()), "beta_nonpos": float((hr <= 0).mean())}, d


def main():
    rows, deltas = [], {}
    for arm in ARMS:
        base, rr, dz, dr = [], [], [], []
        for tn, sl in CELLS:
            b, db_ = cell_stats(BASE_DB, path(arm, tn, sl, False), tn)
            r, dr_ = cell_stats(RR_DB, path(arm, tn, sl, True), tn)
            base.append(b); rr.append(r); dz.append(db_); dr.append(dr_)
        B, R = pd.DataFrame(base), pd.DataFrame(rr)
        idx = sorted(set().union(*[x.index for x in dz + dr]))
        ew = (pd.concat(dr, axis=1).reindex(idx).fillna(0).mean(axis=1)
              - pd.concat(dz, axis=1).reindex(idx).fillna(0).mean(axis=1))
        deltas[arm] = ew.values
        rows.append({"arm": arm,
                     "Sharpe_base": B.sharpe.mean(), "Sharpe_rr": R.sharpe.mean(),
                     "dSharpe": R.sharpe.mean() - B.sharpe.mean(),
                     "dH2": R.h2.mean() - B.h2.mean(),
                     "entries_ratio": R.entries.sum() / max(1, B.entries.sum()),
                     "BE_base": B.be.mean(), "BE_rr": R.be.mean(),
                     "beta_level_med": B.beta_med.median(), "beta_ret_med": R.beta_med.median(),
                     "beta_ret_nonpos": R.beta_nonpos.mean()})
        print(f"  {arm:<10} Sharpe {R.sharpe.mean():+.3f} vs {B.sharpe.mean():+.3f}（Δ {rows[-1]['dSharpe']:+.3f}）"
              f"｜後半期 Δ {rows[-1]['dH2']:+.3f}｜進場 ×{rows[-1]['entries_ratio']:.2f}", flush=True)
    x = pd.DataFrame(rows)
    bt = [bootstrap_test(deltas[a]) for a in ARMS]
    x["ew_ann_delta_pct"] = [deltas[a].mean() * 252 / INITIAL_CAPITAL * 100 for a in ARMS]
    x["CI_lo"] = [b["CI下界"] for b in bt]
    x["CI_hi"] = [b["CI上界"] for b in bt]
    x["p"] = [b["BB p"] for b in bt]
    x["p_BH"] = bh_adjust(x.p.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, index=False, encoding="utf-8-sig")

    med, pos, h2 = x.dSharpe.median(), int((x.dSharpe > 0).sum()), x.dH2.median()
    good = [med >= 0.05, pos >= 10, h2 > 0]
    bad = [med <= -0.05, pos <= 5, h2 < 0]
    print(f"\n== 判準（PREREGISTRATION.md §三）==\n中位 ΔSharpe {med:+.4f}｜為正 {pos}/15｜後半期中位 {h2:+.4f}")
    print(f"「殘差報酬價差較佳」{['過' if c else '否' for c in good]} → {'成立' if all(good) else '不成立'}")
    print(f"「現行價差較佳」{['過' if c else '否' for c in bad]} → {'成立' if all(bad) else '不成立'}")
    if not all(good) and not all(bad):
        print("→ 無差異")
    print("\n== 另報 ==")
    print(x[["arm", "ew_ann_delta_pct", "CI_lo", "CI_hi", "p", "p_BH", "entries_ratio",
             "beta_level_med", "beta_ret_med", "beta_ret_nonpos", "BE_base", "BE_rr"]].round(4).to_string(index=False))
    print(f"\nBH 校正後顯著（p_BH < 0.05）：正 {int(((x.p_BH < 0.05) & (x.ew_ann_delta_pct > 0)).sum())}、"
          f"負 {int(((x.p_BH < 0.05) & (x.ew_ann_delta_pct < 0)).sum())}／15")


if __name__ == "__main__":
    main()
