# -*- coding: utf-8 -*-
"""跨期去重網格的評估（判準見同目錄 PREREGISTRATION.md §五）。

  主要指標：逐格 ΔSharpe = Sharpe(去重) − Sharpe(不去重)，n = 60
  「去重有益」：中位 ≥ +0.05、≥ 40/60 為正、後半期中位 > 0
  「重複持有有益」：中位 ≤ −0.05、≤ 20/60 為正、後半期中位 < 0
  另報：最大回撤、年化波動、平均利用率、進場次數變化、break-even；依 top_n 與排序準則拆解

    python dev/dedup/evaluate.py
"""
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from strategies.config import INITIAL_CAPITAL  # noqa: E402
from strategies.metrics import _daily_frame, metrics, breakeven_roundtrip  # noqa: E402

BASE_DB = "results/result.db"
DD_DB = "results/dedup.db"
OUT = "results/analysis/dedup_eval.csv"
HALF = pd.Timestamp("2014-01-02")
ARMS = [f"{g}_{r}" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM") for r in ("SSD", "DTW", "SDP")]
TOPNS = (3, 5, 10, 20)


def path(arm, top_n, dd=False):
    return f"tiingo/Grid_{arm}/TradeLogs_Top{top_n}_SL0_ZWin0_MSR0" + ("_DD" if dd else "") + ".csv"


def entries(db, p):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute("SELECT Entries, Final_Equity FROM strategy_summaries WHERE _path=?", (p,)).fetchone()
    con.close()
    if r is None:
        raise RuntimeError(f"缺列：{db} {p}")
    return r


def evaluate_cell(db, p, top_n):
    """全期 Sharpe／MDD／利用率、後半期 Sharpe、年化波動（與 db_utils 同一複利報酬定義）。"""
    full = metrics(p, top_n=top_n, result_db=db)
    df = _daily_frame(p, db)
    h2 = metrics(p, dates=df.index[df.index >= HALF], result_db=db)
    eq = INITIAL_CAPITAL + df["d"].cumsum()
    ret = df["d"] / eq.shift(1).fillna(INITIAL_CAPITAL)
    n_ent, final = entries(db, p)
    return {"Sharpe": float(full["Sharpe_Raw"]), "H2": float(h2["Sharpe_Raw"]),
            "MDD": float(full["MDD_Raw"]), "Ann": float(full["Ann_Ret_Raw"]),
            "Vol": float(ret.std(ddof=1) * np.sqrt(252)),
            "Util": float(full.get("Avg_Utilization", np.nan)), "Entries": int(n_ent),
            "BE": breakeven_roundtrip(p, top_n, final, result_db=db)}


def main():
    recs = []
    for arm in ARMS:
        for tn in TOPNS:
            b = evaluate_cell(BASE_DB, path(arm, tn), tn)
            d = evaluate_cell(DD_DB, path(arm, tn, dd=True), tn)
            rec = {"arm": arm, "rank": arm.split("_")[1], "top_n": tn}
            for k in b:
                rec[f"{k}_base"], rec[f"{k}_dd"] = b[k], d[k]
            recs.append(rec)
            print(f"  {arm:<10} Top{tn:<2} Sharpe {d['Sharpe']:+.3f} vs {b['Sharpe']:+.3f} "
                  f"(Δ {d['Sharpe']-b['Sharpe']:+.3f})  進場 {d['Entries']/b['Entries']-1:+.1%}", flush=True)
    x = pd.DataFrame(recs)
    x["dSharpe"] = x.Sharpe_dd - x.Sharpe_base
    x["dH2"] = x.H2_dd - x.H2_base
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, index=False)
    print(f"-> {OUT}（{len(x)} 格）")

    med, pos, h2 = x.dSharpe.median(), int((x.dSharpe > 0).sum()), x.dH2.median()
    print("\n== 判準（PREREGISTRATION.md §五）==")
    good = [med >= 0.05, pos >= 40, h2 > 0]
    bad = [med <= -0.05, pos <= 20, h2 < 0]
    print(f"中位 ΔSharpe {med:+.4f}｜為正 {pos}/60｜後半期中位 {h2:+.4f}")
    print(f"「去重有益」：{['過' if c else '否' for c in good]} → {'成立' if all(good) else '不成立'}")
    print(f"「重複持有有益」：{['過' if c else '否' for c in bad]} → {'成立' if all(bad) else '不成立'}")
    if not all(good) and not all(bad):
        print("→ 無差異")

    print("\n== 另報（不入判準）==")
    for key, g in [("全部", x)] + [(f"Top{t}", g) for t, g in x.groupby("top_n")] + \
                  [(r, g) for r, g in x.groupby("rank")]:
        print(f"{key:<6} n={len(g):>2}｜ΔSharpe 中位 {g.dSharpe.median():+.4f} 為正 {(g.dSharpe > 0).sum():>2}/{len(g)}"
              f"｜後半期 {g.dH2.median():+.4f}｜MDD {g.MDD_dd.median():+.1%} vs {g.MDD_base.median():+.1%}"
              f"｜波動 {g.Vol_dd.median():.2%} vs {g.Vol_base.median():.2%}"
              f"｜利用率 {g.Util_dd.median():.3f} vs {g.Util_base.median():.3f}"
              f"｜進場 {np.median(g.Entries_dd / g.Entries_base - 1):+.1%}"
              f"｜BE {g.BE_dd.median():.3%} vs {g.BE_base.median():.3%}")


if __name__ == "__main__":
    main()
