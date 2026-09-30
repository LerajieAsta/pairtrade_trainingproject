# -*- coding: utf-8 -*-
"""同時持倉上限 K 網格的評估（判準見同目錄 PREREGISTRATION.md §四）。

  0. 驗證：max_active.db 中 15 格無上限 Top10（K ≥ top_n 展開出的格）與 result.db 逐位元相同
  1. 主比較 (20, 10) 對 Top 10：中位 ΔSharpe ≥ +0.05、≥ 11/15 為正、後半期中位 > 0
  2. 次要比較（只描述）：(10,3)、(20,3) 對 Top 3；(10,5)、(20,5) 對 Top 5
  3. 另報（不入判準）：對 Top N（同候選池）、觸頂日比例、進場次數變化、break-even

    python dev/max_active/evaluate.py
"""
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from strategies.metrics import metrics, breakeven_roundtrip  # noqa: E402

BASE_DB = "results/result.db"
MA_DB = "results/max_active.db"
OUT = "results/analysis/max_active_eval.csv"
HALF = pd.Timestamp("2014-01-02")
ARMS = [f"{g}_{r}" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM") for r in ("SSD", "DTW", "SDP")]
CELLS = [(10, 3), (20, 3), (10, 5), (20, 5), (20, 10)]
MAIN = (20, 10)


def path(arm, top_n, k=None):
    return (f"tiingo/Grid_{arm}/TradeLogs_Top{top_n}_SL0_ZWin0_MSR0"
            + (f"_MA{k}" if k else "") + ".csv")


def summary(db, p):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute("SELECT Sharpe_Raw, Final_Equity, Entries, Ann_Ret_Raw, MDD_Raw "
                    "FROM strategy_summaries WHERE _path=?", (p,)).fetchone()
    con.close()
    if r is None:
        raise RuntimeError(f"缺列：{db} {p}")
    return r


def half_sharpe(db, p, _cache={}):
    key = (db, p)
    if key not in _cache:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
        d = pd.read_sql("SELECT DISTINCT Date FROM trade_logs WHERE strategy_id=?", con, params=(p,))
        con.close()
        dates = pd.to_datetime(d.Date)
        _cache[key] = float(metrics(p, dates=dates[dates >= HALF], result_db=db)["Sharpe_Raw"])
    return _cache[key]


def cap_hit(p, k):
    """每期每日持倉數 = K 的 (期, 日) 比例，只計有部位的 (期, 日)。"""
    con = sqlite3.connect(f"file:{MA_DB}?mode=ro", uri=True, timeout=600)
    df = pd.read_sql("SELECT Period_Start, Date FROM trade_logs WHERE strategy_id=? AND Position != 0",
                     con, params=(p,))
    con.close()
    occ = df.groupby(["Period_Start", "Date"]).size()
    return float((occ == k).mean()), int(occ.max())


def main():
    # 0. 驗證
    same = 0
    for arm in ARMS:
        a, b = summary(BASE_DB, path(arm, 10))[:3], summary(MA_DB, path(arm, 10))[:3]
        same += a == b
        if a != b:
            print(f"  [不同] {arm} result.db {a} vs max_active.db {b}")
    print(f"[驗證] 無上限 Top10 與 result.db 逐位元相同：{same}/15")

    recs = []
    for arm in ARMS:
        for tn, k in CELLS:
            pc, pk, pn = path(arm, tn, k), path(arm, k), path(arm, tn)
            sc, sk, sn = summary(MA_DB, pc), summary(BASE_DB, pk), summary(BASE_DB, pn)
            hit, mx = cap_hit(pc, k)
            recs.append({
                "arm": arm, "top_n": tn, "K": k,
                "Sharpe_cap": sc[0], "Sharpe_TopK": sk[0], "Sharpe_TopN": sn[0],
                "dSharpe_vs_TopK": sc[0] - sk[0], "dSharpe_vs_TopN": sc[0] - sn[0],
                "H2_cap": half_sharpe(MA_DB, pc), "H2_TopK": half_sharpe(BASE_DB, pk),
                "Ann_cap": sc[3], "Ann_TopK": sk[3], "MDD_cap": sc[4], "MDD_TopK": sk[4],
                "Entries_cap": sc[2], "Entries_TopK": sk[2], "Entries_TopN": sn[2],
                "cap_hit_share": hit, "max_occupancy": mx,
                "BE_cap": breakeven_roundtrip(pc, tn, sc[1], result_db=MA_DB),
                "BE_TopK": breakeven_roundtrip(pk, k, sk[1], result_db=BASE_DB),
            })
            print(f"  {arm:<10} ({tn},{k:>2}) Sharpe {sc[0]:+.3f} vs Top{k} {sk[0]:+.3f} "
                  f"(Δ {sc[0]-sk[0]:+.3f})  觸頂 {hit:.2f}  最大持倉 {mx}", flush=True)
    d = pd.DataFrame(recs)
    d["dH2_vs_TopK"] = d.H2_cap - d.H2_TopK
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"-> {OUT}（{len(d)} 格）")

    print("\n== 主比較（PREREGISTRATION.md §四）：(20, 10) 對 Top 10 ==")
    m = d[(d.top_n == MAIN[0]) & (d.K == MAIN[1])]
    med, pos, h2 = m.dSharpe_vs_TopK.median(), int((m.dSharpe_vs_TopK > 0).sum()), m.dH2_vs_TopK.median()
    c1, c2, c3 = med >= 0.05, pos >= 11, h2 > 0
    print(f"中位 ΔSharpe {med:+.4f} [{'過' if c1 else '否'}]｜為正 {pos}/15 [{'過' if c2 else '否'}]｜"
          f"後半期中位 {h2:+.4f} [{'過' if c3 else '否'}] → {'有效' if c1 and c2 and c3 else '未過閘'}")

    print("\n== 次要比較（只描述）==")
    for (tn, k), g in d.groupby(["top_n", "K"]):
        print(f"({tn},{k:>2}) 對 Top{k:<2}：中位 Δ {g.dSharpe_vs_TopK.median():+.4f}，"
              f"為正 {(g.dSharpe_vs_TopK > 0).sum()}/15，後半期中位 {g.dH2_vs_TopK.median():+.4f}｜"
              f"對 Top{tn}：中位 Δ {g.dSharpe_vs_TopN.median():+.4f}｜"
              f"觸頂中位 {g.cap_hit_share.median():.2f}｜進場 對 Top{tn} {np.median(g.Entries_cap / g.Entries_TopN - 1):+.1%}"
              f"、對 Top{k} {np.median(g.Entries_cap / g.Entries_TopK - 1):+.1%}｜"
              f"BE 中位 {g.BE_cap.median():.4%} vs {g.BE_TopK.median():.4%}")


if __name__ == "__main__":
    main()
