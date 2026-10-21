# -*- coding: utf-8 -*-
"""殘差報酬訊號＋水準部位的評估（判準見同目錄 PREREGISTRATION.md §四）。

  主要：逐臂 ΔSharpe = (15 格 Sharpe_Raw 平均)_RRL − (15 格 Sharpe_Raw 平均)_現行，n = 15。
  後半期：同定義，Sharpe 由 2014-01-02 起的逐日複利報酬重算。
  效度條件：RRL 15 格等權逐日報酬對成分股等權市場報酬的 β，15 臂中位 ≤ +0.02。
  另報：15 格等權逐日損益差分（4.3 同程序，15 臂 BH）；三種口徑（現行／_RR／RRL）的市場 β 與扣市場後 Sharpe；
        RRL − _RR 的 ΔSharpe；進場次數、勝率、停損。

    python dev/residual_level_hedge/evaluate.py
"""
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.block_bootstrap import bh_adjust, bootstrap_test  # noqa: E402
from strategies.config import INITIAL_CAPITAL  # noqa: E402
from strategies.returns import daily_pnl, returns_from_pnl  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB = {"z": "results/result.db", "rr": "results/residual_spread.db", "rrl": "results/residual_level_hedge.db"}
SUFFIX = {"z": "", "rr": "_RR", "rrl": "_RRL"}
OUT = "results/analysis/residual_level_hedge_eval.csv"
HALF = pd.Timestamp("2014-01-02")
ARMS = [f"{g}_{r}" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM") for r in ("SSD", "DTW", "SDP")]
CELLS = [(t, s) for t in (1, 3, 5, 10, 20) for s in (0, 5, 15)]


def path(arm, kind, tn, sl):
    return f"tiingo/Grid_{arm}/TradeLogs_Top{tn}_SL{sl}_ZWin0_MSR0{SUFFIX[kind]}.csv"


def summ(arm, kind):
    con = sqlite3.connect(f"file:{DB[kind]}?mode=ro", uri=True, timeout=600)
    ps = [path(arm, kind, tn, sl) for tn, sl in CELLS]
    d = pd.read_sql(f"SELECT Sharpe_Raw, Entries, Win_Rate, Stop_Losses FROM strategy_summaries "
                    f"WHERE _path IN ({','.join('?' * len(ps))})", con, params=ps)
    con.close()
    if len(d) != len(CELLS):
        raise RuntimeError(f"{DB[kind]} {arm} 只有 {len(d)}/15 格")
    return d.mean()


def cells(arm, kind):
    ps = [path(arm, kind, tn, sl) for tn, sl in CELLS]
    pnl = daily_pnl(ps, DB[kind], use_cache=False)
    if pnl.shape[1] != len(CELLS):
        raise RuntimeError(f"{DB[kind]} {arm} 逐日紀錄只有 {pnl.shape[1]}/15 格")
    return pnl, pd.DataFrame({c: returns_from_pnl(pnl[c]) for c in pnl.columns}, index=pnl.index)


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def market():
    import strategies.config as C
    from strategies.preprocess_equity import DataProcessor
    pv, *_ = DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME).prepare_backtest_data(
        "2000-01", "2025-12", 252)
    return pv.pct_change(fill_method=None).clip(-0.5, 0.5).mean(axis=1)


def beta_alpha(y, m):
    """回傳 (市場 β, 扣市場後的年化 Sharpe：截距＋殘差)。"""
    ok = y.notna()
    yy, mm = y[ok].values, m.reindex(y.index).fillna(0.0)[ok].values
    x = np.column_stack([np.ones(len(mm)), mm])
    coef = np.linalg.lstsq(x, yy, rcond=None)[0]
    res = yy - coef[1] * mm
    return float(coef[1]), float(res.mean() / res.std() * np.sqrt(252))


def main():
    mkt = market()
    rows, diffs = [], {}
    for arm in ARMS:
        s = {k: summ(arm, k) for k in DB}
        data = {k: cells(arm, k) for k in DB}
        h = lambda r, sel: np.mean([sharpe(r[c][sel(r.index)]) for c in r.columns])
        ew = {k: data[k][0].fillna(0.0).mean(axis=1) for k in DB}
        idx = ew["rrl"].index.union(ew["z"].index)
        d = ew["rrl"].reindex(idx).fillna(0.0) - ew["z"].reindex(idx).fillna(0.0)
        first = min(data["rrl"][1].dropna(how="all").index[0], data["z"][1].dropna(how="all").index[0])
        diffs[arm] = d[d.index >= first]
        ann = lambda x: float(x.mean() * 252 / INITIAL_CAPITAL * 100)
        row = {"arm": arm}
        for k in DB:
            row[f"Sharpe_{k}"] = s[k].Sharpe_Raw
            row[f"Entries_{k}"] = s[k].Entries
            row[f"WinRate_{k}"] = s[k].Win_Rate
            row[f"SL_{k}"] = s[k].Stop_Losses
            b, adj = beta_alpha(data[k][1].mean(axis=1), mkt)
            row[f"beta_{k}"], row[f"mktadj_Sharpe_{k}"] = b, adj
        row["dSharpe"] = row["Sharpe_rrl"] - row["Sharpe_z"]
        row["dSharpe_H2"] = h(data["rrl"][1], lambda i: i >= HALF) - h(data["z"][1], lambda i: i >= HALF)
        row["dSharpe_H1"] = h(data["rrl"][1], lambda i: i < HALF) - h(data["z"][1], lambda i: i < HALF)
        row["dSharpe_vs_RR"] = row["Sharpe_rrl"] - row["Sharpe_rr"]
        row["dEW_ann"] = ann(diffs[arm])
        rows.append(row)
        print(f"  {arm:<10} Sharpe 現行 {row['Sharpe_z']:+.3f}｜_RR {row['Sharpe_rr']:+.3f}｜RRL {row['Sharpe_rrl']:+.3f}"
              f"（Δ {row['dSharpe']:+.3f}；後半 {row['dSharpe_H2']:+.3f}）｜β {row['beta_z']:+.3f}/{row['beta_rr']:+.3f}/"
              f"{row['beta_rrl']:+.3f}｜扣市場 {row['mktadj_Sharpe_z']:+.3f}/{row['mktadj_Sharpe_rr']:+.3f}/"
              f"{row['mktadj_Sharpe_rrl']:+.3f}", flush=True)
    x = pd.DataFrame(rows)
    bt = [bootstrap_test(diffs[a].values) for a in ARMS]
    x["CI_lo"] = [b["CI下界"] for b in bt]
    x["CI_hi"] = [b["CI上界"] for b in bt]
    x["p"] = [b["BB p"] for b in bt]
    x["pBH"] = bh_adjust(x.p.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, index=False, encoding="utf-8-sig")

    pos, med, h2m = int((x.dSharpe > 0).sum()), float(x.dSharpe.median()), float(x.dSharpe_H2.median())
    bmed = float(x.beta_rrl.median())
    better = [med >= 0.05, pos >= 10, h2m > 0]
    worse = [med <= -0.05, pos <= 5, h2m < 0]
    if all(better):
        verdict = "殘差報酬訊號較佳" if bmed <= 0.02 else "曝險混淆，不成立"
    elif all(worse):
        verdict = "現行價差較佳"
    else:
        verdict = "無差異"
    print(f"\n== 判定（PREREGISTRATION.md §四）==\n中位 ΔSharpe {med:+.3f}｜為正 {pos}/15｜後半期中位 {h2m:+.3f}"
          f"｜RRL 市場 β 中位 {bmed:+.4f}"
          f"\n  較佳三條件 {['過' if v else '否' for v in better]}＋效度 {'過' if bmed <= 0.02 else '否'}"
          f"｜現行較佳三條件 {['過' if v else '否' for v in worse]}\n  → {verdict}")
    print("\n== 明細 ==")
    print(x[["arm", "Sharpe_z", "Sharpe_rr", "Sharpe_rrl", "dSharpe", "dSharpe_H1", "dSharpe_H2", "dSharpe_vs_RR"]]
          .round(4).to_string(index=False))
    print(x[["arm", "beta_z", "beta_rr", "beta_rrl", "mktadj_Sharpe_z", "mktadj_Sharpe_rr", "mktadj_Sharpe_rrl",
             "dEW_ann", "CI_lo", "CI_hi", "pBH"]].round(4).to_string(index=False))
    print(x[["arm", "Entries_z", "Entries_rrl", "WinRate_z", "WinRate_rrl", "SL_z", "SL_rrl"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
