# -*- coding: utf-8 -*-
"""獲利潛力排序的評估（判準見同目錄 PREREGISTRATION.md §四）。

  主要：逐臂 ΔSharpe = (15 格 Sharpe_Raw 平均)_POT − (15 格 Sharpe_Raw 平均)_SSD，n = 5。
  後半期：同定義，但 Sharpe 由 2014-01-02 起的逐日複利報酬（`strategies.returns.daily_pnl` → `returns_from_pnl`）重算。
  另報：15 格等權逐日差分（4.3 同程序：循環區塊自助法 L = 126、B = 10,000，5 組 BH）、組合市場 β、
        rf 超額、入選配對重疊率、每期入選數不足 20 的比例。

    python dev/profit_potential/evaluate.py
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

BASE_DB, POT_DB = "results/result.db", "results/profit_potential.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
OUT = "results/analysis/profit_potential_eval.csv"
HALF = pd.Timestamp("2014-01-02")
GROUPS = ["NOGRP", "GICS", "HDB", "AGG", "KM"]
CELLS = [(t, s) for t in (1, 3, 5, 10, 20) for s in (0, 5, 15)]


def path(g, kind, tn, sl):
    return f"tiingo/Grid_{g}_{kind}/TradeLogs_Top{tn}_SL{sl}_ZWin0_MSR0.csv"


def summ(db, g, kind):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    ps = [path(g, kind, tn, sl) for tn, sl in CELLS]
    d = pd.read_sql(f"SELECT Sharpe_Raw, Excess_Ret_RF FROM strategy_summaries WHERE _path IN "
                    f"({','.join('?' * len(ps))})", con, params=ps)
    con.close()
    if len(d) != len(CELLS):
        raise RuntimeError(f"{db} {g}-{kind} 只有 {len(d)}/15 格")
    return float(d.Sharpe_Raw.mean()), float(d.Excess_Ret_RF.mean())


def cells(db, g, kind):
    ps = [path(g, kind, tn, sl) for tn, sl in CELLS]
    pnl = daily_pnl(ps, db, use_cache=False)
    if pnl.shape[1] != len(CELLS):
        raise RuntimeError(f"{db} {g}-{kind} 逐日紀錄只有 {pnl.shape[1]}/15 格")
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


def beta(y, m):
    ok = y.notna()
    x = np.column_stack([np.ones(ok.sum()), m[ok].values])
    return float(np.linalg.lstsq(x, y[ok].values, rcond=None)[0][1])


def formation_diag(g):
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    q = "SELECT Period_Start, Ticker_A, Ticker_B FROM formation_pairs WHERE strategy_id=?"
    pot = pd.read_sql(q, con, params=(f"Grid {g}-POT_MSR0",))
    ssd = pd.read_sql(q, con, params=(f"Grid {g}-SSD_MSR0",))
    con.close()
    key = lambda d: d.groupby("Period_Start").apply(lambda x: set(zip(x.Ticker_A, x.Ticker_B)))
    kp, ks = key(pot), key(ssd)
    common = kp.index.intersection(ks.index)
    overlap = float(np.mean([len(kp[p] & ks[p]) / max(len(kp[p]), 1) for p in common]))
    n = pot.groupby("Period_Start").size()
    return {"periods_POT": len(kp), "overlap_with_SSD": overlap,
            "short_of_20": float((n < 20).mean()), "median_n": float(n.median())}


def main():
    mkt = market()
    rows, diffs = [], {}
    for g in GROUPS:
        sh_p, rf_p = summ(POT_DB, g, "POT")
        sh_s, rf_s = summ(BASE_DB, g, "SSD")
        (pp, rp), (ps_, rs) = cells(POT_DB, g, "POT"), cells(BASE_DB, g, "SSD")
        h2 = lambda r: np.mean([sharpe(r[c][r.index >= HALF]) for c in r.columns])
        h1 = lambda r: np.mean([sharpe(r[c][r.index < HALF]) for c in r.columns])
        # 等權逐日損益差分（4.3）：生命期外損益為 0
        ep, es = pp.fillna(0.0).mean(axis=1), ps_.fillna(0.0).mean(axis=1)
        idx = ep.index.union(es.index)
        d = ep.reindex(idx).fillna(0.0) - es.reindex(idx).fillna(0.0)
        first = min(rp.dropna(how="all").index[0], rs.dropna(how="all").index[0])
        diffs[g] = d[d.index >= first]
        m = mkt.reindex(idx).fillna(0.0)
        ann = lambda s: float(s.mean() * 252 / INITIAL_CAPITAL * 100)
        rows.append({"group": g, "Sharpe_POT": sh_p, "Sharpe_SSD": sh_s, "dSharpe": sh_p - sh_s,
                     "dSharpe_H1": h1(rp) - h1(rs), "dSharpe_H2": h2(rp) - h2(rs),
                     "RF_POT": rf_p, "RF_SSD": rf_s,
                     "dEW_ann": ann(diffs[g]), "dEW_ann_H2": ann(diffs[g][diffs[g].index >= HALF]),
                     "beta_POT": beta(rp.mean(axis=1), m), "beta_SSD": beta(rs.mean(axis=1), m),
                     **formation_diag(g)})
        r = rows[-1]
        print(f"  {g:<6} Sharpe POT {sh_p:+.3f} vs SSD {sh_s:+.3f}（Δ {r['dSharpe']:+.3f}；後半 "
              f"{r['dSharpe_H2']:+.3f}）｜β {r['beta_POT']:+.3f}/{r['beta_SSD']:+.3f}｜重疊 "
              f"{r['overlap_with_SSD']:.1%}｜不足20 {r['short_of_20']:.1%}", flush=True)
    x = pd.DataFrame(rows)
    bt = [bootstrap_test(diffs[g].values) for g in GROUPS]
    x["CI_lo"] = [b["CI下界"] for b in bt]
    x["CI_hi"] = [b["CI上界"] for b in bt]
    x["p"] = [b["BB p"] for b in bt]
    x["pBH"] = bh_adjust(x.p.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, index=False, encoding="utf-8-sig")

    pos, med, h2m = int((x.dSharpe > 0).sum()), float(x.dSharpe.median()), float(x.dSharpe_H2.median())
    better = [pos >= 4, med >= 0.05, h2m > 0]
    worse = [pos <= 1, med <= -0.05, h2m < 0]
    verdict = "獲利潛力排序較佳" if all(better) else ("SSD 較佳" if all(worse) else "無差異")
    print(f"\n== 判定（PREREGISTRATION.md §四）==\nΔSharpe 為正 {pos}/5｜中位 {med:+.3f}｜後半期中位 {h2m:+.3f}"
          f"\n  較佳三條件 {['過' if v else '否' for v in better]}｜SSD 較佳三條件 {['過' if v else '否' for v in worse]}"
          f"\n  → {verdict}")
    print("\n== 明細 ==")
    print(x[["group", "Sharpe_POT", "Sharpe_SSD", "dSharpe", "dSharpe_H1", "dSharpe_H2", "RF_POT", "RF_SSD"]]
          .round(4).to_string(index=False))
    print(x[["group", "dEW_ann", "CI_lo", "CI_hi", "pBH", "dEW_ann_H2", "beta_POT", "beta_SSD",
             "overlap_with_SSD", "short_of_20", "median_n"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
