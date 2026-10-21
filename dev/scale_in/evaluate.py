# -*- coding: utf-8 -*-
"""分層加碼的評估（判準見同目錄 PREREGISTRATION.md §四）。

  主要：逐組 ΔSharpe = (15 格 Sharpe_Raw 平均)_分層 − (15 格 Sharpe_Raw 平均)_現行，n = 8。
  後半期：同定義，Sharpe 由 2014-01-02 起的逐日複利報酬重算（strategies.returns）。
  另報：15 格等權逐日損益差分（4.3 同程序，8 組 BH）、組合市場 β、rf 超額、停損次數、
        第二層觸發率（進場即整份 vs 持有中加碼）、實際平均曝險、依是否加碼分組的平均交易損益（Top20/SL0）。

    python dev/scale_in/evaluate.py
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

BASE_DB, SI_DB = "results/result.db", "results/scale_in.db"
OUT = "results/analysis/scale_in_eval.csv"
HALF = pd.Timestamp("2014-01-02")
SOURCES = ["GICS_SSD", "GICS_SDP", "HDB_SDP", "KM_SSD", "AGG_SSD", "NOGRP_SSD", "NOGRP_DTW", "NOGRP_SDP"]
CELLS = [(t, s) for t in (1, 3, 5, 10, 20) for s in (0, 5, 15)]
SUFFIX = {"base": "", "si": "_SI30"}


def path(src, kind, tn, sl):
    return f"tiingo/Grid_{src}/TradeLogs_Top{tn}_SL{sl}_ZWin0_MSR0{SUFFIX[kind]}.csv"


def summ(db, src, kind):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    ps = [path(src, kind, tn, sl) for tn, sl in CELLS]
    d = pd.read_sql(f"SELECT Sharpe_Raw, Excess_Ret_RF, Stop_Losses, Entries FROM strategy_summaries "
                    f"WHERE _path IN ({','.join('?' * len(ps))})", con, params=ps)
    con.close()
    if len(d) != len(CELLS):
        raise RuntimeError(f"{db} {src}-{kind} 只有 {len(d)}/15 格")
    return d.mean()


def cells(db, src, kind):
    ps = [path(src, kind, tn, sl) for tn, sl in CELLS]
    pnl = daily_pnl(ps, db, use_cache=False)
    if pnl.shape[1] != len(CELLS):
        raise RuntimeError(f"{db} {src}-{kind} 逐日紀錄只有 {pnl.shape[1]}/15 格")
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


def tier_diag(src):
    """Top20/SL0：逐筆交易的最高層數與交易損益；實際平均曝險（第一層 0.5、第二層 1.0）。"""
    con = sqlite3.connect(f"file:{SI_DB}?mode=ro", uri=True, timeout=600)
    d = pd.read_sql("SELECT Period_Start, Ticker_A, Ticker_B, Date, Position, Status, SI_Tier, Trade_PnL "
                    "FROM trade_logs WHERE strategy_id=? ORDER BY Period_Start, Ticker_A, Ticker_B, Date",
                    con, params=(path(src, "si", 20, 0),))
    con.close()
    key = ["Period_Start", "Ticker_A", "Ticker_B"]
    d["entry"] = d.Status.str.startswith("ENTER")
    d["tid"] = d.groupby(key).entry.cumsum()
    open_ = d[(d.Position != 0) & (d.tid > 0)]
    expo = float(open_.SI_Tier.map({1: 0.5, 2: 1.0}).mean())
    g = d[d.tid > 0].groupby(key + ["tid"])
    t = pd.DataFrame({"full_at_entry": g.apply(lambda x: x.SI_Tier.iloc[0] == 2),
                      "scaled": g.apply(lambda x: (x.Status == "SCALE_IN").any()),
                      "pnl": g.Trade_PnL.sum()})
    one = t[~t.full_at_entry & ~t.scaled]
    return {"trades": len(t), "full_at_entry": float(t.full_at_entry.mean()), "scaled_later": float(t.scaled.mean()),
            "avg_exposure_open": expo,
            "pnl_tier1_only": float(one.pnl.mean()), "pnl_scaled_later": float(t[t.scaled].pnl.mean()),
            "pnl_full_at_entry": float(t[t.full_at_entry].pnl.mean())}


def main():
    mkt = market()
    rows, diffs = [], {}
    for src in SOURCES:
        sp, ss = summ(SI_DB, src, "si"), summ(BASE_DB, src, "base")
        (pp, rp), (pb, rb) = cells(SI_DB, src, "si"), cells(BASE_DB, src, "base")
        h = lambda r, sel: np.mean([sharpe(r[c][sel(r.index)]) for c in r.columns])
        ep, eb = pp.fillna(0.0).mean(axis=1), pb.fillna(0.0).mean(axis=1)
        idx = ep.index.union(eb.index)
        dd = ep.reindex(idx).fillna(0.0) - eb.reindex(idx).fillna(0.0)
        first = min(rp.dropna(how="all").index[0], rb.dropna(how="all").index[0])
        diffs[src] = dd[dd.index >= first]
        m = mkt.reindex(idx).fillna(0.0)
        ann = lambda s: float(s.mean() * 252 / INITIAL_CAPITAL * 100)
        rows.append({"source": src, "Sharpe_SI": sp.Sharpe_Raw, "Sharpe_Z": ss.Sharpe_Raw,
                     "dSharpe": sp.Sharpe_Raw - ss.Sharpe_Raw,
                     "dSharpe_H1": h(rp, lambda i: i < HALF) - h(rb, lambda i: i < HALF),
                     "dSharpe_H2": h(rp, lambda i: i >= HALF) - h(rb, lambda i: i >= HALF),
                     "RF_SI": sp.Excess_Ret_RF, "RF_Z": ss.Excess_Ret_RF,
                     "SL_SI": sp.Stop_Losses, "SL_Z": ss.Stop_Losses,
                     "dEW_ann": ann(diffs[src]), "dEW_ann_H2": ann(diffs[src][diffs[src].index >= HALF]),
                     "beta_SI": beta(rp.mean(axis=1), m), "beta_Z": beta(rb.mean(axis=1), m),
                     **tier_diag(src)})
        r = rows[-1]
        print(f"  {src:<10} Sharpe 分層 {r['Sharpe_SI']:+.3f} vs 現行 {r['Sharpe_Z']:+.3f}（Δ {r['dSharpe']:+.3f}；"
              f"後半 {r['dSharpe_H2']:+.3f}）｜β {r['beta_SI']:+.3f}/{r['beta_Z']:+.3f}｜曝險 {r['avg_exposure_open']:.2f}"
              f"｜加碼 {r['scaled_later']:.1%}、進場即整份 {r['full_at_entry']:.1%}", flush=True)
    x = pd.DataFrame(rows)
    bt = [bootstrap_test(diffs[s].values) for s in SOURCES]
    x["CI_lo"] = [b["CI下界"] for b in bt]
    x["CI_hi"] = [b["CI上界"] for b in bt]
    x["p"] = [b["BB p"] for b in bt]
    x["pBH"] = bh_adjust(x.p.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, index=False, encoding="utf-8-sig")

    pos, med, h2m = int((x.dSharpe > 0).sum()), float(x.dSharpe.median()), float(x.dSharpe_H2.median())
    better = [pos >= 6, med >= 0.05, h2m > 0]
    worse = [pos <= 2, med <= -0.05, h2m < 0]
    verdict = "分層加碼較佳" if all(better) else ("現行較佳" if all(worse) else "無差異")
    print(f"\n== 判定（PREREGISTRATION.md §四）==\nΔSharpe 為正 {pos}/8｜中位 {med:+.3f}｜後半期中位 {h2m:+.3f}"
          f"\n  較佳三條件 {['過' if v else '否' for v in better]}｜現行較佳三條件 {['過' if v else '否' for v in worse]}"
          f"\n  → {verdict}")
    print("\n== 明細 ==")
    print(x[["source", "Sharpe_SI", "Sharpe_Z", "dSharpe", "dSharpe_H1", "dSharpe_H2", "RF_SI", "RF_Z",
             "SL_SI", "SL_Z"]].round(4).to_string(index=False))
    print(x[["source", "dEW_ann", "CI_lo", "CI_hi", "pBH", "dEW_ann_H2", "beta_SI", "beta_Z"]]
          .round(4).to_string(index=False))
    print(x[["source", "trades", "full_at_entry", "scaled_later", "avg_exposure_open", "pnl_tier1_only",
             "pnl_scaled_later", "pnl_full_at_entry"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
