"""TW63 對照 NOGRP-DTW：同期比較 + 前後半 + Newey-West。

交易期長度不同會使兩者的期數與交易日集合都不同（TW63 有 298 期、基準 295 期），
故一律截到共同的交易日區間再比較。
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, sqlite3
import statsmodels.api as sm
from regime_cost_dsr_eval import load_daily_returns, deflated_sharpe, TRIAL_CENSUS

CUT = pd.Timestamp("2014-01-01")
c = sqlite3.connect("file:results/result.db?mode=ro", uri=True)
d = pd.read_sql_query(
    "select METHOD,_path,Sharpe_Raw from strategy_summaries "
    "where METHOD in ('Grid (NOGRP-DTW)','Grid (NOGRP-DTW-TW63)')", c)
tw = d[d.METHOD == "Grid (NOGRP-DTW-TW63)"].copy()
bs = d[d.METHOD == "Grid (NOGRP-DTW)"].copy()
if tw.empty:
    sys.exit("TW63 尚無結果")
for x in (tw, bs):
    x["cfg"] = x._path.str.extract(r"TradeLogs_(Top\d+_SL\d+)_")[0]

R = load_daily_returns(sorted(set(tw._path) | set(bs._path)))
lo = max(min(s.dropna().index.min() for p, s in R.items() if p in set(tw._path)),
         min(s.dropna().index.min() for p, s in R.items() if p in set(bs._path)))
hi = min(max(s.dropna().index.max() for p, s in R.items() if p in set(tw._path)),
         max(s.dropna().index.max() for p, s in R.items() if p in set(bs._path)))
print(f"共同交易日區間 {lo.date()} ~ {hi.date()}\n")


def sh(s):
    s = s.dropna()
    return s.mean() / s.std(ddof=1) * np.sqrt(252) if len(s) > 100 else np.nan


rows = []
for cfg in sorted(set(tw.cfg) & set(bs.cfg), key=lambda x: (int(x.split("_")[0][3:]), x)):
    a = R[bs[bs.cfg == cfg]._path.iloc[0]]
    b = R[tw[tw.cfg == cfg]._path.iloc[0]]
    a, b = a[(a.index >= lo) & (a.index <= hi)], b[(b.index >= lo) & (b.index <= hi)]
    rows.append({"配置": cfg,
                 "基準W126": sh(a), "TW63": sh(b), "差": sh(b) - sh(a),
                 "基準前": sh(a[a.index < CUT]), "基準後": sh(a[a.index >= CUT]),
                 "TW63前": sh(b[b.index < CUT]), "TW63後": sh(b[b.index >= CUT])})
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
print(f"\nTW63 勝出 {(T['差'] > 0).sum()}/{len(T)} 格，中位差 {T['差'].median():+.4f}")
print(f"兩半皆正者：基準 {((T['基準前']>0)&(T['基準後']>0)).sum()}/{len(T)}，"
      f"TW63 {((T['TW63前']>0)&(T['TW63後']>0)).sum()}/{len(T)}")

for cfg in ("Top1_SL0",):
    a = R[bs[bs.cfg == cfg]._path.iloc[0]]
    b = R[tw[tw.cfg == cfg]._path.iloc[0]]
    X = pd.DataFrame({"a": a, "b": b}).dropna()
    X = X[(X.index >= lo) & (X.index <= hi)]
    dd = (X.b - X.a).values
    m = sm.OLS(dd, np.ones(len(dd))).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
    print(f"\n{cfg} 逐日報酬差（TW63 − 基準），n={len(dd)}")
    print(f"  年化 {dd.mean()*252*100:+.3f}%   Newey-West(21) t={m.tvalues[0]:+.3f}  p={m.pvalues[0]:.4f}")
    v = TRIAL_CENSUS["method"][1]
    for nm, s in (("基準W126", X.a), ("TW63", X.b)):
        o = deflated_sharpe(s, 46, v)
        print(f"  {nm:<10} Sharpe {o['SR_ann']:.4f}  DSR(N=46) {o['DSR']:.4f}")
