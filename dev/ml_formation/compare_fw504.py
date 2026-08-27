"""FW504 對照 GICS-SSD：同期比較。

W=504 使 2001 全年 12 個視窗歷史不足（run_formation 的守衛），故基準臂
必須截到相同起點重算，否則多出來的一年會混進比較。
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "analysis")
from strategies.metrics import metrics_from_returns
import numpy as np, pandas as pd, sqlite3
from regime_cost_dsr_eval import load_daily_returns, deflated_sharpe, TRIAL_CENSUS

c = sqlite3.connect("file:results/result.db?mode=ro", uri=True)
d = pd.read_sql_query(
    "select METHOD,_path,Sharpe_Raw,Ann_Ret_Raw from strategy_summaries "
    "where METHOD in ('Grid (GICS-SSD)','Grid (GICS-SSD-FW504)')", c)
if d[d.METHOD == "Grid (GICS-SSD-FW504)"].empty:
    sys.exit("FW504 尚無結果")

fw = d[d.METHOD == "Grid (GICS-SSD-FW504)"].copy()
bs = d[(d.METHOD == "Grid (GICS-SSD)") & (~d._path.str.contains("_EZ"))].copy()
for x in (fw, bs):
    x["cfg"] = x._path.str.extract(r"TradeLogs_(Top\d+_SL\d+)_")[0]

R = load_daily_returns(sorted(set(fw._path) | set(bs._path)))
start = min(s.dropna().index.min() for p, s in R.items() if p in set(fw._path))
print(f"FW504 起始交易日 {start.date()}；基準臂一律截自同日重算\n")


def sharpe(s):
    # 原有 `len(s) > 20` 守衛已移除：實測 345 條序列最短 6,035 日，守衛從未觸發。
    # 口徑改由 strategies/metrics.py 統一（此前 fw504 用 >20、tw63 用 >100、
    # regime_cost_dsr_eval 無守衛——三者並存而無人察覺，正因它們都不會生效）。
    return metrics_from_returns(s)['Sharpe_Raw']


rows = []
for cfg in sorted(set(fw.cfg) & set(bs.cfg), key=lambda x: (int(x.split("_")[0][3:]), x)):
    pf = fw[fw.cfg == cfg]._path.iloc[0]
    pb = bs[bs.cfg == cfg]._path.iloc[0]
    a = R[pb][R[pb].index >= start]
    b = R[pf][R[pf].index >= start]
    rows.append({"配置": cfg,
                 "基準(全期)": bs[bs.cfg == cfg].Sharpe_Raw.iloc[0],
                 "基準(同期)": sharpe(a), "FW504": sharpe(b),
                 "差": sharpe(b) - sharpe(a)})
T = pd.DataFrame(rows)
print(T.round(4).to_string(index=False))
print(f"\n同期比較：FW504 勝出 {(T['差'] > 0).sum()}/{len(T)} 格，中位差 {T['差'].median():+.4f}")

i = T.FW504.idxmax()
print(f"\nFW504 最佳格 {T.配置[i]}  Sharpe {T.FW504[i]:.4f}"
      f"（同期基準 {T['基準(同期)'][i]:.4f}）")
v = TRIAL_CENSUS["method"][1]
best = fw[fw.cfg == T.配置[i]]._path.iloc[0]
o = deflated_sharpe(R[best][R[best].index >= start], 45, v)
print(f"  DSR(N=45, 同期 T={o['T']}) = {o['DSR']:.4f}   SR0 {o['SR0_ann']:.4f}")
