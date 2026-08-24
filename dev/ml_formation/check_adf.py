"""驗證批次 ADF 與 statsmodels 逐一計算是否數值一致。"""
import sys; sys.path.insert(0, ".")
import numpy as np
from strategies.formation._utils import _adf_stat
from dev.ml_formation.adf import adf_stat_batch, eg_critical_value
from statsmodels.tsa.adfvalues import mackinnonp

rng = np.random.default_rng(7)
# 混合：純隨機漫步、均值回歸、趨勢殘差
seqs = []
for _ in range(80):
    seqs.append(np.cumsum(rng.normal(size=252)))
for _ in range(80):
    x = np.zeros(252)
    for t in range(1, 252):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    seqs.append(x)
for _ in range(40):
    x = np.zeros(252)
    for t in range(1, 252):
        x[t] = 0.99 * x[t - 1] + rng.normal()
    seqs.append(x)
Y = np.array(seqs)

mine = adf_stat_batch(Y)
ref = np.array([_adf_stat(y, max_lags=1)[0] for y in Y])
d = np.abs(mine - ref)
print(f"n = {len(Y)}   統計量最大絕對差 {d.max():.3e}   中位 {np.median(d):.3e}")

crit = eg_critical_value(0.05, 2)
print(f"\nEG 臨界值（alpha=0.05, N=2）= {crit:.4f}")
print(f"  mackinnonp({crit:.4f}) = {mackinnonp(crit, regression='c', N=2):.6f}")
print(f"  docstring 校準點 -3.337 → {mackinnonp(-3.337, regression='c', N=2):.4f}（應為 0.0498）")
print(f"                  -3.899 → {mackinnonp(-3.899, regression='c', N=2):.4f}（應為 0.0100）")

pv = np.array([_adf_stat(y, max_lags=1)[1] for y in Y])
agree = ((pv < 0.05) == (mine < crit)).mean()
print(f"\n『p<0.05』與『stat<臨界值』判定一致率: {100*agree:.2f}%  (n={len(Y)})")
