# -*- coding: utf-8 -*-
"""P0：配對層級績效的持續性（動量加權的前提）。

預先註冊 §五 P0。**這不是階段一的替代品** —— 無論本檔結果為何，
MO 都要實跑，否則就是拿代理當結果。

量的是：同一 ticker 配對在期 $k$ 的損益，與其在期 $k$ 開始前**已平倉**的
最近 12 期損益總和之間的 Spearman $\rho$。

落後 `lag` 期的必要性見 `stage1._trailing` 的說明：組合重疊 6 期，
「Period_Start 早於本期」不等於「已知」。
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ca_stage1", os.path.join(_ROOT, "dev", "capital_alloc", "stage1.py"))
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)

from strategies.metrics import concurrent_of  # noqa: E402


def main() -> int:
    cl = s1.cells()
    rows, pooled = [], []
    for _, r in cl.iterrows():
        df = s1.load_logs(r._path)
        if df.empty:
            continue
        lag = max(1, concurrent_of(r._path))
        _, _, pp, M, _ = s1._trailing(df, lag)

        m = np.array([M.get((p, pi), np.nan) for p, pi in zip(pp.pair, pp.pi)])
        y = pp.pnl.to_numpy()
        ok = ~np.isnan(m) & ~np.isnan(y)
        n = int(ok.sum())
        if n < 30:
            continue
        rho, p = stats.spearmanr(m[ok], y[ok])
        rows.append({"METHOD": r.METHOD, "top_n": r.top_n, "lag": lag,
                     "n": n, "rho": float(rho), "p": float(p)})
        pooled.append(pd.DataFrame({"m": m[ok], "y": y[ok]}))
        print(f"  {r.METHOD:<20} Top{r.top_n:<3} n={n:>5}  rho={rho:+.4f}  p={p:.3f}",
              flush=True)

    d = pd.DataFrame(rows)
    allp = pd.concat(pooled, ignore_index=True)
    rho_all, p_all = stats.spearmanr(allp.m, allp.y)

    print("\n==== P0 ====")
    print(f"逐格 rho：中位 {d.rho.median():+.4f}  範圍 [{d.rho.min():+.4f}, {d.rho.max():+.4f}]")
    print(f"為正的格數：{int((d.rho > 0).sum())}/{len(d)}")
    print(f"p<0.05 的格數：{int((d.p < 0.05).sum())}/{len(d)}")
    print(f"合併（{len(allp)} 個配對-期）：rho={rho_all:+.4f}  p={p_all:.3g}")
    print(f"\n預測區間 [-0.05, +0.05] —— 合併 rho {'落在區間內' if abs(rho_all) <= 0.05 else '落在區間外'}")

    d.to_csv("results/analysis/capital_alloc_persistence.csv",
             index=False, encoding="utf-8-sig")
    print("-> results/analysis/capital_alloc_persistence.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
