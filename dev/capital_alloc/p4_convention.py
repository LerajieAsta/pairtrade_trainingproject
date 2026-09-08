# -*- coding: utf-8 -*-
"""P4：計量約定本身值多少（預先註冊 §五 P4）。

只跑 ELIM 這一組權重，兩種口徑：

    甲（本專案主口徑）  空倉日補 0，計入分母
    乙（原文約定）      刪除加權後完全空倉的月份

直接檢驗預先註冊 §三保留二：原文 MDD −1.01% ~ −2.47% 是否來自
「未持倉月份以 NA 補齊、不納入分母」。

不含重抽，故比 stage1 快得多。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "ca_stage1", os.path.join(_ROOT, "dev", "capital_alloc", "stage1.py"))
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)

from strategies.metrics import concurrent_of, metrics_from_pnl  # noqa: E402


def main() -> int:
    rows = []
    for _, r in s1.cells().iterrows():
        df = s1.load_logs(r._path)
        if df.empty:
            continue
        lag = max(1, concurrent_of(r._path))
        _, pos, pp, M, V = s1._trailing(df, lag)
        gid = s1.row_gid(df, pp)

        sc = s1.build_scales(df, pos, pp, M, V, "ELIM", lag)
        pnl = s1.apply_scales(df, gid, sc)
        a = metrics_from_pnl(pnl)

        cut = s1.drop_flat_months(pnl, df, gid, sc)
        b = metrics_from_pnl(cut)

        n_all = len(pnl.resample("ME").last())
        n_kept = len(cut.resample("ME").last()) if len(cut) else 0
        rows.append({
            "arm": r.METHOD.replace("Grid ", ""), "top_n": r.top_n,
            "months_all": n_all, "months_kept": n_kept,
            "months_dropped": n_all - n_kept,
            "MDD_full": a.MDD_Raw, "MDD_drop": b.MDD_Raw,
            "Ann_full": a.Ann_Ret_Raw, "Ann_drop": b.Ann_Ret_Raw,
            "Sharpe_full": a.Sharpe_Raw, "Sharpe_drop": b.Sharpe_Raw,
        })
        print(f"  {rows[-1]['arm']:<14} Top{r.top_n:<3} "
              f"刪 {rows[-1]['months_dropped']:>3}/{n_all} 個月  "
              f"MDD {a.MDD_Raw:+.4f} -> {b.MDD_Raw:+.4f}  "
              f"年化 {a.Ann_Ret_Raw*100:+.3f}% -> {b.Ann_Ret_Raw*100:+.3f}%", flush=True)

    d = pd.DataFrame(rows)
    d["MDD_x"] = d.MDD_full / d.MDD_drop
    d["dAnn_pp"] = (d.Ann_drop - d.Ann_full) * 100
    d.to_csv("results/analysis/capital_alloc_p4.csv", index=False, encoding="utf-8-sig")

    print("\n==== P4 ====")
    print(f"刪除月數：中位 {d.months_dropped.median():.0f} / {d.months_all.median():.0f}")
    print(f"MDD 改善倍數：中位 {d.MDD_x.median():.3f}  最大 {d.MDD_x.max():.3f}")
    print(f"年化上偏：中位 {d.dAnn_pp.median():+.3f}pp  最大 {d.dAnn_pp.max():+.3f}pp")
    hit = int((d.MDD_x >= 2.0).sum())
    print(f"\n預測『至少一支臂 MDD 改善 ≥2 倍』 → 達標 {hit}/13 → "
          f"{'成立' if hit >= 1 else '不成立'}")
    print("-> results/analysis/capital_alloc_p4.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
