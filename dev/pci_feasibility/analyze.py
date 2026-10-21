# -*- coding: utf-8 -*-
"""部分共整合可行性診斷：分析（build.py 的輸出）。

可行性閘門（2026-10-21，寫於看到結果之前；通過才寫正式預先登記跑回測）：
  G1. 前 40 名內逐期 AUC（預測收斂）：R²_MR 的平均 − SSD 名次的平均 > 1 個標準誤（配對差的標準誤）；
  G2. 以 R²_MR 在前 40 名內重排取前 20，首筆交易報酬代理的平均 > SSD 前 20，且前、後半期（2014-01-02 起的形成期）皆成立；
  G3. 同上，在前 20 名內以 R²_MR 取前 3 vs SSD 前 3（最佳格為 Top3），前、後半期皆不劣。
另報：ρ、σ_R/σ_M 的 AUC；SSD 前 20 內以 R²_MR < 0.5 剔除後由 21–40 名遞補的版本。

    python dev/pci_feasibility/analyze.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HALF = "2014-01-02"


def per_period_auc(x, col, sign):
    out = []
    for ps, g in x.groupby("Period_Start"):
        g = g.dropna(subset=["conv"])
        if g.conv.nunique() < 2:
            continue
        out.append((ps, roc_auc_score(g.conv, sign * g[col])))
    return pd.DataFrame(out, columns=["Period_Start", col]).set_index("Period_Start")[col]


def pick(x, score, k, pool):
    """每期在 rank ≤ pool 內依 score 由大到小取前 k。"""
    g = x[x["rank"] <= pool].copy()
    g["_s"] = score(g)
    return g.sort_values(["Period_Start", "_s"], ascending=[True, False]).groupby("Period_Start").head(k)


def summ(sel):
    v = sel.dropna(subset=["conv"])
    h1, h2 = v[v.Period_Start < HALF], v[v.Period_Start >= HALF]
    return {"n": len(v), "conv": v.conv.mean(), "cap": v.cap.mean(),
            "cap_H1": h1.cap.mean(), "cap_H2": h2.cap.mean(), "conv_H1": h1.conv.mean(), "conv_H2": h2.conv.mean()}


def main():
    x = pd.read_pickle("results/analysis/pci_feasibility.pkl")
    x["sr_ratio"] = x.sigma_R / x.sigma_M
    print(f"期數 {x.Period_Start.nunique()}，候選列 {len(x)}，有效標籤 {x.conv.notna().sum()}，收斂率 {x.conv.mean():.3f}")
    print(f"R²_MR 分布：中位 {x.R2_MR.median():.3f}，<0.5 的比例 {(x.R2_MR < 0.5).mean():.1%}")

    a = pd.concat({"SSD 名次": per_period_auc(x, "rank", -1), "R²_MR": per_period_auc(x, "R2_MR", 1),
                   "ρ（低）": per_period_auc(x, "rho", -1), "σR/σM（低）": per_period_auc(x, "sr_ratio", -1)}, axis=1).dropna()
    print("\n== 前 40 名內逐期 AUC（預測收斂）==")
    for c in a.columns:
        h1, h2 = a[c][a.index < HALF], a[c][a.index >= HALF]
        print(f"  {c:<10} 平均 {a[c].mean():.4f} ± {a[c].std() / np.sqrt(len(a)):.4f}｜前半 {h1.mean():.4f}｜後半 {h2.mean():.4f}")
    d = a["R²_MR"] - a["SSD 名次"]
    se = d.std() / np.sqrt(len(d))
    g1 = d.mean() > se
    print(f"  R²_MR − SSD：{d.mean():+.4f} ± {se:.4f}（{len(d)} 期）→ G1 {'過' if g1 else '否'}")

    print("\n== 選取（首筆交易報酬代理，每單位資金）==")
    rows = {
        "SSD 前 20": summ(x[x["rank"] <= 20]),
        "R²_MR 重排前 40 取 20": summ(pick(x, lambda g: g.R2_MR, 20, 40)),
        "SSD 前 20 剔除 R²<0.5、21–40 遞補": summ(pick(x, lambda g: (g.R2_MR >= 0.5) * 1000 - g["rank"], 20, 40)),
        "SSD 前 3": summ(x[x["rank"] <= 3]),
        "R²_MR 於前 20 取 3": summ(pick(x, lambda g: g.R2_MR, 3, 20)),
        "SSD 前 5": summ(x[x["rank"] <= 5]),
        "R²_MR 於前 20 取 5": summ(pick(x, lambda g: g.R2_MR, 5, 20)),
    }
    t = pd.DataFrame(rows).T
    print(t.round(4).to_string())
    r20, s20 = rows["R²_MR 重排前 40 取 20"], rows["SSD 前 20"]
    g2 = r20["cap"] > s20["cap"] and r20["cap_H1"] > s20["cap_H1"] and r20["cap_H2"] > s20["cap_H2"]
    r3, s3 = rows["R²_MR 於前 20 取 3"], rows["SSD 前 3"]
    g3 = r3["cap_H1"] >= s3["cap_H1"] and r3["cap_H2"] >= s3["cap_H2"]
    print(f"\nG1 {'過' if g1 else '否'}｜G2 {'過' if g2 else '否'}｜G3 {'過' if g3 else '否'} → "
          f"{'可行，寫正式預先登記' if (g1 and g2 and g3) else '不可行，不進入回測'}")

    print("\n== R²_MR 五分位（前 40 名全部）==")
    q = pd.qcut(x.R2_MR.rank(method="first"), 5, labels=False)
    print(x.groupby(q).agg(R2=("R2_MR", "median"), conv=("conv", "mean"), cap=("cap", "mean"),
                           rank=("rank", "mean")).round(4).to_string())
    print("\ncorr(R²_MR, SSD 名次) =", round(x[["R2_MR", "rank"]].corr(method="spearman").iloc[0, 1], 3))


if __name__ == "__main__":
    main()
