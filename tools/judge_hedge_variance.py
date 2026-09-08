#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判斷 signal 對沖修正的 +0.0838 是否超出重訓雜訊。

判準見 `dev/drl_hedge/PREREGISTRATION.md` §七 與「七之補」（**皆跑前定稿**）：

    主判準  120 格中「修正前 Sharpe < 修正後 k 輪最小值」的比例，
            單尾二項檢定對 p0 = 1/(k+1)，alpha = 0.05
    方向    「修正前 > k 輪最大值」的比例須明顯低於下界比例
    分臂    8 支臂各自列出；若位移只來自 1-2 支臂，只可宣稱那幾支

為何 p0 = 1/(k+1)：k 個可交換抽樣把數線切成 k+1 段，第 k+1 點落在
最小值以下的機率恰為 1/(k+1)。故逐格達標本身不是證據，要看比例。

⚠ `drl_hedge_post.csv` 是**被檢定的那一點**，不併入參考分布。

2026-09-05 實測：`run_drl_variance` 在 `RUNS_CSV` 不存在時，會把 `result.db`
既有的列直接 `harvest(1, ...)` 當第 1 輪（見該檔的 elif 分支），而當時 DB 裡
正是 `rerun_drl_hedge.py` 的修正後結果 —— 故**第 1 輪逐位等於 post.csv**。
本檔自動偵測並剔除與 post 相同的輪次，參考分布只取其餘 k 輪。
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A = "results/analysis"
N_RUNS = 5          # 收成的輪數（含可能與 post 重複者）
MIN_REF = 3         # 剔除重複後，參考分布至少要這麼多輪才判定


def load_runs() -> pd.DataFrame:
    parts = []
    for tag in ("drlhedge", "rlhedge"):
        f = f"{A}/drl_variance_runs_{tag}.csv"
        if not os.path.exists(f):
            sys.exit(f"缺 {f} —— 雜訊量測未跑完，不可判定。")
        parts.append(pd.read_csv(f))
    d = pd.concat(parts, ignore_index=True)
    bad = d.groupby("_path").run_id.nunique()
    bad = bad[bad != N_RUNS]
    if len(bad):
        sys.exit(f"以下 {len(bad)} 格的輪數不是 {N_RUNS}，不可判定：\n{bad.head(10)}")
    return d


def main() -> int:
    runs = load_runs()
    pre = pd.read_csv(f"{A}/drl_hedge_pre_dollar.csv").set_index("_path")
    post = pd.read_csv(f"{A}/drl_hedge_post.csv").set_index("_path")

    g = runs.groupby("_path").Sharpe_Raw
    d = pd.DataFrame({"lo": g.min(), "hi": g.max(), "med": g.median(),
                      "sd": g.std(ddof=1)})
    d["METHOD"] = runs.groupby("_path").METHOD.first()
    d["pre"] = pre.Sharpe_Raw
    d["post"] = post.Sharpe_Raw
    if d[["pre", "post"]].isna().any().any():
        sys.exit("pre/post 與雜訊輪次的 _path 對不上。")

    # 補三：與 post 逐位相同的輪次必須剔除 —— 它是被檢定的那一點本身
    eq = runs.assign(_post=runs._path.map(d.post))
    eq["same"] = np.isclose(eq.Sharpe_Raw, eq._post, atol=1e-12)
    per_run = eq.groupby("run_id").same.sum()
    dup = sorted(per_run[per_run == len(d)].index)
    part = per_run[(per_run > 0) & (per_run < len(d))]
    if len(part):
        sys.exit("以下輪次只有部分格與 post 相同，成因不明，不可自動剔除：" + str(part))
    print(f"與 post.csv 逐位相同的輪次：{dup if dup else '無'}"
          f"（每輪相同格數 {per_run.to_dict()}）")
    if dup:
        runs = runs[~runs.run_id.isin(dup)]
        g = runs.groupby("_path").Sharpe_Raw
        d["lo"], d["hi"] = g.min(), g.max()
        d["med"], d["sd"] = g.median(), g.std(ddof=1)
    k = int(runs.run_id.nunique())
    p0 = 1.0 / (k + 1)
    if k < MIN_REF:
        sys.exit(f"剔除後只剩 {k} 輪參考分布，低於 MIN_REF={MIN_REF}，不判定。")
    print(f"參考分布：第 {sorted(runs.run_id.unique())} 輪，"
          f"k={k} -> p0 = 1/{k + 1} = {p0:.4f}")

    d["below"] = d.pre < d.lo
    d["above"] = d.pre > d.hi
    n = len(d)
    nb, na = int(d.below.sum()), int(d.above.sum())
    p = stats.binomtest(nb, n, p0, alternative="greater").pvalue

    print(f"\n==== 主判準（n={n} 格，k={k} 輪，p0=1/{k+1}={p0:.4f}）====")
    print(f"修正前 < {k} 輪最小值：{nb}/{n} = {nb/n:.1%}   單尾二項 p = {p:.3g}")
    print(f"修正前 > {k} 輪最大值：{na}/{n} = {na/n:.1%}   （反向，作對照）")
    print(f"雜訊本身（k 輪）：跨輪 sd 中位 {d.sd.median():.4f}、全距中位 {(d.hi-d.lo).median():.4f}")
    print(f"pre→post ΔSharpe 中位 {(d.post-d.pre).median():+.4f}、"
          f"為正 {int((d.post>d.pre).sum())}/{n}")

    passed = (p < 0.05) and (nb > na)
    print(f"\n跑前門檻：p<0.05 且 下界比例明顯高於上界比例")
    print(f"=> **{'超出重訓雜訊，可宣稱位移' if passed else '在重訓雜訊範圍內，不可歸因'}**")

    print(f"\n==== 分臂 ====")
    arm = d.groupby("METHOD").agg(n=("below", "size"), below=("below", "sum"),
                                  above=("above", "sum"), sd=("sd", "median"))
    arm["frac"] = arm.below / arm.n
    arm["p"] = [stats.binomtest(int(b), int(m), p0, alternative="greater").pvalue
                for b, m in zip(arm.below, arm.n)]
    print(arm.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n==== 附帶（描述性，不作判準）：輪間兩兩 ΔSharpe ====")
    w = runs.pivot_table(index="_path", columns="run_id", values="Sharpe_Raw")
    pair = np.concatenate([(w[j] - w[i]).to_numpy()
                           for i in w.columns for j in w.columns if j > i])
    print(f"  {len(pair)} 對：中位 {np.median(pair):+.4f}  "
          f"為正 {int((pair>0).sum())}/{len(pair)} ({(pair>0).mean():.0%})  "
          f"sd {pair.std(ddof=1):.4f}  全距 [{pair.min():+.4f}, {pair.max():+.4f}]")
    print(f"  對照 pre→post：為正 {(d.post>d.pre).mean():.0%}")

    d.reset_index().to_csv(f"{A}/drl_hedge_variance_judgement.csv",
                           index=False, encoding="utf-8-sig")
    print(f"\n-> {A}/drl_hedge_variance_judgement.csv")
    return 0 if passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
