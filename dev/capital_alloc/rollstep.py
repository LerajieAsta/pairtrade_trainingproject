# -*- coding: utf-8 -*-
"""延伸二：拉長 rolling_step 是否提高重現率、進而使動量加權有效。

設計與判準見 `dev/capital_alloc/PREREG_ROLLSTEP.md`（**跑前定稿**）。

不需重跑引擎的理由（該檔 §二）
------------------------------
`run_trading.py` 由 `formation_pairs` 取期，而形成期只看 `Period_Start`
往前 252 日、不依賴步長。故 rolling_step = 21k 的期起點是現有 21 日格點的
**子集**，形成結果已存在。資金面 `capital_per_pair = equity/(top_n × conc)`、
`conc = 126/(21k) = 6/k`，故名目額恰乘 k —— 引擎損益對名目額嚴格線性，
「保留每 k 期 + 損益乘 k」是精確重組，不是估計。

近似仍只有複利回饋（同 stage1）。**故本檔屬階段一，不支持績效主張。**

相位：k>=2 有 k 個起始相位，全部跑取平均 —— 挑相位是研究者自由度。
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

KS = [1, 2, 3, 6]           # rolling_step = 21k；conc = 6/k
BASE_CONC = 6


def selection_recurrence(pp: pd.DataFrame, keep_pi: np.ndarray) -> float:
    """配對在**上一個保留期**也被選中的比例。lag 中性，純衡量選取穩定性。"""
    order = {p: i for i, p in enumerate(sorted(keep_pi))}
    sub = pp[pp.pi.isin(keep_pi)].copy()
    sub["slot"] = sub.pi.map(order)
    sets = sub.groupby("slot").pair.apply(set)
    if len(sets) < 2:
        return np.nan
    hits = tot = 0
    for j in range(1, len(sets)):
        cur, prev = sets.iloc[j], sets.iloc[j - 1]
        hits += len(cur & prev)
        tot += len(cur)
    return hits / tot if tot else np.nan


def run_cell(path_key: str) -> list:
    df = s1.load_logs(path_key)
    if df.empty:
        return []
    conc0 = max(1, concurrent_of(path_key))
    periods = sorted(df.Period_Start.unique())
    pos = {p: i for i, p in enumerate(periods)}
    df = df.assign(pi=df.Period_Start.map(pos))

    out = []
    for k in KS:
        conc = max(1, BASE_CONC // k)
        for phase in range(k):
            keep_pi = np.array([i for i in range(len(periods)) if i % k == phase])
            sub = df[df.pi.isin(keep_pi)].copy()
            if sub.empty:
                continue
            # 名目額縮放：cpp_new / cpp_old = conc0 / conc
            scale_k = conc0 / conc
            sub["Daily_Delta"] = sub.Daily_Delta * scale_k

            base_pnl = (pd.Series(sub.Daily_Delta.to_numpy(), index=sub.Date)
                        .groupby(level=0).sum().sort_index())
            ew = metrics_from_pnl(base_pnl)

            # 兩種落後窗定義都跑（預先註冊補充）：
            #   甲 曆期固定 = max(2, 12//k) 期 ≈ 12 個月（忠於原文）
            #   乙 期數固定 = 12 期（觀測數不變，曆期最長 6 年）
            for tag, win in (("cal", max(2, 12 // k)), ("cnt", 12)):
                s1.WINDOW, s1.WARMUP = win, win
                # _trailing 以 sub 自己的期序建 pos2，pi 即保留期的連續編號
                _, pos2, pp, M, V = s1._trailing(sub, conc)
                gid = s1.row_gid(sub, pp)
                sc = s1.build_scales(sub, pos2, pp, M, V, "MO", conc)
                mo = metrics_from_pnl(s1.apply_scales(sub, gid, sc))
                m = np.array([M.get((q, pi), np.nan) for q, pi in zip(pp.pair, pp.pi)])
                out.append({
                    "win": tag, "window": win,
                    "k": k, "rolling_step": 21 * k, "conc": conc, "phase": phase,
                    "n_periods": pp.pi.nunique(),
                    "has_history": float((~np.isnan(m)).mean()),
                    "sel_recur": selection_recurrence(pp, pp.pi.unique()),
                    "Sharpe_EW": ew.Sharpe_Raw, "Ann_EW": ew.Ann_Ret_Raw,
                    "Sharpe_MO": mo.Sharpe_Raw,
                    "dS_MO": mo.Sharpe_Raw - ew.Sharpe_Raw,
                    "notional_ratio": s1.notional_ratio(sub, gid, sc),
                })
    return out


def main() -> int:
    cl = s1.cells()
    rows = []
    for _, r in cl.iterrows():
        for rec in run_cell(r._path):
            rec.update(METHOD=r.METHOD, arm=r.METHOD.replace("Grid ", ""),
                       top_n=r.top_n)
            rows.append(rec)
        print(f"  {r.METHOD:<20} Top{r.top_n:<3} 完成", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv("results/analysis/capital_alloc_rollstep.csv",
             index=False, encoding="utf-8-sig")

    # 相位平均 → 每 (格, k) 一列
    g = (d.groupby(["win", "arm", "top_n", "k", "rolling_step", "conc"], as_index=False)
           .agg(n_periods=("n_periods", "mean"),
                has_history=("has_history", "mean"),
                sel_recur=("sel_recur", "mean"),
                Sharpe_EW=("Sharpe_EW", "mean"),
                dS_MO=("dS_MO", "mean"),
                notional=("notional_ratio", "mean")))

    cal = g[g.win == "cal"]
    print("\n==== P-R1：兩個重現率（甲 cal 口徑）====")
    print(cal.groupby("k")[["n_periods", "has_history", "sel_recur"]].mean()
             .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n乙 cnt 口徑的 has_history：")
    print(g[g.win == "cnt"].groupby("k").has_history.mean()
           .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n==== P-R3：基準本身隨 k 的變化（與窗定義無關）====")
    print(cal.groupby("k").Sharpe_EW.describe()[["mean", "50%", "min", "max"]]
             .to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n==== P-R2：ΔSharpe_MO ====")
    passed = {}
    for tag, name in (("cal", "甲 曆期固定"), ("cnt", "乙 期數固定")):
        gt = g[g.win == tag]
        print(f"\n  [{name}]")
        for k in KS:
            sk = gt[gt.k == k]
            print(f"    k={k} step={21*k} conc={max(1,6//k)} win={sk.window.iloc[0]:.0f}期"
                  f"  中位 {sk.dS_MO.median():+.4f}"
                  f"  為正 {int((sk.dS_MO>0).sum())}/{len(sk)}")
        best = max(KS[1:], key=lambda k: (int((gt[gt.k == k].dS_MO > 0).sum()),
                                          gt[gt.k == k].dS_MO.median()))
        sb = gt[gt.k == best].dS_MO
        passed[tag] = bool((sb > 0).sum() >= 9 and sb.median() >= 0.05)
        print(f"    最佳 k={best}：{int((sb>0).sum())}/13 為正、中位 {sb.median():+.4f}"
              f"  -> {'過閘' if passed[tag] else '未過閘'}")

    print("\n跑前閘門：某 k>1 使 >=9/13 格為正 且 中位 >= +0.05，"
          "且甲乙兩種窗定義同時成立")
    print(f"=> **{'過閘' if all(passed.values()) else '未過閘'}**"
          f"（甲 {'過' if passed['cal'] else '不過'}／乙 {'過' if passed['cnt'] else '不過'}）")
    print(f"\n名目額比值範圍 [{g.notional.min():.4f}, {g.notional.max():.4f}]")

    g.to_csv("results/analysis/capital_alloc_rollstep_avg.csv",
             index=False, encoding="utf-8-sig")
    print("-> results/analysis/capital_alloc_rollstep{,_avg}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
