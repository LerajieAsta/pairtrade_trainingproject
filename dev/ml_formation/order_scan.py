"""管線順序的架構檢定：ADF 先篩 vs DTW 先排。

現行順序「分組 → ADF 篩選 → DTW 排序」使 DTW 只看得到通過 ADF 的 2.2%
（NOGRP 每期約 1,800/84,255）。此順序是計算量逼出來的：對全池跑 DTW 原需
39 小時。dtw_fast 的 numba 實作（948x，逐位相同）使其降為 2.5 分鐘，
本檢定因而第一次可行。

四個變體（皆 w=15，即掃描確認的最優帶寬）：
    A  ADF → DTW → top-K            現行
    B  DTW(全池) → top-K            完全不篩
    C  DTW(全池) → top-100 → ADF → top-K
    D  DTW(全池) → top-500 → ADF → top-K

評價一律分前後半：全期指標今天已多次證實會挑出 2014 後已死的配置。
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.ml_formation.adf import adf_pass_batch
from dev.ml_formation.build import CACHE, FORWARD_DAYS
from dev.ml_formation.dtw_fast import dtw_sc_batch
from dev.ml_formation.pool import (FORMATION_WINDOW, load_groups, load_prices,
                                   roll_indices, window_pairs)
from dev.ml_formation.pipeline_select import normalized, spreads_for

W, ENTRY_Z, COST = 15, 2.0, 0.0058
ARM = "Grid NOGRP-DTW_MSR0"
CUT = pd.Timestamp("2014-01-01")
KS = (1, 3, 5, 20)


def capture(sub, norm_t, mu, sd, sl):
    A = norm_t[sub.Ticker_A.tolist()].values.T
    B = norm_t[sub.Ticker_B.tolist()].values.T
    Z = np.nan_to_num((A - sub.Hedge_Ratio.values[:, None] * B - mu[:, None]) /
                      np.where(sd[:, None] > 1e-12, sd[:, None], np.nan), nan=0.0)
    n, T = Z.shape
    r = np.arange(n)
    hit = np.abs(Z) >= ENTRY_Z
    en = np.where(hit.any(1), hit.argmax(1), -1); has = en >= 0
    zi = Z[r, np.clip(en, 0, T - 1)]; sg = np.sign(zi)
    aft = np.arange(T)[None, :] > en[:, None]
    cv = np.where(sg[:, None] > 0, Z <= 0, Z >= 0) & aft
    zo = np.where(cv.any(1), 0.0, Z[:, -1])
    ba = sub.Hedge_Ratio.abs().values
    unit = sd * (sl[sub.Ticker_A.tolist()].values + ba * sl[sub.Ticker_B.tolist()].values) / (1 + ba)
    return np.where(has, sg * (zi - zo) * unit, np.nan) - COST


def main():
    t0 = time.time()
    pivot, dates, total, first_idx = load_prices()
    groups = load_groups(ARM)
    rec = []
    for k, i in enumerate(roll_indices(total, first_idx), 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        if gm is None:
            continue
        fp = pivot.iloc[i - FORMATION_WINDOW:i]
        tp = pivot.iloc[i:min(i + FORWARD_DAYS, total)]
        pool = window_pairs(fp, gm)
        if len(pool) < 500:
            continue
        us = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
        norm = normalized(fp, us)
        S = spreads_for(pool, norm)
        pa, _s, _ = adf_pass_batch(S)
        mu, sd = S.mean(1), S.std(1, ddof=1)
        lpf = np.log(fp[us].where(fp[us] > 0)); ml, sl = lpf.mean(), lpf.std()
        nt = (np.log(tp[us].where(tp[us] > 0)) - ml) / (sl + 1e-12)
        cap = capture(pool, nt, mu, sd, sl)

        NA = np.ascontiguousarray(norm[pool.Ticker_A.tolist()].values.T)
        NB = np.ascontiguousarray(norm[pool.Ticker_B.tolist()].values.T)
        d_all = dtw_sc_batch(NA, NB, W)                    # 全池 DTW
        order_all = np.argsort(d_all, kind="mergesort")
        idx_adf = np.flatnonzero(pa)
        order_adf = idx_adf[np.argsort(d_all[idx_adf], kind="mergesort")] if len(idx_adf) else np.array([], int)

        row = {"pi": k, "date": dates[i], "n_pool": len(pool), "n_adf": int(pa.sum())}
        for K in KS:
            row[f"A_K{K}"] = np.nansum(cap[order_adf[:K]]) if len(order_adf) >= K else np.nan
            row[f"B_K{K}"] = np.nansum(cap[order_all[:K]])
            for M, tag in ((100, "C"), (500, "D")):
                cand = order_all[:M]
                keep = cand[pa[cand]]
                row[f"{tag}_K{K}"] = np.nansum(cap[keep[:K]]) if len(keep) >= K else np.nan
        rec.append(row)
        if k % 50 == 0:
            print(f"  {k}/295  池 {len(pool):,}  ADF {int(pa.sum()):,}  {time.time()-t0:.0f}s", flush=True)

    R = pd.DataFrame(rec)
    R.to_parquet(os.path.join(CACHE, "order_scan.parquet"))
    print(f"\n完成 {len(R)} 期  {time.time()-t0:.0f}s")
    print(f"平均池 {R.n_pool.mean():,.0f}，ADF 通過 {R.n_adf.mean():,.0f}"
          f"（{100*R.n_adf.mean()/R.n_pool.mean():.1f}%）\n")
    names = {"A": "ADF→DTW（現行）", "B": "DTW全池，不篩", "C": "DTW前100→ADF", "D": "DTW前500→ADF"}
    for K in KS:
        print(f"── top-{K} ──  比值＝每期淨捕獲均值/標準差")
        print(f"{'變體':<18}{'全期':>9}{'前半':>9}{'後半':>9}{'兩半較小':>10}{'有效期數':>9}")
        for v, nm in names.items():
            c = R[f"{v}_K{K}"].dropna()
            a = R[(R.date < CUT)][f"{v}_K{K}"].dropna()
            b = R[(R.date >= CUT)][f"{v}_K{K}"].dropna()
            f = lambda x: x.mean() / x.std() if len(x) > 5 and x.std() > 0 else np.nan
            r0, r1, r2 = f(c), f(a), f(b)
            print(f"{nm:<18}{r0:>9.3f}{r1:>9.3f}{r2:>9.3f}{min(r1,r2):>10.3f}{len(c):>9}")
        print()


if __name__ == "__main__":
    main()
