"""掃描 trading_window（config 中固定 126，從未變動過）。

陷阱：此參數會改變資金結構，而代理在 entry_z 上正是因為忽略資金週轉而失準。
    CONCURRENT_PERIODS = trading_window / rolling_step
    capital_per_pair   = equity / (top_n * CONCURRENT_PERIODS)
交易期加倍 → 每組配對的資金減半。故除了每期捕獲總額，另報「每單位資本捕獲」
（乘上稀釋因子 21/W），後者才對應真實報酬率。

所有視窗長度共用同一組交易期起點（受最長視窗限制），否則比較無效。
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.ml_formation.adf import adf_pass_batch
from dev.ml_formation.build import CACHE
from dev.ml_formation.dtw_fast import dtw_sc_batch
from dev.ml_formation.pool import (FORMATION_WINDOW, load_groups, load_prices,
                                   roll_indices, window_pairs)
from dev.ml_formation.pipeline_select import normalized, spreads_for

TWS = [42, 63, 126, 189, 252, 378]
STEP, W_DTW, ENTRY_Z, COST = 21, 15, 2.0, 0.0058
ARM = os.environ.get("TW_ARM", "Grid NOGRP-DTW_MSR0")
RANK = os.environ.get("TW_RANK", "dtw")          # dtw | ssd
CUT = pd.Timestamp("2014-01-01")


def cap_for(sub, nt, mu, sd, sl):
    A = nt[sub.Ticker_A.tolist()].values.T
    B = nt[sub.Ticker_B.tolist()].values.T
    Z = np.nan_to_num((A - sub.Hedge_Ratio.values[:, None] * B - mu[:, None]) /
                      np.where(sd[:, None] > 1e-12, sd[:, None], np.nan), nan=0.0)
    n, T = Z.shape
    r = np.arange(n)
    hit = np.abs(Z) >= ENTRY_Z
    en = np.where(hit.any(1), hit.argmax(1), -1); has = en >= 0
    zi = Z[r, np.clip(en, 0, T - 1)]; sg = np.sign(zi)
    aft = np.arange(T)[None, :] > en[:, None]
    cv = np.where(sg[:, None] > 0, Z <= 0, Z >= 0) & aft
    conv = cv.any(1) & has
    zo = np.where(cv.any(1), 0.0, Z[:, -1])
    ba = sub.Hedge_Ratio.abs().values
    unit = sd * (sl[sub.Ticker_A.tolist()].values + ba * sl[sub.Ticker_B.tolist()].values) / (1 + ba)
    cap = np.where(has, sg * (zi - zo) * unit, np.nan) - COST
    return cap, conv, has


def main():
    t0 = time.time()
    pivot, dates, total, first_idx = load_prices()
    groups = load_groups(ARM)
    idxs = [i for i in roll_indices(total, first_idx) if i + max(TWS) <= total]
    print(f"臂 {ARM}｜排序 {RANK}｜共同交易期起點 {len(idxs)}（受最長視窗 {max(TWS)} 限制）", flush=True)

    rec = []
    for k, i in enumerate(idxs, 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        if gm is None:
            continue
        fp = pivot.iloc[i - FORMATION_WINDOW:i]
        pool = window_pairs(fp, gm)
        if len(pool) < 200:
            continue
        us = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
        norm = normalized(fp, us)
        S = spreads_for(pool, norm)
        pa, _s, _ = adf_pass_batch(S)
        if pa.sum() < 20:
            continue
        sub = pool[pa].reset_index(drop=True)
        mu, sd = S[pa].mean(1), S[pa].std(1, ddof=1)
        lpf = np.log(fp[us].where(fp[us] > 0)); ml, sl = lpf.mean(), lpf.std()

        if RANK == "dtw":
            NA = np.ascontiguousarray(norm[sub.Ticker_A.tolist()].values.T)
            NB = np.ascontiguousarray(norm[sub.Ticker_B.tolist()].values.T)
            score = dtw_sc_batch(NA, NB, W_DTW)
        else:
            score = sub.SSD.values
        order = np.argsort(score, kind="mergesort")

        row = {"pi": k, "date": dates[i]}
        for W in TWS:
            tp = pivot.iloc[i:i + W]
            nt = (np.log(tp[us].where(tp[us] > 0)) - ml) / (sl + 1e-12)
            cap, conv, has = cap_for(sub, nt, mu, sd, sl)
            for K in (1, 3, 5):
                sel = order[:K]
                row[f"W{W}_K{K}"] = np.nansum(cap[sel])
                row[f"W{W}_K{K}_cv"] = np.nanmean(conv[sel])
        rec.append(row)
        if k % 50 == 0:
            print(f"  {k}/{len(idxs)}  {time.time()-t0:.0f}s", flush=True)

    R = pd.DataFrame(rec)
    R.to_parquet(os.path.join(CACHE, f"tw_scan_{RANK}.parquet"))
    print(f"\n完成 {len(R)} 期  {time.time()-t0:.0f}s\n")

    for K in (1, 3, 5):
        print(f"── top-{K} ──")
        print(f"{'W':>5}{'重疊期':>7}{'收斂率':>8}"
              f"{'每期捕獲':>10}{'比值':>7}"
              f"{'每單位資本':>11}{'比值':>7}{'前半':>8}{'後半':>8}")
        for W in TWS:
            c = R[f"W{W}_K{K}"]
            dil = STEP / W                       # 資金稀釋因子
            a = R[R.date < CUT][f"W{W}_K{K}"]
            b = R[R.date >= CUT][f"W{W}_K{K}"]
            f = lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
            tag = " ←現行" if W == 126 else ""
            print(f"{W:>5}{W//STEP:>7}{R[f'W{W}_K{K}_cv'].mean():>8.3f}"
                  f"{c.mean():>10.5f}{f(c):>7.3f}"
                  f"{c.mean()*dil:>11.5f}{f(c):>7.3f}{f(a):>8.3f}{f(b):>8.3f}{tag}")
        print()


if __name__ == "__main__":
    main()
