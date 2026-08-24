"""掃描 dtw_window（config 中固定 15，從未變動過）。

帶寬 w 是一條連續軸：w=0 時 Sakoe-Chiba 帶退化成對角線，DTW 恰等於 SSD；
w>=T 時為無限制 DTW。故本掃描等於在「SSD」與「完整 DTW」之間插值，
而 config 只取了其中一點。

可行性來自 dev/ml_formation/dtw_fast.py 的 numba 實作（948x，逐位相同）——
原純 Python 版本下本掃描需數十小時。

評價從一開始就分前後半：今天已證實全期指標會系統性挑出 2014 後已死的配置
（formation_window=504 是乾淨實例：前半 +1.037、後半 −0.395）。
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

WINDOWS = [0, 1, 3, 5, 10, 15, 25, 40, 60, 126]
ENTRY_Z, COST = 2.0, 0.0058
ARM = os.environ.get("DTW_ARM", "Grid NOGRP-DTW_MSR0")
CUT = pd.Timestamp("2014-01-01")


def main():
    t0 = time.time()
    pivot, dates, total, first_idx = load_prices()
    groups = load_groups(ARM)
    print(f"臂 {ARM}；分組期數 {len(groups)}", flush=True)

    rec = []
    for k, i in enumerate(roll_indices(total, first_idx), 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        if gm is None:
            continue
        fp = pivot.iloc[i - FORMATION_WINDOW:i]
        tp = pivot.iloc[i:min(i + FORWARD_DAYS, total)]
        pool = window_pairs(fp, gm)
        if pool.empty:
            continue
        us = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
        norm = normalized(fp, us)
        S = spreads_for(pool, norm)
        pa, _s, _ = adf_pass_batch(S)
        if pa.sum() < 20:
            continue
        sub = pool[pa].reset_index(drop=True)
        mu, sd = S[pa].mean(1), S[pa].std(1, ddof=1)

        # 交易期淨捕獲（與 exit_scan 相同的口徑）
        lpf = np.log(fp[us].where(fp[us] > 0)); ml, sl = lpf.mean(), lpf.std()
        nt = (np.log(tp[us].where(tp[us] > 0)) - ml) / (sl + 1e-12)
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
        cv = (np.where(sg[:, None] > 0, Z <= 0, Z >= 0) & aft)
        cd = np.where(cv.any(1), cv.argmax(1), T - 1)
        zo = np.where(cv.any(1), 0.0, Z[:, -1])
        ba = sub.Hedge_Ratio.abs().values
        unit = sd * (sl[sub.Ticker_A.tolist()].values + ba * sl[sub.Ticker_B.tolist()].values) / (1 + ba)
        cap = np.where(has, sg * (zi - zo) * unit, np.nan) - COST

        # 各帶寬的 DTW（僅 ADF 通過者，與管線順序一致）
        NA = np.ascontiguousarray(norm[sub.Ticker_A.tolist()].values.T)
        NB = np.ascontiguousarray(norm[sub.Ticker_B.tolist()].values.T)
        row = {"pi": k, "date": dates[i], "n_adf": n}
        for w in WINDOWS:
            d = dtw_sc_batch(NA, NB, w)
            order = np.argsort(d, kind="mergesort")
            for K in (1, 3, 5, 20):
                sel = order[:K]
                row[f"w{w}_K{K}"] = np.nansum(cap[sel])
        rec.append(row)
        if k % 40 == 0:
            print(f"  {k}/295  n_adf={n}  {time.time()-t0:.0f}s", flush=True)

    R = pd.DataFrame(rec)
    R.to_parquet(os.path.join(CACHE, f"dtw_scan_{ARM.split()[1].split('_')[0]}.parquet"))
    print(f"\n完成 {len(R)} 期，{time.time()-t0:.0f}s\n")

    for K in (1, 3, 5, 20):
        print(f"── top-{K} ──  每期淨捕獲（已扣成本），比值＝均值/標準差")
        print(f"{'w':>5}{'全期':>10}{'比值':>8}{'前半':>10}{'比值':>8}{'後半':>10}{'比值':>8}")
        for w in WINDOWS:
            c = R[f"w{w}_K{K}"].values
            a = R[R.date < CUT][f"w{w}_K{K}"].values
            b = R[R.date >= CUT][f"w{w}_K{K}"].values
            f = lambda x: (x.mean(), x.mean() / x.std() if x.std() > 0 else 0)
            m0, r0 = f(c); m1, r1 = f(a); m2, r2 = f(b)
            tag = " (=SSD)" if w == 0 else (" ←現行" if w == 15 else "")
            print(f"{w:>5}{m0:>10.5f}{r0:>8.3f}{m1:>10.5f}{r1:>8.3f}{m2:>10.5f}{r2:>8.3f}{tag}")
        print()


if __name__ == "__main__":
    main()
