"""掃描 formation_window（config 中全程固定 252，從未變動）。

代理的適用性：先前 entry_z 的外推被回測否證，原因是進場門檻會改變持有期與
資金週轉，而代理只量每筆捕獲。formation_window 不同——交易期仍是 126 日、
進出場門檻不變，改變的只有「用多長的歷史挑配對」。這是純粹的選取參數，
落在代理已驗證的適用範圍內（單次進出 Pearson +0.715 / Spearman +0.881）。

分組沿用 formation_groups：GICS 產業別不依賴視窗長度，故同一個交易期起點
對應的分組標籤可直接重用。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dev.ml_formation.adf import adf_pass_batch
from dev.ml_formation.build import CACHE, FORWARD_DAYS
from dev.ml_formation.pool import load_groups, load_prices, roll_indices, window_pairs
from dev.ml_formation.pipeline_select import normalized, spreads_for

BASE_W = 252
GRID_W = [63, 126, 189, 252, 378, 504]
ENTRY_Z, EXIT_Z, COST = 2.0, 0.0, 0.0058


def one(form_prices, trade_prices, group_map):
    pool = window_pairs(form_prices, group_map)
    if pool.empty:
        return None
    usable = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
    norm = normalized(form_prices, usable)
    S = spreads_for(pool, norm)
    passed, _s, _ = adf_pass_batch(S, alpha=0.05, nvars=2)
    mu_s, sd_s = S.mean(axis=1), S.std(axis=1, ddof=1)

    lp_f = np.log(form_prices[usable].where(form_prices[usable] > 0))
    mu_lp, sd_lp = lp_f.mean(), lp_f.std()
    lp_t = np.log(trade_prices[usable].where(trade_prices[usable] > 0))
    norm_t = (lp_t - mu_lp) / (sd_lp + 1e-12)
    A = norm_t[pool.Ticker_A.tolist()].values.T
    B = norm_t[pool.Ticker_B.tolist()].values.T
    Z = np.nan_to_num((A - pool.Hedge_Ratio.values[:, None] * B - mu_s[:, None]) /
                      np.where(sd_s[:, None] > 1e-12, sd_s[:, None], np.nan), nan=0.0)
    n, T = Z.shape
    rows = np.arange(n)
    hit = np.abs(Z) >= ENTRY_Z
    entry = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    has = entry >= 0
    z_in = Z[rows, np.clip(entry, 0, T - 1)]
    sgn = np.sign(z_in)
    after = np.arange(T)[None, :] > entry[:, None]
    cross = np.where(sgn[:, None] > 0, Z <= EXIT_Z, Z >= -EXIT_Z) & after
    conv = cross.any(axis=1) & has
    z_out = np.where(conv, sgn * EXIT_Z, Z[:, -1])
    ba = pool.Hedge_Ratio.abs().values
    unit = sd_s * (sd_lp[pool.Ticker_A.tolist()].values +
                   ba * sd_lp[pool.Ticker_B.tolist()].values) / (1.0 + ba)
    cap = np.where(has, sgn * (z_in - z_out) * unit, np.nan) - COST
    return pd.DataFrame({"SSD": pool.SSD.values, "adf_pass": passed,
                         "cap": cap, "conv": np.where(has, conv, np.nan)})


def main():
    pivot, dates, total, first_idx = load_prices()
    groups = load_groups()
    # 視窗最長 504，起點需往後推使 idx-W >= 0；各 W 共用同一組交易期起點以求可比
    idxs = [i for i in roll_indices(total, first_idx) if i - max(GRID_W) >= 0]
    print(f"共同交易期起點 {len(idxs)} 個（受最長視窗 {max(GRID_W)} 限制）\n")

    res = {}
    for W in GRID_W:
        parts, t0 = [], time.time()
        for i in idxs:
            gm = groups.get(dates[i - BASE_W].strftime("%Y-%m-%d"))
            if gm is None:
                continue
            df = one(pivot.iloc[i - W:i], pivot.iloc[i:min(i + FORWARD_DAYS, total)], gm)
            if df is None:
                continue
            df["pi"] = i
            parts.append(df)
        full = pd.concat(parts, ignore_index=True)
        V = full[full.adf_pass]
        row = {"池": len(full), "ADF通過率": full.adf_pass.mean()}
        for K in (1, 3, 5, 20):
            caps, cvs = [], []
            for _p, g in V.groupby("pi"):
                t = g.nsmallest(K, "SSD")
                caps.append(np.nansum(t.cap)); cvs.append(np.nanmean(t["conv"]))
            a = np.array(caps)
            row[f"K{K}_每期"] = a.mean()
            row[f"K{K}_比值"] = a.mean() / a.std() if a.std() > 0 else 0.0
            row[f"K{K}_收斂"] = np.nanmean(cvs)
        res[W] = row
        print(f"  W={W:>3}  {time.time()-t0:.0f}s  ADF通過 {row['ADF通過率']:.3f}  "
              f"K1比值 {row['K1_比值']:+.3f}  K3比值 {row['K3_比值']:+.3f}  "
              f"K5比值 {row['K5_比值']:+.3f}", flush=True)

    R = pd.DataFrame(res).T
    R.index.name = "formation_window"
    R.to_csv(os.path.join(CACHE, "fw_scan.csv"))
    print("\n" + R.round(4).to_string())


if __name__ == "__main__":
    main()
