"""掃描 entry_z x exit_z 的聯合網格（exit_z 在 config 中從未被掃過）。

動機：base_params 的 exit_z 固定為 0.0，即「等價差完全回到均值才出場」。
主要虧損機制正是「沒能回到 0」——期末強平 4,744 筆、獲利比例 20.4%、
總損益 −17,322。放寬出場門檻到 |z| = x 會把捕獲從 (entry_z) 個 sigma 降到
(entry_z - x) 個，但成功率上升；只要成功率的提升超過捕獲的折損就划算。

損益代理已對引擎校準（單次進出 3,024 筆：Pearson +0.715、Spearman +0.881、
同號率 89.6%，且十分位均值單調；截距恰等於 0.58% 往返成本）。

代理的已知侷限：只模擬單次進出，不含資金配置與重疊期的槽位競爭。放寬出場
會讓部位更早釋出、增加再進場機會，故本代理**低估**寬出場的好處。相對比較
可信，絕對水準不可直接當回測結果。
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
from dev.ml_formation.pool import (FORMATION_WINDOW, load_groups, load_prices,
                                   roll_indices, window_pairs)
from dev.ml_formation.pipeline_select import normalized, spreads_for

ENTRY_GRID = [1.5, 2.0, 2.5, 3.0]
EXIT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
ROUNDTRIP_COST = 0.0058


def _first_true(mask):
    any_ = mask.any(axis=1)
    return np.where(any_, mask.argmax(axis=1), -1)


def scan_period(form_prices, trade_prices, group_map):
    pool = window_pairs(form_prices, group_map)
    if pool.empty:
        return pd.DataFrame()
    usable = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
    norm = normalized(form_prices, usable)
    S = spreads_for(pool, norm)
    passed, _stat, _ = adf_pass_batch(S, alpha=0.05, nvars=2)
    mu_s, sd_s = S.mean(axis=1), S.std(axis=1, ddof=1)

    lp_f = np.log(form_prices[usable].where(form_prices[usable] > 0))
    mu_lp, sd_lp = lp_f.mean(), lp_f.std()
    lp_t = np.log(trade_prices[usable].where(trade_prices[usable] > 0))
    norm_t = (lp_t - mu_lp) / (sd_lp + 1e-12)
    A = norm_t[pool.Ticker_A.tolist()].values.T
    B = norm_t[pool.Ticker_B.tolist()].values.T
    St = A - pool.Hedge_Ratio.values[:, None] * B
    Z = np.nan_to_num((St - mu_s[:, None]) /
                      np.where(sd_s[:, None] > 1e-12, sd_s[:, None], np.nan), nan=0.0)

    n, T = Z.shape
    cols = np.arange(T)[None, :]
    beta_abs = pool.Hedge_Ratio.abs().values
    unit = sd_s * (sd_lp[pool.Ticker_A.tolist()].values +
                   beta_abs * sd_lp[pool.Ticker_B.tolist()].values) / (1.0 + beta_abs)

    out = {"Group": pool.Group.values, "SSD": pool.SSD.values, "adf_pass": passed,
           "Ticker_A": pool.Ticker_A.values, "Ticker_B": pool.Ticker_B.values}
    rows = np.arange(n)
    for e in ENTRY_GRID:
        entry = _first_true(np.abs(Z) >= e)
        has = entry >= 0
        ei = np.clip(entry, 0, T - 1)
        z_in = Z[rows, ei]
        sgn = np.sign(z_in)
        after = cols > entry[:, None]
        for x in EXIT_GRID:
            # 做空價差（z>0）→ z 跌到 +x 出場；做多 → z 升到 -x 出場
            hit = np.where(sgn[:, None] > 0, Z <= x, Z >= -x) & after
            conv = hit.any(axis=1) & has
            ci = np.clip(_first_true(hit), 0, T - 1)
            z_out = np.where(conv, sgn * x, Z[:, -1])
            cap = np.where(has, sgn * (z_in - z_out) * unit, np.nan)
            tag = f"e{e}_x{x}"
            out[f"cap_{tag}"] = cap - ROUNDTRIP_COST
            out[f"conv_{tag}"] = np.where(has, conv, np.nan)
            out[f"days_{tag}"] = np.where(conv, ci - entry, T - 1 - entry)
    return pd.DataFrame(out)


def main():
    pivot, dates, total, first_idx = load_prices()
    idxs = roll_indices(total, first_idx)
    groups = load_groups()
    parts, t0 = [], time.time()
    for k, i in enumerate(idxs, 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        if gm is None:
            continue
        df = scan_period(pivot.iloc[i - FORMATION_WINDOW:i],
                         pivot.iloc[i:min(i + FORWARD_DAYS, total)], gm)
        if df.empty:
            continue
        df.insert(0, "Period_Start", ps)
        parts.append(df)
        if k % 50 == 0 or k == len(idxs):
            print(f"  {k}/{len(idxs)}  {time.time()-t0:.0f}s", flush=True)
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(os.path.join(CACHE, "exit_scan.parquet"))
    print(f"完成 {len(full):,} 列 x {full.shape[1]} 欄")


if __name__ == "__main__":
    main()
