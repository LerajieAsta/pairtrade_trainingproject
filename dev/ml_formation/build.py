"""單次掃過 295 期，同時產出候選池特徵、收斂標籤、與形成窗序列張量。

分成兩趟會把 spread 的建構做兩次（每期最多近萬對 x 252 天），故合併。
ADF 是本趟最貴的一步，只算一次。

輸出
----
cache/pool.parquet          每列一個候選對-期的純量特徵與標籤
cache/seq/{period}.npy      (n_pairs, 252) float32，形成窗標準化價差序列

標籤定義（決策 6，引擎忠實版）
----------------------------
交易期 = 形成窗後 126 個交易日。z_t = (s_t - mu_s) / sigma_s，其中
s_t = P'_A,t - beta * P'_B,t，而 P' 用**形成窗**的對數價均值與標準差標準化
（與交易引擎一致）。

    entry_day   首次 |z| >= 2.0 的位置；不存在則 label_valid = False
    converged   進場後 z 是否在期末前穿越 0（依進場方向）
    主標籤 not_converged = 1 - converged，僅在 label_valid 時有效

輔助欄一併落地，日後改標籤定義不必重跑百萬級運算。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dev.ml_formation.adf import adf_pass_batch
from dev.ml_formation.pool import (FORMATION_WINDOW, load_groups, load_prices,
                                   roll_indices, window_pairs)
from dev.ml_formation.pipeline_select import normalized, spreads_for

FORWARD_DAYS = 126
ENTRY_Z = 2.0
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SEQ_DIR = os.path.join(CACHE, "seq")


def _first_true(mask: np.ndarray) -> np.ndarray:
    """每列首個 True 的行索引；整列 False 回傳 -1。"""
    any_ = mask.any(axis=1)
    idx = mask.argmax(axis=1)
    return np.where(any_, idx, -1)


def build_period(form_prices: pd.DataFrame, trade_prices: pd.DataFrame,
                 group_map: dict) -> tuple[pd.DataFrame, np.ndarray]:
    pool = window_pairs(form_prices, group_map)
    if pool.empty:
        return pool, np.zeros((0, FORMATION_WINDOW), dtype=np.float32)

    usable = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
    norm = normalized(form_prices, usable)
    S = spreads_for(pool, norm)                       # (n, T) 形成窗 spread
    passed, stat, _ = adf_pass_batch(S, alpha=0.05, nvars=2)
    mu_s = S.mean(axis=1)
    sd_s = S.std(axis=1, ddof=1)
    pool = pool.assign(adf_stat=stat, adf_pass=passed,
                       Spread_Mean=mu_s, Spread_Std_exact=sd_s)

    # 形成窗標準化序列（供 1D-CNN）
    Z_form = ((S - mu_s[:, None]) / np.where(sd_s[:, None] > 1e-12, sd_s[:, None], np.nan))

    # ── 交易期 z 序列：P' 沿用形成窗的對數價均值/標準差 ──────────────
    lp_f = np.log(form_prices[usable].where(form_prices[usable] > 0))
    mu_lp, sd_lp = lp_f.mean(), lp_f.std()
    lp_t = np.log(trade_prices[usable].where(trade_prices[usable] > 0))
    norm_t = (lp_t - mu_lp) / (sd_lp + 1e-12)

    A = norm_t[pool.Ticker_A.tolist()].values.T       # (n, 126)
    B = norm_t[pool.Ticker_B.tolist()].values.T
    St = A - pool.Hedge_Ratio.values[:, None] * B
    Zt = (St - mu_s[:, None]) / np.where(sd_s[:, None] > 1e-12, sd_s[:, None], np.nan)

    valid_days = np.isfinite(Zt).sum(axis=1)
    Zf = np.nan_to_num(Zt, nan=0.0)                   # 缺值視為未觸發

    # 進場：首次 |z| >= 2.0
    hit = np.abs(Zf) >= ENTRY_Z
    entry = _first_true(hit)
    has_entry = entry >= 0
    n, T = Zt.shape
    cols = np.arange(T)[None, :]
    after = cols > entry[:, None]                     # 進場之後（不含當日）
    sgn = np.sign(Zf[np.arange(n), np.clip(entry, 0, T - 1)])
    # 進場方向為正（z>0，做空價差）→ 收斂是 z <= 0；反之 z >= 0
    crossed = np.where(sgn[:, None] > 0, Zf <= 0, Zf >= 0) & after & np.isfinite(Zt)
    conv = crossed.any(axis=1) & has_entry
    conv_day = _first_true(crossed)

    # 輔助欄
    s_t = np.sign(Zf)
    ncross_trade = (np.diff(np.where(s_t == 0, 1, s_t), axis=1) != 0).sum(axis=1)
    z_end = np.where(valid_days > 0, Zf[:, -1], np.nan)
    # 近似經濟捕獲：|z_進場| - |z_出場| 個 sigma_s，再換回分數幅度
    exit_idx = np.where(conv, conv_day, T - 1)
    z_exit = Zf[np.arange(n), np.clip(exit_idx, 0, T - 1)]
    z_in = Zf[np.arange(n), np.clip(entry, 0, T - 1)]
    beta_abs = pool.Hedge_Ratio.abs().values
    sa = sd_lp[pool.Ticker_A.tolist()].values
    sb = sd_lp[pool.Ticker_B.tolist()].values
    unit = sd_s * (sa + beta_abs * sb) / (1.0 + beta_abs)
    capture_frac = np.where(has_entry, (np.abs(z_in) - np.abs(z_exit)) * unit, np.nan)

    out = pool.assign(
        label_valid=has_entry & (valid_days >= FORWARD_DAYS * 0.9),
        not_converged=np.where(conv, 0, 1).astype(np.int8),
        entry_day=entry, conv_day=conv_day,
        days_to_conv=np.where(conv, conv_day - entry, -1),
        n_cross_trade=ncross_trade, z_end=z_end,
        z_entry=np.where(has_entry, z_in, np.nan),
        capture_frac=capture_frac, valid_days=valid_days,
    )
    return out, Z_form.astype(np.float32)


def main(limit: int | None = None):
    os.makedirs(SEQ_DIR, exist_ok=True)
    pivot, dates, total, first_idx = load_prices()
    idxs = roll_indices(total, first_idx)
    groups = load_groups()
    if limit:
        idxs = idxs[:limit]

    parts, t0 = [], time.time()
    for k, i in enumerate(idxs, 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        if gm is None:
            continue
        fw = pivot.iloc[i - FORMATION_WINDOW:i]
        tw = pivot.iloc[i:min(i + FORWARD_DAYS, total)]
        df, seq = build_period(fw, tw, gm)
        if df.empty:
            continue
        df.insert(0, "Period_Start", ps)
        df.insert(1, "Trade_Start", dates[i].strftime("%Y-%m-%d"))
        parts.append(df)
        np.save(os.path.join(SEQ_DIR, f"{ps}.npy"), seq)
        if k % 10 == 0 or k == len(idxs):
            el = time.time() - t0
            print(f"  {k}/{len(idxs)}  {ps}  累計 {sum(len(p) for p in parts):,} 列"
                  f"  {el:.0f}s  (每期 {el/k:.2f}s，預估總計 {el/k*len(idxs)/60:.1f} 分)",
                  flush=True)

    full = pd.concat(parts, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    full.to_parquet(os.path.join(CACHE, "pool.parquet"))
    print(f"\n完成：{len(full):,} 列 → cache/pool.parquet")
    return full


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
