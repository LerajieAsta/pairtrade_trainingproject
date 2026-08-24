"""複製管線的選取邏輯：依 SSD 遞增 → ADF 篩選 → 湊滿 top_n 即止。

ssd_rolling.select_pairs 的迴圈逐一檢定並在收滿 top_n 時 break，故池子後段
從未被檢定。先批次算完全部 ADF 再取「SSD 順序中前 top_n 個通過者」與該迴圈
輸出完全相同——未被碰到的候選本來就不影響結果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dev.ml_formation.adf import adf_pass_batch


def normalized(form_prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    lp = np.log(form_prices[tickers].where(form_prices[tickers] > 0))
    return (lp - lp.mean()) / (lp.std() + 1e-12)


def spreads_for(pool: pd.DataFrame, norm: pd.DataFrame) -> np.ndarray:
    """(n_pairs, T) 的 spread = P'_A - beta * P'_B（ssd_rolling.py:139）。"""
    A = norm[pool.Ticker_A.tolist()].values.T
    B = norm[pool.Ticker_B.tolist()].values.T
    return A - pool.Hedge_Ratio.values[:, None] * B


def select_topn(pool: pd.DataFrame, norm: pd.DataFrame, top_n: int = 20,
                alpha: float = 0.05) -> pd.DataFrame:
    """回傳複製管線後的 top_n，附上 adf_stat 與全池的通過旗標。"""
    if pool.empty:
        return pool
    p = pool.sort_values("SSD", kind="mergesort").reset_index(drop=True)
    S = spreads_for(p, norm)
    passed, stat, _crit = adf_pass_batch(S, alpha=alpha, nvars=2)
    p["adf_stat"] = stat
    p["adf_pass"] = passed
    p["Spread_Mean"] = S.mean(axis=1)
    p["Spread_Std_exact"] = S.std(axis=1, ddof=1)
    sel = p[p.adf_pass].head(top_n).copy()
    sel["Pair_Rank"] = range(1, len(sel) + 1)
    return sel
