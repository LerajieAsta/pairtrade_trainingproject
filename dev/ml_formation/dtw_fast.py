"""Sakoe-Chiba DTW 的 numba 實作，與 DTW_Cointegration_Paper._sakoe_chiba_dtw 數值等價。

原實作是純 Python 巢狀迴圈：n=252、w=15 時每對約 7,800 次迭代，實測 7.1 ms/對。
NOGRP 每期 84,255 組候選，全期不篩選需 49 小時，經 ADF 篩選（約 7% 通過）仍需
3.4 小時——這使得 dtw_window 的敏感性掃描實務上做不了（config 中它固定為 15，
從未變動）。

本模組只換實作、不換演算法：同樣的 DP 遞迴、同樣的 Sakoe-Chiba 帶寬、同樣的
平方距離與邊界條件，故結果須逐位相同（驗證見 verify()）。
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False)
def dtw_sc(x, y, window):
    n = x.shape[0]
    m = y.shape[0]
    INF = np.inf
    # 只保留兩列 DP，記憶體由 O(n*m) 降為 O(m)
    prev = np.full(m + 1, INF)
    cur = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        for k in range(m + 1):
            cur[k] = INF
        start_j = max(1, i - window)
        end_j = min(m, i + window)
        for j in range(start_j, end_j + 1):
            d = x[i - 1] - y[j - 1]
            cost = d * d
            a = prev[j]          # insertion
            b = cur[j - 1]       # deletion
            c = prev[j - 1]      # match
            best = a
            if b < best:
                best = b
            if c < best:
                best = c
            cur[j] = cost + best
        for k in range(m + 1):
            prev[k] = cur[k]
    return prev[m]


@njit(cache=True, parallel=True, fastmath=False)
def dtw_sc_batch(A, B, window):
    """A, B: (n_pairs, T)。回傳 (n_pairs,) 的 DTW 距離。"""
    n = A.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = dtw_sc(A[i], B[i], window)
    return out


def verify(n_cases: int = 300, seed: int = 0) -> dict:
    """與原實作逐位比對。任何差異都應為 0。"""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from strategies.formation.DTW_Cointegration_Paper import _sakoe_chiba_dtw
    rng = np.random.default_rng(seed)
    worst = 0.0
    exact = 0
    for _ in range(n_cases):
        T = int(rng.integers(30, 300))
        w = int(rng.integers(1, 40))
        x = rng.normal(size=T); y = rng.normal(size=T)
        a = _sakoe_chiba_dtw(x, y, w)
        b = float(dtw_sc(x, y, w))
        if a == b:
            exact += 1
        worst = max(worst, abs(a - b))
    return {"n": n_cases, "逐位相同": exact, "最大絕對差": worst}
