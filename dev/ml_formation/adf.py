"""批次 ADF（Engle-Granger 殘差版），與 _utils._adf_stat 數值等價。

_adf_stat 對每組配對呼叫一次 statsmodels.adfuller，全候選池有 151 萬對，
逐一呼叫太慢。此處把同一個迴歸批次化：

    regression="n", maxlag=1 的設定即
        dy[1:] ~ y[1:T-1] + dy[:-1]     （無截距、無時間趨勢）
    取 y[1:T-1] 之係數的 t 統計量。

p 值一樣走 mackinnonp(stat, "c", N=2)。但篩選只需要「p < 0.05」這個布林，
而 mackinnonp 對 stat 單調遞增，故改為一次性以二分法求出臨界值再比大小——
與逐一換算 p 值完全等價，省下 151 萬次呼叫。
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.adfvalues import mackinnonp


def adf_stat_batch(Y: np.ndarray) -> np.ndarray:
    """Y: (N, T) 每列一條殘差序列。回傳 (N,) 的 ADF t 統計量。"""
    Y = np.asarray(Y, dtype=np.float64)
    dy = np.diff(Y, axis=1)                     # (N, T-1)
    lvl = Y[:, 1:-1]                            # (N, T-2)  y_{t-1}
    dlag = dy[:, :-1]                           # (N, T-2)  dy_{t-1}
    resp = dy[:, 1:]                            # (N, T-2)  dy_t
    X = np.stack([lvl, dlag], axis=2)           # (N, T-2, 2)

    XtX = np.einsum("nmi,nmj->nij", X, X)
    Xty = np.einsum("nmi,nm->ni", X, resp)
    # 奇異者（常數序列等）標記為 NaN，交由呼叫端當作不通過
    det = XtX[:, 0, 0] * XtX[:, 1, 1] - XtX[:, 0, 1] * XtX[:, 1, 0]
    ok = np.abs(det) > 1e-12
    beta = np.full((len(Y), 2), np.nan)
    if ok.any():
        beta[ok] = np.linalg.solve(XtX[ok], Xty[ok])
    resid = resp - np.einsum("nmi,ni->nm", X, np.nan_to_num(beta))
    m = X.shape[1]
    s2 = (resid ** 2).sum(axis=1) / (m - 2)
    # inv(XtX)[0,0] = XtX[1,1] / det
    var0 = np.where(ok, s2 * XtX[:, 1, 1] / np.where(ok, det, 1.0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta[:, 0] / np.sqrt(var0)
    return np.where(ok, t, np.nan)


def eg_critical_value(alpha: float = 0.05, nvars: int = 2) -> float:
    """求 mackinnonp(stat, "c", N=nvars) == alpha 的 stat；p 對 stat 單調遞增。"""
    lo, hi = -12.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mackinnonp(mid, regression="c", N=nvars) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def adf_pass_batch(Y: np.ndarray, alpha: float = 0.05, nvars: int = 2):
    """回傳 (通過布林陣列, 統計量陣列, 臨界值)。"""
    stat = adf_stat_batch(Y)
    crit = eg_critical_value(alpha, nvars)
    return (stat < crit) & np.isfinite(stat), stat, crit
