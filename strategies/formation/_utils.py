import numpy as np
from statsmodels.tsa.stattools import adfuller


def _johansen_test(y: np.ndarray, x: np.ndarray) -> tuple[bool, float]:
    """Johansen trace test for cointegration. Returns (is_cointegrated, trace_stat)."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    mat = np.column_stack([y, x])
    try:
        result = coint_johansen(mat, det_order=0, k_ar_diff=1)
        trace_stat = float(result.lr1[0])
        crit_95 = float(result.cvt[0, 1])
        return trace_stat > crit_95, trace_stat
    except Exception:
        return False, 0.0

def _compute_hurst(series: np.ndarray, already_stationary: bool = False) -> float:
    """
    R/S 分析近似 Hurst 指數。
    
    參數:
        series (np.ndarray): 價格或殘差時間序列。
        already_stationary (bool): 若為 True，代表輸入序列已是定態 (如 OLS 殘差或 SSD spread)，
                                 不進行 np.diff 一次差分 (符合 版本B 設計)。
    """
    n = len(series)
    if n < 20:
        return 0.5
    diffs = series if already_stationary else np.diff(series)
    rs_list = []
    for seg_len in [len(diffs) // 4, len(diffs) // 2, len(diffs)]:
        if seg_len < 4:
            continue
        seg = diffs[:seg_len]
        mean_seg = np.mean(seg)
        deviate  = np.cumsum(seg - mean_seg)
        std_val  = np.std(seg, ddof=1)
        if std_val < 1e-8:
            std_val = 1e-8
        rs = (np.max(deviate) - np.min(deviate)) / std_val
        rs_list.append((np.log(seg_len), np.log(rs + 1e-8)))
    if len(rs_list) < 2:
        return 0.5
    xs, ys = zip(*rs_list)
    try:
        h = float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        h = 0.5
    return np.clip(h, 0.0, 1.0)

def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """簡易 OLS：y = alpha + beta * x + resid，回傳 (alpha, beta, residuals)"""
    n = len(y)
    x_mat = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, y - np.mean(y)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    return alpha, beta, y - alpha - beta * x

def _adf_stat(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """ADF 檢定（no constant），同時回傳 (統計量, p 值)"""
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0
