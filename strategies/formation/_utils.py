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


def _residualize_returns(R: np.ndarray, sector_labels=None) -> np.ndarray:
    """
    因子殘差化（研究框架次步 #1）：移除共同因子後保留特殊性報酬。
    分群/相關若建在原始報酬上，會被市場 β 齊漲齊跌主導，且相關結構隨 regime 變動；
    改建在殘差上 → 捕捉「特殊性共動」（才會均值回歸）、更耐 regime。

    步驟：
      1) 市場因子 f_mkt[t] = 橫斷面平均報酬；逐股對 [1, f_mkt] OLS → 取殘差（移除市場 β）。
      2) 若給 sector_labels：產業因子 g_s[t] = 同產業市場殘差的橫斷面平均；
         逐股對 [1, g_{sector}] OLS → 取殘差（移除產業共動）。

    參數:
        R (T×N): 日報酬矩陣。sector_labels: 長度 N 的產業標籤（None 則只移除市場）。
    回傳: 殘差報酬矩陣（T×N）。
    """
    R = np.asarray(R, dtype=np.float64)
    T, N = R.shape
    if T < 10 or N < 2:
        return R

    def _resid_on(factor_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:
        # 對每一欄 y 逐股回歸 [1, factor] 取殘差（factor 相同 → 一次解出所有 β）；
        # 回傳每欄均值為 0 的殘差（後續 PCA/相關正好需要去均值輸入）。
        f = factor_1d - factor_1d.mean()
        var_f = float(f @ f) + 1e-12
        yc = y_2d - y_2d.mean(axis=0)
        beta = (yc.T @ f) / var_f              # (N,)
        return yc - np.outer(f, beta)

    f_mkt = R.mean(axis=1)
    e1 = _resid_on(f_mkt, R)

    if sector_labels is None:
        return e1

    labels = np.asarray(sector_labels)
    e2 = e1.copy()
    for s in np.unique(labels):
        idx = np.where(labels == s)[0]
        if len(idx) < 2:
            continue
        g_s = e1[:, idx].mean(axis=1)               # 該產業的殘差因子
        e2[:, idx] = _resid_on(g_s, e1[:, idx])
    return e2


def _cost_viable(spread_std: float, roundtrip_cost: float = 0.0058,
                 entry_z: float = 2.0, margin: float = 1.0) -> bool:
    """
    成本可行性過濾（研究框架次步 #3）。
    一次往返（entry_z·σ 進場 → 回到 0 出場）的預期擷取 ≈ entry_z × spread_std（分數移動）；
    必須顯著大於往返成本（單邊 0.29% × 2 = 0.58%），否則訊號被費用吃光。
    要求：entry_z × spread_std ≥ margin × roundtrip_cost。
      spread_std：spread（log-price 殘差）標準差，近似分數移動幅度。
    """
    if not np.isfinite(spread_std) or spread_std <= 0:
        return False
    return (entry_z * spread_std) >= (margin * roundtrip_cost)


def _bh_fdr_threshold(pvalues, alpha: float = 0.05) -> float:
    """
    Benjamini–Hochberg FDR 臨界 p 值（研究框架次步 #2）。
    N=500 → ~125k 候選配對，固定 p<0.05 會產生數千個偽共整合（＝出樣本強制平倉）。
    回傳可通過的最大 p 門檻：最大的 p_(k) 使得 p_(k) ≤ (k/m)·alpha；無則回 0（全數拒絕）。
    """
    p = np.sort(np.asarray([x for x in pvalues if np.isfinite(x)], dtype=np.float64))
    m = len(p)
    if m == 0:
        return 0.0
    crit = (np.arange(1, m + 1) / m) * alpha
    passed = np.where(p <= crit)[0]
    return float(p[passed.max()]) if len(passed) else 0.0
