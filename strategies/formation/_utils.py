import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.adfvalues import mackinnonp


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

def _compute_hurst_rs(series: np.ndarray, already_stationary: bool = False,
                      n_scales: int = 12, min_scale: int = 16) -> float:
    """
    標準 R/S 分析估計 Hurst 指數（多尺度 + 不重疊區塊平均）。

    與 `_compute_hurst` 的差異，以及為何需要這一版
    ------------------------------------------------------------------
    `_compute_hurst` 只取 3 個尺度（n//4, n//2, n），且三者是**巢狀前綴**
    （都從序列開頭取），彼此高度相依；再用 3 個點 polyfit 求斜率、最後
    clip 到 [0, 1]。在已知答案的合成序列上實測（n=252）：

        白噪音（真實 H=0.5）          → 0.292
        隨機漫步的差分（真實 H=0.5）  → 0.671

    同一個真值給出 0.29 與 0.67，且真實資料中約 21% 的輸出恰好等於 1.0
    （clip 邊界，即估計飽和）。本版改為 Mandelbrot 的標準作法：

      · 多個尺度（預設 12 個，log 均勻分佈於 [min_scale, n/2]）
      · 每個尺度把序列切成 ⌊n/m⌋ 個**不重疊**區塊，各算 R/S 後取平均
      · 對 (log m, log R/S) 回歸取斜率

    回傳未 clip 的原始斜率（NaN 表示尺度不足），使飽和情形可被看見而非
    被邊界吸收。呼叫端若需要 [0,1] 請自行 clip。
    """
    x = np.asarray(series, dtype=float)
    x = x if already_stationary else np.diff(x)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_scale * 2:
        return np.nan

    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(n // 2), n_scales).astype(int))
    scales = scales[scales >= min_scale]
    xs, ys = [], []
    for m in scales:
        k = n // m
        if k < 1:
            continue
        # 向量化：把前 k*m 點 reshape 成 (k, m) 的不重疊區塊，逐列一次算完。
        # 逐區塊 Python 迴圈在全量回測上會多花數小時，此處省掉。
        blocks = x[:k * m].reshape(k, m)
        s = blocks.std(axis=1, ddof=1)
        dev = np.cumsum(blocks - blocks.mean(axis=1, keepdims=True), axis=1)
        rs = (dev.max(axis=1) - dev.min(axis=1)) / np.where(s < 1e-12, np.nan, s)
        rs = rs[np.isfinite(rs)]
        if rs.size:
            xs.append(np.log(m))
            ys.append(np.log(float(rs.mean())))
    if len(xs) < 3:
        return np.nan
    try:
        return float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        return np.nan


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

def _adf_stat(resid: np.ndarray, max_lags: int = 1,
              eg_nvars: int = 2) -> tuple[float, float]:
    """
    Engle-Granger 第二步的殘差 ADF 檢定，回傳 (統計量, p 值)。

    為何 p 值不能用 `adfuller` 直接回傳的那一個
    ------------------------------------------------------------------
    `adfuller` 的 p 值出自 **Dickey-Fuller 分布**——那是「對一條**觀測到的**
    序列檢定單根」的虛無分布。但這裡送進來的是**估計出來的**共整合迴歸殘差：
    OLS 已經先把殘差的變異最小化過，其 ADF 統計量的虛無分布不是 DF 分布，
    而是依共整合迴歸中變數個數而定的 Phillips-Ouliaris／MacKinnon 分布，
    位置明顯更負。用 DF 臨界值檢定 EG 殘差 → 系統性過度拒絕。

    本專案實測（`formation_ranked` 全部候選對，n=1.5M–20M）：
        Grid GICS-DTW   舊 p<0.05 通過 77.4%   → EG 臨界值下 9.5%
        Grid HDB-DTW    舊 p<0.05 通過 77.7%   → EG 臨界值下 10.0%
    名目 5% 的檢定不可能拒絕 77% 的虛無假設；9.5% 才是合理數字
    （略高於 5%，代表確實存在真共整合）。

    作法與 `statsmodels.tsa.stattools.coint()` 一致：統計量仍以
    `regression="n"` 對殘差計算（殘差依建構已去均值），但 p 值改由
    `mackinnonp(stat, regression="c", N=eg_nvars)` 換算——`"c"` 對應**共整合
    迴歸**含截距，N 為該迴歸的變數個數。

    參數:
        max_lags:  ADF 的落後階（固定，不做 autolag 選擇）
        eg_nvars:  共整合迴歸的變數個數。配對交易為 2（一應變數 + 一解釋變數）；
                   設 1 可還原成一般單根檢定的 DF 校準（僅供對非殘差序列使用）。

    校準對照（N=2）：stat=-3.337 → p=0.0498；stat=-3.899 → p=0.0100，
    與 MacKinnon (2010) 響應曲面的 5%／1% 臨界值相符。

    文獻：Engle & Granger (1987), Econometrica 55(2):251-276；
          Phillips & Ouliaris (1990), Econometrica 58(1):165-193；
          MacKinnon (2010), Queen's Economics Dept. WP 1227。
    """
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        stat = float(result[0])
    except Exception:
        return 0.0, 1.0
    try:
        pval = float(mackinnonp(stat, regression="c", N=eg_nvars))
    except Exception:
        # 換算失敗時退回不拒絕，避免以錯誤的 DF p 值放行
        return stat, 1.0
    return stat, pval


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
        # 需 ≥3 檔才對產業因子去均值：2 檔時 g_s=(e_i+e_j)/2，回歸後兩殘差完全共線，
        # 造成後續 PCA 的 SVD 退化不收斂。
        if len(idx) < 3:
            continue
        g_s = e1[:, idx].mean(axis=1)               # 該產業的殘差因子
        e2[:, idx] = _resid_on(g_s, e1[:, idx])
    return np.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)


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
