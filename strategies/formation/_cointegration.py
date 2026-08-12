"""
形成期共用共整合篩選層（與分組、排序無關）
======================================================================
把「ADF 共整合 + OU 半衰期 + Hurst 指數」三道統計過濾抽成中性函式，
供任何分組方法（GICS / HDBSCAN / Agglomerative / K-means）與任何排序準則
（SSD / DTW / SSD-DTW-PCA）共用，並讓篩選本身成為**可開關的消融維度**：

    passed, stats = screen_pair(spread, ...)          # 單一配對三道過濾
    hl = compute_halflife(spread)                     # OU 半衰期（原三處重複實作）

消融設計（形成期三維度）：
    分組 × 排序 × 篩選(none / coint)
  - filter_mode="none" ：只做分組 + 排序（測「排序準則本身」的貢獻）
  - filter_mode="coint"：加上三道統計過濾（現行所有策略的行為）

方法論依據：
  - Engle & Granger (1987)：共整合殘差的 ADF 單根檢定
  - Krauss, Do & Huck (2016)：OU 半衰期與 Hurst 指數作為均值回歸品質指標
"""
import numpy as np

import os

from strategies.formation._utils import _adf_stat, _compute_hurst, _compute_hurst_rs

# 決策用哪一版 Hurst：legacy（預設，維持既有結果可重現）| rs（標準多尺度 R/S）。
# 無論選哪一版，兩個值都會計算並記錄於 stats，供事後對照無需重跑。
_HURST_METHOD = os.environ.get("HURST_METHOD", "legacy").strip().lower()


def compute_halflife(spread: np.ndarray) -> float:
    """
    OU 半衰期：對 Δs_t = c + λ·s_{t-1} + u_t 最小平方估計，HL = −ln2 / λ。

    回傳 np.inf 表示無均值回歸傾向（λ ≥ 0 或估計失敗）——呼叫端以
    「HL 落在 [min, max] 之外」統一淘汰，語意與原 inline 實作等價
    （原實作為 λ ≥ 0 直接 continue）。

    註：本函式為 ssd_rolling / ssd_basic / DTW_Cointegration_Paper 三處
      逐字重複的 inline lstsq 之等價抽取。
    """
    dy = np.diff(spread)
    y_lag = spread[:-1]
    n_dy = len(dy)
    if n_dy < 2:
        return np.inf
    x_mat = np.column_stack([np.ones(n_dy), y_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, dy, rcond=None)
        lambda_val = coeffs[1]
    except Exception:
        lambda_val = 0.0
    if lambda_val >= 0.0:
        return np.inf
    return -np.log(2) / lambda_val


def screen_pair(
    spread: np.ndarray,
    adf_pvalue_threshold: float = 0.05,
    halflife_min: float = 1.0,
    halflife_max: float = 42.0,
    hurst_threshold: float = 0.50,
    adf_max_lags: int = 1,
    precomputed_adf: tuple = None,
    enabled: bool = True,
    adf_only: bool = False,        # True → 只做 ADF（復現許鈞翔 2025 的篩選層）
) -> tuple[bool, dict]:
    """
    對單一配對的 spread 施加三道統計過濾。

        enabled=False → 直接放行（filter_mode="none" 的消融用），
                        仍回傳可得的統計量供記錄，但不作為淘汰依據。

    回傳 (passed, stats)；stats 含 adf_stat / adf_p / halflife / hurst
    （未計算者為 None）。三道順序與現行策略一致：ADF → 半衰期 → Hurst，
    任一未過即淘汰（短路，省下後續慢速計算）。
    """
    stats = {"adf_stat": None, "adf_p": None,
             "halflife": None, "hurst": None, "hurst_rs": None}

    if not enabled:
        return True, stats

    # 步驟 1：ADF 共整合（過濾隨機漫步）
    # precomputed_adf：呼叫端已算過（如 DTW 的雙向 OLS 取較小 p 值），避免重算
    stat, pval = precomputed_adf if precomputed_adf is not None else \
        _adf_stat(spread, max_lags=adf_max_lags)
    stats["adf_stat"], stats["adf_p"] = stat, pval
    if pval >= adf_pvalue_threshold:
        return False, stats

    # adf_only：復現許鈞翔 (2025) 的篩選層——只做 Engle-Granger 兩步的 ADF，
    # 不施加半衰期與 Hurst。兩者仍計算並記錄，供「多這兩道差多少」的對照。
    if adf_only:
        stats["halflife"] = compute_halflife(spread)
        stats["hurst"] = _compute_hurst(spread, already_stationary=True)
        stats["hurst_rs"] = _compute_hurst_rs(spread, already_stationary=True)
        return True, stats

    # 步驟 2：OU 半衰期
    hl = compute_halflife(spread)
    stats["halflife"] = hl
    if hl < halflife_min or hl > halflife_max:
        return False, stats

    # 步驟 3：Hurst 指數（均值回歸傾向）
    # 兩版都算、都記錄：舊版（3 尺度巢狀前綴）與標準 R/S（多尺度不重疊區塊）。
    # 合成序列實測（n=252，500 次）：真值 H=0.3 時舊版標準差 0.168、誤判率 22.6%，
    # 新版 0.060 / 7.6%。中位數兩版皆略微上偏，差別在**離散度**而非偏誤。
    # 決策用哪一版由 HURST_METHOD 控制（預設 legacy＝維持既有結果可重現）。
    stats["hurst"] = _compute_hurst(spread, already_stationary=True)
    stats["hurst_rs"] = _compute_hurst_rs(spread, already_stationary=True)

    h = stats["hurst_rs"] if _HURST_METHOD == "rs" else stats["hurst"]
    if h is None or not np.isfinite(h) or h >= hurst_threshold:
        return False, stats

    return True, stats
