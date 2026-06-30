"""
HDBSCAN MultiScale 全市場/跨產業聚類回測系統（多時間尺度分段特徵版）
======================================================================

核心思路：
  將形成期切分為多個子期間，衡量「配對關係在不同市場週期下的跨期穩定性」，
  而非僅看單一靜態統計值。

  預設子期間設計（對應 SP500 歷史市場結構，2000–2025）：
    Sub-1：科技泡沫／崩潰期   (2000–2004)
    Sub-2：金融海嘯前牛市     (2005–2009)
    Sub-3：後金融海嘯復甦     (2010–2014)
    Sub-4：低波動牛市         (2015–2019)
    Sub-5：COVID + 升息週期   (2020–2025)

特徵層次：
  層次 A（跨期相關穩定性）：各子期間的 Pearson 相關係數 → 均值、std、最小值
  層次 B（跨期共整合穩定性）：各子期間的 ADF p-value → 通過率、最差期
  層次 C（牛熊條件相關）：牛市 vs 熊市條件相關差異
  層次 D（波動率結構跨期）：各子期間的波動率比率一致性
  層次 E（均值回歸跨期）：各子期間的 Hurst 指數 → 均值、最差期

與 HDBSCAN_UMAP 的核心差異：
  HDBSCAN_UMAP    → 「深度」：用 10 個多維特徵描述整個形成期的配對品質
  HDBSCAN_MultiScale → 「廣度」：用時間切片衡量配對關係是否跨越多個市場週期穩定

  ✅ 兩者均支援跨產業分群（HDBSCAN 不依賴 sector_mapping 分群）
"""

import warnings
import sys
import random
from typing import Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from strategies.formation._utils import _compute_hurst, _ols, _adf_stat

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_LIB = "hdbscan"
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        HDBSCAN_LIB = "sklearn"
    except ImportError:
        raise ImportError("請先安裝 scikit-learn >= 1.3.0 或 hdbscan")

from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# 工具函數
# ══════════════════════════════════════════════════════════════════════

def _make_relative_sub_periods(form_start: str, form_end: str, n_splits: int = 4) -> list[tuple]:
    """
    將形成期 [form_start, form_end] 等分為 n_splits 段，回傳 (start, end, label) 清單。
    取代固定日曆邊界，確保每個滾動窗口都有 n_splits 個子期間可計算。
    """
    s = pd.Timestamp(form_start)
    e = pd.Timestamp(form_end)
    total_days = (e - s).days
    if total_days <= 0 or n_splits <= 0:
        return [(form_start, form_end, "Full")]
    step = total_days / n_splits
    periods = []
    for i in range(n_splits):
        ps = s + pd.Timedelta(days=int(i * step))
        pe = (s + pd.Timedelta(days=int((i + 1) * step)) - pd.Timedelta(days=1)) if i < n_splits - 1 else e
        periods.append((ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d"), f"Q{i+1}"))
    return periods

def _sub_period_corr(
    log_a: np.ndarray,
    log_b: np.ndarray,
    idx:   pd.DatetimeIndex,
    sub_periods: list[tuple],
) -> dict:
    corr_by_period = []
    for (start, end, label) in sub_periods:
        mask = (idx >= start) & (idx <= end)
        if mask.sum() < 30:
            continue
        ca = log_a[mask]
        cb = log_b[mask]
        if np.std(ca) < 1e-10 or np.std(cb) < 1e-10:
            continue
        c = float(np.corrcoef(ca, cb)[0, 1])
        if np.isfinite(c):
            corr_by_period.append(c)

    if not corr_by_period:
        return {"corr_by_period": [], "corr_mean": 0.0,
                "corr_std": 1.0, "corr_min": 0.0, "n_valid": 0}

    return {
        "corr_by_period": corr_by_period,
        "corr_mean":      float(np.mean(corr_by_period)),
        "corr_std":       float(np.std(corr_by_period, ddof=1)) if len(corr_by_period) > 1 else 0.0,
        "corr_min":       float(np.min(corr_by_period)),
        "n_valid":        len(corr_by_period),
    }


def _sub_period_coint(
    log_a: np.ndarray,
    log_b: np.ndarray,
    idx:   pd.DatetimeIndex,
    sub_periods:       list[tuple],
    adf_max_lags:      int   = 1,
    adf_pvalue_thresh: float = 0.05,
) -> dict:
    pval_by_period = []
    for (start, end, label) in sub_periods:
        mask = (idx >= start) & (idx <= end)
        if mask.sum() < 30:
            continue
        ca = log_a[mask]
        cb = log_b[mask]
        try:
            _, _, re_ab = _ols(ca, cb)
            _, pval_ab  = _adf_stat(re_ab, adf_max_lags)
            _, _, re_ba = _ols(cb, ca)
            _, pval_ba  = _adf_stat(re_ba, adf_max_lags)
            pval_by_period.append(min(pval_ab, pval_ba))
        except Exception:
            pval_by_period.append(1.0)

    if not pval_by_period:
        return {"pval_by_period": [], "pass_rate": 0.0,
                "worst_pval": 1.0, "best_pval": 1.0, "n_valid": 0}

    pass_count = sum(1 for p in pval_by_period if p < adf_pvalue_thresh)
    return {
        "pval_by_period": pval_by_period,
        "pass_rate":      float(pass_count / len(pval_by_period)),
        "worst_pval":     float(np.max(pval_by_period)),
        "best_pval":      float(np.min(pval_by_period)),
        "n_valid":        len(pval_by_period),
    }


def _sub_period_hurst(
    log_a:        np.ndarray,
    log_b:        np.ndarray,
    log_a_arr:    np.ndarray,
    log_b_arr:    np.ndarray,
    idx:          pd.DatetimeIndex,
    sub_periods:  list[tuple],
    adf_max_lags: int = 1,
) -> dict:
    hurst_by_period = []
    for (start, end, label) in sub_periods:
        mask = (idx >= start) & (idx <= end)
        if mask.sum() < 60:
            continue
        ca = log_a[mask]
        cb = log_b[mask]
        try:
            _, _, re_ab = _ols(ca, cb)
            _, _, re_ba = _ols(cb, ca)
            h_ab = _compute_hurst(re_ab, already_stationary=True)
            h_ba = _compute_hurst(re_ba, already_stationary=True)
            h    = min(h_ab, h_ba)
            if np.isfinite(h):
                hurst_by_period.append(h)
        except Exception:
            pass

    if not hurst_by_period:
        return {"hurst_by_period": [], "hurst_mean": 0.5,
                "hurst_worst": 0.5, "hurst_std": 0.0}

    return {
        "hurst_by_period": hurst_by_period,
        "hurst_mean":      float(np.mean(hurst_by_period)),
        "hurst_worst":     float(np.max(hurst_by_period)),
        "hurst_std":       float(np.std(hurst_by_period, ddof=1)) if len(hurst_by_period) > 1 else 0.0,
    }


def _bull_bear_corr_diff(
    ret_a:     np.ndarray,
    ret_b:     np.ndarray,
    bear_mask: np.ndarray,   # boolean array, same length as ret_a/ret_b
) -> dict:
    """
    計算牛熊市條件相關差異。
    bear_mask 由動態 MA 方法在呼叫前計算，不依賴硬編碼的歷史日期。
    """
    bull_mask = ~bear_mask
    result    = {"corr_bull": 0.5, "corr_bear": 0.5, "regime_diff": 0.5}

    if bull_mask.sum() >= 30:
        c = np.corrcoef(ret_a[bull_mask], ret_b[bull_mask])[0, 1]
        if np.isfinite(c):
            result["corr_bull"] = float(c)

    if bear_mask.sum() >= 20:
        c = np.corrcoef(ret_a[bear_mask], ret_b[bear_mask])[0, 1]
        if np.isfinite(c):
            result["corr_bear"] = float(c)

    result["regime_diff"] = abs(result["corr_bull"] - result["corr_bear"])
    return result


def _sub_period_vol_ratio_stability(
    ret_a:       np.ndarray,
    ret_b:       np.ndarray,
    idx_ret:     pd.DatetimeIndex,
    sub_periods: list[tuple],
) -> dict:
    vol_ratio_by_period = []
    for (start, end, label) in sub_periods:
        mask = (idx_ret >= start) & (idx_ret <= end)
        if mask.sum() < 20:
            continue
        ra = ret_a[mask]
        rb = ret_b[mask]
        va = float(np.std(ra, ddof=1))
        vb = float(np.std(rb, ddof=1))
        if vb < 1e-12:
            continue
        ratio = va / vb
        if np.isfinite(ratio):
            vol_ratio_by_period.append(ratio)

    if not vol_ratio_by_period:
        return {"vol_ratio_by_period": [], "vol_ratio_std": 1.0, "vol_ratio_mean": 1.0}

    return {
        "vol_ratio_by_period": vol_ratio_by_period,
        "vol_ratio_std":       float(np.std(vol_ratio_by_period, ddof=1)) if len(vol_ratio_by_period) > 1 else 0.0,
        "vol_ratio_mean":      float(np.mean(vol_ratio_by_period)),
    }


def _compute_multiscale_pair_features(
    log_a:             np.ndarray,
    log_b:             np.ndarray,
    idx:               pd.DatetimeIndex,
    sub_periods:       list[tuple],
    bear_mask_ret:     np.ndarray,   # boolean array len = len(idx)-1
    adf_max_lags:      int   = 1,
    adf_pvalue_thresh: float = 0.05,
) -> dict:
    ret_a   = np.diff(log_a)
    ret_b   = np.diff(log_b)
    idx_ret = idx[1:]

    corr_info   = _sub_period_corr(log_a, log_b, idx, sub_periods)
    coint_info  = _sub_period_coint(log_a, log_b, idx, sub_periods, adf_max_lags, adf_pvalue_thresh)
    regime_info = _bull_bear_corr_diff(ret_a, ret_b, bear_mask_ret)
    vol_info    = _sub_period_vol_ratio_stability(ret_a, ret_b, idx_ret, sub_periods)
    hurst_info  = _sub_period_hurst(log_a, log_b, log_a, log_b, idx, sub_periods, adf_max_lags)

    return {
        "corr_mean":        corr_info["corr_mean"],
        "corr_std":         corr_info["corr_std"],
        "corr_min":         corr_info["corr_min"],
        "n_valid_periods":  corr_info["n_valid"],
        "coint_pass_rate":  coint_info["pass_rate"],
        "coint_worst_pval": coint_info["worst_pval"],
        "regime_diff":      regime_info["regime_diff"],
        "corr_bear":        regime_info["corr_bear"],
        "vol_ratio_std":    vol_info["vol_ratio_std"],
        "vol_ratio_mean":   vol_info["vol_ratio_mean"],
        "hurst_mean":       hurst_info["hurst_mean"],
        "hurst_worst":      hurst_info["hurst_worst"],
    }


def _multiscale_quality_score(mf: dict, n_splits: int = 4) -> float:
    s_corr_mean   = max(0.0, (mf["corr_mean"] - 0.5) / 0.5)
    s_corr_std    = max(0.0, 1.0 - mf["corr_std"] / 0.3)
    s_corr_min    = max(0.0, mf["corr_min"])
    s_pass_rate   = mf["coint_pass_rate"]
    s_worst_pval  = max(0.0, 1.0 - mf["coint_worst_pval"] / 0.1)
    s_regime_diff = max(0.0, 1.0 - mf["regime_diff"] / 0.5)
    s_bear_corr   = max(0.0, mf["corr_bear"])
    s_vol_std     = max(0.0, 1.0 - mf["vol_ratio_std"] / 0.5)
    s_vol_mean    = max(0.0, 1.0 - abs(mf["vol_ratio_mean"] - 1.0) / 2.0)
    s_hurst_mean  = max(0.0, (0.5 - mf["hurst_mean"])  / 0.5)
    s_hurst_worst = max(0.0, (0.5 - mf["hurst_worst"]) / 0.5)
    # 分母改為實際子期間數，確保任何滾動窗口下滿分都可達到
    s_coverage    = min(1.0, mf["n_valid_periods"] / max(1, n_splits))

    return float(
        0.10 * s_corr_mean   +
        0.18 * s_corr_std    +
        0.08 * s_corr_min    +
        0.15 * s_pass_rate   +
        0.05 * s_worst_pval  +
        0.12 * s_regime_diff +
        0.08 * s_bear_corr   +
        0.08 * s_vol_std     +
        0.03 * s_vol_mean    +
        0.06 * s_hurst_mean  +
        0.04 * s_hurst_worst +
        0.03 * s_coverage
    )


# ══════════════════════════════════════════════════════════════════════
# 單股特徵萃取（第一層：HDBSCAN 分群用）
# ══════════════════════════════════════════════════════════════════════

def _extract_features_multiscale(
    log_price:   np.ndarray,
    idx:         pd.DatetimeIndex,
    sub_periods: list[tuple],
    market_ret:  Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.diff(log_price)

    sub_vols = []
    for (start, end, label) in sub_periods:
        mask     = (idx >= start) & (idx <= end)
        ret_mask = mask[1:]
        if ret_mask.sum() >= 20:
            v = float(np.std(ret[ret_mask], ddof=1))
        else:
            v = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0
        sub_vols.append(v)

    vol_all        = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0
    hurst_all      = _compute_hurst(log_price)
    vol_period_std = float(np.std(sub_vols)) if len(sub_vols) > 1 else 0.0

    beta = 1.0
    if market_ret is not None and len(market_ret) == len(ret):
        mask_fin = np.isfinite(ret) & np.isfinite(market_ret)
        if mask_fin.sum() >= 30:
            try:
                cov   = np.cov(ret[mask_fin], market_ret[mask_fin])
                var_m = cov[1, 1]
                if var_m > 1e-12:
                    beta = float(cov[0, 1] / var_m)
            except Exception:
                pass

    features = np.array(
        sub_vols + [vol_all, vol_period_std, hurst_all, beta],
        dtype=np.float64
    )
    return np.where(np.isfinite(features), features, 0.0)


# ══════════════════════════════════════════════════════════════════════
# Formation Class
# ══════════════════════════════════════════════════════════════════════

class Formation:
    """
    HDBSCAN MultiScale 配對交易形成期模組（多時間尺度分段特徵）

    跨產業分群說明：
      HDBSCAN 的分群完全基於量化特徵，不使用 sector_mapping 作為分群依據。
      sector_mapping 僅用於結果記錄與後置多元化篩選（max_sector_ratio）。
    """

    def __init__(
        self,
        price_df:    pd.DataFrame,
        form_start:  str,
        form_end:    str,
        top_n:                   int   = 20,
        sector_mapping:          dict  = None,
        min_tickers_for_pairing: int   = 2,
        n_splits:                int   = 4,    # 形成期內部等分數（取代固定日曆子期間）
        bear_ma_window:          int   = 60,   # 動態熊市判斷：等權指數低於 N 日 MA 即為熊市
        min_corr_mean:           float = 0.50,
        min_corr_min:            float = 0.10,
        max_corr_std:            float = 0.30,
        min_coint_pass_rate:     float = 0.40,
        max_regime_diff:         float = 0.50,
        max_vol_ratio_std:       float = 0.80,
        min_hurst_improvement:   float = 0.0,
        adf_max_lags:            int   = 1,
        adf_pvalue_threshold:    float = 0.05,
        adf_sub_pvalue:          float = 0.10,
        halflife_min:            float = 1.0,
        halflife_max:            float = 60.0,
        hurst_threshold:         float = 0.5,
        min_zero_crossings:      int   = 5,
        hdbscan_min_cluster_size: int  = 5,
        hdbscan_min_samples:      int  = 2,
        hdbscan_metric:           str  = "euclidean",
        reduce_method:           str   = "umap",
        umap_n_components:       int   = 5,
        umap_n_neighbors:        int   = 40,
        umap_min_dist:           float = 0.01,
        umap_random_state:       int   = 42,
        pca_n_components:        int   = 15,
        max_sector_ratio:        float = 0.3,
        use_mom1_filter:         bool  = True,
        market_returns: Optional[pd.Series] = None,
    ):
        self.price_df   = price_df.copy()
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping          = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.max_sector_ratio        = max_sector_ratio
        self.n_splits                = n_splits
        self.bear_ma_window          = bear_ma_window

        # 從形成期內部等分產生子期間，確保每個滾動窗口都有 n_splits 段可計算
        self.sub_periods = _make_relative_sub_periods(form_start, form_end, n_splits)

        self.min_corr_mean         = min_corr_mean
        self.min_corr_min         = min_corr_min
        self.max_corr_std         = max_corr_std
        self.min_coint_pass_rate  = min_coint_pass_rate
        self.max_regime_diff      = max_regime_diff
        self.max_vol_ratio_std    = max_vol_ratio_std
        self.adf_max_lags         = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.adf_sub_pvalue       = adf_sub_pvalue
        self.halflife_min         = halflife_min
        self.halflife_max         = halflife_max
        self.hurst_threshold      = hurst_threshold
        self.min_zero_crossings   = min_zero_crossings

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        self.reduce_method     = reduce_method.lower()
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors  = umap_n_neighbors
        self.umap_min_dist     = umap_min_dist
        self.umap_random_state = umap_random_state
        self.pca_n_components  = pca_n_components

        if self.reduce_method == "umap" and not UMAP_AVAILABLE:
            raise RuntimeError("umap-learn 未安裝：pip install umap-learn")

        self.use_mom1_filter = use_mom1_filter
        self.market_returns  = market_returns

        self.selected_pairs:  pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict         = {}

    def _build_dynamic_bear_mask(self) -> np.ndarray:
        """
        動態計算熊市 mask（長度 = len(price_df) - 1，對應日報酬序列）。
        以等權平均 log-price 指數對比其 bear_ma_window 日 MA：
          - 指數低於 MA → 熊市（True）
          - 指數高於 MA → 牛市（False）
        完全基於形成期內部資料，不使用任何硬編碼歷史日期。
        """
        log_prices   = np.log(self.price_df)
        eq_idx       = log_prices.mean(axis=1)          # 等權 log-price 指數
        prices_exp   = np.exp(eq_idx.values)
        ma_window    = max(5, self.bear_ma_window)
        min_periods  = max(1, ma_window // 2)
        ma           = (pd.Series(prices_exp)
                        .rolling(ma_window, min_periods=min_periods)
                        .mean().values)
        bear_price   = prices_exp < ma                  # 長度 = len(price_df)
        return bear_price[1:]                           # 截去第一天（對齊日報酬長度）

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        log_prices = np.log(self.price_df)
        tickers    = log_prices.columns.tolist()
        idx        = pd.DatetimeIndex(self.price_df.index)

        mkt_arr = None
        if self.market_returns is not None:
            aligned = self.market_returns.reindex(self.price_df.index)
            mkt_arr = np.diff(np.log(aligned.ffill().values + 1e-12))

        feat_rows, valid_tickers = [], []
        for ticker in tickers:
            series = log_prices[ticker].values
            if len(series) < 60 or not np.all(np.isfinite(series)):
                continue
            ticker_ret = np.diff(series)
            mkt_sub    = mkt_arr if (mkt_arr is not None and len(mkt_arr) == len(ticker_ret)) else None
            feat_rows.append(_extract_features_multiscale(series, idx, self.sub_periods, mkt_sub))
            valid_tickers.append(ticker)

        if not feat_rows:
            return np.empty((0, 0)), []

        return RobustScaler().fit_transform(np.vstack(feat_rows)), valid_tickers

    def _umap_reduce(self, X: np.ndarray) -> np.ndarray:
        n       = X.shape[0]
        n_comp  = min(self.umap_n_components, n - 1)
        n_neigh = min(self.umap_n_neighbors,  n - 1)
        if n_comp < 1 or n_neigh < 1:
            return X
        for init in ["spectral", "random"]:
            try:
                return umap.UMAP(
                    n_components = n_comp,
                    n_neighbors  = n_neigh,
                    min_dist     = self.umap_min_dist,
                    random_state = self.umap_random_state,
                    low_memory   = True,
                    init         = init,
                ).fit_transform(X)
            except Exception:
                continue
        return X

    def _pca_reduce(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        n_comp = min(self.pca_n_components, X.shape[0] - 1, X.shape[1])
        if n_comp < 1:
            return X
        return PCA(n_components=n_comp, random_state=self.umap_random_state).fit_transform(X)

    def _pca_umap_reduce(self, X: np.ndarray) -> np.ndarray:
        """先 PCA 降維去除噪音，再 UMAP 非線性嵌入，降低退化解機率。"""
        from sklearn.decomposition import PCA
        pca_comp = min(self.pca_n_components, X.shape[0] - 1, X.shape[1])
        if pca_comp >= 1:
            X = PCA(n_components=pca_comp, random_state=self.umap_random_state).fit_transform(X)
        return self._umap_reduce(X)

    def _hdbscan_cluster(self, X: np.ndarray) -> np.ndarray:
        min_cs   = self.hdbscan_min_cluster_size
        min_samp = self.hdbscan_min_samples

        while min_cs >= 2:
            cs   = min(min_cs, max(2, X.shape[0] // 5))
            samp = max(1, min(min_samp, cs - 1))

            if HDBSCAN_LIB == "hdbscan":
                clf = hdbscan.HDBSCAN(
                    min_cluster_size = cs,
                    min_samples      = samp,
                    metric           = self.hdbscan_metric,
                    core_dist_n_jobs = -1,
                )
            else:
                clf = sklearn_HDBSCAN(
                    min_cluster_size = cs,
                    min_samples      = samp,
                    metric           = self.hdbscan_metric,
                    n_jobs           = -1,
                )

            labels = clf.fit_predict(X)
            if len(set(labels) - {-1}) >= 1:
                return labels

            min_cs   //= 2
            min_samp   = max(1, min_samp // 2)

        print("  [FormationV4] HDBSCAN 未找到任何群落。")
        return np.full(X.shape[0], -1)

    def _cointegration_within_clusters(
        self,
        tickers:      list[str],
        labels:       np.ndarray,
        bear_mask_ret: np.ndarray,
    ) -> pd.DataFrame:
        log_prices    = np.log(self.price_df[tickers])
        idx           = pd.DatetimeIndex(self.price_df.index)
        unique_labels = set(labels) - {-1}

        if not unique_labels:
            print("  [FormationV4] 無有效群落。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(
            f"  [FormationV4] 聚類結果：{len(unique_labels)} 群落 | "
            f"{noise_count} 噪音 | "
            f"子期間數：{len(self.sub_periods)} | "
            f"共整合門檻 p≤{self.adf_pvalue_threshold}"
        )

        group_map: dict[int, list[str]] = {}
        for t, lbl in zip(tickers, labels):
            if lbl != -1:
                group_map.setdefault(int(lbl), []).append(t)

        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}
        if not valid_groups:
            return pd.DataFrame()

        print(f"  [FormationV4] 有效群落：{len(valid_groups)} 組")

        eg_records, passed_count, rejected_count = [], 0, 0

        for cluster_lbl, group_tickers in sorted(valid_groups.items()):
            if len(group_tickers) > 100:
                random.seed(self.umap_random_state)
                group_tickers = random.sample(group_tickers, 100)

            for i, ta in enumerate(group_tickers):
                log_a = log_prices[ta].values
                sec_a = self.sector_mapping.get(ta.upper(), "Unknown")

                for j in range(i + 1, len(group_tickers)):
                    tb    = group_tickers[j]
                    log_b = log_prices[tb].values
                    sec_b = self.sector_mapping.get(tb.upper(), "Unknown")

                    # 步驟 1：全期相關係數初篩
                    corr_full = float(np.corrcoef(log_a, log_b)[0, 1])
                    if not np.isfinite(corr_full) or corr_full < self.min_corr_mean:
                        rejected_count += 1
                        continue

                    # 步驟 2：全期 ADF 共整合
                    al_ab, be_ab, re_ab = _ols(log_a, log_b)
                    stat_ab, pval_ab    = _adf_stat(re_ab, self.adf_max_lags)
                    al_ba, be_ba, re_ba = _ols(log_b, log_a)
                    stat_ba, pval_ba    = _adf_stat(re_ba, self.adf_max_lags)

                    if min(pval_ab, pval_ba) >= self.adf_pvalue_threshold:
                        rejected_count += 1
                        continue

                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta_coef, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ta, tb
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta_coef, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = tb, ta

                    # 步驟 3：全期半衰期
                    dy    = np.diff(best_resid)
                    y_lag = best_resid[:-1]
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(
                            np.column_stack([np.ones(len(dy)), y_lag]),
                            dy, rcond=None
                        )
                        lam = coeffs[1]
                    except Exception:
                        lam = 0.0

                    if lam >= 0.0:
                        rejected_count += 1
                        continue
                    halflife = float(-np.log(2) / lam)
                    if not (self.halflife_min <= halflife <= self.halflife_max):
                        rejected_count += 1
                        continue

                    # 步驟 4：全期 Hurst
                    hurst = _compute_hurst(best_resid, already_stationary=True)
                    if hurst >= self.hurst_threshold:
                        rejected_count += 1
                        continue


                    # 步驟 5：零穿越次數
                    demeaned = best_resid - np.mean(best_resid)
                    zc = int(np.sum(np.diff(np.sign(demeaned)) != 0))
                    if zc < self.min_zero_crossings:
                        rejected_count += 1
                        continue

                    # 步驟 6–10：方案四核心多時間尺度特徵
                    mf = _compute_multiscale_pair_features(
                        log_a             = log_a,
                        log_b             = log_b,
                        idx               = idx,
                        sub_periods       = self.sub_periods,
                        bear_mask_ret     = bear_mask_ret,
                        adf_max_lags      = self.adf_max_lags,
                        adf_pvalue_thresh = self.adf_sub_pvalue,
                    )

                    if mf["corr_mean"]       < self.min_corr_mean:       rejected_count += 1; continue
                    if mf["corr_min"]        < self.min_corr_min:        rejected_count += 1; continue
                    if mf["corr_std"]        > self.max_corr_std:        rejected_count += 1; continue
                    if mf["coint_pass_rate"] < self.min_coint_pass_rate: rejected_count += 1; continue
                    if mf["regime_diff"]     > self.max_regime_diff:     rejected_count += 1; continue
                    if mf["vol_ratio_std"]   > self.max_vol_ratio_std:   rejected_count += 1; continue

                    quality_score   = _multiscale_quality_score(mf, n_splits=self.n_splits)
                    passed_count   += 1
                    assigned_sector = sec_a if sec_a == sec_b else f"Cluster_{cluster_lbl}"

                    eg_records.append({
                        "Form_Start":       self.form_start,
                        "Form_End":         self.form_end,
                        "Sector":           assigned_sector,
                        "Sector_A":         sec_a,
                        "Sector_B":         sec_b,
                        "Cluster_Label":    cluster_lbl,
                        "Ticker_A":         best_a,
                        "Ticker_B":         best_b,
                        "ADF_Stat":         round(best_stat,      6),
                        "ADF_PValue":       round(best_pval,      6),
                        "Hedge_Ratio":      round(best_beta_coef, 6),
                        "OLS_Alpha":        round(best_alpha,     6),
                        "Spread_Mean":      round(float(np.mean(best_resid)),        6),
                        "Spread_Std":       round(float(np.std(best_resid, ddof=1)), 6),
                        "Halflife":         round(halflife,  2),
                        "Hurst_Spread":     round(hurst,     4),
                        "Zero_Crossings":   int(zc),
                        "Correlation":      round(corr_full, 6),
                        "Corr_Mean":        round(mf["corr_mean"],        4),
                        "Corr_Std":         round(mf["corr_std"],         4),
                        "Corr_Min":         round(mf["corr_min"],         4),
                        "N_Valid_Periods":  int(mf["n_valid_periods"]),
                        "Coint_Pass_Rate":  round(mf["coint_pass_rate"],  4),
                        "Coint_Worst_Pval": round(mf["coint_worst_pval"], 6),
                        "Regime_Diff":      round(mf["regime_diff"],      4),
                        "Corr_Bear":        round(mf["corr_bear"],        4),
                        "Vol_Ratio_Std":    round(mf["vol_ratio_std"],    4),
                        "Vol_Ratio_Mean":   round(mf["vol_ratio_mean"],   4),
                        "Hurst_Mean":       round(mf["hurst_mean"],       4),
                        "Hurst_Worst":      round(mf["hurst_worst"],      4),
                        "Quality_Score":    round(quality_score,          4),
                        "Mom1_Diff":        0.0,
                    })

        print(f"  [FormationV4] 篩選結果：通過 {passed_count} 對 | 排除 {rejected_count} 對")

        if not eg_records:
            return pd.DataFrame()

        # Mom1 篩選（Han et al. 2021）
        if self.use_mom1_filter and len(eg_records) > 1:
            mom1_map: dict[str, float] = {}
            for ticker in tickers:
                series = np.log(self.price_df[ticker].values)
                if len(series) >= 21:
                    mom1_map[ticker] = float(series[-1] - series[-21])
                elif len(series) >= 2:
                    mom1_map[ticker] = float(series[-1] - series[0])
                else:
                    mom1_map[ticker] = 0.0

            for rec in eg_records:
                rec["Mom1_Diff"] = round(
                    abs(mom1_map.get(rec["Ticker_A"], 0.0) -
                        mom1_map.get(rec["Ticker_B"], 0.0)), 6
                )

            all_diffs      = [r["Mom1_Diff"] for r in eg_records]
            mom1_threshold = float(np.std(all_diffs, ddof=1)) if len(all_diffs) > 1 else 0.0
            before         = len(eg_records)
            eg_records     = [r for r in eg_records if r["Mom1_Diff"] < mom1_threshold]

            print(
                f"  [FormationV4] Mom1 篩選：門檻={mom1_threshold:.4f} | "
                f"{before} → {len(eg_records)} 對"
            )

            if not eg_records:
                return pd.DataFrame()

        return (
            pd.DataFrame(eg_records)
            .sort_values("Quality_Score", ascending=False)
            .reset_index(drop=True)
        )

    def run(self) -> pd.DataFrame:
        X, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if self.reduce_method == "pca":
            X_embed = self._pca_reduce(X)
        elif self.reduce_method == "pca_umap":
            X_embed = self._pca_umap_reduce(X)
        else:
            X_embed = self._umap_reduce(X)
        labels  = self._hdbscan_cluster(X_embed)
        self.cluster_labels_ = {t: int(lbl) for t, lbl in zip(valid_tickers, labels)}

        bear_mask_ret = self._build_dynamic_bear_mask()
        eg_df = self._cointegration_within_clusters(valid_tickers, labels, bear_mask_ret)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if self.max_sector_ratio > 0:
            # 以真實產業（Sector_A / Sector_B）計算曝險，而非群集標籤，
            # 避免跨產業配對因 Sector 欄位被改填為 Cluster_N 而規避產業集中度上限
            max_per_sec = max(1, int(self.top_n * self.max_sector_ratio))
            sec_exposure: dict[str, int] = {}
            diversified = []
            for _, row in eg_df.iterrows():
                sa, sb = row["Sector_A"], row["Sector_B"]
                exp_a = sec_exposure.get(sa, 0)
                exp_b = sec_exposure.get(sb, 0)
                if exp_a < max_per_sec and exp_b < max_per_sec:
                    diversified.append(row)
                    sec_exposure[sa] = exp_a + 1
                    if sb != sa:
                        sec_exposure[sb] = exp_b + 1
                if len(diversified) >= self.top_n:
                    break
            selected = pd.DataFrame(diversified).copy()
        else:
            selected = eg_df.head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        log_prices = np.log(self.price_df)
        selected["Log_Mean_A"] = selected["Ticker_A"].map(log_prices.mean())
        selected["Log_Std_A"]  = selected["Ticker_A"].map(log_prices.std())
        selected["Log_Mean_B"] = selected["Ticker_B"].map(log_prices.mean())
        selected["Log_Std_B"]  = selected["Ticker_B"].map(log_prices.std())

        self.selected_pairs = selected
        return self.selected_pairs


