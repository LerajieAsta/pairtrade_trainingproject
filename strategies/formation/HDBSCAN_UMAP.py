"""
HDBSCAN UMAP 全市場/跨產業聚類回測系統（雙層特徵評分版）
======================================================================

架構：
  第一層（單股特徵，10 維）→ UMAP 降維 → HDBSCAN 聚類
    特徵：波動率結構、自相關、Hurst 指數、Beta、價格水準
    目的：把行為相似的股票分到同一群落，縮小候選配對池

  第二層（配對層特徵，10 維）→ 品質評分排序
    層次一（均值回歸核心） ：Hurst_spread、halflife、zero_crossings
    層次二（相關穩定性）   ：corr_mean、corr_stability、corr_regime_diff
    層次三（風險暴露對齊） ：beta_diff、vol_ratio
    層次四（流動性可行性） ：volume_corr、adv_ratio

與 HDBSCAN_MultiScale 的核心差異：
  HDBSCAN_UMAP       → 「深度」：10 維特徵描述整個形成期的配對品質
  HDBSCAN_MultiScale → 「廣度」：時間切片衡量配對跨市場週期的穩定性
"""

import warnings
import sys
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

def _compute_beta(ret: np.ndarray, market_ret: np.ndarray) -> float:
    try:
        mask = np.isfinite(ret) & np.isfinite(market_ret)
        r, m = ret[mask], market_ret[mask]
        if len(r) < 30:
            return 1.0
        cov   = np.cov(r, m)
        var_m = cov[1, 1]
        if var_m < 1e-12:
            return 1.0
        return float(cov[0, 1] / var_m)
    except Exception:
        return 1.0


def _rolling_corr_stability(
    ret_a:  np.ndarray,
    ret_b:  np.ndarray,
    window: int = 60,
) -> tuple[float, float]:
    if len(ret_a) < window * 2:
        c = np.corrcoef(ret_a, ret_b)[0, 1]
        return float(c), 0.5

    roll_corrs = []
    for start in range(0, len(ret_a) - window + 1, window // 2):
        c = np.corrcoef(ret_a[start:start + window], ret_b[start:start + window])[0, 1]
        if np.isfinite(c):
            roll_corrs.append(c)

    if not roll_corrs:
        return 0.0, 1.0
    return float(np.mean(roll_corrs)), float(np.std(roll_corrs, ddof=1))


def _compute_adv_ratio(vol_a: np.ndarray, vol_b: np.ndarray) -> float:
    adv_a = float(np.mean(vol_a[vol_a > 0])) if np.any(vol_a > 0) else 1.0
    adv_b = float(np.mean(vol_b[vol_b > 0])) if np.any(vol_b > 0) else 1.0
    if max(adv_a, adv_b) < 1e-8:
        return 1.0
    return float(min(adv_a, adv_b) / max(adv_a, adv_b))


def _compute_halflife(residuals: np.ndarray) -> float:
    dy    = np.diff(residuals)
    y_lag = residuals[:-1]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            np.column_stack([np.ones(len(dy)), y_lag]),
            dy, rcond=None
        )
        lam = coeffs[1]
    except Exception:
        return np.inf
    if lam >= 0.0:
        return np.inf
    return float(-np.log(2) / lam)


# ══════════════════════════════════════════════════════════════════════
# [A] 單股特徵萃取（10 維）
# ══════════════════════════════════════════════════════════════════════

def _extract_features(
    log_price:  np.ndarray,
    market_ret: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.diff(log_price)

    vol_21  = float(np.std(ret[-21:], ddof=1)) if len(ret) >= 21 else 0.0
    vol_63  = float(np.std(ret[-63:], ddof=1)) if len(ret) >= 63 else 0.0
    vol_all = float(np.std(ret,       ddof=1)) if len(ret) >  1  else 0.0
    vol_ratio_short_long = vol_21 / (vol_63 + 1e-12)

    autocorr_1 = float(np.corrcoef(ret[:-1], ret[1:])[0, 1]) if len(ret) > 1 else 0.0
    skew_all   = float(pd.Series(ret).skew()) if len(ret) > 2 else 0.0
    kurt_all   = float(pd.Series(ret).kurt()) if len(ret) > 3 else 0.0
    hurst      = _compute_hurst(log_price)
    beta       = _compute_beta(ret, market_ret) if (market_ret is not None and len(market_ret) == len(ret)) else 1.0
    log_price_level = float(np.mean(log_price))

    features = np.array([
        vol_21, vol_63, vol_all, vol_ratio_short_long,
        autocorr_1, skew_all, kurt_all,
        hurst, beta, log_price_level,
    ], dtype=np.float64)
    return np.where(np.isfinite(features), features, 0.0)


# ══════════════════════════════════════════════════════════════════════
# [B] 配對層特徵（10 維）
# ══════════════════════════════════════════════════════════════════════

def _compute_pair_features(
    log_a:       np.ndarray,
    log_b:       np.ndarray,
    ret_a:       np.ndarray,
    ret_b:       np.ndarray,
    vol_a:       np.ndarray,
    vol_b:       np.ndarray,
    residuals:   np.ndarray,
    beta_a:      float = 1.0,
    beta_b:      float = 1.0,
    roll_window: int   = 60,
) -> dict:
    hurst_spread   = _compute_hurst(residuals, already_stationary=True)
    halflife       = _compute_halflife(residuals)
    demeaned       = residuals - np.mean(residuals)
    zero_crossings = int(np.sum(np.diff(np.sign(demeaned)) != 0))

    corr_mean, corr_std = _rolling_corr_stability(ret_a, ret_b, window=roll_window)

    mkt_proxy = (ret_a + ret_b) / 2
    bull_mask  = mkt_proxy > np.median(mkt_proxy)
    bear_mask  = ~bull_mask
    if bull_mask.sum() >= 20 and bear_mask.sum() >= 20:
        corr_bull = np.corrcoef(ret_a[bull_mask], ret_b[bull_mask])[0, 1]
        corr_bear = np.corrcoef(ret_a[bear_mask], ret_b[bear_mask])[0, 1]
        corr_regime_diff = float(abs(corr_bull - corr_bear))
    else:
        corr_regime_diff = 0.5

    beta_diff = float(abs(beta_a - beta_b))
    v_a = float(np.std(ret_a, ddof=1)) if len(ret_a) > 1 else 1.0
    v_b = float(np.std(ret_b, ddof=1)) if len(ret_b) > 1 else 1.0
    vol_ratio = max(v_a, v_b) / (min(v_a, v_b) + 1e-12)

    log_vol_a   = np.log1p(vol_a.astype(float))
    log_vol_b   = np.log1p(vol_b.astype(float))
    volume_corr = float(np.corrcoef(log_vol_a, log_vol_b)[0, 1]) if len(log_vol_a) > 10 else 0.0
    adv_ratio   = _compute_adv_ratio(vol_a, vol_b)

    return {
        "hurst_spread":      hurst_spread,
        "halflife":          halflife if np.isfinite(halflife) else 999.0,
        "zero_crossings":    zero_crossings,
        "corr_mean":         corr_mean,
        "corr_stability":    corr_std,
        "corr_regime_diff":  corr_regime_diff,
        "beta_diff":         beta_diff,
        "vol_ratio":         vol_ratio,
        "volume_corr":       volume_corr,
        "adv_ratio":         adv_ratio,
    }


def _pair_quality_score(pf: dict) -> float:
    s_hurst = max(0.0, (0.5 - pf["hurst_spread"]) / 0.5)

    hl = pf["halflife"]
    if   5 <= hl <= 30:  s_halflife = 1.0
    elif hl < 5:         s_halflife = hl / 5.0
    elif hl <= 60:       s_halflife = (60 - hl) / 30.0
    else:                s_halflife = 0.0

    s_zc        = min(1.0, pf["zero_crossings"] / 200.0)
    s_corr_mean = max(0.0, (pf["corr_mean"] - 0.5) / 0.5)
    s_corr_stab = max(0.0, 1.0 - pf["corr_stability"]    / 0.3)
    s_regime    = max(0.0, 1.0 - pf["corr_regime_diff"]  / 0.4)
    s_beta      = max(0.0, 1.0 - pf["beta_diff"]         / 1.0)
    s_vol       = max(0.0, 1.0 - (pf["vol_ratio"] - 1.0) / 2.0)
    s_vol_corr  = max(0.0, pf["volume_corr"])
    s_adv       = pf["adv_ratio"]

    return float(
        0.20 * s_hurst     +
        0.15 * s_halflife  +
        0.10 * s_zc        +
        0.15 * s_corr_mean +
        0.15 * s_corr_stab +
        0.05 * s_regime    +
        0.08 * s_beta      +
        0.05 * s_vol       +
        0.04 * s_vol_corr  +
        0.03 * s_adv
    )


# ══════════════════════════════════════════════════════════════════════
# Formation Class
# ══════════════════════════════════════════════════════════════════════

class Formation:
    """
    HDBSCAN + UMAP 配對交易形成期模組

    第一層：單股特徵（10 維）→ UMAP 降維 → HDBSCAN 聚類
    第二層：配對層特徵（10 維）→ 品質評分排序
    """

    def __init__(
        self,
        price_df:    pd.DataFrame,
        form_start:  str,
        form_end:    str,
        top_n:                   int   = 20,
        sector_mapping:          dict  = None,
        min_tickers_for_pairing: int   = 2,
        hdbscan_min_cluster_size: int  = 5,
        hdbscan_min_samples:      int  = 2,
        hdbscan_metric:           str  = "euclidean",
        reduce_method:           str   = "umap",
        umap_n_components:       int   = 5,
        umap_n_neighbors:        int   = 40,
        umap_min_dist:           float = 0.01,
        umap_random_state:       int   = 42,
        adf_max_lags:            int   = 1,
        adf_pvalue_threshold:    float = 0.01,
        max_sector_ratio:        float = 0.3,
        min_corr:                float = 0.50,
        min_zero_crossings:      int   = 5,
        hurst_threshold:         float = 0.5,
        halflife_min:            float = 1.0,
        halflife_max:            float = 60.0,
        roll_corr_window:        int   = 60,
        max_beta_diff:           float = 0.8,
        max_vol_ratio:           float = 3.0,
        min_adv_ratio:           float = 0.1,
        use_mom1_filter:         bool  = True,
        market_returns: Optional[pd.Series] = None,
        feature_mode:            str   = "stats10",
    ):
        self.price_df   = price_df.copy()
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping          = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.max_sector_ratio        = max_sector_ratio

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        self.reduce_method     = reduce_method.lower()
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors  = umap_n_neighbors
        self.umap_min_dist     = umap_min_dist
        self.umap_random_state = umap_random_state

        if self.reduce_method == "umap" and not UMAP_AVAILABLE:
            raise RuntimeError("umap-learn 未安裝：pip install umap-learn")

        self.adf_max_lags         = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.min_corr             = min_corr
        self.min_zero_crossings   = min_zero_crossings
        self.hurst_threshold      = hurst_threshold
        self.halflife_min         = halflife_min
        self.halflife_max         = halflife_max
        self.roll_corr_window     = roll_corr_window
        self.max_beta_diff        = max_beta_diff
        self.max_vol_ratio        = max_vol_ratio
        self.min_adv_ratio        = min_adv_ratio
        self.use_mom1_filter      = use_mom1_filter
        self.market_returns       = market_returns
        self.feature_mode         = feature_mode.lower()

        self.selected_pairs:  pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict         = {}
        self._beta_map:       dict         = {}

    def _build_beta_cache(self, log_prices: pd.DataFrame, tickers: list[str]) -> None:
        if self.market_returns is None:
            self._beta_map = {t: 1.0 for t in tickers}
            return
        for ticker in tickers:
            ret     = np.diff(log_prices[ticker].values)
            mkt_arr = np.diff(self.market_returns.reindex(log_prices.index).values)
            min_len = min(len(ret), len(mkt_arr))
            self._beta_map[ticker] = _compute_beta(ret[:min_len], mkt_arr[:min_len])

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        log_prices = np.log(self.price_df)
        tickers    = log_prices.columns.tolist()

        mkt_arr = None
        if self.market_returns is not None:
            aligned = self.market_returns.reindex(self.price_df.index)
            mkt_arr = np.diff(np.log(aligned.ffill().values + 1e-12))

        feat_rows, valid_tickers = [], []
        for ticker in tickers:
            series = log_prices[ticker].values
            if len(series) < 30 or not np.all(np.isfinite(series)):
                continue
            ticker_ret = np.diff(series)
            mkt_sub    = mkt_arr if (mkt_arr is not None and len(mkt_arr) == len(ticker_ret)) else None
            feat_rows.append(_extract_features(series, market_ret=mkt_sub))
            valid_tickers.append(ticker)

        if not feat_rows:
            return np.empty((0, 0)), []

        return RobustScaler().fit_transform(np.vstack(feat_rows)), valid_tickers

    def _umap_reduce(self, X: np.ndarray) -> np.ndarray:
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        n_neigh  = min(self.umap_n_neighbors,  n_stocks - 1)
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
        n_comp = min(self.umap_n_components, X.shape[0] - 1)
        if n_comp < 1:
            return X
        return PCA(n_components=n_comp, random_state=self.umap_random_state).fit_transform(X)

    def _pca_umap_reduce(self, X: np.ndarray, pca_components: int = 15) -> np.ndarray:
        """先 PCA 降維去除噪音，再 UMAP 非線性嵌入，降低退化解機率。"""
        from sklearn.decomposition import PCA
        n = X.shape[0]
        pca_comp = min(pca_components, n - 1, X.shape[1])
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

        print("  [Formation] HDBSCAN 未找到任何群落。")
        return np.full(X.shape[0], -1)

    def _cointegration_within_clusters(
        self,
        tickers: list[str],
        labels:  np.ndarray,
    ) -> pd.DataFrame:
        log_prices    = np.log(self.price_df[tickers])
        unique_labels = set(labels) - {-1}

        if not unique_labels:
            print("  [Formation] 無有效群落。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(
            f"  [Formation] 聚類結果：{len(unique_labels)} 群落 | "
            f"{noise_count} 噪音 | "
            f"ADF p≤{self.adf_pvalue_threshold} | corr≥{self.min_corr}"
        )

        self._build_beta_cache(log_prices, tickers)

        group_map: dict[int, list[str]] = {}
        for t, lbl in zip(tickers, labels):
            if lbl != -1:
                group_map.setdefault(int(lbl), []).append(t)

        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}
        if not valid_groups:
            return pd.DataFrame()

        print(f"  [Formation] 有效群落：{len(valid_groups)} 組")

        eg_records, passed_count, rejected_count = [], 0, 0

        for cluster_lbl, group_tickers in sorted(valid_groups.items()):
            if len(group_tickers) > 100:
                import random
                random.seed(self.umap_random_state)
                group_tickers = random.sample(group_tickers, 100)

            for i, ta in enumerate(group_tickers):
                log_a  = log_prices[ta].values
                ret_a  = np.diff(log_a)
                sec_a  = self.sector_mapping.get(ta.upper(), "Unknown")
                beta_a = self._beta_map.get(ta, 1.0)
                vol_a_raw = (
                    self.price_df[ta + "_vol"].values
                    if (ta + "_vol") in self.price_df.columns
                    else np.ones(len(log_a))
                )

                for j in range(i + 1, len(group_tickers)):
                    tb     = group_tickers[j]
                    log_b  = log_prices[tb].values
                    ret_b  = np.diff(log_b)
                    sec_b  = self.sector_mapping.get(tb.upper(), "Unknown")
                    beta_b = self._beta_map.get(tb, 1.0)
                    vol_b_raw = (
                        self.price_df[tb + "_vol"].values
                        if (tb + "_vol") in self.price_df.columns
                        else np.ones(len(log_b))
                    )

                    # 步驟 1：相關係數初篩
                    corr = float(np.corrcoef(log_a, log_b)[0, 1])
                    if corr < self.min_corr:
                        rejected_count += 1
                        continue

                    # 步驟 2：ADF 共整合
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

                    # 步驟 3：半衰期
                    halflife = _compute_halflife(best_resid)
                    if not (self.halflife_min <= halflife <= self.halflife_max):
                        rejected_count += 1
                        continue

                    # 步驟 4：Hurst
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

                    # 步驟 6：Beta 差異
                    if abs(beta_a - beta_b) > self.max_beta_diff:
                        rejected_count += 1
                        continue

                    # 步驟 7：波動率比率
                    v_a = float(np.std(ret_a, ddof=1)) if len(ret_a) > 1 else 1.0
                    v_b = float(np.std(ret_b, ddof=1)) if len(ret_b) > 1 else 1.0
                    if max(v_a, v_b) / (min(v_a, v_b) + 1e-12) > self.max_vol_ratio:
                        rejected_count += 1
                        continue

                    # 步驟 8：ADV 比率
                    min_len_v = min(len(vol_a_raw), len(vol_b_raw))
                    if _compute_adv_ratio(vol_a_raw[:min_len_v], vol_b_raw[:min_len_v]) < self.min_adv_ratio:
                        rejected_count += 1
                        continue

                    # 步驟 9：配對特徵與品質評分
                    min_len_r  = min(len(ret_a), len(ret_b))
                    pair_feats = _compute_pair_features(
                        log_a       = log_a,
                        log_b       = log_b,
                        ret_a       = ret_a[:min_len_r],
                        ret_b       = ret_b[:min_len_r],
                        vol_a       = vol_a_raw[:min_len_v],
                        vol_b       = vol_b_raw[:min_len_v],
                        residuals   = best_resid,
                        beta_a      = beta_a,
                        beta_b      = beta_b,
                        roll_window = self.roll_corr_window,
                    )
                    quality_score   = _pair_quality_score(pair_feats)
                    passed_count   += 1
                    assigned_sector = sec_a if sec_a == sec_b else f"Cluster_{cluster_lbl}"

                    eg_records.append({
                        "Form_Start":       self.form_start,
                        "Form_End":         self.form_end,
                        "Sector":           assigned_sector,
                        "Cluster_Label":    cluster_lbl,
                        "Ticker_A":         best_a,
                        "Ticker_B":         best_b,
                        "ADF_Stat":         round(best_stat,      6),
                        "ADF_PValue":       round(best_pval,      6),
                        "Hedge_Ratio":      round(best_beta_coef, 6),
                        "OLS_Alpha":        round(best_alpha,     6),
                        "Spread_Mean":      round(float(np.mean(best_resid)),        6),
                        "Spread_Std":       round(float(np.std(best_resid, ddof=1)), 6),
                        "Hurst_Spread":     round(pair_feats["hurst_spread"],        4),
                        "Halflife":         round(pair_feats["halflife"],            2),
                        "Zero_Crossings":   int(pair_feats["zero_crossings"]),
                        "Corr_Mean":        round(pair_feats["corr_mean"],           4),
                        "Corr_Stability":   round(pair_feats["corr_stability"],      4),
                        "Corr_Regime_Diff": round(pair_feats["corr_regime_diff"],    4),
                        "Correlation":      round(corr,                              6),
                        "Beta_A":           round(beta_a,                            4),
                        "Beta_B":           round(beta_b,                            4),
                        "Beta_Diff":        round(pair_feats["beta_diff"],           4),
                        "Vol_Ratio":        round(pair_feats["vol_ratio"],           4),
                        "Volume_Corr":      round(pair_feats["volume_corr"],         4),
                        "ADV_Ratio":        round(pair_feats["adv_ratio"],           4),
                        "Quality_Score":    round(quality_score,                     4),
                        "Mom1_Diff":        0.0,
                    })

        print(f"  [Formation] 篩選結果：通過 {passed_count} 對 | 排除 {rejected_count} 對")

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
                f"  [Formation] Mom1 篩選：門檻={mom1_threshold:.4f} | "
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

        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if self.max_sector_ratio > 0:
            max_per_sec = max(1, int(self.top_n * self.max_sector_ratio))
            sec_counts: dict[str, int] = {}
            diversified = []
            for _, row in eg_df.iterrows():
                sec = row["Sector"]
                sec_counts[sec] = sec_counts.get(sec, 0)
                if sec_counts[sec] < max_per_sec:
                    diversified.append(row)
                    sec_counts[sec] += 1
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
