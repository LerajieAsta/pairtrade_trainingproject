import sqlite3
import warnings
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Try to load UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Try to load HDBSCAN
try:
    import hdbscan
    HDBSCAN_LIB = "hdbscan"
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        HDBSCAN_LIB = "sklearn"
    except ImportError:
        raise ImportError("請先安裝 scikit-learn >= 1.3.0 或 hdbscan：pip install scikit-learn hdbscan")

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from strategies.formation._utils import _compute_hurst, _ols, _adf_stat

def _extract_features(log_price: np.ndarray) -> np.ndarray:
    """從對數價格序列萃取多維特徵向量，用於 HDBSCAN 分群。"""
    n = len(log_price)
    ret = np.diff(log_price)

    def safe_ret(days):
        if n >= days:
            return float(log_price[-1] - log_price[-days])
        return 0.0

    def roll_vol(days):
        if len(ret) >= days:
            return float(np.std(ret[-days:], ddof=1))
        return 0.0

    def autocorr(lag):
        if len(ret) <= lag:
            return 0.0
        x1, x2 = ret[:-lag], ret[lag:]
        try:
            return float(np.corrcoef(x1, x2)[0, 1])
        except Exception:
            return 0.0



    vol_all  = float(np.std(ret, ddof=1)) if n > 1 else 0.0
    skew_all = float(pd.Series(ret).skew()) if n > 2 else 0.0
    kurt_all = float(pd.Series(ret).kurt()) if n > 3 else 0.0

    features = np.array([
        safe_ret(5),   safe_ret(21),  safe_ret(63),  safe_ret(126),  # 動量
        roll_vol(21),  roll_vol(63),  vol_all,                        # 波動率
        autocorr(1),   autocorr(5),   autocorr(21),                  # 自相關
        skew_all,      kurt_all,      _compute_hurst(log_price),                 # 統計矩
    ], dtype=np.float64)

    features = np.where(np.isfinite(features), features, 0.0)
    return features

class Formation:
    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        hdbscan_min_cluster_size: int = 30,
        hdbscan_min_samples: int = 10,
        hdbscan_metric: str = "euclidean",
        reduce_method: str = "umap",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 40,
        umap_min_dist: float = 0.01,
        umap_random_state: int = 42,
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.01,
        max_sector_ratio: float = 0.3,
        feature_mode: str = "stats13",
        **kwargs
    ):
        self.price_df = price_df.copy()
        self.max_sector_ratio = max_sector_ratio
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        self.reduce_method = reduce_method.lower()
        if self.reduce_method == "umap" and not UMAP_AVAILABLE:
            raise RuntimeError("umap-learn 未安裝，請執行：pip install umap-learn")
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors  = umap_n_neighbors
        self.umap_min_dist     = umap_min_dist
        self.umap_random_state = umap_random_state

        self.adf_max_lags           = adf_max_lags
        self.adf_pvalue_threshold   = adf_pvalue_threshold
        self.feature_mode           = feature_mode.lower()

        self.selected_pairs: pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict = {}

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        log_prices = np.log(self.price_df)
        tickers    = log_prices.columns.tolist()

        if self.feature_mode == "path":
            price_rows, valid_tickers = [], []
            for ticker in tickers:
                series = log_prices[ticker].values
                if len(series) < 30 or not np.all(np.isfinite(series)):
                    continue
                norm_series = (series - np.mean(series)) / (np.std(series) + 1e-12)
                price_rows.append(norm_series)
                valid_tickers.append(ticker)
            if not price_rows:
                return np.empty((0, 0)), []
            return np.vstack(price_rows), valid_tickers

        feat_rows, valid_tickers = [], []
        for ticker in tickers:
            series = log_prices[ticker].values
            if len(series) < 30 or not np.all(np.isfinite(series)):
                continue
            feat_rows.append(_extract_features(series))
            valid_tickers.append(ticker)

        if not feat_rows:
            return np.empty((0, 0)), []

        X = StandardScaler().fit_transform(np.vstack(feat_rows))
        return X, valid_tickers

    def _umap_reduce(self, X: np.ndarray) -> np.ndarray:
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        n_neigh  = min(self.umap_n_neighbors,  n_stocks - 1)
        if n_comp < 1 or n_neigh < 1:
            return X
        try:
            reducer = umap.UMAP(
                n_components  = n_comp,
                n_neighbors   = n_neigh,
                min_dist      = self.umap_min_dist,
                random_state  = self.umap_random_state,
                low_memory    = True,
            )
            return reducer.fit_transform(X)
        except Exception:
            reducer = umap.UMAP(
                n_components  = n_comp,
                n_neighbors   = n_neigh,
                min_dist      = self.umap_min_dist,
                random_state  = self.umap_random_state,
                low_memory    = True,
                init          = "random"
            )
            return reducer.fit_transform(X)

    def _pca_reduce(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        if n_comp < 1:
            return X
        pca = PCA(n_components=n_comp, random_state=self.umap_random_state)
        return pca.fit_transform(X)

    def _hdbscan_cluster(self, X: np.ndarray) -> np.ndarray:
        current_min_cs = self.hdbscan_min_cluster_size
        current_min_samples = self.hdbscan_min_samples
        
        while current_min_cs >= 2:
            min_cs = min(current_min_cs, max(2, X.shape[0] // 5))
            min_samp = min(current_min_samples, min_cs - 1) if min_cs > 1 else 1
            min_samp = max(1, min_samp)
            
            if HDBSCAN_LIB == "hdbscan":
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size = min_cs,
                    min_samples      = min_samp,
                    metric           = self.hdbscan_metric,
                    core_dist_n_jobs = -1,
                )
            else:
                clusterer = sklearn_HDBSCAN(
                    min_cluster_size = min_cs,
                    min_samples      = min_samp,
                    metric           = self.hdbscan_metric,
                    n_jobs           = -1,
                )
            labels = clusterer.fit_predict(X)
            unique_labels = set(labels) - {-1}
            if len(unique_labels) >= 1:
                # if current_min_cs != self.hdbscan_min_cluster_size:
                #     print(f"  [Formation] HDBSCAN parameter fallback succeeded with min_cluster_size={min_cs}, min_samples={min_samp}!")
                return labels
            
            current_min_cs = current_min_cs // 2
            current_min_samples = max(1, current_min_samples // 2)
            
        print("  [Formation] HDBSCAN failed to find any clusters even with min_cluster_size=2. Returning all noise.")
        return np.full(X.shape[0], -1)

    def _cointegration_within_clusters(
        self, tickers: list[str], labels: np.ndarray
    ) -> pd.DataFrame:
        log_prices = np.log(self.price_df[tickers])
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("  [Formation] HDBSCAN 未找到任何有效群落（全為噪音點）。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(f"  [Formation] 全市場 HDBSCAN 分群結果：{len(unique_labels)} 個群落，"
              f"{noise_count} 個噪音點排除，"
              f"ADF p 值門檻 = {self.adf_pvalue_threshold:.2f}")

        group_map = {}
        for t, lbl in zip(tickers, labels):
            if lbl == -1:
                continue
            group_map.setdefault(int(lbl), []).append(t)

        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}
        if not valid_groups:
            print("  [Formation] 全市場聚類後無有效配對組合。")
            return pd.DataFrame()

        print(f"  [Formation] 有效群落組數：{len(valid_groups)} 組")

        eg_records = []
        passed_count = 0
        rejected_count = 0

        for cluster_lbl, group_tickers in sorted(valid_groups.items()):
            if len(group_tickers) > 100:
                print(f"  [Formation] 群落 {cluster_lbl} 規模過大 ({len(group_tickers)} 檔)，限制測試前 100 檔。")
                group_tickers = group_tickers[:100]

            for i, ta in enumerate(group_tickers):
                log_a = log_prices[ta].values
                sec_a = self.sector_mapping.get(ta.upper(), "Unknown")
                for j in range(i + 1, len(group_tickers)):
                    tb    = group_tickers[j]
                    log_b = log_prices[tb].values
                    sec_b = self.sector_mapping.get(tb.upper(), "Unknown")

                    al_ab, be_ab, re_ab = _ols(log_a, log_b)
                    stat_ab, pval_ab = _adf_stat(re_ab, self.adf_max_lags)

                    al_ba, be_ba, re_ba = _ols(log_b, log_a)
                    stat_ba, pval_ba = _adf_stat(re_ba, self.adf_max_lags)

                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ta, tb
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = tb, ta

                    if best_pval >= self.adf_pvalue_threshold:
                        rejected_count += 1
                        continue

                    dy = np.diff(best_resid)
                    y_lag = best_resid[:-1]
                    n_dy = len(dy)
                    x_mat = np.column_stack([np.ones(n_dy), y_lag])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(x_mat, dy, rcond=None)
                        lambda_val = coeffs[1]
                    except Exception:
                        lambda_val = 0.0

                    if lambda_val >= 0.0:
                        rejected_count += 1
                        continue
                    
                    halflife = -np.log(2) / lambda_val
                    if halflife < 1.0 or halflife > 60.0:
                        rejected_count += 1
                        continue

                    hurst = _compute_hurst(best_resid, already_stationary=True)
                    if hurst >= 0.50:
                        rejected_count += 1
                        continue
                    
                    passed_count += 1
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    assigned_sector = sec_a if sec_a == sec_b else f"Cluster_{cluster_lbl}"

                    eg_records.append({
                        "Form_Start":    self.form_start,
                        "Form_End":      self.form_end,
                        "Sector":        assigned_sector,
                        "Cluster_Label": cluster_lbl,
                        "Ticker_A":      best_a,
                        "Ticker_B":      best_b,
                        "ADF_Stat":      round(best_stat,   6),
                        "ADF_PValue":    round(best_pval,   6),
                        "Hedge_Ratio":   round(best_beta,   6),
                        "OLS_Alpha":     round(best_alpha,  6),
                        "Spread_Mean":   round(spread_mean, 6),
                        "Spread_Std":    round(spread_std,  6),
                    })

        print(f"  [Formation] EG 檢定：{passed_count} 對通過，{rejected_count} 對被排除。")

        if not eg_records:
            return pd.DataFrame()

        return pd.DataFrame(eg_records).sort_values("ADF_Stat").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        X, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if self.reduce_method == "pca":
            X_embed = self._pca_reduce(X)
        else:
            X_embed = self._umap_reduce(X)

        labels = self._hdbscan_cluster(X_embed)
        self.cluster_labels_ = {t: int(lbl) for t, lbl in zip(valid_tickers, labels)}

        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if getattr(self, "max_sector_ratio", 0) > 0:
            max_pairs_per_sec = max(1, int(self.top_n * self.max_sector_ratio))
            sector_counts = {}
            diversified_records = []
            for _, row in eg_df.iterrows():
                sec = row["Sector"]
                if sec not in sector_counts:
                    sector_counts[sec] = 0
                if sector_counts[sec] < max_pairs_per_sec:
                    diversified_records.append(row)
                    sector_counts[sec] += 1
                if len(diversified_records) >= self.top_n:
                    break
            selected = pd.DataFrame(diversified_records).copy()
        else:
            selected = eg_df.head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        log_prices  = np.log(self.price_df)
        mean_prices = log_prices.mean()
        std_prices  = log_prices.std()

        selected["Log_Mean_A"] = selected["Ticker_A"].map(mean_prices)
        selected["Log_Std_A"]  = selected["Ticker_A"].map(std_prices)
        selected["Log_Mean_B"] = selected["Ticker_B"].map(mean_prices)
        selected["Log_Std_B"]  = selected["Ticker_B"].map(std_prices)

        self.selected_pairs = selected
        return self.selected_pairs
