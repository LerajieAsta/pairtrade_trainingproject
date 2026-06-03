# ======================================================================
"""
HDBSCAN 分群配對交易滾動回測系統 (交易明細版) - 跨產業優化版
核心功能：以 HDBSCAN 對全市場（不按產業預分）形成期特徵向量進行一次性密度分群，
          排除噪音點後，在同一個分群內允許跨產業股票進行 Engle-Granger 共整合配對。
"""

import sqlite3
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def _compute_hurst(series: np.ndarray) -> float:
    """R/S 分析近似 Hurst 指數"""
    n = len(series)
    if n < 20:
        return 0.5
    diffs = np.diff(series)
    rs_list = []
    for seg_len in [n // 4, n // 2, n]:
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

# 試著載入 UMAP，如果安裝了 umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# 試著載入 HDBSCAN
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

    def hurst_approx():
        if len(log_price) < 20:
            return 0.5
        diffs = np.diff(log_price)
        rs_list = []
        for seg_len in [len(diffs) // 4, len(diffs) // 2, len(diffs)]:
            if seg_len < 4:
                continue
            seg = diffs[:seg_len]
            mean_seg = np.mean(seg)
            deviate  = np.cumsum(seg - mean_seg)
            rs = (np.max(deviate) - np.min(deviate)) / (np.std(seg, ddof=1) + 1e-8)
            rs_list.append((np.log(seg_len), np.log(rs + 1e-8)))
        if len(rs_list) < 2:
            return 0.5
        xs, ys = zip(*rs_list)
        try:
            h = float(np.polyfit(xs, ys, 1)[0])
        except Exception:
            h = 0.5
        return np.clip(h, 0.0, 1.0)

    vol_all  = float(np.std(ret, ddof=1)) if n > 1 else 0.0
    skew_all = float(pd.Series(ret).skew()) if n > 2 else 0.0
    kurt_all = float(pd.Series(ret).kurt()) if n > 3 else 0.0

    features = np.array([
        safe_ret(5),   safe_ret(21),  safe_ret(63),  safe_ret(126),  # 動量
        roll_vol(21),  roll_vol(63),  vol_all,                        # 波動率
        autocorr(1),   autocorr(5),   autocorr(21),                  # 自相關
        skew_all,      kurt_all,      hurst_approx(),                 # 統計矩
    ], dtype=np.float64)

    features = np.where(np.isfinite(features), features, 0.0)
    return features

# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）- HDBSCAN 跨產業/全市場聚類版本
# ══════════════════════════════════════════════════════════════════════════════

class Formation:
    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,           # 產業分類，僅用於記錄與後置分散化
        min_tickers_for_pairing: int = 2,
        # HDBSCAN 參數
        hdbscan_min_cluster_size: int = 30,    # 全市場聚類，調大 min_cluster_size
        hdbscan_min_samples: int = 10,
        hdbscan_metric: str = "euclidean",
        # 降維方法：'umap' 或 'pca'
        reduce_method: str = "umap",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 40,
        umap_min_dist: float = 0.01,
        umap_random_state: int = 42,
        # ADF p 值門檻
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.01,   # 預設更嚴格以防止跨產業假共整合
        max_sector_ratio: float = 0.3,
        feature_mode: str = "stats13",
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
                if current_min_cs != self.hdbscan_min_cluster_size:
                    print(f"  [Formation] HDBSCAN parameter fallback succeeded with min_cluster_size={min_cs}, min_samples={min_samp}!")
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

        # 僅按 cluster_label 分組，排除噪音 -1，不限產業
        group_map: dict[int, list[str]] = {}
        for t, lbl in zip(tickers, labels):
            if lbl == -1:
                continue
            group_map.setdefault(int(lbl), []).append(t)

        # 排除成員數不足的群組
        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}
        if not valid_groups:
            print("  [Formation] 全市場聚類後無有效配對組合。")
            return pd.DataFrame()

        print(f"  [Formation] 有效群落組數：{len(valid_groups)} 組")

        eg_records = []
        passed_count = 0
        rejected_count = 0

        for cluster_lbl, group_tickers in sorted(valid_groups.items()):
            # 限制單個 Cluster 內最大配對測試數以防卡死（通常 S&P500 的單群不會大於 100，保險起見設上限）
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

                    # OLS 擬合
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

                    # 篩選條件 1：ADF P-Value
                    if best_pval >= self.adf_pvalue_threshold:
                        rejected_count += 1
                        continue

                    # 篩選條件 2：半衰期
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
                    if halflife < 2.0 or halflife > 60.0:
                        rejected_count += 1
                        continue

                    # 篩選條件 3：Hurst 指數
                    hurst = _compute_hurst(best_resid)
                    if hurst >= 0.40:
                        rejected_count += 1
                        continue
                    
                    passed_count += 1
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    # 為了符合回測引擎以 "Sector" 進行限制，若兩者同行業則顯示該行業，否則標記為 "Cluster_X"
                    assigned_sector = sec_a if sec_a == sec_b else f"Cluster_{cluster_lbl}"

                    eg_records.append({
                        "Form_Start":    self.form_start,
                        "Form_End":      self.form_end,
                        "Sector":        assigned_sector,  # 設為 Cluster 編號或同行業名稱
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

        # ── 優化核心：全市場一次性降維與聚類 ──
        if self.reduce_method == "pca":
            X_embed = self._pca_reduce(X)
        else:
            X_embed = self._umap_reduce(X)

        labels = self._hdbscan_cluster(X_embed)
        self.cluster_labels_ = {t: int(lbl) for t, lbl in zip(valid_tickers, labels)}

        # 共整合篩選
        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 實施分散化限制
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

        # 附加對數統計量（交易期使用）
        log_prices  = np.log(self.price_df)
        mean_prices = log_prices.mean()
        std_prices  = log_prices.std()

        selected["Log_Mean_A"] = selected["Ticker_A"].map(mean_prices)
        selected["Log_Std_A"]  = selected["Ticker_A"].map(std_prices)
        selected["Log_Mean_B"] = selected["Ticker_B"].map(mean_prices)
        selected["Log_Std_B"]  = selected["Ticker_B"].map(std_prices)

        self.selected_pairs = selected
        return self.selected_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Class 2, 3, 4：與原本 HDBSCAN.py 大架構一致，無縫相容 Trading、DataProcessor 與 RollingBacktester
# ══════════════════════════════════════════════════════════════════════════════

# 我們可以直接繼承或重新綁定，但為了獨立執行性，我們將其餘部分複製於此。
# 這樣能保證 HDBSCAN_CrossSector 可以作為獨立模組被主程序 import。

from strategies.HDBSCAN import Trading, PairState, DataProcessor, RollingBacktester

def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    import inspect
    from pathlib import Path
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}
    
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester (跨產業聚類優化版)...")
    
    # 創建自訂的 RollingBacktester，以調用本檔案內優化的 Formation 類
    class CustomRollingBacktester(RollingBacktester):
        def run(self, price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping):
            max_concurrent = self.trading_window // self.rolling_step
            states = {}
            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)] = {
                    "logs":  [],
                    "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                              for _ in range(max_concurrent)],
                }

            roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
            
            for roll_idx, trade_start_idx in enumerate(roll_start_indices):
                form_start_idx = trade_start_idx - self.formation_window
                form_end_idx   = trade_start_idx
                trade_end_idx  = min(trade_start_idx + self.trading_window, total_days)

                form_data   = price_pivot.iloc[form_start_idx:form_end_idx]
                trade_data  = price_pivot.iloc[trade_start_idx:trade_end_idx]
                extended_start = max(0, trade_start_idx - max(self.zscore_window_list))
                extended_data_raw = price_pivot.iloc[extended_start:trade_end_idx]
                valid_cols  = (form_data.isnull().sum() + extended_data_raw.isnull().sum()) == 0
                form_data   = form_data.loc[:, valid_cols]
                trade_dates = trade_data.index
                extended_data  = extended_data_raw.loc[:, valid_cols]

                if form_data.shape[1] < 2 or trade_data.empty:
                    continue

                ts_str = str(all_dates[trade_start_idx])[:10]
                te_str = str(all_dates[trade_end_idx - 1])[:10]
                fs_str = str(all_dates[form_start_idx])[:10]
                fe_str = str(all_dates[form_end_idx - 1])[:10]
                print(f"  ▶ 處理中：第 {roll_idx+1:02d} 期 (交易: {ts_str} ~ {te_str})")

                # 使用本檔案定義的 Formation (全市場聚類 ＆ 跨產業配對)
                formation = Formation(
                    price_df=form_data,
                    form_start=fs_str, form_end=fe_str,
                    top_n=max(self.top_n_list) * 5,
                    sector_mapping=sector_mapping,
                    min_tickers_for_pairing=self.min_tickers_for_pairing,
                    hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,
                    hdbscan_min_samples=self.hdbscan_min_samples,
                    hdbscan_metric=self.hdbscan_metric,
                    umap_n_components=self.umap_n_components,
                    umap_n_neighbors=self.umap_n_neighbors,
                    umap_min_dist=self.umap_min_dist,
                    umap_random_state=self.umap_random_state,
                    adf_max_lags=self.adf_max_lags,
                    adf_pvalue_threshold=self.adf_pvalue_threshold,
                    reduce_method=getattr(self, "reduce_method", "umap"),
                    feature_mode=getattr(self, "feature_mode", "stats13"),
                    max_sector_ratio=0,
                )
                max_selected_pairs = formation.run()

                if max_selected_pairs.empty:
                    continue

                for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                    self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                    self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
                ):
                    if sec_ratio > 0:
                        max_pairs_per_sector = max(1, int(n * sec_ratio))
                        sector_counts = {}
                        diversified_records = []
                        for _, row in max_selected_pairs.iterrows():
                            sec = row["Sector"]
                            if sec not in sector_counts:
                                sector_counts[sec] = 0
                            if sector_counts[sec] < max_pairs_per_sector:
                                diversified_records.append(row)
                                sector_counts[sec] += 1
                            if len(diversified_records) >= n:
                                break
                        selected_pairs = pd.DataFrame(diversified_records).copy()
                    else:
                        selected_pairs = max_selected_pairs.head(n).copy()

                    if selected_pairs.empty:
                        continue

                    selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                    state  = states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)]
                    slots  = state["slots"]

                    free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                    slot_idx   = free_slots[0] if free_slots else min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                    cap_period   = slots[slot_idx]["capital"]
                    cap_per_pair = cap_period / n

                    trading = Trading(
                        price_df=extended_data, trade_dates=trade_dates,
                        selected_pairs=selected_pairs,
                        capital_per_pair=cap_per_pair,
                        fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
                        stop_loss_pct=sl, entry_z=self.entry_z, exit_z=self.exit_z,
                        zscore_window=z_win, allow_reentry=self.allow_reentry,
                        zscore_clip=self.zscore_clip, min_spread_std=self.min_spread_std,
                        use_dynamic_stop=(dyn_z > 0),
                        dynamic_stop_z=dyn_z,
                        portfolio_stop_loss_pct=p_stop,
                        use_vol_adjust=vol_adj,
                    )

                    trade_log_df, period_pnl = trading.run(ts_str, te_str)

                    if not trade_log_df.empty:
                        state["logs"].append(trade_log_df)

                    slots[slot_idx]["capital"]   = max(0, cap_period + period_pnl)
                    slots[slot_idx]["avail_idx"] = trade_end_idx

            self._export_results(states)

        def _export_results(self, states):
            print("\n✅ 回測完成！正在匯出交易紀錄...")
            for (n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj), state in states.items():
                if state["logs"]:
                    full_log = pd.concat(state["logs"], ignore_index=True)
                    sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                    rm_str   = getattr(self, "reduce_method", "umap").upper() + "_CrossSector"
                    psl_str = f"PSL{int(p_stop*100)}" if p_stop > 0 else "PSL0"
                    msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
                    dsz_str = f"DSZ{int(dyn_z)}" if dyn_z > 0 else "DSZ0"
                    vol_str  = "VolAdj" if vol_adj else "NoVol"
                    filename = f"HDBSCAN_{rm_str}_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
                    filepath = self.output_dir / filename
                    full_log.to_csv(filepath, index=False)
                    print(f"  - 已輸出: {filename} (共 {len(full_log)} 筆紀錄)")
            print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")

    engine = CustomRollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
