# ======================================================================
"""
HDBSCAN 分群配對交易滾動回測系統 (交易明細版) - 時序多因子特徵空間版
核心功能：以 6 大穩健金融因子作為分群特徵空間，跳過降維步驟，
          直接以 HDBSCAN 密度分群，從同群內挑選最優配對，
          結合 Engle-Granger OLS Spread 建構 Z-Score 執行配對交易。
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
        deviate = np.cumsum(seg - mean_seg)
        std_val = np.std(seg, ddof=1)
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
    return float(np.clip(h, 0.0, 1.0))


# HDBSCAN
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


# ══════════════════════════════════════════════════════════════════════════════
# 輔助函式
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    ADF 檢定（no constant），同時回傳 (統計量, p 值)。
    """
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）- 多因子特徵空間 HDBSCAN 版本
# ══════════════════════════════════════════════════════════════════════════════

class Formation:
    """
    形成期四步驟（時序多因子特徵空間版）：
      1. 特徵萃取  → 每支股票提取 6 大金融時序因子（標準化後）
      2. 降維跳過  → 不降維，直接使用 6 維因子特徵矩陣
      3. HDBSCAN   → 密度分群，自動決定群數，噪音點（label=-1）排除
      4. 同產業 × 同群落 雙重篩選 → EG 共整合，ADF p 值門檻過濾後依統計量升序選 top_n（含行業分散限制）
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,           # 產業分類字典，強制同產業配對
        min_tickers_for_pairing: int = 2,
        # HDBSCAN 參數
        hdbscan_min_cluster_size: int = 3,     # 最小群落大小
        hdbscan_min_samples: int = 1,          # 核心點最小鄰居數
        hdbscan_metric: str = "euclidean",     # 距離度量
        # 降維參數（接口保留但跳過）
        reduce_method: str = "none",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        # ADF p 值門檻
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.05,
        max_sector_ratio: float = 0.3,
    ):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        self.reduce_method = "none"
        self.adf_max_lags           = adf_max_lags
        self.adf_pvalue_threshold   = adf_pvalue_threshold
        self.max_sector_ratio       = max_sector_ratio

        self.selected_pairs: pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict = {}

    # ── Step 1：特徵萃取與標準化 ─────────────────────────────────────────────
    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        log_prices = np.log(self.price_df)
        tickers    = log_prices.columns.tolist()

        returns_df = log_prices.diff().dropna()
        if returns_df.empty or len(returns_df.columns) < 2:
            return np.empty((0, 0)), []

        market_returns = returns_df.mean(axis=1).values
        feat_rows, valid_tickers = [], []
        t_indices = np.arange(len(log_prices))

        for ticker in tickers:
            prices = log_prices[ticker].values
            if len(prices) < 30 or not np.all(np.isfinite(prices)):
                continue

            ticker_ret = returns_df[ticker].values

            # --- 1. Market Beta ---
            cov_mat = np.cov(ticker_ret, market_returns)
            beta = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12) if cov_mat[1, 1] > 1e-12 else 0.0

            # --- 2. Volatility ---
            vol = np.std(ticker_ret, ddof=1) if len(ticker_ret) > 1 else 0.0

            # --- 3. Skewness ---
            skew = float(pd.Series(ticker_ret).skew())

            # --- 4. Kurtosis ---
            kurt = float(pd.Series(ticker_ret).kurt())

            # --- 5. Trend Slope ---
            try:
                x_mat = np.column_stack([np.ones(len(prices)), t_indices])
                coeffs, _, _, _ = np.linalg.lstsq(x_mat, prices, rcond=None)
                slope = float(coeffs[1])
            except Exception:
                slope = 0.0

            # --- 6. Idiosyncratic Volatility ---
            try:
                alpha, beta_val, resid = _ols(ticker_ret, market_returns)
                idio_vol = np.std(resid, ddof=1) if len(resid) > 1 else 0.0
            except Exception:
                idio_vol = vol

            feats = np.array([beta, vol, skew, kurt, slope, idio_vol], dtype=np.float64)
            feats = np.where(np.isfinite(feats), feats, 0.0)

            feat_rows.append(feats)
            valid_tickers.append(ticker)

        if not feat_rows:
            return np.empty((0, 0)), []

        X = np.vstack(feat_rows)
        X = StandardScaler().fit_transform(X)
        return X, valid_tickers

    # ── Step 3：HDBSCAN 分群 ────────────────────────────────────────────────
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

    # ── Step 4：同產業 × 同群落 雙重篩選 + EG 共整合 ──────────────────────────
    def _cointegration_within_clusters(
        self, tickers: list[str], labels: np.ndarray
    ) -> pd.DataFrame:
        log_prices = np.log(self.price_df[tickers])
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("  [Formation] HDBSCAN 未找到任何有效群落（全為噪音點）。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(f"  [Formation] HDBSCAN 分群結果：{len(unique_labels)} 個群落，"
              f"{noise_count} 個噪音點排除，"
              f"ADF p 值門檻 = {self.adf_pvalue_threshold:.2f} "
              f"({'保守 1%' if self.adf_pvalue_threshold <= 0.01 else '積極 5%'})")

        ticker_meta: dict[str, tuple[str, int]] = {}
        for t, lbl in zip(tickers, labels):
            sector = self.sector_mapping.get(t.upper(), "Unknown")
            ticker_meta[t] = (sector, int(lbl))

        group_map: dict[tuple[str, int], list[str]] = {}
        for t, (sec, lbl) in ticker_meta.items():
            if sec == "Unknown" or lbl == -1:
                continue
            group_map.setdefault((sec, lbl), []).append(t)

        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}
        if not valid_groups:
            print("  [Formation] 同產業 × 同群落後無有效配對組合。")
            return pd.DataFrame()

        eg_records = []
        passed_count = 0
        rejected_count = 0

        for (sector, cluster_lbl), group_tickers in sorted(valid_groups.items()):
            for i, ta in enumerate(group_tickers):
                log_a = log_prices[ta].values
                for j in range(i + 1, len(group_tickers)):
                    tb    = group_tickers[j]
                    log_b = log_prices[tb].values

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
                    if halflife < 2.0 or halflife > 60.0:
                        rejected_count += 1
                        continue

                    # C. Hurst 指數篩選 (Hurst < 0.40，強烈均值回歸傾向)
                    hurst = _compute_hurst(best_resid)
                    if hurst >= 0.40:
                        rejected_count += 1
                        continue
                    
                    passed_count += 1
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    eg_records.append({
                        "Form_Start":    self.form_start,
                        "Form_End":      self.form_end,
                        "Sector":        sector,
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

        print(f"  [Formation] EG 檢定：{passed_count} 對通過 p < {self.adf_pvalue_threshold}，"
              f"{rejected_count} 對被 p 值門檻排除。")

        if not eg_records:
            return pd.DataFrame()

        return pd.DataFrame(eg_records).sort_values("ADF_Stat").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        X, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 初始化一個全為噪音 (-1) 的 labels 陣列
        labels = np.full(len(valid_tickers), -1, dtype=int)
        ticker_to_idx = {t: i for i, t in enumerate(valid_tickers)}

        # 按 GICS 產業別進行分組聚類
        sector_to_tickers = {}
        for t in valid_tickers:
            sec = self.sector_mapping.get(t.upper(), "Unknown")
            if sec != "Unknown":
                sector_to_tickers.setdefault(sec, []).append(t)

        global_cluster_counter = 0
        self.cluster_labels_ = {}

        # 逐產業執行 HDBSCAN 分群
        for sector, sec_tickers in sorted(sector_to_tickers.items()):
            n_sec = len(sec_tickers)
            if n_sec < self.min_tickers_for_pairing:
                continue

            # 取得該產業股票的特徵子矩陣
            sec_indices = [ticker_to_idx[t] for t in sec_tickers]
            X_sec = X[sec_indices]

            # HDBSCAN 分群
            sec_labels = self._hdbscan_cluster(X_sec)

            # 將局部 label 映射到全局唯一的 label
            for t, lbl in zip(sec_tickers, sec_labels):
                idx = ticker_to_idx[t]
                if lbl != -1:
                    labels[idx] = global_cluster_counter + lbl
                    self.cluster_labels_[t] = global_cluster_counter + lbl
                else:
                    labels[idx] = -1
                    self.cluster_labels_[t] = -1

            unique_sec_labels = set(sec_labels) - {-1}
            if unique_sec_labels:
                global_cluster_counter += max(unique_sec_labels) + 1

        # 確保 self.cluster_labels_ 包含所有 valid_tickers
        for t in valid_tickers:
            if t not in self.cluster_labels_:
                self.cluster_labels_[t] = -1

        # Step 4：同產業 × 同群落 EG 共整合
        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if getattr(self, "max_sector_ratio", 0) > 0:
            max_pairs_per_cluster = max(1, int(self.top_n * self.max_sector_ratio))
            cluster_counts = {}
            diversified_records = []
            for _, row in eg_df.iterrows():
                cluster_lbl = row["Cluster_Label"]
                if cluster_lbl not in cluster_counts:
                    cluster_counts[cluster_lbl] = 0
                if cluster_counts[cluster_lbl] < max_pairs_per_cluster:
                    diversified_records.append(row)
                    cluster_counts[cluster_lbl] += 1
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


# ══════════════════════════════════════════════════════════════════════════════
# Class 2：Trading（交易期模組）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PairState:
    position: int = 0
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    cooldown_dir: int = 0
    prev_total_pnl: float = 0.0


class Trading:
    def __init__(
        self,
        price_df: pd.DataFrame,
        trade_dates: pd.DatetimeIndex,
        selected_pairs: pd.DataFrame,
        capital_per_pair: float,
        fee_rate: float,
        slippage_rate: float,
        stop_loss_pct: float,
        entry_z: float,
        exit_z: float,
        zscore_window: int,
        allow_reentry: bool = False,
        zscore_clip: float = 10.0,
        min_spread_std: float = 1e-6,
        use_dynamic_stop: bool = False,
        dynamic_stop_z: float = 3.0,
        portfolio_stop_loss_pct: float = 0.10,
        use_vol_adjust: bool = False,
    ):
        self.price_df        = price_df.copy()
        self.trade_dates     = trade_dates
        self.selected_pairs  = selected_pairs
        self.capital_per_pair = capital_per_pair
        self.friction_rate   = fee_rate + slippage_rate
        self.stop_loss_pct   = stop_loss_pct
        self.entry_z         = entry_z
        self.exit_z          = exit_z
        self.zscore_window   = zscore_window
        self.allow_reentry   = allow_reentry
        self.zscore_clip     = zscore_clip
        self.min_spread_std  = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z  = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust  = use_vol_adjust
        self.period_pnl: float = 0.0

    def _execute_entry(self, state, z, p_a, p_b, hedge_ratio):
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)

        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b =  v_b / p_b
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a =  v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a  = p_a
        state.entry_price_b  = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state, current_trade_pnl, stop_loss=False):
        state.realized_pnl += current_trade_pnl
        if stop_loss:
            state.is_stopped = True if not self.allow_reentry else False
            if self.allow_reentry:
                state.cooldown_dir = state.position
        else:
            state.cooldown_dir = state.position
        state.position = 0
        state.shares_a = 0.0
        state.shares_b = 0.0
        state.entry_price_a = 0.0
        state.entry_price_b = 0.0
        state.trade_entry_fee = 0.0

    def _simulate_pair(
        self, period_start, period_end, sector, ticker_a, ticker_b, pair_rank,
        hedge_ratio, ols_alpha, form_spread_mean, form_spread_std,
        log_mean_a, log_std_a, log_mean_b, log_std_b,
        cluster_label, cluster_group,
    ) -> pd.DataFrame:

        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns:
            return pd.DataFrame()

        price_a = self.price_df[ticker_a].dropna()
        price_b = self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a    = price_a.loc[common_idx]
        price_b    = price_b.loc[common_idx]

        if len(price_a) < 5:
            return pd.DataFrame()

        log_a = np.log(price_a)
        log_b = np.log(price_b)

        # Z-Score 計算
        if self.zscore_window == 0:
            spread   = log_a - ols_alpha - hedge_ratio * log_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore   = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = pd.Series(hedge_ratio, index=common_idx)
            alpha_series = pd.Series(ols_alpha,   index=common_idx)
        else:
            w = self.zscore_window
            n = len(log_a)
            la_vals, lb_vals = log_a.values, log_b.values
            roll_alpha = np.full(n, np.nan)
            roll_beta  = np.full(n, np.nan)
            roll_mean  = np.full(n, np.nan)
            roll_std   = np.full(n, np.nan)

            for k in range(w - 1, n):
                ya = la_vals[k - w + 1: k + 1]
                xb = lb_vals[k - w + 1: k + 1]
                a_, b_, r_ = _ols(ya, xb)
                roll_alpha[k] = a_
                roll_beta[k]  = b_
                roll_mean[k]  = float(np.mean(r_))
                roll_std[k]   = float(np.std(r_, ddof=1)) if len(r_) > 1 else 0.0

            roll_alpha_s = pd.Series(roll_alpha, index=common_idx)
            roll_beta_s  = pd.Series(roll_beta,  index=common_idx)
            roll_mean_s  = pd.Series(roll_mean,  index=common_idx)
            roll_std_s   = pd.Series(roll_std,   index=common_idx)

            spread     = log_a - roll_alpha_s - roll_beta_s * log_b
            safe_std_s = np.maximum(roll_std_s, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std_s * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std_s
            zscore     = np.clip((spread - roll_mean_s) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = roll_beta_s
            alpha_series = roll_alpha_s

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()

        price_a      = price_a.loc[valid_idx]
        price_b      = price_b.loc[valid_idx]
        zscore       = zscore.loc[valid_idx]
        beta_series  = beta_series.loc[valid_idx]
        alpha_series = alpha_series.loc[valid_idx]

        dates_arr  = valid_idx
        zscore_arr = zscore.values
        pa_arr     = price_a.values
        pb_arr     = price_b.values
        beta_arr   = beta_series.values
        alpha_arr  = alpha_series.values

        base_log = {
            "Period_Start":   period_start,   "Period_End":     period_end,
            "Sector":         sector,          "Cluster_Label":  cluster_label,
            "Pair_Rank":      pair_rank,
            "Ticker_A":       ticker_a,        "Ticker_B":       ticker_b,
            "Log_Mean_A":     log_mean_a,      "Log_Std_A":      log_std_a,
            "Log_Mean_B":     log_mean_b,      "Log_Std_B":      log_std_b,
        }

        state = PairState()
        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_ols_alpha, out_z, out_pos = [], [], [], []
        out_unr, out_rea, out_cum = [], [], []
        out_status, out_tpnl, out_days, out_delta = [], [], [], []

        def _append_row(date, p_a, p_b, c_beta, c_alpha, z_val, pos,
                        unr, rea, cum, status, tpnl, days, delta):
            out_dates.append(date);      out_pa.append(round(p_a, 4));     out_pb.append(round(p_b, 4))
            out_hr.append(round(c_beta, 4)); out_ols_alpha.append(round(c_alpha, 6))
            out_z.append(round(z_val, 4));   out_pos.append(pos)
            out_unr.append(round(unr, 4));   out_rea.append(round(rea, 4)); out_cum.append(round(cum, 4))
            out_status.append(status);   out_tpnl.append(round(tpnl, 4))
            out_days.append(days);        out_delta.append(round(delta, 4))

        for i in range(len(dates_arr)):
            date    = dates_arr[i]
            z       = 0.0 if np.isnan(zscore_arr[i]) else float(zscore_arr[i])
            p_a, p_b = float(pa_arr[i]), float(pb_arr[i])
            c_beta   = float(beta_arr[i])  if not np.isnan(beta_arr[i])  else hedge_ratio
            c_alpha  = float(alpha_arr[i]) if not np.isnan(alpha_arr[i]) else ols_alpha

            unr, tpnl, status = 0.0, 0.0, "HOLD_CASH"

            if state.is_stopped:
                _append_row(date, p_a, p_b, c_beta, c_alpha, z, 0,
                            0.0, state.realized_pnl, state.realized_pnl,
                            "STOPPED", 0.0, 0, 0.0)
                continue

            if   state.cooldown_dir == -1 and z <= self.exit_z:  state.cooldown_dir = 0
            elif state.cooldown_dir ==  1 and z >= -self.exit_z: state.cooldown_dir = 0

            if state.position != 0:
                state.days_held += 1
                raw_unr  = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                cur_tpnl = raw_unr - state.trade_entry_fee - exit_fee

                is_cap_stop = self.stop_loss_pct > 0 and (-cur_tpnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, cur_tpnl, stop_loss=True)
                    tpnl, status = cur_tpnl, "STOP_LOSS_TRIGGERED"
                elif (state.position == -1 and z <= self.exit_z) or (state.position == 1 and z >= -self.exit_z):
                    self._execute_close(state, cur_tpnl, stop_loss=False)
                    tpnl, status = cur_tpnl, "EXIT"
                else:
                    exit_fee_est = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                    unr    = raw_unr - state.trade_entry_fee - exit_fee_est
                    status = "HOLDING"
            else:
                if abs(z) > self.entry_z:
                    entered, unr = self._execute_entry(state, z, p_a, p_b, c_beta)
                    status = ("ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A") if entered else "HOLD_CASH (COOLDOWN)"
                else:
                    status = "HOLD_CASH"

            cum   = state.realized_pnl + unr
            delta = cum - state.prev_total_pnl
            state.prev_total_pnl = cum

            _append_row(date, p_a, p_b, c_beta, c_alpha, z, state.position,
                        unr, state.realized_pnl, cum, status, tpnl, state.days_held, delta)

            if status in ("STOP_LOSS_TRIGGERED", "EXIT"):
                state.days_held = 0

            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    _append_row(
                        dates_arr[j], float(pa_arr[j]), float(pb_arr[j]),
                        float(beta_arr[j]) if not np.isnan(beta_arr[j]) else hedge_ratio,
                        float(alpha_arr[j]) if not np.isnan(alpha_arr[j]) else ols_alpha,
                        0.0 if np.isnan(zscore_arr[j]) else float(zscore_arr[j]),
                        0, 0.0, state.realized_pnl, state.realized_pnl,
                        "STOPPED", 0.0, 0, 0.0
                    )
                break

        if state.position != 0 and out_status:
            if out_status[-1] not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                p_a_last, p_b_last = float(pa_arr[-1]), float(pb_arr[-1])
                raw_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                fee_final = (abs(state.shares_a) * p_a_last + abs(state.shares_b) * p_b_last) * self.friction_rate
                final_tpnl = raw_final - state.trade_entry_fee - fee_final
                state.realized_pnl += final_tpnl
                pnl_prev = out_cum[-2] if len(out_cum) > 1 else 0.0

                out_status[-1]     = "PERIOD_END_EXIT"
                out_rea[-1]        = round(state.realized_pnl, 4)
                out_cum[-1]        = round(state.realized_pnl, 4)
                out_unr[-1]        = 0.0
                out_tpnl[-1]       = round(final_tpnl, 4)
                out_delta[-1]      = round(state.realized_pnl - pnl_prev, 4)
                out_days[-1]       = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "OLS_Alpha": out_ols_alpha,
            "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unr, "Realized_PnL": out_rea,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_tpnl, "Days_Held": out_days, "Daily_Delta": out_delta,
        })
        for k, v in base_log.items():
            df_out[k] = v
        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start, period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"], ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                ols_alpha=float(row.get("OLS_Alpha", 0.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A",  1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B",  1.0)),
                cluster_label=int(row.get("Cluster_Label", -1)),
                cluster_group=str(row.get("Sector", "Unknown")),
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # ---- 實作後置投資組合總體止損斷路器 ----
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()
            
            cutoff_date = None
            for date_val, pnl_val in daily_cum_pnl.items():
                if pnl_val / total_cap <= -self.portfolio_stop_loss_pct:
                    cutoff_date = date_val
                    break
            
            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date
                    
                    df_before = df[before_mask]
                    
                    df_at = df[at_mask].copy()
                    final_realized = 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        final_realized = row_at["Cumulative_PnL"]
                        if row_at["Position"] != 0:
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Trade_PnL"]
                        else:
                            if row_at["Status"] not in ("STOPPED", "STOP_LOSS_TRIGGERED", "EXIT"):
                                df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOPPED"
                            final_realized = row_at["Realized_PnL"]
                            
                    df_after = df[after_mask].copy()
                    if not df_after.empty:
                        df_after.loc[df_after.index, "Position"] = 0
                        df_after.loc[df_after.index, "Unrealized_PnL"] = 0.0
                        df_after.loc[df_after.index, "Realized_PnL"] = final_realized
                        df_after.loc[df_after.index, "Cumulative_PnL"] = final_realized
                        df_after.loc[df_after.index, "Status"] = "STOPPED"
                        df_after.loc[df_after.index, "Trade_PnL"] = 0.0
                        df_after.loc[df_after.index, "Daily_Delta"] = 0.0
                        
                    new_dfs.append(pd.concat([df_before, df_at, df_after], ignore_index=True))
                dfs = new_dfs
        # ----------------------------------------------

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily.sum()) if not period_daily.empty else 0.0
        return log_df, self.period_pnl


# ══════════════════════════════════════════════════════════════════════════════
# Class 3：RollingBacktester
# ══════════════════════════════════════════════════════════════════════════════

class RollingBacktester:
    def __init__(
        self,
        top_n_list: list,
        stop_loss_list: list,
        zscore_window_list: list,
        entry_z: float,
        exit_z: float,
        formation_window: int,
        trading_window: int,
        rolling_step: int,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        allow_reentry: bool,
        zscore_clip: float,
        min_spread_std: float,
        min_tickers_for_pairing: int,
        hdbscan_min_cluster_size: int,
        hdbscan_min_samples: int,
        hdbscan_metric: str,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.05,
        output_dir: Path = None,
        reduce_method: str = "none",
        portfolio_stop_loss_pct_list: list = None,
        max_sector_ratio_list: list = None,
        dynamic_stop_z_list: list = None,
        use_vol_adjust_list: list = None,
    ):
        self.top_n_list = top_n_list
        self.stop_loss_list = stop_loss_list
        self.zscore_window_list = zscore_window_list
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.rolling_step = rolling_step
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing
        
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.adf_max_lags = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.output_dir = output_dir
        self.reduce_method = reduce_method

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

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
        print(f"\n🚀 開始 HDBSCAN MultiFactor Grid Search，共 {len(roll_start_indices)} 期，每期 {len(states)} 種參數組合...")

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
            print(f"  ▶ 第 {roll_idx+1:02d} 期 (交易: {ts_str} ~ {te_str})")

            # ── HDBSCAN 形成期（以最大 top_n * 5 計算一次以提供充足的配對池）
            formation = Formation(
                price_df=form_data,
                form_start=fs_str, form_end=fe_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing,
                hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,
                hdbscan_min_samples=self.hdbscan_min_samples,
                hdbscan_metric=self.hdbscan_metric,
                adf_max_lags=self.adf_max_lags,
                adf_pvalue_threshold=self.adf_pvalue_threshold,
                max_sector_ratio=0, # 在外部網格進行產業過濾
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty:
                continue

            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                # 產業上限過濾與 top_n 擷取
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
                rm_str   = "MULTIFACTOR"
                psl_str = f"PSL{int(p_stop*100)}" if p_stop > 0 else "PSL0"
                msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
                dsz_str = f"DSZ{int(dyn_z)}" if dyn_z > 0 else "DSZ0"
                vol_str  = "VolAdj" if vol_adj else "NoVol"
                filename = f"HDBSCAN_{rm_str}_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
                filepath = self.output_dir / filename
                full_log.to_csv(filepath, index=False)
                print(f"  - 已輸出: {filename} (共 {len(full_log)} 筆紀錄)")
        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 標準化策略進入點接口 (Unified Strategy Entry Point)
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    標準化調用接口，完全解耦資料載入 I/O
    """
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
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
