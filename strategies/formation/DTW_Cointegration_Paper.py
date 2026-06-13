# ======================================================================
"""
DTW 配對交易滾動回測系統 (交易明細版) - 許鈞翔 (2025) 論文對齊版
核心功能：
  1. 篩選出 Engle-Granger 共整合檢定 ADF p-value < adf_pvalue_threshold 的股票對。
  2. 對通過共整合的股票對，計算 Z-Score 標準化對數價格的 SSD 與 DTW 距離。
  3. 根據 DTW 距離（對照組）或基於 PCA 融合 SSD 與 DTW 距離的第一主成分 PC1 得分（實驗組）進行升序排序，挑選前 Top N。
"""

import sqlite3
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def _adf_stat(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """ADF 檢定（no constant），同時回傳 (統計量, p 值)"""
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0

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

def _sakoe_chiba_dtw(x: np.ndarray, y: np.ndarray, window: int = 15) -> float:
    """
    Sakoe-Chiba 限制窗口的快速 DTW (Dynamic Time Warping) 距離。
    時間軸扭曲限制在 `window` 天之內，時間複雜度為 O(N * W)。
    """
    n = len(x)
    m = len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    
    for i in range(1, n + 1):
        start_j = max(1, i - window)
        end_j = min(m, i + window)
        for j in range(start_j, end_j + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dp[i, j] = cost + min(
                dp[i - 1, j],     # Insertion
                dp[i, j - 1],     # Deletion
                dp[i - 1, j - 1]  # Match
            )
            
    return float(dp[n, m])

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

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）- 許鈞翔論文重構版
# ══════════════════════════════════════════════════════════════════════════════
class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    流程：先共整合檢定過濾 -> 計算 SSD 與 DTW 距離 -> 基於 DTW 或是 SSD+DTW (PCA) 排序。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, 
                 sector_mapping: dict = None, min_tickers_for_pairing: int = 2, dtw_window: int = 15,
                 method: str = "dtw", adf_pvalue_threshold: float = 0.01):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.dtw_window = dtw_window
        self.method = method.lower()
        self.adf_pvalue_threshold = adf_pvalue_threshold

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.mean_prices: pd.Series = pd.Series(dtype=float)
        self.std_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """對數價格標準化"""
        log_prices = np.log(self.price_df)
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        self.normalized_df = (log_prices - self.mean_prices) / self.std_prices
        return self.normalized_df

    def compute_pairs(self) -> pd.DataFrame:
        """先共整合篩選，再計算 SSD / DTW 距離"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        records = []

        # 根據產業分類分組
        sector_groups = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        skipped_unknown_count = 0
        for sector, sector_tickers in sector_groups.items():
            if sector == "Unknown":
                skipped_unknown_count = len(sector_tickers)
                continue
            
            if len(sector_tickers) < self.min_tickers_for_pairing: 
                continue

            n_sec = len(sector_tickers)
            for i in range(n_sec):
                ticker_b = sector_tickers[i]
                x_val = self.normalized_df[ticker_b].values
                var_x = np.var(x_val, ddof=1)
                
                for j in range(i + 1, n_sec):
                    ticker_a = sector_tickers[j]
                    y_val = self.normalized_df[ticker_a].values
                    
                    # 1. 雙向 OLS 擬合與共整合檢定
                    al_ab, be_ab, re_ab = _ols(y_val, x_val)
                    stat_ab, pval_ab = _adf_stat(re_ab, 1)

                    al_ba, be_ba, re_ba = _ols(x_val, y_val)
                    stat_ba, pval_ba = _adf_stat(re_ba, 1)

                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ticker_a, ticker_b
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = ticker_b, ticker_a

                    # A. ADF 共整合篩選 (依論文預設 adf_pvalue_threshold 為 0.01)
                    if best_pval >= self.adf_pvalue_threshold:
                        continue
                        
                    # B. Ornstein-Uhlenbeck 半衰期過濾 (2.0 <= halflife <= 40.0 天)
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
                        continue
                        
                    halflife = -np.log(2) / lambda_val
                    if halflife < 2.0 or halflife > 40.0:
                        continue
                        
                    # C. Hurst 指數篩選 (Hurst < 0.40)
                    hurst = _compute_hurst(best_resid)
                    if hurst >= 0.40:
                        continue
                    
                    # 通過篩選後，計算 SSD 與 DTW 距離
                    norm_a = self.normalized_df[best_a].values
                    norm_b = self.normalized_df[best_b].values
                    
                    # SSD 距離
                    ssd_dist = float(np.sum((norm_a - norm_b) ** 2))
                    
                    # DTW 距離 (Sakoe-Chiba window)
                    dtw_dist = _sakoe_chiba_dtw(norm_a, norm_b, window=self.dtw_window)
                    
                    spread_mean = np.mean(best_resid)
                    spread_std = np.std(best_resid, ddof=1) if len(best_resid) > 1 else 0.0
                    
                    records.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": best_a, "Ticker_B": best_b,
                        "SSD": round(ssd_dist, 6), "DTW_Dist": round(dtw_dist, 6), 
                        "Hedge_Ratio": round(best_beta, 4),
                        "OLS_Alpha": round(best_alpha, 6),
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std": round(spread_std, 6)
                    })

        # if skipped_unknown_count > 0:
        #     print(f"  [Formation] 跳過 {skipped_unknown_count} 支未分類股票。")

        if not records:
            return pd.DataFrame()
            
        return pd.DataFrame(records)

    def select_pairs(self) -> pd.DataFrame:
        pairs_df = self.compute_pairs()
        if pairs_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 根據指定模式進行排序
        if self.method == "dtw":
            # 論文對照組：依 DTW 距離升序排序
            selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()
        elif self.method == "ssd_dtw_pca":
            # 論文實驗組：標準化後使用 PCA 提取第一主成分排序
            if len(pairs_df) < 2:
                # 樣本太少無法做 PCA，退回 DTW
                selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                
                feats = pairs_df[["SSD", "DTW_Dist"]].values
                feats_scaled = StandardScaler().fit_transform(feats)
                
                pca = PCA(n_components=1, random_state=42)
                scores = pca.fit_transform(feats_scaled).flatten()
                
                # 調整得分方向，確保 loadings 為正，即綜合距離越小，得分越小
                loadings = pca.components_[0]
                if loadings[0] < 0:
                    scores = -scores
                
                pairs_df["PC1_Score"] = scores
                selected = pairs_df.sort_values("PC1_Score").head(self.top_n).copy()
        else:
            selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        mean_a_list, std_a_list, mean_b_list, std_b_list = [], [], [], []
        for _, row in selected.iterrows():
            mean_a_list.append(self.mean_prices[row["Ticker_A"]])
            std_a_list.append(self.std_prices[row["Ticker_A"]])
            mean_b_list.append(self.mean_prices[row["Ticker_B"]])
            std_b_list.append(self.std_prices[row["Ticker_B"]])
            
        selected["Log_Mean_A"] = mean_a_list
        selected["Log_Std_A"] = std_a_list
        selected["Log_Mean_B"] = mean_b_list
        selected["Log_Std_B"] = std_b_list

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs


# ══════════════════════════════════════════════════════════════════════════════
