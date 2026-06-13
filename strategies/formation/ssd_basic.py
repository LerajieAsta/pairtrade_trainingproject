# ======================================================================
"""
SSD 配對交易滾動回測系統 (交易明細版)
核心功能：基於 SSD (Sum of Squared Differences) 與 Z-Score 的配對交易回測。
針對大型網格搜索進行效能最佳化，僅輸出詳細交易紀錄。
"""

import sqlite3
import warnings
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
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


# 忽略不必要的 Pandas/Numpy 警告以保持輸出整潔
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）
# ══════════════════════════════════════════════════════════════════════════════
class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    透過計算累積總回報指數 (Cumulative Total Returns Index) 的 SSD 來尋找走勢相近的股票對。
    累積總回報指數第一天價格歸一化為 1.0 (完全對應 Gatev et al. 2006 論文)。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, sector_mapping: dict = None, min_tickers_for_pairing: int = 2):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.first_day_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """將價格正規化為累積總回報指數 (首日價格設為 1.0)"""
        self.first_day_prices = self.price_df.iloc[0]
        # 防範分母為 0，若首日價格 <= 0 則設為 1.0
        safe_first_prices = np.where(self.first_day_prices > 1e-8, self.first_day_prices, 1.0)
        self.normalized_df = self.price_df / safe_first_prices
        return self.normalized_df

    def compute_ssd(self) -> pd.DataFrame:
        """計算產業內所有可能配對的 SSD (Sum of Squared Differences)"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        ssd_records = []

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

            # 向量化最佳化：使用 scipy 進行快速的兩兩配對距離計算
            norm_vals = self.normalized_df[sector_tickers].values.T
            
            # 使用 squared euclidean 計算 SSD
            ssd_matrix = ssd.pdist(norm_vals, metric='sqeuclidean')
            
            # 將 pdist 的 1D 陣列轉換為 Pair 組合
            # 距離法中 Beta 恆定為 1.0 (等權重 dollar weighting)
            idx = 0
            for i in range(len(sector_tickers)):
                ticker_b = sector_tickers[i]
                
                for j in range(i + 1, len(sector_tickers)):
                    ticker_a = sector_tickers[j]
                    
                    ssd_value = ssd_matrix[idx]
                    idx += 1
                    
                    beta = 1.0 # 距離法無 Beta 估計，等同於 1.0
                    
                    ssd_records.append({
                        "Sector": sector, "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                        "SSD": float(ssd_value), "Hedge_Ratio": float(beta),
                    })

        # if skipped_unknown_count > 0:
        #     print(f"  [Formation] 跳過 {skipped_unknown_count} 支未分類 (Unknown) 股票。")

        if not ssd_records: 
            return pd.DataFrame()

        # 先按 SSD 升序排序
        all_pairs_df = pd.DataFrame(ssd_records).sort_values("SSD").reset_index(drop=True)
        
        # 智慧型【先初篩再過濾】優化：
        # 既然最後只需要 top_n，我們只需要對 SSD 最接近的前 top_n * 15 組候選配對進行慢速共整合和 Hurst 檢驗
        # 這能將慢速統計擬合次數從 30,000+ 暴降至 ~300，速度直接飆升 30~50 倍！
        candidates_limit = max(200, self.top_n * 15)
        candidates = all_pairs_df.head(candidates_limit)
        
        filtered_records = []
        for _, row in candidates.iterrows():
            x_val = self.normalized_df[row["Ticker_B"]].values
            y_val = self.normalized_df[row["Ticker_A"]].values
            beta = row["Hedge_Ratio"]
            
            spread = y_val - beta * x_val
            
            # A. ADF 共整合檢驗 (p-value < 0.05，過濾隨機漫步)
            stat, pval = _adf_stat(spread, max_lags=1)
            if pval >= 0.05:
                continue
                
            # B. Ornstein-Uhlenbeck 半衰期過濾 (2.0 <= halflife <= 40.0 天)
            dy = np.diff(spread)
            y_lag = spread[:-1]
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
                
            # C. Hurst 指數篩選 (Hurst < 0.40，強均值回歸傾向)
            hurst = _compute_hurst(spread)
            if hurst >= 0.40:
                continue
                
            spread_mean = np.mean(spread)
            spread_std = np.std(spread, ddof=1) if len(spread) > 1 else 0.0
            
            filtered_records.append({
                "Form_Start": self.form_start, "Form_End": self.form_end,
                "Sector": row["Sector"], "Ticker_A": row["Ticker_A"], "Ticker_B": row["Ticker_B"],
                "SSD": round(row["SSD"], 6), "Hedge_Ratio": round(beta, 4),
                "Spread_Mean": round(spread_mean, 6),
                "Spread_Std": round(spread_std, 6)
            })
            
            if len(filtered_records) >= self.top_n * 5:
                break

        if not filtered_records:
            return pd.DataFrame()
            
        return pd.DataFrame(filtered_records).sort_values("SSD").reset_index(drop=True)

    def select_pairs(self) -> pd.DataFrame:
        """選出 SSD 最小的前 N 組配對"""
        ssd_df = self.compute_ssd()
        if ssd_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        selected = ssd_df.head(self.top_n).copy()
        selected["Rank"] = range(1, len(selected) + 1)

        first_price_a_list, first_price_b_list = [], []
        for _, row in selected.iterrows():
            first_price_a_list.append(self.price_df[row["Ticker_A"]].iloc[0])
            first_price_b_list.append(self.price_df[row["Ticker_B"]].iloc[0])
            
        selected["First_Price_A"] = first_price_a_list
        selected["First_Price_B"] = first_price_b_list

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        """執行形成期流程並回傳選定的配對"""
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs

# ══════════════════════════════════════════════════════════════════════════════
