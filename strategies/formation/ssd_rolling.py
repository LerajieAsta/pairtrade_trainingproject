"""
SSD Rolling 配對交易回測系統（滾動多期版）
核心功能：基於 SSD (Sum of Squared Differences) 進行滾動視窗的股票對篩選，
搭配 Z-Score 交易信號執行配對交易。為 ssd_basic 的滾動延伸版本。
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
from strategies.formation._utils import _compute_hurst, _ols, _adf_stat


warnings.filterwarnings("ignore")


class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    透過計算正規化對數價格的 SSD 來尋找走勢相近的股票對。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, sector_mapping: dict = None, min_tickers_for_pairing: int = 2, adf_pvalue_threshold: float = 0.05, **kwargs):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.halflife_max = kwargs.get("trading_window", 126) / 3.0

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.mean_prices: pd.Series = pd.Series(dtype=float)
        self.std_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """將價格轉換為對數價格，並進行 Z-Score 正規化"""
        log_prices = np.log(np.maximum(self.price_df, 1e-8))
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        self.normalized_df = (log_prices - self.mean_prices) / (self.std_prices + 1e-12)
        return self.normalized_df

    def compute_ssd(self) -> pd.DataFrame:
        """計算產業內所有可能配對的 SSD (Sum of Squared Differences)"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        ssd_records = []

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

            norm_vals = self.normalized_df[sector_tickers].values.T
            ssd_matrix = ssd.pdist(norm_vals, metric='sqeuclidean')
            # 協方差矩陣用於 OLS Beta 估計（外層 i=X=Ticker_B，內層 j=Y=Ticker_A）
            cov_matrix = np.cov(norm_vals)
            var_diag = np.diag(cov_matrix)
            idx = 0
            for i in range(len(sector_tickers)):
                ticker_b = sector_tickers[i]
                var_x = var_diag[i]
                
                for j in range(i + 1, len(sector_tickers)):
                    ticker_a = sector_tickers[j]
                    
                    ssd_value = ssd_matrix[idx]
                    idx += 1
                    
                    cov_xy = cov_matrix[i, j]
                    beta = cov_xy / var_x if var_x > 1e-8 else 0.0
                    
                    ssd_records.append({
                        "Sector": sector, "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                        "SSD": float(ssd_value), "Hedge_Ratio": float(beta),
                    })

        if not ssd_records: 
            return pd.DataFrame()

        all_pairs_df = pd.DataFrame(ssd_records).sort_values("SSD").reset_index(drop=True)
        # 先按 SSD 初篩，只對最近鄰候選對做共整合/Hurst，大幅減少慢速統計計算次數
        candidates_limit = max(200, self.top_n * 15)
        candidates = all_pairs_df.head(candidates_limit)
        
        filtered_records = []
        for _, row in candidates.iterrows():
            x_val = self.normalized_df[row["Ticker_B"]].values
            y_val = self.normalized_df[row["Ticker_A"]].values
            beta = row["Hedge_Ratio"]
            
            spread = y_val - beta * x_val
            
            # 步驟 1：ADF 共整合（過濾隨機漫步）
            stat, pval = _adf_stat(spread, max_lags=1)
            if pval >= self.adf_pvalue_threshold:
                continue

            # 步驟 2：OU 半衰期
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
            if halflife < 1.0 or halflife > self.halflife_max:
                continue

            # 步驟 3：Hurst 指數（均值回歸傾向）
            hurst = _compute_hurst(spread, already_stationary=True)
            if hurst >= 0.50:
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

        selected["Log_Mean_A"] = [self.mean_prices[t] for t in selected["Ticker_A"]]
        selected["Log_Std_A"]  = [self.std_prices[t]  for t in selected["Ticker_A"]]
        selected["Log_Mean_B"] = [self.mean_prices[t] for t in selected["Ticker_B"]]
        selected["Log_Std_B"]  = [self.std_prices[t]  for t in selected["Ticker_B"]]

        self.selected_pairs = selected
        return self.selected_pairs


    def run(self) -> pd.DataFrame:
        """執行形成期流程並回傳選定的配對"""
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs
