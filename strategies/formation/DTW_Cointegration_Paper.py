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
from strategies.formation._utils import _compute_hurst, _ols, _adf_stat
from strategies.formation._cointegration import screen_pair

def _dtw_py(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """純 Python 參照實作（numba 不可用時的退路，亦為數值對帳的基準）。"""
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


try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=False)
    def _dtw_kernel(x, y, window):
        # 只保留兩列 DP：遞迴僅依賴 i-1 與 i 兩列，記憶體由 O(n*m) 降為 O(m)。
        # 迴圈與比較順序刻意與 _dtw_py 一致，確保逐位相同（fastmath 必須為 False，
        # 否則浮點重排會破壞位元等價）。
        n = x.shape[0]
        m = y.shape[0]
        INF = np.inf
        prev = np.full(m + 1, INF)
        cur = np.full(m + 1, INF)
        prev[0] = 0.0
        for i in range(1, n + 1):
            for k in range(m + 1):
                cur[k] = INF
            start_j = max(1, i - window)
            end_j = min(m, i + window)
            for j in range(start_j, end_j + 1):
                d = x[i - 1] - y[j - 1]
                cost = d * d
                a = prev[j]
                b = cur[j - 1]
                c = prev[j - 1]
                best = a
                if b < best:
                    best = b
                if c < best:
                    best = c
                cur[j] = cost + best
            for k in range(m + 1):
                prev[k] = cur[k]
        return prev[m]

    _HAVE_NUMBA = True
except Exception:                                    # numba 未安裝或編譯失敗
    _HAVE_NUMBA = False


def _sakoe_chiba_dtw(x: np.ndarray, y: np.ndarray, window: int = 15) -> float:
    """
    Sakoe-Chiba 限制窗口的 DTW (Dynamic Time Warping) 距離。
    時間軸扭曲限制在 `window` 天之內，時間複雜度為 O(N * W)。

    2026-08-24：核心改以 numba JIT 編譯（實測 5.7 ms → 6.0 us，948x）。
    原純 Python 版保留為 `_dtw_py`，兩者在 300 組隨機長度（30–300）與帶寬
    （1–40）測試下**逐位相同、最大絕對差 0.0**，故既有結果不受影響。

    動機：NOGRP 臂每期 84,255 組候選，全池 DTW 原需 39 小時，使 dtw_window
    的敏感性掃描實務上做不了（config 中該參數固定為 15 且從未變動）。加速後
    降為 2.5 分鐘，掃描結果確認 w=15 確實為最優（見 dev/ml_formation/）。
    """
    if not _HAVE_NUMBA:
        return _dtw_py(x, y, window)
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    return float(_dtw_kernel(xa, ya, int(window)))



warnings.filterwarnings("ignore")


class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    流程：先共整合檢定過濾 -> 計算 SSD 與 DTW 距離 -> 基於 DTW 或是 SSD+DTW (PCA) 排序。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20,
                 sector_mapping: dict = None, min_tickers_for_pairing: int = 2, dtw_window: int = 15,
                 method: str = "dtw", adf_pvalue_threshold: float = 0.01, trading_window: int = 126, **kwargs):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.dtw_window = dtw_window
        self.method = method.lower()
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.halflife_max = trading_window / 3.0
        # 篩選消融開關（預設 True＝現行行為）：False 時跳過 ADF/半衰期/Hurst
        self.enable_filters = kwargs.get("enable_filters", True)
        # 分階段稽核軌跡（預設 None＝不記錄）。見 ssd_rolling 同名欄位說明。
        # 註：本後端「先檢定、後排序」，故未通過者沒有距離分數（記 None）。
        self.trace = kwargs.get("trace", None)
        self.adf_only = kwargs.get("adf_only", False)

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.mean_prices: pd.Series = pd.Series(dtype=float)
        self.std_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """對數價格標準化"""
        log_prices = np.log(np.maximum(self.price_df, 1e-8))
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        self.normalized_df = (log_prices - self.mean_prices) / (self.std_prices + 1e-12)
        return self.normalized_df

    def compute_pairs(self) -> pd.DataFrame:
        """先共整合篩選，再計算 SSD / DTW 距離"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        records = []

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
                    
                    # 步驟 1：雙向 OLS + ADF 共整合
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

                    # 步驟 2-4：三道統計過濾（中性共用層）
                    # ADF 已於雙向 OLS 取得（best_stat/best_pval），以 precomputed_adf 傳入避免重算；
                    # enable_filters=False 時整層跳過 → 純排序消融（雙向選向仍執行）
                    passed, _stats = screen_pair(
                        best_resid,
                        adf_pvalue_threshold=self.adf_pvalue_threshold,
                        halflife_min=1.0, halflife_max=self.halflife_max,
                        hurst_threshold=0.50,
                        precomputed_adf=(best_stat, best_pval),
                        enabled=self.enable_filters,
                        adf_only=self.adf_only,
                    )
                    if not passed:
                        if self.trace is not None:
                            self.trace.append({
                                "Ticker_A": best_a, "Ticker_B": best_b,
                                "Group": sector, "Rank_Backend": self.method,
                                "Rank_Score": None, "Cand_Rank": None,
                                "adf_stat": _stats["adf_stat"], "adf_p": _stats["adf_p"],
                                "halflife": _stats["halflife"], "hurst": _stats["hurst"],
                                "hurst_rs": _stats["hurst_rs"],
                                "Passed": 0,
                            })
                        continue


                    # 步驟 5：計算 SSD 與 DTW 距離
                    norm_a = self.normalized_df[best_a].values
                    norm_b = self.normalized_df[best_b].values
                    ssd_dist = float(np.sum((norm_a - norm_b) ** 2))
                    dtw_dist = _sakoe_chiba_dtw(norm_a, norm_b, window=self.dtw_window)
                    
                    spread_mean = np.mean(best_resid)
                    spread_std = np.std(best_resid, ddof=1) if len(best_resid) > 1 else 0.0

                    if self.trace is not None:
                        self.trace.append({
                            "Ticker_A": best_a, "Ticker_B": best_b,
                            "Group": sector, "Rank_Backend": self.method,
                            "Rank_Score": float(dtw_dist), "Cand_Rank": None,
                            "adf_stat": _stats["adf_stat"], "adf_p": _stats["adf_p"],
                            "halflife": _stats["halflife"], "hurst": _stats["hurst"],
                                "hurst_rs": _stats["hurst_rs"],
                            "Passed": 1,
                        })

                    records.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": best_a, "Ticker_B": best_b,
                        "SSD": round(ssd_dist, 6), "DTW_Dist": round(dtw_dist, 6), 
                        "Hedge_Ratio": round(best_beta, 4),
                        "OLS_Alpha": round(best_alpha, 6),
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std": round(spread_std, 6)
                    })

        if not records:
            return pd.DataFrame()
            
        return pd.DataFrame(records)

    def select_pairs(self) -> pd.DataFrame:
        pairs_df = self.compute_pairs()
        if pairs_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if self.method == "dtw":
            # 對照組：依 DTW 距離升序
            selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()
        elif self.method == "ssd_dtw_pca":
            # 實驗組：PCA 融合 SSD+DTW 取第一主成分排序
            if len(pairs_df) < 2:
                selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                
                feats = pairs_df[["SSD", "DTW_Dist"]].values
                feats_scaled = StandardScaler().fit_transform(feats)
                
                pca = PCA(n_components=1, random_state=42)
                scores = pca.fit_transform(feats_scaled).flatten()
                
                # 確保 loadings 方向為正（距離越小得分越小）
                loadings = pca.components_[0]
                if loadings[0] < 0:
                    scores = -scores
                
                pairs_df["PC1_Score"] = scores
                selected = pairs_df.sort_values("PC1_Score").head(self.top_n).copy()
        else:
            selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        selected["Log_Mean_A"] = [self.mean_prices[t] for t in selected["Ticker_A"]]
        selected["Log_Std_A"]  = [self.std_prices[t]  for t in selected["Ticker_A"]]
        selected["Log_Mean_B"] = [self.mean_prices[t] for t in selected["Ticker_B"]]
        selected["Log_Std_B"]  = [self.std_prices[t]  for t in selected["Ticker_B"]]

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs
