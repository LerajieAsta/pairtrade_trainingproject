# ======================================================================
"""
純 DTW 配對交易滾動回測系統 (交易明細版) - 依照純DTW配對交易-[版本2].ipynb 轉譯
核心功能：
  1. 正規化價格為累積總回報指數 (首日價格設為 1.0)。
  2. 計算產業內所有可能配對的 Pure DTW 距離，完全不套用任何統計檢定過濾（如 ADF、Hurst 或半衰期）。
  3. 依 DTW 距離由小到大排序，挑選前 Top N 做為交易配對。
  4. 訊號與交易邏輯：
     - Entry：當價差從外面交叉回歸進來 (cross back inside the bands)。
     - Exit：當價差回歸到均值。
"""

import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dtaidistance import dtw
from joblib import Parallel, delayed


# 忽略不必要的 Pandas/Numpy 警告以保持輸出整潔
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 模組級輔助函式：平行化計算單一配對的 DTW 距離與形成期統計量
# ══════════════════════════════════════════════════════════════════════════════
def _compute_single_pair_dtw(ticker_a: str, ticker_b: str, norm_a: np.ndarray, norm_b: np.ndarray, sector: str) -> dict:
    """計算單一配對的 DTW 距離與價差統計量"""
    try:
        # 使用 dtaidistance 計算 DTW 距離 (會自動調用快速的 C 實作)
        dist = float(dtw.distance(norm_a.astype(np.float64), norm_b.astype(np.float64)))
    except Exception:
        dist = 999999.0
        
    spread = norm_a - norm_b
    spread_mean = float(np.mean(spread))
    # 依照 notebook，標準差使用 ddof=0
    spread_std = float(np.std(spread, ddof=0))
    
    return {
        "Sector": sector,
        "Ticker_A": ticker_a,
        "Ticker_B": ticker_b,
        "DTW_Dist": dist,
        "SSD": float(np.sum(spread ** 2)),
        "Hedge_Ratio": 1.0,  # 距離法無 Beta 估計，等同於 1.0
        "Spread_Mean": spread_mean,
        "Spread_Std": spread_std
    }


# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）
# ══════════════════════════════════════════════════════════════════════════════
class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    流程：對價格正規化 (首日價格設為 1.0) -> 計算所有配對的 Pure DTW 距離 -> 依 DTW 距離升序挑選 Top N。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, 
                 sector_mapping: dict = None, min_tickers_for_pairing: int = 2, **kwargs):
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
        """價格正規化為累積總回報指數 (首日價格設為 1.0)"""
        self.first_day_prices = self.price_df.iloc[0]
        # 完全對應 notebook 的正規化方式：
        # 1. 計算日收益率，並用 0 填補首日或缺失值
        df_ret = self.price_df.pct_change().fillna(0)
        # 2. 計算累積總回報指數 (以 1.0 為基準)
        self.normalized_df = (1 + df_ret).cumprod()
        # 3. 剔除「整個期間都沒變化」的常數序列 (nunique <= 1)
        self.normalized_df = self.normalized_df.loc[:, self.normalized_df.nunique() > 1]
        return self.normalized_df

    def compute_pairs(self) -> pd.DataFrame:
        """計算產業內所有可能配對的 Pure DTW 距離 (平行化優化)"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        
        # 根據產業分類分組
        sector_groups = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        tasks = []
        for sector, sector_tickers in sector_groups.items():
            if sector == "Unknown":
                continue
            if len(sector_tickers) < self.min_tickers_for_pairing: 
                continue

            n_sec = len(sector_tickers)
            for i in range(n_sec):
                ticker_b = sector_tickers[i]
                norm_b = self.normalized_df[ticker_b].values
                for j in range(i + 1, n_sec):
                    ticker_a = sector_tickers[j]
                    norm_a = self.normalized_df[ticker_a].values
                    
                    tasks.append((ticker_a, ticker_b, norm_a, norm_b, sector))

        if not tasks:
            return pd.DataFrame()

        # 使用 joblib 進行多進程平行計算，與 notebook 的 Parallel(n_jobs=-1) 對齊
        results = Parallel(n_jobs=-1)(
            delayed(_compute_single_pair_dtw)(ta, tb, na, nb, sec)
            for ta, tb, na, nb, sec in tasks
        )
        
        # 過濾空結果並轉為 DataFrame
        records = [r for r in results if r is not None]
        if not records:
            return pd.DataFrame()
            
        df_all = pd.DataFrame(records)
        
        # 依 DTW 距離由小到大排序 (Pure DTW)
        df_all = df_all.sort_values("DTW_Dist").reset_index(drop=True)
        return df_all

    def select_pairs(self) -> pd.DataFrame:
        """選出 DTW 距離最小的前 N 組配對"""
        pairs_df = self.compute_pairs()
        if pairs_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        selected = pairs_df.head(self.top_n).copy()
        selected["Rank"] = range(1, len(selected) + 1)

        first_price_a_list, first_price_b_list = [], []
        for _, row in selected.iterrows():
            first_price_a_list.append(self.price_df[row["Ticker_A"]].iloc[0])
            first_price_b_list.append(self.price_df[row["Ticker_B"]].iloc[0])
            
        selected["First_Price_A"] = first_price_a_list
        selected["First_Price_B"] = first_price_b_list
        selected["Form_Start"] = self.form_start
        selected["Form_End"] = self.form_end

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs


# ══════════════════════════════════════════════════════════════════════════════
