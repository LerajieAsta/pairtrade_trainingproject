"""
SSD Basic 形成期模組 — Gatev et al. (2006) 基準模型復刻版
======================================================================
以 Zhu (2024) "Examining Pairs Trading Profitability"（Gatev 距離法的
現代複製研究）記載的基準規格為原型：

  1. 價格正規化為累積總回報指數（首日 = 1.0，即「首日投入 $1 的價值」）。
  2. 距離 D_ij = (1/T) * Σ_t (P_i,t − P_j,t)^2（全市場 N(N−1)/2 對兩兩計算，
     不限制同產業）。
  3. 依距離升序直接取前 top_n 對——不做 ADF / 半衰期 / Hurst 等任何
     額外統計過濾（原論文模型沒有這些步驟）。
  4. 對沖比例固定 β = 1（等金額對沖，無回歸估計）。
  5. 形成期價差標準差 s_ij（交易期 2 標準差開倉門檻的基準）。

原論文中的 wait-one-day 開倉規則與「價差符號翻轉」平倉規則屬交易期
行為，由共用交易引擎 zscore_trading 以 |Z|>2 進場 / Z 穿越 0 出場近似，
差異記載於 notebooks/formation/ssd_basic.ipynb「與原論文的已知差異」。

2026-07-06 重寫：移除先前版本自加的三道統計過濾與同產業限制，
回歸論文原始規格（使用者指示「盡可能復刻其論文模型」）。
"""

import warnings

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd

warnings.filterwarnings("ignore")


class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    透過計算累積總回報指數的 SSD（均方差距離）尋找走勢相近的股票對。
    全市場搜尋、純距離排序、β=1——完全對應 Gatev et al. (2006) 基準模型。
    """

    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str,
                 top_n: int = 20, sector_mapping: dict = None,
                 min_tickers_for_pairing: int = 2, **kwargs):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        # 原論文不限制產業；sector_mapping 僅用於結果記錄（Sector_A/B 欄位）
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.first_day_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def _lookup_sector(self, ticker: str) -> str:
        return self.sector_mapping.get(
            ticker.upper(), self.sector_mapping.get(ticker, "Unknown"))

    def normalize_prices(self) -> pd.DataFrame:
        """價格正規化為累積總回報指數（首日 = 1.0）"""
        self.first_day_prices = self.price_df.iloc[0]
        safe_first = np.where(self.first_day_prices > 1e-8, self.first_day_prices, 1.0)
        self.normalized_df = self.price_df / safe_first
        return self.normalized_df

    def compute_ssd(self) -> pd.DataFrame:
        """全市場兩兩計算均方差距離 D_ij = (1/T) Σ (P_i − P_j)^2，升序排列"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        if len(tickers) < self.min_tickers_for_pairing:
            return pd.DataFrame()

        norm_vals = self.normalized_df[tickers].values.T   # (N, T)
        T = norm_vals.shape[1]
        # sqeuclidean = Σ(P_i − P_j)^2；除以 T 即 Zhu (2024) 的 D_ij 定義
        # （常數縮放不影響排序，保留以對齊論文公式）
        dist_matrix = ssd.pdist(norm_vals, metric="sqeuclidean") / T

        records = []
        idx = 0
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                records.append({
                    "Ticker_A": tickers[j], "Ticker_B": tickers[i],
                    "SSD": float(dist_matrix[idx]),
                })
                idx += 1

        all_pairs = pd.DataFrame(records).sort_values("SSD").reset_index(drop=True)

        # 純距離排序取前 top_n——無任何統計過濾（對應原論文模型）
        selected = all_pairs.head(self.top_n).copy()

        out = []
        for _, row in selected.iterrows():
            a, b = row["Ticker_A"], row["Ticker_B"]
            spread = self.normalized_df[a].values - self.normalized_df[b].values
            spread_mean = float(np.mean(spread))
            # Zhu (2024) eq.(1)：形成期價差的樣本變異（population，1/T）
            spread_std = float(np.std(spread, ddof=0))
            sec_a, sec_b = self._lookup_sector(a), self._lookup_sector(b)
            out.append({
                "Form_Start": self.form_start, "Form_End": self.form_end,
                "Sector": sec_a if sec_a == sec_b else "CrossSector",
                "Ticker_A": a, "Ticker_B": b,
                "SSD": round(row["SSD"], 6),
                "Hedge_Ratio": 1.0,                    # 等金額對沖（β ≡ 1）
                "Spread_Mean": round(spread_mean, 6),
                "Spread_Std": round(spread_std, 6),
                "Sector_A": sec_a, "Sector_B": sec_b,
            })

        return pd.DataFrame(out)

    def select_pairs(self) -> pd.DataFrame:
        """選出距離最小的前 N 組配對並補上交易期重建所需欄位"""
        selected = self.compute_ssd()
        if selected.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        selected = selected.copy()
        selected["Rank"] = range(1, len(selected) + 1)
        selected["First_Price_A"] = [self.price_df[t].iloc[0] for t in selected["Ticker_A"]]
        selected["First_Price_B"] = [self.price_df[t].iloc[0] for t in selected["Ticker_B"]]

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        """執行形成期流程並回傳選定的配對"""
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs
