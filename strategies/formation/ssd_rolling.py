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
from strategies.formation._utils import (_compute_hurst, _ols, _adf_stat,
                                         _adf_stat_batch)
from strategies.formation._cointegration import screen_pair


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
        # 篩選消融開關（預設 True＝現行行為）：False 時跳過 ADF/半衰期/Hurst
        self.enable_filters = kwargs.get("enable_filters", True)
        # 分階段稽核軌跡（預設 None＝不記錄，行為與既往完全相同）。
        # 給一個 list 時，每個被評估的候選對會附上排序分數與三道檢定的統計量，
        # 供 run_formation 落地成 formation_ranked / formation_filtered。
        # 這些統計量本來就會算，只是過去被丟棄；記錄不增加任何計算。
        self.trace = kwargs.get("trace", None)
        # filter_mode="adf_only"：只做 ADF（復現許鈞翔 2025 的篩選層）
        self.adf_only = kwargs.get("adf_only", False)
        # 候選池上限：None＝不截斷，對群內全部配對施加篩選後才排序（現行預設）。
        #
        # 舊行為為 max(200, top_n*15)：先全域按 SSD 排序、只對最近的那些候選做統計
        # 檢定。該截斷是**全域**而非逐群，故 SSD 分布整體偏大的群可能一組配對都
        # 輪不到被檢定——此即「排序先於篩選」，與四層架構所宣告的順序不一致，
        # 且使截斷與分組方法產生交互作用。
        #
        # 保留此參數僅為可重現舊結果；正式設定一律 None。
        self.candidate_limit = kwargs.get("candidate_limit", None)

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
        # 四層架構的順序：分組 → 篩選 → 排序。
        # 群內**全部**配對都要經過三道統計檢定，通過者才進入距離排序。
        # 此處先依 SSD 排序僅為使 trace 的 Cand_Rank 有意義、並讓後續 head(top_n)
        # 免去再排序；沒有任何配對因排序而被排除在檢定之外。
        candidates = (all_pairs_df if self.candidate_limit is None
                      else all_pairs_df.head(int(self.candidate_limit)))

        # ── ADF 分塊批次預算 ────────────────────────────────────────────
        # 瓶頸實測：NOGRP 臂單一視窗 81,003 個候選需 89 秒，其中約 91% 花在
        # 逐一呼叫 statsmodels.adfuller。改以 _adf_stat_batch 一次算一整塊，
        # 再經 screen_pair 既有的 precomputed_adf 參數傳入——控制流與判定邏輯
        # 完全不動，只換「算 ADF」這一步。
        #
        # 為何要分塊而非全批：下方的提前中止（湊滿 top_n 即停）是精確等價的
        # 最佳化，全批會把它作廢、反而把 NOGRP 的 8 萬個候選全算一遍。塊大小
        # 取 top_n×40，實測通過率 2–7%，故通常第一塊就湊滿。
        _cand_n = len(candidates)
        _tickA = candidates["Ticker_A"].tolist()
        _tickB = candidates["Ticker_B"].tolist()
        _betas = candidates["Hedge_Ratio"].to_numpy()
        _chunk = max(200, int(self.top_n) * 40)
        _adf_cache: dict[int, tuple] = {}

        def _adf_for(pos: int, spread_vec: np.ndarray) -> tuple:
            """取第 pos 個候選的 (stat, pval)；未算過則批次補算該塊。"""
            if pos not in _adf_cache:
                lo = (pos // _chunk) * _chunk
                hi = min(lo + _chunk, _cand_n)
                A = self.normalized_df[_tickA[lo:hi]].to_numpy().T
                B = self.normalized_df[_tickB[lo:hi]].to_numpy().T
                Sp = A - _betas[lo:hi, None] * B
                st, pv = _adf_stat_batch(Sp, max_lags=1)
                for j in range(hi - lo):
                    _adf_cache[lo + j] = (float(st[j]), float(pv[j]))
            return _adf_cache[pos]

        filtered_records = []
        for _cand_rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            x_val = self.normalized_df[row["Ticker_B"]].values
            y_val = self.normalized_df[row["Ticker_A"]].values
            beta = row["Hedge_Ratio"]

            spread = y_val - beta * x_val

            # 三道統計過濾（中性共用層；enable_filters=False 時整層跳過 → 純排序消融）
            passed, _stats = screen_pair(
                spread,
                adf_pvalue_threshold=self.adf_pvalue_threshold,
                halflife_min=1.0, halflife_max=self.halflife_max,
                hurst_threshold=0.50, adf_max_lags=1,
                precomputed_adf=(_adf_for(_cand_rank - 1, spread)
                                 if self.enable_filters else None),
                enabled=self.enable_filters,
                adf_only=self.adf_only,
            )
            if self.trace is not None:
                self.trace.append({
                    "Ticker_A": row["Ticker_A"], "Ticker_B": row["Ticker_B"],
                    "Group": row["Sector"], "Rank_Backend": "ssd",
                    "Rank_Score": float(row["SSD"]), "Cand_Rank": _cand_rank,
                    "adf_stat": _stats["adf_stat"], "adf_p": _stats["adf_p"],
                    "halflife": _stats["halflife"], "hurst": _stats["hurst"],
                    "hurst_rs": _stats["hurst_rs"],
                    "Passed": int(passed),
                })
            if not passed:
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
            
            # 精確等價的提前中止（非近似）：
            #
            # 最終選取為「通過者中 SSD 最小的前 top_n 組」。候選依 SSD 遞增順序
            # 處理，故一旦湊滿 top_n 個通過者，尚未檢定者的 SSD 皆大於已收集者，
            # 不可能擠進前 top_n。此時停止與掃完群內全部配對**輸出完全相同**。
            #
            # 與舊實作的差別在於：舊版另有 max(200, top_n*15) 的**候選截斷**，
            # 會在湊滿 top_n 之前就被迫停手，導致選不滿名額（實測 11.5/20）。
            # 移除截斷後，掃描長度由「湊滿 top_n 所需」決定，而非固定上限。
            if len(filtered_records) >= self.top_n:
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
