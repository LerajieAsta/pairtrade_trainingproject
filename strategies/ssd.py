

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
    透過計算正規化對數價格的 SSD 來尋找走勢相近的股票對。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, sector_mapping: dict = None, min_tickers_for_pairing: int = 2):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.mean_prices: pd.Series = pd.Series(dtype=float)
        self.std_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """將價格轉換為對數價格，並進行 Z-Score 正規化"""
        log_prices = np.log(self.price_df)
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        self.normalized_df = (log_prices - self.mean_prices) / self.std_prices
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
            
            # 建立相關係數與變異數矩陣以計算 Beta
            cov_matrix = np.cov(norm_vals)
            var_diag = np.diag(cov_matrix)
            
            # 將 pdist 的 1D 陣列轉換為 Pair 組合
            # 註：此處外層迴圈 i (x_val) 視為自變數 X (Ticker_B)，內層迴圈 j (y_val) 視為因變數 Y (Ticker_A)
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

        if skipped_unknown_count > 0:
            print(f"  [Formation] 跳過 {skipped_unknown_count} 支未分類 (Unknown) 股票。")

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
        """執行形成期流程並回傳選定的配對"""
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs

# ══════════════════════════════════════════════════════════════════════════════
# Class 2：Trading（交易期模組）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class PairState:
    """單一配對在模擬過程中的內部狀態 (使用 slots 降低記憶體開銷)"""
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
    """負責在交易期 (Trading Period) 模擬配對交易並產生詳細交易紀錄"""
    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, selected_pairs: pd.DataFrame, capital_per_pair: float, 
                 fee_rate: float, slippage_rate: float, stop_loss_pct: float, entry_z: float, exit_z: float, zscore_window: int, allow_reentry: bool = False,
                 zscore_clip: float = 10.0, min_spread_std: float = 1e-6, use_dynamic_stop: bool = False, dynamic_stop_z: float = 3.0,
                 portfolio_stop_loss_pct: float = 0.10, use_vol_adjust: bool = False):
        self.price_df = price_df.copy()
        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair
        
        self.friction_rate = fee_rate + slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.zscore_window = zscore_window
        self.allow_reentry = allow_reentry  
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust = use_vol_adjust

        self.period_pnl: float = 0.0

    def _execute_entry(self, state: PairState, z: float, p_a: float, p_b: float, hedge_ratio: float) -> tuple[bool, float]:
        """處理進場邏輯與資金分配"""
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)
        
        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b = v_b / p_b
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a = v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state: PairState, current_trade_pnl: float, stop_loss: bool = False):
        """處理平倉與停損邏輯"""
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

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float, 
                       form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float) -> pd.DataFrame:
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: return pd.DataFrame()

        log_p_a = np.log(price_a)
        log_p_b = np.log(price_b)

        norm_p_a = (log_p_a - log_mean_a) / log_std_a
        norm_p_b = (log_p_b - log_mean_b) / log_std_b
        
        if self.zscore_window == 0:
            spread = norm_p_a - hedge_ratio * norm_p_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series = pd.Series(hedge_ratio, index=common_idx)
        else:
            roll_cov = norm_p_b.rolling(window=self.zscore_window).cov(norm_p_a)
            roll_var = norm_p_b.rolling(window=self.zscore_window).var()
            
            roll_beta = np.where(roll_var > 1e-8, roll_cov / roll_var, 0.0)
            roll_beta = pd.Series(roll_beta, index=common_idx)
            
            roll_mean_a = norm_p_a.rolling(window=self.zscore_window).mean()
            roll_mean_b = norm_p_b.rolling(window=self.zscore_window).mean()
            roll_alpha = roll_mean_a - roll_beta * roll_mean_b
            
            spread = norm_p_a - roll_alpha - roll_beta * norm_p_b
            
            roll_var_a = norm_p_a.rolling(window=self.zscore_window).var()
            roll_res_var = roll_var_a - roll_beta * roll_cov
            roll_std = np.sqrt(np.maximum(roll_res_var, 0))
            
            if (roll_std < self.min_spread_std * 10).mean() > 0.5:
                return pd.DataFrame()
            
            safe_std = np.maximum(roll_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore = np.clip(spread / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series = roll_beta

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0: return pd.DataFrame()
        
        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        zscore = zscore.loc[valid_idx]
        beta_series = beta_series.loc[valid_idx]

        dates_arr = valid_idx
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values
        beta_arr = beta_series.values

        base_log = {
            "Period_Start": period_start, "Period_End": period_end,
            "Sector": sector, "Pair_Rank": pair_rank,
            "Ticker_A": ticker_a, "Ticker_B": ticker_b,
            "Log_Mean_A": log_mean_a, "Log_Std_A": log_std_a,
            "Log_Mean_B": log_mean_b, "Log_Std_B": log_std_b
        }

        state = PairState()
        
        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_z, out_pos = [], [], []
        out_unrealized, out_realized, out_cum = [], [], []
        out_status, out_trade_pnl, out_days, out_delta = [], [], [], []

        for i in range(len(dates_arr)):
            date = dates_arr[i]
            z = 0.0 if np.isnan(zscore_arr[i]) else zscore_arr[i]
            p_a, p_b = pa_arr[i], pb_arr[i]
            
            c_beta = beta_arr[i] if not np.isnan(beta_arr[i]) else hedge_ratio

            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0 
            daily_delta = 0.0
            current_status = "HOLD_CASH"

            if state.is_stopped:
                out_dates.append(date)
                out_pa.append(round(p_a, 4))
                out_pb.append(round(p_b, 4))
                out_hr.append(round(float(c_beta), 4))
                out_z.append(round(float(z), 4))
                out_pos.append(0)
                out_unrealized.append(0.0)
                out_realized.append(round(float(state.realized_pnl), 4))
                out_cum.append(round(float(state.realized_pnl), 4))
                out_status.append("STOPPED")
                out_trade_pnl.append(0.0)
                out_days.append(0)
                out_delta.append(0.0)
                continue

            if state.cooldown_dir == -1 and z <= self.exit_z:
                state.cooldown_dir = 0
            elif state.cooldown_dir == 1 and z >= -self.exit_z:
                state.cooldown_dir = 0

            if state.position != 0:
                state.days_held += 1
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                
                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est
                
                is_cap_stop = self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED"
                else:
                    is_exit_short = (state.position == -1) and (z <= self.exit_z)  
                    is_exit_long  = (state.position == 1)  and (z >= -self.exit_z)  
                    
                    if is_exit_short or is_exit_long:
                        self._execute_close(state, current_trade_pnl, stop_loss=False)
                        closed_trade_pnl = current_trade_pnl
                        current_status = "EXIT"
                    else:
                        unrealized_pnl = current_trade_pnl 
                        current_status = "HOLDING"
            else: 
                if abs(z) > self.entry_z:
                    entered, unrealized_pnl = self._execute_entry(state, z, p_a, p_b, c_beta)
                    if entered:
                        current_status = "ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A"
                    else:
                        current_status = "HOLD_CASH (COOLDOWN)"
                else:
                    current_status = "HOLD_CASH"

            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl
            
            out_dates.append(date)
            out_pa.append(round(p_a, 4))
            out_pb.append(round(p_b, 4))
            out_hr.append(round(float(c_beta), 4))
            out_z.append(round(float(z), 4))
            out_pos.append(state.position)
            out_unrealized.append(round(float(unrealized_pnl), 4))
            out_realized.append(round(float(state.realized_pnl), 4))
            out_cum.append(round(float(cumulative_pnl), 4))
            out_status.append(current_status)
            out_trade_pnl.append(round(float(closed_trade_pnl), 4))
            out_days.append(state.days_held)
            out_delta.append(round(float(daily_delta), 4))

            if current_status in ["STOP_LOSS_TRIGGERED", "EXIT"]:
                state.days_held = 0 

            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    rd = dates_arr[j]
                    r_z = 0.0 if np.isnan(zscore_arr[j]) else zscore_arr[j]
                    r_pa, r_pb = pa_arr[j], pb_arr[j]
                    r_beta = beta_arr[j] if not np.isnan(beta_arr[j]) else hedge_ratio
                    
                    out_dates.append(rd)
                    out_pa.append(round(r_pa, 4))
                    out_pb.append(round(r_pb, 4))
                    out_hr.append(round(float(r_beta), 4))
                    out_z.append(round(float(r_z), 4))
                    out_pos.append(0)
                    out_unrealized.append(0.0)
                    out_realized.append(round(float(state.realized_pnl), 4))
                    out_cum.append(round(float(state.realized_pnl), 4))
                    out_status.append("STOPPED")
                    out_trade_pnl.append(0.0)
                    out_days.append(0)
                    out_delta.append(0.0)
                break 

        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                pnl_before_last_day = out_cum[-2] if len(out_cum) > 1 else 0.0
                
                p_a_last, p_b_last = pa_arr[-1], pb_arr[-1]
                raw_unrealized_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                exit_fee = (abs(state.shares_a)*p_a_last + abs(state.shares_b)*p_b_last) * self.friction_rate
                
                closed_trade_pnl = raw_unrealized_final - state.trade_entry_fee - exit_fee
                state.realized_pnl += closed_trade_pnl
                daily_delta = state.realized_pnl - pnl_before_last_day 
                
                out_status[-1] = "PERIOD_END_EXIT"
                out_realized[-1] = round(state.realized_pnl, 4)
                out_cum[-1] = round(state.realized_pnl, 4)
                out_unrealized[-1] = 0.0
                out_trade_pnl[-1] = round(closed_trade_pnl, 4)
                out_delta[-1] = round(daily_delta, 4)
                out_days[-1] = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb, 
            "Hedge_Ratio": out_hr, "ZScore": out_z, "Position": out_pos, 
            "Unrealized_PnL": out_unrealized, "Realized_PnL": out_realized, 
            "Cumulative_PnL": out_cum, "Status": out_status, 
            "Trade_PnL": out_trade_pnl, "Days_Held": out_days, "Daily_Delta": out_delta
        })
        
        for k, v in base_log.items():
            df_out[k] = v
            
        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """執行該期所有配對的交易模擬"""
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start,
                period_end=period_end,
                sector=row.get("Sector", "Unknown"), 
                ticker_a=row["Ticker_A"], 
                ticker_b=row["Ticker_B"], 
                pair_rank=row["Rank"], 
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)), 
                form_spread_mean=float(row.get("Spread_Mean", 0.0)), 
                form_spread_std=float(row.get("Spread_Std", 1.0)), 
                log_mean_a=float(row.get("Log_Mean_A", 0.0)), 
                log_std_a=float(row.get("Log_Std_A", 1.0)), 
                log_mean_b=float(row.get("Log_Mean_B", 0.0)), 
                log_std_b=float(row.get("Log_Std_B", 1.0))
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
        period_daily_delta = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily_delta.sum()) if not period_daily_delta.empty else 0.0
        
        return log_df, self.period_pnl

# ══════════════════════════════════════════════════════════════════════════════
# Class 3：DataProcessor（數據清理與前置處理模組）
# ══════════════════════════════════════════════════════════════════════════════
class DataProcessor:
    """處理歷史價格載入、產業對應、以及清洗缺失值"""
    def __init__(self, db_path: str, table_name: str = "daily_prices"):
        self.db_path, self.table_name = db_path, table_name

    def load_sector_mapping(self, info_table: str, ticker_col: str = "ticker", sector_col: str = "sector") -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT {ticker_col}, {sector_col} FROM {info_table}", conn)
            conn.close()
            mapping = {}
            for k, v in zip(df[ticker_col], df[sector_col]):
                if pd.notna(k) and pd.notna(v):
                    mapping[str(k).strip().upper()] = str(v).strip()
            print(f"✅ 成功載入產業分類表 '{info_table}'，共取得 {len(mapping)} 檔標的分類。")
            return mapping
        except Exception as e: 
            print(f"⚠️ [警告] 無法載入產業分類表 '{info_table}'！錯誤原因：{e}")
            print(f"⚠️ 系統將退回「全市場(All_Market)」跨產業配對模式。")
            return {}

    def prepare_backtest_data(self, backtest_start: str, backtest_end: str, formation_window: int):
        conn = sqlite3.connect(self.db_path)
        # 支援 Close 或 Adj_Close
        raw_df = pd.read_sql_query(f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC", conn)
        conn.close()

        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]
        
        # 建立 Pivot 價格矩陣
        price_pivot = raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()
        # A. 移除遺失值超過 20% 的標的（欄位）
        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20]
        # B. 向前填補最多 5 天（避免長缺口被不合理填補）
        price_pivot = price_pivot.ffill(limit=5)
        # C. 移除同日遺失值超過 10% 的日期（列）
        price_pivot = price_pivot.loc[price_pivot.isnull().mean(axis=1) <= 0.10]
        # D. 再次移除遺失值過多的標的（以目前長度的 90% 為門檻）
        price_pivot.dropna(axis=1, thresh=int(len(price_pivot) * 0.9), inplace=True)
        
        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                if is_end and len(str(d_str).strip()) == 7:
                    return dt + pd.offsets.MonthEnd(0)
                return dt
            except Exception:
                return None

        bt_start_ts = _safe_parse(backtest_start)
        bt_end_ts = _safe_parse(backtest_end, is_end=True)
        all_dates = price_pivot.index.tolist()

        start_indices = [i for i, d in enumerate(all_dates) if d >= bt_start_ts] if bt_start_ts else []
        first_idx = start_indices[0] if start_indices else 0
        
        # 確保回推足夠的 Formation Window
        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end = bt_end_ts if bt_end_ts else price_pivot.index[-1]
        price_pivot = price_pivot.loc[data_slice_start:data_slice_end]

        sliced_dates = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_trade_idx = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_trade_idx, formation_window)


# ══════════════════════════════════════════════════════════════════════════════
# Class 4：RollingBacktester（滾動回測引擎）
# ══════════════════════════════════════════════════════════════════════════════
class RollingBacktester:
    """負責處理滾動視窗、參數網格搜尋以及交易排程的主引擎"""
    def __init__(self, top_n_list: list, stop_loss_list: list, zscore_window_list: list,
                 entry_z: float, exit_z: float, formation_window: int, trading_window: int, rolling_step: int,
                 fee_rate: float, slippage_rate: float, initial_capital: float,
                 allow_reentry: bool, zscore_clip: float, min_spread_std: float,
                 min_tickers_for_pairing: int, output_dir: Path,
                 portfolio_stop_loss_pct_list: list = None,
                 max_sector_ratio_list: list = None,
                 dynamic_stop_z_list: list = None,
                 use_vol_adjust_list: list = None,
                 **kwargs):
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
        self.output_dir = output_dir

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

        for k, v in kwargs.items():
            setattr(self, k, v)

    def run(self, price_pivot: pd.DataFrame, all_dates: list, total_days: int, local_first_trade_idx: int, sector_mapping: dict):
        """執行網格搜索與滾動回測"""
        max_concurrent = self.trading_window // self.rolling_step
        states = {}

        for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
            self.top_n_list, self.stop_loss_list, self.zscore_window_list,
            self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
        ):
            states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)] = {
                "logs": [], 
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent} for _ in range(max_concurrent)]
            }

        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始進行 Grid Search，共 {len(roll_start_indices)} 期，每期處理 {len(states)} 種參數組合...")

        # 執行滾動回測主迴圈
        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx, form_end_idx = trade_start_idx - self.formation_window, trade_start_idx
            trade_end_idx = min(trade_start_idx + self.trading_window, total_days)

            form_data_raw = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data_raw = price_pivot.iloc[trade_start_idx:trade_end_idx]
            
            # 準備包含 Z-Score 計算所需的延伸歷史資料
            extended_trade_start_idx = max(0, trade_start_idx - max(self.zscore_window_list))
            extended_trade_data_raw = price_pivot.iloc[extended_trade_start_idx:trade_end_idx]
            
            # 過濾在形成期與延伸交易期存在 NaN 的標的，確保滾動計算不會遇到 NaN
            valid_cols = (form_data_raw.isnull().sum() + extended_trade_data_raw.isnull().sum()) == 0
            
            form_data = form_data_raw.loc[:, valid_cols]
            trade_data = trade_data_raw.loc[:, valid_cols]
            trade_dates = trade_data.index
            
            extended_trade_data = extended_trade_data_raw.loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty: continue

            trade_start_str, trade_end_str = str(all_dates[trade_start_idx])[:10], str(all_dates[trade_end_idx - 1])[:10]
            form_start_str, form_end_str = str(all_dates[form_start_idx])[:10], str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 處理中：第 {roll_idx+1:02d} 期 (交易: {trade_start_str} ~ {trade_end_str})")

            # 進行配對篩選 (top_n 設為安全上限以供網格過濾)
            formation = Formation(
                price_df=form_data, 
                form_start=form_start_str, 
                form_end=form_end_str, 
                top_n=max(self.top_n_list) * 5, 
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty: continue

            # 對所有參數組合進行交易模擬
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
                state = states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)]
                slots = state["slots"]
                
                # 分配可用資金槽
                free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                if free_slots:
                    slot_idx = free_slots[0]
                else:
                    slot_idx = min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                current_period_capital = slots[slot_idx]["capital"]
                current_capital_per_pair = current_period_capital / n

                trading = Trading(
                    price_df=extended_trade_data,
                    trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=current_capital_per_pair,
                    fee_rate=self.fee_rate,
                    slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl,
                    entry_z=self.entry_z,
                    exit_z=self.exit_z,
                    zscore_window=z_win,
                    allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip,
                    min_spread_std=self.min_spread_std,
                    use_dynamic_stop=(dyn_z > 0),
                    dynamic_stop_z=dyn_z,
                    portfolio_stop_loss_pct=p_stop,
                    use_vol_adjust=vol_adj
                )
                
                trade_log_df, period_pnl = trading.run(trade_start_str, trade_end_str)
                
                if not trade_log_df.empty:
                    state["logs"].append(trade_log_df)
                    
                slots[slot_idx]["capital"] = max(0, current_period_capital + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        # 迴圈結束後呼叫匯出功能
        self._export_results(states)

    def _export_results(self, states: dict):
        """將每種參數組合的紀錄匯出為獨立 CSV"""
        print("\n✅ 回測完成！正在匯出交易紀錄檔案...")
        for (n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj), state in states.items():
            if state["logs"]:
                full_log_df = pd.concat(state["logs"], ignore_index=True)
                sl_str = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                psl_str = f"PSL{int(p_stop*100)}" if p_stop > 0 else "PSL0"
                msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
                dsz_str = f"DSZ{int(dyn_z)}" if dyn_z > 0 else "DSZ0"
                vol_str = "VolAdj" if vol_adj else "NoVol"
                filename = f"TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
                filepath = self.output_dir / filename
                full_log_df.to_csv(filepath, index=False)
                print(f"  - 已輸出: {filename} (共 {len(full_log_df)} 筆紀錄)")
                
        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 標準化策略進入點接口 (Unified Strategy Entry Point)
# ══════════════════════════════════════════════════════════════════════════════
#
# 注意：本檔案下方仍保留唯一的 `run_strategy` 實作；此處刪除重複定義以避免覆蓋/混淆。

# ══════════════════════════════════════════════════════════════════════════════
# 主程式：自動參數網格搜尋 
# ══════════════════════════════════════════════════════════════════════════════


# ======================================================================
# 原 Main 區塊被自動化註釋解耦如下：
# if __name__ == "__main__":

#     # --- 1. 基本參數與路徑設定 ---
#     MIN_TICKERS_FOR_PAIRING = 2  
#     ZSCORE_CLIP = 10.0           
#     MIN_SPREAD_STD = 1e-6

#     # 確保您有此相對路徑下的 db 檔案
#     DB_PATH, TABLE_NAME = r"../data/sp500_Current.db", "Daily_Prices"
#     BACKTEST_START, BACKTEST_END = "2000-01", "2025-12"
#     INFO_TABLE_NAME, TICKER_COL_NAME, SECTOR_COL_NAME = "Constituents", "Symbol", "GICS_Sector"

#     ALLOW_REENTRY = False            # 是否允許再進場
#     OUTPUT_DIR = Path(r"../results/current/SSD_NoReEntry")
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     # --- 2. 網格搜尋參數設定 ---
#     TOP_N_LIST = [1, 5, 20]
#     STOP_LOSS_LIST = [0, 0.05, 0.15]      # 0 表示不停損
#     ZSCORE_WINDOW_LIST = [0, 20, 60]      # 0 表示固定 Z 值

#     ENTRY_Z, EXIT_Z = 2.0, 0.0
#     FORMATION_WINDOW, TRADING_WINDOW, ROLLING_STEP = 252, 126, 21

#     # --- 3. 交易成本與環境設定 ---
#     FEE_RATE = 0.001
#     SLIPPAGE_RATE = 0.001 
#     INITIAL_CAPITAL = 10000 
#     USE_SECTOR_PAIRING = True       # 是否使用產業分類

#     # --- 4. 資料前處理 ---
#     processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)

#     if USE_SECTOR_PAIRING:
#         sector_mapping = processor.load_sector_mapping(INFO_TABLE_NAME, TICKER_COL_NAME, SECTOR_COL_NAME)
#     else:
#         sector_mapping = {}
#         print("ℹ️ 產業分類配對已關閉 (USE_SECTOR_PAIRING = False)，系統將進行全市場全標的配對。")

#     # 執行資料載入與格式化
#     try:
#         price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)

#         # --- 5. 啟動回測引擎 ---
#         # 將所有參數送入新建立的 RollingBacktester，並呼叫 run()
#         engine = RollingBacktester(
#             top_n_list=TOP_N_LIST, stop_loss_list=STOP_LOSS_LIST, zscore_window_list=ZSCORE_WINDOW_LIST,
#             entry_z=ENTRY_Z, exit_z=EXIT_Z, formation_window=FORMATION_WINDOW,
#             trading_window=TRADING_WINDOW, rolling_step=ROLLING_STEP, fee_rate=FEE_RATE,
#             slippage_rate=SLIPPAGE_RATE, initial_capital=INITIAL_CAPITAL, allow_reentry=ALLOW_REENTRY,
#             zscore_clip=ZSCORE_CLIP, min_spread_std=MIN_SPREAD_STD,
#             min_tickers_for_pairing=MIN_TICKERS_FOR_PAIRING, output_dir=OUTPUT_DIR
#         )

#         engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)

#     except Exception as e:
#         print(f"\n❌ 執行發生錯誤：{e}")
#         print("💡 提示: 請確認資料庫 `../data/sp500.db` 存在，且資料表格式符合預期。")


# ══════════════════════════════════════════════════════════════════════════════
# 標準化策略進入點接口 (Unified Strategy Entry Point)
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    標準化調用接口，接受外部傳入的價格資料與回測參數，完全解耦資料載入 I/O
    """
    import inspect
    from pathlib import Path
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取 RollingBacktester 的 __init__ 參數列表
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}
    
    # 動態將外部 params 對應並過濾為該策略 RollingBacktester 支援的參數
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    
    # 初始化回測引擎
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    # 執行回測
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
