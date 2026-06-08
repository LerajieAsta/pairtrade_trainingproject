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

from strategies.ssd import PairState, RollingBacktester, Trading

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
# Class 2：Trading（交易期模組）
# ══════════════════════════════════════════════════════════════════════════════
class PureDTWTrading(Trading):
    """
    自訂交易模擬器：修改進出場邏輯以對齊純 DTW 策略 (Jupyter Notebook 邏輯)：
    1. Entry 條件：當 Z-score 從外面交叉回歸進來 (cross back inside the bands)
       - Short entry (position = -1): prev_z > entry_z and curr_z <= entry_z
       - Long entry (position = 1): prev_z < -entry_z and curr_z >= -entry_z
    2. Exit 條件：當 Z-score 回歸到均值
       - Short exit (position == -1): curr_z <= exit_z
       - Long exit (position == 1): curr_z >= -exit_z
    """
    def _execute_entry_custom(self, state: PairState, direction: int, p_a: float, p_b: float) -> tuple[bool, float]:
        """自訂進場邏輯 (等市值分配：Beta 固定為 1.0)"""
        v_a = self.capital_per_pair * 0.5
        v_b = self.capital_per_pair * 0.5
        
        if direction == -1:    # Short Entry: 賣空 A，買入 B
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b = v_b / p_b
        elif direction == 1:   # Long Entry: 買入 A，賣空 B
            state.position = 1
            state.shares_a = v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        # 計算交易手續費 + 滑價 (friction_rate)
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float, 
                       form_spread_mean: float, form_spread_std: float, **kwargs) -> pd.DataFrame:
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: 
            return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: 
            return pd.DataFrame()

        # 從 selected_pairs 提取此配對的形成期首日價格
        pair_row = self.selected_pairs[
            (self.selected_pairs["Ticker_A"] == ticker_a) & 
            (self.selected_pairs["Ticker_B"] == ticker_b)
        ]
        if not pair_row.empty:
            first_price_a = float(pair_row["First_Price_A"].iloc[0])
            first_price_b = float(pair_row["First_Price_B"].iloc[0])
        else:
            first_price_a = float(price_a.iloc[0])
            first_price_b = float(price_b.iloc[0])

        # 正規化：以形成期首日價格做為基準除數，確保連續性 (完全對應 notebook)
        norm_p_a = price_a / (first_price_a if first_price_a > 1e-8 else 1.0)
        norm_p_b = price_b / (first_price_b if first_price_b > 1e-8 else 1.0)
        
        # 價差與 Z-score
        spread = norm_p_a - norm_p_b
        safe_std = max(form_spread_std, self.min_spread_std)
        
        # 波動度調節 (若啟用)
        if getattr(self, "use_vol_adjust", False):
            roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
            vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
            adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
        else:
            adjusted_std = safe_std
            
        zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)

        # 篩選出屬於交易期的日期
        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0: 
            return pd.DataFrame()
        
        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        zscore = zscore.loc[valid_idx]

        dates_arr = valid_idx
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values

        base_log = {
            "Period_Start": period_start, "Period_End": period_end,
            "Sector": sector, "Pair_Rank": pair_rank,
            "Ticker_A": ticker_a, "Ticker_B": ticker_b,
            "First_Price_A": first_price_a, "First_Price_B": first_price_b
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
            
            # 前一日 Z-score
            prev_z = 0.0 if i == 0 or np.isnan(zscore_arr[i-1]) else zscore_arr[i-1]

            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0 
            daily_delta = 0.0
            current_status = "HOLD_CASH"

            if state.is_stopped:
                out_dates.append(date)
                out_pa.append(round(p_a, 4))
                out_pb.append(round(p_b, 4))
                out_hr.append(1.0)
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

            # 冷卻期重設
            if state.cooldown_dir == -1 and z <= self.exit_z:
                state.cooldown_dir = 0
            elif state.cooldown_dir == 1 and z >= -self.exit_z:
                state.cooldown_dir = 0

            if state.position != 0:
                state.days_held += 1
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                
                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est
                
                # 個別配對停損判定
                is_cap_stop = self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED"
                else:
                    # 出場判斷 (Z-score 回歸均值)
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
                # 進場判斷 (Z-score 從外面交叉回歸進來)
                is_entry_short = (prev_z > self.entry_z) and (z <= self.entry_z) and (state.cooldown_dir != -1)
                is_entry_long  = (prev_z < -self.entry_z) and (z >= -self.entry_z) and (state.cooldown_dir != 1)
                
                if is_entry_short:
                    entered, unrealized_pnl = self._execute_entry_custom(state, -1, p_a, p_b)
                    if entered:
                        current_status = "ENTER_SHORT_A"
                    else:
                        current_status = "HOLD_CASH"
                elif is_entry_long:
                    entered, unrealized_pnl = self._execute_entry_custom(state, 1, p_a, p_b)
                    if entered:
                        current_status = "ENTER_LONG_A"
                    else:
                        current_status = "HOLD_CASH"
                else:
                    current_status = "HOLD_CASH (COOLDOWN)" if state.cooldown_dir != 0 else "HOLD_CASH"

            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl
            
            out_dates.append(date)
            out_pa.append(round(p_a, 4))
            out_pb.append(round(p_b, 4))
            out_hr.append(1.0)
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

            # 如果已經停損且不允許再進場，則直接填補後續所有交易日並退出
            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    rd = dates_arr[j]
                    r_z = 0.0 if np.isnan(zscore_arr[j]) else zscore_arr[j]
                    r_pa, r_pb = pa_arr[j], pb_arr[j]
                    
                    out_dates.append(rd)
                    out_pa.append(round(r_pa, 4))
                    out_pb.append(round(r_pb, 4))
                    out_hr.append(1.0)
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

        # 每期結束時強制平倉
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


# ══════════════════════════════════════════════════════════════════════════════
# Class 3：RollingBacktester（滾動回測引擎）
# ══════════════════════════════════════════════════════════════════════════════
class PureDTWRollingBacktester(RollingBacktester):
    """
    滾動回測引擎 subclass，覆寫 run() 以調用 PureDTW 的 Formation 與 PureDTWTrading。
    """
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
        print(f"\n🚀 [Pure DTW] 開始 Grid Search，共 {len(roll_start_indices)} 期...")
        
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

            # 調用 Pure DTW Formation
            formation = Formation(
                price_df=form_data,
                form_start=fs_str, form_end=fe_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing,
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty:
                continue

            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                # 產業多元化過濾
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

                # 調用 PureDTWTrading 進行交易模擬
                trading = PureDTWTrading(
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

        # 匯出至 SQLite 與 CSV
        self._export_results(states)


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point：主入口
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir, db_method='Unknown', dataset_name='Unknown', db_path='results/result.db'):
    import inspect
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 動態取得參數
    init_sig = inspect.signature(PureDTWRollingBacktester.__init__)
    valid_params = {}
    
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[PURE_DTW] 正在初始化 PureDTWRollingBacktester (轉譯 Jupyter Notebook 邏輯)...")
    
    engine = PureDTWRollingBacktester(
        output_dir=out_dir,
        db_method=db_method,
        dataset_name=dataset_name,
        db_path=db_path,
        **valid_params
    )
    
    print(f"[PURE_DTW] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[PURE_DTW] 回測執行完畢。")
