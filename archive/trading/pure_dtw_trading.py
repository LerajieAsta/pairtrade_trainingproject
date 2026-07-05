import pandas as pd
import numpy as np
from strategies.trading.zscore_trading import Trading, PairState

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
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float, 
                       form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float,
                       first_price_a: float = 0.0, first_price_b: float = 0.0, **kwargs) -> pd.DataFrame:
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: 
            return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5:
            return pd.DataFrame()


        # Spread 使用 OLS 殘差空間（與 DTW_Cointegration_Paper 形成期一致）
        ols_alpha_val = kwargs.get("ols_alpha", None)
        log_mean_a    = kwargs.get("log_mean_a", None)
        log_mean_b    = kwargs.get("log_mean_b", None)
        log_std_a     = kwargs.get("log_std_a", 1.0) or 1.0
        log_std_b     = kwargs.get("log_std_b", 1.0) or 1.0

        log_p_a = np.log(np.maximum(price_a, 1e-8))
        log_p_b = np.log(np.maximum(price_b, 1e-8))

        if ols_alpha_val is not None:
            # DTW：OLS 殘差 = log_A - alpha - beta * log_B（與形成期相同空間）
            spread = log_p_a - ols_alpha_val - hedge_ratio * log_p_b
        else:
            # Fallback：Z-score 標準化對數價格空間
            norm_p_a = (log_p_a - (log_mean_a or 0.0)) / (log_std_a if log_std_a > 1e-12 else 1.0)
            norm_p_b = (log_p_b - (log_mean_b or 0.0)) / (log_std_b if log_std_b > 1e-12 else 1.0)
            spread = norm_p_a - hedge_ratio * norm_p_b

        safe_std = max(form_spread_std, self.min_spread_std)

        if getattr(self, "use_vol_adjust", False):
            roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
            vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
            adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
        else:
            adjusted_std = safe_std

        zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)

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

            # 冷卻期重設：等 Z-score 回穿 0 才允許再進場
            if state.cooldown_dir == -1 and z >= 0.0:
                state.cooldown_dir = 0
            elif state.cooldown_dir == 1 and z <= 0.0:
                state.cooldown_dir = 0

            if state.position != 0:
                state.days_held += 1
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                
                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est
                
                is_cap_stop = self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and self.dynamic_stop_z > 0 and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    cooldown_to_set = state.position
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    state.cooldown_dir = cooldown_to_set
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED"
                else:
                    # 出場判斷 (Z-score 回歸均值)
                    is_exit_short = (state.position == -1) and (z <= self.exit_z)  
                    is_exit_long  = (state.position == 1)  and (z >= -self.exit_z)  
                    
                    if is_exit_short or is_exit_long:
                        cooldown_to_set = state.position
                        self._execute_close(state, current_trade_pnl, stop_loss=False)
                        state.cooldown_dir = cooldown_to_set
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

            # 停損後填補剩餘交易日並退出（vectorized bulk-extend）
            if state.is_stopped and i < len(dates_arr) - 1:
                remain = slice(i + 1, len(dates_arr))
                n_remain = len(dates_arr) - (i + 1)
                final_realized = round(float(state.realized_pnl), 4)
                out_dates.extend(dates_arr[remain])
                out_pa.extend([round(v, 4) for v in pa_arr[remain].tolist()])
                out_pb.extend([round(v, 4) for v in pb_arr[remain].tolist()])
                out_hr.extend([1.0] * n_remain)
                out_z.extend([round(float(v), 4) for v in np.where(np.isnan(zscore_arr[remain]), 0.0, zscore_arr[remain]).tolist()])
                out_pos.extend([0] * n_remain)
                out_unrealized.extend([0.0] * n_remain)
                out_realized.extend([final_realized] * n_remain)
                out_cum.extend([final_realized] * n_remain)
                out_status.extend(["STOPPED"] * n_remain)
                out_trade_pnl.extend([0.0] * n_remain)
                out_days.extend([0] * n_remain)
                out_delta.extend([0.0] * n_remain)
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

