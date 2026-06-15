import pandas as pd
import numpy as np
from dataclasses import dataclass

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
                       form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float,
                       first_price_a: float = 0.0, first_price_b: float = 0.0,
                       ols_alpha: float = None) -> pd.DataFrame:
        """
        ols_alpha: OLS 截距，由 HDBSCAN 等 OLS 形成期傳入。
          - 若非 None：使用原始 log-price 空間計算 spread（與形成期一致），修復 Bug #1/#2。
          - 若 None：使用標準化 norm_p 空間（SSD / DTW 等舊路徑，保持向下相容）。
        """
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: return pd.DataFrame()

        # ── 路徑 A：OLS log-price 空間（HDBSCAN 系列） ──────────────────────
        # 條件：ols_alpha 被明確傳入（非 None），使用原始 log-price 空間，
        # spread = log_A - ols_alpha - hedge_ratio * log_B，與形成期 OLS 殘差完全一致。
        if ols_alpha is not None:
            log_p_a = np.log(price_a)
            log_p_b = np.log(price_b)

            if self.zscore_window == 0:
                # 靜態：使用形成期 OLS 參數（alpha, beta）重建殘差
                spread = log_p_a - ols_alpha - hedge_ratio * log_p_b
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
                # 滾動：在原始 log-price 空間做 rolling OLS
                roll_cov = log_p_b.rolling(window=self.zscore_window).cov(log_p_a)
                roll_var = log_p_b.rolling(window=self.zscore_window).var()
                roll_beta = np.where(roll_var > 1e-8, roll_cov / roll_var, 0.0)
                roll_beta = pd.Series(roll_beta, index=common_idx)
                roll_mean_a = log_p_a.rolling(window=self.zscore_window).mean()
                roll_mean_b = log_p_b.rolling(window=self.zscore_window).mean()
                roll_alpha = roll_mean_a - roll_beta * roll_mean_b
                spread = log_p_a - roll_alpha - roll_beta * log_p_b
                roll_var_a = log_p_a.rolling(window=self.zscore_window).var()
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

        # ── 路徑 B：標準化 norm_p 空間（SSD / DTW，向下相容） ────────────────
        else:
            if first_price_a > 0.0 and first_price_b > 0.0 and log_mean_a == 0.0:
                norm_p_a = price_a / first_price_a
                norm_p_b = price_b / first_price_b
            else:
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
        
        base_log = {
            "Period_Start": period_start, "Period_End": period_end,
            "Sector": sector, "Pair_Rank": pair_rank,
            "Ticker_A": ticker_a, "Ticker_B": ticker_b,
        }
        if first_price_a > 0.0 and log_mean_a == 0.0:
            base_log["First_Price_A"] = first_price_a
            base_log["First_Price_B"] = first_price_b
        else:
            base_log["Log_Mean_A"] = log_mean_a
            base_log["Log_Std_A"] = log_std_a
            base_log["Log_Mean_B"] = log_mean_b
            base_log["Log_Std_B"] = log_std_b
            
        for k, v in base_log.items():
            df_out[k] = v
            
        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """執行該期所有配對的交易模擬"""
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            # OLS_Alpha 存在時（HDBSCAN 系列）走 log-price 路徑；
            # 不存在時（SSD/DTW）走舊有標準化路徑（向下相容）
            raw_alpha = row.get("OLS_Alpha", None)
            ols_alpha_val = float(raw_alpha) if raw_alpha is not None and not pd.isna(raw_alpha) else None
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
                log_std_b=float(row.get("Log_Std_B", 1.0)),
                first_price_a=float(row.get("First_Price_A", 0.0)),
                first_price_b=float(row.get("First_Price_B", 0.0)),
                ols_alpha=ols_alpha_val
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
