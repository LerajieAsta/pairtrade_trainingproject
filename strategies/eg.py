

# ======================================================================
"""
Engle-Granger 共整合配對交易滾動回測系統
核心模組：
 1. 形成期 (Formation)：
    - Pearson 相關係數快篩 (>=0.7) 確保強烈同向性。
    - OLS 估計迴歸係數 β，ADF 檢定殘差 (p-value < 0.01) 確立共整合。
    - 統一 A 為被解釋變數 (Y)，依 ADF_Stat 挑選最穩健配對。
 2. 交易期 (Trading) - Optimal Double Stopping：
    - 突破 ±2 不進場，等待反向穿越才建倉 (濾除假突破)。
    - 防跳空保護：單日直接越過均值，直接解除武裝不進場。
    - 停損永久冷凍：觸發停損代表共整合破裂，本期不再交易。
 3. 資金管理：
    - 進場當日不計費，合併於持倉與出場時認列雙端手續費，避免雙重扣費。
    - 採用動態總資本池依 Top N 均分。
"""

import sqlite3
import warnings
import itertools
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ══════════════════════════════════════════════════════════════════════════════
# 輔助函式：OLS 估計與 ADF 檢定
# ══════════════════════════════════════════════════════════════════════════════

def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """執行 OLS：y = alpha + beta * x + resid"""
    n = len(y)
    x_mat = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, y - np.mean(y)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    resid = y - alpha - beta * x
    return alpha, beta, resid

def _adf_test(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """對殘差序列做 ADF 檢定，回傳 (ADF_Stat, P-value)"""
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="c", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0

# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）
# ══════════════════════════════════════════════════════════════════════════════

class Formation:
    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        adf_max_lags: int = 1,
        p_value_threshold: float = 0.01,
    ):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.adf_max_lags = adf_max_lags
        self.p_value_threshold = p_value_threshold
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def compute_cointegration(self) -> pd.DataFrame:
        log_prices = np.log(self.price_df)
        tickers = log_prices.columns.tolist()

        sector_groups: dict[str, list[str]] = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        eg_records = []
        for sector, sector_tickers in sector_groups.items():
            if sector == "Unknown" or len(sector_tickers) < self.min_tickers_for_pairing:
                continue

            for i, ticker_1 in enumerate(sector_tickers):
                log_1 = log_prices[ticker_1].values
                for j in range(i + 1, len(sector_tickers)):
                    ticker_2 = sector_tickers[j]
                    log_2 = log_prices[ticker_2].values

                    # Pearson 相關係數快篩
                    if np.corrcoef(log_1, log_2)[0, 1] < 0.7:
                        continue

                    # 雙向檢定
                    alpha_12, beta_12, resid_12 = _ols(log_1, log_2)
                    adf_12, pval_12 = _adf_test(resid_12, max_lags=self.adf_max_lags)

                    alpha_21, beta_21, resid_21 = _ols(log_2, log_1)
                    adf_21, pval_21 = _adf_test(resid_21, max_lags=self.adf_max_lags)

                    # 綁定最適合的 Y 變數為 Ticker_A
                    if pval_12 <= pval_21:
                        best_pval, best_adf = pval_12, adf_12
                        best_alpha, best_beta, best_resid = alpha_12, beta_12, resid_12
                        ticker_y, ticker_x = ticker_1, ticker_2
                    else:
                        best_pval, best_adf = pval_21, adf_21
                        best_alpha, best_beta, best_resid = alpha_21, beta_21, resid_21
                        ticker_y, ticker_x = ticker_2, ticker_1

                    # 檢定門檻與排除無效的負相關配對
                    if best_pval >= self.p_value_threshold or best_beta <= 0:
                        continue
                    
                    equation = f"Log({ticker_y}) = {round(best_alpha,4)} + {round(best_beta,4)}*Log({ticker_x})"
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    eg_records.append({
                        "Form_Start":  self.form_start,
                        "Form_End":    self.form_end,
                        "Sector":      sector,
                        "Ticker_A":    ticker_y,
                        "Ticker_B":    ticker_x,
                        "Hedge_Equation": equation, 
                        "ADF_Stat":    round(best_adf,  6),
                        "P_Value":     round(best_pval, 6),
                        "Hedge_Ratio": round(best_beta, 6),   
                        "OLS_Alpha":   round(best_alpha, 6),  
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std":  round(spread_std,  6),
                    })

        if not eg_records:
            return pd.DataFrame()

        df = pd.DataFrame(eg_records).sort_values("ADF_Stat", ascending=True).reset_index(drop=True)
        return df

    def run(self) -> pd.DataFrame:
        eg_df = self.compute_cointegration()
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        selected = eg_df.head(self.top_n).copy()
        selected["Rank"] = range(1, len(selected) + 1)

        # 計算對數價格統計量
        log_prices = np.log(self.price_df)
        mean_prices = log_prices.mean()
        std_prices = log_prices.std()

        selected["Log_Mean_A"] = selected["Ticker_A"].map(mean_prices)
        selected["Log_Std_A"]  = selected["Ticker_A"].map(std_prices)
        selected["Log_Mean_B"] = selected["Ticker_B"].map(mean_prices)
        selected["Log_Std_B"]  = selected["Ticker_B"].map(std_prices)

        self.selected_pairs = selected
        return self.selected_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Class 2：Trading（交易期模組）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PairState:
    position: int = 0         
    armed: int = 0            
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    prev_total_pnl: float = 0.0


class Trading:
    def __init__(
        self,
        price_df: pd.DataFrame,
        trade_dates: pd.DatetimeIndex,
        selected_pairs: pd.DataFrame,
        capital_per_pair: float,
        fee_rate: float,
        slippage_rate: float,
        stop_loss_pct: float,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        exit_buffer: float = 0.05,
        zscore_clip: float = 10.0,
        min_spread_std: float = 1e-6,
    ):
        self.price_df = price_df.copy()
        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair

        self.friction_rate = fee_rate + slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.exit_buffer = exit_buffer
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.period_pnl: float = 0.0

    def _execute_entry(
        self, state: PairState, dir: int, p_a: float, p_b: float, hedge_ratio: float
    ) -> tuple[bool, float]:
        weight_a = 1.0 / (1.0 + abs(hedge_ratio))
        weight_b = abs(hedge_ratio) / (1.0 + abs(hedge_ratio))
        
        v_a = self.capital_per_pair * weight_a
        v_b = self.capital_per_pair * weight_b

        state.position = dir
        beta_sign = 1 if hedge_ratio >= 0 else -1

        if dir == 1:
            state.shares_a = v_a / p_a
            state.shares_b = -beta_sign * (v_b / p_b)
        elif dir == -1:
            state.shares_a = -v_a / p_a
            state.shares_b = beta_sign * (v_b / p_b)

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        state.armed = 0 

        # 進場日不計費，延遲至持倉評估與平倉時認列
        return True, 0.0 

    def _execute_close(self, state: PairState, current_trade_pnl: float, stop_loss: bool = False):
        state.realized_pnl += current_trade_pnl
        if stop_loss:
            state.is_stopped = True
        state.position = 0
        state.armed = 0

    def _simulate_pair(
        self,
        period_start: str,
        period_end: str,
        sector: str,
        ticker_a: str,
        ticker_b: str,
        pair_rank: int,
        hedge_equation: str,
        hedge_ratio: float,
        ols_alpha: float,
        form_spread_mean: float,
        form_spread_std: float,
        log_mean_a: float,
        log_std_a: float,
        log_mean_b: float,
        log_std_b: float,
    ) -> pd.DataFrame:

        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns:
            return pd.DataFrame()

        price_a = self.price_df[ticker_a].dropna()
        price_b = self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        
        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()

        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        
        log_a = np.log(price_a)
        log_b = np.log(price_b)

        spread = log_a - ols_alpha - hedge_ratio * log_b
        safe_std = max(form_spread_std, self.min_spread_std)
        zscore = np.clip(
            (spread - form_spread_mean) / safe_std,
            -self.zscore_clip, self.zscore_clip
        )

        dates_arr = valid_idx.values
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values

        state = PairState()
        
        out_dates, out_pa, out_pb = [], [], []
        out_z, out_pos = [], []
        out_unrealized, out_realized, out_cum = [], [], []
        out_status, out_trade_pnl, out_days, out_delta = [], [], [], []

        for i in range(len(dates_arr)):
            date = dates_arr[i]
            z = float(zscore_arr[i])
            p_a, p_b = float(pa_arr[i]), float(pb_arr[i])
            
            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0
            current_status = "FLAT"

            if state.is_stopped:
                current_status = "STOPPED"
                unrealized_pnl = 0.0

            elif state.position != 0:
                state.days_held += 1
                raw_unrealized = (
                    state.shares_a * (p_a - state.entry_price_a) +
                    state.shares_b * (p_b - state.entry_price_b)
                )
                exit_fee_est = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                
                # 評估損益時同步扣除雙端手續費
                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est

                if self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct:
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED"
                else:
                    is_exit_short = (state.position == -1) and (z <= self.exit_z + self.exit_buffer)
                    is_exit_long  = (state.position == 1) and (z >= self.exit_z - self.exit_buffer)
                    
                    if is_exit_short or is_exit_long:
                        self._execute_close(state, current_trade_pnl, stop_loss=False)
                        closed_trade_pnl = current_trade_pnl
                        current_status = "EXIT"
                    else:
                        unrealized_pnl = current_trade_pnl
                        current_status = "HOLDING"
            else:
                if state.armed == 0:
                    if z > self.entry_z:
                        state.armed = -1 
                        current_status = "ARMED_SHORT"
                    elif z < -self.entry_z:
                        state.armed = 1  
                        current_status = "ARMED_LONG"
                        
                elif state.armed == -1:
                    if z < -self.entry_z:
                        state.armed = 1  
                        current_status = "ARMED_LONG"
                    elif z <= self.exit_z:
                        state.armed = 0  
                        current_status = "MISSED_OPPORTUNITY"
                    elif self.exit_z < z < self.entry_z:
                        _, unrealized_pnl = self._execute_entry(state, -1, p_a, p_b, hedge_ratio)
                        current_status = "ENTER_SHORT"
                    else:
                        current_status = "ARMED_SHORT"
                        
                elif state.armed == 1:
                    if z > self.entry_z:
                        state.armed = -1 
                        current_status = "ARMED_SHORT"
                    elif z >= self.exit_z:
                        state.armed = 0  
                        current_status = "MISSED_OPPORTUNITY"
                    elif -self.entry_z < z < self.exit_z:
                        _, unrealized_pnl = self._execute_entry(state, 1, p_a, p_b, hedge_ratio)
                        current_status = "ENTER_LONG"
                    else:
                        current_status = "ARMED_LONG"

            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl

            out_dates.append(date); out_pa.append(round(p_a, 4)); out_pb.append(round(p_b, 4))
            out_z.append(round(z, 4)); out_pos.append(state.position)
            out_unrealized.append(round(unrealized_pnl, 4)); out_realized.append(round(state.realized_pnl, 4))
            out_cum.append(round(cumulative_pnl, 4)); out_status.append(current_status)
            out_trade_pnl.append(round(closed_trade_pnl, 4)); out_days.append(state.days_held)
            out_delta.append(round(daily_delta, 4))

        # 期末強制平倉
        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                pnl_before_last = out_cum[-2] if len(out_cum) > 1 else 0.0

                p_a_last, p_b_last = float(pa_arr[-1]), float(pb_arr[-1])
                raw_unrealized_final = (
                    state.shares_a * (p_a_last - state.entry_price_a) +
                    state.shares_b * (p_b_last - state.entry_price_b)
                )
                exit_fee = (abs(state.shares_a) * p_a_last + abs(state.shares_b) * p_b_last) * self.friction_rate
                closed_trade_pnl = raw_unrealized_final - state.trade_entry_fee - exit_fee
                
                state.realized_pnl += closed_trade_pnl
                daily_delta = state.realized_pnl - pnl_before_last

                out_status[-1]    = "PERIOD_END_EXIT"
                out_realized[-1]  = round(state.realized_pnl, 4)
                out_cum[-1]       = round(state.realized_pnl, 4)
                out_unrealized[-1] = 0.0
                out_trade_pnl[-1] = round(closed_trade_pnl, 4)
                out_delta[-1]     = round(daily_delta, 4)
                out_days[-1]      = state.days_held

        df_out = pd.DataFrame({
            "Date":           out_dates,
            "Price_A":        out_pa,
            "Price_B":        out_pb,
            "Hedge_Equation": hedge_equation, 
            "Hedge_Ratio":    hedge_ratio,
            "ZScore":         out_z,
            "Position":       out_pos,
            "Unrealized_PnL": out_unrealized,
            "Realized_PnL":   out_realized,
            "Cumulative_PnL": out_cum,
            "Status":         out_status,
            "Trade_PnL":      out_trade_pnl,
            "Days_Held":      out_days,
            "Daily_Delta":    out_delta,
            "Period_Start":   period_start,
            "Period_End":     period_end,
            "Sector":         sector,
            "Pair_Rank":      pair_rank,
            "Ticker_A":       ticker_a,
            "Ticker_B":       ticker_b,
            "Log_Mean_A":     log_mean_a,
            "Log_Std_A":      log_std_a,
            "Log_Mean_B":     log_mean_b,
            "Log_Std_B":      log_std_b,
        })
        
        ordered_cols = [
            "Date", "Price_A", "Price_B", "Hedge_Equation", "Hedge_Ratio", "ZScore", "Position", 
            "Unrealized_PnL", "Realized_PnL", "Cumulative_PnL", "Status", 
            "Trade_PnL", "Days_Held", "Daily_Delta", "Period_Start", "Period_End", 
            "Sector", "Pair_Rank", "Ticker_A", "Ticker_B", "Log_Mean_A", 
            "Log_Std_A", "Log_Mean_B", "Log_Std_B"
        ]
        return df_out[ordered_cols]

    def run(self, period_start: str, period_end: str) -> tuple[pd.DataFrame, float]:
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start,
                period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"],
                ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_equation=str(row.get("Hedge_Equation", "")),
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                ols_alpha=float(row.get("OLS_Alpha", 0.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A", 1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B", 1.0)),
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily_delta = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily_delta.sum()) if not period_daily_delta.empty else 0.0

        return log_df, self.period_pnl

# ══════════════════════════════════════════════════════════════════════════════
# Class 3：DataProcessor（數據清理與前置處理模組）
# ══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
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
            return mapping
        except Exception as e:
            logging.warning(f"載入產業分類表失敗，切換為全市場配對。({e})")
            return {}

    def prepare_backtest_data(self, backtest_start: str, backtest_end: str, formation_window: int):
        conn = sqlite3.connect(self.db_path)
        raw_df = pd.read_sql_query(
            f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price "
            f"FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC",
            conn,
        )
        conn.close()

        raw_df["date"]  = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]

        price_pivot = raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()

        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                return dt + pd.offsets.MonthEnd(0) if is_end and len(str(d_str).strip()) == 7 else dt
            except Exception:
                return None

        bt_start_ts = _safe_parse(backtest_start)
        bt_end_ts   = _safe_parse(backtest_end, is_end=True)
        all_dates   = price_pivot.index.tolist()

        # [修正]：先找到切片起點
        first_idx = next((i for i, d in enumerate(all_dates) if d >= bt_start_ts), 0) if bt_start_ts else 0

        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end   = bt_end_ts if bt_end_ts else price_pivot.index[-1]

        # [修正]：先切片到回測所需時段
        price_pivot = price_pivot.loc[data_slice_start:data_slice_end]

        # [修正]：再做缺值處理，避免誤殺較晚上市的標的
        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20].ffill(limit=5)
        price_pivot.dropna(axis=1, how='any', inplace=True)

        sliced_dates = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_trade_idx = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_trade_idx, formation_window)

# ══════════════════════════════════════════════════════════════════════════════
# Class 4：RollingBacktester（滾動回測引擎）
# ══════════════════════════════════════════════════════════════════════════════

class RollingBacktester:
    def __init__(
        self,
        top_n_list: list,
        stop_loss_list: list,
        entry_z: float,
        exit_z: float,
        exit_buffer: float,
        formation_window: int,
        trading_window: int,
        rolling_step: int,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        zscore_clip: float,
        min_spread_std: float,
        min_tickers_for_pairing: int,
        adf_max_lags: int,
        p_value_threshold: float,
        output_dir: Path,
    ):
        self.top_n_list            = top_n_list
        self.stop_loss_list        = stop_loss_list
        self.entry_z               = entry_z
        self.exit_z                = exit_z
        self.exit_buffer           = exit_buffer
        self.formation_window      = formation_window
        self.trading_window        = trading_window
        self.rolling_step          = rolling_step
        self.fee_rate              = fee_rate
        self.slippage_rate         = slippage_rate
        self.initial_capital       = initial_capital
        self.zscore_clip           = zscore_clip
        self.min_spread_std        = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.adf_max_lags          = adf_max_lags
        self.p_value_threshold     = p_value_threshold
        self.output_dir            = output_dir

    def run(
        self,
        price_pivot: pd.DataFrame,
        all_dates: list,
        total_days: int,
        local_first_trade_idx: int,
        sector_mapping: dict,
    ):
        states = {}
        for n, sl in itertools.product(self.top_n_list, self.stop_loss_list):
            states[(n, sl)] = {"logs": [], "total_capital": self.initial_capital}

        roll_start_indices = list(
            range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step)
        )

        logging.info(f"🚀 開始執行回測，共 {len(roll_start_indices)} 期，每期 {len(states)} 種組合...")

        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx = trade_start_idx - self.formation_window
            form_end_idx   = trade_start_idx
            trade_end_idx  = min(trade_start_idx + self.trading_window, total_days)

            form_data  = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data = price_pivot.iloc[trade_start_idx:trade_end_idx]
            valid_cols = (form_data.isnull().sum() + trade_data.isnull().sum()) == 0

            form_data  = form_data.loc[:, valid_cols]
            trade_data = trade_data.loc[:, valid_cols]
            trade_dates = trade_data.index

            if form_data.shape[1] < 2 or trade_data.empty:
                continue

            trade_start_str = str(all_dates[trade_start_idx])[:10]
            trade_end_str   = str(all_dates[trade_end_idx - 1])[:10]
            form_start_str  = str(all_dates[form_start_idx])[:10]
            form_end_str    = str(all_dates[form_end_idx - 1])[:10]
            logging.info(f"▶ 處理中：第 {roll_idx+1:02d} 期 ({trade_start_str} ~ {trade_end_str})")

            formation = Formation(
                price_df=form_data,
                form_start=form_start_str,
                form_end=form_end_str,
                top_n=max(self.top_n_list),
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing,
                adf_max_lags=self.adf_max_lags,
                p_value_threshold=self.p_value_threshold,
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty:
                continue

            for n, sl in itertools.product(self.top_n_list, self.stop_loss_list):
                selected_pairs = max_selected_pairs.head(n)
                state  = states[(n, sl)]
                
                current_capital_per_pair = state["total_capital"] / n

                trading = Trading(
                    price_df=trade_data,
                    trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=current_capital_per_pair,
                    fee_rate=self.fee_rate,
                    slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl,
                    entry_z=self.entry_z,
                    exit_z=self.exit_z,
                    exit_buffer=self.exit_buffer,
                    zscore_clip=self.zscore_clip,
                    min_spread_std=self.min_spread_std,
                )

                trade_log_df, period_pnl = trading.run(trade_start_str, trade_end_str)

                if not trade_log_df.empty:
                    state["logs"].append(trade_log_df)

                state["total_capital"] = max(0, state["total_capital"] + period_pnl)

        self._export_results(states)

    def _export_results(self, states: dict):
        for (n, sl), state in states.items():
            if state["logs"]:
                full_log_df = pd.concat(state["logs"], ignore_index=True)
                sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                filename = f"EG_TradeLogs_Top{n}_{sl_str}.csv"
                filepath = self.output_dir / filename
                full_log_df.to_csv(filepath, index=False)
        logging.info(f"✅ 交易紀錄已儲存至: {self.output_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════



# ======================================================================
# 原 Main 區塊被自動化註釋解耦如下：
# if __name__ == "__main__":

#     MIN_TICKERS_FOR_PAIRING = 2
#     ZSCORE_CLIP             = 10.0
#     MIN_SPREAD_STD          = 1e-6

#     ADF_MAX_LAGS      = 1      
#     P_VALUE_THRESHOLD = 0.01   

#     DB_PATH, TABLE_NAME      = r"../data/sp500.db", "Daily_Prices"
#     BACKTEST_START, BACKTEST_END = "2000-01", "2025-12"
#     INFO_TABLE_NAME, TICKER_COL_NAME, SECTOR_COL_NAME = "Constituents", "Symbol", "GICS_Sector"

#     OUTPUT_DIR = Path(r"../results/full/EG_NoReEntry")
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     TOP_N_LIST     = [1, 5, 20]
#     STOP_LOSS_LIST = [0, 0.05, 0.15]

#     ENTRY_Z, EXIT_Z, EXIT_BUFFER         = 2.0, 0.0, 0.05
#     FORMATION_WINDOW, TRADING_WINDOW, ROLLING_STEP = 252, 126, 21

#     FEE_RATE          = 0.001
#     SLIPPAGE_RATE     = 0.001
#     INITIAL_CAPITAL   = 10_000
#     USE_SECTOR_PAIRING = True

#     processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)

#     if USE_SECTOR_PAIRING:
#         sector_mapping = processor.load_sector_mapping(INFO_TABLE_NAME, TICKER_COL_NAME, SECTOR_COL_NAME)
#     else:
#         sector_mapping = {}

#     price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
#         BACKTEST_START, BACKTEST_END, FORMATION_WINDOW
#     )

#     engine = RollingBacktester(
#         top_n_list=TOP_N_LIST,
#         stop_loss_list=STOP_LOSS_LIST,
#         entry_z=ENTRY_Z,
#         exit_z=EXIT_Z,
#         exit_buffer=EXIT_BUFFER,
#         formation_window=FORMATION_WINDOW,
#         trading_window=TRADING_WINDOW,
#         rolling_step=ROLLING_STEP,
#         fee_rate=FEE_RATE,
#         slippage_rate=SLIPPAGE_RATE,
#         initial_capital=INITIAL_CAPITAL,
#         zscore_clip=ZSCORE_CLIP,
#         min_spread_std=MIN_SPREAD_STD,
#         min_tickers_for_pairing=MIN_TICKERS_FOR_PAIRING,
#         adf_max_lags=ADF_MAX_LAGS,
#         p_value_threshold=P_VALUE_THRESHOLD, 
#         output_dir=OUTPUT_DIR,
#     )

#     engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)


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
