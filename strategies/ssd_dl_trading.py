# ======================================================================
"""
SSD + 深度學習動態交易期進出場分析策略 (SSD_DL_Trading)
核心功能：基於 SSD (Sum of Squared Differences) 篩選配對，並以 PyTorch MLP 進行交易期動態信號預測與過濾。
"""

import sqlite3
import warnings
import itertools
import inspect
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import torch
import torch.nn as nn
import torch.optim as optim

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Class 0：深度學習模型 (PyTorch MLP Regressor)
# ══════════════════════════════════════════════════════════════════════════════
class MLPRegressor(nn.Module):
    """
    輕量級三層 MLP 迴歸模型，輸入 6 維特徵，預測未來第 5 天的 Z-Score。
    """
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_models_for_pairs(form_data: pd.DataFrame, selected_pairs: pd.DataFrame, epochs: int = 50) -> dict:
    """
    在形成期 (Formation Period) 數據上快速訓練每個配對的深度學習模型。
    """
    models = {}
    if selected_pairs.empty or form_data.empty:
        return models
        
    log_prices = np.log(form_data)
    
    for _, row in selected_pairs.iterrows():
        ta = row["Ticker_A"]
        tb = row["Ticker_B"]
        beta = float(row["Hedge_Ratio"])
        
        if ta not in log_prices.columns or tb not in log_prices.columns:
            continue
            
        # 計算形成期價差 Z-Score
        mean_a = log_prices[ta].mean()
        std_a = log_prices[ta].std()
        mean_b = log_prices[tb].mean()
        std_b = log_prices[tb].std()
        
        norm_a = (log_prices[ta] - mean_a) / (std_a if std_a > 1e-8 else 1.0)
        norm_b = (log_prices[tb] - mean_b) / (std_b if std_b > 1e-8 else 1.0)
        
        spread = norm_a - beta * norm_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread, ddof=1) if len(spread) > 1 else 1.0
        safe_std = max(spread_std, 1e-6)
        
        zscore_series = (spread - spread_mean) / safe_std
        z_vals = zscore_series.values
        
        # 構建訓練集 (輸入 6 維特徵，預測 5 天後的 Z-Score)
        X, Y = [], []
        for t in range(9, len(z_vals) - 5):
            feat = [
                z_vals[t],
                z_vals[t] - z_vals[t-1],
                z_vals[t] - z_vals[t-3],
                z_vals[t] - z_vals[t-5],
                float(np.std(z_vals[t-9:t+1])),
                float(np.mean(z_vals[t-4:t+1]))
            ]
            X.append(feat)
            Y.append(z_vals[t+5])
            
        if len(X) < 10:
            continue
            
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
        
        # 宣告並訓練 PyTorch 模型 (強制使用 CPU 以確保多行程安全性與速度)
        model = MLPRegressor(input_dim=6)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        model.train()
        X_t = torch.tensor(X)
        Y_t = torch.tensor(Y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, Y_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        models[(ta, tb)] = model
        
    return models

# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）
# ══════════════════════════════════════════════════════════════════════════════
class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    與進階 SSD (OLS) 相同，採用對數價格與 OLS 殘差滾動。
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
        """對數價格與 Z-Score 正規化"""
        log_prices = np.log(self.price_df)
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        self.normalized_df = (log_prices - self.mean_prices) / self.std_prices
        return self.normalized_df

    def compute_ssd(self) -> pd.DataFrame:
        """計算同產業配對的 SSD 距離並以 OLS 估算 Beta"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        ssd_records = []

        # 產業分組
        sector_groups = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        for sector, sector_tickers in sector_groups.items():
            if sector == "Unknown" or len(sector_tickers) < self.min_tickers_for_pairing: 
                continue

            norm_vals = self.normalized_df[sector_tickers].values.T
            ssd_matrix = ssd.pdist(norm_vals, metric='sqeuclidean')
            
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
                    
                    spread = norm_vals[j] - beta * norm_vals[i]
                    spread_mean = np.mean(spread)
                    spread_std = np.std(spread, ddof=1) if len(spread) > 1 else 0.0
                    
                    ssd_records.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                        "SSD": round(ssd_value, 6), "Hedge_Ratio": round(beta, 4),
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std": round(spread_std, 6)
                    })

        if not ssd_records: 
            return pd.DataFrame()
            
        return pd.DataFrame(ssd_records).sort_values("SSD").reset_index(drop=True)

    def select_pairs(self) -> pd.DataFrame:
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
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs

# ══════════════════════════════════════════════════════════════════════════════
# Class 2：Trading（交易期模組）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class PairState:
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
    """
    負責在交易期 (Trading Period) 模擬配對交易，並結合已訓練之 PyTorch 預測模型進行動態過濾。
    """
    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, selected_pairs: pd.DataFrame, 
                 trained_models: dict, capital_per_pair: float, fee_rate: float, slippage_rate: float, 
                 stop_loss_pct: float, entry_z: float, exit_z: float, zscore_window: int, allow_reentry: bool = False,
                 zscore_clip: float = 10.0, min_spread_std: float = 1e-6):
        self.price_df = price_df.copy()
        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.trained_models = trained_models
        self.capital_per_pair = capital_per_pair
        
        self.friction_rate = fee_rate + slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.zscore_window = zscore_window
        self.allow_reentry = allow_reentry  
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std

        self.period_pnl: float = 0.0

    def _execute_entry(self, state: PairState, z: float, pred_z_future: float, p_a: float, p_b: float, hedge_ratio: float) -> tuple[bool, float]:
        """
        結合深度學習預測值進行開倉過濾。
        """
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)
        
        # 1. 開空條件：Z > entry_z 且 DL 預測未來會下跌 (均值復歸)
        if z > self.entry_z and state.cooldown_dir != -1:
            if pred_z_future < z - 0.3: # 深度學習過濾器：預估會下跌至少 0.3
                state.position = -1
                state.shares_a = -v_a / p_a
                state.shares_b = v_b / p_b
            else:
                return False, 0.0 # 拒絕開倉
        # 2. 開多條件：Z < -entry_z 且 DL 預測未來會上漲 (均值復歸)
        elif z < -self.entry_z and state.cooldown_dir != 1:
            if pred_z_future > z + 0.3: # 深度學習過濾器：預估會反彈至少 0.3
                state.position = +1
                state.shares_a = v_a / p_a
                state.shares_b = -v_b / p_b
            else:
                return False, 0.0 # 拒絕開倉
        else:
            return False, 0.0

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state: PairState, current_trade_pnl: float, stop_loss: bool = False):
        state.realized_pnl += current_trade_pnl 
        if stop_loss:
            if not self.allow_reentry:
                state.is_stopped = True
            else:
                state.cooldown_dir = state.position 
        else:
            state.cooldown_dir = 0  
                
        state.position = 0
        state.shares_a = 0.0
        state.shares_b = 0.0
        state.entry_price_a = 0.0
        state.entry_price_b = 0.0
        state.trade_entry_fee = 0.0

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float, 
                       form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float) -> pd.DataFrame:
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: 
            return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: 
            return pd.DataFrame()

        log_p_a = np.log(price_a)
        log_p_b = np.log(price_b)

        norm_p_a = (log_p_a - log_mean_a) / log_std_a
        norm_p_b = (log_p_b - log_mean_b) / log_std_b
        
        # 1. 估計 Z-Score 序列
        if self.zscore_window == 0:
            spread = norm_p_a - hedge_ratio * norm_p_b
            safe_std = max(form_spread_std, self.min_spread_std)
            zscore = np.clip((spread - form_spread_mean) / safe_std, -self.zscore_clip, self.zscore_clip)
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
            zscore = np.clip(spread / safe_std, -self.zscore_clip, self.zscore_clip)
            beta_series = roll_beta

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0: 
            return pd.DataFrame()
        
        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        zscore = zscore.loc[valid_idx]
        beta_series = beta_series.loc[valid_idx]

        # 取得深度學習預測模型
        dl_model = self.trained_models.get((ticker_a, ticker_b))

        # 加速模擬之陣列解構
        dates_arr = valid_idx
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values
        beta_arr = beta_series.values

        # 為了能在 trading 期間取得完整的 10 天特徵歷史，我們使用 common_idx 上的 zscore 陣列
        common_z_dict = {d: val for d, val in zip(common_idx, zscore.values)}
        common_z_list = zscore.values.tolist()
        common_date_to_idx = {d: i for i, d in enumerate(common_idx)}

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

            # 解除冷卻
            if state.cooldown_dir == -1 and z <= self.exit_z:
                state.cooldown_dir = 0
            elif state.cooldown_dir == 1 and z >= -self.exit_z:
                state.cooldown_dir = 0

            # 提取當前時間點的價差特徵，呼叫 DL 模型進行預測
            pred_z_future = z # 預設為當前值
            if dl_model is not None:
                c_idx = common_date_to_idx.get(date, -1)
                if c_idx >= 9:
                    hist = common_z_list[c_idx-9 : c_idx+1]
                    feat = [
                        hist[-1],
                        hist[-1] - hist[-2],
                        hist[-1] - hist[-4],
                        hist[-1] - hist[-6],
                        float(np.std(hist)),
                        float(np.mean(hist[-5:]))
                    ]
                    # 輸入特徵轉 Tensor
                    feat_t = torch.tensor([feat], dtype=torch.float32)
                    with torch.no_grad():
                        pred_z_future = float(dl_model(feat_t).item())

            # 持倉邏輯
            if state.position != 0:
                state.days_held += 1
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est
                
                # 判斷是否止損
                is_sl = self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                
                # 深度學習動態止損/提前平倉
                # 如果持有多單，但 DL 預測未來會下跌；或者持有空單，預測未來會上漲
                is_dl_early_exit = False
                if state.position == 1 and pred_z_future < z - 0.5:
                    is_dl_early_exit = True
                elif state.position == -1 and pred_z_future > z + 0.5:
                    is_dl_early_exit = True
                
                if is_sl or is_dl_early_exit:
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED" if is_sl else "DL_EARLY_EXIT"
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
                # 空手狀態下開倉，引入深度學習過濾器
                if abs(z) > self.entry_z:
                    entered, unrealized_pnl = self._execute_entry(state, z, pred_z_future, p_a, p_b, c_beta)
                    if entered:
                        current_status = "ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A"
                    else:
                        current_status = "HOLD_CASH (COOLDOWN_OR_DL_FILTER)"
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

            if current_status in ["STOP_LOSS_TRIGGERED", "EXIT", "DL_EARLY_EXIT"]:
                state.days_held = 0 

            # 已停損跳轉
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

        # 期末強制平倉
        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED", "DL_EARLY_EXIT"):
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
            
        log_df = pd.concat(dfs, ignore_index=True)
        period_daily_delta = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily_delta.sum()) if not period_daily_delta.empty else 0.0
        
        return log_df, self.period_pnl

# ══════════════════════════════════════════════════════════════════════════════
# Class 3：RollingBacktester（滾動回測引擎）
# ══════════════════════════════════════════════════════════════════════════════
class RollingBacktester:
    def __init__(self, top_n_list: list, stop_loss_list: list, zscore_window_list: list,
                 entry_z: float, exit_z: float, formation_window: int, trading_window: int, rolling_step: int,
                 fee_rate: float, slippage_rate: float, initial_capital: float,
                 allow_reentry: bool, zscore_clip: float, min_spread_std: float,
                 min_tickers_for_pairing: int, output_dir: Path):
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

    def run(self, price_pivot: pd.DataFrame, all_dates: list, total_days: int, local_first_trade_idx: int, sector_mapping: dict):
        max_concurrent = self.trading_window // self.rolling_step
        states = {}

        for n, sl, z_win in itertools.product(self.top_n_list, self.stop_loss_list, self.zscore_window_list):
            states[(n, sl, z_win)] = {
                "logs": [], 
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent} for _ in range(max_concurrent)]
            }

        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始進行 Grid Search (SSD + DL)，共 {len(roll_start_indices)} 期，每期處理 {len(states)} 種參數組合...")

        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx, form_end_idx = trade_start_idx - self.formation_window, trade_start_idx
            trade_end_idx = min(trade_start_idx + self.trading_window, total_days)

            form_data_raw = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data_raw = price_pivot.iloc[trade_start_idx:trade_end_idx]
            
            extended_trade_start_idx = max(0, trade_start_idx - max(self.zscore_window_list) - 10) # 預留多 10 天作為 DL 特徵緩衝
            extended_trade_data_raw = price_pivot.iloc[extended_trade_start_idx:trade_end_idx]
            
            valid_cols = (form_data_raw.isnull().sum() + extended_trade_data_raw.isnull().sum()) == 0
            
            form_data = form_data_raw.loc[:, valid_cols]
            trade_data = trade_data_raw.loc[:, valid_cols]
            trade_dates = trade_data.index
            extended_trade_data = extended_trade_data_raw.loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty: 
                continue

            trade_start_str, trade_end_str = str(all_dates[trade_start_idx])[:10], str(all_dates[trade_end_idx - 1])[:10]
            form_start_str, form_end_str = str(all_dates[form_start_idx])[:10], str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 處理中：第 {roll_idx+1:02d} 期 (交易: {trade_start_str} ~ {trade_end_str})")

            # 進行配對篩選
            formation = Formation(
                price_df=form_data, 
                form_start=form_start_str, 
                form_end=form_end_str, 
                top_n=max(self.top_n_list), 
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing
            )
            max_selected_pairs = formation.select_pairs()

            if max_selected_pairs.empty: 
                continue

            # 深度學習特有步驟：在形成期數據上訓練每個配對的 PyTorch 模型
            # 這大幅提昇了速度，因為只需在形成期訓練一次，而非在交易期每天訓練
            trained_models = train_models_for_pairs(form_data, max_selected_pairs, epochs=50)

            # 對所有參數組合進行交易模擬
            for n, sl, z_win in itertools.product(self.top_n_list, self.stop_loss_list, self.zscore_window_list):
                selected_pairs = max_selected_pairs.head(n)
                state = states[(n, sl, z_win)]
                slots = state["slots"]
                
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
                    trained_models=trained_models,
                    capital_per_pair=current_capital_per_pair,
                    fee_rate=self.fee_rate,
                    slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl,
                    entry_z=self.entry_z,
                    exit_z=self.exit_z,
                    zscore_window=z_win,
                    allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip,
                    min_spread_std=self.min_spread_std
                )
                
                trade_log_df, period_pnl = trading.run(trade_start_str, trade_end_str)
                
                if not trade_log_df.empty:
                    state["logs"].append(trade_log_df)
                    
                slots[slot_idx]["capital"] = max(0, current_period_capital + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        # 匯出紀錄
        self._export_results(states)

    def _export_results(self, states: dict):
        print("\n✅ 回測完成！正在匯出交易紀錄檔案...")
        for (n, sl, z_win), state in states.items():
            if state["logs"]:
                full_log_df = pd.concat(state["logs"], ignore_index=True)
                sl_str = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                filename = f"TradeLogs_Top{n}_{sl_str}_ZWin{z_win}.csv"
                filepath = self.output_dir / filename
                full_log_df.to_csv(filepath, index=False)
                print(f"  - 已輸出: {filename} (共 {len(full_log_df)} 筆紀錄)")
                
        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# 標準化策略進入點接口
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}
    
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
