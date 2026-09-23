"""
DRL 探索版 D — 離散表格式價值迭代（Tabular Value Iteration，無函數逼近）
======================================================================
純探索、未預先登記，不是論文正式結果。v3/v3m/v3s/v3ms 四個變體全部屬於
同一個演算法家族——用梯度下降擬合一個參數化 Q 函數（LSTM 或 MLP）逼近
Bellman 目標。本檔換一個完全不同的家族：**把狀態離散化成 ZScore 區間
（bin），用精確的表格式 Bellman 回代（value iteration）求解**——沒有
神經網路、沒有梯度下降、沒有優化器超參數，是與「函數逼近＋SGD」正交的
方法家族。若這條臂在同一批配對上依然輸給門檻選擇式（DL-THR），
就更難把「逐日自由持倉表現差」歸咎於某個特定學習演算法的優化瑕疵。

狀態＝(ZScore 離散區間, 持倉)，動作＝目標持倉（與其餘四個變體相同的
{Flat, Long Spread, Short Spread} 動作空間、相同的獎勵與費用定義，
唯一變因是「用表格 + 精確回代」取代「用神經網路 + SGD」）。

介面與 v3 完全相同（run_trading.py 直接可用）。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

_FEAT_COLS = ["ZScore_n", "Rel_Return_n", "MA_Dist_n", "Vol_n", "Trend_n", "TTM"]
N_FEAT = len(_FEAT_COLS)
_POS_SIGN = np.array([0.0, 1.0, -1.0], dtype=np.float32)
_TURNOVER = np.abs(_POS_SIGN[:, None] - _POS_SIGN[None, :])

#: ZScore 離散化區間數與範圍。邊界外的值夾到頭尾兩格。
N_BINS = 41
_BIN_EDGES = np.linspace(-6.0, 6.0, N_BINS + 1)


def _digitize(z: np.ndarray) -> np.ndarray:
    idx = np.digitize(z, _BIN_EDGES[1:-1])  # 0..N_BINS-1
    return np.clip(idx, 0, N_BINS - 1).astype(np.int64)


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
    prev_total_pnl: float = 0.0


class Trading:
    """離散表格式價值迭代 Trading Strategy Interface for run_trading.py"""

    _shared_agents: dict = {}
    _MAX_CACHED_AGENTS: int = 40

    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex,
                 selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float,
                 drl_episodes: int = 100,          # 解讀為 value iteration 的 sweep 數
                 drl_batch_size: int = 4096,        # 未使用（表格法無 minibatch），保留以相容介面
                 drl_gamma: float = 0.99,
                 drl_lr: float = 1e-3,              # 未使用，保留以相容介面
                 drl_hidden_size: int = 64, drl_num_layers: int = 1,   # 未使用
                 drl_finetune_episodes: int = 0,
                 drl_scope: str = "period",
                 drl_buffer_periods: int = 24,
                 full_price_df: pd.DataFrame = None,
                 formation_start: str = None, formation_end: str = None, **kwargs):

        _pct_clean = lambda df: df.where(df.pct_change().abs() <= 0.50).ffill().bfill() if df is not None else None
        self.trade_prices = _pct_clean(price_df.copy())
        self.full_price_df = _pct_clean(full_price_df.copy() if full_price_df is not None else None)

        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair
        self.friction_rate = fee_rate + slippage_rate

        self.sweeps = max(50, drl_episodes)
        self.gamma = drl_gamma
        self.scope = drl_scope.lower()
        self.buffer_periods = max(1, drl_buffer_periods)

        self.formation_start = formation_start
        self.formation_end = formation_end

    # ── 特徵工程：與 v3 逐位相同（保留全部欄位以便未來擴充，實際只用 ZScore） ──
    def _prepare_features(self, p_a: pd.Series, p_b: pd.Series, hedge_ratio: float,
                          log_mean_a, log_std_a, log_mean_b, log_std_b,
                          spread_mean=None, spread_std=None):
        df = pd.DataFrame({"Price_A": p_a, "Price_B": p_b})
        norm_a = (np.log(df["Price_A"]) - log_mean_a) / log_std_a
        norm_b = (np.log(df["Price_B"]) - log_mean_b) / log_std_b
        spread = norm_a - hedge_ratio * norm_b

        if spread_mean is None:
            spread_mean = spread.mean()
        if spread_std is None or spread_std <= 0:
            spread_std = max(float(spread.std()), 1e-8)

        z = (spread - spread_mean) / spread_std
        df["ZScore"] = z
        ret_a = df["Price_A"].pct_change().fillna(0)
        ret_b = df["Price_B"].pct_change().fillna(0)
        df["Ret_A"] = ret_a
        df["Ret_B"] = ret_b
        return df.fillna(0)

    def _enumerate_transitions(self, feat: pd.DataFrame, hedge_ratio: float):
        """
        枚舉單配對形成期軌跡的全部 (t, p, a) 轉移，並附上離散化的
        state-bin(t) 與 next-bin(t+1)。獎勵/費用定義與 v3 逐位相同。
        """
        T = len(feat)
        if T < 3:
            return None
        w_b = abs(hedge_ratio) / (1.0 + abs(hedge_ratio))
        w_a = 1.0 - w_b
        ret_a = feat["Ret_A"].values.astype(np.float32)
        ret_b = feat["Ret_B"].values.astype(np.float32)
        pos_pnl = (w_a * ret_a[1:] - w_b * ret_b[1:]) * 100.0
        fee_unit = self.friction_rate * 100.0

        bins = _digitize(feat["ZScore"].values.astype(np.float64))

        n_dec = T - 1
        t_idx = np.repeat(np.arange(n_dec, dtype=np.int64), 9)
        p_idx = np.tile(np.repeat(np.arange(3, dtype=np.int64), 3), n_dec)
        a_idx = np.tile(np.arange(3, dtype=np.int64), 3 * n_dec)

        r = _POS_SIGN[a_idx] * pos_pnl[t_idx] - _TURNOVER[p_idx, a_idx] * fee_unit
        done = (t_idx == n_dec - 1)
        r = r - np.where(done, np.abs(_POS_SIGN[a_idx]) * fee_unit, 0.0)

        bin_t = bins[t_idx]
        bin_next = bins[t_idx + 1]
        return {"bin_t": bin_t, "bin_next": bin_next, "p": p_idx, "a": a_idx,
                "r": r.astype(np.float64), "done": done}

    def _value_iteration(self, ag: dict, n_sweeps: int):
        """精確表格式 Bellman 回代（無函數逼近、無梯度、無隨機性）。"""
        datasets = [d for plist in ag["period_data"].values() for d in plist]
        if not datasets:
            return
        bin_t = np.concatenate([d["bin_t"] for d in datasets])
        bin_next = np.concatenate([d["bin_next"] for d in datasets])
        p_g = np.concatenate([d["p"] for d in datasets])
        a_g = np.concatenate([d["a"] for d in datasets])
        r_g = np.concatenate([d["r"] for d in datasets])
        dn_g = np.concatenate([d["done"] for d in datasets]).astype(np.float64)

        flat_idx = bin_t * 9 + p_g * 3 + a_g
        Q = ag["Q"]

        for _ in range(n_sweeps):
            # 下一狀態的持倉＝本次選定的動作（環境動態：position(t+1) = a），
            # 故對 next-bin 只在「位置＝a_g」那一列上對動作取 max，不是對 bin_next 整塊取 max。
            q_next_max = Q[bin_next, a_g].max(axis=1)      # (N,)
            target = r_g + self.gamma * q_next_max * (1.0 - dn_g)

            sums = np.zeros(N_BINS * 3 * 3, dtype=np.float64)
            counts = np.zeros(N_BINS * 3 * 3, dtype=np.float64)
            np.add.at(sums, flat_idx, target)
            np.add.at(counts, flat_idx, 1.0)
            observed = counts.reshape(N_BINS, 3, 3) > 0
            new_vals = np.divide(sums, np.maximum(counts, 1.0)).reshape(N_BINS, 3, 3)
            Q = np.where(observed, new_vals, Q)

        ag["Q"] = Q

    def _get_agent(self, period_start: str, ticker_a: str, ticker_b: str, hedge_ratio: float,
                   form_spread_mean, form_spread_std, log_mean_a, log_std_a, log_mean_b, log_std_b):
        is_global = self.scope == "global"
        agent_key = "GLOBAL" if is_global else f"{self.formation_start}_{period_start}"
        pair_key = f"{period_start}_{ticker_a}_{ticker_b}"

        ag = Trading._shared_agents.get(agent_key)
        if ag is None:
            if len(Trading._shared_agents) >= Trading._MAX_CACHED_AGENTS:
                for old in list(Trading._shared_agents.keys())[:1]:
                    del Trading._shared_agents[old]
            ag = {"Q": np.zeros((N_BINS, 3, 3), dtype=np.float64),
                  "period_data": {}, "trained_pairs": set(), "initialized": False}
            Trading._shared_agents[agent_key] = ag
            print(f"    [DRL-TabularVI] New {'GLOBAL walk-forward' if is_global else 'period-shared'} "
                  f"agent (period {period_start})")

        if pair_key in ag["trained_pairs"]:
            return ag

        form_prices = self.full_price_df.loc[self.formation_start:self.formation_end]
        p_a = form_prices[ticker_a].dropna()
        p_b = form_prices[ticker_b].dropna()
        common = p_a.index.intersection(p_b.index)
        if len(common) > 50:
            feat = self._prepare_features(p_a.loc[common], p_b.loc[common], hedge_ratio,
                                          log_mean_a, log_std_a, log_mean_b, log_std_b,
                                          form_spread_mean, form_spread_std)
            trans = self._enumerate_transitions(feat, hedge_ratio)
            if trans is not None:
                new_period = period_start not in ag["period_data"]
                if new_period:
                    ag["period_data"][period_start] = []
                    while len(ag["period_data"]) > self.buffer_periods:
                        oldest = next(iter(ag["period_data"]))
                        del ag["period_data"][oldest]
                ag["period_data"][period_start].append(trans)
                self._value_iteration(ag, self.sweeps)
                ag["initialized"] = True

        ag["trained_pairs"].add(pair_key)
        return ag

    def _simulate_pair(self, period_start: str, period_end: str, sector: str,
                       ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float,
                       form_spread_mean: float, form_spread_std: float,
                       log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float,
                       **kwargs) -> pd.DataFrame:

        ag = self._get_agent(period_start, ticker_a, ticker_b, hedge_ratio,
                             form_spread_mean, form_spread_std,
                             log_mean_a, log_std_a, log_mean_b, log_std_b)
        Q = ag["Q"]

        price_a = self.trade_prices[ticker_a].dropna()
        price_b = self.trade_prices[ticker_b].dropna()
        common = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common], price_b.loc[common]
        if len(price_a) < 5:
            return pd.DataFrame()

        feat_df = self._prepare_features(price_a, price_b, hedge_ratio,
                                         log_mean_a, log_std_a, log_mean_b, log_std_b,
                                         form_spread_mean, form_spread_std)
        valid_idx = common.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()
        feat_df = feat_df.loc[valid_idx]
        dates_arr = valid_idx
        T = len(dates_arr)

        bins = _digitize(feat_df["ZScore"].values.astype(np.float64))
        Q_by_day = Q[bins]  # (T, 3持倉, 3動作)

        state = PairState()
        out = {k: [] for k in ["dates", "pa", "pb", "hr", "z", "pos", "unreal", "real",
                               "cum", "status", "tpnl", "days", "delta"]}
        pa_arr = feat_df["Price_A"].values
        pb_arr = feat_df["Price_B"].values
        z_arr = feat_df["ZScore"].values
        pos_to_idx = {0: 0, 1: 1, -1: 2}

        for i in range(T):
            p_a, p_b, z_raw = float(pa_arr[i]), float(pb_arr[i]), float(z_arr[i])
            if i < T - 1:
                action_idx = int(np.argmax(Q_by_day[i, pos_to_idx[state.position]]))
                action = 0
                if action_idx == 1:
                    action = 1
                elif action_idx == 2:
                    action = -1
            else:
                action = state.position

            unrealized = 0.0
            closed_pnl = 0.0
            status = "HOLD_CASH"

            if action != state.position:
                was_flat = state.position == 0
                if state.position != 0:
                    raw = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                    exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                    closed_pnl = raw - state.trade_entry_fee - exit_fee
                    state.realized_pnl += closed_pnl
                    state.shares_a = state.shares_b = 0.0
                    state.trade_entry_fee = 0.0
                    state.days_held = 0
                    status = "EXIT"
                if action != 0:
                    tw = 1.0 + abs(hedge_ratio)
                    v_a = self.capital_per_pair / tw
                    v_b = self.capital_per_pair * abs(hedge_ratio) / tw
                    if action == 1:
                        state.shares_a, state.shares_b = v_a / p_a, -v_b / p_b
                        status = "ENTER_LONG_A" if was_flat else "REVERSE_LONG_A"
                    else:
                        state.shares_a, state.shares_b = -v_a / p_a, v_b / p_b
                        status = "ENTER_SHORT_A" if was_flat else "REVERSE_SHORT_A"
                    state.entry_price_a, state.entry_price_b = p_a, p_b
                    state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                state.position = action
            elif state.position != 0:
                state.days_held += 1
                raw = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                unrealized = raw - state.trade_entry_fee - exit_fee
                status = "HOLDING"

            cum = state.realized_pnl + unrealized
            delta = cum - state.prev_total_pnl
            state.prev_total_pnl = cum

            out["dates"].append(dates_arr[i])
            out["pa"].append(round(p_a, 4))
            out["pb"].append(round(p_b, 4))
            out["hr"].append(round(float(hedge_ratio), 4))
            out["z"].append(round(z_raw, 4))
            out["pos"].append(state.position)
            out["unreal"].append(round(unrealized, 4))
            out["real"].append(round(state.realized_pnl, 4))
            out["cum"].append(round(cum, 4))
            out["status"].append(status)
            out["tpnl"].append(round(closed_pnl, 4))
            out["days"].append(state.days_held)
            out["delta"].append(round(delta, 4))

        if state.position != 0 and out["status"]:
            if out["status"][-1] not in ("EXIT", "PERIOD_END_EXIT"):
                before = out["cum"][-2] if len(out["cum"]) > 1 else 0.0
                p_a, p_b = out["pa"][-1], out["pb"][-1]
                raw = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                closed_pnl = raw - state.trade_entry_fee - exit_fee
                state.realized_pnl += closed_pnl
                out["status"][-1] = "PERIOD_END_EXIT"
                out["real"][-1] = round(state.realized_pnl, 4)
                out["cum"][-1] = round(state.realized_pnl, 4)
                out["unreal"][-1] = 0.0
                out["tpnl"][-1] = round(closed_pnl, 4)
                out["delta"][-1] = round(state.realized_pnl - before, 4)
                out["days"][-1] = state.days_held

        df_out = pd.DataFrame({
            "Date": out["dates"], "Price_A": out["pa"], "Price_B": out["pb"],
            "Hedge_Ratio": out["hr"], "ZScore": out["z"], "Position": out["pos"],
            "Unrealized_PnL": out["unreal"], "Realized_PnL": out["real"],
            "Cumulative_PnL": out["cum"], "Status": out["status"],
            "Trade_PnL": out["tpnl"], "Days_Held": out["days"], "Daily_Delta": out["delta"],
        })
        for k, v in {"Period_Start": period_start, "Period_End": period_end,
                     "Sector": sector, "Pair_Rank": pair_rank,
                     "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                     "Log_Mean_A": log_mean_a, "Log_Std_A": log_std_a,
                     "Log_Mean_B": log_mean_b, "Log_Std_B": log_std_b}.items():
            df_out[k] = v
        return df_out
