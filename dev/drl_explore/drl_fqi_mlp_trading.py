"""
DRL FQI 探索版 A — 把 LSTM 市場編碼器換成 MLP（去掉遞迴）
======================================================================
純探索、未預先登記，不是論文正式結果。目的：v3（`archive/trading/
drl_fqi_trading.py`）用 LSTM 編碼 10 日特徵窗，本檔只把編碼器換成
「攤平視窗 + MLP」，其餘（Double-Q FQI 訓練、獎勵、回測會計）逐字相同，
用來檢驗「遞迴本身是否必要」——若 MLP 版表現與 v3 相近甚至更好，代表
v1/v2/v3 三代共同的 LSTM 選擇不是失敗的原因；若明顯更差，則 LSTM
至少沒有拖累。

介面與 v3 完全相同（run_trading.py 直接可用）。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

torch.set_num_threads(1)

_FEAT_COLS = ["ZScore_n", "Rel_Return_n", "MA_Dist_n", "Vol_n", "Trend_n", "TTM"]
N_FEAT = len(_FEAT_COLS)
SEQ_LEN = 10
_POS_SIGN = np.array([0.0, 1.0, -1.0], dtype=np.float32)
_TURNOVER = np.abs(_POS_SIGN[:, None] - _POS_SIGN[None, :])


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


class MLPQNet(nn.Module):
    """攤平視窗 + MLP 市場編碼器（無遞迴）＋ 持倉條件 Q-head。"""

    def __init__(self, feat_dim=N_FEAT, hidden_dim=64, num_layers=1, seq_len=SEQ_LEN):
        super().__init__()
        in_dim = feat_dim * seq_len
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def encode(self, seqs):                       # (B, SEQ_LEN, F) -> (B, H)
        B = seqs.shape[0]
        return self.encoder(seqs.reshape(B, -1))

    def q_from_h(self, h, pos_onehot):
        return self.head(torch.cat([h, pos_onehot], dim=1))


class Trading:
    """DRL FQI（MLP 編碼器）Trading Strategy Interface for run_trading.py"""

    _shared_agents: dict = {}
    _MAX_CACHED_AGENTS: int = 40

    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex,
                 selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float,
                 drl_episodes: int = 100,
                 drl_batch_size: int = 4096,
                 drl_gamma: float = 0.99,
                 drl_lr: float = 1e-3,
                 drl_hidden_size: int = 64, drl_num_layers: int = 1,
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
        self.finetune_sweeps = drl_finetune_episodes or max(20, self.sweeps // 3)
        self.batch_size = drl_batch_size
        self.scope = drl_scope.lower()
        self.buffer_periods = max(1, drl_buffer_periods)
        self.gamma = drl_gamma
        self.lr = drl_lr
        self.hidden = drl_hidden_size
        self.layers = drl_num_layers

        self.formation_start = formation_start
        self.formation_end = formation_end
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        df["ZScore_n"] = np.clip(z, -6.0, 6.0) / 3.0
        ret_a = df["Price_A"].pct_change().fillna(0)
        ret_b = df["Price_B"].pct_change().fillna(0)
        df["Ret_A"] = ret_a
        df["Ret_B"] = ret_b
        df["Rel_Return_n"] = np.clip((ret_a - ret_b) * 50.0, -3.0, 3.0)
        ma_s = spread.rolling(5, min_periods=1).mean()
        ma_l = spread.rolling(21, min_periods=1).mean()
        df["MA_Dist_n"] = np.clip((ma_s - ma_l) / spread_std, -3.0, 3.0)
        roll_std = spread.rolling(20, min_periods=5).std()
        df["Vol_n"] = np.clip((roll_std / (roll_std.mean() + 1e-8)).fillna(1.0) - 1.0, -3.0, 3.0)
        df["Trend_n"] = np.clip((z - z.shift(5).fillna(z)), -6.0, 6.0) / 3.0
        n = len(df)
        df["TTM"] = np.linspace(1.0, 1.0 / max(n, 1), n)
        return df.fillna(0)

    @staticmethod
    def _build_seqs(feat_np: np.ndarray) -> np.ndarray:
        T = feat_np.shape[0]
        padded = np.concatenate([np.repeat(feat_np[:1], SEQ_LEN - 1, axis=0), feat_np], axis=0)
        return np.stack([padded[t:t + SEQ_LEN] for t in range(T)], axis=0)

    def _enumerate_transitions(self, feat: pd.DataFrame, hedge_ratio: float):
        T = len(feat)
        if T < SEQ_LEN + 2:
            return None
        w_b = abs(hedge_ratio) / (1.0 + abs(hedge_ratio))
        w_a = 1.0 - w_b
        ret_a = feat["Ret_A"].values.astype(np.float32)
        ret_b = feat["Ret_B"].values.astype(np.float32)
        pos_pnl = (w_a * ret_a[1:] - w_b * ret_b[1:]) * 100.0
        fee_unit = self.friction_rate * 100.0

        n_dec = T - 1
        t_idx = np.repeat(np.arange(n_dec, dtype=np.int64), 9)
        p_idx = np.tile(np.repeat(np.arange(3, dtype=np.int64), 3), n_dec)
        a_idx = np.tile(np.arange(3, dtype=np.int64), 3 * n_dec)

        r = _POS_SIGN[a_idx] * pos_pnl[t_idx] - _TURNOVER[p_idx, a_idx] * fee_unit
        done = (t_idx == n_dec - 1)
        r = r - np.where(done, np.abs(_POS_SIGN[a_idx]) * fee_unit, 0.0)

        seqs = self._build_seqs(feat[_FEAT_COLS].values.astype(np.float32))
        return {"seqs": seqs, "t": t_idx, "p": p_idx, "a": a_idx,
                "r": r.astype(np.float32), "done": done}

    def _fqi_sweeps(self, ag: dict, n_sweeps: int):
        model, target, opt = ag["model"], ag["target"], ag["opt"]
        datasets = [d for plist in ag["period_data"].values() for d in plist]
        if not datasets:
            return
        loss_fn = nn.MSELoss()
        eye3 = torch.eye(3, device=self.device)

        seq_pool = torch.from_numpy(np.concatenate([d["seqs"] for d in datasets])).to(self.device)
        offs, off = [], 0
        for d in datasets:
            offs.append(off)
            off += d["seqs"].shape[0]
        t_g = torch.from_numpy(np.concatenate([d["t"] + o for d, o in zip(datasets, offs)])).to(self.device)
        p_g = torch.from_numpy(np.concatenate([d["p"] for d in datasets])).to(self.device)
        a_g = torch.from_numpy(np.concatenate([d["a"] for d in datasets])).to(self.device)
        r_g = torch.from_numpy(np.concatenate([d["r"] for d in datasets])).to(self.device)
        dn_g = torch.from_numpy(np.concatenate([d["done"] for d in datasets])).to(self.device).float()
        N = len(t_g)

        n_seq = seq_pool.shape[0]
        chunk = max(1024, int(self.batch_size))
        chunk_bounds = list(range(0, n_seq, chunk))
        chunk_idx = []
        for lo in chunk_bounds:
            hi = min(lo + chunk, n_seq)
            chunk_idx.append(((t_g >= lo) & (t_g < hi)).nonzero(as_tuple=True)[0])

        model.train()
        for sweep in range(n_sweeps):
            if sweep % 15 == 0:
                target.load_state_dict(model.state_dict())
                target.eval()
            if sweep % 5 == 0:
                with torch.no_grad():
                    H_t = target.encode(seq_pool)
                    H_m = model.encode(seq_pool)
                    hn_t, hn_m = H_t[t_g + 1], H_m[t_g + 1]
                    q_next_cols = []
                    for a_idx in range(3):
                        oh = eye3[a_idx].expand(N, 3)
                        a_star = model.q_from_h(hn_m, oh).argmax(1, keepdim=True)
                        q_next_cols.append(target.q_from_h(hn_t, oh).gather(1, a_star).squeeze(1))
                    q_next = torch.stack(q_next_cols, dim=1)
                    y_all = r_g + self.gamma * q_next.gather(1, a_g.unsqueeze(1)).squeeze(1) * (1 - dn_g)

            for lo, idx in zip(chunk_bounds, chunk_idx):
                if idx.numel() == 0:
                    continue
                hi = min(lo + chunk, n_seq)
                H = model.encode(seq_pool[lo:hi])
                q = model.q_from_h(H[t_g[idx] - lo], eye3[p_g[idx]])
                q_sa = q.gather(1, a_g[idx].unsqueeze(1)).squeeze(1)
                loss = loss_fn(q_sa, y_all[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        target.load_state_dict(model.state_dict())
        target.eval()

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
            model = MLPQNet(N_FEAT, self.hidden, self.layers).to(self.device)
            target = MLPQNet(N_FEAT, self.hidden, self.layers).to(self.device)
            target.load_state_dict(model.state_dict())
            target.eval()
            ag = {"model": model, "target": target,
                  "opt": optim.Adam(model.parameters(), lr=self.lr),
                  "period_data": {}, "trained_pairs": set(), "initialized": False}
            Trading._shared_agents[agent_key] = ag
            print(f"    [DRL-FQI-MLP] New {'GLOBAL walk-forward' if is_global else 'period-shared'} "
                  f"agent (period {period_start}, device={self.device})")

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

                if is_global:
                    if new_period:
                        n = self.sweeps if not ag["initialized"] else self.finetune_sweeps
                        self._fqi_sweeps(ag, n)
                        ag["initialized"] = True
                else:
                    n = self.sweeps if not ag["initialized"] else self.finetune_sweeps
                    self._fqi_sweeps(ag, n)
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
        model = ag["model"]
        model.eval()

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

        seqs = torch.from_numpy(self._build_seqs(feat_df[_FEAT_COLS].values.astype(np.float32))).to(self.device)
        eye3 = torch.eye(3, device=self.device)
        with torch.no_grad():
            H = model.encode(seqs)
            Q_all = torch.stack([model.q_from_h(H, eye3[p].expand(T, 3)) for p in range(3)], dim=1)
        Q_np = Q_all.cpu().numpy()

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
                action_idx = int(np.argmax(Q_np[i, pos_to_idx[state.position]]))
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
