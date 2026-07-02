"""
DRL LSTM-DQN 交易模組 v2 — 修復版
======================================================================

v1（drl_lstm_trading.py）診斷出四個缺陷，v2 逐一修復：

1. 假共享、真過擬合：v1 的 agent_key 含 ticker，實際上每配對每期獨立訓練，
   150 episodes 重複學同一條 252 天軌跡 → 背路徑而非學策略。
   v2：agent_key 只含期間 → 同期所有配對共享一個 agent；
   首配對完整訓練，後續配對增量微調（replay buffer 跨配對混合經驗）。

2. 獎勵重複計算：v1 持有期每日給 0.3× 日損益、平倉再給全額 PnL，同一筆利潤
   獎勵兩次，且 γ 折現誘導提早平倉（v1 平均持有 6.5 天、3000+ 次進出被費用磨死）。
   v2：單一獎勵來源 r_t = Δequity_t / capital × 100，
   equity = realized + unrealized（扣進場費與預估出場費）。
   開倉當日 equity 立即下降一個來回費用 → agent 自然學會「進場必須值回成本」。

3. epsilon 沒退完：v1 decay=0.995 × 150 episodes → 訓練結束仍有 47% 隨機動作。
   v2：依 episodes 自動計算 decay，訓練 80% 進度時到達 epsilon_end。

4. 觀測未標準化：v1 的 ZScore（±10）與 Rel_Return（±0.01）尺度差千倍直接餵 LSTM。
   v2：所有觀測特徵標準化至 ~[-3, 3]。

介面與 v1 完全相同（run_trading.py 直接可用）。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dataclasses import dataclass

torch.set_num_threads(1)


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
    prev_total_pnl: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# RL Environment（v2：Δequity 單一獎勵來源）
# ══════════════════════════════════════════════════════════════════════
class PairTradingEnv:
    """
    單配對交易環境。狀態 8 維（皆已標準化）：
      [ZScore_n, Rel_Return_n, MA_Dist_n, TTM, Vol_n, position, days_held_norm, Trend_n]
    動作：0=Flat, 1=Long Spread, 2=Short Spread
    獎勵：每日淨值變化（含摩擦成本），無事件型獎勵、無重複計算。
    """
    OBS_DIM = 8

    def __init__(self, df: pd.DataFrame, max_steps: int, friction_rate: float,
                 hedge_ratio: float, capital: float):
        self.df = df.reset_index(drop=True)
        self.max_steps = min(max_steps, len(self.df))
        self.friction_rate = friction_rate
        self.hedge_ratio = hedge_ratio
        self.capital = capital
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.shares_a = 0.0
        self.shares_b = 0.0
        self.entry_price_a = 0.0
        self.entry_price_b = 0.0
        self.entry_fee = 0.0
        self.days_held = 0
        self.realized_pnl = 0.0
        self.prev_equity = 0.0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[min(self.current_step, len(self.df) - 1)]
        ttm = (self.max_steps - self.current_step) / max(self.max_steps, 1)
        return np.array([
            row["ZScore_n"], row["Rel_Return_n"], row["MA_Dist_n"], ttm,
            row["Vol_n"], float(self.position),
            self.days_held / max(self.max_steps, 1), row["Trend_n"],
        ], dtype=np.float32)

    def _equity(self, p_a: float, p_b: float) -> float:
        """realized + 未實現淨損益（扣已付進場費與預估出場費）"""
        if self.position == 0:
            return self.realized_pnl
        raw = self.shares_a * (p_a - self.entry_price_a) + self.shares_b * (p_b - self.entry_price_b)
        exit_fee_est = (abs(self.shares_a) * p_a + abs(self.shares_b) * p_b) * self.friction_rate
        return self.realized_pnl + raw - self.entry_fee - exit_fee_est

    def step(self, action_idx: int):
        action = 0
        if action_idx == 1:
            action = 1
        elif action_idx == 2:
            action = -1

        row = self.df.iloc[self.current_step]
        p_a, p_b = float(row["Price_A"]), float(row["Price_B"])

        # ── 執行動作 ────────────────────────────────────────────
        if action != self.position:
            if self.position != 0:  # 平倉
                raw = self.shares_a * (p_a - self.entry_price_a) + self.shares_b * (p_b - self.entry_price_b)
                exit_fee = (abs(self.shares_a) * p_a + abs(self.shares_b) * p_b) * self.friction_rate
                self.realized_pnl += raw - self.entry_fee - exit_fee
                self.shares_a = self.shares_b = 0.0
                self.entry_fee = 0.0
                self.position = 0
                self.days_held = 0
            if action != 0:  # 開倉
                total_w = 1.0 + abs(self.hedge_ratio)
                v_a = self.capital / total_w
                v_b = self.capital * abs(self.hedge_ratio) / total_w
                if action == 1:
                    self.shares_a, self.shares_b = v_a / p_a, -v_b / p_b
                else:
                    self.shares_a, self.shares_b = -v_a / p_a, v_b / p_b
                self.entry_price_a, self.entry_price_b = p_a, p_b
                self.entry_fee = (abs(self.shares_a) * p_a + abs(self.shares_b) * p_b) * self.friction_rate
                self.position = action
                self.days_held = 0
        elif self.position != 0:
            self.days_held += 1

        # ── 獎勵 = 當日淨值變化（用下一日價格 mark-to-market） ────
        self.current_step += 1
        done = self.current_step >= self.max_steps
        if not done:
            nrow = self.df.iloc[self.current_step]
            equity = self._equity(float(nrow["Price_A"]), float(nrow["Price_B"]))
        else:
            # 期末強制平倉結算
            if self.position != 0:
                raw = self.shares_a * (p_a - self.entry_price_a) + self.shares_b * (p_b - self.entry_price_b)
                exit_fee = (abs(self.shares_a) * p_a + abs(self.shares_b) * p_b) * self.friction_rate
                self.realized_pnl += raw - self.entry_fee - exit_fee
                self.position = 0
                self.shares_a = self.shares_b = 0.0
            equity = self.realized_pnl

        reward = (equity - self.prev_equity) / self.capital * 100.0
        self.prev_equity = equity
        return self._get_obs(), float(reward), done


# ══════════════════════════════════════════════════════════════════════
# LSTM-DQN Agent（架構同 v1；epsilon 排程修正）
# ══════════════════════════════════════════════════════════════════════
class LSTM_DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=1, lr=1e-3,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim
        self.seq_len = 10

        self.model = LSTM_DQN(state_dim, hidden_dim, action_dim, num_layers).to(self.device)
        self.target_model = LSTM_DQN(state_dim, hidden_dim, action_dim, num_layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=100000)
        self.loss_fn = nn.MSELoss()

    def act(self, state_seq):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(np.array(state_seq)).unsqueeze(0).to(self.device)
            return self.model(t).argmax().item()

    def store_transition(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        s = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        a = torch.LongTensor([b[1] for b in batch]).to(self.device)
        r = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        d = torch.FloatTensor([b[4] for b in batch]).to(self.device)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_model(ns).max(1)[0]
            target = r + self.gamma * nq * (1 - d)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# ══════════════════════════════════════════════════════════════════════
# Trading（介面同 v1）
# ══════════════════════════════════════════════════════════════════════
class Trading:
    """DRL LSTM v2 Trading Strategy Interface for run_trading.py"""

    _shared_agents: dict = {}   # {period key -> (DQNAgent, set(trained pair keys))}
    _MAX_CACHED_AGENTS: int = 60

    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex,
                 selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float,
                 drl_episodes: int = 100, drl_batch_size: int = 64, drl_gamma: float = 0.99,
                 drl_epsilon_start: float = 1.0, drl_epsilon_end: float = 0.05,
                 drl_epsilon_decay: float = 0.0,   # 0 = 依 episodes 自動計算
                 drl_lr: float = 1e-3, drl_hidden_size: int = 64, drl_num_layers: int = 1,
                 drl_finetune_episodes: int = 0,   # 0 = episodes//5
                 full_price_df: pd.DataFrame = None,
                 formation_start: str = None, formation_end: str = None, **kwargs):

        _pct_clean = lambda df: df.where(df.pct_change().abs() <= 0.50).ffill().bfill() if df is not None else None
        self.trade_prices = _pct_clean(price_df.copy())
        self.full_price_df = _pct_clean(full_price_df.copy() if full_price_df is not None else None)

        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair
        self.friction_rate = fee_rate + slippage_rate

        self.drl_episodes = drl_episodes
        self.drl_batch_size = drl_batch_size
        self.drl_gamma = drl_gamma
        self.drl_epsilon_start = drl_epsilon_start
        self.drl_epsilon_end = drl_epsilon_end
        self.drl_epsilon_decay = drl_epsilon_decay
        self.drl_lr = drl_lr
        self.drl_hidden_size = drl_hidden_size
        self.drl_num_layers = drl_num_layers
        self.drl_finetune_episodes = drl_finetune_episodes

        self.formation_start = formation_start
        self.formation_end = formation_end

    # ── 特徵工程（v2：全特徵標準化至 ~[-3, 3]） ─────────────────────────
    def _prepare_features(self, p_a: pd.Series, p_b: pd.Series, hedge_ratio: float,
                          log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float,
                          spread_mean: float = None, spread_std: float = None):
        df = pd.DataFrame({"Price_A": p_a, "Price_B": p_b})

        norm_a = (np.log(df["Price_A"]) - log_mean_a) / log_std_a
        norm_b = (np.log(df["Price_B"]) - log_mean_b) / log_std_b
        spread = norm_a - hedge_ratio * norm_b

        if spread_mean is None:
            spread_mean = spread.mean()
        if spread_std is None or spread_std <= 0:
            spread_std = max(float(spread.std()), 1e-8)

        z = (spread - spread_mean) / spread_std
        df["ZScore"] = z                                   # 原始 z（供交易紀錄輸出）
        df["ZScore_n"] = np.clip(z, -6.0, 6.0) / 3.0

        ret_a = df["Price_A"].pct_change().fillna(0)
        ret_b = df["Price_B"].pct_change().fillna(0)
        df["Rel_Return_n"] = np.clip((ret_a - ret_b) * 50.0, -3.0, 3.0)

        ma_s = spread.rolling(5, min_periods=1).mean()
        ma_l = spread.rolling(21, min_periods=1).mean()
        df["MA_Dist_n"] = np.clip((ma_s - ma_l) / spread_std, -3.0, 3.0)

        roll_std = spread.rolling(20, min_periods=5).std()
        df["Vol_n"] = np.clip((roll_std / (roll_std.mean() + 1e-8)).fillna(1.0) - 1.0, -3.0, 3.0)

        df["Trend_n"] = np.clip((z - z.shift(5).fillna(z)), -6.0, 6.0) / 3.0

        return df.fillna(0)

    # ── 期間級共享 agent：首配對完整訓練，後續配對增量微調 ─────────────
    def _train_shared_agent(self, period_start: str, trade_start: str, ticker_a: str, ticker_b: str,
                            hedge_ratio: float, form_spread_mean: float, form_spread_std: float,
                            log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float):
        agent_key = f"{period_start}_{trade_start}"          # v2：不含 ticker → 真共享
        pair_key = f"{ticker_a}_{ticker_b}"

        cached = Trading._shared_agents.get(agent_key)
        if cached is not None:
            agent, trained_pairs = cached
            if pair_key in trained_pairs:
                return agent
            n_episodes = self.drl_finetune_episodes or max(5, self.drl_episodes // 5)
            eps_start = 0.2                                   # 微調：小幅探索
        else:
            if len(Trading._shared_agents) >= Trading._MAX_CACHED_AGENTS:
                for old in list(Trading._shared_agents.keys())[:len(Trading._shared_agents) - Trading._MAX_CACHED_AGENTS + 1]:
                    del Trading._shared_agents[old]
            agent = DQNAgent(
                state_dim=PairTradingEnv.OBS_DIM, action_dim=3,
                hidden_dim=self.drl_hidden_size, num_layers=self.drl_num_layers,
                lr=self.drl_lr, gamma=self.drl_gamma,
                epsilon_start=self.drl_epsilon_start, epsilon_end=self.drl_epsilon_end,
            )
            trained_pairs = set()
            Trading._shared_agents[agent_key] = (agent, trained_pairs)
            n_episodes = self.drl_episodes
            eps_start = self.drl_epsilon_start
            print(f"    [DRLv2] New shared agent for period {period_start} (device={agent.device})")

        # epsilon 排程：訓練 80% 進度時到達 epsilon_end（v1 缺陷 #3 修正）
        agent.epsilon = eps_start
        if self.drl_epsilon_decay > 0:
            agent.epsilon_decay = self.drl_epsilon_decay
        else:
            horizon = max(1, int(0.8 * n_episodes))
            agent.epsilon_decay = (self.drl_epsilon_end / max(eps_start, 1e-8)) ** (1.0 / horizon)

        # 形成期資料訓練（不觸碰交易期 → 無前視）
        form_prices = self.full_price_df.loc[self.formation_start:self.formation_end]
        p_a = form_prices[ticker_a].dropna()
        p_b = form_prices[ticker_b].dropna()
        common = p_a.index.intersection(p_b.index)
        if len(common) > 50:
            feat = self._prepare_features(p_a.loc[common], p_b.loc[common], hedge_ratio,
                                          log_mean_a, log_std_a, log_mean_b, log_std_b,
                                          form_spread_mean, form_spread_std)
            env = PairTradingEnv(feat, max_steps=len(feat), friction_rate=self.friction_rate,
                                 hedge_ratio=hedge_ratio, capital=self.capital_per_pair)
            agent.model.train()
            for ep in range(n_episodes):
                obs = env.reset()
                seq = deque([obs] * agent.seq_len, maxlen=agent.seq_len)
                done = False
                while not done:
                    action = agent.act(list(seq))
                    nobs, reward, done = env.step(action)
                    nseq = seq.copy()
                    nseq.append(nobs)
                    agent.store_transition(list(seq), action, reward, list(nseq), done)
                    if env.current_step % 4 == 0:
                        agent.replay(self.drl_batch_size)
                    seq = nseq
                agent.update_epsilon()
                if ep % 5 == 0:
                    agent.update_target_model()
            agent.update_target_model()

        trained_pairs.add(pair_key)
        return agent

    # ── 交易期模擬（貪婪推理；輸出格式同 v1 / zscore_trading） ──────────
    def _simulate_pair(self, period_start: str, period_end: str, sector: str,
                       ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float,
                       form_spread_mean: float, form_spread_std: float,
                       log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float,
                       **kwargs) -> pd.DataFrame:

        agent = self._train_shared_agent(self.formation_start, period_start, ticker_a, ticker_b,
                                         hedge_ratio, form_spread_mean, form_spread_std,
                                         log_mean_a, log_std_a, log_mean_b, log_std_b)
        agent.model.eval()

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

        # Warm-start：形成期末 seq_len 天初始化觀測序列
        seq_buf = np.zeros((agent.seq_len, PairTradingEnv.OBS_DIM), dtype=np.float32)
        try:
            form_prices = self.full_price_df.loc[self.formation_start:self.formation_end]
            fp_a = form_prices[ticker_a].dropna()
            fp_b = form_prices[ticker_b].dropna()
            cf = fp_a.index.intersection(fp_b.index)
            if len(cf) >= agent.seq_len + 5:
                ff = self._prepare_features(fp_a.loc[cf], fp_b.loc[cf], hedge_ratio,
                                            log_mean_a, log_std_a, log_mean_b, log_std_b,
                                            form_spread_mean, form_spread_std)
                tail = ff.iloc[-agent.seq_len:]
                for k in range(len(tail)):
                    r = tail.iloc[k]
                    seq_buf[k] = [r["ZScore_n"], r["Rel_Return_n"], r["MA_Dist_n"],
                                  (len(tail) - k) / max(len(tail), 1),
                                  r["Vol_n"], 0.0, 0.0, r["Trend_n"]]
        except Exception:
            pass

        state = PairState()
        out = {k: [] for k in ["dates", "pa", "pb", "hr", "z", "pos", "unreal", "real",
                               "cum", "status", "tpnl", "days", "delta"]}
        feat_np = feat_df[["Price_A", "Price_B", "ZScore", "ZScore_n", "Rel_Return_n",
                           "MA_Dist_n", "Vol_n", "Trend_n"]].values.astype(np.float32)
        total = len(dates_arr)
        inv_total = 1.0 / max(total, 1)

        for i in range(total):
            rn = feat_np[i]
            p_a, p_b, z_raw = float(rn[0]), float(rn[1]), float(rn[2])
            obs = np.array([rn[3], rn[4], rn[5], (total - i) * inv_total,
                            rn[6], float(state.position),
                            state.days_held * inv_total, rn[7]], dtype=np.float32)
            seq_buf[:-1] = seq_buf[1:]
            seq_buf[-1] = obs

            with torch.no_grad():
                q = agent.model(torch.from_numpy(seq_buf).unsqueeze(0).to(agent.device))
                action_idx = q.argmax().item()
            action = 0
            if action_idx == 1:
                action = 1
            elif action_idx == 2:
                action = -1

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

        # 期末強制平倉
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
