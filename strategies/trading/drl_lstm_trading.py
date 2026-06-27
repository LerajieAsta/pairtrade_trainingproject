import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import math
from dataclasses import dataclass

# Avoid CPU thread thrashing when running PyTorch in multiprocessing
torch.set_num_threads(1)

# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════
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
# RL Environment
# ══════════════════════════════════════════════════════════════════════
class PairTradingEnv(gym.Env):
    """
    自定義 Gymnasium 環境：模擬單一配對在時間序列上的交易。
    狀態包含: Spread Z-Score, 相對報酬率, 均線距離, Time-to-Maturity, 動態波動度。
    """
    def __init__(self, df: pd.DataFrame, max_steps: int, friction_rate: float, hedge_ratio: float, capital: float):
        super(PairTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = min(max_steps, len(self.df) - 1)
        self.friction_rate = friction_rate
        self.hedge_ratio = hedge_ratio
        self.capital = capital
        
        # Action: 0 (Flat), 1 (Long Spread: Buy A, Sell B), 2 (Short Spread: Sell A, Buy B)
        self.action_space = spaces.Discrete(3)
        # 8 features: ZScore, Rel_Return, MA_Dist, time_to_maturity, Spread_Std, position, days_held_norm, Spread_Trend
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.current_step = 0
        self.position = 0
        self.realized_pnl = 0.0
        self.shares_a = 0.0
        self.shares_b = 0.0
        self.entry_price_a = 0.0
        self.entry_price_b = 0.0
        self.entry_fee = 0.0
        self.days_held = 0

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0 # 0=flat, 1=long, -1=short
        self.realized_pnl = 0.0
        self.shares_a = 0.0
        self.shares_b = 0.0
        self.entry_price_a = 0.0
        self.entry_price_b = 0.0
        self.entry_fee = 0.0
        self.days_held = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        time_to_maturity = (self.max_steps - self.current_step) / self.max_steps
        volatility = float(row['Spread_Std']) if 'Spread_Std' in row else 1.0
        spread_trend = float(row['Spread_Trend']) if 'Spread_Trend' in row else 0.0

        obs = np.array([
            row['ZScore'],                                    # Spread Z-Score
            row['Rel_Return'],                                # 相對報酬率
            row['MA_Dist'],                                   # 均線距離
            time_to_maturity,                                 # 剩餘交易天數
            volatility,                                       # 正規化滾動波動度
            float(self.position),                             # 倉位方向 {-1, 0, 1}
            self.days_held / max(self.max_steps, 1),          # 持倉天數（正規化）
            spread_trend,                                     # Z-Score 5 日動量
        ], dtype=np.float32)
        return obs
        
    def step(self, action_idx):
        # map action_idx to position: 0 -> 0, 1 -> 1, 2 -> -1
        action = 0
        if action_idx == 1: action = 1
        elif action_idx == 2: action = -1
        
        row = self.df.iloc[self.current_step]
        p_a = row['Price_A']
        p_b = row['Price_B']
        
        reward = 0.0
        done = False
        prev_position = self.position

        # 動作激勵/懲罰與事件型獎勵
        if action != self.position:
            # 平倉事件 (Close position)
            if self.position != 0:
                raw_unrealized = self.shares_a * (p_a - self.entry_price_a) + self.shares_b * (p_b - self.entry_price_b)
                exit_fee = (abs(self.shares_a)*p_a + abs(self.shares_b)*p_b) * self.friction_rate
                trade_pnl = raw_unrealized - self.entry_fee - exit_fee
                        reward += (trade_pnl / self.capital) * 100.0

                self.realized_pnl += trade_pnl
                self.shares_a = 0.0
                self.shares_b = 0.0
                self.position = 0
                self.entry_fee = 0.0

            # 開倉事件 (Open position)
            if action != 0:
                total_weight = 1.0 + abs(self.hedge_ratio)
                v_a = self.capital * (1.0 / total_weight)
                v_b = self.capital * (abs(self.hedge_ratio) / total_weight)

                if action == 1: # Long Spread -> Buy A, Sell B
                    self.shares_a = v_a / p_a
                    self.shares_b = -v_b / p_b
                else: # Short Spread -> Sell A, Buy B
                    self.shares_a = -v_a / p_a
                    self.shares_b = v_b / p_b

                self.entry_price_a = p_a
                self.entry_price_b = p_b
                self.entry_fee = (abs(self.shares_a)*p_a + abs(self.shares_b)*p_b) * self.friction_rate
                self.position = action

                # Action Penalty (減少過度交易)
                reward -= (self.entry_fee / self.capital) * 100.0 * 0.5

        if self.position == prev_position and self.position != 0:
            self.days_held += 1
        else:
            self.days_held = 0

        # Daily holding reward: 引導 agent 感知每日損益方向
        if self.position != 0 and self.current_step + 1 < len(self.df):
            next_row = self.df.iloc[self.current_step + 1]
            daily_delta = (
                self.shares_a * (next_row['Price_A'] - p_a) +
                self.shares_b * (next_row['Price_B'] - p_b)
            )
            reward += (daily_delta / self.capital) * 100.0 * 0.3

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            # 強制平倉
            if self.position != 0:
                row_last = self.df.iloc[self.current_step - 1]
                p_a = row_last['Price_A']
                p_b = row_last['Price_B']
                raw_unrealized = self.shares_a * (p_a - self.entry_price_a) + self.shares_b * (p_b - self.entry_price_b)
                exit_fee = (abs(self.shares_a)*p_a + abs(self.shares_b)*p_b) * self.friction_rate
                trade_pnl = raw_unrealized - self.entry_fee - exit_fee
                reward += (trade_pnl / self.capital) * 100.0 
                self.realized_pnl += trade_pnl
            
        next_obs = self._get_obs()
        return next_obs, reward, done, False, {}

    def render(self):
        pass

# ══════════════════════════════════════════════════════════════════════
# RL Agent & Model
# ══════════════════════════════════════════════════════════════════════
class LSTM_DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_DQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last hidden state
        return self.fc(out)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=1, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim
        self.seq_len = 10 # 固定的時間序列長度
        
        self.model = LSTM_DQN(state_dim, hidden_dim, action_dim, num_layers).to(self.device)
        self.target_model = LSTM_DQN(state_dim, hidden_dim, action_dim, num_layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()
        
    def act(self, state_seq):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
            
    def store_transition(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        action_batch = torch.LongTensor([b[1] for b in batch]).to(self.device)
        reward_batch = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        done_batch = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        
        q_values = self.model(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            next_state_values = next_q_values.max(1)[0]
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
            
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ══════════════════════════════════════════════════════════════════════
# Integration Strategy Class
# ══════════════════════════════════════════════════════════════════════
class Trading:
    """DRL LSTM Trading Strategy Interface for run_trading.py"""

    _shared_agents: dict = {}  # class-level cache: {formation+trade+pair key -> DQNAgent}

    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float, 
                 drl_episodes: int = 100, drl_batch_size: int = 64, drl_gamma: float = 0.99,
                 drl_epsilon_start: float = 1.0, drl_epsilon_end: float = 0.05, drl_epsilon_decay: float = 0.995,
                 drl_lr: float = 1e-3, drl_hidden_size: int = 64, drl_num_layers: int = 1,
                 full_price_df: pd.DataFrame = None, formation_start: str = None, formation_end: str = None, **kwargs):
        
        self.trade_prices = price_df.copy()
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
        
        self.full_price_df = full_price_df
        self.formation_start = formation_start
        self.formation_end = formation_end
        
    def _prepare_features(self, p_a: pd.Series, p_b: pd.Series, hedge_ratio: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float, spread_mean: float = None, spread_std: float = None):
        """Prepare RL features matching the formation/trading period."""
        df = pd.DataFrame({'Price_A': p_a, 'Price_B': p_b})
        
        norm_a = (np.log(df['Price_A']) - log_mean_a) / log_std_a
        norm_b = (np.log(df['Price_B']) - log_mean_b) / log_std_b
        
        # Spread & Z-Score
        spread = norm_a - hedge_ratio * norm_b
        df['Spread'] = spread
        
        if spread_mean is None:
            spread_mean = spread.mean()
        if spread_std is None:
            spread_std = spread.std()
            
        df['ZScore'] = (spread - spread_mean) / (spread_std if spread_std > 0 else 1.0)
        
        # Relative Return
        ret_a = df['Price_A'].pct_change().fillna(0)
        ret_b = df['Price_B'].pct_change().fillna(0)
        df['Rel_Return'] = ret_a - ret_b
        
        # MA Distance (Short vs Long)
        ma_short = spread.rolling(window=5, min_periods=1).mean()
        ma_long = spread.rolling(window=21, min_periods=1).mean()
        df['MA_Dist'] = ma_short - ma_long
        
        # Volatility (normalized rolling std)
        roll_std = spread.rolling(window=20, min_periods=5).std()
        df['Spread_Std'] = (roll_std / (roll_std.mean() + 1e-8)).fillna(1.0)

        # Spread Trend (5-day Z-score momentum, clipped for stability)
        df['Spread_Trend'] = (df['ZScore'] - df['ZScore'].shift(5).fillna(df['ZScore'])).clip(-5.0, 5.0)

        return df.fillna(0)

    def _train_shared_agent(self, period_start: str, trade_start: str, sector: str, ticker_a: str, ticker_b: str, hedge_ratio: float, 
                            form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float):
        """Trains or retrieves the shared agent for this period."""
        agent_key = f"{period_start}_{trade_start}_{ticker_a}_{ticker_b}"

        if agent_key in Trading._shared_agents:
            return Trading._shared_agents[agent_key]
            
        print(f"    [DRL] Initializing and training new Shared Agent for period {period_start}...")
        
        agent = DQNAgent(state_dim=8, action_dim=3, hidden_dim=self.drl_hidden_size, num_layers=self.drl_num_layers,
                         lr=self.drl_lr, gamma=self.drl_gamma, epsilon_start=self.drl_epsilon_start, 
                         epsilon_end=self.drl_epsilon_end, epsilon_decay=self.drl_epsilon_decay)
                         
        # 使用當前配對的形成期資料訓練；隨配對迭代持續累積，作為共享 agent
        form_start_idx = self.full_price_df.index.get_loc(pd.to_datetime(self.formation_start))
        form_end_idx = self.full_price_df.index.get_loc(pd.to_datetime(self.formation_end))
        form_prices = self.full_price_df.iloc[form_start_idx:form_end_idx]
        
        p_a = form_prices[ticker_a].dropna()
        p_b = form_prices[ticker_b].dropna()
        common_idx = p_a.index.intersection(p_b.index)
        
        if len(common_idx) > 50:
            p_a, p_b = p_a.loc[common_idx], p_b.loc[common_idx]
            feat_df = self._prepare_features(p_a, p_b, hedge_ratio, log_mean_a, log_std_a, log_mean_b, log_std_b, form_spread_mean, form_spread_std)
            
            env = PairTradingEnv(feat_df, max_steps=len(feat_df), friction_rate=self.friction_rate, hedge_ratio=hedge_ratio, capital=self.capital_per_pair)
            
            # Fast Training Loop
            for ep in range(self.drl_episodes):
                obs, _ = env.reset()
                state_seq = deque([obs]*agent.seq_len, maxlen=agent.seq_len)
                done = False
                
                while not done:
                    action = agent.act(list(state_seq))
                    next_obs, reward, done, _, _ = env.step(action)
                    next_state_seq = state_seq.copy()
                    next_state_seq.append(next_obs)
                    
                    agent.store_transition(list(state_seq), action, reward, list(next_state_seq), done)
                    
                    # Update model every 4 steps to speed up training
                    if env.current_step % 4 == 0:
                        agent.replay(self.drl_batch_size)
                    
                    state_seq = next_state_seq
                    
                agent.update_epsilon()
                if ep % 5 == 0:
                    agent.update_target_model()
                    
        Trading._shared_agents[agent_key] = agent
        return agent

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float, 
                       form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float, **kwargs) -> pd.DataFrame:
        
        agent = self._train_shared_agent(self.formation_start, period_start, sector, ticker_a, ticker_b, hedge_ratio, 
                                         form_spread_mean, form_spread_std, log_mean_a, log_std_a, log_mean_b, log_std_b)
        agent.model.eval()
        
        price_a, price_b = self.trade_prices[ticker_a].dropna(), self.trade_prices[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        # ── 資料清洗：過濾單日漲跌幅超過 50% 的異常點 ──────────────────────
        _max_daily_move = 0.50
        price_a = price_a.where(price_a.pct_change().abs() <= _max_daily_move).ffill().bfill()
        price_b = price_b.where(price_b.pct_change().abs() <= _max_daily_move).ffill().bfill()
        common_idx = price_a.dropna().index.intersection(price_b.dropna().index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: return pd.DataFrame()

        feat_df = self._prepare_features(price_a, price_b, hedge_ratio, log_mean_a, log_std_a, log_mean_b, log_std_b, form_spread_mean, form_spread_std)
        valid_idx = common_idx.intersection(self.trade_dates)
        
        if len(valid_idx) == 0: return pd.DataFrame()
        
        feat_df = feat_df.loc[valid_idx]
        dates_arr = valid_idx
        
        state = PairState()
        out_dates, out_pa, out_pb, out_hr, out_z, out_pos = [], [], [], [], [], []
        out_unrealized, out_realized, out_cum, out_status, out_trade_pnl, out_days, out_delta = [], [], [], [], [], [], []

        STATE_DIM = 8

        # Warm-start: 用 formation 期最後 seq_len 天初始化 state_seq，消除冷啟動偏誤
        try:
            form_prices = self.full_price_df.loc[self.formation_start:self.formation_end]
            fp_a = form_prices[ticker_a].dropna()
            fp_b = form_prices[ticker_b].dropna()
            common_form = fp_a.index.intersection(fp_b.index)
            if len(common_form) >= agent.seq_len + 5 and self.full_price_df is not None:
                fp_a = fp_a.loc[common_form].where(fp_a.loc[common_form].pct_change().abs() <= 0.5).ffill().bfill()
                fp_b = fp_b.loc[common_form].where(fp_b.loc[common_form].pct_change().abs() <= 0.5).ffill().bfill()
                form_feat = self._prepare_features(fp_a, fp_b, hedge_ratio, log_mean_a, log_std_a, log_mean_b, log_std_b, form_spread_mean, form_spread_std)
                tail = form_feat.iloc[-agent.seq_len:]
                n_tail = len(tail)
                init_states = []
                for k in range(n_tail):
                    f_row = tail.iloc[k]
                    f_obs = np.array([
                        f_row['ZScore'], f_row['Rel_Return'], f_row['MA_Dist'],
                        (n_tail - k) / max(n_tail, 1),
                        f_row['Spread_Std'], 0.0, 0.0, f_row['Spread_Trend'],
                    ], dtype=np.float32)
                    init_states.append(f_obs)
                state_seq = deque(init_states, maxlen=agent.seq_len)
            else:
                state_seq = deque([np.zeros(STATE_DIM, dtype=np.float32)] * agent.seq_len, maxlen=agent.seq_len)
        except Exception:
            state_seq = deque([np.zeros(STATE_DIM, dtype=np.float32)] * agent.seq_len, maxlen=agent.seq_len)

        total_steps = len(dates_arr)

        for i in range(total_steps):
            date = dates_arr[i]
            row = feat_df.iloc[i]
            p_a, p_b = row['Price_A'], row['Price_B']
            z = row['ZScore']

            time_to_mat = (total_steps - i) / total_steps
            obs = np.array([
                z, row['Rel_Return'], row['MA_Dist'], time_to_mat,
                row['Spread_Std'], float(state.position),
                state.days_held / max(total_steps, 1), row['Spread_Trend'],
            ], dtype=np.float32)
            state_seq.append(obs)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(list(state_seq)).unsqueeze(0).to(agent.device)
                q_values = agent.model(state_tensor)
                action_idx = q_values.argmax().item()
            
            action = 0
            if action_idx == 1: action = 1
            elif action_idx == 2: action = -1
            
            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0
            current_status = "HOLD_CASH"
            
            if action != state.position:
                was_flat = (state.position == 0)
                # Close existing position
                if state.position != 0:
                    raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                    exit_fee = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                    closed_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee
                    state.realized_pnl += closed_trade_pnl
                    state.shares_a = 0.0
                    state.shares_b = 0.0
                    state.trade_entry_fee = 0.0
                    state.days_held = 0
                    current_status = "EXIT"
                    
                # Open new position
                if action != 0:
                    total_weight = 1.0 + abs(hedge_ratio)
                    v_a = self.capital_per_pair * (1.0 / total_weight)
                    v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)
                    
                    if action == 1:
                        state.shares_a = v_a / p_a
                        state.shares_b = -v_b / p_b
                        current_status = "ENTER_LONG_A" if was_flat else "REVERSE_LONG_A"
                    else:
                        state.shares_a = -v_a / p_a
                        state.shares_b = v_b / p_b
                        current_status = "ENTER_SHORT_A" if was_flat else "REVERSE_SHORT_A"
                        
                    state.entry_price_a = p_a
                    state.entry_price_b = p_b
                    state.trade_entry_fee = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                    
                state.position = action
            else:
                if state.position != 0:
                    state.days_held += 1
                    raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                    exit_fee = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate
                    unrealized_pnl = raw_unrealized - state.trade_entry_fee - exit_fee
                    current_status = "HOLDING"
                    
            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl
            
            out_dates.append(date)
            out_pa.append(round(p_a, 4))
            out_pb.append(round(p_b, 4))
            out_hr.append(round(float(hedge_ratio), 4))
            out_z.append(round(float(z), 4))
            out_pos.append(state.position)
            out_unrealized.append(round(float(unrealized_pnl), 4))
            out_realized.append(round(float(state.realized_pnl), 4))
            out_cum.append(round(float(cumulative_pnl), 4))
            out_status.append(current_status)
            out_trade_pnl.append(round(float(closed_trade_pnl), 4))
            out_days.append(state.days_held)
            out_delta.append(round(float(daily_delta), 4))
            
        # Force Close at end of period
        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "PERIOD_END_EXIT"):
                pnl_before_last_day = out_cum[-2] if len(out_cum) > 1 else 0.0
                
                p_a_last, p_b_last = out_pa[-1], out_pb[-1]
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
            "Log_Mean_A": log_mean_a, "Log_Std_A": log_std_a,
            "Log_Mean_B": log_mean_b, "Log_Std_B": log_std_b
        }
        for k, v in base_log.items():
            df_out[k] = v
            
        return df_out
