import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class PairsTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_df, transaction_cost=0.0029, stop_loss_pct=-0.05, beta=1.0, ablation_vix=False):
        super(PairsTradingEnv, self).__init__()
        
        self.df = data_df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        
        self.tc = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.beta = beta
        self.ablation_vix = ablation_vix
        
        self.action_space = spaces.Discrete(3)
        self.obs_dim = 5 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.current_step = 0
        self.current_pos = 0       
        self.entry_price_a = 0.0
        self.entry_price_b = 0.0
        self.total_reward = 0.0
        self.pnl_history = [0.0]
        
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_pos = 0
        self.entry_price_a = 0.0
        self.entry_price_b = 0.0
        self.total_reward = 0.0
        self.pnl_history = [0.0]
        
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        unrealized_pnl = 0.0
        if self.current_pos != 0 and self.entry_price_a > 0:
            current_pa, current_pb = row['price_a'], row['price_b']
            if not pd.isna(current_pa) and not pd.isna(current_pb):
                unrealized_pnl = self.current_pos * (
                    ((current_pa - self.entry_price_a)/self.entry_price_a - 
                    self.beta * (current_pb - self.entry_price_b)/self.entry_price_b) / (1 + self.beta)
                )
        
        vix_val = 0.0 if self.ablation_vix else (row['vix'] if not pd.isna(row['vix']) else 20.0)
        zscore = row['z_score'] if not pd.isna(row['z_score']) else 0.0
        hl = row['half_life'] if not pd.isna(row['half_life']) else 15.0
        
        obs = np.array([zscore, vix_val, hl, float(self.current_pos), unrealized_pnl], dtype=np.float32)
        return obs

    def step(self, action):
        target_pos = 0
        if action == 1: target_pos = 1
        elif action == 2: target_pos = -1

        row = self.df.iloc[self.current_step]
        current_pa = row['price_a']
        current_pb = row['price_b']
        
        # [Survivorship Bias Fix] Detect Delisting
        is_delisted = pd.isna(current_pa) or pd.isna(current_pb) or current_pa <= 0 or current_pb <= 0
        if is_delisted:
            target_pos = 0
            
        step_pnl = 0.0
        unrealized_pnl = 0.0
        
        if self.current_pos != 0 and self.current_step > 0:
            prev_row = self.df.iloc[self.current_step - 1]
            prev_pa, prev_pb = prev_row['price_a'], prev_row['price_b']
            
            if is_delisted:
                unrealized_pnl = self.stop_loss_pct * 2.0
                step_pnl = unrealized_pnl
            else:
                try:
                    step_pnl = self.current_pos * (((current_pa - prev_pa) / prev_pa - self.beta * (current_pb - prev_pb) / prev_pb) / (1 + self.beta))
                    unrealized_pnl = self.current_pos * (((current_pa - self.entry_price_a) / self.entry_price_a - self.beta * (current_pb - self.entry_price_b) / self.entry_price_b) / (1 + self.beta))
                except:
                    step_pnl, unrealized_pnl = 0.0, 0.0

        forced_stop_loss = False
        if self.current_pos != 0 and (unrealized_pnl <= self.stop_loss_pct or is_delisted):
            forced_stop_loss = True
            target_pos = 0  
            
        trade_executed = 0
        if target_pos != self.current_pos:
            trades_needed = abs(target_pos - self.current_pos)
            trade_penalty = trades_needed * self.tc
            step_pnl -= trade_penalty
            if target_pos != 0:
                trade_executed = 1
            
            if target_pos != 0 and not is_delisted:
                self.entry_price_a = current_pa
                self.entry_price_b = current_pb
            else:
                self.entry_price_a, self.entry_price_b, unrealized_pnl = 0.0, 0.0, 0.0

        self.current_pos = target_pos
        self.pnl_history.append(step_pnl)
        
        # 累積資產高峰追蹤
        self.cumulative_pnl += step_pnl
        if self.cumulative_pnl > self.peak_pnl:
            self.peak_pnl = self.cumulative_pnl
        current_drawdown = self.peak_pnl - self.cumulative_pnl
        
        if len(self.pnl_history) > 10:
            rolling_std = np.std(self.pnl_history[-20:]) + 1e-6
            reward = step_pnl / rolling_std
        else:
            reward = step_pnl
            
        # 1. MDD 平方懲罰 (壓制無腦抱單)
        if current_drawdown > 0.02:
            reward -= (current_drawdown ** 2) * 100.0 
            
        # 2. 強力停損扣分 (-5.0) 
        if forced_stop_loss:
            reward -= 5.0
            
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        info = {
            'step_pnl': step_pnl, 'unrealized_pnl': unrealized_pnl,
            'forced_stop_loss': forced_stop_loss, 'current_position': self.current_pos,
            'delisted': is_delisted, 'trade_executed': trade_executed
        }
        return self._get_obs(), float(np.clip(reward, -10.0, 10.0)), terminated, False, info

    def render(self, mode='human'):
        pass
