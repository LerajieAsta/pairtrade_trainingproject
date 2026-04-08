import pandas as pd
import numpy as np
import statsmodels.api as sm
import gc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# =========================================================================
# 執行時間紀錄參數
# =========================================================================
import time
from datetime import datetime

# 1. 記錄開始時間
start_wall_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"🚀 訓練開始時間: {start_timestamp}")

# --- 你的 DRL 訓練核心代碼 ---
# 例如: 
# model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01, n_steps=2048, ...)
# model.learn(total_timesteps=1000000) 
# -------------------------

# 2. 記錄結束時間並計算耗時
end_wall_time = time.time()
end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
total_seconds = end_wall_time - start_wall_time

# 3. 轉換格式 (小時:分鐘:秒)
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)

# =========================================================================
# 預處理函數
# =========================================================================

def safe_half_life(ts):
    if len(ts) < 30: return 15.0
    z_lag = np.roll(ts, 1)
    z_lag[0] = 0
    z_ret = ts - z_lag
    z_ret[0] = 0
    z_lag2 = sm.add_constant(z_lag)
    try:
        res = sm.OLS(z_ret[1:], z_lag2[1:]).fit()
        hl = -np.log(2) / res.params[1]
        return hl if (hl > 0 and hl < 100) else 15.0
    except:
        return 15.0

def prepare_rl_features(df, beta=1.0, z_window=60):
    df = df.copy()
    df['spread'] = df['price_a'] - beta * df['price_b']
    
    roll_mean = df['spread'].rolling(window=z_window).mean()
    roll_std = df['spread'].rolling(window=z_window).std()
    df['z_score'] = (df['spread'] - roll_mean) / roll_std
    
    hl_list = []
    last_valid_hl = 15.0 
    
    for i in range(len(df)):
        if i < z_window:
            hl_list.append(last_valid_hl)
        elif i % 20 == 0: 
            window_ts = df['spread'].iloc[i-z_window:i].dropna().values
            current_hl = safe_half_life(window_ts)
            hl_list.append(current_hl)
            last_valid_hl = current_hl
        else:
            hl_list.append(hl_list[-1])
            
    df['half_life'] = hl_list
    return df.ffill().fillna(0)

# =========================================================================
# WALK-FORWARD OPTIMIZATION: 日級資產累加引擎 (Daily Portfolio Manager)
# =========================================================================

# 確保 panel有雙重對齊
price_panel_df = prices_raw.copy()
if 'date' not in price_panel_df.index.names:
    price_panel_df.set_index(['date', 'ticker'], inplace=True)

# 系統級參數設定
ROLLING_STEP = ROLLING_WINDOW
TRANCHE_ALLOCATION = INITIAL_CAPITAL / CAPITAL_TRANCHES

print(f"🚀 啟動多視窗整合回測引擎 (Daily Portfolio Manager)")
print(f"📦 總資金: ${INITIAL_CAPITAL} | 單注: ${TRANCHE_ALLOCATION} | 梯隊滾動: {ROLLING_STEP}天")

all_dates = price_pivot.index
total_length = len(all_dates)

cash = INITIAL_CAPITAL
active_tranches = []
completed_tranches_count = 0
trade_frequencies = []
historical_dates = []
historical_portfolio_nav = []

failed_due_to_margin = False

# 每日推演主時間軸
for t in range(FORMATION_WINDOW, total_length):
    current_date = all_dates[t]
    
    # 1. 每日掃描，回收過期梯隊資金
    still_active = []
    for tr in active_tranches:
        if current_date >= tr['end_date']:
            final_nav = tr['series'].iloc[-1]
            cash += final_nav
            completed_tranches_count += 1
        else:
            still_active.append(tr)
    active_tranches = still_active

    # --- [新增] 準確計算每日整個系統的總淨值 ---
    daily_total_nav = cash
    for tr in active_tranches:
        latest_val = tr['series'].loc[:current_date]
        if not latest_val.empty:
            daily_total_nav += latest_val.iloc[-1]
            
    # --- [新增] 全系統破產底線檢查 ---
    if daily_total_nav < 1000.0:
        print(f"❌💥 破產警報！系統總資產 (${daily_total_nav:,.2f}) 已低於 $1000 最低維運標準。交易強制永久終止。")
        failed_due_to_margin = True
        break

    # 2. 定期滾動啟動新梯隊
    if (t - FORMATION_WINDOW) % ROLLING_STEP == 0:
        
        # --- [重點修改] 動態提撥總資產的 10% 建倉 ---
        # TRANCHE_ALLOCATION = INITIAL_CAPITAL * 0.10
        
        if cash < TRANCHE_ALLOCATION:
            # 這不是破產，只是這天現金卡在別的部位裡。不停止整個系統，只是這 20 天休息不開新單。
            print(f"⚠️ 現金水位 (${cash:,.2f}) 暫時不足以提撥 10% 資金 (${TRANCHE_ALLOCATION:,.2f})。新梯隊輪空！")
        else:
            train_start_date = all_dates[t - FORMATION_WINDOW].date()
            
        train_start_date = all_dates[t - FORMATION_WINDOW].date()
        train_end_date = all_dates[t].date()
        test_start_date = all_dates[t].date()
        test_end_idx = min(t + TRADING_WINDOW, total_length - 1)
        test_end_date = all_dates[test_end_idx].date()
        
        print(f"\n--- 🔄 新梯隊啟動 | 系統剩餘現金: ${cash:.2f} | 預計執行區間: {test_start_date} ~ {test_end_date} ---")
        
        train_panel = price_panel_df.loc[str(train_start_date):str(train_end_date)]
        train_pivot = price_pivot.loc[str(train_start_date):str(train_end_date)]
        train_vix = vix_features.loc[str(train_start_date):str(train_end_date)]['VIX']
        
        train_features_std, train_scaler = feature_engineering_pipeline(train_panel, train_vix)
        top_pairs_dict = select_pairs_with_hdbscan(train_pivot, train_features_std, sector_map, top_ns=[MAX_PAIRS_PER_TRANCHE])
        top_pairs = top_pairs_dict.get(MAX_PAIRS_PER_TRANCHE, [])
        
        if not top_pairs:
            print("⚠️ 無高品質共整配對，放棄建倉，資金輪空不使用。")
        else:
            cash -= TRANCHE_ALLOCATION 
            actual_pairs = top_pairs[:MAX_PAIRS_PER_TRANCHE]
            sub_allocation = TRANCHE_ALLOCATION / len(actual_pairs)
            
            print(f"✅ 款項抵扣 ${TRANCHE_ALLOCATION}，準備對 {len(actual_pairs)} 組配對執行預演(每組分配: ${sub_allocation:.2f})")
            
            tranche_nav_records = None
            total_trades_count = 0
            
            for p_idx, pair in enumerate(actual_pairs):
                    
                stk_a, stk_b, beta_coef = pair['stock_a'], pair['stock_b'], pair['beta']
                print(f"  - 組合 {p_idx+1}: {stk_a} vs {stk_b} (Beta: {beta_coef:.4f})")
                
                rl_train_raw = pd.DataFrame({'price_a': train_pivot[stk_a], 'price_b': train_pivot[stk_b], 'vix': train_vix}).dropna()
                rl_train_df = prepare_rl_features(rl_train_raw, beta_coef)
                env_train = PairsTradingEnv(data_df=rl_train_df, transaction_cost=TRANSACTION_COST, beta=beta_coef)
                vec_env_train = DummyVecEnv([lambda: env_train])
                
                model = PPO('MlpPolicy', vec_env_train, verbose=0, learning_rate=0.0003, gamma=0.99)
                model.learn(total_timesteps=30000)
                
                ext_test_start = pd.to_datetime(test_start_date) - pd.Timedelta(days=120)
                test_pivot = price_pivot.loc[ext_test_start:str(test_end_date)]
                test_vix = vix_features.loc[ext_test_start:str(test_end_date)]['VIX']
                
                rl_test_raw = pd.DataFrame({'price_a': test_pivot.get(stk_a, pd.Series(dtype=float)), 'price_b': test_pivot.get(stk_b, pd.Series(dtype=float)), 'vix': test_vix})
                rl_test_full = prepare_rl_features(rl_test_raw, beta_coef)
                actual_test_df = rl_test_full[rl_test_full.index >= pd.to_datetime(test_start_date)].reset_index()
                
                if len(actual_test_df) >= 10:
                    env_test = PairsTradingEnv(data_df=actual_test_df, transaction_cost=TRANSACTION_COST, beta=beta_coef)
                    vec_env_test = DummyVecEnv([lambda: env_test])
                    
                    obs = vec_env_test.reset()
                    dones = [False]
                    
                    sub_nav = sub_allocation
                    sub_records = [sub_nav]
                    tranche_dates = actual_test_df['date'].tolist()
                    
                    pair_trades_count = 0
                    while not dones[0]:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, rewards, dones, infos = vec_env_test.step(action)
                        pair_trades_count += infos[0].get('trade_executed', 0)
                        total_trades_count += infos[0].get('trade_executed', 0)
                        
                        sub_nav += infos[0].get('step_pnl', 0.0) * sub_nav 
                        sub_records.append(sub_nav)
                            
                    if tranche_nav_records is None:
                        tranche_nav_records = np.array(sub_records)
                    else:
                        tranche_nav_records += np.array(sub_records)
                    
                        
                    
                    
                    pair_series = pd.Series(sub_records, index=tranche_dates)
                    pair_final_nav = pair_series.iloc[-1]
                    pair_ret = (pair_final_nav / sub_allocation) - 1.0
                    pair_peak = pair_series.cummax()
                    pair_mdd = ((pair_series - pair_peak) / pair_peak).min()
                    print(f"    ↳ [組合 {p_idx+1} 結算] 淨值: ${pair_final_nav:.2f} | 報酬率: {pair_ret*100:.2f}% | MDD: {pair_mdd*100:.2f}% | 交易次數: {pair_trades_count}")
                    
                    del env_test, vec_env_test
                
                    
                del env_train, vec_env_train, model
                gc.collect()
            
            if tranche_nav_records is not None:
                trade_frequencies.append(total_trades_count)
                daily_series = pd.Series(tranche_nav_records, index=tranche_dates)
                
                    
                # --- [介入點 A] ---
                final_nav = daily_series.iloc[-1]
                    
                ret = (final_nav / TRANCHE_ALLOCATION) - 1.0
                peak = daily_series.cummax()
                mdd = ((daily_series - peak) / peak).min()
                print(f"🎯 梯隊預演結算 | 最終淨值: ${final_nav:.2f} | 總報酬: {ret*100:.2f}% | MDD: {mdd*100:.2f}% | 交易次數: {total_trades_count}")
                
                active_tranches.append({'end_date': tranche_dates[-1], 'series': daily_series})

    # 3. 每日掃描：計算總資產 NAV_t
    historical_dates.append(current_date)
    historical_portfolio_nav.append(daily_total_nav)

print("\n🎉 回測迴圈結束！")
print(f"\n✅ 訓練結束時間: {end_timestamp}")
print(f"⏱️ 總共執行耗時: {hours} 小時 {minutes} 分 {seconds} 秒")

