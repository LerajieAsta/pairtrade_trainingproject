# =======================================================
# 驗證單元：環境冒煙測試 (Smoke Test with Dummy Data)
# =======================================================
print('🧪 Starting environment sanity check...')

# 建立模擬數據
dates = pd.date_range('2023-01-01', periods=100)
dummy_df = pd.DataFrame({
    'price_a': 100 + np.cumsum(np.random.normal(0, 1, 100)),
    'price_b': 100 + np.cumsum(np.random.normal(0, 1, 100)),
    'z_score': np.random.normal(0, 1, 100),
    'vix': 20 + np.random.normal(0, 2, 100),
    'half_life': [20.0] * 100
}, index=dates)

# 實例化環境
env_test = PairsTradingEnv(data_df=dummy_df, beta=1.0)
obs, info = env_test.reset()
print(f'✅ Reset successful. Initial Observation: {obs}')

# 執行一個隨機動作並檢查 reward
action = 1 # Long Spread
next_obs, reward, terminated, truncated, info = env_test.step(action)
print(f'✅ Step successful. Reward: {reward:.4f}, Position: {info["current_position"]}')
print('🚀 Sanity check passed! Environment is ready for RL training.')
