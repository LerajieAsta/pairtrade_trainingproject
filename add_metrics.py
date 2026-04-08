import json

nb_path = 'd:/Unknown/Papper/Code/notebooks/pt_step3_1150408.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last code cell
last_cell = nb['cells'][-1]
src = "".join(last_cell['source'])

# The replacement targets
target_metrics = """    total_return = (results_df['Agent_NAV'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    
    print("\\n📊 【滾動梯隊資金管理 - 最終總結結算】 📊")"""

new_metrics = """    total_return = (results_df['Agent_NAV'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # --- 計算經濟學指標 (Economic Metrics) ---
    results_df['Daily_Return'] = results_df['Agent_NAV'].pct_change().fillna(0)
    trading_days = len(results_df)
    years = trading_days / 252.0
    
    # 1. 年化報酬率 (CAGR)
    annualized_return = ((results_df['Agent_NAV'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) if years > 0 else 0.0
    
    # 2. 年化波動率 (Annualized Volatility)
    annualized_volatility = results_df['Daily_Return'].std() * (252 ** 0.5)
    
    # 3. 年化夏普比率 (Sharpe Ratio, 假設 Risk-Free Rate = 0)
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0.0
    
    # 4. 卡瑪比率 (Calmar Ratio)
    mdd = results_df['Drawdown'].min()
    calmar_ratio = (annualized_return / abs(mdd)) if mdd < 0 else 0.0
    
    print("\\n📊 【滾動梯隊資金管理 - 最終總結結算】 📊")"""

target_print = """    print(f"📈 累計淨報酬率: {total_return * 100:.2f}%")
    print(f"📉 最大回撤 (MDD): {results_df['Drawdown'].min() * 100:.2f}%")"""

new_print = """    print(f"📈 累計淨報酬率: {total_return * 100:.2f}%")
    print(f"🌍 年化報酬率 (CAGR): {annualized_return * 100:.2f}%")
    print(f"🌪 年化波動率 (Volatility): {annualized_volatility * 100:.2f}%")
    print(f"📉 最大回撤 (MDD): {results_df['Drawdown'].min() * 100:.2f}%")
    print(f"⚖️ 年化夏普比率 (Sharpe Ratio): {sharpe_ratio:.2f}")
    print(f"🛡️ 卡瑪比率 (Calmar Ratio): {calmar_ratio:.2f}")"""

src = src.replace(target_metrics, new_metrics)
src = src.replace(target_print, new_print)

last_cell['source'] = [line + ("\\n" if i < len(src.split("\\n"))-1 else "") for i, line in enumerate(src.split("\\n"))]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("done")
