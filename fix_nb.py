import json
import re

nb_path = 'd:/Unknown/Papper/Code/notebooks/pt_step3_1150408.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modifying Cell 6: remove Hurst >= 0.5
src_6 = "".join(nb['cells'][6]['source'])
src_6 = src_6.replace(
    "features_df = features_df.dropna() # 清理計算中斷不穩定的死點",
    "features_df = features_df.dropna() # 清理計算中斷不穩定的死點\n    \n    # 1. 剔除 Hurst >= 0.5 的標的\n    features_df = features_df[features_df['Hurst'] < 0.5]"
)
nb['cells'][6]['source'] = [line + ('\n' if i < len(src_6.split('\n'))-1 else '') for i, line in enumerate(src_6.split('\n'))]

# Modifying Cell 7: half life, profit space, filter, sorting
src_7 = "".join(nb['cells'][7]['source'])

replacement1 = """            # 計算過零率 (均值回歸特徵)
            centered = spread - spread.mean()
            zero_crossings = ((centered.shift(1) * centered) < 0).sum()
            p['zero_cross'] = zero_crossings
            
            # 計算半衰期與獲利空間
            p['half_life'] = compute_half_life(spread.values)
            p['profit_space'] = spread.max() - spread.min()
        except:
            p['ssd'] = np.inf
            p['zero_cross'] = 0
            p['half_life'] = np.inf
            p['profit_space'] = 0"""

src_7 = src_7.replace(
"""            # 計算過零率 (均值回歸特徵)\n            centered = spread - spread.mean()\n            zero_crossings = ((centered.shift(1) * centered) < 0).sum()\n            p['zero_cross'] = zero_crossings\n        except:\n            p['ssd'] = np.inf\n            p['zero_cross'] = 0""",
replacement1
)

replacement2 = """    # 依照 SSD 排序並選出 Top-N
    df_passed = pd.DataFrame(passed)
    if df_passed.empty:
        return {n: [] for n in top_ns}
        
    # 2. 強行過濾零交叉跟半衰期
    try:
        FORMATION_WINDOW = 252 # 取自全域變數或預設值
    except:
        FORMATION_WINDOW = 252 
        
    df_passed = df_passed[~((df_passed['zero_cross'] < 12) & (df_passed['half_life'] > FORMATION_WINDOW / 2))]
    if df_passed.empty:
        return {n: [] for n in top_ns}
        
    # 3. 綜合評分指標：(獲利空間 / 交易次數)
    df_passed['trade_count'] = df_passed['zero_cross'].replace(0, 1)
    df_passed['profit_per_trade'] = df_passed['profit_space'] / df_passed['trade_count']
    
    df_passed['ssd_rank'] = df_passed['ssd'].rank(ascending=True)
    df_passed['profit_rank'] = df_passed['profit_per_trade'].rank(ascending=False)
    df_passed['final_score'] = df_passed['ssd_rank'] + df_passed['profit_rank']
    
    df_passed = df_passed.sort_values('final_score')"""

src_7 = src_7.replace(
"""    # 依照 SSD 排序並選出 Top-N\n    df_passed = pd.DataFrame(passed)\n    if df_passed.empty:\n        return {n: [] for n in top_ns}\n        \n    df_passed = df_passed.sort_values('ssd')""",
replacement2
)

nb['cells'][7]['source'] = [line + ('\n' if i < len(src_7.split('\n'))-1 else '') for i, line in enumerate(src_7.split('\n'))]

with open('d:/Unknown/Papper/Code/notebooks/pt_step3_1150408.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done!')
