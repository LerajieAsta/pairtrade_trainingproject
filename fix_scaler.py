import json

nb_path = 'd:/Unknown/Papper/Code/notebooks/pt_step3_1150408.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The pipeline code is at nb['cells'][6]
cell = nb['cells'][6]
src = "".join(cell['source'])

# Replace target
target_logic = """    # 過濾出成功計算出特徵的欄位，防止錯誤
    valid_cols = [c for c in feature_columns if c in features_df.columns]
    if fit_mode:
        std_matrix = scaler.fit_transform(features_df[valid_cols])
    else:
        std_matrix = scaler.transform(features_df[valid_cols])"""

new_logic = """    # 過濾出成功計算出特徵的欄位，防止錯誤
    valid_cols = [c for c in feature_columns if c in features_df.columns]
    
    if features_df.empty:
        print("⚠️ 警告：經過 Hurst < 0.45 與 Half_life < 126 篩選後，剩餘 0 檔股票。跳過此梯隊。")
        # 造一個假 scaler 以免 caller 崩潰
        if fit_mode:
            import numpy as np
            scaler.fit(np.zeros((1, len(valid_cols)))) 
        return pd.DataFrame(columns=valid_cols), scaler

    if fit_mode:
        std_matrix = scaler.fit_transform(features_df[valid_cols])
    else:
        std_matrix = scaler.transform(features_df[valid_cols])"""

src = src.replace(target_logic, new_logic)
cell['source'] = [line + ("\n" if i < len(src.split("\n"))-1 else "") for i, line in enumerate(src.split("\n"))]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("done")
