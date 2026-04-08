import json

nb_path = 'd:/Unknown/Papper/Code/notebooks/pt_step3_1150408.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The pipeline code is at nb['cells'][10], wait earlier it was index 10 (Cell 10) because I inserted markdown blocks!
# Wait! In nb['cells'], index 0 is markdown, index 1 is c0, index 5 is markdown, index 6 is c4 (where feature_engineering_pipeline lives)
cell = nb['cells'][6]
src = "".join(cell['source'])

# The replacement targets
target_logic = """    if features_df.empty:
        print("⚠️ 警告：經過 Hurst < 0.45 與 Half_life < 126 篩選後，剩餘 0 檔股票。跳過此梯隊。")
        # 造一個假 scaler 以免 caller 崩潰
        if fit_mode:
            import numpy as np
            scaler.fit(np.zeros((1, len(valid_cols)))) 
        return pd.DataFrame(columns=valid_cols), scaler"""

new_logic = """    if features_df.empty:
        print("⚠️ 警告：經過 Hurst < 0.45 與 Half_life < 126 篩選後，剩餘 0 檔股票。跳過此梯隊。")
        # 造一個假 scaler 以免 caller 崩潰
        if fit_mode:
            # np is already imported globally
            scaler.fit(np.zeros((1, len(valid_cols)))) 
        return pd.DataFrame(columns=valid_cols), scaler"""

src = src.replace(target_logic, new_logic)
cell['source'] = [line + ("\n" if i < len(src.split("\n"))-1 else "") for i, line in enumerate(src.split("\n"))]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("done")
