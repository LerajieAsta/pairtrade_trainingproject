import os
import json
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

# 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering (即時刷屏)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

INITIAL_CAPITAL = 10000.0

# =======================================================================
# 智慧型自適應路徑追溯 (Adaptive Path Escalation)
# =======================================================================
results_dir = Path("results")
notebooks_dir = Path("notebooks")
tmp_dir = Path("tmp")

is_escalated = False
if not results_dir.exists() and Path("../results").exists():
    print("💡 Auto-Detection: Running inside tohtml/ folder. Path escalated to parent directory.")
    results_dir = Path("../results")
    notebooks_dir = Path("../notebooks")
    tmp_dir = Path("../tmp")
    is_escalated = True

db_path = results_dir / "result.db"

if not db_path.exists():
    print(f"❌ 找不到 SQLite 資料庫: {db_path}！請確認是否已執行回測並匯入資料庫。")
    exit(1)

print(f"🔍 檢測到可用資料庫: {db_path}。啟動【資料庫直連 - 雙通道全動態最優參數掃描與指標計算】...")

# 策略標籤定義 (動態表格展示用)
strategy_meta_info = {
    "SSD_Basic": {"label": "經典 SSD (Basic)", "color": "#4ade80"},
    "SSD_OLS": {"label": "進階 SSD (OLS) 🌟", "color": "#60a5fa"},
    "HDBSCAN_Handcrafted": {"label": "HDBSCAN (UMAP)", "color": "#fbd38d"},
    "HDBSCAN_MultiFactor": {"label": "HDBSCAN (MF)", "color": "#2b6cb0"},
    "HDBSCAN_PCA": {"label": "HDBSCAN (PCA)", "color": "#f87171"},
    "Pure_DTW": {"label": "純 DTW (Notebook)", "color": "#a855f7"}
}

# 資料庫 METHOD 與策略 KEY 的映射
db_method_to_key = {
    "SSD (Basic)": "SSD_Basic",
    "SSD": "SSD_OLS",
    "HDBSCAN (UMAP)": "HDBSCAN_Handcrafted",
    "HDBSCAN (MF)": "HDBSCAN_MultiFactor",
    "HDBSCAN (PCA)": "HDBSCAN_PCA",
    "Pure_DTW": "Pure_DTW"
}

def format_params(row):
    top_n = row["TOP N"]
    sl_pct = row["STOP LOSS %"]
    zwin = row["Z-WINDOW"]
    
    psl_val = str(row["PORT SL %"])
    psl = "無" if psl_val in ["0%", "0.0%", "0", "0.0"] else psl_val
    
    msr_val = str(row["MAX SEC %"])
    msr = "無" if msr_val in ["0%", "0.0%", "0", "0.0"] else msr_val
    
    dsz_val = str(row["DYN Z"])
    dsz = "無" if dsz_val in ["0", "0.0"] else dsz_val
    
    vol_val = row["VOL ADJ"]
    vol = "有" if vol_val == "VolAdj" else "無"
    
    return f"{top_n}, SL: {sl_pct}, ZWin: {zwin}, PSL: {psl}, MSR: {msr}, DSZ: {dsz}, VolAdj: {vol}"

# 儲存最終合併的數據與計算好的指標
merged_df = None
all_computed_metrics = {}

# 連接 SQLite 資料庫
conn = sqlite3.connect(db_path)

# 讀取所有的 strategy_summaries
summaries_df = pd.read_sql_query("SELECT * FROM strategy_summaries", conn)

# 我們只關心在資料庫中存在的 DATASET
available_datasets = summaries_df["DATASET"].unique()
print(f"資料庫中可用的資料集: {available_datasets}")

for dataset in available_datasets:
    # 統一將資料集名稱對應回小寫路徑 (如 Current -> current)
    dataset_lower = dataset.lower()
    print(f"\n📂 ==================== 正在由資料庫提取 {dataset.upper()} 資料集 ====================")
    
    dataset_summaries = summaries_df[summaries_df["DATASET"] == dataset]
    
    for db_method, key in db_method_to_key.items():
        method_df = dataset_summaries[dataset_summaries["METHOD"] == db_method]
        if method_df.empty:
            print(f"   ⚠️ 策略 {db_method} 在資料集 {dataset} 中無數據，跳過。")
            continue
            
        # 尋找 Final_Equity 最高的行
        best_row_idx = method_df["Final_Equity"].idxmax()
        best_row = method_df.loc[best_row_idx]
        
        best_path = best_row["_path"]
        best_equity = best_row["Final_Equity"]
        params_str = format_params(best_row)
        
        print(f"   🏆 {db_method} 最佳回測路徑: {best_path}")
        print(f"      -> 參數解構: {params_str} (最終淨值: ${best_equity:,.2f})")
        
        # 1. 自 trade_logs 載入最佳策略的 Daily_Delta 做時間序列合併
        daily_query = "SELECT Date, Daily_Delta FROM trade_logs WHERE strategy_id = ?"
        daily_df = pd.read_sql_query(daily_query, conn, params=(best_path,))
        
        if daily_df.empty:
            print(f"      ⚠️ 警告: trade_logs 中找不到此策略的每日交易明細，跳過淨值線合併。")
            continue
            
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        daily = daily_df.groupby("Date")["Daily_Delta"].sum().reset_index()
        daily = daily.sort_values("Date").reset_index(drop=True)
        
        col_name = f"{key}_{dataset_lower}"
        daily[col_name] = INITIAL_CAPITAL + daily["Daily_Delta"].cumsum()
        daily_equity = daily[["Date", col_name]]
        
        if merged_df is None:
            merged_df = daily_equity
        else:
            merged_df = pd.merge(merged_df, daily_equity, on="Date", how="outer")
            
        # 2. 收集指標以供動態注入 Notebook
        all_computed_metrics[(dataset_lower, key)] = {
            "metrics": {
                "final_equity": best_equity,
                "ann_ret": best_row["Ann_Ret_Raw"],
                "mdd": best_row["MDD_Raw"],
                "sharpe": best_row["Sharpe_Raw"],
                "rcc": best_row["RCC_Raw"],
                "rec": best_row["REC_Raw"]
            },
            "params_str": params_str
        }

conn.close()

# 填補缺失值
if merged_df is not None:
    merged_df = merged_df.sort_values("Date").ffill().fillna(INITIAL_CAPITAL)

# =======================================================================
# 1. 雙向同步寫入 - equity_curves.csv (Dual-Write CSV Sync)
# =======================================================================
primary_csv_path = notebooks_dir / "equity_curves.csv"
os.makedirs(primary_csv_path.parent, exist_ok=True)
merged_df.to_csv(primary_csv_path, index=False)
print(f"\n🎉 成功生成最優資產淨值數據！路徑: {primary_csv_path}")

if is_escalated:
    local_csv_path = Path("notebooks") / "equity_curves.csv"
    os.makedirs(local_csv_path.parent, exist_ok=True)
    merged_df.to_csv(local_csv_path, index=False)
    print(f"⚡ [雙向同步] 資料集同步複製至當前測試簡報路徑: {local_csv_path}")

# =======================================================================
# 2. 雙向同步寫入 - selected_dataset.txt (Dual-Write Temp File Sync)
# =======================================================================
# 預設使用第一個可用的 dataset 作為選中 dataset
active_dataset = "current" if "current" in [d.lower() for d in available_datasets] else "full"
primary_txt_path = tmp_dir / "selected_dataset.txt"
os.makedirs(primary_txt_path.parent, exist_ok=True)
with open(primary_txt_path, "w", encoding="utf-8") as temp_f:
    temp_f.write(active_dataset)

if is_escalated:
    local_txt_path = Path("tmp") / "selected_dataset.txt"
    os.makedirs(local_txt_path.parent, exist_ok=True)
    with open(local_txt_path, "w", encoding="utf-8") as temp_f:
        temp_f.write(active_dataset)
    print(f"⚡ [雙向同步] 暫存狀態檔同步複製至當前測試路徑: {local_txt_path}")

# =======================================================================
# 3. 雙向同步寫入 - analysis.ipynb 表格動態注入
# =======================================================================
def patch_notebook(nb_file):
    if not nb_file.exists():
        print(f"⚠️ 找不到 Notebook 檔案: {nb_file}，跳過。")
        return
        
    print(f"🛠️ 正在動態重構 {nb_file} 中的性能對比表格...")
    with open(nb_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    # 動態拼接最精美、100% 真實數據的 HTML 績效對比行 (tr)
    table_rows = []
    is_even = False
    
    # 遍歷資料集與策略
    for dataset_name in ["current", "full"]:
        for key, meta in strategy_meta_info.items():
            entry = all_computed_metrics.get((dataset_name, key))
            if not entry:
                continue
                
            metrics = entry["metrics"]
            params_str = entry["params_str"]
            
            ann_color = "#38a169" if metrics["ann_ret"] >= 0 else "#e53e3e"
            rcc_color = "#38a169" if metrics["rcc"] >= 0 else "#e53e3e"
            rec_color = "#38a169" if metrics["rec"] >= 0 else "#e53e3e"
            
            if key == "SSD_OLS":
                row_style = "background-color: #f0f7ff; font-weight: bold; border: 2px solid #3182ce;"
                text_color = "color: #2b6cb0;"
            else:
                row_style = "background-color: #f8fafc;" if is_even else "background-color: #ffffff;"
                text_color = ""
                
            is_even = not is_even
            
            tr_html = f"""    <tr style="{row_style}">
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0; font-weight: bold; text-align: left; {text_color}">{meta["label"]} ({dataset_name.upper()})</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0;">{params_str}</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0;">${metrics["final_equity"]:,.2f}</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0; color: {ann_color};">{metrics["ann_ret"]*100:+.2f}%</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0;">{metrics["mdd"]*100:.2f}%</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0;">{metrics["sharpe"]:.2f}</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0; color: {rcc_color};">{metrics["rcc"]*100:+.2f}%</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0; color: {rec_color};">{metrics["rec"]*100:+.2f}%</td>
    </tr>"""
            table_rows.append(tr_html)
            
    full_table_body = "\n".join(table_rows)
    
    dynamic_table_html = f"""## 📈 六大策略最優參數回測效能對比 (實時更新)

<table style="width: 100%; border-collapse: collapse; font-family: 'Inter', 'Outfit', sans-serif; font-size: 0.52em; margin: 10px auto; text-align: center; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
  <thead>
    <tr style="background-color: #1a365d; color: #ffffff; font-weight: 600; text-transform: uppercase;">
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">策略名稱 (Method)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">最佳參數組合 (Optimal Params)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">最終淨值 (Final Equity)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">年化報酬 (Ann. Return)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">最大回撤 (Max DD)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">夏普值 (Sharpe)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">RCC (%)</th>
      <th style="padding: 10px 8px; border: 1px solid #cbd5e1;">REC (%)</th>
    </tr>
  </thead>
  <tbody>
{full_table_body}
  </tbody>
</table>

> [!NOTE]
> **RCC (Return on Capital Constraint)**: 基於回測期分配總資金 $10,000 計算。
> **REC (Return on Engaged Capital)**: 基於實際動用且對齊 Beta 避險權重的保證金資金計算。
> *資料更新時間: 當前電腦編譯實時生成。
"""
    
    cell_patched = False
    for cell in nb_data["cells"]:
        if cell["cell_type"] == "markdown" and "六大策略最優參數回測效能對比" in "".join(cell["source"]):
            cell["source"] = [line + "\n" for line in dynamic_table_html.split("\n")]
            cell_patched = True
            break
            
    if cell_patched:
        with open(nb_file, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, ensure_ascii=False, indent=1)
        print(f"✅ 性能表格已成功動態改寫: {nb_file}")
    else:
        print(f"⚠️ Warning: 找不到性能對比表格 cell，無法自動改寫 {nb_file}")

# 同時改寫主要路徑與當前本機測試路徑的 Notebook
patch_notebook(notebooks_dir / "analysis.ipynb")
if is_escalated:
    patch_notebook(Path("notebooks") / "analysis.ipynb")

# =======================================================================
# 4. 雙向同步寫入 - default_name.txt (Dual-Write Default Name Sync)
# =======================================================================
import datetime
d = datetime.date.today()
minguo_date = f"{d.year-1911}{d.month:02d}{d.day:02d}"

primary_date_path = tmp_dir / "default_name.txt"
os.makedirs(primary_date_path.parent, exist_ok=True)
with open(primary_date_path, "w", encoding="utf-8") as df_f:
    df_f.write(minguo_date)

if is_escalated:
    local_date_path = Path("tmp") / "default_name.txt"
    os.makedirs(local_date_path.parent, exist_ok=True)
    with open(local_date_path, "w", encoding="utf-8") as df_f:
        df_f.write(minguo_date)
    print(f"⚡ [雙向同步] 預設檔名同步複製至當前測試路徑: {local_date_path}")
