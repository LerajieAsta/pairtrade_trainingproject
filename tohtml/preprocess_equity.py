import os
import json
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering (即時刷屏)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

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

INITIAL_CAPITAL = 10000.0

if not results_dir.exists():
    print("❌ 找不到 results 資料夾！請確認該電腦上是否存在回測輸出 results 目錄。")
    exit(1)

# 自動檢測 results 下可用的資料集子目錄
target_datasets = []
if (results_dir / "current").exists():
    target_datasets.append("current")
if (results_dir / "full").exists():
    target_datasets.append("full")

if not target_datasets:
    print("❌ 未在 results/ 下找到 'current' 或 'full' 資料夾，請確認回測路徑結構！")
    exit(1)

print(f"🔍 檢測到可用資料集: {target_datasets}。啟動【極速雙通道並行載入與動態指標計算】...")

# 100% 精準復現 dashboard.py 的底層指標計算函數 (無任何 Streamlit 依賴)
def compute_metrics(csv_path, top_n):
    try:
        df = pd.read_csv(csv_path, usecols=["Date", "Daily_Delta", "Position", "Ticker_A", "Ticker_B"])
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        
        # 每日盈虧分組
        portfolio_daily = df.groupby("Date")["Daily_Delta"].sum().reset_index()
        portfolio_daily = portfolio_daily.sort_values("Date").reset_index(drop=True)
        portfolio_daily['Cumulative_PnL'] = portfolio_daily['Daily_Delta'].cumsum()
        portfolio_daily['Equity'] = INITIAL_CAPITAL + portfolio_daily['Cumulative_PnL']
        
        final_equity = portfolio_daily['Equity'].iloc[-1] if len(portfolio_daily) > 0 else INITIAL_CAPITAL
        
        # A. 年化報酬
        monthly_equity = portfolio_daily.set_index('Date')['Equity'].resample('ME').last().dropna()
        n_months = len(monthly_equity)
        cum_ret = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        ann_ret = ((final_equity / INITIAL_CAPITAL) ** (12 / n_months)) - 1 if n_months > 0 else 0
        
        # B. 夏普值
        daily_returns = portfolio_daily['Daily_Delta'] / INITIAL_CAPITAL
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        # C. 最大回撤
        roll_max = portfolio_daily['Cumulative_PnL'].cummax()
        drawdown = portfolio_daily['Cumulative_PnL'] - roll_max
        mdd_pct = drawdown.min() / INITIAL_CAPITAL
        
        # D. RCC
        rcc = cum_ret
        
        # E. REC
        if 'Position' in df.columns and 'Ticker_A' in df.columns:
            n_traded = len(df[df['Position'] != 0].drop_duplicates(subset=['Ticker_A', 'Ticker_B']))
        else:
            n_traded = 0
        c_pair = INITIAL_CAPITAL / top_n if top_n > 0 else INITIAL_CAPITAL
        engaged_capital = n_traded * c_pair
        rec = (final_equity - INITIAL_CAPITAL) / engaged_capital if engaged_capital > 0 else 0
        
        return {
            "final_equity": final_equity,
            "ann_ret": ann_ret,
            "mdd": mdd_pct,
            "sharpe": sharpe,
            "rcc": rcc,
            "rec": rec
        }
    except Exception as e:
        print(f"   ⚠️ 指標計算失敗 ({csv_path.name}): {e}")
        return None

# 儲存最終合併的數據與計算好的指標
merged_df = None
all_computed_metrics = {}

# 最佳參數組合映射
optimal_params_map = {
    "SSD_Basic": {"file": "TradeLogs_Top5_SL0_ZWin0.csv", "top_n": 5, "label": "經典 SSD (Basic)", "params_str": "Top 5, SL: 0%, ZWin: 0"},
    "SSD_OLS": {"file": "TradeLogs_Top10_SL0_ZWin0.csv", "top_n": 10, "label": "進階 SSD (OLS) 🌟", "params_str": "Top 10, SL: 0%, ZWin: 0"},
    "SSD_DL": {"file": "TradeLogs_Top5_SL0_ZWin0.csv", "top_n": 5, "label": "SSD + 深度學習交易 🚀", "params_str": "Top 5, SL: 0%, ZWin: 0"},
    "Engle_Granger": {"file": "EG_TradeLogs_Top10_SL5.csv", "top_n": 10, "label": "Engle-Granger 共整合", "params_str": "Top 10, SL: 5%, ZWin: 0"},
    "HDBSCAN_Handcrafted": {"file": "HDBSCAN_UMAP_TradeLogs_Top5_SL0_ZWin0_VolAdj.csv", "top_n": 5, "label": "HDBSCAN (UMAP)", "params_str": "Top 5, SL: 0%, ZWin: 0"},
    "HDBSCAN_Autoencoder": {"file": "HDBSCAN_AE_UMAP_TradeLogs_Top5_SL0_ZWin0_VolAdj.csv", "top_n": 5, "label": "HDBSCAN (AE UMAP)", "params_str": "Top 5, SL: 0%, ZWin: 0"},
    "HDBSCAN_MultiFactor": {"file": "HDBSCAN_MULTIFACTOR_TradeLogs_Top20_SL0_ZWin0_VolAdj.csv", "top_n": 20, "label": "HDBSCAN (MF)", "params_str": "Top 20, SL: 0%, ZWin: 0"}
}

# 雙通道載入與動態指標計算
for dataset in target_datasets:
    print(f"\n📂 ==================== 正在處理 {dataset.upper()} 資料集 ====================")
    base_dir = results_dir / dataset
    
    # 策略目錄映射
    folders = {
        "SSD_Basic": base_dir / "SSD_Basic_ReEntry",
        "SSD_OLS": base_dir / "SSD_ReEntry",
        "SSD_DL": base_dir / "SSD_DL_ReEntry",
        "Engle_Granger": base_dir / "EG_NoReEntry",
        "HDBSCAN_Handcrafted": base_dir / "HDBSCAN_UMAP_ReEntry",
        "HDBSCAN_Autoencoder": base_dir / "HDBSCAN_AE_UMAP_ReEntry",
        "HDBSCAN_MultiFactor": base_dir / "HDBSCAN_MultiFactor_ReEntry"
    }
    
    for key, opt in optimal_params_map.items():
        path = folders[key] / opt["file"]
        if not path.exists():
            print(f"   ⚠️ 跳過: 找不到 {key} ({opt['file']})")
            continue
            
        print(f"   🔄 正在合併並計算指標: {key} ({opt['file']})...")
        
        df = pd.read_csv(path, usecols=["Date", "Daily_Delta"])
        df["Date"] = pd.to_datetime(df["Date"])
        daily = df.groupby("Date")["Daily_Delta"].sum().reset_index()
        daily = daily.sort_values("Date").reset_index(drop=True)
        
        col_name = f"{key}_{dataset}"
        daily[col_name] = INITIAL_CAPITAL + daily["Daily_Delta"].cumsum()
        daily_equity = daily[["Date", col_name]]
        
        if merged_df is None:
            merged_df = daily_equity
        else:
            merged_df = pd.merge(merged_df, daily_equity, on="Date", how="outer")
            
        # 計算該策略在此電腦的動態量化指標！
        metrics = compute_metrics(path, opt["top_n"])
        if metrics:
            all_computed_metrics[(dataset, key)] = metrics

# 填補缺失值
if merged_df is not None:
    merged_df = merged_df.sort_values("Date").ffill().fillna(INITIAL_CAPITAL)

# =======================================================================
# 1. 雙向同步寫入 - equity_curves.csv (Dual-Write CSV Sync)
# =======================================================================
primary_csv_path = notebooks_dir / "equity_curves.csv"
os.makedirs(primary_csv_path.parent, exist_ok=True)
merged_df.to_csv(primary_csv_path, index=False)
print(f"\n🎉 主要數據儲存成功！路徑: {primary_csv_path}")

if is_escalated:
    local_csv_path = Path("notebooks") / "equity_curves.csv"
    os.makedirs(local_csv_path.parent, exist_ok=True)
    merged_df.to_csv(local_csv_path, index=False)
    print(f"⚡ [雙向同步] 資料集同步複製至當前測試簡報路徑: {local_csv_path}")

# =======================================================================
# 2. 雙向同步寫入 - selected_dataset.txt (Dual-Write Temp File Sync)
# =======================================================================
primary_txt_path = tmp_dir / "selected_dataset.txt"
os.makedirs(primary_txt_path.parent, exist_ok=True)
with open(primary_txt_path, "w", encoding="utf-8") as temp_f:
    temp_f.write(dataset)

if is_escalated:
    local_txt_path = Path("tmp") / "selected_dataset.txt"
    os.makedirs(local_txt_path.parent, exist_ok=True)
    with open(local_txt_path, "w", encoding="utf-8") as temp_f:
        temp_f.write(dataset)
    print(f"⚡ [雙向同步] 暫存狀態檔同步複製至當前測試路徑: {local_txt_path}")

# =======================================================================
# 3. 雙向同步寫入 - analysis.ipynb 表格動態注入
# =======================================================================
def patch_notebook(nb_file):
    if not nb_file.exists():
        return
        
    print(f"🛠️ 正在動態重構 {nb_file} 中的性能對比表格...")
    with open(nb_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    # 動態拼接最精美、100% 真實數據的 HTML 績效對比行 (tr)
    table_rows = []
    is_even = False
    
    for dataset_name in target_datasets:
        for key, opt in optimal_params_map.items():
            metrics = all_computed_metrics.get((dataset_name, key))
            if not metrics:
                continue
                
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
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0; font-weight: bold; text-align: left; {text_color}">{opt["label"]} ({dataset_name.upper()})</td>
      <td style="padding: 8px 6px; border: 1px solid #e2e8f0;">{opt["params_str"]}</td>
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
