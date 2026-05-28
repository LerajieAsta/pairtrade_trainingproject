import os
import json
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import concurrent.futures

# 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering (即時刷屏)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

INITIAL_CAPITAL = 10000.0

def get_csv_equity(csv_file):
    try:
        # 僅讀取 Daily_Delta 這欄以大幅節省記憶體與時間，並使用 float32 類型加速
        df_temp = pd.read_csv(csv_file, usecols=["Daily_Delta"], dtype={"Daily_Delta": np.float32})
        total_pnl = df_temp["Daily_Delta"].sum()
        final_equity = INITIAL_CAPITAL + total_pnl
        return csv_file, final_equity
    except Exception:
        return csv_file, -999999.0

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

print(f"🔍 檢測到可用資料集: {target_datasets}。啟動【雙通道全動態最優參數掃描與指標計算】...")

# 100% 精準復現底層指標計算函數 (無任何 Streamlit 依賴)
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

# 從 CSV 檔名中利用正則表達式動態提取最優參數結構的函數
def parse_params_from_filename(filename):
    filename_lower = filename.lower()
    
    # 1. 匹配 Top N
    match_top = re.search(r'top(\d+)', filename_lower)
    top_n = int(match_top.group(1)) if match_top else 5
    
    # 2. 匹配 Stop Loss %
    match_sl = re.search(r'sl(\d+)', filename_lower)
    sl_pct = f"{match_sl.group(1)}%" if match_sl else "0%"
    
    # 3. 匹配 Zscore Window
    match_zwin = re.search(r'zwin(\d+)', filename_lower)
    zwin = match_zwin.group(1) if match_zwin else "0"
    
    params_str = f"Top {top_n}, SL: {sl_pct}, ZWin: {zwin}"
    return top_n, params_str

# 策略標籤定義 (動態表格展示用)
strategy_meta_info = {
    "SSD_Basic": {"label": "經典 SSD (Basic)"},
    "SSD_OLS": {"label": "進階 SSD (OLS) 🌟"},
    "SSD_DL": {"label": "SSD + 深度學習交易 🚀"},
    "Engle_Granger": {"label": "Engle-Granger 共整合"},
    "HDBSCAN_Handcrafted": {"label": "HDBSCAN (UMAP)"},
    "HDBSCAN_MultiFactor": {"label": "HDBSCAN (MF)"}
}

# 儲存最終合併的數據與計算好的指標
merged_df = None
all_computed_metrics = {} # {(dataset, strategy_key): {"metrics": dict, "params_str": str}}

# 雙通道載入與動態指標計算
for dataset in target_datasets:
    print(f"\n📂 ==================== 正在全動態掃描 {dataset.upper()} 資料集 ====================")
    base_dir = results_dir / dataset
    
    # 建立一個 dict 來收集每個 strategy_key 對應到的所有 CSV 檔案
    strategy_csv_groups = {key: [] for key in strategy_meta_info.keys()}
    
    # 用 Path.glob 遞迴搜尋所有 *.csv 檔案
    all_csvs = list(base_dir.rglob("*.csv"))
    
    for csv_file in all_csvs:
        # 排除日誌與 summary 檔案
        if "logs" in csv_file.parts or csv_file.name.lower() == "summary.csv":
            continue
            
        # 智慧型判斷這份 CSV 屬於哪個策略
        path_str = str(csv_file).lower()
        filename_lower = csv_file.name.lower()
        combined = f"{path_str} {filename_lower}"
        
        matched_key = None
        if "ssd_basic" in combined:
            matched_key = "SSD_Basic"
        elif "ssd_dl" in combined:
            matched_key = "SSD_DL"
        elif "ssd" in combined:
            matched_key = "SSD_OLS"  # 進階 SSD (OLS)
        elif "eg_" in combined or "engle" in combined or "eg_no" in combined or "eg_re" in combined:
            matched_key = "Engle_Granger"
        elif "hdbscan" in combined:
            if "multi" in combined or "mf" in combined or "factor" in combined:
                matched_key = "HDBSCAN_MultiFactor"
            else:
                matched_key = "HDBSCAN_Handcrafted"
        
        if matched_key:
            strategy_csv_groups[matched_key].append(csv_file)
            
    # 遍歷六大策略，挑選出該策略底下 Final Equity 最優的 CSV 進行指標計算與資產曲線合併
    for key in strategy_meta_info.keys():
        csv_files = strategy_csv_groups[key]
        if not csv_files:
            print(f"   ⚠️ 策略 {key} 未在 results/{dataset}/ 底下掃描到符合的 CSV 檔案，跳過。")
            continue
            
        best_file = None
        best_equity = -999999.0
        
        # 使用 ThreadPoolExecutor 並行讀取 CSV，榨乾多核心效能！
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_pool = executor.map(get_csv_equity, csv_files)
            for csv_file, final_equity in results_pool:
                if final_equity > best_equity:
                    best_equity = final_equity
                    best_file = csv_file
                
        if not best_file:
            print(f"   ⚠️ 策略 {key} 下所有 CSV 讀取或計算失敗，跳過。")
            continue
            
        # 動態從最優 CSV 檔名中解構參數
        top_n, params_str = parse_params_from_filename(best_file.name)
        print(f"   🏆 {key} 最佳回測檔案: {best_file.relative_to(results_dir)}")
        print(f"      -> 參數解構: {params_str} (最終淨值: ${best_equity:,.2f})")
        
        # 1. 載入最佳 CSV 的 Daily_Delta 做時間序列合併
        df = pd.read_csv(best_file, usecols=["Date", "Daily_Delta"])
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
            
        # 2. 計算該策略在此電腦的動態量化指標
        metrics = compute_metrics(best_file, top_n)
        if metrics:
            all_computed_metrics[(dataset, key)] = {
                "metrics": metrics,
                "params_str": params_str
            }

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
