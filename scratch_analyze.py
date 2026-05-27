import os
import re
import sys
import pandas as pd
import numpy as np
import concurrent.futures

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# 設置路徑
RESULTS_DIR = "results"
INITIAL_CAPITAL = 10000.0

def scan_strategies(base_dir):
    strategies = []
    if not os.path.exists(base_dir): return strategies
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "TradeLogs" in file and "HDBSCAN" in root:
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                strategies.append(rel_path.replace("\\", "/"))
    return sorted(strategies)

def load_data(strategy_path):
    file_path = os.path.join(RESULTS_DIR, strategy_path)
    if not os.path.exists(file_path): return pd.DataFrame()
    
    try:
        # 只讀取需要的欄位
        target_cols = ['Date', 'Position', 'Ticker_A', 'Ticker_B', 'Daily_Delta', 'Status']
        
        sample = pd.read_csv(file_path, nrows=0)
        original_header = sample.columns.tolist()
        clean_header = [str(c).strip() for c in original_header]
        
        col_map = {orig: clean for orig, clean in zip(original_header, clean_header) if clean in target_cols}
        cols_to_use_orig = list(col_map.keys())
        
        dtypes_to_use = {}
        for orig, clean in col_map.items():
            if clean == 'Position':
                dtypes_to_use[orig] = 'float32'
            elif clean == 'Daily_Delta':
                dtypes_to_use[orig] = 'float32'
            elif clean == 'Status':
                dtypes_to_use[orig] = 'category'
                
        df = pd.read_csv(file_path, usecols=cols_to_use_orig, dtype=dtypes_to_use)
        df.rename(columns=col_map, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        if 'Position' in df.columns:
            df['Position'] = df['Position'].fillna(0).astype(int)
        
        if 'Ticker_A' in df.columns and 'Ticker_B' in df.columns:
            df['Ticker_A'] = df['Ticker_A'].fillna('UNKNOWN').astype(str)
            df['Ticker_B'] = df['Ticker_B'].fillna('UNKNOWN').astype(str)
            df = df.sort_values(by=['Ticker_A', 'Ticker_B', 'Date']).reset_index(drop=True)
            
        return df
    except Exception as e:
        print(f"Error loading {strategy_path}: {str(e)}")
        return pd.DataFrame()

def extract_features_from_path(path):
    path_lower = path.lower()
    dataset = "Full" if "full" in path_lower else "Current" if "current" in path_lower else "Unknown"
    reentry = "NoReEntry" if "noreentry" in path_lower else "ReEntry" if "reentry" in path_lower else "Unknown"
    
    is_mf = "multifactor" in path_lower
    is_ae = "_ae_" in path_lower or "hdbscan_ae" in path_lower
    is_pca = "_pca_" in path_lower or "hdbscan_pca" in path_lower
    
    if is_mf:
        method = "HDBSCAN (MultiFactor)"
    elif is_ae:
        method = "HDBSCAN (AE PCA)" if is_pca else "HDBSCAN (AE UMAP)"
    else:
        method = "HDBSCAN (PCA)" if is_pca else "HDBSCAN (UMAP)"

    top_n = 20
    match_n = re.search(r'top(\d+)', path_lower)
    if match_n: top_n = int(match_n.group(1))
        
    sl_pct = 0.0
    match_sl = re.search(r'sl(\d+)', path_lower)
    if match_sl: sl_pct = float(match_sl.group(1)) / 100.0
        
    zwin = 0
    match_zwin = re.search(r'zwin(\d+)', path_lower)
    if match_zwin: zwin = int(match_zwin.group(1))
        
    vol_adj = "VolAdj" if "voladj" in path_lower and "novoladj" not in path_lower else "NoVolAdj"
        
    return dataset, reentry, method, top_n, sl_pct, zwin, vol_adj

def calculate_metrics_raw(strategy_path):
    df = load_data(strategy_path)
    if df.empty: return None

    dataset, reentry, method, top_n, sl_pct, zwin, vol_adj = extract_features_from_path(strategy_path)
    
    if 'Daily_Delta' in df.columns:
        portfolio_daily = df.groupby('Date')['Daily_Delta'].sum().reset_index()
        portfolio_daily = portfolio_daily.sort_values('Date').reset_index(drop=True)
    else:
        portfolio_daily = pd.DataFrame({'Date': df['Date'].unique(), 'Daily_Delta': 0})
        
    portfolio_daily['Cumulative_PnL'] = portfolio_daily['Daily_Delta'].cumsum()
    portfolio_daily['Equity'] = INITIAL_CAPITAL + portfolio_daily['Cumulative_PnL']
    
    final_pnl = portfolio_daily['Cumulative_PnL'].iloc[-1] if not portfolio_daily.empty else 0
    final_equity = INITIAL_CAPITAL + final_pnl
    
    if len(portfolio_daily) > 0:
        portfolio_daily_idx = portfolio_daily.set_index('Date')
        monthly_equity = portfolio_daily_idx['Equity'].resample('ME').last().dropna()
        
        if len(monthly_equity) > 0:
            monthly_returns = monthly_equity.pct_change().fillna(0)
            cum_ret = np.prod(1 + monthly_returns) - 1
            n_months = len(monthly_returns)
            ann_ret = ((1 + cum_ret) ** (12 / n_months)) - 1 if n_months > 0 else 0
        else:
            cum_ret = ann_ret = 0
            
        daily_returns = portfolio_daily_idx['Daily_Delta'] / INITIAL_CAPITAL
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        roll_max = portfolio_daily['Cumulative_PnL'].cummax()
        drawdown = portfolio_daily['Cumulative_PnL'] - roll_max
        mdd_pct = drawdown.min() / INITIAL_CAPITAL
    else:
        cum_ret = ann_ret = sharpe = mdd_pct = 0

    win_rate = 0.0
    profit_factor = 0.0
    total_trades = 0
    winning_trades = 0
    
    if 'Position' in df.columns and 'Ticker_A' in df.columns:
        df['Prev_Pos'] = df.groupby(['Ticker_A', 'Ticker_B'])['Position'].shift(1).fillna(0)
        
        direction_change = df['Position'] != df['Prev_Pos']
        exit_mask = direction_change & (df['Prev_Pos'] != 0)
        n_exits_total = exit_mask.sum()
        
        last_rows = df.groupby(['Ticker_A', 'Ticker_B']).tail(1)
        n_forced_close = (last_rows['Position'] != 0).sum()
        n_entries = n_exits_total + n_forced_close
        
        if 'Status' in df.columns:
            n_stop_loss = df[exit_mask]['Status'].astype(str).str.contains('stop|sl|停損', case=False, na=False).sum()
        else:
            n_stop_loss = -1
            
        if 'Daily_Delta' in df.columns:
            state_change = df['Position'] != df['Prev_Pos']
            df['State_ID'] = state_change.groupby([df['Ticker_A'], df['Ticker_B']]).cumsum()
            df['Prev_State_ID'] = df.groupby(['Ticker_A', 'Ticker_B'])['State_ID'].shift(1).fillna(0)
            active_mask = (df['Prev_Pos'] != 0) | (df['Daily_Delta'] != 0)
            
            if active_mask.any():
                trade_pnls = df[active_mask].groupby(['Ticker_A', 'Ticker_B', 'Prev_State_ID'])['Daily_Delta'].sum()
                gross_profit = float(trade_pnls[trade_pnls > 0].sum())
                gross_loss = float(trade_pnls[trade_pnls < 0].sum())
                total_trades = len(trade_pnls)
                winning_trades = len(trade_pnls[trade_pnls > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else (gross_profit if gross_profit > 0 else 0.0)
            else:
                gross_profit = gross_loss = 0.0
        else:
            gross_profit = gross_loss = 0.0
    else:
        n_entries = n_stop_loss = 0
        gross_profit = gross_loss = 0.0

    return {
        'Dataset': dataset,
        'ReEntry': reentry,
        'Method': method,
        'Top_N': top_n,
        'Stop_Loss': sl_pct,
        'Z_Window': zwin,
        'Vol_Adjust': vol_adj,
        'Final_Equity': final_equity,
        'Cum_Ret': cum_ret,
        'Ann_Ret': ann_ret,
        'Sharpe': sharpe,
        'MDD': mdd_pct,
        'Entries': n_entries,
        'Stop_Losses': n_stop_loss,
        'Gross_Profit': gross_profit,
        'Gross_Loss': gross_loss,
        'Total_Trades': total_trades,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        '_path': strategy_path
    }

def process_single(s):
    try:
        return calculate_metrics_raw(s)
    except Exception as e:
        print(f"處理 {s} 發生錯誤: {e}")
        return None

def main():
    print("正在掃描 HDBSCAN 交易日誌...")
    strategies = scan_strategies(RESULTS_DIR)
    total_tasks = len(strategies)
    print(f"找到 {total_tasks} 個 HDBSCAN 策略參數組合。啟動平行分析...")
    
    records = []
    
    max_workers = min(os.cpu_count() or 4, 12)
    print(f"並行工作行程數 (Workers): {max_workers}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, s): (i, s) for i, s in enumerate(strategies)}
        
        for future in concurrent.futures.as_completed(futures):
            i, s = futures[future]
            res = future.result()
            if res:
                records.append(res)
            print(f"[{len(records)}/{total_tasks}] 已完成：{s.split('/')[-1]}")
            
    df = pd.DataFrame(records)
    
    if df.empty:
        print("沒有成功解析任何績效數據！")
        return
        
    df.to_csv("scratch/hdbscan_metrics_summary.csv", index=False)
    print("已將所有明細存檔至 scratch/hdbscan_metrics_summary.csv")
    
    # 進行分組統計並印出
    print("\n" + "="*80)
    print("                 🏆 各 HDBSCAN 策略之最優參數組合 (依 Sharpe Ratio 排序) 🏆")
    print("="*80)
    
    for (dataset, method), group in df.groupby(['Dataset', 'Method']):
        print(f"\n📁 數據集: {dataset} | 策略: {method}")
        best_row = group.loc[group['Sharpe'].idxmax()]
        print(f"  🔥 最佳 Sharpe: {best_row['Sharpe']:.2f}")
        print(f"  📈 年化報酬率 : {best_row['Ann_Ret']:.2%}")
        print(f"  📉 最大回撤   : {best_row['MDD']:.2%}")
        print(f"  🛠️ 最佳參數組合: Top_N={best_row['Top_N']}, SL={best_row['Stop_Loss']:.0%}, Z_Window={best_row['Z_Window']}, Vol_Adjust={best_row['Vol_Adjust']}")
        print(f"  📊 交易次數   : {best_row['Total_Trades']} | 勝率: {best_row['Win_Rate']:.2%} | 獲利因子: {best_row['Profit_Factor']:.2f}")

    # 橫向對比 HDBSCAN 策略的平均表現 (Current 數據集)
    print("\n" + "="*80)
    print("             📊 所有 HDBSCAN 策略平均表現對比 (Current 數據集) 📊")
    print("="*80)
    current_df = df[df['Dataset'] == 'Current']
    if not current_df.empty:
        method_stats = current_df.groupby('Method').agg({
            'Sharpe': 'mean',
            'Ann_Ret': 'mean',
            'MDD': 'mean',
            'Total_Trades': 'mean',
            'Win_Rate': 'mean',
            'Profit_Factor': 'mean'
        }).rename(columns={
            'Sharpe': '平均 Sharpe',
            'Ann_Ret': '平均年化報酬',
            'MDD': '平均最大回撤',
            'Total_Trades': '平均交易次數',
            'Win_Rate': '平均勝率',
            'Profit_Factor': '平均獲利因子'
        })
        print(method_stats.to_string())

    # 波動度調節 (Vol_Adjust) 的效益分析
    print("\n" + "="*80)
    print("          📈 波動度自適應調節 (Vol_Adjust) 效益分析 (Current) 📈")
    print("="*80)
    if not current_df.empty:
        vol_stats = current_df.groupby('Vol_Adjust').agg({
            'Sharpe': 'mean',
            'Ann_Ret': 'mean',
            'MDD': 'mean',
            'Win_Rate': 'mean',
            'Profit_Factor': 'mean'
        }).rename(columns={
            'Sharpe': '平均 Sharpe',
            'Ann_Ret': '平均年化報酬',
            'MDD': '平均最大回撤',
            'Win_Rate': '平均勝率',
            'Profit_Factor': '平均獲利因子'
        })
        print(vol_stats.to_string())

    # 分析參數對績效的影響
    print("\n" + "="*80)
    print("          ⚙️ 參數靈敏度分析 (所有 HDBSCAN 策略在 Current 的平均值) ⚙️")
    print("="*80)
    if not current_df.empty:
        print("\n1. 停損比例 Stop Loss 的影響:")
        print(current_df.groupby('Stop_Loss')[['Sharpe', 'Ann_Ret', 'MDD', 'Win_Rate']].mean().to_string())
        
        print("\n2. Z-Score 窗口大小 Z_Window 的影響:")
        print(current_df.groupby('Z_Window')[['Sharpe', 'Ann_Ret', 'MDD', 'Win_Rate']].mean().to_string())
        
        print("\n3. 最優配對組數 Top N 的影響:")
        print(current_df.groupby('Top_N')[['Sharpe', 'Ann_Ret', 'MDD', 'Win_Rate']].mean().to_string())

if __name__ == "__main__":
    os.makedirs("scratch", exist_ok=True)
    main()
