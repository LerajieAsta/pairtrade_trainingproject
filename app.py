import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pairs Trading Comparison", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS FOR DARK THEME AESTHETICS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    /* Force dark theme aesthetics */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117 !important;
        color: #f8fafc !important;
    }
    
    /* Header Section */
    .top-label {
        color: #60a5fa; /* Lighter blue for dark mode */
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .main-title {
        font-size: 42px;
        font-weight: 900;
        color: #f1f5f9;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 15px;
    }
    
    /* Displayed Rows Badge */
    .row-badge {
        background-color: #1e293b;
        color: white;
        border-radius: 12px;
        padding: 15px 30px;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .row-badge-title { font-size: 13px; color: #94a3b8; margin-bottom: 4px; }
    .row-badge-val { font-size: 36px; font-weight: 900; line-height: 1; color: #f8fafc; }
    
    /* Metric Cards */
    .metric-card {
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    /* Dark mode card variants */
    .card-green { border: 1px solid #16a34a; background-color: rgba(22, 163, 74, 0.05); }
    .card-blue { border: 1px solid #2563eb; background-color: rgba(37, 99, 235, 0.05); }
    .card-purple { border: 1px solid #9333ea; background-color: rgba(147, 51, 234, 0.05); }
    .card-orange { border: 1px solid #ea580c; background-color: rgba(234, 88, 12, 0.05); }
    .card-red { border: 1px solid #dc2626; background-color: rgba(220, 38, 38, 0.05); }
    
    .card-title {
        font-size: 13px;
        font-weight: 700;
        color: #cbd5e1;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .card-value {
        font-size: 38px;
        font-weight: 900;
        margin-bottom: 8px;
        line-height: 1;
    }
    /* Brighter colors for text on dark background */
    .val-green { color: #4ade80; }
    .val-blue { color: #60a5fa; }
    .val-purple { color: #c084fc; }
    .val-orange { color: #fb923c; }
    .val-red { color: #f87171; }
    
    .card-desc {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 400;
    }

    /* DataFrame Styling Tweaks */
    [data-testid="stDataFrame"] {
        margin-top: -10px;
    }
    
    [data-testid="stMetricValue"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CONSTANTS & CONFIG
# ==========================================
RESULTS_DIR = "results"
INITIAL_CAPITAL = 10000.0

@st.cache_data
def scan_strategies(base_dir):
    """Scan directory recursively to find all strategies."""
    strategies = []
    if not os.path.exists(base_dir):
        return strategies
    for root, dirs, files in os.walk(base_dir):
        if "detailed_trade_logs.csv" in files:
            rel_path = os.path.relpath(root, base_dir)
            strategies.append(rel_path.replace("\\", "/"))
    return sorted(strategies)

@st.cache_data
def load_data(strategy_path):
    """Load CSV with strict downcasting for memory optimization."""
    file_path = os.path.join(RESULTS_DIR, strategy_path, "detailed_trade_logs.csv")
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    dtypes = {
        'Position': 'int8',
        'Daily_Delta': 'float32',
        'Trade_PnL': 'float32',
    }
    # Load necessary columns for logic
    cols_to_use = ['Date', 'Position', 'Ticker_A', 'Ticker_B', 'Daily_Delta', 'Trade_PnL']
    
    try:
        df = pd.read_csv(file_path, usecols=cols_to_use, dtype=dtypes, parse_dates=['Date'])
        return df
    except Exception:
        df = pd.read_csv(file_path, dtype=dtypes, parse_dates=['Date'])
        return df


# ==========================================
# METRICS & AGGREGATION LOGIC
# ==========================================
def extract_features_from_path(strategy_path):
    """Parse the directory path to extract features."""
    method = "-"
    top_n = "Top 20"
    sl_pct = "5%"
    zwin = "0"
    
    path_lower = strategy_path.lower()
    if "eg" in path_lower: method = "EG"
    elif "ssd" in path_lower: method = "SSD"
    
    match_top = re.search(r'top(\d+)', strategy_path, re.IGNORECASE)
    if match_top: top_n = f"Top {match_top.group(1)}"
        
    match_sl = re.search(r'sl(\d+)', strategy_path, re.IGNORECASE)
    if match_sl: sl_pct = f"{match_sl.group(1)}%"
        
    match_zwin = re.search(r'zwin(\d+)', strategy_path, re.IGNORECASE)
    if match_zwin: zwin = f"{match_zwin.group(1)}"
        
    # 新增分類解析邏輯
    dataset = "Full" if "full" in path_lower else "Current" if "current" in path_lower else "-"
    sector = "NoSector" if "nosector" in path_lower else "Sector" if "sector" in path_lower else "-"
    reentry = "NoReEntry" if "noreentry" in path_lower else "ReEntry" if "reentry" in path_lower else "-"
        
    return dataset, sector, reentry, method, top_n, sl_pct, zwin

def calculate_metrics_raw(df, strategy_path):
    """Calculate RCC, REC, and other metrics based on formulas."""
    dataset, sector, reentry, method, top_n, sl_pct, zwin = extract_features_from_path(strategy_path)
    
    if df.empty:
        return None

    # Aggregate to Portfolio Level
    portfolio_daily = df.groupby('Date')['Daily_Delta'].sum().reset_index()
    portfolio_daily['Cumulative_PnL'] = portfolio_daily['Daily_Delta'].cumsum()
    
    final_pnl = portfolio_daily['Cumulative_PnL'].iloc[-1] if not portfolio_daily.empty else 0
    final_equity = INITIAL_CAPITAL + final_pnl
    
    # Extract numerical Top N for capital calculations
    match_top = re.search(r'\d+', top_n)
    top_n_int = int(match_top.group()) if match_top else 20
    
    # --- RCC (Return on Committed Capital) ---
    # R_cc = Period PnL / C_period
    c_period = INITIAL_CAPITAL
    rcc = final_pnl / c_period 
    
    # --- REC (Return on Engaged Capital) ---
    # R_ec = Period PnL / (N_traded * C_pair)
    # 1. Count unique pairs traded
    if 'Position' in df.columns:
        n_traded = len(df[df['Position'] != 0].drop_duplicates(subset=['Ticker_A', 'Ticker_B']))
        n_trades = len(df[df['Position'] != 0]) # Total individual positions entered/held
    else:
        n_traded = 0
        n_trades = 0
        
    # 2. Capital per pair
    c_pair = c_period / top_n_int if top_n_int > 0 else c_period
    
    # 3. Calc REC
    engaged_capital = n_traded * c_pair
    rec = final_pnl / engaged_capital if engaged_capital > 0 else 0
        
    # --- 依據圖片公式計算每月、累積與年化報酬率 ---
    if not portfolio_daily.empty:
        # 確保 Date 是 datetime 格式
        portfolio_daily['Date'] = pd.to_datetime(portfolio_daily['Date'])
        portfolio_daily['Equity'] = INITIAL_CAPITAL + portfolio_daily['Cumulative_PnL']
        
        # 取得每個月底的淨值 (使用 to_period 安全分群)
        portfolio_daily['YearMonth'] = portfolio_daily['Date'].dt.to_period('M')
        eom_equity = portfolio_daily.groupby('YearMonth')['Equity'].last()
        
        if len(eom_equity) > 0:
            # 插入初始本金作為第 0 個月的值，以計算第 1 個月的報酬率
            equity_values = np.insert(eom_equity.values, 0, INITIAL_CAPITAL)
            
            # 計算 Ri (第 i 個月的報酬率)
            monthly_returns = (equity_values[1:] / equity_values[:-1]) - 1
            
            n_months = len(monthly_returns) # n = 總月數
            
            if n_months > 0:
                # 累積報酬率 = 1 * (1+r1) * (1+r2) * ... * (1+rn) - 1
                cum_ret = np.prod(1 + monthly_returns) - 1
                
                # 年化報酬率 = ( 累積報酬率 + 1 ) ^ (12/n) - 1
                ann_ret = np.power(cum_ret + 1, 12 / n_months) - 1
            else:
                cum_ret, ann_ret = 0.0, 0.0
        else:
            cum_ret, ann_ret = 0.0, 0.0
    else:
        cum_ret, ann_ret = 0.0, 0.0

    # --- Sharpe Ratio ---
    daily_returns = portfolio_daily['Daily_Delta'] / INITIAL_CAPITAL
    if daily_returns.std() != 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0
        
    # --- Max Drawdown % ---
    roll_max = portfolio_daily['Cumulative_PnL'].cummax()
    drawdown = portfolio_daily['Cumulative_PnL'] - roll_max
    mdd = drawdown.min()
    mdd_pct = mdd / INITIAL_CAPITAL
        
    return {
        'DATASET': dataset,
        'SECTOR': sector,
        'RE-ENTRY': reentry,
        'METHOD': method,
        'TOP N': top_n,
        'STOP LOSS %': sl_pct,
        'Z-WINDOW': zwin,
        'Final_Equity': final_equity,
        'RCC_Raw': rcc,
        'REC_Raw': rec,
        'Cum_Ret_Raw': cum_ret,
        'Ann_Ret_Raw': ann_ret,
        'Sharpe_Raw': sharpe,
        'MDD_Raw': mdd_pct,
        'Total_Trades': n_trades,
        '_path': strategy_path
    }

@st.cache_data(show_spinner=False)
def build_master_dataframe(strategies):
    """Builds a single master dataframe of all metrics for all strategies."""
    records = []
    for strat in strategies:
        metrics = calculate_metrics_raw(load_data(strat), strat)
        if metrics:
            records.append(metrics)
    return pd.DataFrame(records) if records else pd.DataFrame()


# ==========================================
# MAIN APP
# ==========================================
def main():
    if not os.path.exists(RESULTS_DIR):
        st.error(f"Directory '{RESULTS_DIR}' not found. Please ensure the app is run from the project root.")
        return

    available_strategies = scan_strategies(RESULTS_DIR)
    if not available_strategies:
        st.warning("No strategy logs found. Please generate backtest logs.")
        return

    with st.spinner("Compiling strategy database..."):
        master_df = build_master_dataframe(available_strategies)
        
    if master_df.empty:
        st.error("Could not parse metrics from the provided data.")
        return

    total_strategies = len(master_df)

    # --- TOP HEADER SECTION ---
    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 30px; margin-top: 10px;">
        <div style="flex-grow: 1;">
            <div class="top-label">QUANTITATIVE PERFORMANCE DATA</div>
            <div class="main-title">Pairs Trading Comparison</div>
            <div class="sub-title">Full dataset: {total_strategies} strategy combinations evaluated.</div>
        </div>
        <div class="row-badge" style="min-width: 150px;">
            <div class="row-badge-title">Displayed Rows</div>
            <div class="row-badge-val" id="row-count">{total_strategies}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # --- FILTERS ---
    f_cols1 = st.columns(4)
    f_cols2 = st.columns(3)
    
    with f_cols1[0]:
        options = ["All"] + sorted(master_df['DATASET'].unique().tolist())
        sel_dataset = st.selectbox("DATASET", options)
    with f_cols1[1]:
        options = ["All"] + sorted(master_df['SECTOR'].unique().tolist())
        sel_sector = st.selectbox("SECTOR", options)
    with f_cols1[2]:
        options = ["All"] + sorted(master_df['RE-ENTRY'].unique().tolist())
        sel_reentry = st.selectbox("RE-ENTRY", options)
    with f_cols1[3]:
        options = ["All"] + sorted(master_df['METHOD'].unique().tolist())
        sel_method = st.selectbox("METHOD", options)

    with f_cols2[0]:
        options = ["All"] + sorted(master_df['TOP N'].unique().tolist(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        sel_topn = st.selectbox("TOP N", options)
    with f_cols2[1]:
        options = ["All"] + sorted(master_df['STOP LOSS %'].unique().tolist(), key=lambda x: int(x.replace('%','')) if x.replace('%','').isdigit() else 0)
        sel_sl = st.selectbox("STOP LOSS %", options)
    with f_cols2[2]:
        options = ["All"] + sorted(master_df['Z-WINDOW'].unique().tolist(), key=lambda x: int(x) if x.isdigit() else 0)
        sel_zwin = st.selectbox("Z-WINDOW", options)

    # --- APPLY FILTERS ---
    filtered_df = master_df.copy()
    if sel_dataset != "All": filtered_df = filtered_df[filtered_df['DATASET'] == sel_dataset]
    if sel_sector != "All": filtered_df = filtered_df[filtered_df['SECTOR'] == sel_sector]
    if sel_reentry != "All": filtered_df = filtered_df[filtered_df['RE-ENTRY'] == sel_reentry]
    if sel_method != "All": filtered_df = filtered_df[filtered_df['METHOD'] == sel_method]
    if sel_topn != "All": filtered_df = filtered_df[filtered_df['TOP N'] == sel_topn]
    if sel_sl != "All": filtered_df = filtered_df[filtered_df['STOP LOSS %'] == sel_sl]
    if sel_zwin != "All": filtered_df = filtered_df[filtered_df['Z-WINDOW'] == sel_zwin]

    st.markdown(
        f"<script>document.getElementById('row-count').innerText = '{len(filtered_df)}';</script>", 
        unsafe_allow_html=True
    )

    # --- CALCULATE BEST METRICS FOR CARDS ---
    if len(filtered_df) > 0:
        best_cum = filtered_df.loc[filtered_df['Cum_Ret_Raw'].idxmax()]
        best_ann = filtered_df.loc[filtered_df['Ann_Ret_Raw'].idxmax()]
        best_rcc = filtered_df.loc[filtered_df['RCC_Raw'].idxmax()]
        best_shp = filtered_df.loc[filtered_df['Sharpe_Raw'].idxmax()]
        low_dd = filtered_df.loc[filtered_df['MDD_Raw'].idxmax()] 
    else:
        empty_series = pd.Series({
            'RCC_Raw': 0, 'REC_Raw': 0, 'Cum_Ret_Raw': 0, 'Ann_Ret_Raw': 0, 
            'Sharpe_Raw': 0, 'MDD_Raw': 0,
            'DATASET': '-', 'SECTOR': '-', 'RE-ENTRY': '-',
            'METHOD': '-', 'TOP N': '-', 'STOP LOSS %': '-', 'Z-WINDOW': '-'
        })
        best_cum = best_ann = best_rcc = best_shp = low_dd = empty_series

    def make_desc(row): return f"{row['DATASET']} · {row['SECTOR']} · {row['RE-ENTRY']} · {row['METHOD']} · {row['TOP N']} · SL {row['STOP LOSS %']} · ZWin {row['Z-WINDOW']}"

    # --- METRIC CARDS ---
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.markdown(f"""
        <div class="metric-card card-purple">
            <div class="card-title">BEST CUMULATIVE</div>
            <div class="card-value val-purple">{best_cum['Cum_Ret_Raw']:.2%}</div>
            <div class="card-desc">{make_desc(best_cum)}</div>
        </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
        <div class="metric-card card-blue">
            <div class="card-title">BEST ANNUALIZED</div>
            <div class="card-value val-blue">{best_ann['Ann_Ret_Raw']:.2%}</div>
            <div class="card-desc">{make_desc(best_ann)}</div>
        </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
        <div class="metric-card card-green">
            <div class="card-title">BEST RCC</div>
            <div class="card-value val-green">{best_rcc['RCC_Raw']:.2%}</div>
            <div class="card-desc">{make_desc(best_rcc)}</div>
        </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
        <div class="metric-card card-orange">
            <div class="card-title">BEST SHARPE</div>
            <div class="card-value val-orange">{best_shp['Sharpe_Raw']:.2f}</div>
            <div class="card-desc">{make_desc(best_shp)}</div>
        </div>
    """, unsafe_allow_html=True)
    
    c5.markdown(f"""
        <div class="metric-card card-red">
            <div class="card-title">LOWEST DRAWDOWN</div>
            <div class="card-value val-red">{low_dd['MDD_Raw']:.2%}</div>
            <div class="card-desc">{make_desc(low_dd)}</div>
        </div>
    """, unsafe_allow_html=True)

    # --- PERFORMANCE TABLE ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <h3 style='margin-bottom:0; font-weight: 900; color: #f1f5f9;'>Complete Performance Table</h3>
        <p style='color: #94a3b8; font-size: 14px; margin-top: 5px; margin-bottom: 15px;'>
        Click any column header to sort. Use filters above to narrow down parameters. <br>
        <b style='color: #60a5fa;'>💡 點擊表格最左側的 Checkbox 勾選框，即可在下方繪製歷史走勢圖（最多 5 個）。</b></p>
    """, unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.warning("No strategies match the selected filters.")
        return

    # 先對 filtered_df 進行排序並重設 Index，確保後續依據 Index 繪圖時能精準對應
    filtered_df = filtered_df.sort_values(by='RCC_Raw', ascending=False).reset_index(drop=True)

    # Prepare Display DataFrame
    display_df = filtered_df.copy()
    display_df['FINAL EQUITY'] = display_df['Final_Equity']
    display_df['CUM. RETURN %'] = display_df['Cum_Ret_Raw']
    display_df['ANN. RETURN %'] = display_df['Ann_Ret_Raw']
    display_df['RCC %'] = display_df['RCC_Raw']
    display_df['REC %'] = display_df['REC_Raw']
    display_df['SHARPE'] = display_df['Sharpe_Raw']
    display_df['MAX DRAWDOWN %'] = display_df['MDD_Raw']
    display_df['TOTAL TRADES'] = display_df['Total_Trades'].apply(lambda x: f"{int(x):,}")

    cols = ['DATASET', 'SECTOR', 'RE-ENTRY', 'METHOD', 'TOP N', 'STOP LOSS %', 'Z-WINDOW', 
            'FINAL EQUITY', 'CUM. RETURN %', 'ANN. RETURN %', 'RCC %', 'REC %', 'SHARPE', 'MAX DRAWDOWN %', 'TOTAL TRADES']
    display_df = display_df[cols]

    def apply_color(val, color):
        return f'color: {color}; font-weight: 700;'
    
    styled_df = display_df.style.format({
        'FINAL EQUITY': '${:,.2f}',
        'CUM. RETURN %': '{:.2%}',
        'ANN. RETURN %': '{:.2%}',
        'RCC %': '{:.2%}',
        'REC %': '{:.2%}',
        'SHARPE': '{:.2f}',
        'MAX DRAWDOWN %': '{:.2%}'
    })\
    .map(lambda x: apply_color(x, '#f8fafc'), subset=['FINAL EQUITY'])\
    .map(lambda x: apply_color(x, '#c084fc'), subset=['CUM. RETURN %'])\
    .map(lambda x: apply_color(x, '#38bdf8'), subset=['ANN. RETURN %'])\
    .map(lambda x: apply_color(x, '#4ade80'), subset=['RCC %'])\
    .map(lambda x: apply_color(x, '#60a5fa'), subset=['REC %'])\
    .map(lambda x: apply_color(x, '#fb923c'), subset=['SHARPE'])\
    .map(lambda x: apply_color(x, '#f87171'), subset=['MAX DRAWDOWN %'])\
    .map(lambda x: 'color: #94a3b8; font-weight: 600;', subset=['DATASET', 'SECTOR', 'RE-ENTRY', 'METHOD', 'TOP N', 'STOP LOSS %', 'Z-WINDOW'])

    # 啟用原生 Checkbox 列選取功能
    selection_event = st.dataframe(
        styled_df, 
        use_container_width=True,
        hide_index=True,
        height=400,
        on_select="rerun",
        selection_mode="multi-row"
    )

    # 獲取使用者勾選的行數清單 (回傳的是 Index 列表，例如 [0, 2, 5])
    selected_rows = selection_event.selection.rows

    # --- EQUITY CURVE COMPARISON ---
    st.markdown("<br><hr style='border-color: #334155;'><br>", unsafe_allow_html=True)
    st.markdown("""
        <h3 style='margin-bottom:0; font-weight: 900; color: #f1f5f9;'>📈 Historical Performance Comparison</h3>
    """, unsafe_allow_html=True)

    # 限制繪圖數量上限為 5 個
    if len(selected_rows) > 5:
        st.warning("⚠️ 最多只能選擇 5 個策略進行繪圖，目前將為您顯示最先勾選的 5 個。")
        selected_rows = selected_rows[:5]

    if selected_rows:
        fig = go.Figure()
        
        for idx in selected_rows:
            # 透過勾選的 Index 對應回 filtered_df 取出資料路徑
            row_data = filtered_df.iloc[idx]
            strat_path = row_data['_path']
            legend_name = f"{row_data['DATASET']} · {row_data['SECTOR']} · {row_data['RE-ENTRY']} · {row_data['METHOD']} · {row_data['TOP N']} · SL {row_data['STOP LOSS %']} · ZWin {row_data['Z-WINDOW']}"
            
            df = load_data(strat_path)
            
            if not df.empty:
                # Recalculate daily equity for plotting
                port_daily = df.groupby('Date')['Daily_Delta'].sum().reset_index()
                port_daily['Cumulative_PnL'] = port_daily['Daily_Delta'].cumsum()
                port_daily['Equity'] = INITIAL_CAPITAL + port_daily['Cumulative_PnL']
                
                fig.add_trace(go.Scatter(
                    x=port_daily['Date'], 
                    y=port_daily['Equity'], 
                    mode='lines', 
                    name=legend_name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Final Equity ($)",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h", 
                yanchor="top", 
                y=-0.2, 
                xanchor="center", 
                x=0.5,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, t=30, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 請在上方表格最左側勾選您想查看的策略（至多 5 個），以生成歷史淨值走勢圖。")

if __name__ == "__main__":
    main()