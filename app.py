import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pairs Trading Comparison Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS FOR DYNAMIC THEME SUPPORT
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-title {
        color: var(--text-color); font-size: 2.5rem !important;
        font-weight: 800; margin-bottom: 0.2rem; letter-spacing: -0.02em;
    }
    .sub-title { color: var(--text-color); opacity: 0.7; font-size: 1rem; margin-bottom: 2rem; }
    .blue-subtitle {
        color: #3b82f6; font-size: 0.8rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;
    }
    
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="metric-container"] label {
        color: var(--text-color) !important; opacity: 0.8 !important; 
        font-weight: 600 !important; font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
RESULTS_DIR = "results"
INITIAL_CAPITAL = 10000.0

def natural_sort_key(s):
    return [int(text) if text.isdigit() else str(text).lower() for text in re.split(r'(\d+)', str(s))]

def scan_strategies(base_dir):
    strategies = []
    if not os.path.exists(base_dir): return strategies
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "TradeLogs" in file or "detailed_trade_logs" in file:
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                strategies.append(rel_path.replace("\\", "/"))
    return sorted(strategies)

@st.cache_data(show_spinner=False)
def load_data(strategy_path):
    file_path = os.path.join(RESULTS_DIR, strategy_path)
    if not os.path.exists(file_path): return pd.DataFrame()
    
    try:
        sample = pd.read_csv(file_path, nrows=0)
        original_header = sample.columns.tolist()
        clean_header = [str(c).strip() for c in original_header]
        
        target_cols = [
            'Date', 'Position', 'Ticker_A', 'Ticker_B', 'Daily_Delta', 
            'Trade_PnL', 'Status', 'Hedge_Ratio', 'Price_A', 'Price_B', 'Days_Held',
            'Period_Start', 'Period_End'
        ]
        
        col_map = {orig: clean for orig, clean in zip(original_header, clean_header) if clean in target_cols}
        cols_to_use_orig = list(col_map.keys())
        
        dtypes_target = {
            'Position': 'float32', 'Daily_Delta': 'float32', 'Trade_PnL': 'float32',
            'Hedge_Ratio': 'float32', 'Price_A': 'float32', 'Price_B': 'float32',
            'Days_Held': 'float32', 'Status': 'category', 
            'Period_Start': 'string', 'Period_End': 'string'
        }
        
        dtypes_to_use = {orig: dtypes_target[clean] for orig, clean in col_map.items() if clean in dtypes_target}
        
        df = pd.read_csv(file_path, usecols=cols_to_use_orig, dtype=dtypes_to_use, parse_dates=False)
        df.rename(columns=col_map, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        if 'Position' in df.columns:
            df['Position'] = df['Position'].fillna(0).astype(int)
        if 'Days_Held' in df.columns:
            df['Days_Held'] = df['Days_Held'].fillna(0).astype(int)
        
        if 'Ticker_A' in df.columns and 'Ticker_B' in df.columns:
            df['Ticker_A'] = df['Ticker_A'].fillna('UNKNOWN').astype(str)
            df['Ticker_B'] = df['Ticker_B'].fillna('UNKNOWN').astype(str)
            df = df.sort_values(by=['Ticker_A', 'Ticker_B', 'Date']).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading {strategy_path}: {str(e)}")
        return pd.DataFrame()

def extract_features_from_path(path):
    path_lower = path.lower()
    dataset = "Full" if "full" in path_lower else "Current" if "current" in path_lower else "Unknown"
    reentry = "NoReEntry" if "noreentry" in path_lower else "ReEntry" if "reentry" in path_lower else "Unknown"
    
    method = "Unknown"
    if "ssd_basic" in path_lower:
        method = "SSD (Basic)"
    elif "ssd" in path_lower:
        method = "SSD"
    elif "eg" in path_lower:
        method = "EG"
    elif "hdbscan" in path_lower:
        is_ae = "_ae_" in path_lower or "hdbscan_ae" in path_lower
        is_pca = "_pca_" in path_lower or "hdbscan_pca" in path_lower
        if is_ae:
            method = "HDBSCAN (AE PCA)" if is_pca else "HDBSCAN (AE UMAP)"
        else:
            method = "HDBSCAN (PCA)" if is_pca else "HDBSCAN (UMAP)"

    top_n = "Top 20"
    match_n = re.search(r'top(\d+)', path_lower)
    if match_n: top_n = f"Top {match_n.group(1)}"
        
    sl_pct = "0%"
    match_sl = re.search(r'sl(\d+)', path_lower)
    if match_sl: sl_pct = f"{match_sl.group(1)}%"
        
    zwin = "0"
    match_zwin = re.search(r'zwin(\d+)', path_lower)
    if match_zwin: zwin = match_zwin.group(1)
        
    return dataset, reentry, method, top_n, sl_pct, zwin

@st.cache_data(show_spinner=False)
def calculate_metrics_raw(strategy_path):
    df = load_data(strategy_path)
    if df.empty: return None

    dataset, reentry, method, top_n, sl_pct, zwin = extract_features_from_path(strategy_path)
    top_n_int = int(top_n.replace('Top ', '')) if 'Top' in top_n else 20
    c_period = INITIAL_CAPITAL
    
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

    rcc = final_pnl / c_period if c_period > 0 else 0
    
    if 'Position' in df.columns and 'Ticker_A' in df.columns:
        n_traded = len(df[df['Position'] != 0].drop_duplicates(subset=['Ticker_A', 'Ticker_B']))
        df['Prev_Pos'] = df.groupby(['Ticker_A', 'Ticker_B'])['Position'].shift(1).fillna(0)
        
        direction_change = df['Position'] != df['Prev_Pos']
        exit_mask = direction_change & (df['Prev_Pos'] != 0)
        n_exits_total = exit_mask.sum()
        
        last_rows = df.groupby(['Ticker_A', 'Ticker_B']).tail(1)
        n_forced_close = (last_rows['Position'] != 0).sum()
        n_entries = n_exits_total + n_forced_close
        
        if 'Status' in df.columns:
            n_stop_loss = df[exit_mask]['Status'].astype(str).str.contains('stop|sl|停損', case=False, na=False).sum()
            n_normal_exits = n_exits_total - n_stop_loss
        else:
            n_stop_loss = -1
            n_normal_exits = n_exits_total
            
        if 'Daily_Delta' in df.columns:
            state_change = df['Position'] != df['Prev_Pos']
            df['State_ID'] = state_change.groupby([df['Ticker_A'], df['Ticker_B']]).cumsum()
            df['Prev_State_ID'] = df.groupby(['Ticker_A', 'Ticker_B'])['State_ID'].shift(1).fillna(0)
            active_mask = (df['Prev_Pos'] != 0) | (df['Daily_Delta'] != 0)
            
            if active_mask.any():
                trade_pnls = df[active_mask].groupby(['Ticker_A', 'Ticker_B', 'Prev_State_ID'])['Daily_Delta'].sum()
                gross_profit = float(trade_pnls[trade_pnls > 0].sum())
                gross_loss = float(trade_pnls[trade_pnls < 0].sum())
            else:
                gross_profit = gross_loss = 0.0
        else:
            gross_profit = gross_loss = 0.0
            
    else:
        n_traded = n_entries = n_normal_exits = 0
        n_stop_loss = -1
        n_forced_close = 0
        gross_profit = gross_loss = 0.0
        
    c_pair = c_period / top_n_int if top_n_int > 0 else c_period
    engaged_capital = n_traded * c_pair
    rec = final_pnl / engaged_capital if engaged_capital > 0 else 0

    return {
        'DATASET': dataset, 'RE-ENTRY': reentry,
        'METHOD': method, 'TOP N': top_n, 'STOP LOSS %': sl_pct, 'Z-WINDOW': zwin,
        'Final_Equity': final_equity, 'RCC_Raw': rcc, 'REC_Raw': rec,
        'Cum_Ret_Raw': cum_ret, 'Ann_Ret_Raw': ann_ret, 'Sharpe_Raw': sharpe, 'MDD_Raw': mdd_pct,
        'Entries': n_entries, 'Exits': n_normal_exits, 'Stop_Losses': n_stop_loss, 'Forced_Closes': n_forced_close,
        'Gross_Profit': gross_profit, 'Gross_Loss': gross_loss,
        '_path': strategy_path
    }

@st.cache_data(show_spinner=False)
def build_master_dataframe(strategies):
    records = []
    for s in strategies:
        res = calculate_metrics_raw(s)
        if res: records.append(res)
    return pd.DataFrame(records)

def make_desc(row): 
    return f"{row['DATASET']} · {row['RE-ENTRY']} · {row['METHOD']} · {row['TOP N']} · SL {row['STOP LOSS %']} · ZWin {row['Z-WINDOW']}"

@st.fragment
def render_deep_dive(target_row):
    st.markdown("---")
    st.markdown(f"### 🔍 Strategy Deep Dive: <span style='color:#3b82f6; font-size:1.5rem;'>{make_desc(target_row)}</span>", unsafe_allow_html=True)
    
    raw_target_df = load_data(target_row['_path'])
    
    if raw_target_df.empty or 'Period_Start' not in raw_target_df.columns:
        st.warning("Missing required columns ('Period_Start', 'Period_End') for periodic analysis.")
        return

    raw_target_df['Trading_Period'] = raw_target_df['Period_Start'].astype(str) + " ~ " + raw_target_df['Period_End'].astype(str)
    
    period_stats = raw_target_df.groupby('Trading_Period').agg(
        Start_Date=('Period_Start', 'first'),
        Return=('Daily_Delta', 'sum')
    ).reset_index()
    
    period_stats = period_stats.sort_values('Start_Date').reset_index(drop=True)
    period_stats.rename(columns={'Trading_Period': 'Trading Period', 'Return': 'Period Return ($)'}, inplace=True)
    disp_periods = period_stats[['Trading Period', 'Period Return ($)']]
    
    col_p1, col_p2 = st.columns([1, 1.5])
    
    with col_p1:
        st.markdown("##### 1. Trading Periods (Select to view pairs)")
        format_p = {'Period Return ($)': '${:,.2f}'}
        styled_periods = disp_periods.style.format(format_p).map(
            lambda x: 'color: #4ade80; font-weight:bold;' if pd.notna(x) and x > 0 else ('color: #f87171; font-weight:bold;' if pd.notna(x) and x < 0 else ''), subset=['Period Return ($)']
        )
        
        period_event = st.dataframe(
            styled_periods, width='stretch', height=400, hide_index=True,
            selection_mode="single-row", on_select="rerun"
        )
        
    with col_p2:
        sel_p_row = period_event.selection.rows
        if not sel_p_row:
            st.info("👈 Select a trading period to view the pairs traded during that time.")
            return

        sel_period_str = disp_periods.iloc[sel_p_row[0]]['Trading Period']
        st.markdown(f"##### 2. Pairs Traded in [{sel_period_str}]")
        
        period_df = raw_target_df[raw_target_df['Trading_Period'] == sel_period_str].copy()
        traded_mask = period_df.groupby(['Ticker_A', 'Ticker_B'])['Position'].transform(lambda x: (x != 0).any())
        period_traded_df = period_df[traded_mask].copy()
        
        if period_traded_df.empty:
            st.warning("No positions were opened during this period.")
            return

        pair_stats = period_traded_df.groupby(['Ticker_A', 'Ticker_B']).agg(
            Pair_Return=('Daily_Delta', 'sum')
        ).reset_index()
        
        pair_stats.rename(columns={'Ticker_A': 'Stock A', 'Ticker_B': 'Stock B', 'Pair_Return': 'Return ($)'}, inplace=True)
        pair_stats = pair_stats.sort_values(by='Return ($)', ascending=False).reset_index(drop=True)
        
        format_pair = {'Return ($)': '${:,.2f}'}
        styled_pairs = pair_stats.style.format(format_pair).map(
            lambda x: 'color: #4ade80; font-weight:bold;' if pd.notna(x) and x > 0 else ('color: #f87171; font-weight:bold;' if pd.notna(x) and x < 0 else ''), subset=['Return ($)']
        )
        
        pair_event = st.dataframe(
            styled_pairs, width='stretch', height=400, hide_index=True,
            selection_mode="single-row", on_select="rerun"
        )
        
        sel_pair_row = pair_event.selection.rows
        
    if sel_pair_row:
        t_a = pair_stats.iloc[sel_pair_row[0]]['Stock A']
        t_b = pair_stats.iloc[sel_pair_row[0]]['Stock B']
        
        st.markdown("---")
        st.markdown(f"##### 3. Trade Visualizer: {t_a} vs {t_b} (Period: {sel_period_str})")
        
        pair_full = period_df[(period_df['Ticker_A'] == t_a) & (period_df['Ticker_B'] == t_b)].copy()
        pair_full = pair_full.sort_values('Date')
        
        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        
        if 'Price_A' in pair_full.columns and 'Price_B' in pair_full.columns and not pair_full['Price_A'].isna().all():
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=pair_full['Price_A'], name=f"{t_a} Price", line=dict(color='rgba(96, 165, 250, 0.6)', width=2)), secondary_y=False)
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=pair_full['Price_B'], name=f"{t_b} Price", line=dict(color='rgba(251, 211, 141, 0.6)', width=2)), secondary_y=True)
            fig_p.update_yaxes(title_text=f"{t_a} Price", secondary_y=False)
            fig_p.update_yaxes(title_text=f"{t_b} Price", secondary_y=True)
            marker_y_col = 'Price_A'
        else:
            if 'Daily_Delta' not in pair_full.columns: pair_full['Daily_Delta'] = 0.0
            pair_full['Cum_PnL'] = pair_full['Daily_Delta'].cumsum()
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=pair_full['Cum_PnL'], name="Cumulative Return ($)", line=dict(color='rgba(74, 222, 128, 0.6)', width=2)), secondary_y=False)
            fig_p.update_yaxes(title_text="Cumulative Return ($)", secondary_y=False)
            marker_y_col = 'Cum_PnL'

        holding_pos = 0 
        start_date = None
        
        long_x, long_y, short_x, short_y = [], [], [], []
        tp_x, tp_y, sl_x, sl_y = [], [], [], []
        
        for idx, row in pair_full.iterrows():
            pos = row['Position']
            date = row['Date']
            y_val = row[marker_y_col]
            status = str(row.get('Status', '')).lower()
            
            if pos != holding_pos:
                if holding_pos != 0 and (pos == 0 or np.sign(pos) != np.sign(holding_pos)):
                    end_date = date
                    if 'stop' in status or 'sl' in status or '停損' in status:
                        sl_x.append(date); sl_y.append(y_val)
                    else:
                        tp_x.append(date); tp_y.append(y_val)
                        
                    if 'Daily_Delta' in pair_full.columns:
                        pnl = pair_full.loc[(pair_full['Date'] >= start_date) & (pair_full['Date'] <= end_date), 'Daily_Delta'].sum()
                    else:
                        pnl = 0
                        
                    f_color = "rgba(74, 222, 128, 0.15)" if pnl > 0 else "rgba(248, 113, 113, 0.15)"
                    fig_p.add_vrect(
                        x0=start_date, x1=end_date, fillcolor=f_color,
                        opacity=1, layer="below", line_width=0,
                        annotation_text=f"{'Win' if pnl>0 else 'Loss'}", annotation_position="top left", annotation_font_color="var(--text-color)"
                    )
                    holding_pos = 0
                
                if pos != 0 and holding_pos == 0:
                    holding_pos = pos
                    start_date = date
                    if pos > 0:
                        long_x.append(date); long_y.append(y_val)
                    else:
                        short_x.append(date); short_y.append(y_val)

        if holding_pos != 0:
            end_date = pair_full['Date'].iloc[-1]
            if 'Daily_Delta' in pair_full.columns:
                pnl = pair_full.loc[(pair_full['Date'] >= start_date) & (pair_full['Date'] <= end_date), 'Daily_Delta'].sum()
            else:
                pnl = 0
            f_color = "rgba(74, 222, 128, 0.15)" if pnl > 0 else "rgba(248, 113, 113, 0.15)"
            fig_p.add_vrect(x0=start_date, x1=end_date, fillcolor=f_color, opacity=1, layer="below", line_width=0)
        
        # 移除了強制深色的外框線，使用透明邊界 (rgba(0,0,0,0))
        if long_x: fig_p.add_trace(go.Scatter(x=long_x, y=long_y, mode='markers', name='Buy Long', marker=dict(symbol='triangle-up', size=14, color='#4ade80', line=dict(width=1, color='rgba(0,0,0,0)'))), secondary_y=False)
        if short_x: fig_p.add_trace(go.Scatter(x=short_x, y=short_y, mode='markers', name='Sell Short', marker=dict(symbol='triangle-down', size=14, color='#f87171', line=dict(width=1, color='rgba(0,0,0,0)'))), secondary_y=False)
        if tp_x: fig_p.add_trace(go.Scatter(x=tp_x, y=tp_y, mode='markers', name='Take Profit / Close', marker=dict(symbol='circle', size=12, color='#60a5fa', line=dict(width=1, color='rgba(0,0,0,0)'))), secondary_y=False)
        if sl_x: fig_p.add_trace(go.Scatter(x=sl_x, y=sl_y, mode='markers', name='Stop Loss', marker=dict(symbol='x', size=10, color='#fbd38d', line=dict(width=2.5, color='#fbd38d'))), secondary_y=False)

        # 移除強制寫死的深色背景，使用 Streamlit 預設的主題相容設定
        fig_p.update_layout(
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_p, width='stretch')

def main():
    st.markdown('<div class="blue-subtitle">QUANTITATIVE PERFORMANCE DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Pairs Trading Comparison Dashboard</div>', unsafe_allow_html=True)
    
    available_strategies = scan_strategies(RESULTS_DIR)
    st.markdown(f'<div class="sub-title">Full dataset: {len(available_strategies)} strategy combinations currently loaded.</div>', unsafe_allow_html=True)
    
    if not available_strategies:
        st.warning("No strategy logs found in the results directory.")
        return
        
    with st.spinner("Compiling massive dataset metrics..."):
        master_df = build_master_dataframe(available_strategies)
        
    if master_df.empty:
        st.error("Could not parse any valid strategy data.")
        return

    st.markdown("### Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1: sel_dataset = st.selectbox("DATASET", ["All"] + sorted(master_df['DATASET'].unique(), key=natural_sort_key))
    with f_col2: sel_reentry = st.selectbox("RE-ENTRY", ["All"] + sorted(master_df['RE-ENTRY'].unique(), key=natural_sort_key))
    with f_col3: sel_method = st.selectbox("METHOD", ["All"] + sorted(master_df['METHOD'].unique(), key=natural_sort_key))
    
    f_col4, f_col5, f_col6 = st.columns(3)
    with f_col4: sel_topn = st.selectbox("TOP N", ["All"] + sorted(master_df['TOP N'].unique(), key=natural_sort_key))
    with f_col5: sel_sl = st.selectbox("STOP LOSS %", ["All"] + sorted(master_df['STOP LOSS %'].unique(), key=natural_sort_key))
    with f_col6: sel_zwin = st.selectbox("Z-WINDOW", ["All"] + sorted(master_df['Z-WINDOW'].unique(), key=natural_sort_key))

    filtered_df = master_df.copy()
    if sel_dataset != "All": filtered_df = filtered_df[filtered_df['DATASET'] == sel_dataset]
    if sel_reentry != "All": filtered_df = filtered_df[filtered_df['RE-ENTRY'] == sel_reentry]
    if sel_method != "All": filtered_df = filtered_df[filtered_df['METHOD'] == sel_method]
    if sel_topn != "All": filtered_df = filtered_df[filtered_df['TOP N'] == sel_topn]
    if sel_sl != "All": filtered_df = filtered_df[filtered_df['STOP LOSS %'] == sel_sl]
    if sel_zwin != "All": filtered_df = filtered_df[filtered_df['Z-WINDOW'] == sel_zwin]

    st.markdown("<br>", unsafe_allow_html=True)
    if len(filtered_df) > 0:
        best_cum = filtered_df.loc[filtered_df['Cum_Ret_Raw'].idxmax()]
        best_ann = filtered_df.loc[filtered_df['Ann_Ret_Raw'].idxmax()]
        best_rcc = filtered_df.loc[filtered_df['RCC_Raw'].idxmax()]
        best_shp = filtered_df.loc[filtered_df['Sharpe_Raw'].idxmax()]
        low_dd = filtered_df.loc[filtered_df['MDD_Raw'].idxmax()] 
    else:
        empty_series = pd.Series({
            'RCC_Raw': 0, 'REC_Raw': 0, 'Cum_Ret_Raw': 0, 'Ann_Ret_Raw': 0, 
            'Sharpe_Raw': 0, 'MDD_Raw': 0, 'Entries': 0, 'Exits': 0, 'Stop_Losses': 0, 'Forced_Closes': 0,
            'Gross_Profit': 0.0, 'Gross_Loss': 0.0,
            'DATASET': '-', 'RE-ENTRY': '-', 'METHOD': '-', 'TOP N': '-', 'STOP LOSS %': '-', 'Z-WINDOW': '-'
        })
        best_cum = best_ann = best_rcc = best_shp = low_dd = empty_series

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BEST CUMULATIVE RET", f"{best_cum['Cum_Ret_Raw']:.2%}", make_desc(best_cum))
    c2.metric("BEST ANNUALIZED RET", f"{best_ann['Ann_Ret_Raw']:.2%}", make_desc(best_ann))
    c3.metric("BEST RCC", f"{best_rcc['RCC_Raw']:.2%}", make_desc(best_rcc))
    c4.metric("BEST SHARPE", f"{best_shp['Sharpe_Raw']:.2f}", make_desc(best_shp))
    c5.metric("LOWEST DRAWDOWN", f"{abs(low_dd['MDD_Raw']):.2%}", make_desc(low_dd))

    st.markdown("### Complete Performance Table")
    st.markdown("Select checkboxes on the left to plot strategy equity curves below (Max 5). The **first checked row** will display detailed trading periods.", unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.info("No data matches the current filters.")
        return

    display_df = filtered_df.copy().sort_values(by='Ann_Ret_Raw', ascending=False).reset_index(drop=True)

    display_df['FINAL EQUITY ($)'] = display_df['Final_Equity'].apply(lambda x: f"${x:,.2f}")
    display_df['CUM. RETURN (%)'] = display_df['Cum_Ret_Raw']
    display_df['ANN. RETURN (%)'] = display_df['Ann_Ret_Raw']
    display_df['RCC (%)'] = display_df['RCC_Raw']
    display_df['REC (%)'] = display_df['REC_Raw']
    display_df['SHARPE'] = display_df['Sharpe_Raw']
    display_df['MAX DRAWDOWN (%)'] = display_df['MDD_Raw']
    display_df['ENTRIES (Count)'] = display_df['Entries'].apply(lambda x: f"{int(x):,}")
    display_df['EXITS (Count)'] = display_df['Exits'].apply(lambda x: f"{int(x):,}")
    display_df['STOP LOSSES (Count)'] = display_df['Stop_Losses'].apply(lambda x: f"{int(x):,}" if x >= 0 else "N/A")
    display_df['FORCED CLOSES (Count)'] = display_df['Forced_Closes'].apply(lambda x: f"{int(x):,}")
    display_df['GROSS PROFIT ($)'] = display_df['Gross_Profit'].apply(lambda x: f"${x:,.2f}")
    display_df['GROSS LOSS ($)'] = display_df['Gross_Loss'].apply(lambda x: f"${x:,.2f}")

    cols = ['DATASET', 'RE-ENTRY', 'METHOD', 'TOP N', 'STOP LOSS %', 'Z-WINDOW', 
            'FINAL EQUITY ($)', 'CUM. RETURN (%)', 'ANN. RETURN (%)', 'RCC (%)', 'REC (%)', 'SHARPE', 'MAX DRAWDOWN (%)', 
            'ENTRIES (Count)', 'EXITS (Count)', 'STOP LOSSES (Count)', 'FORCED CLOSES (Count)', 'GROSS PROFIT ($)', 'GROSS LOSS ($)']
    
    df_styled = display_df[cols].style.format({
        'CUM. RETURN (%)': '{:.2%}', 'ANN. RETURN (%)': '{:.2%}',
        'RCC (%)': '{:.2%}', 'REC (%)': '{:.2%}', 'SHARPE': '{:.2f}', 'MAX DRAWDOWN (%)': '{:.2%}'
    }).map(lambda _: 'color: #4ade80; font-weight: bold;', subset=['CUM. RETURN (%)', 'GROSS PROFIT ($)']) \
      .map(lambda _: 'color: #60a5fa; font-weight: bold;', subset=['ANN. RETURN (%)']) \
      .map(lambda _: 'color: #fbd38d; font-weight: bold;', subset=['SHARPE']) \
      .map(lambda _: 'color: #f87171; font-weight: bold;', subset=['MAX DRAWDOWN (%)', 'GROSS LOSS ($)'])

    event = st.dataframe(
        df_styled, width='stretch', hide_index=True, height=350,
        on_select="rerun", selection_mode="multi-row"
    )

    selected_rows = event.selection.rows

    if len(selected_rows) > 0:
        st.markdown("### 📈 Selected Strategies Equity Curves")
        plot_rows = selected_rows[:5]
            
        fig_eq = go.Figure()
        colors = ['#4ade80', '#60a5fa', '#fbd38d', '#f87171', '#c084fc']
        
        for i, row_idx in enumerate(plot_rows):
            path = display_df.iloc[row_idx]['_path']
            desc = make_desc(display_df.iloc[row_idx])
            raw_df = load_data(path)
            if not raw_df.empty and 'Daily_Delta' in raw_df.columns:
                port_daily = raw_df.groupby('Date')['Daily_Delta'].sum().reset_index()
                port_daily['Cumulative_PnL'] = port_daily['Daily_Delta'].cumsum()
                port_daily['Equity'] = INITIAL_CAPITAL + port_daily['Cumulative_PnL']
                
                fig_eq.add_trace(go.Scatter(
                    x=port_daily['Date'], y=port_daily['Equity'], mode='lines', 
                    name=desc, line=dict(width=2, color=colors[i % len(colors)])
                ))
                
        # 移除強制寫死的深色背景，使用 Streamlit 預設的主題相容設定
        fig_eq.update_layout(
            xaxis_title="Date", yaxis_title="Account Equity ($)",
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, width='stretch')

        target_row = display_df.iloc[selected_rows[0]]
        render_deep_dive(target_row)

if __name__ == "__main__":
    main()