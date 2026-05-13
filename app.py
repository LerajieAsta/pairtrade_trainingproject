import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pair Trading Backtest Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS FOR PREMIUM AESTHETICS
# ==========================================
st.markdown("""
<style>
    /* Main Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Title */
    .main-title {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle / Description */
    .sub-title {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(0, 201, 255, 0.5);
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #1E1E2E !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CONSTANTS & CONFIG
# ==========================================
RESULTS_DIR = "results"
INITIAL_CAPITAL = 10000.0


# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def scan_strategies(base_dir):
    """Scan directory recursively to find all strategies (lazy loading)."""
    strategies = []
    if not os.path.exists(base_dir):
        return strategies
    for root, dirs, files in os.walk(base_dir):
        if "detailed_trade_logs.csv" in files:
            # Get relative path e.g., SSD_NoReEntry_NoSector/Top20_SL5_ZWin0
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
        'Sector': 'category',
        'Ticker_A': 'category',
        'Ticker_B': 'category',
        'Status': 'category',
        'Pair_Rank': 'int16',
        'Position': 'int8',
        'Days_Held': 'int16',
        'Log_Mean_A': 'float32',
        'Log_Std_A': 'float32',
        'Log_Mean_B': 'float32',
        'Log_Std_B': 'float32',
        'Price_A': 'float32',
        'Price_B': 'float32',
        'Hedge_Ratio': 'float32',
        'ZScore': 'float32',
        'Unrealized_PnL': 'float32',
        'Realized_PnL': 'float32',
        'Cumulative_PnL': 'float32',
        'Trade_PnL': 'float32',
        'Daily_Delta': 'float32',
    }
    
    df = pd.read_csv(file_path, dtype=dtypes, parse_dates=['Date', 'Period_Start', 'Period_End'])
    return df


# ==========================================
# CALCULATIONS
# ==========================================
def calculate_metrics(df, strategy_name):
    """Calculate performance metrics based on user-defined formulas."""
    
    # 1. Parse number of total pairs from strategy name
    n_total_pairs = 20 # Default fallback
    match = re.search(r'Top(\d+)', strategy_name)
    if match:
        n_total_pairs = int(match.group(1))
        
    c_period = INITIAL_CAPITAL
    c_pair = INITIAL_CAPITAL / n_total_pairs if n_total_pairs > 0 else INITIAL_CAPITAL
    
    # 2. Aggregate to Portfolio Level
    portfolio_daily = df.groupby('Date')['Daily_Delta'].sum().reset_index()
    portfolio_daily['Cumulative_PnL'] = portfolio_daily['Daily_Delta'].cumsum()
    
    final_pnl = portfolio_daily['Cumulative_PnL'].iloc[-1] if not portfolio_daily.empty else 0
    final_equity = INITIAL_CAPITAL + final_pnl
    
    # 3. Annualized Return
    # Assuming ~252 trading days/year
    years = len(portfolio_daily) / 252.0 if not portfolio_daily.empty else 1.0
    if final_equity > 0 and years > 0:
        annualized_return = (final_equity / INITIAL_CAPITAL) ** (1/years) - 1
    else:
        annualized_return = 0
        
    # 4. Sharpe Ratio
    daily_returns = portfolio_daily['Daily_Delta'] / INITIAL_CAPITAL
    if daily_returns.std() != 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0
        
    # 5. Max Drawdown
    roll_max = portfolio_daily['Cumulative_PnL'].cummax()
    drawdown = portfolio_daily['Cumulative_PnL'] - roll_max
    mdd = drawdown.min()
    
    # 6. RCC & REC
    rcc = final_pnl / c_period if c_period > 0 else 0
    
    # N_traded: number of unique pairs that actually entered a position
    traded_pairs_df = df[df['Position'] != 0].groupby(['Ticker_A', 'Ticker_B']).size().reset_index()
    n_traded = len(traded_pairs_df)
    
    engaged_capital = n_traded * c_pair
    rec = final_pnl / engaged_capital if engaged_capital > 0 else 0
    
    # 7. Win Rate & Profit/Loss Ratio
    # Filter only closed trades (where Trade_PnL is recorded)
    closed_trades = df[df['Trade_PnL'] != 0]['Trade_PnL']
    win_trades = closed_trades[closed_trades > 0]
    loss_trades = closed_trades[closed_trades < 0]
    
    win_rate = len(win_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
    
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 0
    pnl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    return {
        'Strategy': strategy_name.split('/')[-1], # Keep it concise for display
        'Full_Path': strategy_name,
        'Final Equity': final_equity,
        'Ann. Return': annualized_return,
        'Sharpe': sharpe,
        'MDD': mdd,
        'RCC': rcc,
        'REC': rec,
        'Win Rate': win_rate,
        'PnL Ratio': pnl_ratio
    }, portfolio_daily


# ==========================================
# MAIN APP
# ==========================================
def main():
    st.markdown('<div class="main-title">Pair Trading Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive Dashboard for Cointegration Backtest Results</div>', unsafe_allow_html=True)
    
    if not os.path.exists(RESULTS_DIR):
        st.error(f"Directory '{RESULTS_DIR}' not found. Please ensure the app is run from the project root.")
        return
        
    available_strategies = scan_strategies(RESULTS_DIR)
    
    if not available_strategies:
        st.warning("No strategy logs found in the results directory.")
        return
        
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### ⚙️ Strategy Selection")
        selected_strategies = st.multiselect(
            "Select up to 5 strategies to compare:",
            options=available_strategies,
            max_selections=5,
            default=available_strategies[:1] if available_strategies else None
        )
        st.markdown("---")
        st.info("💡 **Memory Optimization Active**: Data is lazily loaded and downcasted upon selection.")
        
    if not selected_strategies:
        st.info("👈 Please select at least one strategy from the sidebar to view results.")
        return
        
    # --- DATA PROCESSING ---
    metrics_list = []
    portfolio_dfs = {}
    
    with st.spinner("Crunching numbers and caching data..."):
        for strategy in selected_strategies:
            df = load_data(strategy)
            if df.empty:
                st.error(f"No data found for strategy {strategy}.")
                continue
            
            metrics, portfolio_daily = calculate_metrics(df, strategy)
            metrics_list.append(metrics)
            portfolio_dfs[strategy] = portfolio_daily
            
    if not metrics_list:
        return
        
    # --- HIGHLIGHT METRICS (If single strategy selected) ---
    if len(metrics_list) == 1:
        m = metrics_list[0]
        cols = st.columns(4)
        cols[0].metric("Final Equity", f"${m['Final Equity']:,.2f}", f"{m['Ann. Return']:.2%} Ann.")
        cols[1].metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
        cols[2].metric("RCC / REC", f"{m['RCC']:.2%} / {m['REC']:.2%}")
        cols[3].metric("Win Rate", f"{m['Win Rate']:.2%}", f"{m['PnL Ratio']:.2f} P/L Ratio")
        st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPARISON TABLE ---
    st.markdown("### 📊 Performance Metrics Overview")
    metrics_df = pd.DataFrame(metrics_list).drop(columns=['Full_Path'])
    
    # Format the dataframe for display
    display_df = metrics_df.copy()
    display_df['Final Equity'] = display_df['Final Equity'].map('${:,.2f}'.format)
    display_df['Ann. Return'] = display_df['Ann. Return'].map('{:.2%}'.format)
    display_df['Sharpe'] = display_df['Sharpe'].map('{:.2f}'.format)
    display_df['MDD'] = display_df['MDD'].map('${:,.2f}'.format)
    display_df['RCC'] = display_df['RCC'].map('{:.2%}'.format)
    display_df['REC'] = display_df['REC'].map('{:.2%}'.format)
    display_df['Win Rate'] = display_df['Win Rate'].map('{:.2%}'.format)
    display_df['PnL Ratio'] = display_df['PnL Ratio'].map('{:.2f}'.format)
    
    st.dataframe(display_df.set_index('Strategy'), use_container_width=True)
    
    # --- EQUITY CURVE ---
    st.markdown("<br>### 📈 Cumulative Equity Curve", unsafe_allow_html=True)
    
    # Use Plotly Dark template for modern look
    fig_eq = go.Figure()
    
    # Color palette
    colors = ['#00C9FF', '#FF007A', '#00FFA3', '#FFC800', '#B200FF']
    
    for i, (strategy, pdf) in enumerate(portfolio_dfs.items()):
        short_name = strategy.split('/')[-1]
        fig_eq.add_trace(go.Scatter(
            x=pdf['Date'], 
            y=pdf['Cumulative_PnL'], 
            mode='lines', 
            name=short_name,
            line=dict(width=2, color=colors[i % len(colors)]),
            fill='tozeroy' if len(selected_strategies) == 1 else 'none',
            fillcolor=f"rgba{tuple(list(int(colors[i % len(colors)].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.1])}" if len(selected_strategies) == 1 else None
        ))
    
    fig_eq.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL ($)",
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255,255,255,0.05)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_eq, use_container_width=True)
    
    # --- PARAMETER HEATMAP ---
    if len(selected_strategies) > 1:
        st.markdown("<br>### 🎛️ Parameter Heatmap Analysis", unsafe_allow_html=True)
        
        heatmap_data = []
        for metrics in metrics_list:
            full_path = metrics['Full_Path']
            
            # Extract parameters: Stop Loss (SL) and Z-Score Window (ZWin)
            sl_match = re.search(r'SL(\d+)', full_path)
            zwin_match = re.search(r'ZWin(\d+)', full_path)
            
            sl = int(sl_match.group(1)) if sl_match else "N/A"
            zwin = int(zwin_match.group(1)) if zwin_match else "N/A"
            
            heatmap_data.append({
                'Stop Loss (SL)': str(sl),
                'Z-Score Window (ZWin)': str(zwin),
                'Sharpe Ratio': metrics['Sharpe'],
                'Strategy': metrics['Strategy']
            })
            
        hm_df = pd.DataFrame(heatmap_data)
        
        # We need at least some variation to plot a meaningful heatmap
        if len(hm_df['Stop Loss (SL)'].unique()) > 1 or len(hm_df['Z-Score Window (ZWin)'].unique()) > 1:
            pivot_df = hm_df.pivot_table(
                index='Stop Loss (SL)', 
                columns='Z-Score Window (ZWin)', 
                values='Sharpe Ratio', 
                aggfunc='mean'
            )
            
            fig_hm = px.imshow(
                pivot_df, 
                text_auto='.2f', 
                color_continuous_scale='Mint',
                aspect="auto"
            )
            fig_hm.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("⚠️ Not enough variation in parameters (SL, ZWin) among selected strategies to generate a meaningful heatmap.")

if __name__ == "__main__":
    main()
