import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from scipy import stats

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

def load_master_dataframe_from_db(db_path="results/result.db"):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        import sqlite3
        # 使用 URI 模式與 read-only 模式以確保即使資料庫被寫入鎖定也不會發生讀取崩潰
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        df = pd.read_sql_query("SELECT * FROM strategy_summaries", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading master DataFrame from database: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data_from_db(strategy_path, db_path="results/result.db"):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        query = """
        SELECT Date, Position, Ticker_A, Ticker_B, Daily_Delta, Trade_PnL, Status, 
               Hedge_Ratio, Price_A, Price_B, Days_Held, Period_Start, Period_End 
        FROM trade_logs 
        WHERE strategy_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(strategy_path,))
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
            
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Position'] = df['Position'].fillna(0).astype(int)
        df['Days_Held'] = df['Days_Held'].fillna(0).astype(int)
        df['Ticker_A'] = df['Ticker_A'].fillna('UNKNOWN').astype(str)
        df['Ticker_B'] = df['Ticker_B'].fillna('UNKNOWN').astype(str)
        df = df.sort_values(by=['Ticker_A', 'Ticker_B', 'Date']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading trade logs for {strategy_path}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data(strategy_path):
    # 先試著從 SQLite 資料庫載入
    db_path = os.path.join(RESULTS_DIR, "result.db")
    if os.path.exists(db_path):
        df_db = load_data_from_db(strategy_path, db_path)
        if not df_db.empty:
            return df_db

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
    dataset = "Full" if "full" in path_lower else "Current" if "current" in path_lower else "Full"  # 無前綴預設 Full（DB路徑無 full/ 開頭）
    reentry = "NoReEntry" if "noreentry" in path_lower else "ReEntry" if "reentry" in path_lower else "Unknown"
    
    # 全域提取 VolAdj 屬性
    if "novoladj" in path_lower:
        voladj = "NoVolAdj"
    elif "voladj" in path_lower:
        voladj = "VolAdj"
    else:
        voladj = "N/A"
    
    method = "Unknown"
    # ── 識別順序：包含長名稱的先判斷，防止被短名稱吸收 ──
    if "ssd_basic" in path_lower:
        method = "SSD (Basic)"
    elif "ssd_dtw" in path_lower or "ssd-dtw" in path_lower:
        # SSD_DTW_PCA_NoReEntry 等形式
        is_pca = "pca" in path_lower
        method = "SSD-DTW-PCA (Paper)" if is_pca else "SSD (DTW)"
    elif "ssd" in path_lower:
        if "rolling" in path_lower:
            method = "SSD (Rolling)"
        else:
            method = "SSD"
    elif "eg" in path_lower:
        method = "EG"
    elif "pure_dtw" in path_lower or "pure-dtw" in path_lower:
        method = "Pure_DTW"
    elif "dtw" in path_lower:
        if "paper" in path_lower:
            method = "DTW (Paper)"
        else:
            method = "DTW"
    elif "hdbscan" in path_lower:
        is_ss = "_ss_" in path_lower
        is_cs = "_cs_" in path_lower or "crosssector" in path_lower or "cross_sector" in path_lower or "cross-sector" in path_lower
        is_macro = "_macro_" in path_lower or "macro" in path_lower
        is_ae = "_ae_" in path_lower or "hdbscan_ae" in path_lower
        is_pca = "_pca_" in path_lower or "hdbscan_pca" in path_lower
        is_mf = "_mf_" in path_lower or "multifactor" in path_lower
        
        if is_mf:
            method = "HDBSCAN (CS-MF)" if is_cs else "HDBSCAN (SS-MF)" if is_ss else "HDBSCAN (MF)"
        elif is_macro:
            method = "HDBSCAN (Macro-UMAP)"
        elif is_ss:
            method = "HDBSCAN (SS-PCA)" if is_pca else "HDBSCAN (SS-UMAP)"
        elif is_cs:
            method = "HDBSCAN (CS-PCA)" if is_pca else "HDBSCAN (CS-UMAP)"
        else:
            if is_ae:
                method = "HDBSCAN (AE PCA)" if is_pca else "HDBSCAN (AE UMAP)"
            else:
                method = "HDBSCAN (PCA)" if is_pca else "HDBSCAN (UMAP)"

    top_n = "Top 20"
    match_n = re.search(r'top(\d+)', path_lower)
    if match_n: top_n = f"Top {match_n.group(1)}"

    # ── 修正：先抓 PSL（組合停損），再用負向前瞻抓單配對 SL，避免 PSL 干擾 SL 解析 ──
    sl_pct = "0%"
    match_sl = re.search(r'(?<!p)sl(\d+)', path_lower)   # 不匹配 psl 前綴
    if match_sl: sl_pct = f"{match_sl.group(1)}%"

    zwin = "0"
    match_zwin = re.search(r'zwin(\d+)', path_lower)
    if match_zwin: zwin = match_zwin.group(1)

    # ── 新增：解析 PSL（全域組合停損 %）──
    psl_pct = "0%"
    match_psl = re.search(r'psl(\d+)', path_lower)
    if match_psl: psl_pct = f"{match_psl.group(1)}%"

    # ── 新增：解析 MSR（最大產業配對比例 %）──
    msr_pct = "0%"
    match_msr = re.search(r'msr(\d+)', path_lower)
    if match_msr: msr_pct = f"{match_msr.group(1)}%"

    # ── 新增：解析 DSZ（動態 Z 停損門檻）──
    dsz_val = "0"
    match_dsz = re.search(r'dsz(\d+)', path_lower)
    if match_dsz: dsz_val = match_dsz.group(1)

    return dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val

@st.cache_data(show_spinner=False)
def calculate_metrics_raw(strategy_path):
    df = load_data(strategy_path)
    if df.empty: return None

    dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val = extract_features_from_path(strategy_path)
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
    
    # ── 初始化統計量 ──
    t_stat = t_pval = nw_t_stat = nw_t_pval = np.nan

    if len(portfolio_daily) > 0:
        portfolio_daily_idx = portfolio_daily.set_index('Date')
        monthly_equity = portfolio_daily_idx['Equity'].resample('ME').last().dropna()
        
        if len(monthly_equity) > 0:
            monthly_returns = monthly_equity.pct_change().fillna(0)
            cum_ret = np.prod(1 + monthly_returns) - 1
            n_months = len(monthly_returns)
            ann_ret = ((1 + cum_ret) ** (12 / n_months)) - 1 if n_months > 0 else 0

            # ── 單一樣本 T 檢定（H0: μ ≤ 0，右尾檢定）──
            mr = monthly_returns.values
            n = len(mr)
            if n >= 3 and mr.std(ddof=1) > 0:
                # 標準 t 檢定
                t_result = stats.ttest_1samp(mr, popmean=0, alternative='greater')
                t_stat  = float(t_result.statistic)
                t_pval  = float(t_result.pvalue)

                # Newey-West 調整後標準誤（修正序列自相關）
                # 使用 Bartlett 核，最大滯後 lags = floor(4*(n/100)^(2/9))
                mu_hat  = mr.mean()
                e       = mr - mu_hat           # 殘差序列
                lags_nw = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
                # 計算 Newey-West 長期方差
                gamma0  = float(np.dot(e, e) / n)
                nw_var  = gamma0
                for lag in range(1, lags_nw + 1):
                    w       = 1 - lag / (lags_nw + 1)   # Bartlett 權重
                    gamma_l = float(np.dot(e[lag:], e[:-lag]) / n)
                    nw_var += 2 * w * gamma_l
                nw_se   = np.sqrt(max(nw_var, 1e-20) / n)
                nw_t_stat = float(mu_hat / nw_se)
                # 右尾 p 值（自由度 n-1）
                nw_t_pval = float(1 - stats.t.cdf(nw_t_stat, df=n - 1))
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
        'DATASET': dataset, 'RE-ENTRY': reentry, 'VOL ADJ': voladj,
        'METHOD': method, 'TOP N': top_n, 'STOP LOSS %': sl_pct, 'Z-WINDOW': zwin,
        'PORT SL %': psl_pct, 'MAX SEC %': msr_pct, 'DYN Z': dsz_val,
        'Final_Equity': final_equity, 'RCC_Raw': rcc, 'REC_Raw': rec,
        'Cum_Ret_Raw': cum_ret, 'Ann_Ret_Raw': ann_ret, 'Sharpe_Raw': sharpe, 'MDD_Raw': mdd_pct,
        'Entries': n_entries, 'Exits': n_normal_exits, 'Stop_Losses': n_stop_loss, 'Forced_Closes': n_forced_close,
        'Gross_Profit': gross_profit, 'Gross_Loss': gross_loss,
        'T_Stat': t_stat, 'T_Pval': t_pval,
        'NW_T_Stat': nw_t_stat, 'NW_T_Pval': nw_t_pval,
        '_path': strategy_path
    }

def build_master_dataframe(strategies):
    records = []
    total = len(strategies)
    if total == 0:
        return pd.DataFrame()
        
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    for i, s in enumerate(strategies):
        pct = float((i + 1) / total)
        progress_bar.progress(pct)
        
        # 格式化顯示名稱，更為美觀
        short_name = s.split('/')[-1] if '/' in s else s
        status_text.markdown(f"📊 **Compiling Metrics** ({i+1}/{total}): `{short_name}`")
        
        res = calculate_metrics_raw(s)
        if res: 
            records.append(res)
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def compute_all_ttests(paths_tuple: tuple) -> pd.DataFrame:
    """
    對所有策略整批計算 T 檢定統計量。
    使用 tuple 作為快取 key，將整張 DataFrame 一次快取，
    不再每次重新跨越迫圈。
    DB 路徑：直接用 SQL GROUP BY Date 聚合，只擈日報酬總和。
    """
    db_path = os.path.join(RESULTS_DIR, "result.db")
    use_db  = os.path.exists(db_path)

    # 建立進度展示儲置元件（在函式內初始化，快取命中時不會執行）
    pb  = st.progress(0.0)
    txt = st.empty()
    txt.markdown("🔬 **首次計算 T 檢定統計量**（僅需執行一次，之後快取生效）")

    records = []
    total   = len(paths_tuple)

    if use_db:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60.0)
        sql = """
            SELECT Date, SUM(Daily_Delta) AS Daily_Delta
            FROM trade_logs
            WHERE strategy_id = ?
            GROUP BY Date ORDER BY Date
        """

    for i, path in enumerate(paths_tuple):
        parts  = path.split('/')
        folder = parts[-2] if len(parts) >= 2 else path
        fid    = parts[-1].replace('TradeLogs_', '').replace('.csv', '')[:20]
        txt.markdown(f"&nbsp;&nbsp;🔬 DB查詢中 `({i+1}/{total})` — **{folder}** · `{fid}`")
        pb.progress((i + 1) / total)

        t_stat = t_pval = nw_t_stat = nw_t_pval = np.nan
        try:
            if use_db:
                port_daily = pd.read_sql_query(sql, conn, params=(path,))
                if port_daily.empty:
                    port_daily = None
                else:
                    port_daily['Date'] = pd.to_datetime(port_daily['Date'])
            else:
                port_daily = None

            if port_daily is None:
                file_path = os.path.join(RESULTS_DIR, path)
                if os.path.exists(file_path):
                    df_csv = pd.read_csv(file_path, usecols=['Date', 'Daily_Delta'],
                                         dtype={'Daily_Delta': 'float32'})
                    df_csv['Date'] = pd.to_datetime(df_csv['Date'], errors='coerce')
                    port_daily = df_csv.groupby('Date')['Daily_Delta'].sum().reset_index()

            if port_daily is not None and not port_daily.empty:
                port_daily = port_daily.sort_values('Date').set_index('Date')
                port_daily['Equity'] = INITIAL_CAPITAL + port_daily['Daily_Delta'].cumsum()
                monthly_equity = port_daily['Equity'].resample('ME').last().dropna()
                if len(monthly_equity) >= 3:
                    mr = monthly_equity.pct_change().fillna(0).values
                    n  = len(mr)
                    if mr.std(ddof=1) > 0:
                        res       = stats.ttest_1samp(mr, popmean=0, alternative='greater')
                        t_stat    = float(res.statistic)
                        t_pval    = float(res.pvalue)
                        mu_hat    = mr.mean()
                        e         = mr - mu_hat
                        lags_nw   = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
                        nw_var    = float(np.dot(e, e) / n)
                        for lag in range(1, lags_nw + 1):
                            w = 1 - lag / (lags_nw + 1)
                            nw_var += 2 * w * float(np.dot(e[lag:], e[:-lag]) / n)
                        nw_se     = np.sqrt(max(nw_var, 1e-20) / n)
                        nw_t_stat = float(mu_hat / nw_se)
                        nw_t_pval = float(1 - stats.t.cdf(nw_t_stat, df=n - 1))
        except Exception:
            pass

        records.append({'_path': path,
                        'T_Stat': t_stat, 'T_Pval': t_pval,
                        'NW_T_Stat': nw_t_stat, 'NW_T_Pval': nw_t_pval})

    if use_db:
        conn.close()

    pb.empty()
    txt.empty()
    return pd.DataFrame(records)


def make_desc(row): 
    vol_part = f" · {row['VOL ADJ']}" if 'VOL ADJ' in row and row['VOL ADJ'] not in ['NoVolAdj', 'N/A', ''] else ""
    reentry_part = f" · {row['RE-ENTRY']}" if 'RE-ENTRY' in row and row['RE-ENTRY'] not in ['NoReEntry', ''] else ""
    psl_part = f" · PSL {row['PORT SL %']}" if 'PORT SL %' in row and row['PORT SL %'] not in ['0%', '0.0%', ''] else ""
    dsz_part = f" · DSZ {row['DYN Z']}" if 'DYN Z' in row and row['DYN Z'] not in ['0', '0.0', ''] else ""
    zwin_part = f" · ZWin {row['Z-WINDOW']}" if 'Z-WINDOW' in row and row['Z-WINDOW'] not in ['0', ''] else ""
    return f"{row['DATASET']}{reentry_part}{vol_part} · {row['METHOD']} · {row['TOP N']} · SL {row['STOP LOSS %']} · MSR {row['MAX SEC %']}{zwin_part}{psl_part}{dsz_part}"

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

@st.cache_data(show_spinner=False)
def get_sector_mapping():
    mapping = {}
    # 優先從 data/sp500.db 讀取（有 843 檔，含下市公司，最完整）
    for db_file in ["data/sp500.db", "data/SP500_Current.db"]:
        if os.path.exists(db_file):
            try:
                import sqlite3
                conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
                df = pd.read_sql_query("SELECT Symbol, GICS_Sector FROM Constituents", conn)
                conn.close()
                for k, v in zip(df['Symbol'], df['GICS_Sector']):
                    if k and v:
                        mapping[str(k).strip().upper()] = str(v).strip()
                break # 成功載入最完整的就停止
            except Exception:
                pass
    # 備份：從 data/imputed_sectors.csv 補齊
    csv_file = "data/imputed_sectors.csv"
    if os.path.exists(csv_file):
        try:
            df_csv = pd.read_csv(csv_file)
            for k, v in zip(df_csv['ticker'], df_csv['sector']):
                ticker_upper = str(k).strip().upper()
                if k and v and ticker_upper not in mapping:
                    mapping[ticker_upper] = str(v).strip()
        except Exception:
            pass
    return mapping

def save_ttests_to_db(ttest_df: pd.DataFrame, db_path: str):
    if ttest_df.empty or not os.path.exists(db_path):
        return
    try:
        import sqlite3
        conn = sqlite3.connect(db_path, timeout=60.0)
        cursor = conn.cursor()
        
        # 動態為 strategy_summaries 新增缺失的欄位
        cursor.execute("PRAGMA table_info(strategy_summaries);")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        for col in ['T_Stat', 'T_Pval', 'NW_T_Stat', 'NW_T_Pval']:
            if col not in existing_cols:
                try:
                    cursor.execute(f'ALTER TABLE strategy_summaries ADD COLUMN "{col}" REAL;')
                except Exception:
                    pass
                    
        # 批量更新數據
        update_sql = """
        UPDATE strategy_summaries 
        SET T_Stat = ?, T_Pval = ?, NW_T_Stat = ?, NW_T_Pval = ?
        WHERE _path = ?
        """
        update_data = []
        for _, row in ttest_df.iterrows():
            t_s = float(row['T_Stat']) if pd.notna(row['T_Stat']) else None
            t_p = float(row['T_Pval']) if pd.notna(row['T_Pval']) else None
            nw_t = float(row['NW_T_Stat']) if pd.notna(row['NW_T_Stat']) else None
            nw_p = float(row['NW_T_Pval']) if pd.notna(row['NW_T_Pval']) else None
            update_data.append((t_s, t_p, nw_t, nw_p, row['_path']))
            
        cursor.executemany(update_sql, update_data)
        conn.commit()
        conn.close()
    except Exception:
        pass

def render_pair_consistency():
    db_path = os.path.join(RESULTS_DIR, "result.db")
    st.markdown("### 🤝 股票對分析與一致性比對 (Pair Consistency Analysis)")
    st.markdown("此工具讓您直接從資料庫查詢各策略所選出的股票對，並進行一致性與交集分析，無須掃描億級每日交易明細。")
    
    if not os.path.exists(db_path):
        st.warning("找不到 SQLite 資料庫，請先執行歷史回填或回測以建立資料。")
        return
        
    def format_strategy_id(path):
        try:
            dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val = extract_features_from_path(path)
            vol_part = f" · {voladj}" if voladj != 'N/A' else ""
            psl_part = f" · PSL {psl_pct}" if psl_pct != '0%' else ""
            msr_part = f" · MSR {msr_pct}" if msr_pct != '0%' else ""
            dsz_part = f" · DSZ {dsz_val}" if dsz_val != '0' else ""
            desc = f"{dataset} · {reentry}{vol_part} · {method} · {top_n} · SL {sl_pct} · ZWin {zwin}{psl_part}{msr_part}{dsz_part}"
            short_name = path.split('/')[-1] if '/' in path else path
            return f"{desc} ({short_name})"
        except Exception:
            return path

    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        cursor = conn.cursor()
        
        # 檢查 strategy_pairs 表是否存在且有資料
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_pairs';")
        if not cursor.fetchone():
            st.info("資料庫中尚未建立 strategy_pairs 資料表，請先執行回補指令碼。")
            conn.close()
            return
            
        cursor.execute("SELECT count(*) FROM strategy_pairs;")
        total_pairs = cursor.fetchone()[0]
        if total_pairs == 0:
            st.info("💡 股票配對表 (strategy_pairs) 目前為空。請稍候，背景回補程式正在提取配對紀錄。")
            conn.close()
            return
            
        st.markdown(f"**目前配對資料庫中共存有 `{total_pairs:,}` 筆獨特配對選擇紀錄。**")
        
        # 1. 策略配對明細 (Strategy Pair Details)
        st.markdown("---")
        st.markdown("#### 🔍 策略配對明細查詢")
        
        # 取得所有策略路徑
        cursor.execute("SELECT DISTINCT strategy_id FROM strategy_pairs;")
        all_strat_ids = sorted([row[0] for row in cursor.fetchall()])
        
        selected_strat = st.selectbox(
            "選擇要查詢的策略 (Select Strategy ID)", 
            all_strat_ids, 
            format_func=format_strategy_id,
            key="sb_strat_details"
        )
        
        # 獲取產業分類對照表
        sec_map = get_sector_mapping()
        
        if selected_strat:
            pairs_query = """
            SELECT Ticker_A as "股票 A", Ticker_B as "股票 B", Period_Start as "交易期起點", Period_End as "交易期終點", Hedge_Ratio as "避險比例"
            FROM strategy_pairs
            WHERE strategy_id = ?
            ORDER BY Period_Start DESC
            """
            strat_pairs_df = pd.read_sql_query(pairs_query, conn, params=(selected_strat,))
            
            # 分別映射 股票 A 與 股票 B 的產業板塊
            strat_pairs_df.insert(2, "股票 A 產業", strat_pairs_df["股票 A"].apply(lambda x: sec_map.get(str(x).upper(), "Unknown")))
            strat_pairs_df.insert(3, "股票 B 產業", strat_pairs_df["股票 B"].apply(lambda x: sec_map.get(str(x).upper(), "Unknown")))
            
            st.markdown(f"該策略共選出 **{len(strat_pairs_df)}** 個滾動配對關係：")
            st.dataframe(strat_pairs_df, use_container_width=True, hide_index=True)
            
        # 2. 策略配對交集比對 (Strategy Pair Intersection)
        st.markdown("---")
        st.markdown("#### 🤝 策略選股配對交集比對 (Strategy Pair Intersection)")
        st.markdown("選取兩個策略，直接找出它們**共同選中的股票對（交集）**與各自獨有擁有的股票對，並顯示各自的交易期間 (年月日)。")
        
        col_venn1, col_venn2 = st.columns(2)
        with col_venn1:
            strat_a = st.selectbox(
                "選擇第一個策略 (Strategy A)", 
                all_strat_ids, 
                index=0 if all_strat_ids else None, 
                format_func=format_strategy_id,
                key="sb_strat_a"
            )
        with col_venn2:
            strat_b = st.selectbox(
                "選擇第二個策略 (Strategy B)", 
                all_strat_ids, 
                index=(1 if len(all_strat_ids) > 1 else 0) if all_strat_ids else None, 
                format_func=format_strategy_id,
                key="sb_strat_b"
            )
            
        if strat_a and strat_b:
            cursor.execute("SELECT DISTINCT Ticker_A, Ticker_B FROM strategy_pairs WHERE strategy_id = ?;", (strat_a,))
            pairs_a = {(row[0], row[1]) for row in cursor.fetchall()}
            
            cursor.execute("SELECT DISTINCT Ticker_A, Ticker_B FROM strategy_pairs WHERE strategy_id = ?;", (strat_b,))
            pairs_b = {(row[0], row[1]) for row in cursor.fetchall()}
            
            intersection = pairs_a.intersection(pairs_b)
            union = pairs_a.union(pairs_b)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Strategy A 總配對數", f"{len(pairs_a)}")
            col_m2.metric("Strategy B 總配對數", f"{len(pairs_b)}")
            col_m3.metric("🤝 共同交集配對", f"{len(intersection)}", f"{len(intersection)/len(union):.1%} 重合度" if len(union) > 0 else "0.0%")
            col_m4.metric("📊 聯集總計配對", f"{len(union)}")
            
            if intersection:
                st.markdown(f"##### 🤝 共同選出的股票對 ({len(intersection)})：")
                
                # 查詢 Strategy A 的所有明細
                cursor.execute("""
                    SELECT Ticker_A, Ticker_B, Period_Start, Period_End
                    FROM strategy_pairs
                    WHERE strategy_id = ?
                """, (strat_a,))
                rows_a = cursor.fetchall()
                
                # 查詢 Strategy B 的所有明細
                cursor.execute("""
                    SELECT Ticker_A, Ticker_B, Period_Start, Period_End
                    FROM strategy_pairs
                    WHERE strategy_id = ?
                """, (strat_b,))
                rows_b = cursor.fetchall()
                
                # 建立 A 期間字典
                periods_a = {}
                for row in rows_a:
                    pair_key = (row[0], row[1])
                    if pair_key in intersection:
                        p_str = f"{row[2]} ~ {row[3]}"
                        periods_a.setdefault(pair_key, []).append(p_str)
                        
                # 建立 B 期間字典
                periods_b = {}
                for row in rows_b:
                    pair_key = (row[0], row[1])
                    if pair_key in intersection:
                        p_str = f"{row[2]} ~ {row[3]}"
                        periods_b.setdefault(pair_key, []).append(p_str)
                
                # 建立展示用的 DataFrame
                intersection_rows = []
                for p in sorted(list(intersection)):
                    t_a, t_b = p
                    sec_a = sec_map.get(t_a.upper(), "Unknown")
                    sec_b = sec_map.get(t_b.upper(), "Unknown")
                    
                    # 排序期間以確保時間先後順序
                    p_list_a = sorted(list(set(periods_a.get(p, []))))
                    p_list_b = sorted(list(set(periods_b.get(p, []))))
                    
                    p_str_a = ", ".join(p_list_a) if p_list_a else "-"
                    p_str_b = ", ".join(p_list_b) if p_list_b else "-"
                    
                    intersection_rows.append({
                        "股票 A": t_a,
                        "股票 B": t_b,
                        "股票 A 產業": sec_a,
                        "股票 B 產業": sec_b,
                        "Strategy A 交易期間 (年月日)": p_str_a,
                        "Strategy B 交易期間 (年月日)": p_str_b
                    })
                    
                intersection_df = pd.DataFrame(intersection_rows)
                st.dataframe(intersection_df, use_container_width=True, hide_index=True)
            else:
                st.info("兩個策略之間沒有共同選出的股票對。")
                
        conn.close()
    except Exception as e:
        st.error(f"Error rendering pair consistency analysis: {e}")

def main():
    # ── 模組導覽列 ───────────────────────────────────────
    st.write("") # 留空
    tab_selection = st.radio(
        "選擇功能模組", 
        ["📊 策略績效與回測比較 (Strategy Performance)", "🤝 股票對分析與一致性比對 (Pair Consistency Analysis)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if "一致性比對" in tab_selection:
        render_pair_consistency()
        return

    st.markdown('<div class="blue-subtitle">QUANTITATIVE PERFORMANCE DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Pairs Trading Comparison Dashboard</div>', unsafe_allow_html=True)
    
    db_path = os.path.join(RESULTS_DIR, "result.db")
    master_df = pd.DataFrame()
    use_db = False
    
    if os.path.exists(db_path):
        master_df = load_master_dataframe_from_db(db_path)
        if not master_df.empty:
            use_db = True
            available_strategies = master_df['_path'].tolist()

            # ── 每次載入後重新解析路徑欄位 ──
            # 只覆蓋可從路徑字串直接推導的欄位（METHOD、RE-ENTRY、TOP N 等）。
            # DATASET 優先保留 DB 的值（因為 DB 路徑沒有 full/ 前綴，
            # 純路徑解析無法區分 Full / Current，由回測寫入時最準確）。
            parsed = master_df['_path'].apply(
                lambda p: extract_features_from_path(p)
            )
            parsed_df = pd.DataFrame(
                parsed.tolist(),
                columns=['DATASET', 'RE-ENTRY', 'VOL ADJ', 'METHOD',
                         'TOP N', 'STOP LOSS %', 'Z-WINDOW',
                         'PORT SL %', 'MAX SEC %', 'DYN Z'],
                index=master_df.index
            )
            # ── 每次載入後對 DB 缺失/Unknown 的欄位進行回補 ──
            # 優先保留 DB 中的真實值（如 HDBSCAN (SS-UMAP) 等），僅在資料庫值為 'Unknown'、空值或缺失時才用路徑解析值進行回補。
            for col in ['DATASET', 'RE-ENTRY', 'VOL ADJ', 'METHOD', 'TOP N', 'STOP LOSS %', 'Z-WINDOW', 'PORT SL %', 'MAX SEC %', 'DYN Z']:
                if col in master_df.columns:
                    need_fill = master_df[col].isna() | master_df[col].isin(['Unknown', ''])
                    master_df.loc[need_fill, col] = parsed_df.loc[need_fill, col]
            
    if not use_db:
        available_strategies = scan_strategies(RESULTS_DIR)
        
    st.markdown(f'<div class="sub-title">Full dataset: {len(available_strategies)} strategy combinations currently loaded.</div>', unsafe_allow_html=True)
    
    if not available_strategies:
        st.warning("No strategy logs found in the results directory.")
        return
        
    t_test_needed = False
    missing_paths = []
    
    if use_db:
        # 已從 SQLite 資料庫極速載入完成。
        db_has_ttest_cols = 'T_Stat' in master_df.columns
        if not db_has_ttest_cols:
            # 資料庫缺少 T 檢定欄位，動態建立為 NaN 欄位以達到極速啟動
            for col in ['T_Stat', 'T_Pval', 'NW_T_Stat', 'NW_T_Pval']:
                master_df[col] = np.nan
            t_test_needed = True
            missing_paths = master_df['_path'].tolist()
        else:
            # 檢查是否有缺失的部分（例如增量更新）
            isna_mask = master_df['T_Stat'].isna()
            if isna_mask.any():
                missing_paths = master_df.loc[isna_mask, '_path'].tolist()
                # 如果是全部都缺失，就不在啟動時自動計算，避免卡住
                if len(missing_paths) == len(master_df):
                    t_test_needed = True
                else:
                    # 只有少部分缺失，可以進行背景/即時增量更新
                    with st.spinner(f"正在增量計算 {len(missing_paths)} 組新策略的 T 檢定..."):
                        ttest_df = compute_all_ttests(tuple(missing_paths))
                        save_ttests_to_db(ttest_df, db_path)
                        # 更新 master_df 中的值
                        master_df.set_index('_path', inplace=True)
                        ttest_df.set_index('_path', inplace=True)
                        for col in ['T_Stat', 'T_Pval', 'NW_T_Stat', 'NW_T_Pval']:
                            master_df.update(ttest_df[[col]])
                        master_df.reset_index(inplace=True)
    else:
        with st.spinner("Compiling massive dataset metrics..."):
            master_df = build_master_dataframe(available_strategies)
    
    if t_test_needed:
        st.sidebar.warning("⚠️ 目前資料庫中缺少 T 檢定快取數據。")
        if st.sidebar.button("🔬 立即計算並快取 T 檢定統計量", use_container_width=True):
            with st.spinner("正在計算所有策略的 T 檢定統計量，並寫入資料庫快取..."):
                ttest_df = compute_all_ttests(tuple(missing_paths))
                save_ttests_to_db(ttest_df, db_path)
            st.sidebar.success("🎉 T 檢定統計量計算並寫入完畢！")
            st.rerun()
            
    if master_df.empty:
        st.error("Could not parse any valid strategy data.")
        return

    # ── 動態篩選：只渲染有兩個以上唯一值的欄位 ──
    FILTER_DEFS = [
        ('DATASET',    'DATASET'),
        ('RE-ENTRY',   'RE-ENTRY'),
        ('VOL ADJ',    'VOL ADJ'),
        ('METHOD',     'METHOD'),
        ('TOP N',      'TOP N'),
        ('STOP LOSS %','STOP LOSS %'),
        ('Z-WINDOW',   'Z-WIN'),
        ('PORT SL %',  'PORT SL % (全域組合停損)'),
        ('MAX SEC %',  'MAX SEC % (產業上限)'),
        ('DYN Z',      'DYN Z (動態 Z 停損)'),
    ]
    # 只保留有超過 1 個唯一值的欄位
    active_filters = [
        (col, label) for col, label in FILTER_DEFS
        if col in master_df.columns and master_df[col].nunique() > 1
    ]

    sel_vals = {}   # col -> 選擇值
    if active_filters:
        st.markdown("### Filters")
        # 每行最多 4 個
        COLS_PER_ROW = 4
        for row_start in range(0, len(active_filters), COLS_PER_ROW):
            row_filters = active_filters[row_start : row_start + COLS_PER_ROW]
            cols_ui = st.columns(len(row_filters))
            for ui_col, (col, label) in zip(cols_ui, row_filters):
                opts = ["All"] + sorted(master_df[col].unique(), key=natural_sort_key)
                sel_vals[col] = ui_col.selectbox(label, opts, key=f"filter_{col}")

    filtered_df = master_df.copy()
    for col, chosen in sel_vals.items():
        if chosen != "All":
            filtered_df = filtered_df[filtered_df[col] == chosen]

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
            'DATASET': '-', 'RE-ENTRY': '-', 'VOL ADJ': '-', 'METHOD': '-', 'TOP N': '-', 'STOP LOSS %': '-', 'Z-WINDOW': '-',
            'PORT SL %': '-', 'MAX SEC %': '-', 'DYN Z': '-',
            'T_Stat': np.nan, 'T_Pval': np.nan, 'NW_T_Stat': np.nan, 'NW_T_Pval': np.nan
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

    display_df['FINAL EQUITY ($)'] = display_df['Final_Equity']
    display_df['CUM. RETURN (%)'] = display_df['Cum_Ret_Raw']
    display_df['ANN. RETURN (%)'] = display_df['Ann_Ret_Raw']
    display_df['RCC (%)'] = display_df['RCC_Raw']
    display_df['REC (%)'] = display_df['REC_Raw']
    display_df['SHARPE'] = display_df['Sharpe_Raw']
    display_df['MAX DRAWDOWN (%)'] = display_df['MDD_Raw']
    display_df['ENTRIES (Count)'] = display_df['Entries']
    display_df['EXITS (Count)'] = display_df['Exits']
    display_df['STOP LOSSES (Count)'] = display_df['Stop_Losses'].apply(lambda x: f"{int(x):,}" if x >= 0 else "N/A")
    display_df['FORCED CLOSES (Count)'] = display_df['Forced_Closes']
    display_df['GROSS PROFIT ($)'] = display_df['Gross_Profit']
    display_df['GROSS LOSS ($)'] = display_df['Gross_Loss']
    # ── T 檢定欄位（月報酬率 H0: μ≤0 右尾檢定）──
    # 使用 .get() 防止舊版 SQLite 快取缺少欄位時拋出 KeyError
    _nan_col = np.nan
    display_df['T-STAT']    = display_df.get('T_Stat',    _nan_col)
    display_df['T-PVAL']    = display_df.get('T_Pval',    _nan_col)
    display_df['NW T-STAT'] = display_df.get('NW_T_Stat', _nan_col)
    display_df['NW T-PVAL'] = display_df.get('NW_T_Pval', _nan_col)

    # ------------------------------------------
    # CONTROLS FOR DYNAMIC COLUMN DISPLAY
    # ------------------------------------------
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        expand_config = st.toggle("顯示完整參數欄位 (Separate Config Columns)", value=False, help="預設將參數合併為單一 Strategic Config 欄位以節省空間。開啟此選項可分開顯示各參數欄位。")
    with col_t2:
        show_detailed_metrics = st.toggle("顯示詳細效能指標 (Detailed Metrics)", value=False, help="預設只顯示最關鍵的核心效能指標（Final Equity, Ann. Return, Sharpe, Max Drawdown）。開啟此選項可顯示更多詳細指標與交易計數。")

    display_df['STRATEGY CONFIG'] = display_df.apply(make_desc, axis=1)

    # 決定顯示欄位
    if expand_config:
        config_cols = [col for col, _ in FILTER_DEFS if col in master_df.columns and master_df[col].nunique() > 1]
    else:
        config_cols = ['STRATEGY CONFIG']

    if show_detailed_metrics:
        metrics_cols = [
            'FINAL EQUITY ($)', 'CUM. RETURN (%)', 'ANN. RETURN (%)', 'RCC (%)', 'REC (%)', 'SHARPE', 'MAX DRAWDOWN (%)',
            'T-STAT', 'T-PVAL', 'NW T-STAT', 'NW T-PVAL',
            'ENTRIES (Count)', 'EXITS (Count)', 'STOP LOSSES (Count)', 'FORCED CLOSES (Count)', 'GROSS PROFIT ($)', 'GROSS LOSS ($)'
        ]
    else:
        metrics_cols = [
            'FINAL EQUITY ($)', 'ANN. RETURN (%)', 'SHARPE', 'MAX DRAWDOWN (%)',
            'T-STAT', 'T-PVAL', 'NW T-STAT', 'NW T-PVAL'
        ]

    cols = config_cols + metrics_cols

    # 動態格式化與著色，避免 subset KeyError
    # ── p 值格式化：NaN 顯示為 N/A ──
    def fmt_pval(v):
        try:
            if np.isnan(v): return 'N/A'
            return f'{v:.4f}'
        except Exception:
            return 'N/A'

    def fmt_tstat(v):
        try:
            if np.isnan(v): return 'N/A'
            return f'{v:.3f}'
        except Exception:
            return 'N/A'

    all_formats = {
        'FINAL EQUITY ($)': '${:,.2f}',
        'CUM. RETURN (%)': '{:.2%}', 
        'ANN. RETURN (%)': '{:.2%}',
        'RCC (%)': '{:.2%}', 
        'REC (%)': '{:.2%}', 
        'SHARPE': '{:.2f}', 
        'MAX DRAWDOWN (%)': '{:.2%}',
        'T-STAT':    fmt_tstat,
        'T-PVAL':    fmt_pval,
        'NW T-STAT': fmt_tstat,
        'NW T-PVAL': fmt_pval,
        'ENTRIES (Count)': '{:,.0f}',
        'EXITS (Count)': '{:,.0f}',
        'FORCED CLOSES (Count)': '{:,.0f}',
        'GROSS PROFIT ($)': '${:,.2f}',
        'GROSS LOSS ($)': '${:,.2f}'
    }
    active_formats = {k: v for k, v in all_formats.items() if k in cols}
    
    df_styled = display_df[cols].style.format(active_formats)
    
    # 數值正負著色函數 (正數綠字、負數紅字)
    def color_by_value(val):
        try:
            numeric_val = float(val)
            if numeric_val > 0:
                return 'color: #4ade80; font-weight: bold;'
            elif numeric_val < 0:
                return 'color: #f87171; font-weight: bold;'
        except (ValueError, TypeError):
            pass
        return ''

    # 帳戶權益著色函數 (以 10000 初始資金為基準)
    def color_equity(val):
        try:
            numeric_val = float(val)
            if numeric_val > 10000.0:
                return 'color: #4ade80; font-weight: bold;'
            elif numeric_val < 10000.0:
                return 'color: #f87171; font-weight: bold;'
        except (ValueError, TypeError):
            pass
        return ''

    # 套用著色
    if 'FINAL EQUITY ($)' in cols:
        df_styled = df_styled.map(color_equity, subset=['FINAL EQUITY ($)'])

    value_color_cols = [
        'CUM. RETURN (%)', 'ANN. RETURN (%)', 'RCC (%)', 'REC (%)', 
        'SHARPE', 'MAX DRAWDOWN (%)', 'GROSS PROFIT ($)', 'GROSS LOSS ($)',
        'T-STAT', 'NW T-STAT'
    ]
    for col in value_color_cols:
        if col in cols:
            df_styled = df_styled.map(color_by_value, subset=[col])

    # p 值著色：< 0.05 綠色顯著，>= 0.05 正常色
    def color_pval(v):
        try:
            fv = float(v)
            if fv < 0.05:  return 'color: #4ade80; font-weight: bold;'
            if fv < 0.10:  return 'color: #fbd38d;'
        except Exception:
            pass
        return ''
    for pcol in ['T-PVAL', 'NW T-PVAL']:
        if pcol in cols:
            df_styled = df_styled.map(color_pval, subset=[pcol])

    # 精美欄位配置，自訂最佳寬度
    column_config = {
        "STRATEGY CONFIG": st.column_config.TextColumn("Strategy Config 🛠️", width="large"),
        "DATASET": st.column_config.TextColumn("Dataset 📁", width="small"),
        "RE-ENTRY": st.column_config.TextColumn("Re-Entry 🔄", width="small"),
        "VOL ADJ": st.column_config.TextColumn("Vol Adj ⚡", width="small"),
        "METHOD": st.column_config.TextColumn("Method 🧮", width="medium"),
        "TOP N": st.column_config.TextColumn("Top N 🏆", width="small"),
        "STOP LOSS %": st.column_config.TextColumn("Stop Loss 📉", width="small"),
        "Z-WINDOW": st.column_config.TextColumn("Z-Win ⏱️", width="small"),
        "PORT SL %": st.column_config.TextColumn("Port SL % 🛡️", width="small"),
        "MAX SEC %": st.column_config.TextColumn("Max Sec % 🏭", width="small"),
        "DYN Z": st.column_config.TextColumn("Dyn Z 🎯", width="small"),
        "T-STAT":    st.column_config.TextColumn("T-Stat 📐",    width="small"),
        "T-PVAL":    st.column_config.TextColumn("p-val 📐",     width="small"),
        "NW T-STAT": st.column_config.TextColumn("NW T-Stat 🔬", width="small"),
        "NW T-PVAL": st.column_config.TextColumn("NW p-val 🔬",  width="small"),
        
        "FINAL EQUITY ($)": st.column_config.TextColumn("Final Equity 💰", width="small"),
        "CUM. RETURN (%)": st.column_config.TextColumn("Cum. Return 📈", width="small"),
        "ANN. RETURN (%)": st.column_config.TextColumn("Ann. Return ⚡", width="small"),
        "RCC (%)": st.column_config.TextColumn("RCC 📊", width="small"),
        "REC (%)": st.column_config.TextColumn("REC 📈", width="small"),
        "SHARPE": st.column_config.TextColumn("Sharpe 📊", width="small"),
        "MAX DRAWDOWN (%)": st.column_config.TextColumn("Max DD ⚠️", width="small"),
        
        "ENTRIES (Count)": st.column_config.TextColumn("Entries 🔑", width="small"),
        "EXITS (Count)": st.column_config.TextColumn("Exits 🚪", width="small"),
        "STOP LOSSES (Count)": st.column_config.TextColumn("Stop Losses 🚨", width="small"),
        "FORCED CLOSES (Count)": st.column_config.TextColumn("Forced Closes 🛡️", width="small"),
        "GROSS PROFIT ($)": st.column_config.TextColumn("Gross Profit 🟢", width="small"),
        "GROSS LOSS ($)": st.column_config.TextColumn("Gross Loss 🔴", width="small"),
    }
    active_config = {k: v for k, v in column_config.items() if k in cols}

    event = st.dataframe(
        df_styled, width='stretch', hide_index=True, height=350,
        on_select="rerun", selection_mode="multi-row",
        column_config=active_config
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