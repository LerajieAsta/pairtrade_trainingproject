import streamlit as st
import os
import pandas as pd
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
# CUSTOM CSS
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
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
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
from strategies.config import INITIAL_CAPITAL, FORWARD_DAYS, rolling_step, CONCURRENT_PERIODS

# 最大並行部署資本（用於 daily return 正規化）= INITIAL_CAPITAL（run_trading 已修正）
_CONCURRENT_PERIODS = CONCURRENT_PERIODS


def _portfolio_capital() -> float:
    """正規化用的總部署資本基準（= INITIAL_CAPITAL，因為 run_trading 的 PM
    已將 max_pairs 乘以 CONCURRENT_PERIODS，Daily_Delta 總量不超過 INITIAL_CAPITAL）
    """
    return INITIAL_CAPITAL


def natural_sort_key(s):
    return [int(t) if t.isdigit() else str(t).lower() for t in re.split(r'(\d+)', str(s))]


def scan_strategies(base_dir):
    strategies = []
    if not os.path.exists(base_dir):
        return strategies
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "TradeLogs" in file or "detailed_trade_logs" in file:
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                strategies.append(rel_path.replace("\\", "/"))
    return sorted(strategies)


@st.cache_data(ttl=60, show_spinner=False)
def load_master_dataframe_from_db(db_path="results/result.db"):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        df = pd.read_sql_query("SELECT * FROM strategy_summaries", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading master DataFrame: {e}")
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
        FROM trade_logs WHERE strategy_id = ?
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
        df = df.sort_values(['Ticker_A', 'Ticker_B', 'Date']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading trade logs for {strategy_path}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data(strategy_path):
    db_path = os.path.join(RESULTS_DIR, "result.db")
    if os.path.exists(db_path):
        df_db = load_data_from_db(strategy_path, db_path)
        if not df_db.empty:
            return df_db

    file_path = os.path.join(RESULTS_DIR, strategy_path)
    if not os.path.exists(file_path):
        if strategy_path.lower().startswith("full/"):
            sub_path = strategy_path[5:]
            for alt in [os.path.join(RESULTS_DIR, "tiingo", sub_path),
                        os.path.join(RESULTS_DIR, "current", sub_path)]:
                if os.path.exists(alt):
                    file_path = alt
                    break
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    try:
        sample = pd.read_csv(file_path, nrows=0)
        original_header = sample.columns.tolist()
        clean_header = [str(c).strip() for c in original_header]
        target_cols = ['Date', 'Position', 'Ticker_A', 'Ticker_B', 'Daily_Delta',
                       'Trade_PnL', 'Status', 'Hedge_Ratio', 'Price_A', 'Price_B',
                       'Days_Held', 'Period_Start', 'Period_End']
        col_map = {orig: clean for orig, clean in zip(original_header, clean_header)
                   if clean in target_cols}
        dtypes_target = {
            'Position': 'float32', 'Daily_Delta': 'float32', 'Trade_PnL': 'float32',
            'Hedge_Ratio': 'float32', 'Price_A': 'float32', 'Price_B': 'float32',
            'Days_Held': 'float32', 'Status': 'category',
            'Period_Start': 'string', 'Period_End': 'string'
        }
        dtypes_to_use = {orig: dtypes_target[clean]
                         for orig, clean in col_map.items() if clean in dtypes_target}
        df = pd.read_csv(file_path, usecols=list(col_map.keys()),
                         dtype=dtypes_to_use, parse_dates=False)
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
            df = df.sort_values(['Ticker_A', 'Ticker_B', 'Date']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading {strategy_path}: {str(e)}")
        return pd.DataFrame()


def extract_features_from_path(path):
    path_lower = path.lower()
    dataset = "Current" if "current" in path_lower else "Tiingo" if "tiingo" in path_lower else "Full"
    reentry = "NoReEntry" if "noreentry" in path_lower else "ReEntry" if "reentry" in path_lower else "Unknown"

    if "novoladj" in path_lower:
        voladj = "NoVolAdj"
    elif "voladj" in path_lower:
        voladj = "VolAdj"
    else:
        voladj = "N/A"

    method = "Unknown"
    if "ensemble" in path_lower:
        if "hdbscan" in path_lower:
            method = "Ensemble (HDBSCAN)"
        elif "ssd" in path_lower and "dtw" in path_lower:
            method = "Ensemble (SSD-DTW)"
        else:
            method = "Ensemble"
    elif "ssd_basic" in path_lower:
        method = "SSD (Basic)"
    elif "ssd_dtw" in path_lower or "ssd-dtw" in path_lower:
        method = "SSD-DTW-PCA (Paper)" if "pca" in path_lower else "SSD (DTW)"
    elif "ssd" in path_lower:
        method = "SSD (Rolling)" if "rolling" in path_lower else "SSD"
    elif "eg" in path_lower:
        method = "EG"
    elif "pure_dtw" in path_lower or "pure-dtw" in path_lower:
        method = "Pure_DTW"
    elif "dtw" in path_lower:
        method = "DTW (Paper)" if "paper" in path_lower else "DTW"
    elif "hdbscan" in path_lower:
        is_ss = "_ss_" in path_lower
        is_cs = ("_cs_" in path_lower or "crosssector" in path_lower
                 or "cross_sector" in path_lower or "cross-sector" in path_lower)
        is_macro = "_macro_" in path_lower or "macro" in path_lower
        is_ae = "_ae_" in path_lower or "hdbscan_ae" in path_lower
        is_pca_umap = "pca_umap" in path_lower or "pcaumap" in path_lower
        is_pca = ("_pca_" in path_lower or "hdbscan_pca" in path_lower) and not is_pca_umap
        is_mf = "_mf_" in path_lower or "multifactor" in path_lower
        is_multiscale = "multiscale" in path_lower or "multi_scale" in path_lower
        if is_mf:
            method = "HDBSCAN (CS-MF)" if is_cs else "HDBSCAN (SS-MF)" if is_ss else "HDBSCAN (MF)"
        elif is_macro:
            method = "HDBSCAN (Macro-UMAP)"
        elif is_pca_umap:
            method = "HDBSCAN MultiScale (PCA-UMAP)" if is_multiscale else "HDBSCAN (PCA-UMAP)"
        elif is_ss:
            method = "HDBSCAN (SS-PCA)" if is_pca else "HDBSCAN (SS-UMAP)"
        elif is_cs:
            method = "HDBSCAN (CS-PCA)" if is_pca else "HDBSCAN (CS-UMAP)"
        elif is_ae:
            method = "HDBSCAN (AE PCA)" if is_pca else "HDBSCAN (AE UMAP)"
        elif is_multiscale:
            method = "HDBSCAN MultiScale (PCA)" if is_pca else "HDBSCAN MultiScale"
        else:
            method = "HDBSCAN (PCA)" if is_pca else "HDBSCAN (UMAP)"
    elif "drl" in path_lower:
        method = "DRL (LSTM)" if "lstm" in path_lower else "DRL"

    top_n = "Top 20"
    m = re.search(r'top(\d+)', path_lower)
    if m:
        top_n = f"Top {m.group(1)}"

    sl_pct = "0%"
    m = re.search(r'(?<!p)sl(\d+)', path_lower)
    if m:
        sl_pct = f"{m.group(1)}%"

    zwin = "0"
    m = re.search(r'zwin(\d+)', path_lower)
    if m:
        zwin = m.group(1)

    psl_pct = "0%"
    m = re.search(r'psl(\d+)', path_lower)
    if m:
        psl_pct = f"{m.group(1)}%"

    msr_pct = "0%"
    m = re.search(r'msr(\d+)', path_lower)
    if m:
        msr_pct = f"{m.group(1)}%"

    dsz_val = "0"
    m = re.search(r'dsz(\d+)', path_lower)
    if m:
        dsz_val = m.group(1)

    return dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val


@st.cache_data(show_spinner=False)
def calculate_metrics_raw(strategy_path):
    df = load_data(strategy_path)
    if df.empty:
        return None

    dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val = \
        extract_features_from_path(strategy_path)
    top_n_int = int(top_n.replace('Top ', '')) if 'Top' in top_n else 20
    c_period = _portfolio_capital()   # 任意時點最大並行部署資本（6 × $10,000）

    if 'Daily_Delta' in df.columns:
        portfolio_daily = df.groupby('Date')['Daily_Delta'].sum().reset_index()
        portfolio_daily = portfolio_daily.sort_values('Date').reset_index(drop=True)
    else:
        portfolio_daily = pd.DataFrame({'Date': df['Date'].unique(), 'Daily_Delta': 0})

    portfolio_daily['Cumulative_PnL'] = portfolio_daily['Daily_Delta'].cumsum()
    # 複利資本模型：PM 已確保 Daily_Delta 總量 ≤ current_equity，直接累加即反映真實複利
    portfolio_daily['Equity'] = INITIAL_CAPITAL + portfolio_daily['Cumulative_PnL']

    final_pnl = float(portfolio_daily['Cumulative_PnL'].iloc[-1]) if not portfolio_daily.empty else 0.0
    final_equity = INITIAL_CAPITAL + final_pnl

    t_stat = t_pval = nw_t_stat = nw_t_pval = np.nan
    cum_ret = ann_ret = mdd_pct = 0.0
    sharpe_full = sharpe_active = 0.0

    if len(portfolio_daily) > 0:
        portfolio_daily_idx = portfolio_daily.set_index('Date')
        monthly_equity = portfolio_daily_idx['Equity'].resample('ME').last().dropna()

        if len(monthly_equity) > 1:
            # dropna() 後 pct_change 第一個仍 NaN，再 fillna(0) 不影響 prod (1+0=1)
            monthly_returns = monthly_equity.pct_change().fillna(0)
            cum_ret = float(np.prod(1 + monthly_returns) - 1)
            n_months = len(monthly_returns)
            if n_months > 0:
                base = 1 + cum_ret
                if base > 0:
                    ann_ret = float(base ** (12 / n_months) - 1)
                else:
                    # 累積虧損超過 100%，無法計算幾何年化報酬，改用算術近似
                    ann_ret = float(cum_ret * (12 / n_months))
            else:
                ann_ret = 0.0

            mr = monthly_returns.values
            n = len(mr)
            if n >= 3 and mr.std(ddof=1) > 0:
                t_result = stats.ttest_1samp(mr, popmean=0, alternative='greater')
                t_stat = float(t_result.statistic)
                t_pval = float(t_result.pvalue)
                mu_hat = mr.mean()
                e = mr - mu_hat
                lags_nw = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
                gamma0 = float(np.dot(e, e) / n)
                nw_var = gamma0
                for lag in range(1, lags_nw + 1):
                    w = 1 - lag / (lags_nw + 1)
                    nw_var += 2 * w * float(np.dot(e[lag:], e[:-lag]) / n)
                nw_se = np.sqrt(max(nw_var, 1e-20) / n)
                nw_t_stat = float(mu_hat / nw_se)
                nw_t_pval = float(1 - stats.t.cdf(nw_t_stat, df=n - 1))

        # Sharpe：全期含空置日（與 buy-and-hold 可比）
        prev_equity = portfolio_daily_idx['Equity'].shift(1).fillna(c_period)
        daily_returns = portfolio_daily_idx['Daily_Delta'] / prev_equity
        sharpe_full = float(
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() != 0 else 0
        )
        # Sharpe：僅持倉日（反映純交易品質）
        active_returns = daily_returns[daily_returns != 0]
        sharpe_active = float(
            np.sqrt(252) * active_returns.mean() / active_returns.std()
            if len(active_returns) > 1 and active_returns.std() != 0 else 0
        )

        # MDD（比例型，負值）
        roll_max = portfolio_daily['Equity'].cummax()
        mdd_pct = float((portfolio_daily['Equity'] - roll_max).divide(roll_max).min())

    rcc = final_pnl / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 0

    n_traded = n_entries = n_normal_exits = 0
    n_stop_loss = -1
    n_forced_close = 0
    gross_profit = gross_loss = 0.0
    n_wins = n_total_trades = 0
    win_rate = 0.0

    if 'Position' in df.columns and 'Ticker_A' in df.columns:
        n_traded = len(df[df['Position'] != 0].drop_duplicates(subset=['Ticker_A', 'Ticker_B']))
        df['Prev_Pos'] = df.groupby(['Ticker_A', 'Ticker_B'])['Position'].shift(1).fillna(0)

        direction_change = df['Position'] != df['Prev_Pos']
        exit_mask = direction_change & (df['Prev_Pos'] != 0)
        n_exits_total = int(exit_mask.sum())

        last_rows = df.groupby(['Ticker_A', 'Ticker_B']).tail(1)
        n_forced_close = int((last_rows['Position'] != 0).sum())
        n_entries = n_exits_total + n_forced_close

        if 'Status' in df.columns:
            n_stop_loss = int(df[exit_mask]['Status'].astype(str)
                              .str.contains('stop|sl|停損', case=False, na=False).sum())
            n_normal_exits = n_exits_total - n_stop_loss
        else:
            n_stop_loss = -1
            n_normal_exits = n_exits_total

        if 'Trade_PnL' in df.columns and df['Trade_PnL'].notna().any():
            completed_trades = df[df['Trade_PnL'] != 0]['Trade_PnL']
            gross_profit = float(completed_trades[completed_trades > 0].sum())
            gross_loss   = float(completed_trades[completed_trades < 0].sum())
            n_wins       = int((completed_trades > 0).sum())
            n_total_trades = len(completed_trades)
            win_rate     = n_wins / n_total_trades if n_total_trades > 0 else 0.0
        elif 'Daily_Delta' in df.columns:
            state_change = df['Position'] != df['Prev_Pos']
            df['_state_id'] = state_change.groupby([df['Ticker_A'], df['Ticker_B']]).cumsum()
            df['_prev_sid'] = df.groupby(['Ticker_A', 'Ticker_B'])['_state_id'].shift(1).fillna(0)
            active_mask = (df['Prev_Pos'] != 0) | (df['Daily_Delta'] != 0)
            if active_mask.any():
                trade_pnls = (df[active_mask]
                              .groupby(['Ticker_A', 'Ticker_B', '_prev_sid'])['Daily_Delta'].sum())
                gross_profit = float(trade_pnls[trade_pnls > 0].sum())
                gross_loss   = float(trade_pnls[trade_pnls < 0].sum())
                n_wins       = int((trade_pnls > 0).sum())
                n_total_trades = len(trade_pnls)
                win_rate     = n_wins / n_total_trades if n_total_trades > 0 else 0.0

    c_pair = INITIAL_CAPITAL / top_n_int if top_n_int > 0 else INITIAL_CAPITAL
    engaged_capital = n_traded * c_pair
    rec = final_pnl / engaged_capital if engaged_capital > 0 else 0

    # Calmar Ratio：年化報酬 / 最大回撤絕對值（MDD 為負數）
    calmar = ann_ret / abs(mdd_pct) if mdd_pct < 0 else 0.0
    # Profit Factor：總獲利 / 總虧損絕對值
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else 0.0

    return {
        'DATASET': dataset, 'RE-ENTRY': reentry, 'VOL ADJ': voladj,
        'METHOD': method, 'TOP N': top_n, 'STOP LOSS %': sl_pct, 'Z-WINDOW': zwin,
        'PORT SL %': psl_pct, 'MAX SEC %': msr_pct, 'DYN Z': dsz_val,
        'Final_Equity': final_equity, 'RCC_Raw': rcc, 'REC_Raw': rec,
        'Cum_Ret_Raw': cum_ret, 'Ann_Ret_Raw': ann_ret,
        'Sharpe_Raw': sharpe_full, 'Sharpe_Active_Raw': sharpe_active,
        'MDD_Raw': mdd_pct, 'Calmar_Raw': calmar, 'PF_Raw': profit_factor,
        'Entries': n_entries, 'Exits': n_normal_exits,
        'Stop_Losses': n_stop_loss, 'Forced_Closes': n_forced_close,
        'Gross_Profit': gross_profit, 'Gross_Loss': gross_loss,
        'Win_Rate_Raw': win_rate, 'Total_Trades': n_total_trades,
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
        progress_bar.progress(float((i + 1) / total))
        short_name = s.split('/')[-1] if '/' in s else s
        status_text.markdown(f"**Compiling Metrics** ({i+1}/{total}): `{short_name}`")
        res = calculate_metrics_raw(s)
        if res:
            records.append(res)
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def compute_all_ttests(paths_tuple: tuple) -> pd.DataFrame:
    db_path = os.path.join(RESULTS_DIR, "result.db")
    use_db = os.path.exists(db_path)
    pb = st.progress(0.0)
    txt = st.empty()
    try:
        txt.markdown("**首次計算 T 檢定統計量**（僅需執行一次）")
        records = []
        total = len(paths_tuple)
        conn = None
        sql = """
            SELECT Date, SUM(Daily_Delta) AS Daily_Delta
            FROM trade_logs WHERE strategy_id = ?
            GROUP BY Date ORDER BY Date
        """
        if use_db:
            import sqlite3
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60.0)
        try:
            for i, path in enumerate(paths_tuple):
                pb.progress((i + 1) / total)
                t_stat = t_pval = nw_t_stat = nw_t_pval = np.nan
                try:
                    port_daily = None
                    if use_db and conn:
                        port_daily = pd.read_sql_query(sql, conn, params=(path,))
                        if port_daily.empty:
                            port_daily = None
                        else:
                            port_daily['Date'] = pd.to_datetime(port_daily['Date'])
                    if port_daily is None:
                        fp = os.path.join(RESULTS_DIR, path)
                        if os.path.exists(fp):
                            df_csv = pd.read_csv(fp, usecols=['Date', 'Daily_Delta'],
                                                 dtype={'Daily_Delta': 'float32'})
                            df_csv['Date'] = pd.to_datetime(df_csv['Date'], errors='coerce')
                            port_daily = df_csv.groupby('Date')['Daily_Delta'].sum().reset_index()
                    if port_daily is not None and not port_daily.empty:
                        port_daily = port_daily.sort_values('Date').set_index('Date')
                        port_daily['Equity'] = INITIAL_CAPITAL + port_daily['Daily_Delta'].cumsum()
                        monthly_equity = port_daily['Equity'].resample('ME').last().dropna()
                        if len(monthly_equity) >= 3:
                            mr = monthly_equity.pct_change().fillna(0).values
                            n = len(mr)
                            if mr.std(ddof=1) > 0:
                                res = stats.ttest_1samp(mr, popmean=0, alternative='greater')
                                t_stat = float(res.statistic)
                                t_pval = float(res.pvalue)
                                mu_hat = mr.mean()
                                e = mr - mu_hat
                                lags_nw = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
                                nw_var = float(np.dot(e, e) / n)
                                for lag in range(1, lags_nw + 1):
                                    w = 1 - lag / (lags_nw + 1)
                                    nw_var += 2 * w * float(np.dot(e[lag:], e[:-lag]) / n)
                                nw_se = np.sqrt(max(nw_var, 1e-20) / n)
                                nw_t_stat = float(mu_hat / nw_se)
                                nw_t_pval = float(1 - stats.t.cdf(nw_t_stat, df=n - 1))
                except Exception:
                    pass
                records.append({'_path': path, 'T_Stat': t_stat, 'T_Pval': t_pval,
                                'NW_T_Stat': nw_t_stat, 'NW_T_Pval': nw_t_pval})
        finally:
            if conn:
                conn.close()
    finally:
        pb.empty()
        txt.empty()
    return pd.DataFrame(records)


def make_desc(row):
    vol_part = f" · {row['VOL ADJ']}" if row.get('VOL ADJ') not in ['NoVolAdj', 'N/A', '', None] else ""
    reentry_part = f" · {row['RE-ENTRY']}" if row.get('RE-ENTRY') not in ['NoReEntry', 'Unknown', '', None] else ""
    psl_part = f" · PSL {row['PORT SL %']}" if row.get('PORT SL %') not in ['0%', '0.0%', '', None] else ""
    dsz_part = f" · DSZ {row['DYN Z']}" if row.get('DYN Z') not in ['0', '0.0', '', None] else ""
    zwin_part = f" · ZWin {row['Z-WINDOW']}" if row.get('Z-WINDOW') not in ['0', '', None] else ""
    return (f"{row['DATASET']}{reentry_part}{vol_part} · {row['METHOD']} · "
            f"{row['TOP N']} · SL {row['STOP LOSS %']} · MSR {row['MAX SEC %']}"
            f"{zwin_part}{psl_part}{dsz_part}")


@st.fragment
def render_deep_dive(target_row):
    st.markdown("---")
    st.markdown(
        f"### Strategy Deep Dive: "
        f"<span style='color:#3b82f6; font-size:1.3rem;'>{make_desc(target_row)}</span>",
        unsafe_allow_html=True
    )

    raw_target_df = load_data(target_row['_path'])
    if raw_target_df.empty or 'Period_Start' not in raw_target_df.columns:
        st.warning("Missing required columns ('Period_Start', 'Period_End') for periodic analysis.")
        return

    raw_target_df['Trading_Period'] = (raw_target_df['Period_Start'].astype(str)
                                       + " ~ " + raw_target_df['Period_End'].astype(str))

    period_stats = raw_target_df.groupby('Trading_Period').agg(
        Start_Date=('Period_Start', 'first'),
        Return=('Daily_Delta', 'sum')
    ).reset_index().sort_values('Start_Date').reset_index(drop=True)
    period_stats.rename(columns={'Trading_Period': 'Trading Period',
                                 'Return': 'Period Return ($)'}, inplace=True)
    disp_periods = period_stats[['Trading Period', 'Period Return ($)']]

    col_p1, col_p2 = st.columns([1, 1.5])

    with col_p1:
        st.markdown("##### 1. Trading Periods")
        styled_periods = disp_periods.style.format({'Period Return ($)': '${:,.2f}'}).map(
            lambda x: ('color: #4ade80; font-weight:bold;' if pd.notna(x) and x > 0
                       else ('color: #f87171; font-weight:bold;' if pd.notna(x) and x < 0 else '')),
            subset=['Period Return ($)']
        )
        period_event = st.dataframe(
            styled_periods, width='stretch', height=400, hide_index=True,
            selection_mode="single-row", on_select="rerun"
        )

    with col_p2:
        sel_p_row = period_event.selection.rows
        if not sel_p_row:
            st.info("Select a trading period on the left to view pairs.")
            return

        sel_period_str = disp_periods.iloc[sel_p_row[0]]['Trading Period']
        st.markdown(f"##### 2. Pairs in [{sel_period_str}]")

        period_df = raw_target_df[raw_target_df['Trading_Period'] == sel_period_str].copy()
        traded_mask = period_df.groupby(['Ticker_A', 'Ticker_B'])['Position'].transform(
            lambda x: (x != 0).any()
        )
        period_traded_df = period_df[traded_mask].copy()

        if period_traded_df.empty:
            st.warning("No positions were opened during this period.")
            return

        # 豐富的配對統計：Return, Trades, Win Rate, Avg Hold
        def _pair_stats(grp):
            ret = grp['Daily_Delta'].sum() if 'Daily_Delta' in grp.columns else 0.0
            if 'Trade_PnL' in grp.columns:
                trades = grp[grp['Trade_PnL'] != 0]['Trade_PnL']
                n_t = len(trades)
                wr = float((trades > 0).sum() / n_t) if n_t > 0 else 0.0
            else:
                n_t, wr = 0, 0.0
            avg_hold = float(grp[grp['Days_Held'] > 0]['Days_Held'].mean()) if 'Days_Held' in grp.columns else 0.0
            return pd.Series({'Return ($)': ret, 'Trades': n_t, 'Win Rate': wr,
                              'Avg Hold (days)': avg_hold})

        try:
            pair_stats = (period_traded_df.groupby(['Ticker_A', 'Ticker_B'])
                          .apply(_pair_stats, include_groups=False).reset_index())
        except TypeError:
            # Pandas < 2.2 不支援 include_groups 參數
            pair_stats = (period_traded_df.groupby(['Ticker_A', 'Ticker_B'])
                          .apply(_pair_stats).reset_index())
        pair_stats.rename(columns={'Ticker_A': 'Stock A', 'Ticker_B': 'Stock B'}, inplace=True)
        pair_stats = pair_stats.sort_values('Return ($)', ascending=False).reset_index(drop=True)

        styled_pairs = pair_stats.style.format({
            'Return ($)': '${:,.2f}', 'Win Rate': '{:.1%}', 'Avg Hold (days)': '{:.1f}'
        }).map(
            lambda x: ('color: #4ade80; font-weight:bold;' if pd.notna(x) and x > 0
                       else ('color: #f87171; font-weight:bold;' if pd.notna(x) and x < 0 else '')),
            subset=['Return ($)']
        )
        pair_event = st.dataframe(
            styled_pairs, width='stretch', height=400, hide_index=True,
            selection_mode="single-row", on_select="rerun"
        )
        sel_pair_row = pair_event.selection.rows

    if not sel_pair_row:
        return

    t_a = pair_stats.iloc[sel_pair_row[0]]['Stock A']
    t_b = pair_stats.iloc[sel_pair_row[0]]['Stock B']

    st.markdown("---")
    st.markdown(f"##### 3. Trade Visualizer: {t_a} vs {t_b}  (Period: {sel_period_str})")

    # 視覺化選項
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        normalize_prices = st.checkbox("Normalize prices (start = 1.0)", value=False,
                                       key=f"norm_{t_a}_{t_b}")
    with viz_col2:
        show_zscore = st.checkbox("Show Spread / Z-Score subplot", value=True,
                                  key=f"zsc_{t_a}_{t_b}")

    pair_full = period_df[(period_df['Ticker_A'] == t_a) & (period_df['Ticker_B'] == t_b)].copy()
    pair_full = pair_full.sort_values('Date').reset_index(drop=True)

    has_prices = ('Price_A' in pair_full.columns and 'Price_B' in pair_full.columns
                  and not pair_full['Price_A'].isna().all())

    n_rows = 2 if (show_zscore and has_prices) else 1
    row_heights = [0.65, 0.35] if n_rows == 2 else [1.0]
    fig_p = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.06,
        subplot_titles=(
            [f"{t_a} vs {t_b} — Prices", "Spread Z-Score"] if n_rows == 2
            else [f"{t_a} vs {t_b}"]
        )
    )

    if has_prices:
        p_a = pair_full['Price_A'].values.astype(float)
        p_b = pair_full['Price_B'].values.astype(float)
        if normalize_prices:
            first_a = p_a[~np.isnan(p_a)][0] if not np.all(np.isnan(p_a)) else 1.0
            first_b = p_b[~np.isnan(p_b)][0] if not np.all(np.isnan(p_b)) else 1.0
            y_a = p_a / first_a
            y_b = p_b / first_b
            y_a_label = f"{t_a} (norm)"
            y_b_label = f"{t_b} (norm)"
            use_secondary = False
        else:
            y_a, y_b = p_a, p_b
            y_a_label, y_b_label = f"{t_a} Price", f"{t_b} Price"
            use_secondary = True

        if use_secondary:
            # 重建帶雙 Y 軸的 subplot（row1 secondary_y，row2 單軸）
            specs_list = [[{"secondary_y": True}]]
            if n_rows == 2:
                specs_list.append([{"secondary_y": False}])
            fig_p = make_subplots(
                rows=n_rows, cols=1, shared_xaxes=True,
                specs=specs_list,
                row_heights=row_heights, vertical_spacing=0.06,
                subplot_titles=(
                    [f"{t_a} vs {t_b} — Prices", "Spread Z-Score"] if n_rows == 2
                    else [f"{t_a} vs {t_b}"]
                )
            )
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=y_a, name=y_a_label,
                                       line=dict(color='rgba(96,165,250,0.8)', width=2)),
                            row=1, col=1, secondary_y=False)
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=y_b, name=y_b_label,
                                       line=dict(color='rgba(251,211,141,0.8)', width=2)),
                            row=1, col=1, secondary_y=True)
            fig_p.update_yaxes(title_text=y_a_label, secondary_y=False, row=1)
            fig_p.update_yaxes(title_text=y_b_label, secondary_y=True, row=1)
        else:
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=y_a, name=y_a_label,
                                       line=dict(color='rgba(96,165,250,0.8)', width=2)), row=1, col=1)
            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=y_b, name=y_b_label,
                                       line=dict(color='rgba(251,211,141,0.8)', width=2)), row=1, col=1)
            fig_p.update_yaxes(title_text="Normalized Price", row=1)

        marker_y = y_a  # 進出場標記跟隨 Price_A 軸

        # Z-Score 子圖
        if show_zscore and n_rows == 2:
            hr_val = float(pair_full['Hedge_Ratio'].iloc[0]) if 'Hedge_Ratio' in pair_full.columns else 1.0
            spread = p_a - hr_val * p_b
            s_mean, s_std = spread.mean(), spread.std()
            z_vals = (spread - s_mean) / s_std if s_std > 0 else spread * 0

            fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=z_vals, name="Z-Score",
                                       line=dict(color='rgba(192,132,252,0.9)', width=1.5)),
                            row=2, col=1)
            for level, color, dash in [(2, 'rgba(248,113,113,0.6)', 'dash'),
                                        (-2, 'rgba(74,222,128,0.6)', 'dash'),
                                        (0, 'rgba(128,128,128,0.4)', 'dot')]:
                fig_p.add_hline(y=level, line=dict(color=color, dash=dash, width=1),
                                row=2, col=1)
            fig_p.update_yaxes(title_text="Z-Score", row=2)
    else:
        if 'Daily_Delta' not in pair_full.columns:
            pair_full['Daily_Delta'] = 0.0
        pair_full['Cum_PnL'] = pair_full['Daily_Delta'].cumsum()
        fig_p.add_trace(go.Scatter(x=pair_full['Date'], y=pair_full['Cum_PnL'],
                                   name="Cumulative PnL ($)",
                                   line=dict(color='rgba(74,222,128,0.8)', width=2)), row=1, col=1)
        fig_p.update_yaxes(title_text="Cumulative PnL ($)", row=1)
        marker_y = pair_full['Cum_PnL'].values

    # ── 向量化進出場訊號偵測（取代 iterrows）──
    pos_arr = pair_full['Position'].values
    prev_pos = np.concatenate([[0], pos_arr[:-1]])
    change = pos_arr != prev_pos

    # 分配 trade_id：每次狀態改變遞增
    trade_id = np.cumsum(change)
    in_trade = pos_arr != 0
    pair_full['_tid'] = trade_id
    pair_full['_in'] = in_trade

    long_x, long_y_pts = [], []
    short_x, short_y_pts = [], []
    tp_x, tp_y_pts = [], []
    sl_x, sl_y_pts = [], []

    for tid, grp in pair_full[pair_full['_in']].groupby('_tid'):
        start_dt = grp['Date'].iloc[0]
        end_dt = grp['Date'].iloc[-1]
        direction = int(grp['Position'].iloc[0])
        pnl = float(grp['Daily_Delta'].sum()) if 'Daily_Delta' in grp.columns else 0.0
        trade_width_days = (end_dt - start_dt).days if hasattr(end_dt - start_dt, 'days') else 1

        f_color = "rgba(74,222,128,0.13)" if pnl >= 0 else "rgba(248,113,113,0.13)"
        # 僅在交易時間夠長時顯示文字標注，避免密集重疊
        ann_text = ("Win" if pnl >= 0 else "Loss") if trade_width_days >= 5 else ""
        fig_p.add_vrect(x0=start_dt, x1=end_dt, fillcolor=f_color,
                        opacity=1, layer="below", line_width=0,
                        annotation_text=ann_text,
                        annotation_position="top left",
                        annotation_font_color="rgba(200,200,200,0.8)",
                        row=1, col=1)

        # grp.index 是 pair_full 的行號（已 reset_index），可直接作為 marker_y array 索引
        entry_idx_arr = int(grp.index[0])
        exit_idx_arr  = int(grp.index[-1])
        entry_y = float(marker_y[entry_idx_arr]) if len(marker_y) > entry_idx_arr else 0.0

        if direction > 0:
            long_x.append(start_dt); long_y_pts.append(entry_y)
        else:
            short_x.append(start_dt); short_y_pts.append(entry_y)

        last_status = str(grp['Status'].iloc[-1]).lower() if 'Status' in grp.columns else ''
        exit_y = float(marker_y[exit_idx_arr]) if len(marker_y) > exit_idx_arr else 0.0
        if 'stop' in last_status or 'sl' in last_status or '停損' in last_status:
            sl_x.append(end_dt); sl_y_pts.append(exit_y)
        else:
            tp_x.append(end_dt); tp_y_pts.append(exit_y)

    marker_kwargs = dict(row=1, col=1)
    if long_x:
        fig_p.add_trace(go.Scatter(x=long_x, y=long_y_pts, mode='markers', name='Buy Long',
                                   marker=dict(symbol='triangle-up', size=14, color='#4ade80',
                                               line=dict(width=1, color='rgba(0,0,0,0)'))),
                        **marker_kwargs)
    if short_x:
        fig_p.add_trace(go.Scatter(x=short_x, y=short_y_pts, mode='markers', name='Sell Short',
                                   marker=dict(symbol='triangle-down', size=14, color='#f87171',
                                               line=dict(width=1, color='rgba(0,0,0,0)'))),
                        **marker_kwargs)
    if tp_x:
        fig_p.add_trace(go.Scatter(x=tp_x, y=tp_y_pts, mode='markers', name='Take Profit',
                                   marker=dict(symbol='circle', size=11, color='#60a5fa',
                                               line=dict(width=1, color='rgba(0,0,0,0)'))),
                        **marker_kwargs)
    if sl_x:
        fig_p.add_trace(go.Scatter(x=sl_x, y=sl_y_pts, mode='markers', name='Stop Loss',
                                   marker=dict(symbol='x', size=10, color='#fbd38d',
                                               line=dict(width=2, color='#fbd38d'))),
                        **marker_kwargs)

    fig_p.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500 if n_rows == 2 else 350
    )
    st.plotly_chart(fig_p, width='stretch')


@st.cache_data(show_spinner=False)
def get_sector_mapping():
    mapping = {}
    for db_file in ["data/sp500Full.db", "data/sp500_Current.db",
                    "dataset/sp500_Tiingo.db", "dataset/sp500_yF.db",
                    "data/sp500.db", "dataset/SP500_Current.db"]:
        if os.path.exists(db_file):
            try:
                import sqlite3
                conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
                df = pd.read_sql_query("SELECT Symbol, GICS_Sector FROM Constituents", conn)
                conn.close()
                for k, v in zip(df['Symbol'], df['GICS_Sector']):
                    if k and v:
                        mapping[str(k).strip().upper()] = str(v).strip()
                break
            except Exception:
                pass
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
        cursor.execute("PRAGMA table_info(strategy_summaries);")
        existing_cols = {row[1] for row in cursor.fetchall()}
        for col in ['T_Stat', 'T_Pval', 'NW_T_Stat', 'NW_T_Pval']:
            if col not in existing_cols:
                try:
                    cursor.execute(f'ALTER TABLE strategy_summaries ADD COLUMN "{col}" REAL;')
                except Exception:
                    pass
        update_sql = """
        UPDATE strategy_summaries
        SET T_Stat=?, T_Pval=?, NW_T_Stat=?, NW_T_Pval=?
        WHERE _path=?
        """
        update_data = []
        for _, row in ttest_df.iterrows():
            update_data.append((
                float(row['T_Stat']) if pd.notna(row['T_Stat']) else None,
                float(row['T_Pval']) if pd.notna(row['T_Pval']) else None,
                float(row['NW_T_Stat']) if pd.notna(row['NW_T_Stat']) else None,
                float(row['NW_T_Pval']) if pd.notna(row['NW_T_Pval']) else None,
                row['_path']
            ))
        cursor.executemany(update_sql, update_data)
        conn.commit()
        conn.close()
    except Exception:
        pass


def render_pair_consistency():
    db_path = os.path.join(RESULTS_DIR, "result.db")
    st.markdown("### Stock Pair Consistency Analysis")
    if not os.path.exists(db_path):
        st.warning("SQLite DB not found. Run backtests to generate data.")
        return

    dataset_map = {}

    def format_strategy_id(path):
        try:
            dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val = \
                extract_features_from_path(path)
            if path in dataset_map and dataset_map[path] not in ["Unknown", ""]:
                dataset = dataset_map[path]
            parts = [dataset]
            if reentry not in ['NoReEntry', 'Unknown', '']:
                parts.append(reentry)
            if voladj not in ['NoVolAdj', 'N/A', '']:
                parts.append(voladj)
            parts += [method, top_n, f"SL {sl_pct}", f"ZWin {zwin}"]
            if psl_pct not in ['0%', '0.0%', '']:
                parts.append(f"PSL {psl_pct}")
            if msr_pct not in ['0%', '0.0%', '']:
                parts.append(f"MSR {msr_pct}")
            if dsz_val not in ['0', '0.0', '']:
                parts.append(f"DSZ {dsz_val}")
            return " · ".join(parts) + f" ({path.split('/')[-1]})"
        except Exception:
            return path

    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT _path, DATASET FROM strategy_summaries;")
            for r in cursor.fetchall():
                dataset_map[r[0]] = r[1]
        except Exception:
            pass

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_pairs';")
        if not cursor.fetchone():
            st.info("strategy_pairs table not found. Run backfill script first.")
            conn.close()
            return

        cursor.execute("SELECT count(*) FROM strategy_pairs;")
        total_pairs = cursor.fetchone()[0]
        if total_pairs == 0:
            st.info("strategy_pairs table is empty.")
            conn.close()
            return

        st.markdown(f"**{total_pairs:,} unique pair records in database.**")

        st.markdown("---")
        st.markdown("#### Strategy Pair Details")
        cursor.execute("SELECT DISTINCT strategy_id FROM strategy_pairs;")
        all_strat_ids = sorted([row[0] for row in cursor.fetchall()])

        selected_strat = st.selectbox("Select Strategy", all_strat_ids,
                                      format_func=format_strategy_id,
                                      key="sb_strat_details")
        sec_map = get_sector_mapping()

        if selected_strat:
            strat_pairs_df = pd.read_sql_query("""
                SELECT Ticker_A, Ticker_B, Period_Start, Period_End, Hedge_Ratio
                FROM strategy_pairs WHERE strategy_id = ?
                ORDER BY Period_Start DESC
            """, conn, params=(selected_strat,))
            strat_pairs_df.rename(columns={
                'Ticker_A': 'Stock A', 'Ticker_B': 'Stock B',
                'Period_Start': 'Trade Start', 'Period_End': 'Trade End',
                'Hedge_Ratio': 'Hedge Ratio'
            }, inplace=True)
            strat_pairs_df.insert(2, "Sector A", strat_pairs_df["Stock A"].apply(
                lambda x: sec_map.get(str(x).upper(), "Unknown")))
            strat_pairs_df.insert(3, "Sector B", strat_pairs_df["Stock B"].apply(
                lambda x: sec_map.get(str(x).upper(), "Unknown")))
            st.markdown(f"**{len(strat_pairs_df)}** pair relationships found:")
            st.dataframe(strat_pairs_df, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### Strategy Pair Intersection")
        col_venn1, col_venn2 = st.columns(2)
        with col_venn1:
            strat_a = st.selectbox("Strategy A", all_strat_ids, index=0 if all_strat_ids else None,
                                   format_func=format_strategy_id, key="sb_strat_a")
        with col_venn2:
            strat_b = st.selectbox("Strategy B", all_strat_ids,
                                   index=(1 if len(all_strat_ids) > 1 else 0) if all_strat_ids else None,
                                   format_func=format_strategy_id, key="sb_strat_b")

        if strat_a and strat_b:
            def _canonical_db(a, b):
                return (min(str(a), str(b)), max(str(a), str(b)))

            cursor.execute("SELECT DISTINCT Ticker_A, Ticker_B FROM strategy_pairs WHERE strategy_id=?;",
                           (strat_a,))
            pairs_a = {_canonical_db(r[0], r[1]) for r in cursor.fetchall()}
            cursor.execute("SELECT DISTINCT Ticker_A, Ticker_B FROM strategy_pairs WHERE strategy_id=?;",
                           (strat_b,))
            pairs_b = {_canonical_db(r[0], r[1]) for r in cursor.fetchall()}

            intersection = pairs_a & pairs_b
            union = pairs_a | pairs_b

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Strategy A Pairs", f"{len(pairs_a)}")
            col_m2.metric("Strategy B Pairs", f"{len(pairs_b)}")
            col_m3.metric("Common Pairs", f"{len(intersection)}",
                          f"{len(intersection)/len(union):.1%} overlap" if union else "0%")
            col_m4.metric("Union Total", f"{len(union)}")

            if intersection:
                cursor.execute("SELECT Ticker_A, Ticker_B, Period_Start, Period_End "
                               "FROM strategy_pairs WHERE strategy_id=?", (strat_a,))
                rows_a = cursor.fetchall()
                cursor.execute("SELECT Ticker_A, Ticker_B, Period_Start, Period_End "
                               "FROM strategy_pairs WHERE strategy_id=?", (strat_b,))
                rows_b = cursor.fetchall()

                periods_a, periods_b = {}, {}
                for rows, periods in [(rows_a, periods_a), (rows_b, periods_b)]:
                    for row in rows:
                        key = _canonical_db(row[0], row[1])
                        if key in intersection:
                            periods.setdefault(key, []).append(f"{row[2]} ~ {row[3]}")

                st.markdown(f"##### Common pairs ({len(intersection)}):")
                records = []
                for p in sorted(intersection):
                    t_a2, t_b2 = p
                    records.append({
                        "Stock A": t_a2, "Sector A": sec_map.get(t_a2.upper(), "Unknown"),
                        "Stock B": t_b2, "Sector B": sec_map.get(t_b2.upper(), "Unknown"),
                        "Strategy A Periods": ", ".join(sorted(set(periods_a.get(p, [])))),
                        "Strategy B Periods": ", ".join(sorted(set(periods_b.get(p, [])))),
                    })
                st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
            else:
                st.info("No common pairs between the two strategies.")

        conn.close()
    except Exception as e:
        st.error(f"Error in pair consistency analysis: {e}")


def main():
    st.write("")
    tab_selection = st.radio(
        "Module",
        ["Strategy Performance", "Pair Consistency Analysis"],
        horizontal=True, label_visibility="collapsed"
    )

    if "Pair Consistency" in tab_selection:
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

            parsed = master_df['_path'].apply(extract_features_from_path)
            parsed_df = pd.DataFrame(
                parsed.tolist(),
                columns=['DATASET', 'RE-ENTRY', 'VOL ADJ', 'METHOD', 'TOP N',
                         'STOP LOSS %', 'Z-WINDOW', 'PORT SL %', 'MAX SEC %', 'DYN Z'],
                index=master_df.index
            )
            for col in parsed_df.columns:
                if col in master_df.columns:
                    need_fill = master_df[col].isna() | master_df[col].isin(['Unknown', ''])
                    master_df.loc[need_fill, col] = parsed_df.loc[need_fill, col]

    if not use_db:
        available_strategies = scan_strategies(RESULTS_DIR)

    st.markdown(
        f'<div class="sub-title">{len(available_strategies)} strategy combinations loaded.</div>',
        unsafe_allow_html=True
    )

    if not available_strategies:
        st.warning("No strategy logs found.")
        return

    t_test_needed = False
    missing_paths = []

    if use_db:
        if 'T_Stat' not in master_df.columns:
            for col in ['T_Stat', 'T_Pval', 'NW_T_Stat', 'NW_T_Pval']:
                master_df[col] = np.nan
            t_test_needed = True
            missing_paths = master_df['_path'].tolist()
        else:
            isna_mask = master_df['T_Stat'].isna()
            if isna_mask.any():
                missing_paths = master_df.loc[isna_mask, '_path'].tolist()
                if len(missing_paths) == len(master_df):
                    t_test_needed = True
                else:
                    with st.spinner(f"Incrementally computing T-tests for {len(missing_paths)} strategies..."):
                        ttest_df = compute_all_ttests(tuple(missing_paths))
                        save_ttests_to_db(ttest_df, db_path)
                        master_df.set_index('_path', inplace=True)
                        ttest_df.set_index('_path', inplace=True)
                        for col in ['T_Stat', 'T_Pval', 'NW_T_Stat', 'NW_T_Pval']:
                            master_df.update(ttest_df[[col]])
                        master_df.reset_index(inplace=True)
    else:
        with st.spinner("Compiling metrics..."):
            master_df = build_master_dataframe(available_strategies)

    if t_test_needed:
        st.sidebar.warning("T-test cache missing.")
        if st.sidebar.button("Compute & Cache T-tests", use_container_width=True):
            with st.spinner("Computing T-tests..."):
                ttest_df = compute_all_ttests(tuple(missing_paths))
                save_ttests_to_db(ttest_df, db_path)
            st.sidebar.success("Done!")
            st.rerun()

    if master_df.empty:
        st.error("Could not parse any valid strategy data.")
        return

    # ══════════════════════════════════════════════
    # FILTERS
    # ══════════════════════════════════════════════
    FILTER_DEFS = [
        ('DATASET',     'Dataset'),
        ('RE-ENTRY',    'Re-Entry'),
        ('VOL ADJ',     'Vol Adj'),
        ('TOP N',       'Top N'),
        ('STOP LOSS %', 'Stop Loss %'),
        ('Z-WINDOW',    'Z-Win'),
        ('PORT SL %',   'Port SL %'),
        ('MAX SEC %',   'Max Sec %'),
        ('DYN Z',       'Dyn Z'),
    ]
    active_filters = [(col, label) for col, label in FILTER_DEFS
                      if col in master_df.columns and master_df[col].nunique() > 1]

    st.markdown("### Filters")

    # METHOD 用 multiselect（可同時選多個策略類型）
    method_col_present = 'METHOD' in master_df.columns and master_df['METHOD'].nunique() > 1
    sel_methods = []
    if method_col_present:
        method_opts = sorted(master_df['METHOD'].dropna().unique(), key=natural_sort_key)
        sel_methods = st.multiselect("Method (multi-select)", method_opts,
                                     placeholder="All methods", key="filter_METHOD")

    # 其餘分類篩選：selectbox（每行最多 4 個）
    sel_vals = {}
    COLS_PER_ROW = 4
    for row_start in range(0, len(active_filters), COLS_PER_ROW):
        row_filters = active_filters[row_start: row_start + COLS_PER_ROW]
        cols_ui = st.columns(len(row_filters))
        for ui_col, (col, label) in zip(cols_ui, row_filters):
            opts = ["All"] + sorted(master_df[col].dropna().unique(), key=natural_sort_key)
            sel_vals[col] = ui_col.selectbox(label, opts, key=f"filter_{col}")

    # 快捷數值篩選
    st.markdown("**Quick Filters:**")
    qf1, qf2, qf3 = st.columns([1, 1, 3])
    with qf1:
        qf_profitable = st.checkbox("Profitable Only (Ann.Ret > 0)", value=False)
    with qf2:
        qf_high_sharpe = st.checkbox("Sharpe > 1.0", value=False)

    # 年份篩選（影響 Equity Curve 顯示，表格仍為全期績效）
    st.markdown("**Year Range (affects Equity Curve display):**")
    yr_c1, yr_c2, yr_c3 = st.columns([1, 1, 3])
    with yr_c1:
        yr_start = st.number_input("Start Year", min_value=2001, max_value=2025,
                                   value=2001, step=1, key="yr_start")
    with yr_c2:
        yr_end = st.number_input("End Year", min_value=2001, max_value=2025,
                                 value=2025, step=1, key="yr_end")
    if yr_start > yr_end:
        yr_start, yr_end = yr_end, yr_start
    yr_filter_active = (yr_start > 2001 or yr_end < 2025)

    # 套用所有篩選
    filtered_df = master_df.copy()
    if sel_methods:
        filtered_df = filtered_df[filtered_df['METHOD'].isin(sel_methods)]
    for col, chosen in sel_vals.items():
        if chosen != "All":
            filtered_df = filtered_df[filtered_df[col] == chosen]
    if qf_profitable and 'Ann_Ret_Raw' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Ann_Ret_Raw'] > 0]
    if qf_high_sharpe and 'Sharpe_Raw' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sharpe_Raw'] > 1.0]

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # DISPLAY CONTROLS（必須在 display_df 欄位賦值之前）
    # ══════════════════════════════════════════════
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        expand_config = st.toggle("Expand Config Columns", value=False,
                                  help="Split strategy config into individual columns.")
    with ctrl2:
        show_detailed_metrics = st.toggle("Detailed Metrics", value=False,
                                          help="Show full metric set including trade counts.")
    with ctrl3:
        sharpe_active_mode = st.toggle("Sharpe: Active Days Only", value=False,
                                       help="Off: all trading days (comparable to buy-and-hold). "
                                            "On: only days with open positions.")

    # ══════════════════════════════════════════════
    # SUMMARY METRIC CARDS（篩選後）
    # ══════════════════════════════════════════════
    empty_series = pd.Series({
        'RCC_Raw': 0, 'REC_Raw': 0, 'Cum_Ret_Raw': 0, 'Ann_Ret_Raw': 0,
        'Sharpe_Raw': 0, 'Sharpe_Active_Raw': 0, 'MDD_Raw': 0,
        'Calmar_Raw': 0, 'PF_Raw': 0,
        'Entries': 0, 'Exits': 0, 'Stop_Losses': 0, 'Forced_Closes': 0,
        'Gross_Profit': 0.0, 'Gross_Loss': 0.0, 'Win_Rate_Raw': 0.0, 'Total_Trades': 0,
        'DATASET': '-', 'RE-ENTRY': '-', 'VOL ADJ': '-', 'METHOD': '-',
        'TOP N': '-', 'STOP LOSS %': '-', 'Z-WINDOW': '-',
        'PORT SL %': '-', 'MAX SEC %': '-', 'DYN Z': '-',
        'T_Stat': np.nan, 'T_Pval': np.nan, 'NW_T_Stat': np.nan, 'NW_T_Pval': np.nan
    })

    def safe_best(col, df=None):
        d = df if df is not None else filtered_df
        if col in d.columns and d[col].notna().any():
            return d.loc[d[col].idxmax()]
        return empty_series

    if len(filtered_df) > 0:
        best_ann  = safe_best('Ann_Ret_Raw')
        best_shp  = safe_best('Sharpe_Raw')
        # MDD 是負數；idxmax 取「最接近 0」= 最小回撤 = 最佳
        best_mdd  = safe_best('MDD_Raw')
        best_cal  = safe_best('Calmar_Raw')
        best_pf   = safe_best('PF_Raw')
        best_wr   = safe_best('Win_Rate_Raw')
    else:
        best_ann = best_shp = best_mdd = best_cal = best_pf = best_wr = empty_series

    filter_note = " (filtered)" if (sel_methods or any(v != "All" for v in sel_vals.values())
                                    or qf_profitable or qf_high_sharpe) else ""

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"BEST ANN. RETURN{filter_note}",
              f"{best_ann['Ann_Ret_Raw']:.2%}", best_ann['METHOD'])
    c2.metric(f"BEST SHARPE{filter_note}",
              f"{best_shp['Sharpe_Raw']:.2f}", best_shp['METHOD'])
    c3.metric(f"LOWEST DRAWDOWN{filter_note}",
              f"{abs(best_mdd['MDD_Raw']):.2%}", best_mdd['METHOD'])
    c4.metric(f"BEST CALMAR{filter_note}",
              f"{best_cal['Calmar_Raw']:.2f}", best_cal['METHOD'])
    c5.metric(f"BEST PROFIT FACTOR{filter_note}",
              f"{best_pf['PF_Raw']:.2f}", best_pf['METHOD'])
    c6.metric(f"BEST WIN RATE{filter_note}",
              f"{best_wr['Win_Rate_Raw']:.1%}", best_wr['METHOD'])

    # ══════════════════════════════════════════════
    # PERFORMANCE TABLE
    # ══════════════════════════════════════════════
    st.markdown("### Complete Performance Table")
    st.markdown(
        "Select rows to plot equity curves (max 5). "
        "**First selected row** shows trade detail below.",
        unsafe_allow_html=True
    )

    if len(filtered_df) == 0:
        st.info("No data matches current filters.")
        return

    _sort_options = {
        "Ann. Return":   ("Ann_Ret_Raw",  False),
        "Final Equity":  ("Final_Equity", False),
        "Sharpe":        ("Sharpe_Raw",   False),
        "Calmar":        ("Calmar_Raw",   False),
        "Max Drawdown":  ("MDD_Raw",      True),   # ascending = least negative first
        "Win Rate":      ("Win_Rate",     False),
        "Profit Factor": ("Profit_Factor",False),
        "Entries":       ("Entries",      False),
    }
    _sort_key = st.selectbox(
        "Sort table by", list(_sort_options.keys()), index=0, key="perf_sort_col"
    )
    _col, _asc = _sort_options[_sort_key]

    display_df = (filtered_df.copy()
                  .sort_values(_col, ascending=_asc)
                  .reset_index(drop=True))

    # 排名欄
    display_df.insert(0, '#', range(1, len(display_df) + 1))

    # 指標欄
    display_df['FINAL EQUITY ($)']  = display_df['Final_Equity']
    display_df['CUM. RETURN (%)']   = display_df['Cum_Ret_Raw']
    display_df['ANN. RETURN (%)']   = display_df['Ann_Ret_Raw']
    display_df['RCC (%)']           = display_df['RCC_Raw']
    display_df['REC (%)']           = display_df['REC_Raw']
    sharpe_src = 'Sharpe_Active_Raw' if sharpe_active_mode else 'Sharpe_Raw'
    display_df['SHARPE']            = display_df.get(sharpe_src, display_df.get('Sharpe_Raw', 0))
    display_df['MAX DRAWDOWN (%)']  = display_df['MDD_Raw']
    display_df['CALMAR']            = display_df.get('Calmar_Raw', 0.0)
    display_df['PROFIT FACTOR']     = display_df.get('PF_Raw', 0.0)
    display_df['WIN RATE (%)']      = display_df.get('Win_Rate_Raw', 0.0)
    _tt = display_df.get('Total_Trades', pd.Series(0, index=display_df.index))
    display_df['TOTAL TRADES']      = _tt.fillna(0).astype(int)
    display_df['ENTRIES']           = display_df['Entries']
    display_df['EXITS']             = display_df['Exits']
    display_df['STOP LOSSES']       = display_df['Stop_Losses'].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) and x >= 0 else "N/A")
    display_df['FORCED CLOSES']     = display_df['Forced_Closes']
    display_df['GROSS PROFIT ($)']  = display_df['Gross_Profit']
    display_df['GROSS LOSS ($)']    = display_df['Gross_Loss']

    _nan_s = pd.Series(np.nan, index=display_df.index)
    display_df['T-STAT']    = display_df.get('T_Stat',    _nan_s)
    display_df['T-PVAL']    = display_df.get('T_Pval',    _nan_s)
    display_df['NW T-STAT'] = display_df.get('NW_T_Stat', _nan_s)
    display_df['NW T-PVAL'] = display_df.get('NW_T_Pval', _nan_s)

    display_df['STRATEGY CONFIG'] = display_df.apply(make_desc, axis=1)

    # 決定欄位集合
    if expand_config:
        config_cols = (['#'] +
                       [col for col, _ in FILTER_DEFS
                        if col in master_df.columns and master_df[col].nunique() > 1])
    else:
        config_cols = ['#', 'STRATEGY CONFIG']

    if show_detailed_metrics:
        metrics_cols = [
            'FINAL EQUITY ($)', 'CUM. RETURN (%)', 'ANN. RETURN (%)',
            'RCC (%)', 'REC (%)', 'SHARPE', 'MAX DRAWDOWN (%)',
            'CALMAR', 'PROFIT FACTOR', 'WIN RATE (%)', 'TOTAL TRADES',
            'T-STAT', 'T-PVAL', 'NW T-STAT', 'NW T-PVAL',
            'ENTRIES', 'EXITS', 'STOP LOSSES', 'FORCED CLOSES',
            'GROSS PROFIT ($)', 'GROSS LOSS ($)'
        ]
    else:
        metrics_cols = [
            'FINAL EQUITY ($)', 'ANN. RETURN (%)', 'SHARPE', 'MAX DRAWDOWN (%)',
            'CALMAR', 'PROFIT FACTOR', 'WIN RATE (%)', 'TOTAL TRADES',
            'T-STAT', 'T-PVAL', 'NW T-STAT', 'NW T-PVAL'
        ]

    cols = config_cols + metrics_cols

    # ── 格式化 ──
    def fmt_pval(v):
        try:
            return 'N/A' if np.isnan(float(v)) else f'{float(v):.4f}'
        except Exception:
            return 'N/A'

    def fmt_tstat(v):
        try:
            return 'N/A' if np.isnan(float(v)) else f'{float(v):.3f}'
        except Exception:
            return 'N/A'

    all_formats = {
        'FINAL EQUITY ($)': '${:,.2f}',
        'CUM. RETURN (%)':  '{:.2%}',
        'ANN. RETURN (%)':  '{:.2%}',
        'RCC (%)':          '{:.2%}',
        'REC (%)':          '{:.2%}',
        'SHARPE':           '{:.2f}',
        'MAX DRAWDOWN (%)': '{:.2%}',
        'CALMAR':           '{:.2f}',
        'PROFIT FACTOR':    '{:.2f}',
        'WIN RATE (%)':     '{:.1%}',
        'TOTAL TRADES':     '{:,.0f}',
        'T-STAT':    fmt_tstat,
        'T-PVAL':    fmt_pval,
        'NW T-STAT': fmt_tstat,
        'NW T-PVAL': fmt_pval,
        'ENTRIES':          '{:,.0f}',
        'EXITS':            '{:,.0f}',
        'FORCED CLOSES':    '{:,.0f}',
        'GROSS PROFIT ($)': '${:,.2f}',
        'GROSS LOSS ($)':   '${:,.2f}',
    }
    active_formats = {k: v for k, v in all_formats.items() if k in cols}

    df_styled = display_df[cols].style.format(active_formats)

    def color_pos_neg(val):
        try:
            v = float(val)
            if v > 0: return 'color: #4ade80; font-weight: bold;'
            if v < 0: return 'color: #f87171; font-weight: bold;'
        except Exception:
            pass
        return ''

    def color_equity(val):
        try:
            v = float(val)
            # 以 0 為基準（Final_Equity 已是 PnL + portfolio_capital，正值代表獲利）
            if v > 0: return 'color: #4ade80; font-weight: bold;'
            if v < 0: return 'color: #f87171; font-weight: bold;'
        except Exception:
            pass
        return ''

    def color_winrate(val):
        try:
            v = float(val)
            if v >= 0.5: return 'color: #4ade80; font-weight: bold;'
            if v > 0:    return 'color: #f87171; font-weight: bold;'
        except Exception:
            pass
        return ''

    def color_pval(v):
        try:
            fv = float(v)
            if fv < 0.05: return 'color: #4ade80; font-weight: bold;'
            if fv < 0.10: return 'color: #fbd38d;'
        except Exception:
            pass
        return ''

    if 'FINAL EQUITY ($)' in cols:
        df_styled = df_styled.map(color_equity, subset=['FINAL EQUITY ($)'])
    for c in ['CUM. RETURN (%)', 'ANN. RETURN (%)', 'RCC (%)', 'REC (%)',
              'SHARPE', 'MAX DRAWDOWN (%)', 'CALMAR', 'PROFIT FACTOR',
              'GROSS PROFIT ($)', 'GROSS LOSS ($)', 'T-STAT', 'NW T-STAT']:
        if c in cols:
            df_styled = df_styled.map(color_pos_neg, subset=[c])
    if 'WIN RATE (%)' in cols:
        df_styled = df_styled.map(color_winrate, subset=['WIN RATE (%)'])
    for pc in ['T-PVAL', 'NW T-PVAL']:
        if pc in cols:
            df_styled = df_styled.map(color_pval, subset=[pc])

    column_config = {
        '#':                st.column_config.NumberColumn("#", width="small"),
        'STRATEGY CONFIG':  st.column_config.TextColumn("Strategy Config", width="large"),
        'DATASET':          st.column_config.TextColumn("Dataset", width="small"),
        'RE-ENTRY':         st.column_config.TextColumn("Re-Entry", width="small"),
        'VOL ADJ':          st.column_config.TextColumn("Vol Adj", width="small"),
        'METHOD':           st.column_config.TextColumn("Method", width="medium"),
        'TOP N':            st.column_config.TextColumn("Top N", width="small"),
        'STOP LOSS %':      st.column_config.TextColumn("Stop Loss", width="small"),
        'Z-WINDOW':         st.column_config.TextColumn("Z-Win", width="small"),
        'PORT SL %':        st.column_config.TextColumn("Port SL", width="small"),
        'MAX SEC %':        st.column_config.TextColumn("Max Sec", width="small"),
        'DYN Z':            st.column_config.TextColumn("Dyn Z", width="small"),
        'FINAL EQUITY ($)': st.column_config.TextColumn("Final Equity", width="small"),
        'CUM. RETURN (%)':  st.column_config.TextColumn("Cum. Ret", width="small"),
        'ANN. RETURN (%)':  st.column_config.TextColumn("Ann. Ret", width="small"),
        'RCC (%)':          st.column_config.TextColumn("RCC", width="small"),
        'REC (%)':          st.column_config.TextColumn("REC", width="small"),
        'SHARPE':           st.column_config.TextColumn("Sharpe", width="small"),
        'MAX DRAWDOWN (%)': st.column_config.TextColumn("Max DD", width="small"),
        'CALMAR':           st.column_config.TextColumn("Calmar", width="small"),
        'PROFIT FACTOR':    st.column_config.TextColumn("Profit Factor", width="small"),
        'WIN RATE (%)':     st.column_config.TextColumn("Win Rate", width="small"),
        'TOTAL TRADES':     st.column_config.TextColumn("Trades", width="small"),
        'T-STAT':           st.column_config.TextColumn("T-Stat", width="small"),
        'T-PVAL':           st.column_config.TextColumn("p-val", width="small"),
        'NW T-STAT':        st.column_config.TextColumn("NW T-Stat", width="small"),
        'NW T-PVAL':        st.column_config.TextColumn("NW p-val", width="small"),
        'ENTRIES':          st.column_config.TextColumn("Entries", width="small"),
        'EXITS':            st.column_config.TextColumn("Exits", width="small"),
        'STOP LOSSES':      st.column_config.TextColumn("Stop Loss #", width="small"),
        'FORCED CLOSES':    st.column_config.TextColumn("Forced Close", width="small"),
        'GROSS PROFIT ($)': st.column_config.TextColumn("Gross Profit", width="small"),
        'GROSS LOSS ($)':   st.column_config.TextColumn("Gross Loss", width="small"),
    }
    active_config = {k: v for k, v in column_config.items() if k in cols}

    event = st.dataframe(
        df_styled, width='stretch', hide_index=True, height=350,
        on_select="rerun", selection_mode="multi-row",
        column_config=active_config
    )

    selected_rows = event.selection.rows

    # ══════════════════════════════════════════════
    # EQUITY CURVE + DRAWDOWN CHART
    # ══════════════════════════════════════════════
    if len(selected_rows) > 0:
        eq_title = f"### Equity Curves"
        if yr_filter_active:
            eq_title += f" &nbsp;<span style='font-size:0.85em;color:#94a3b8'>({yr_start}–{yr_end}, re-based to ${INITIAL_CAPITAL:,.0f})</span>"
        st.markdown(eq_title, unsafe_allow_html=True)

        eq_ctrl1, eq_ctrl2 = st.columns(2)
        with eq_ctrl1:
            equity_pct_mode = st.toggle("Show as % Return (not $)", value=False,
                                        key="eq_pct_mode")
        with eq_ctrl2:
            show_drawdown = st.toggle("Show Drawdown subplot", value=True,
                                      key="eq_show_dd")

        plot_rows = selected_rows[:5]
        colors = ['#4ade80', '#60a5fa', '#fbd38d', '#f87171', '#c084fc']

        n_rows_eq = 2 if show_drawdown else 1
        row_h = [0.65, 0.35] if show_drawdown else [1.0]
        fig_eq = make_subplots(
            rows=n_rows_eq, cols=1, shared_xaxes=True,
            row_heights=row_h, vertical_spacing=0.05,
            subplot_titles=(["Equity / Return", "Drawdown (%)"] if show_drawdown
                            else ["Equity / Return"])
        )

        y_label = "Return (%)" if equity_pct_mode else "Account Equity ($)"

        for i, row_idx in enumerate(plot_rows):
            path = display_df.iloc[row_idx]['_path']
            label = display_df.iloc[row_idx]['METHOD']
            rank  = display_df.iloc[row_idx]['#']
            desc  = f"#{rank} {label}"
            raw_df = load_data(path)
            if raw_df.empty or 'Daily_Delta' not in raw_df.columns:
                continue

            port = raw_df.groupby('Date')['Daily_Delta'].sum().reset_index()
            port = port.sort_values('Date').reset_index(drop=True)
            port['Equity'] = INITIAL_CAPITAL + port['Daily_Delta'].cumsum()

            # 年份範圍篩選：重設基準至 INITIAL_CAPITAL，方便同期比較
            if yr_filter_active:
                port = port[
                    (port['Date'].dt.year >= yr_start) &
                    (port['Date'].dt.year <= yr_end)
                ].copy().reset_index(drop=True)
                if not port.empty:
                    port['Equity'] = INITIAL_CAPITAL + (
                        port['Daily_Delta'].cumsum()
                        - port['Daily_Delta'].cumsum().iloc[0]
                        + port['Equity'].iloc[0]
                        - INITIAL_CAPITAL
                    )

            if port.empty:
                continue

            if equity_pct_mode:
                y_vals = (port['Equity'] - INITIAL_CAPITAL) / INITIAL_CAPITAL
            else:
                y_vals = port['Equity']

            fig_eq.add_trace(go.Scatter(
                x=port['Date'], y=y_vals, mode='lines', name=desc,
                line=dict(width=2, color=colors[i % len(colors)])
            ), row=1, col=1)

            if show_drawdown:
                roll_max = port['Equity'].cummax()
                dd = (port['Equity'] - roll_max) / roll_max
                fig_eq.add_trace(go.Scatter(
                    x=port['Date'], y=dd, mode='lines', name=f"{desc} DD",
                    line=dict(width=1.5, color=colors[i % len(colors)], dash='dot'),
                    showlegend=False
                ), row=2, col=1)

        # 基準線
        if equity_pct_mode:
            fig_eq.add_hline(y=0, line=dict(color='rgba(128,128,128,0.5)', dash='dash', width=1),
                             row=1, col=1)
        else:
            fig_eq.add_hline(y=INITIAL_CAPITAL,
                             line=dict(color='rgba(128,128,128,0.5)', dash='dash', width=1),
                             annotation_text=f"Initial Capital ${INITIAL_CAPITAL:,.0f}",
                             annotation_position="bottom right",
                             row=1, col=1)
        if show_drawdown:
            fig_eq.add_hline(y=0, line=dict(color='rgba(128,128,128,0.3)', width=1),
                             row=2, col=1)

        fig_eq.update_yaxes(title_text=y_label, row=1)
        if show_drawdown:
            fig_eq.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2)
        fig_eq.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=40, b=0),
            height=480 if show_drawdown else 320
        )
        st.plotly_chart(fig_eq, width='stretch')

        target_row = display_df.iloc[selected_rows[0]]
        render_deep_dive(target_row)


if __name__ == "__main__":
    main()
