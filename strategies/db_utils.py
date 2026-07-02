"""
資料庫與回測指標公用工具模組
============================

本模組提供配對交易回測系統中所需的資料庫管理與指標運算功能，主要包含：
1. SQLite 資料庫連線管理與高效能參數設定 (WAL 模式)。
2. 資料表動態欄位擴充與初始化。
3. 自回測產出檔案路徑中解析策略參數。
4. 計算策略的關鍵效能指標 (Equity、CAGR、Sharpe、MDD、交易次數、毛利/毛損等)。
5. 批量匯入回測詳細 CSV 紀錄至 SQLite 資料庫。
"""

import os
import re
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# 確保輸出編碼為 UTF-8，避免 Windows 環境下輸出中文字元時產生編碼錯誤
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 定義策略初始資金常數，用於計算報酬率與部位佔比
from strategies.config import INITIAL_CAPITAL, CONCURRENT_PERIODS, RF_ANNUAL


def get_db_connection(db_path="results/result.db", max_retries=5, retry_delay=0.5):
    """
    建立並取得 SQLite 資料庫連線，並配置效能優化參數。
    具備重試機制，防止多行程並發寫入時發生 database is locked。

    參數:
        db_path (str): SQLite 資料庫檔案路徑，預設為 "results/result.db"。
        max_retries (int): 當遇上 lock 時的最大重試次數。
        retry_delay (float): 每次重試間隔的秒數。

    回傳:
        sqlite3.Connection: 資料庫連線物件。
    """
    import time
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    for attempt in range(max_retries):
        try:
            # timeout=60.0 本身就會讓 SQLite 內部自動等待解鎖
            conn = sqlite3.connect(db_path, timeout=60.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=OFF;")
            conn.execute("PRAGMA cache_size=-2000000;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            return conn
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise e
    return None


def add_missing_columns_to_table(conn, table_name, df):
    """
    動態檢查 Pandas DataFrame 中的欄位，若 SQLite 資料表中缺失，則自動進行 ALTER TABLE 新增。

    參數:
        conn (sqlite3.Connection): 資料庫連線物件。
        table_name (str): 目標資料表名稱。
        df (pd.DataFrame): 準備寫入資料庫的 DataFrame。
    """
    cursor = conn.cursor()

    # 取得資料表目前已有的欄位資訊
    cursor.execute(f"PRAGMA table_info({table_name});")
    existing_cols = {row[1] for row in cursor.fetchall()}

    # 如果資料表尚未建立（無任何欄位），則跳過動態擴充，交由後續 schema 初始化或 to_sql 處理
    if not existing_cols:
        return

    # 遍歷 DataFrame 欄位，檢查是否有尚未存在於資料庫中的新欄位
    for col in df.columns:
        if col not in existing_cols:
            # 依據 Pandas 的資料型態決定 SQLite 的資料庫欄位型態
            col_type = "REAL"
            if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.StringDtype):
                col_type = "TEXT"
            elif df[col].dtype in ['int64', 'int32', 'int16']:
                col_type = "INTEGER"

            # 使用雙引號包裹欄位名稱，避免特殊字元或保留字導致 SQL 語法解析失敗
            safe_col = f'"{col}"'
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {safe_col} {col_type};")
            except Exception as e:
                print(f"  ⚠️ ALTER TABLE 失敗 {table_name} ADD COLUMN {col}: {e}")
                
    conn.commit()


def init_db(db_path="results/result.db"):
    """
    初始化 SQLite 資料庫，建立策略統計摘要、詳細交易日誌以及策略配對標的等資料表與索引。

    參數:
        db_path (str): SQLite 資料庫檔案路徑。
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. 建立 strategy_summaries 表：存放各組策略回測的最終績效統計
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_summaries (
        "_path" TEXT PRIMARY KEY,
        "DATASET" TEXT,
        "METHOD" TEXT,
        "TRADE_METHOD" TEXT,
        "TOP N" TEXT,
        "STOP LOSS %" TEXT,
        "MAX SEC %" TEXT,
        "Final_Equity" REAL,
        "RCC_Raw" REAL,
        "REC_Raw" REAL,
        "Cum_Ret_Raw" REAL,
        "Ann_Ret_Raw" REAL,
        "Sharpe_Raw" REAL,
        "Sortino_Raw" REAL,
        "Calmar_Raw" REAL,
        "MDD_Raw" REAL,
        "Win_Rate" REAL,
        "Profit_Factor" REAL,
        "Avg_Trade_Days" REAL,
        "Entries" INTEGER,
        "Exits" INTEGER,
        "Stop_Losses" INTEGER,
        "Forced_Closes" INTEGER,
        "Gross_Profit" REAL,
        "Gross_Loss" REAL,
        "Avg_Utilization" REAL,
        "Ann_Ret_Employed" REAL,
        "Excess_Ret_RF" REAL
    );
    """)

    try:
        cursor.execute('ALTER TABLE strategy_summaries ADD COLUMN "TRADE_METHOD" TEXT;')
    except Exception:
        pass

    # 舊資料庫遷移：文獻口徑績效欄位（GGR fully-invested / rf 計息超額）
    for _col in ("Avg_Utilization", "Ann_Ret_Employed", "Excess_Ret_RF"):
        try:
            cursor.execute(f'ALTER TABLE strategy_summaries ADD COLUMN "{_col}" REAL;')
        except Exception:
            pass

    # 2. 建立 trade_logs 表：存放每日明細，包含價格、Z-Score、部位及未實現/已實現損益等
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_logs (
        "strategy_id" TEXT,
        "Date" TEXT,
        "Price_A" REAL,
        "Price_B" REAL,
        "Hedge_Ratio" REAL,
        "ZScore" REAL,
        "Position" INTEGER,
        "Unrealized_PnL" REAL,
        "Realized_PnL" REAL,
        "Cumulative_PnL" REAL,
        "Status" TEXT,
        "Trade_PnL" REAL,
        "Days_Held" INTEGER,
        "Daily_Delta" REAL,
        "Period_Start" TEXT,
        "Period_End" TEXT,
        "Sector" TEXT,
        "Pair_Rank" INTEGER,
        "Ticker_A" TEXT,
        "Ticker_B" TEXT,
        "Log_Mean_A" REAL,
        "Log_Std_A" REAL,
        "Log_Mean_B" REAL,
        "Log_Std_B" REAL
    );
    """)

    # 3. 建立 strategy_pairs 表：存放策略各期選定之配對及其形成期參數與權重
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_pairs (
        "strategy_id" TEXT,
        "Ticker_A" TEXT,
        "Ticker_B" TEXT,
        "Period_Start" TEXT,
        "Period_End" TEXT,
        "Hedge_Ratio" REAL,
        "Sector" TEXT,
        "Pair_Rank" INTEGER,
        "Log_Mean_A" REAL,
        "Log_Std_A" REAL,
        "Log_Mean_B" REAL,
        "Log_Std_B" REAL,
        PRIMARY KEY ("strategy_id", "Ticker_A", "Ticker_B", "Period_Start", "Period_End")
    );
    """)

    # 4. 建立效能索引，優化後續分析查詢速度
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_logs_strat ON trade_logs (strategy_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_logs_date ON trade_logs (Date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_logs_strat_date ON trade_logs (strategy_id, Date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_pairs_strat ON strategy_pairs (strategy_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_pairs_pair ON strategy_pairs (Ticker_A, Ticker_B);")

    conn.commit()
    conn.close()


def calculate_metrics_from_params(df, strategy_name, params, dataset_name, path_key, trade_method=""):
    """
    分析每日交易明細 DataFrame，並結合傳入的參數字典計算全方位的量化回測指標。

    參數:
        df (pd.DataFrame): 包含每日交易明細的原始 DataFrame。
        strategy_name (str): 策略名稱，例如 "SSD (Basic)"。
        params (dict): 策略參數字典。
        dataset_name (str): 資料集名稱，例如 "Current" 或 "Full"。
        path_key (str): 資料庫中的唯一識別鍵（通常是對應參數組合的偽路徑名稱）。
        trade_method (str): 交易期的策略方法名稱。

    回傳:
        dict: 整理好的關鍵績效指標與策略特徵，若 DataFrame 為空則回傳 None。
    """
    if df.empty:
        return None

    df = df.copy()
    c_period = INITIAL_CAPITAL
    top_n_int = params.get("top_n", 20)

    # 參數對應與格式化
    reentry = "ReEntry" if params.get("allow_reentry", False) else "NoReEntry"
    voladj = "VolAdj" if params.get("use_vol_adjust", False) else "NoVolAdj"
    
    sl_pct = f"{int(params.get('stop_loss_pct', 0) * 100)}%"
    psl_pct = f"{int(params.get('portfolio_stop_loss_pct', 0) * 100)}%"
    msr_pct = f"{int(params.get('max_sector_ratio', 0) * 100)}%"
    dsz_val = f"{int(params.get('dynamic_stop_z', 0))}"
    zwin = str(params.get('zscore_window', 0))
    top_n_str = f"Top {top_n_int}"

    # 計算每日投資組合層級的損益與權益曲線 (Equity Curve)
    if 'Daily_Delta' in df.columns:
        portfolio_daily = df.groupby('Date')['Daily_Delta'].sum().reset_index()
        portfolio_daily = portfolio_daily.sort_values('Date').reset_index(drop=True)
    else:
        portfolio_daily = pd.DataFrame({'Date': df['Date'].unique(), 'Daily_Delta': 0})

    portfolio_daily['Cumulative_PnL'] = portfolio_daily['Daily_Delta'].cumsum()
    portfolio_daily['Equity'] = INITIAL_CAPITAL + portfolio_daily['Cumulative_PnL']

    final_pnl = portfolio_daily['Cumulative_PnL'].iloc[-1] if not portfolio_daily.empty else 0
    final_equity = INITIAL_CAPITAL + final_pnl

    # 計算報酬率相關績效指標 (年化報酬率、夏普值、最大回撤等)
    if len(portfolio_daily) > 0:
        portfolio_daily_idx = portfolio_daily.set_index('Date')
        portfolio_daily_idx.index = pd.to_datetime(portfolio_daily_idx.index)
        monthly_equity = portfolio_daily_idx['Equity'].resample('ME').last().dropna()

        if len(monthly_equity) > 0:
            monthly_returns = monthly_equity.pct_change().fillna(0)
            cum_ret = np.prod(1 + monthly_returns) - 1
            n_months = len(monthly_returns)
            ann_ret = ((1 + cum_ret) ** (12 / n_months)) - 1 if n_months > 0 else 0
        else:
            cum_ret = ann_ret = 0

        # 以真實權益計算每日報酬率 (避免第一天 NaN，使用 shift(1) 並填充 INITIAL_CAPITAL)
        prev_equity = portfolio_daily_idx['Equity'].shift(1).fillna(INITIAL_CAPITAL)
        daily_returns = portfolio_daily_idx['Daily_Delta'] / prev_equity
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

        # Sortino Ratio
        neg_ret = daily_returns[daily_returns < 0]
        sortino = (np.sqrt(252) * daily_returns.mean() / neg_ret.std()) if (len(neg_ret) > 0 and neg_ret.std() != 0) else 0.0

        # 修正 MDD 算法：以真實動態權益計算最大回撤比例
        roll_max_equity = portfolio_daily['Equity'].cummax()
        drawdown_pct = (portfolio_daily['Equity'] - roll_max_equity) / roll_max_equity
        mdd_pct = drawdown_pct.min()

        # Calmar Ratio
        calmar = ann_ret / abs(mdd_pct) if mdd_pct != 0 else 0.0
    else:
        cum_ret = ann_ret = sharpe = mdd_pct = 0
        sortino = calmar = 0.0

    rcc = final_pnl / c_period if c_period > 0 else 0

    # ── 文獻口徑績效（GGR 2006 fully-invested measure / rf 計息超額） ──────
    # Avg_Utilization：平均資金利用率 = 日均持倉配對數 / 最大配對槽位
    # Ann_Ret_Employed：動用資本年化 = 總損益 / (日均動用資金 × 年數)
    # Excess_Ret_RF：閒置現金計 rf 利息後的超額年化
    #   = 承諾資本算術年化 + rf×(1−利用率) − rf = 算術年化 − rf×利用率
    max_pairs = top_n_int * CONCURRENT_PERIODS
    avg_utilization = ann_ret_employed = excess_ret_rf = 0.0
    if 'Position' in df.columns and max_pairs > 0 and len(portfolio_daily) > 0:
        _daily_open = (
            df[df['Position'] != 0].groupby('Date').size()
            .reindex(portfolio_daily['Date'], fill_value=0)
        )
        years = len(portfolio_daily) / 252.0
        if years > 0:
            avg_utilization = float(_daily_open.mean() / max_pairs)
            ann_ret_arith = final_pnl / c_period / years
            avg_employed = _daily_open.mean() * (c_period / max_pairs)
            if avg_employed > 0:
                ann_ret_employed = float(final_pnl / avg_employed / years)
            excess_ret_rf = float(ann_ret_arith - RF_ANNUAL * avg_utilization)

    # 計算交易統計
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
                win_rate = float((trade_pnls > 0).mean()) if len(trade_pnls) > 0 else 0.0
                profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0.0
            else:
                gross_profit = gross_loss = 0.0
                win_rate = profit_factor = 0.0
        else:
            gross_profit = gross_loss = 0.0
            win_rate = profit_factor = 0.0

        # Average trade duration (days)
        if 'Days_Held' in df.columns and 'Trade_PnL' in df.columns:
            closed_trades = df[df['Trade_PnL'] != 0]
            avg_trade_days = float(closed_trades['Days_Held'].mean()) if len(closed_trades) > 0 else 0.0
        else:
            avg_trade_days = 0.0
    else:
        n_traded = n_entries = n_normal_exits = 0
        n_stop_loss = -1
        n_forced_close = 0
        gross_profit = gross_loss = 0.0
        win_rate = profit_factor = avg_trade_days = 0.0

    c_pair = c_period / top_n_int if top_n_int > 0 else c_period
    engaged_capital = n_traded * c_pair
    rec = final_pnl / engaged_capital if engaged_capital > 0 else 0

    return {
        'DATASET': dataset_name, 'RE-ENTRY': reentry, 'VOL ADJ': voladj,
        'METHOD': strategy_name, 'TRADE_METHOD': trade_method, 'TOP N': top_n_str, 'STOP LOSS %': sl_pct, 'Z-WINDOW': zwin,
        'PORT SL %': psl_pct, 'MAX SEC %': msr_pct, 'DYN Z': dsz_val,
        'Final_Equity': float(final_equity),
        'RCC_Raw': float(rcc), 'REC_Raw': float(rec),
        'Cum_Ret_Raw': float(cum_ret), 'Ann_Ret_Raw': float(ann_ret),
        'Sharpe_Raw': float(sharpe), 'Sortino_Raw': float(sortino),
        'Calmar_Raw': float(calmar), 'MDD_Raw': float(mdd_pct),
        'Win_Rate': float(win_rate), 'Profit_Factor': float(profit_factor),
        'Avg_Trade_Days': float(avg_trade_days),
        'Entries': int(n_entries), 'Exits': int(n_normal_exits),
        'Stop_Losses': int(n_stop_loss), 'Forced_Closes': int(n_forced_close),
        'Gross_Profit': float(gross_profit), 'Gross_Loss': float(gross_loss),
        'Avg_Utilization': float(avg_utilization),
        'Ann_Ret_Employed': float(ann_ret_employed),
        'Excess_Ret_RF': float(excess_ret_rf),
        '_path': path_key
    }


def export_df_to_db(df, strategy_name, params, dataset_name, path_key, db_path="results/result.db", overwrite=True, trade_method=""):
    """
    直接將子行程算出的 DataFrame 與績效指標寫入 SQLite，不透過 CSV 檔案轉發。

    參數:
        df (pd.DataFrame): 單個策略/網格產生的完整交易明細。
        strategy_name (str): 策略的方法名稱，例如 "SSD (Basic)"，必須對應 dashboard 的 mapping。
        params (dict): 本次跑此策略網格的參數字典。
        dataset_name (str): "Current" 或 "Full"。
        path_key (str): 偽路徑，作為資料庫中該次網格回測的主鍵 (例如 "SSD_Basic_ReEntry/TradeLogs_...").
        db_path (str): SQLite 儲存路徑。
        overwrite (bool): 是否刪除舊的相同 path_key 數據。
        trade_method (str): 交易期的策略方法名稱。
        
    回傳:
        bool: 寫入是否成功。
    """
    if df.empty:
        return False

    metrics = calculate_metrics_from_params(df.copy(), strategy_name, params, dataset_name, path_key, trade_method)
    if not metrics:
        return False

    init_db(db_path)
    conn = get_db_connection(db_path)
    if not conn:
        print(f"  ❌ 無法取得資料庫連線，寫入失敗: {path_key}")
        return False

    cursor = conn.cursor()

    try:
        if overwrite:
            cursor.execute("DELETE FROM strategy_summaries WHERE _path = ?;", (path_key,))
            cursor.execute("DELETE FROM trade_logs WHERE strategy_id = ?;", (path_key,))
            cursor.execute("DELETE FROM strategy_pairs WHERE strategy_id = ?;", (path_key,))

        cursor.execute("""
        INSERT INTO strategy_summaries (
            "_path", "DATASET", "METHOD", "TRADE_METHOD", "TOP N", "STOP LOSS %",
            "MAX SEC %", "Final_Equity", "RCC_Raw",
            "REC_Raw", "Cum_Ret_Raw", "Ann_Ret_Raw", "Sharpe_Raw", "Sortino_Raw", "Calmar_Raw", "MDD_Raw",
            "Win_Rate", "Profit_Factor", "Avg_Trade_Days",
            "Entries", "Exits", "Stop_Losses", "Forced_Closes", "Gross_Profit", "Gross_Loss",
            "Avg_Utilization", "Ann_Ret_Employed", "Excess_Ret_RF"
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            metrics['_path'], metrics['DATASET'],
            metrics['METHOD'], metrics['TRADE_METHOD'], metrics['TOP N'], metrics['STOP LOSS %'],
            metrics['MAX SEC %'],
            metrics['Final_Equity'], metrics['RCC_Raw'], metrics['REC_Raw'],
            metrics['Cum_Ret_Raw'], metrics['Ann_Ret_Raw'], metrics['Sharpe_Raw'],
            metrics['Sortino_Raw'], metrics['Calmar_Raw'], metrics['MDD_Raw'],
            metrics['Win_Rate'], metrics['Profit_Factor'], metrics['Avg_Trade_Days'],
            metrics['Entries'], metrics['Exits'], metrics['Stop_Losses'], metrics['Forced_Closes'],
            metrics['Gross_Profit'], metrics['Gross_Loss'],
            metrics['Avg_Utilization'], metrics['Ann_Ret_Employed'], metrics['Excess_Ret_RF']
        ))

        df_db = df.copy()
        df_db.insert(0, 'strategy_id', path_key)

        float_cols = ['Price_A', 'Price_B', 'Hedge_Ratio', 'ZScore', 'Unrealized_PnL', 'Realized_PnL', 'Cumulative_PnL', 'Trade_PnL', 'Daily_Delta', 'Log_Mean_A', 'Log_Std_A', 'Log_Mean_B', 'Log_Std_B']
        int_cols = ['Position', 'Days_Held', 'Pair_Rank']

        for c in float_cols:
            if c in df_db.columns:
                df_db[c] = pd.to_numeric(df_db[c], errors='coerce').fillna(0.0)

        for c in int_cols:
            if c in df_db.columns:
                df_db[c] = pd.to_numeric(df_db[c], errors='coerce').fillna(0).astype(int)

        add_missing_columns_to_table(conn, "trade_logs", df_db)
        df_db.to_sql("trade_logs", conn, if_exists="append", index=False, chunksize=20000)

        pair_cols = ['strategy_id', 'Ticker_A', 'Ticker_B', 'Period_Start', 'Period_End',
                     'Hedge_Ratio', 'Sector', 'Pair_Rank',
                     'Log_Mean_A', 'Log_Std_A', 'Log_Mean_B', 'Log_Std_B']
        existing_pair_cols = [c for c in pair_cols if c in df_db.columns]

        pairs_df = df_db[existing_pair_cols].drop_duplicates(
            subset=['strategy_id', 'Ticker_A', 'Ticker_B', 'Period_Start', 'Period_End']
        ).copy()

        p_float_cols = ['Hedge_Ratio', 'Log_Mean_A', 'Log_Std_A', 'Log_Mean_B', 'Log_Std_B']
        p_int_cols = ['Pair_Rank']

        for c in p_float_cols:
            if c in pairs_df.columns:
                pairs_df[c] = pd.to_numeric(pairs_df[c], errors='coerce').fillna(0.0)
        for c in p_int_cols:
            if c in pairs_df.columns:
                pairs_df[c] = pd.to_numeric(pairs_df[c], errors='coerce').fillna(0).astype(int)

        add_missing_columns_to_table(conn, "strategy_pairs", pairs_df)
        pairs_df.to_sql("strategy_pairs", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"  ❌ 資料庫寫入失敗 {path_key}: {e}")
        try:
            conn.rollback()
            conn.close()
        except:
            pass
        return False


def init_formation_db(db_path="data/formation_pairs.db"):
    """
    初始化 Formation 期間選出的配對資料庫。
    """
    conn = get_db_connection(db_path)
    if conn is None: return None
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS formation_pairs (
        "strategy_id" TEXT,
        "Period_Start" TEXT,
        "Trade_Start" TEXT,
        "Trade_End" TEXT,
        "Ticker_A" TEXT,
        "Ticker_B" TEXT,
        "Sector_A" TEXT,
        "Sector_B" TEXT,
        "Pair_Rank" INTEGER,
        "Formation_Params" TEXT,
        PRIMARY KEY ("strategy_id", "Period_Start", "Ticker_A", "Ticker_B")
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_formation_strat ON formation_pairs (strategy_id);")
    conn.commit()
    return conn
