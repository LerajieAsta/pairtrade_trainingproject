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
INITIAL_CAPITAL = 10000.0


def get_db_connection(db_path="results/result.db"):
    """
    建立並取得 SQLite 資料庫連線，並配置效能優化參數。

    參數:
        db_path (str): SQLite 資料庫檔案路徑，預設為 "results/result.db"。

    回傳:
        sqlite3.Connection: 資料庫連線物件。
    """
    # 確保資料庫存放目錄存在，若不存在則自動創建
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # 建立 SQLite 連線，並設定逾時時間為 60 秒以防鎖定衝突
    conn = sqlite3.connect(db_path, timeout=60.0)

    # 啟用 WAL (Write-Ahead Logging) 模式，大幅提升併發讀寫效能
    conn.execute("PRAGMA journal_mode=WAL;")
    # 將 synchronous 設為 OFF，最大化批次寫入與索引更新速度
    conn.execute("PRAGMA synchronous=OFF;")
    # 設定快取大小為 2GB，減少 I/O 次數
    conn.execute("PRAGMA cache_size=-2000000;")
    # 設定臨時資料存放在記憶體中
    conn.execute("PRAGMA temp_store=MEMORY;")
    
    return conn


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
        "RE-ENTRY" TEXT,
        "VOL ADJ" TEXT,
        "METHOD" TEXT,
        "TOP N" TEXT,
        "STOP LOSS %" TEXT,
        "Z-WINDOW" TEXT,
        "PORT SL %" TEXT,
        "MAX SEC %" TEXT,
        "DYN Z" TEXT,
        "Final_Equity" REAL,
        "RCC_Raw" REAL,
        "REC_Raw" REAL,
        "Cum_Ret_Raw" REAL,
        "Ann_Ret_Raw" REAL,
        "Sharpe_Raw" REAL,
        "MDD_Raw" REAL,
        "Entries" INTEGER,
        "Exits" INTEGER,
        "Stop_Losses" INTEGER,
        "Forced_Closes" INTEGER,
        "Gross_Profit" REAL,
        "Gross_Loss" REAL
    );
    """)

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


def extract_features_from_path(strategy_path):
    """
    從回測結果的檔案路徑/檔名中，解析出該策略運作的關鍵參數特徵。

    參數:
        strategy_path (str): 策略回測記錄檔的相對或絕對路徑。

    回傳:
        tuple: 包含 dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val 等解析結果。
    """
    path_lower = strategy_path.lower()
    
    # 判斷資料集範疇 (Full 全期 / Current 當前)
    dataset = "Full" if "full" in path_lower else "Current" if "current" in path_lower else "Unknown"
    
    # 判斷是否允許再次進場 (ReEntry / NoReEntry)
    reentry = "NoReEntry" if "noreentry" in path_lower else "ReEntry" if "reentry" in path_lower else "Unknown"

    # 判斷是否使用波動率調整 (VolAdj / NoVolAdj)
    if "novoladj" in path_lower:
        voladj = "NoVolAdj"
    elif "voladj" in path_lower:
        voladj = "VolAdj"
    else:
        voladj = "N/A"

    # 判斷配對形成演算法分類
    method = "Unknown"
    if "ssd_basic" in path_lower:
        method = "SSD (Basic)"
    elif "ssd" in path_lower:
        method = "SSD"
    elif "eg" in path_lower:
        method = "EG"
    elif "hdbscan" in path_lower:
        if "multifactor" in path_lower:
            method = "HDBSCAN (MF)"
        else:
            is_ae = "_ae_" in path_lower or "hdbscan_ae" in path_lower
            is_pca = "_pca_" in path_lower or "hdbscan_pca" in path_lower
            if is_ae:
                method = "HDBSCAN (AE PCA)" if is_pca else "HDBSCAN (AE UMAP)"
            else:
                method = "HDBSCAN (PCA)" if is_pca else "HDBSCAN (UMAP)"

    # 使用正則表達式提取數值型參數 (Top N, Stop Loss, Z-score Window, Portfolio Stop Loss 等)
    top_n = "Top 20"
    match_n = re.search(r'top(\d+)', path_lower)
    if match_n: 
        top_n = f"Top {match_n.group(1)}"

    sl_pct = "0%"
    match_sl = re.search(r'(?<!p)sl(\d+)', path_lower)
    if match_sl: 
        sl_pct = f"{match_sl.group(1)}%"

    zwin = "0"
    match_zwin = re.search(r'zwin(\d+)', path_lower)
    if match_zwin: 
        zwin = match_zwin.group(1)

    psl_pct = "0%"
    match_psl = re.search(r'psl(\d+)', path_lower)
    if match_psl: 
        psl_pct = f"{match_psl.group(1)}%"

    msr_pct = "0%"
    match_msr = re.search(r'msr(\d+)', path_lower)
    if match_msr: 
        msr_pct = f"{match_msr.group(1)}%"

    dsz_val = "0"
    match_dsz = re.search(r'dsz(\d+)', path_lower)
    if match_dsz: 
        dsz_val = match_dsz.group(1)

    return dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val


def calculate_metrics(df, strategy_path):
    """
    分析每日交易明細 DataFrame，計算全方位的量化回測指標。

    參數:
        df (pd.DataFrame): 包含每日交易明細的原始 DataFrame。
        strategy_path (str): 該回測檔案的路徑，用於解析參數。

    回傳:
        dict: 整理好的關鍵績效指標與策略特徵，若 DataFrame 為空則回傳 None。
    """
    if df.empty:
        return None

    # 解析路徑特徵
    dataset, reentry, voladj, method, top_n, sl_pct, zwin, psl_pct, msr_pct, dsz_val = extract_features_from_path(strategy_path)
    top_n_int = int(top_n.replace('Top ', '')) if 'Top' in top_n else 20
    c_period = INITIAL_CAPITAL

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
        # 以月底最後一天價格做 resample，計算月度報酬率
        monthly_equity = portfolio_daily_idx['Equity'].resample('ME').last().dropna()

        if len(monthly_equity) > 0:
            monthly_returns = monthly_equity.pct_change().fillna(0)
            cum_ret = np.prod(1 + monthly_returns) - 1
            n_months = len(monthly_returns)
            # 依據總月數幾何年化
            ann_ret = ((1 + cum_ret) ** (12 / n_months)) - 1 if n_months > 0 else 0
        else:
            cum_ret = ann_ret = 0

        # 計算夏普值 (Sharpe Ratio, 以 252 交易日年化)
        daily_returns = portfolio_daily_idx['Daily_Delta'] / INITIAL_CAPITAL
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        # 計算最大回撤比例 (Maximum Drawdown)
        roll_max = portfolio_daily['Cumulative_PnL'].cummax()
        drawdown = portfolio_daily['Cumulative_PnL'] - roll_max
        mdd_pct = drawdown.min() / INITIAL_CAPITAL
    else:
        cum_ret = ann_ret = sharpe = mdd_pct = 0

    # 計算原始資金報酬率 (Return on Capital Constraint, RCC)
    rcc = final_pnl / c_period if c_period > 0 else 0

    # 計算交易統計 (建倉次數、出場次數、停損次數、強制平倉次數以及毛利/毛損)
    if 'Position' in df.columns and 'Ticker_A' in df.columns:
        # 計算實際有交易的配對數量
        n_traded = len(df[df['Position'] != 0].drop_duplicates(subset=['Ticker_A', 'Ticker_B']))
        df['Prev_Pos'] = df.groupby(['Ticker_A', 'Ticker_B'])['Position'].shift(1).fillna(0)

        # 藉由部位轉變識別出場與建倉次數
        direction_change = df['Position'] != df['Prev_Pos']
        exit_mask = direction_change & (df['Prev_Pos'] != 0)
        n_exits_total = exit_mask.sum()

        # 期末仍持有部位的配對，記為強制平倉
        last_rows = df.groupby(['Ticker_A', 'Ticker_B']).tail(1)
        n_forced_close = (last_rows['Position'] != 0).sum()
        n_entries = n_exits_total + n_forced_close

        # 從交易狀態中統計停損次數
        if 'Status' in df.columns:
            n_stop_loss = df[exit_mask]['Status'].astype(str).str.contains('stop|sl|停損', case=False, na=False).sum()
            n_normal_exits = n_exits_total - n_stop_loss
        else:
            n_stop_loss = -1
            n_normal_exits = n_exits_total

        # 計算毛利 (Gross Profit) 與毛損 (Gross Loss)
        if 'Daily_Delta' in df.columns:
            state_change = df['Position'] != df['Prev_Pos']
            # 為每一筆交易狀態變化標註獨立的 State_ID
            df['State_ID'] = state_change.groupby([df['Ticker_A'], df['Ticker_B']]).cumsum()
            df['Prev_State_ID'] = df.groupby(['Ticker_A', 'Ticker_B'])['State_ID'].shift(1).fillna(0)
            active_mask = (df['Prev_Pos'] != 0) | (df['Daily_Delta'] != 0)

            if active_mask.any():
                # 統計每筆獨立持倉交易區間的累積 Delta
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

    # 計算實際參與資金報酬率 (Return on Engaged Capital, REC)
    c_pair = c_period / top_n_int if top_n_int > 0 else c_period
    engaged_capital = n_traded * c_pair
    rec = final_pnl / engaged_capital if engaged_capital > 0 else 0

    return {
        'DATASET': dataset, 'RE-ENTRY': reentry, 'VOL ADJ': voladj,
        'METHOD': method, 'TOP N': top_n, 'STOP LOSS %': sl_pct, 'Z-WINDOW': zwin,
        'PORT SL %': psl_pct, 'MAX SEC %': msr_pct, 'DYN Z': dsz_val,
        'Final_Equity': float(final_equity),
        'RCC_Raw': float(rcc), 'REC_Raw': float(rec),
        'Cum_Ret_Raw': float(cum_ret), 'Ann_Ret_Raw': float(ann_ret),
        'Sharpe_Raw': float(sharpe), 'MDD_Raw': float(mdd_pct),
        'Entries': int(n_entries), 'Exits': int(n_normal_exits),
        'Stop_Losses': int(n_stop_loss), 'Forced_Closes': int(n_forced_close),
        'Gross_Profit': float(gross_profit), 'Gross_Loss': float(gross_loss),
        '_path': strategy_path
    }


def import_csv_to_db(csv_filepath, db_path="results/result.db", overwrite=True):
    """
    讀取單一策略回測交易日誌 CSV 檔案，進行指標計算，並寫入/更新至 SQLite 資料庫對應的資料表中。

    參數:
        csv_filepath (str/Path): CSV 交易紀錄檔案路徑。
        db_path (str): SQLite 資料庫路徑。
        overwrite (bool): 若為 True，則會刪除該 strategy_id 在資料庫中的歷史舊資料後重新寫入。

    回傳:
        bool: 匯入成功回傳 True，否則回傳 False。
    """
    csv_path = Path(csv_filepath)
    if not csv_path.exists():
        return False

    # 嘗試將實體絕對路徑轉換成相對於 results 目錄的相對路徑，作為資料庫中的唯一識別鍵 `_path`
    results_dir = Path("results").resolve()
    try:
        rel_path = csv_path.resolve().relative_to(results_dir).as_posix()
    except ValueError:
        rel_path = csv_path.name

    rel_path = rel_path.replace("\\", "/")

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"  ❌ 讀取 CSV 失敗 {csv_filepath}: {e}")
        return False

    if df.empty:
        return False

    # 計算策略的效能指標
    metrics = calculate_metrics(df.copy(), rel_path)
    if not metrics:
        return False

    # 初始化資料表結構
    init_db(db_path)

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 若啟用了覆寫模式，則清理此 strategy_id 的所有歷史紀錄，避免重複寫入
    if overwrite:
        cursor.execute("DELETE FROM strategy_summaries WHERE _path = ?;", (rel_path,))
        cursor.execute("DELETE FROM trade_logs WHERE strategy_id = ?;", (rel_path,))
        cursor.execute("DELETE FROM strategy_pairs WHERE strategy_id = ?;", (rel_path,))

    # 1. 寫入績效統計摘要 (strategy_summaries)
    cursor.execute("""
    INSERT INTO strategy_summaries (
        "_path", "DATASET", "RE-ENTRY", "VOL ADJ", "METHOD", "TOP N", "STOP LOSS %",
        "Z-WINDOW", "PORT SL %", "MAX SEC %", "DYN Z", "Final_Equity", "RCC_Raw",
        "REC_Raw", "Cum_Ret_Raw", "Ann_Ret_Raw", "Sharpe_Raw", "MDD_Raw",
        "Entries", "Exits", "Stop_Losses", "Forced_Closes", "Gross_Profit", "Gross_Loss"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        metrics['_path'], metrics['DATASET'], metrics['RE-ENTRY'], metrics['VOL ADJ'],
        metrics['METHOD'], metrics['TOP N'], metrics['STOP LOSS %'], metrics['Z-WINDOW'],
        metrics['PORT SL %'], metrics['MAX SEC %'], metrics['DYN Z'],
        metrics['Final_Equity'], metrics['RCC_Raw'], metrics['REC_Raw'],
        metrics['Cum_Ret_Raw'], metrics['Ann_Ret_Raw'], metrics['Sharpe_Raw'], metrics['MDD_Raw'],
        metrics['Entries'], metrics['Exits'], metrics['Stop_Losses'], metrics['Forced_Closes'],
        metrics['Gross_Profit'], metrics['Gross_Loss']
    ))

    # 插入外鍵關聯列 `strategy_id` 至明細 DataFrame 中
    df.insert(0, 'strategy_id', rel_path)

    # 確保寫入資料庫時，數值型欄位沒有 NaN，否則會影響後續 SQL 聚合計算，統一填充為預設值
    float_cols = ['Price_A', 'Price_B', 'Hedge_Ratio', 'ZScore', 'Unrealized_PnL', 'Realized_PnL', 'Cumulative_PnL', 'Trade_PnL', 'Daily_Delta', 'Log_Mean_A', 'Log_Std_A', 'Log_Mean_B', 'Log_Std_B']
    int_cols = ['Position', 'Days_Held', 'Pair_Rank']

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # 動態擴充 trade_logs 資料表缺失欄位，並將每日交易明細寫入資料庫
    add_missing_columns_to_table(conn, "trade_logs", df)
    df.to_sql("trade_logs", conn, if_exists="append", index=False, chunksize=20000)

    # 2. 篩選出策略各交易期選定的配對組合及其形成期參數，並寫入 strategy_pairs 表中
    pair_cols = ['strategy_id', 'Ticker_A', 'Ticker_B', 'Period_Start', 'Period_End',
                 'Hedge_Ratio', 'Sector', 'Pair_Rank',
                 'Log_Mean_A', 'Log_Std_A', 'Log_Mean_B', 'Log_Std_B']
    existing_pair_cols = [c for c in pair_cols if c in df.columns]

    pairs_df = df[existing_pair_cols].drop_duplicates(
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

    # 動態擴充 strategy_pairs 資料表缺失欄位，並匯入資料庫
    add_missing_columns_to_table(conn, "strategy_pairs", pairs_df)
    pairs_df.to_sql("strategy_pairs", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    return True


def import_all_csvs_in_dir(results_dir="results", db_path="results/result.db", delete_after_import=True):
    """
    掃描目標資料夾下所有匹配的回測 CSV 交易日誌，將資料以串流方式匯入 SQLite 資料庫中，
    並依設定決定是否在匯入完成後自動刪除該 CSV 檔案。

    參數:
        results_dir (str): 回測檔案存放之根目錄。
        db_path (str): SQLite 資料庫路徑。
        delete_after_import (bool): 匯入成功後是否自動刪除 CSV 來源檔以節省磁碟空間。
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return

    # 搜尋匹配特定命名格式的回測詳細紀錄檔
    csv_files = list(results_path.glob("**/*TradeLogs_*.csv")) + list(results_path.glob("**/*detailed_trade_logs_*.csv"))
    if not csv_files:
        return

    print(f"\n📥 正在掃描並匯入 {len(csv_files)} 個 CSV 至 SQLite result.db...")
    init_db(db_path)

    success_count = 0
    for csv_file in csv_files:
        if csv_file.name.endswith(".csv"):
            success = import_csv_to_db(csv_file, db_path, overwrite=True)
            if success:
                success_count += 1
                # 若設定刪除已匯入檔案，則呼叫 unlink 進行移除
                if delete_after_import:
                    try:
                        csv_file.unlink()
                    except Exception as e:
                        print(f"  ⚠️ 無法刪除已處理的 CSV {csv_file}: {e}")

    print(f"✨ 匯入完成！成功匯入 {success_count}/{len(csv_files)} 個策略回測資料。\n")
