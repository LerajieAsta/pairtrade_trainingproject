import pandas as pd
import sqlite3
import time
import requests
import os
import signal
import sys
from datetime import datetime, timedelta

# ==========================================
# 請在此填入您在 Tiingo 註冊取得的 API Key
# ==========================================
TIINGO_API_KEY = "432374758dbd3bdcad66bf3cf990393b2fd37579"

# ==========================================
# 全域變數：用於安全中斷
# ==========================================
_conn_global = None       # 全域 connection，供 signal handler 使用
_interrupted = False      # 旗標：是否已被 Ctrl+C 中斷

def _safe_exit_handler(signum, frame):
    """
    捕捉 Ctrl+C (SIGINT)，確保資料已 commit 後再退出。
    """
    global _interrupted
    if _interrupted:
        # 第二次按 Ctrl+C 強制退出
        print("\n強制退出！")
        sys.exit(1)

    _interrupted = True
    print("\n\n⚠️  偵測到 Ctrl+C，正在安全儲存已下載資料，請稍候...")
    if _conn_global:
        try:
            _conn_global.commit()
            print("✅ 資料已成功 commit 至資料庫，可以安全退出。")
        except Exception as e:
            print(f"❌ Commit 時發生錯誤: {e}")
        finally:
            _conn_global.close()
            print("🔒 資料庫連線已關閉。")
    print("若要繼續下載，重新執行程式即可（會自動從斷點續傳）。")
    sys.exit(0)

# 註冊 signal handler
signal.signal(signal.SIGINT, _safe_exit_handler)


# ==========================================
# 網路請求工具：加入 retry 機制
# ==========================================
def robust_get(url, headers, max_retries=3, retry_delay=5):
    """
    帶有重試機制的 GET 請求。
    - DNS / 連線錯誤：最多重試 max_retries 次
    - 空回應（非 JSON）：直接回傳 None，不拋例外
    - 429 / 403：直接回傳 response，由呼叫端處理
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=15)

            # 空回應（例如 rate limit 軟性封鎖）
            if not response.text.strip():
                print(f"  ⚠️  收到空回應 (attempt {attempt}/{max_retries})，跳過。")
                return None, None  # (response, parsed_json)

            return response, None  # json 留給呼叫端自行解析

        except requests.exceptions.ConnectionError as e:
            err_str = str(e)
            if "NameResolutionError" in err_str or "getaddrinfo" in err_str:
                print(f"  ❌ DNS 解析失敗 (attempt {attempt}/{max_retries})：{url}")
                print(f"     原因：{err_str[:120]}")
            else:
                print(f"  ❌ 連線錯誤 (attempt {attempt}/{max_retries})：{err_str[:120]}")

            if attempt < max_retries:
                print(f"     {retry_delay} 秒後重試...")
                time.sleep(retry_delay)
            else:
                print("  ⛔ 已達最大重試次數，略過此請求。")
                return None, None

        except requests.exceptions.Timeout:
            print(f"  ⏱️  請求逾時 (attempt {attempt}/{max_retries})，{retry_delay} 秒後重試...")
            time.sleep(retry_delay)

        except Exception as e:
            print(f"  ❌ 未知錯誤: {e}")
            return None, None

    return None, None


# ==========================================
# 取得 S&P 500 成份股清單
# ==========================================
def get_sp500_tickers():
    print("正在從 Wikipedia 獲取 S&P 500 成份股清單...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        tables = pd.read_html(response.text)

        df_current = tables[0]
        current_tickers = df_current['Symbol'].tolist()

        df_adjustments = tables[1]
        added_tickers = []
        removed_tickers = []

        if 'Added' in df_adjustments.columns:
            col = df_adjustments['Added']
            added_tickers = (col['Ticker'].dropna().tolist()
                             if isinstance(col, pd.DataFrame)
                             else col.dropna().tolist())
        if 'Removed' in df_adjustments.columns:
            col = df_adjustments['Removed']
            removed_tickers = (col['Ticker'].dropna().tolist()
                               if isinstance(col, pd.DataFrame)
                               else col.dropna().tolist())

        all_tickers = list(set(current_tickers + added_tickers + removed_tickers))
        all_tickers = [str(t).replace('.', '-') for t in all_tickers if pd.notna(t)]

        print(f"成功！共找到 {len(all_tickers)} 支歷年成分股。")
        return all_tickers, df_current

    except Exception as e:
        print(f"獲取清單失敗: {e}")
        return [], pd.DataFrame()


# ==========================================
# 初始化資料庫
# ==========================================
def setup_database(db_name="sp500Full.db"):
    conn = sqlite3.connect(db_name, isolation_level=None)  # autocommit off，手動控制
    conn.execute("PRAGMA journal_mode=WAL;")               # WAL 模式，減少資料損毀風險
    conn.execute("BEGIN")                                  # 開啟交易
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Constituents (
            Symbol TEXT PRIMARY KEY,
            Security TEXT,
            GICS_Sector TEXT,
            GICS_Sub_Industry TEXT,
            Headquarters_Location TEXT,
            Date_Added TEXT,
            CIK TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Daily_Prices (
            Date TEXT,
            Symbol TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume INTEGER,
            PRIMARY KEY (Date, Symbol)
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_daily_symbol
        ON Daily_Prices (Symbol, Date)
    ''')

    # Skip_List：記錄確認無資料或 API 找不到的 (Symbol, 區間) 組合
    # reason: NO_DATA（API 回空）| NOT_FOUND（404）| JSON_ERROR（解析失敗）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Skip_List (
            Symbol      TEXT,
            Start_Date  TEXT,
            End_Date    TEXT,
            Reason      TEXT,
            Logged_At   TEXT,
            PRIMARY KEY (Symbol, Start_Date, End_Date)
        )
    ''')

    conn.commit()
    conn.execute("BEGIN")  # 重新開啟交易供後續使用
    return conn


# ==========================================
# 下載並儲存股價資料
# ==========================================
COMMIT_EVERY = 20   # 每下載 N 支股票 commit 一次，平衡效能與安全性

def download_and_save_data(tickers, conn, start_date="2000-01-01", end_date="2025-12-31"):
    global _interrupted

    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {TIINGO_API_KEY}'
    }

    total = len(tickers)
    for i, ticker in enumerate(tickers):

        # 若已被 Ctrl+C 中斷，跳出主迴圈
        if _interrupted:
            break

        cursor = conn.cursor()
        cursor.execute(
            "SELECT MIN(Date), MAX(Date) FROM Daily_Prices WHERE Symbol=?", (ticker,)
        )
        min_date_str, max_date_str = cursor.fetchone()

        tasks = []

        # 回溯補償 (Backfill)
        # 差距 <= 7 天內必然沒有交易日（假日/週末），跳過以節省 API 額度
        BACKFILL_MIN_DAYS = 7
        FORWARD_MIN_DAYS  = 3
        if min_date_str and min_date_str > start_date:
            gap_days = (
                datetime.strptime(min_date_str, '%Y-%m-%d')
                - datetime.strptime(start_date, '%Y-%m-%d')
            ).days
            if gap_days > BACKFILL_MIN_DAYS:
                backfill_end = (
                    datetime.strptime(min_date_str, '%Y-%m-%d') - timedelta(days=1)
                ).strftime('%Y-%m-%d')
                tasks.append((start_date, backfill_end, "回溯歷史"))
            # else: 差距太小，跳過，不浪費 API

        # 向上更新 (Forward-fill)
        if max_date_str:
            if max_date_str < end_date:
                gap_days = (
                    datetime.strptime(end_date, '%Y-%m-%d')
                    - datetime.strptime(max_date_str, '%Y-%m-%d')
                ).days
                if gap_days > FORWARD_MIN_DAYS:
                    forward_start = (
                        datetime.strptime(max_date_str, '%Y-%m-%d') + timedelta(days=1)
                    ).strftime('%Y-%m-%d')
                    if forward_start <= end_date:
                        tasks.append((forward_start, end_date, "同步新資料"))
                # else: 差距太小（週末），跳過
        else:
            tasks.append((start_date, end_date, "首次下載"))

        for f_start, f_end, task_type in tasks:
            if _interrupted:
                break
            if f_start > f_end:
                continue

            # 查 Skip_List：若此 (ticker, 區間) 已確認無資料，直接跳過
            cursor.execute(
                "SELECT Reason FROM Skip_List WHERE Symbol=? AND Start_Date=? AND End_Date=?",
                (ticker, f_start, f_end)
            )
            skip_row = cursor.fetchone()
            if skip_row:
                print(f"[{i+1}/{total}] {ticker} {task_type}: 略過（已記錄 {skip_row[0]}）")
                continue

            print(f"[{i+1}/{total}] {ticker} {task_type}: {f_start} -> {f_end}")

            url = (
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
                f"?startDate={f_start}&endDate={f_end}"
            )
            response, _ = robust_get(url, headers)

            if response is None:
                # DNS 失敗或空回應，已在 robust_get 內印出訊息
                continue

            if response.status_code == 404:
                print(f"  提示: Tiingo 找不到 {ticker} (可能已下市或代碼變更)，記錄至 Skip_List。")
                cursor.execute(
                    "INSERT OR IGNORE INTO Skip_List VALUES (?,?,?,?,?)",
                    (ticker, f_start, f_end, 'NOT_FOUND', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
                continue

            if response.status_code in (429, 403):
                print(
                    f"\n  ⛔ 已達 Tiingo API 請求限制 (狀態碼: {response.status_code})。"
                    f"\n  下個月再執行會自動從斷點續傳！"
                )
                _interrupted = True  # 標記中斷，讓外層迴圈跳出
                break

            if response.status_code != 200:
                print(f"  警告: {ticker} 獲取失敗 (狀態碼: {response.status_code}) - {response.text[:100]}")
                continue

            # 解析 JSON（防止空字串或格式錯誤）
            try:
                data = response.json()
            except Exception as e:
                print(f"  ⚠️  {ticker} JSON 解析失敗（回應可能為空）: {e}，記錄至 Skip_List。")
                cursor.execute(
                    "INSERT OR IGNORE INTO Skip_List VALUES (?,?,?,?,?)",
                    (ticker, f_start, f_end, 'JSON_ERROR', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
                continue

            if not data:
                print(f"  ℹ️  {ticker} 在 {f_start}~{f_end} 區間無資料，記錄至 Skip_List。")
                cursor.execute(
                    "INSERT OR IGNORE INTO Skip_List VALUES (?,?,?,?,?)",
                    (ticker, f_start, f_end, 'NO_DATA', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
                continue

            try:
                df = pd.DataFrame(data)
                
                # 確保按日期從新到舊排序以計算累積拆股因子
                df['date_dt'] = pd.to_datetime(df['date'])
                df = df.sort_values('date_dt', ascending=False)
                
                # 處理 splitFactor，若為 NaN 或 <= 0 則設為 1.0
                df['splitFactor'] = df['splitFactor'].fillna(1.0)
                df.loc[df['splitFactor'] <= 0, 'splitFactor'] = 1.0
                
                # 計算累積拆股因子 (從最新日期向歷史回溯累積)
                # 拆股因子影響的是當天之前的歷史價格，故使用 shift(1)
                df['split_inv'] = 1.0 / df['splitFactor']
                df['cum_split'] = df['split_inv'].shift(1).fillna(1.0).cumprod()
                
                # 1. 價格僅進行「拆股調整」(Split-adjusted)，符合 yFinance 的 Open/High/Low/Close 定義
                df['open'] = df['open'] * df['cum_split']
                df['high'] = df['high'] * df['cum_split']
                df['low'] = df['low'] * df['cum_split']
                df['close'] = df['close'] * df['cum_split']
                
                # 2. 交易量僅進行「拆股調整」(Split-adjusted)，符合 yFinance 的 Volume 定義
                df['volume'] = (df['volume'] / df['cum_split']).fillna(0).astype('int64')
                
                # 3. Adj_Close 同時進行「拆股與除權息調整」(Split & Dividend Adjusted)
                # 直接使用 Tiingo 提供的 adjClose
                df['adjClose'] = df['adjClose']
                
                # 恢復成原來的日期升序排序以存入資料庫
                df = df.sort_values('date_dt', ascending=True)
                
                df['date'] = df['date_dt'].dt.strftime('%Y-%m-%d')
                df = df.rename(columns={
                    'date': 'Date', 'open': 'Open', 'high': 'High',
                    'low': 'Low', 'close': 'Close',
                    'adjClose': 'Adj_Close', 'volume': 'Volume'
                })
                df['Symbol'] = ticker
                df_to_save = df[['Date', 'Symbol', 'Open', 'High', 'Low',
                                 'Close', 'Adj_Close', 'Volume']]
                df_to_save.to_sql('Daily_Prices', conn, if_exists='append', index=False)
            except Exception as e:
                print(f"  ❌ 儲存 {ticker} 資料時發生錯誤: {e}")
                continue

            time.sleep(0.1)

        # 每 COMMIT_EVERY 支 commit 一次
        if (i + 1) % COMMIT_EVERY == 0:
            conn.commit()
            conn.execute("BEGIN")
            print(f"  💾 已 commit（已處理 {i+1}/{total} 支）")

        time.sleep(0.2)

    # 結束時最後 commit
    conn.commit()
    print("\n✅ 所有資料已安全儲存。")


# ==========================================
# 稽核工具：找出尚未完成的股票
# ==========================================
def audit_missing_data(conn, tickers, start_date="2000-01-01", end_date="2025-12-31"):
    """
    比對每支股票在資料庫中的覆蓋狀況，印出：
    1. 完全沒有資料的股票
    2. 資料起點比 start_date 晚（缺歷史）
    3. 資料終點比 end_date 早（缺近期）
    4. 日期區間內有異常的缺口（天數差異遠大於正常週末假日）
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')

    cursor = conn.cursor()

    # 一次撈出所有 symbol 的統計（效率較佳）
    cursor.execute("""
        SELECT Symbol, MIN(Date) as min_d, MAX(Date) as max_d, COUNT(*) as cnt
        FROM Daily_Prices
        GROUP BY Symbol
    """)
    rows = cursor.fetchall()
    db_stats = {row[0]: {'min': row[1], 'max': row[2], 'cnt': row[3]} for row in rows}

    # 撈出 Skip_List 中有記錄的 symbol 集合（已確認無資料，視為已完成處理）
    cursor.execute("SELECT DISTINCT Symbol FROM Skip_List")
    skipped_symbols = {row[0] for row in cursor.fetchall()}

    no_data = []
    missing_history = []
    missing_recent = []
    gap_suspects = []

    # 計算目標區間的「日曆天數」，用來估算預期交易日數（約 ÷ 7 × 5，再扣假日約 10%）
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    total_calendar_days = (end_dt - start_dt).days
    # 粗估預期交易日（保守估計，不含假日）
    expected_trading_days_approx = int(total_calendar_days * 5 / 7 * 0.94)

    EARLY_THRESHOLD_DAYS = 10   # 起點誤差超過 10 天才算缺失
    LATE_THRESHOLD_DAYS  = 10   # 終點誤差超過 10 天才算缺失
    GAP_RATIO_THRESHOLD  = 0.6  # 實際筆數 < 預期 × 0.6 才懷疑有缺口

    for ticker in tickers:
        stats = db_stats.get(ticker)

        if stats is None:
            # Skip_List 有記錄 → 已確認處理完畢，不列為未下載
            if ticker in skipped_symbols:
                continue
            no_data.append(ticker)
            continue

        min_d = datetime.strptime(stats['min'], '%Y-%m-%d')
        max_d = datetime.strptime(stats['max'], '%Y-%m-%d')
        cnt = stats['cnt']

        # 起點太晚
        if (min_d - start_dt).days > EARLY_THRESHOLD_DAYS:
            missing_history.append((ticker, stats['min'], stats['max'], cnt))

        # 終點太早
        if (end_dt - max_d).days > LATE_THRESHOLD_DAYS:
            missing_recent.append((ticker, stats['min'], stats['max'], cnt))

        # 日期內部缺口嫌疑（實際筆數比預期少太多）
        # 用股票自身的日期範圍來估算預期天數，更精準
        ticker_calendar_days = (max_d - min_d).days
        ticker_expected = int(ticker_calendar_days * 5 / 7 * 0.94)
        if ticker_expected > 50 and cnt < ticker_expected * GAP_RATIO_THRESHOLD:
            gap_suspects.append((ticker, stats['min'], stats['max'], cnt, ticker_expected))

    # ---- 計算彙總數字 ----
    # 「歷史資料不足」= missing_history ∪ missing_recent ∪ gap_suspects（去重）
    incomplete_set = set(
        [t for t, *_ in missing_history]
        + [t for t, *_ in missing_recent]
        + [t for t, *_ in gap_suspects]
    )
    no_data_set    = set(no_data)
    total          = len(tickers)
    cnt_no_data    = len(no_data_set)
    cnt_incomplete = len(incomplete_set - no_data_set)  # 有資料但不完整
    cnt_skipped    = len(skipped_symbols & set(tickers)) # Skip_List 中屬於本次清單的
    cnt_complete   = total - cnt_no_data - cnt_incomplete

    # ---- 印出標題框（含彙總） ----
    print(f"\n{'='*60}")
    print(f"📊 資料完整性稽核報告")
    print(f"   目標區間    : {start_date} ~ {end_date}")
    print(f"   股票總數    : {total:>5}")
    print(f"   已完成股票數: {cnt_complete:>5}  （含 {cnt_skipped} 支確認無資料）")
    print(f"   歷史資料不足: {cnt_incomplete:>5}  （含近期缺漏、內部缺口）")
    print(f"   未下載股票數: {cnt_no_data:>5}")
    print(f"{'='*60}")

    # 匯出 CSV 方便後續追蹤
    summary_rows = []
    for t in no_data:
        summary_rows.append({'Symbol': t, 'Status': 'NO_DATA', 'Min_Date': '', 'Max_Date': '', 'Count': 0})
    for t, mn, mx, c in missing_history:
        summary_rows.append({'Symbol': t, 'Status': 'MISSING_HISTORY', 'Min_Date': mn, 'Max_Date': mx, 'Count': c})
    for t, mn, mx, c in missing_recent:
        summary_rows.append({'Symbol': t, 'Status': 'MISSING_RECENT', 'Min_Date': mn, 'Max_Date': mx, 'Count': c})
    for t, mn, mx, c, exp in gap_suspects:
        summary_rows.append({'Symbol': t, 'Status': 'GAP_SUSPECT', 'Min_Date': mn, 'Max_Date': mx, 'Count': c})

    if summary_rows:
        df_report = pd.DataFrame(summary_rows).drop_duplicates(subset=['Symbol', 'Status'])
        # 取得 db 檔案所在目錄，將報告存在同一層
        try:
            db_dir = os.path.dirname(os.path.abspath(conn.execute("PRAGMA database_list").fetchone()[2]))
        except Exception:
            db_dir = os.getcwd()
        report_path = os.path.join(db_dir, "audit_report.csv")
        df_report.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\n📄 稽核報告已匯出至: {report_path}")

    print(f"\n{'='*60}\n")
    return no_data, missing_history, missing_recent, gap_suspects


# ==========================================
# 儲存成份股基本資訊
# ==========================================
def save_constituents_info(df_current, conn):
    if df_current.empty:
        return

    df_current = df_current.copy()
    df_current.columns = [c.replace(' ', '_') for c in df_current.columns]

    mapping = {
        'Symbol': 'Symbol',
        'Security': 'Security',
        'GICS_Sector': 'GICS_Sector',
        'GICS_Sub-Industry': 'GICS_Sub_Industry',
        'Headquarters_Location': 'Headquarters_Location',
        'Date_added': 'Date_Added',
        'CIK': 'CIK'
    }
    df_current = df_current.rename(columns=mapping)
    df_current['Symbol'] = df_current['Symbol'].str.replace('.', '-')

    valid_cols = ['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry',
                  'Headquarters_Location', 'Date_Added', 'CIK']
    df_current = df_current[[c for c in valid_cols if c in df_current.columns]]
    df_current.to_sql('Constituents', conn, if_exists='replace', index=False)
    print("成分股基本資訊已更新。")


# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "dataset")  
    os.makedirs(data_dir, exist_ok=True)
    db_file = os.path.join(data_dir, "sp500Full.db")  
    abs_db_path = os.path.abspath(db_file)

    print(f"\n{'='*55}")
    print(f"📁 資料庫路徑: {abs_db_path}")
    print(f"{'='*55}\n")

    all_tickers, df_info = get_sp500_tickers()

    if not all_tickers:
        print("無法取得股票清單，程式終止。")
        sys.exit(1)

    connection = setup_database(abs_db_path)
    _conn_global = connection  # 讓 signal handler 能存取

    try:
        save_constituents_info(df_info, connection)

        # 執行模式選擇
        import argparse
        parser = argparse.ArgumentParser(description="S&P 500 資料下載工具")
        parser.add_argument('--audit-only', action='store_true',
                            help='只執行稽核，不下載新資料')
        args, _ = parser.parse_known_args()

        if args.audit_only:
            audit_missing_data(connection, all_tickers,
                               start_date="2000-01-01", end_date="2025-12-31")
        else:
            download_and_save_data(all_tickers, connection,
                                   start_date="2000-01-01", end_date="2025-12-31")
            # 下載完成後自動稽核
            audit_missing_data(connection, all_tickers,
                               start_date="2000-01-01", end_date="2025-12-31")
            print(f"\n🎉 任務完成！資料已儲存至 {abs_db_path}")

    except Exception as e:
        print(f"\n❌ 主程式發生未預期錯誤: {e}")
        connection.commit()  # 盡力儲存
        raise
    finally:
        if not _interrupted:  # 若已被 signal handler 關閉，不要重複關閉
            try:
                connection.commit()
                connection.close()
            except Exception:
                pass