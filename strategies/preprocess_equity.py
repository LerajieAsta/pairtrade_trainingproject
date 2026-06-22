import sqlite3
import pandas as pd
from strategies.db_utils import get_db_connection

class DataProcessor:
    """
    資料庫載入與資料特徵前處理器。
    負責從 SQLite 中載入 Adj_Close，進行數據過濾（如移除停牌或缺值率過高的股票），填補缺失值，解析回測時間等。
    """
    
    def __init__(self, db_path: str, table_name: str = "daily_prices"):
        """
        參數:
            db_path (str): SQLite 資料庫路徑。
            table_name (str): 歷史日收盤價資料表名稱。
        """
        self.db_path, self.table_name = db_path, table_name

    def load_sector_mapping(self, info_table: str, ticker_col: str = "ticker", sector_col: str = "sector") -> dict:
        """
        從指定資料表載入個股對應之產業分類映射字典。

        參數:
            info_table (str): 分類資訊資料表名稱。
            ticker_col (str): 股票代碼欄位名稱。
            sector_col (str): 產業分類欄位名稱。

        回傳:
            dict: {股票代碼: 產業分類名稱} 的映射字典。
        """
        try:
            conn = get_db_connection(self.db_path)
            df = pd.read_sql_query(f"SELECT {ticker_col}, {sector_col} FROM {info_table}", conn)
            conn.close()
            mapping = {}
            for k, v in zip(df[ticker_col], df[sector_col]):
                if pd.notna(k) and pd.notna(v):
                    mapping[str(k).strip().upper()] = str(v).strip()
            print(f"✅ 成功載入產業分類表 '{info_table}'，共取得 {len(mapping)} 檔標的分類。")
            return mapping
        except Exception as e:
            print(f"⚠️ [警告] 無法載入產業分類表 '{info_table}'！錯誤原因：{e}")
            print(f"⚠️ 系統將退回「全市場(All_Market)」跨產業配對模式。")
            return {}

    def prepare_backtest_data(self, backtest_start: str, backtest_end: str, formation_window: int):
        """
        載入並重塑價格矩陣，過濾流動性差與大量缺值的個股。

        參數:
            backtest_start (str): 回測起始日期。
            backtest_end (str): 回測結束日期。
            formation_window (int): 形成期天數（需往前多載入資料以供第一期選配）。

        回傳:
            tuple: (價格 Pivot 矩陣, 時間索引序列, 總天數, 交易啟動本地日期索引位置)
        """
        conn = get_db_connection(self.db_path)
        # 合理載入 Adj_Close 或 Close 價格
        raw_df = pd.read_sql_query(f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC", conn)
        conn.close()

        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]

        # 重塑為以日期為 Index，Symbol 為 Columns 的寬表
        price_pivot = raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()

        # 剔除回測區間中缺失值比例大於 20% 的劣質標的，其餘以 forward fill 填補 (限制最多連續填 5 天)
        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20].ffill(limit=5)
        # 進一步確保剩餘股票含有至少 90% 的非空值，防止後續計算報錯
        price_pivot.dropna(axis=1, thresh=int(len(price_pivot) * 0.9), inplace=True)

        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                if is_end and len(str(d_str).strip()) == 7:
                    # 處理如 '2023-12' 自動抓月底日期
                    return dt + pd.offsets.MonthEnd(0)
                return dt
            except Exception:
                return None

        bt_start_ts = _safe_parse(backtest_start)
        bt_end_ts = _safe_parse(backtest_end, is_end=True)
        all_dates = price_pivot.index.tolist()

        start_indices = [i for i, d in enumerate(all_dates) if d >= bt_start_ts] if bt_start_ts else []
        first_idx = start_indices[0] if start_indices else 0

        # 將數據切片起點往前推一個 formation_window，供第一期 Formation 計算
        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end = bt_end_ts if bt_end_ts else price_pivot.index[-1]
        price_pivot = price_pivot.loc[data_slice_start:data_slice_end]

        sliced_dates = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_trade_idx = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_trade_idx, formation_window)


