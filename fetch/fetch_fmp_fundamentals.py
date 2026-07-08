import requests
import pandas as pd
import numpy as np
import time
import os
import sqlite3
from typing import List, Dict, Set, Optional


class FMPPITDataPipeline:
    """
    Financial Modeling Prep (FMP) Point-in-Time (PIT) 數據處理管道
    專為 S&P 500 成分股（2000–2025）長週期回測設計，具備動態成分股重建與無前視偏誤特徵計算。
    """

    def __init__(self, api_key: str, cache_dir: str = "dataset/fundamental/fmp_cache", tiingo_db_path: str = "dataset/price/sp500_Tiingo.db"):
        self.api_key = api_key
        # FMP 於 2024-08 停用 /api/v3 legacy 端點（回 403 Legacy Endpoint），
        # 全面遷移至 /stable，改以 ?symbol= query 參數呼叫。
        self.base_url = "https://financialmodelingprep.com/stable"
        self.cache_dir = cache_dir
        self.tiingo_db_path = tiingo_db_path
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_request(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """
        帶有指數退避重試機制的 API 請求模組
        """
        if params is None:
            params = {}
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        retries = 5
        delay = 1
        for i in range(retries):
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    # 不再靜默回傳 []：把 HTTP 錯誤明確拋出，避免像 403 Legacy /
                    # 402 Restricted 這類權限問題被吞掉、產生全 NaN 的資料集而不自知。
                    raise RuntimeError(
                        f"FMP API {response.status_code} @ {endpoint}: {response.text[:200]}"
                    )
            except requests.exceptions.RequestException:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"FMP API 重試 {retries} 次仍失敗 @ {endpoint}")

    def fetch_sp500_current_constituents(self) -> List[Dict]:
        """
        獲取當前 S&P 500 成分股與行業分類資訊 (從本地 Tiingo DB)
        """
        if not os.path.exists(self.tiingo_db_path):
            raise FileNotFoundError(f"找不到 Tiingo 資料庫: {self.tiingo_db_path}")
        conn = sqlite3.connect(self.tiingo_db_path)
        df = pd.read_sql("SELECT Symbol as symbol, GICS_Sector as sector FROM Constituents", conn)
        conn.close()
        return df.to_dict('records')

    def reconstruct_sp500_constituents_by_date(
        self, 
        target_dates: List[pd.Timestamp]
    ) -> Dict[pd.Timestamp, Set[str]]:
        """
        依據 Tiingo 的 index_memberships 表，查詢特定日期哪些股票屬於 S&P 500。
        """
        if not os.path.exists(self.tiingo_db_path):
            raise FileNotFoundError(f"找不到 Tiingo 資料庫: {self.tiingo_db_path}")
            
        conn = sqlite3.connect(self.tiingo_db_path)
        mem_df = pd.read_sql("SELECT Symbol, start_date, end_date FROM index_memberships", conn)
        conn.close()
        
        mem_df['start_date'] = pd.to_datetime(mem_df['start_date'], errors='coerce')
        mem_df['end_date'] = pd.to_datetime(mem_df['end_date'], errors='coerce')
        
        date_constituents = {}
        for current_date in target_dates:
            mask = (mem_df['start_date'] <= current_date) & \
                   (mem_df['end_date'].fillna(pd.Timestamp('2099-12-31')) >= current_date)
            active_set = set(mem_df[mask]['Symbol'])
            date_constituents[current_date] = active_set
            
        return date_constituents

    def fetch_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        獲取歷史每日收盤價，優先從本地 Tiingo 資料庫獲取，若無則降級為 FMP API 並存入快取
        """
        # 1. 優先嘗試從 Tiingo DB 獲取 Adj_Close
        if os.path.exists(self.tiingo_db_path):
            try:
                conn = sqlite3.connect(self.tiingo_db_path)
                query = "SELECT Date as date, Adj_Close as close FROM Daily_Prices WHERE Symbol = ? AND Date BETWEEN ? AND ?"
                df = pd.read_sql(query, conn, params=(ticker, start_date, end_date), parse_dates=["date"])
                conn.close()
                if not df.empty:
                    return df.sort_values("date").reset_index(drop=True)
            except Exception as e:
                print(f"  [Warning] 從 Tiingo DB 獲取 {ticker} 股價失敗 ({e})，降級為 FMP API...")

        # 2. 降級：從本地快取或 FMP API 獲取
        cache_file = os.path.join(self.cache_dir, f"{ticker}_prices.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=["date"])
            # 篩選符合請求範圍的子集
            mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
            if not df[mask].empty:
                return df[mask].reset_index(drop=True)

        # /stable：historical-price-eod/full?symbol=，回傳為扁平 list（非 {"historical": [...]}）
        try:
            data = self._get_request("historical-price-eod/full",
                                     {"symbol": ticker, "from": start_date, "to": end_date})
        except RuntimeError as e:
            print(f"  [Warning] 股價抓取失敗 {ticker}: {e}")
            return pd.DataFrame()

        rows = data.get("historical") if isinstance(data, dict) else data
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df.to_csv(cache_file, index=False)
        return df[["date", "close"]]

    def fetch_historical_market_cap(self, ticker: str, limit: int = 10000) -> pd.DataFrame:
        """
        獲取歷史每日市值，優先讀取本地快取
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_mcap.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=["date"])
            return df

        # /stable：symbol 改為 query 參數
        try:
            data = self._get_request("historical-market-capitalization",
                                     {"symbol": ticker, "limit": limit})
        except RuntimeError as e:
            print(f"  [Warning] 市值抓取失敗 {ticker}: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df[["date", "marketCap"]].to_csv(cache_file, index=False)
        return df[["date", "marketCap"]]

    def fetch_quarterly_eps_pit(self, ticker: str) -> pd.DataFrame:
        """
        獲取歷史季度每股盈餘（EPS），並保留申報日（fillingDate）作為 PIT 對齊點
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_eps.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=["fiscal_date", "filling_date"])

        # /stable：symbol 改為 query 參數；欄位改為 stable 命名（epsDiluted / filingDate），
        # 並相容 v3 舊命名（epsdiluted / fillingDate）以防混用。
        try:
            data = self._get_request("income-statement",
                                     {"symbol": ticker, "period": "quarter", "limit": 120})
        except RuntimeError as e:
            print(f"  [Warning] 財報抓取失敗 {ticker}: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        eps_col = "epsDiluted" if "epsDiluted" in df.columns else "epsdiluted"
        fil_col = "filingDate" if "filingDate" in df.columns else "fillingDate"
        for col in ("date", fil_col, eps_col):
            if col not in df.columns:
                df[col] = np.nan

        df["fiscal_date"] = pd.to_datetime(df["date"])
        df["filling_date"] = pd.to_datetime(df[fil_col])
        df["filling_date"] = df["filling_date"].fillna(df["fiscal_date"] + pd.Timedelta(days=45))
        df["epsdiluted"] = pd.to_numeric(df[eps_col], errors="coerce")

        result_df = df[["fiscal_date", "filling_date", "epsdiluted"]].sort_values("filling_date").reset_index(drop=True)
        result_df.to_csv(cache_file, index=False)
        return result_df

    def calculate_pit_ttm_eps(self, eps_df: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """
        在不產生前視偏誤的前提下，計算每日的 Point-in-Time TTM EPS。
        """
        if eps_df.empty:
            return pd.DataFrame(index=date_range, columns=["ttm_eps"], data=np.nan)

        unique_filling_dates = eps_df["filling_date"].dropna().unique()
        unique_filling_dates = sorted([d for d in unique_filling_dates if d in date_range or d < date_range.max()])
        
        pit_records = []
        for current_date in unique_filling_dates:
            available_eps = eps_df[eps_df["filling_date"] <= current_date]
            if len(available_eps) < 4:
                continue
            
            latest_4_quarters = available_eps.sort_values("fiscal_date", ascending=False).head(4)
            if len(latest_4_quarters) == 4:
                ttm_eps = latest_4_quarters["epsdiluted"].sum()
                pit_records.append({"date": current_date, "ttm_eps": ttm_eps})
                
        if not pit_records:
            return pd.DataFrame(index=date_range, columns=["ttm_eps"], data=np.nan)
            
        pit_df = pd.DataFrame(pit_records)
        pit_df["date"] = pd.to_datetime(pit_df["date"])
        pit_df = pit_df.set_index("date").reindex(date_range).ffill()
        return pit_df

    def build_single_stock_pit_features(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        建立單一股票無前視偏誤的每日 PIT 特徵矩陣
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        
        price_df = self.fetch_historical_prices(ticker, start_date, end_date)
        if price_df.empty:
            return pd.DataFrame(index=date_range, columns=["close", "market_cap", "pe_ratio"])
            
        price_df = price_df.set_index("date").reindex(date_range)
        
        mcap_df = self.fetch_historical_market_cap(ticker)
        if not mcap_df.empty:
            mcap_df = mcap_df.set_index("date").reindex(date_range).ffill()
        else:
            mcap_df = pd.DataFrame(index=date_range, columns=["marketCap"], data=np.nan)
            
        eps_df = self.fetch_quarterly_eps_pit(ticker)
        ttm_eps_df = self.calculate_pit_ttm_eps(eps_df, date_range)
        
        features = pd.DataFrame(index=date_range)
        features["close"] = price_df["close"]
        features["market_cap"] = mcap_df["marketCap"]
        features["ttm_eps"] = ttm_eps_df["ttm_eps"]
        
        features["pe_ratio"] = np.where(
            features["ttm_eps"] > 0,
            features["close"] / features["ttm_eps"],
            np.nan
        )
        
        features = features.drop(columns=["ttm_eps"])
        return features


def process_sp500_pipeline(
    api_key: str,
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    rebalance_freq: str = "ME",  # 'ME' 代表月底重平衡, 可改為 'QE' 季底或 'D' 每日
    output_path: str = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet"
) -> pd.DataFrame:
    """
    S&P 500 長週期 Point-in-Time 數據處理主函數
    """
    pipeline = FMPPITDataPipeline(api_key)
    
    # 1. 定義重平衡觀測點時間序列
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    
    print(f"正在動態計算自 {start_date} 至 {end_date} 間各觀測點的 S&P 500 歷史真實成分股...")
    # 重建成分股對照表 (Date -> Set of Active Tickers)
    historical_constituents = pipeline.reconstruct_sp500_constituents_by_date(list(rebalance_dates))
    
    # 2. 彙整這 26 年內曾經進入過 S&P 500 指數的所有股票清單，進行資料下載
    all_unique_tickers = set()
    for tickers in historical_constituents.values():
        all_unique_tickers.update(tickers)
    
    print(f"歷史觀測區間內共涉及 {len(all_unique_tickers)} 檔獨特股票。開始建立 Point-in-Time 數據庫...")

    # 3. 獲取當前行業分類對照，便於後續中位數插補
    current_const_data = pipeline.fetch_sp500_current_constituents()
    industry_map = {item["symbol"]: item.get("sector", "Unknown") for item in current_const_data}

    all_stock_features = {}
    total_tickers = len(all_unique_tickers)
    
    for i, ticker in enumerate(all_unique_tickers, 1):
        print(f"[{i}/{total_tickers}] 正在下載並計算 {ticker} 的 PIT 數據...")
        # 建立該股完整的日頻 PIT 數據架構
        df = pipeline.build_single_stock_pit_features(ticker, start_date, end_date)
        df["industry"] = industry_map.get(ticker, "Unknown")
        all_stock_features[ticker] = df
        # 避免觸發 API 頻率限制
        time.sleep(0.05)

    # 4. 轉換為 MultiIndex DataFrame (date, ticker) 並過濾出僅限重平衡觀測點的數據
    print("正在合併數據並篩選重平衡日期點...")
    m_df = pd.concat(all_stock_features, names=["ticker", "date"])
    m_df = m_df.reorder_levels(["date", "ticker"]).sort_index()
    
    # 僅篩選出重平衡觀測點的日期
    filtered_df = m_df.loc[m_df.index.get_level_values("date").isin(rebalance_dates)].copy()

    # 5. 確保在每個重平衡時點，僅篩選出當時真正屬於 S&P 500 的成分股
    print("正在執行時點成分股過濾與生存者偏差校正...")
    valid_records = []
    
    for current_date, group in filtered_df.groupby(level="date"):
        active_tickers_on_date = historical_constituents.get(current_date, set())
        # 僅保留當時在指數內部的股票
        active_group = group.loc[group.index.get_level_values("ticker").isin(active_tickers_on_date)].copy()
        
        # 針對這批當時被納入指數但存在資料缺失或已停牌/下市的股票，進行動態行業插補
        medians = active_group.groupby("industry")[["market_cap", "pe_ratio"]].transform("median")
        market_median = active_group[["market_cap", "pe_ratio"]].median()
        
        active_group["market_cap"] = active_group["market_cap"].fillna(medians["market_cap"]).fillna(market_median["market_cap"])
        active_group["pe_ratio"] = active_group["pe_ratio"].fillna(medians["pe_ratio"]).fillna(market_median["pe_ratio"])
        
        valid_records.append(active_group)

    final_dataset = pd.concat(valid_records).sort_index()
    
    # 儲存為 HDF5 或 Parquet 以利後續分析
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_dataset.to_parquet(output_path)
    print(f"數據已導出至 {output_path}")
    
    return final_dataset


if __name__ == "__main__":
    # API key 改由環境變數提供（勿再硬編碼於原始碼）：
    #   Windows PowerShell:  $env:FMP_API_KEY="你的key"; python fetch/fetch_fmp_fundamentals.py
    #   Linux/macOS:         FMP_API_KEY=你的key python fetch/fetch_fmp_fundamentals.py
    FMP_API_KEY = os.environ.get("FMP_API_KEY", "").strip()
    if not FMP_API_KEY:
        raise SystemExit("請先設定環境變數 FMP_API_KEY（FMP /stable API 金鑰）。")

    # 前置檢查：先以單一 ticker 測試付費端點，權限不足時立即明確中止，
    # 避免整批抓完才發現 market_cap/pe 全 NaN。
    _pf = FMPPITDataPipeline(FMP_API_KEY)
    try:
        _probe = _pf.fetch_historical_market_cap("AAPL", limit=2)
        if _probe.empty:
            raise SystemExit("前置檢查：historical-market-capitalization 無資料，請確認金鑰方案。")
        print(f"✅ 前置檢查通過：AAPL 市值端點可用（樣本 {len(_probe)} 筆）。")
    except Exception as e:
        raise SystemExit(f"前置檢查失敗：{e}")

    # 回測時間範圍：2000-01-01 至 2025-12-31
    START = "2000-01-01"
    END = "2025-12-31"
    
    # 預設採用月底重平衡 'ME'（Month End），可依策略改為 'QE' 季底重平衡
    final_data = process_sp500_pipeline(
        api_key=FMP_API_KEY,
        start_date=START,
        end_date=END,
        rebalance_freq="ME",
        output_path="dataset/fundamental/sp500_pit_2000_2025_monthly.parquet"
    )
    
    print("\nS&P 500 PIT 數據處理與生存者偏差校正完成。")
    print("最終資料集維度:", final_data.shape)
    print("\n數據樣本 (前 20 筆):")
    print(final_data.head(20))
