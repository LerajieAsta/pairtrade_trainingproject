# ======================================================================
"""
DTW 配對交易滾動回測系統 (交易明細版) - 許鈞翔 (2025) 論文對齊版
核心功能：
  1. 篩選出 Engle-Granger 共整合檢定 ADF p-value < adf_pvalue_threshold 的股票對。
  2. 對通過共整合的股票對，計算 Z-Score 標準化對數價格的 SSD 與 DTW 距離。
  3. 根據 DTW 距離（對照組）或基於 PCA 融合 SSD 與 DTW 距離的第一主成分 PC1 得分（實驗組）進行升序排序，挑選前 Top N。
"""

import sqlite3
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def _adf_stat(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """ADF 檢定（no constant），同時回傳 (統計量, p 值)"""
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0

def _compute_hurst(series: np.ndarray) -> float:
    """R/S 分析近似 Hurst 指數"""
    n = len(series)
    if n < 20:
        return 0.5
    diffs = np.diff(series)
    rs_list = []
    for seg_len in [n // 4, n // 2, n]:
        if seg_len < 4:
            continue
        seg = diffs[:seg_len]
        mean_seg = np.mean(seg)
        deviate = np.cumsum(seg - mean_seg)
        std_val = np.std(seg, ddof=1)
        if std_val < 1e-8:
            std_val = 1e-8
        rs = (np.max(deviate) - np.min(deviate)) / std_val
        rs_list.append((np.log(seg_len), np.log(rs + 1e-8)))
    if len(rs_list) < 2:
        return 0.5
    xs, ys = zip(*rs_list)
    try:
        h = float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        h = 0.5
    return float(np.clip(h, 0.0, 1.0))

def _sakoe_chiba_dtw(x: np.ndarray, y: np.ndarray, window: int = 15) -> float:
    """
    Sakoe-Chiba 限制窗口的快速 DTW (Dynamic Time Warping) 距離。
    時間軸扭曲限制在 `window` 天之內，時間複雜度為 O(N * W)。
    """
    n = len(x)
    m = len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    
    for i in range(1, n + 1):
        start_j = max(1, i - window)
        end_j = min(m, i + window)
        for j in range(start_j, end_j + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dp[i, j] = cost + min(
                dp[i - 1, j],     # Insertion
                dp[i, j - 1],     # Deletion
                dp[i - 1, j - 1]  # Match
            )
            
    return float(dp[n, m])

def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """簡易 OLS：y = alpha + beta * x + resid，回傳 (alpha, beta, residuals)"""
    n = len(y)
    x_mat = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, y - np.mean(y)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    return alpha, beta, y - alpha - beta * x

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）- 許鈞翔論文重構版
# ══════════════════════════════════════════════════════════════════════════════
class Formation:
    """
    負責在形成期 (Formation Period) 篩選最佳配對。
    流程：先共整合檢定過濾 -> 計算 SSD 與 DTW 距離 -> 基於 DTW 或是 SSD+DTW (PCA) 排序。
    """
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, 
                 sector_mapping: dict = None, min_tickers_for_pairing: int = 2, dtw_window: int = 15,
                 method: str = "dtw", adf_pvalue_threshold: float = 0.01):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.dtw_window = dtw_window
        self.method = method.lower()
        self.adf_pvalue_threshold = adf_pvalue_threshold

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.mean_prices: pd.Series = pd.Series(dtype=float)
        self.std_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """對數價格標準化"""
        log_prices = np.log(self.price_df)
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        self.normalized_df = (log_prices - self.mean_prices) / self.std_prices
        return self.normalized_df

    def compute_pairs(self) -> pd.DataFrame:
        """先共整合篩選，再計算 SSD / DTW 距離"""
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        records = []

        # 根據產業分類分組
        sector_groups = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        skipped_unknown_count = 0
        for sector, sector_tickers in sector_groups.items():
            if sector == "Unknown":
                skipped_unknown_count = len(sector_tickers)
                continue
            
            if len(sector_tickers) < self.min_tickers_for_pairing: 
                continue

            n_sec = len(sector_tickers)
            for i in range(n_sec):
                ticker_b = sector_tickers[i]
                x_val = self.normalized_df[ticker_b].values
                var_x = np.var(x_val, ddof=1)
                
                for j in range(i + 1, n_sec):
                    ticker_a = sector_tickers[j]
                    y_val = self.normalized_df[ticker_a].values
                    
                    # 1. 雙向 OLS 擬合與共整合檢定
                    al_ab, be_ab, re_ab = _ols(y_val, x_val)
                    stat_ab, pval_ab = _adf_stat(re_ab, 1)

                    al_ba, be_ba, re_ba = _ols(x_val, y_val)
                    stat_ba, pval_ba = _adf_stat(re_ba, 1)

                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ticker_a, ticker_b
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = ticker_b, ticker_a

                    # A. ADF 共整合篩選 (依論文預設 adf_pvalue_threshold 為 0.01)
                    if best_pval >= self.adf_pvalue_threshold:
                        continue
                        
                    # B. Ornstein-Uhlenbeck 半衰期過濾 (2.0 <= halflife <= 40.0 天)
                    dy = np.diff(best_resid)
                    y_lag = best_resid[:-1]
                    n_dy = len(dy)
                    x_mat = np.column_stack([np.ones(n_dy), y_lag])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(x_mat, dy, rcond=None)
                        lambda_val = coeffs[1]
                    except Exception:
                        lambda_val = 0.0
                        
                    if lambda_val >= 0.0:
                        continue
                        
                    halflife = -np.log(2) / lambda_val
                    if halflife < 2.0 or halflife > 40.0:
                        continue
                        
                    # C. Hurst 指數篩選 (Hurst < 0.40)
                    hurst = _compute_hurst(best_resid)
                    if hurst >= 0.40:
                        continue
                    
                    # 通過篩選後，計算 SSD 與 DTW 距離
                    norm_a = self.normalized_df[best_a].values
                    norm_b = self.normalized_df[best_b].values
                    
                    # SSD 距離
                    ssd_dist = float(np.sum((norm_a - norm_b) ** 2))
                    
                    # DTW 距離 (Sakoe-Chiba window)
                    dtw_dist = _sakoe_chiba_dtw(norm_a, norm_b, window=self.dtw_window)
                    
                    spread_mean = np.mean(best_resid)
                    spread_std = np.std(best_resid, ddof=1) if len(best_resid) > 1 else 0.0
                    
                    records.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": best_a, "Ticker_B": best_b,
                        "SSD": round(ssd_dist, 6), "DTW_Dist": round(dtw_dist, 6), 
                        "Hedge_Ratio": round(best_beta, 4),
                        "OLS_Alpha": round(best_alpha, 6),
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std": round(spread_std, 6)
                    })

        if skipped_unknown_count > 0:
            print(f"  [Formation] 跳過 {skipped_unknown_count} 支未分類股票。")

        if not records:
            return pd.DataFrame()
            
        return pd.DataFrame(records)

    def select_pairs(self) -> pd.DataFrame:
        pairs_df = self.compute_pairs()
        if pairs_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 根據指定模式進行排序
        if self.method == "dtw":
            # 論文對照組：依 DTW 距離升序排序
            selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()
        elif self.method == "ssd_dtw_pca":
            # 論文實驗組：標準化後使用 PCA 提取第一主成分排序
            if len(pairs_df) < 2:
                # 樣本太少無法做 PCA，退回 DTW
                selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                
                feats = pairs_df[["SSD", "DTW_Dist"]].values
                feats_scaled = StandardScaler().fit_transform(feats)
                
                pca = PCA(n_components=1, random_state=42)
                scores = pca.fit_transform(feats_scaled).flatten()
                
                # 調整得分方向，確保 loadings 為正，即綜合距離越小，得分越小
                loadings = pca.components_[0]
                if loadings[0] < 0:
                    scores = -scores
                
                pairs_df["PC1_Score"] = scores
                selected = pairs_df.sort_values("PC1_Score").head(self.top_n).copy()
        else:
            selected = pairs_df.sort_values("DTW_Dist").head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        mean_a_list, std_a_list, mean_b_list, std_b_list = [], [], [], []
        for _, row in selected.iterrows():
            mean_a_list.append(self.mean_prices[row["Ticker_A"]])
            std_a_list.append(self.std_prices[row["Ticker_A"]])
            mean_b_list.append(self.mean_prices[row["Ticker_B"]])
            std_b_list.append(self.std_prices[row["Ticker_B"]])
            
        selected["Log_Mean_A"] = mean_a_list
        selected["Log_Std_A"] = std_a_list
        selected["Log_Mean_B"] = mean_b_list
        selected["Log_Std_B"] = std_b_list

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Class 2, 3, 4：無縫相容原版回測引擎
# ══════════════════════════════════════════════════════════════════════════════

from strategies.ssd import Trading, PairState, DataProcessor, RollingBacktester

def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir, db_method='Unknown', dataset_name='Unknown', db_path='results/result.db'):
    import inspect
    from pathlib import Path
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}
    
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester (許鈞翔論文版)...")
    
    class CustomDTWRollingBacktester(RollingBacktester):
        def run(self, price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping):
            max_concurrent = self.trading_window // self.rolling_step
            states = {}
            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)] = {
                    "logs":  [],
                    "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                              for _ in range(max_concurrent)],
                }

            roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
            
            for roll_idx, trade_start_idx in enumerate(roll_start_indices):
                form_start_idx = trade_start_idx - self.formation_window
                form_end_idx   = trade_start_idx
                trade_end_idx  = min(trade_start_idx + self.trading_window, total_days)

                form_data   = price_pivot.iloc[form_start_idx:form_end_idx]
                trade_data  = price_pivot.iloc[trade_start_idx:trade_end_idx]
                extended_start = max(0, trade_start_idx - max(self.zscore_window_list))
                extended_data_raw = price_pivot.iloc[extended_start:trade_end_idx]
                valid_cols  = (form_data.isnull().sum() + extended_data_raw.isnull().sum()) == 0
                form_data   = form_data.loc[:, valid_cols]
                trade_dates = trade_data.index
                extended_data  = extended_data_raw.loc[:, valid_cols]

                if form_data.shape[1] < 2 or trade_data.empty:
                    continue

                ts_str = str(all_dates[trade_start_idx])[:10]
                te_str = str(all_dates[trade_end_idx - 1])[:10]
                fs_str = str(all_dates[form_start_idx])[:10]
                fe_str = str(all_dates[form_end_idx - 1])[:10]
                print(f"  ▶ 處理中：第 {roll_idx+1:02d} 期 (交易: {ts_str} ~ {te_str})")

                # 使用本檔案定義的 Formation
                formation = Formation(
                    price_df=form_data,
                    form_start=fs_str, form_end=fe_str,
                    top_n=max(self.top_n_list) * 5,
                    sector_mapping=sector_mapping,
                    min_tickers_for_pairing=self.min_tickers_for_pairing,
                    dtw_window=15,
                    method=params.get("method", "dtw"),
                    adf_pvalue_threshold=params.get("adf_pvalue_threshold", 0.01),
                )
                max_selected_pairs = formation.run()

                if max_selected_pairs.empty:
                    continue

                for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                    self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                    self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
                ):
                    if sec_ratio > 0:
                        max_pairs_per_sector = max(1, int(n * sec_ratio))
                        sector_counts = {}
                        diversified_records = []
                        for _, row in max_selected_pairs.iterrows():
                            sec = row["Sector"]
                            if sec not in sector_counts:
                                sector_counts[sec] = 0
                            if sector_counts[sec] < max_pairs_per_sector:
                                diversified_records.append(row)
                                sector_counts[sec] += 1
                            if len(diversified_records) >= n:
                                break
                        selected_pairs = pd.DataFrame(diversified_records).copy()
                    else:
                        selected_pairs = max_selected_pairs.head(n).copy()

                    if selected_pairs.empty:
                        continue

                    selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                    state  = states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)]
                    slots  = state["slots"]

                    free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                    slot_idx   = free_slots[0] if free_slots else min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                    cap_period   = slots[slot_idx]["capital"]
                    cap_per_pair = cap_period / n

                    trading = Trading(
                        price_df=extended_data, trade_dates=trade_dates,
                        selected_pairs=selected_pairs,
                        capital_per_pair=cap_per_pair,
                        fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
                        stop_loss_pct=sl, entry_z=self.entry_z, exit_z=self.exit_z,
                        zscore_window=z_win, allow_reentry=self.allow_reentry,
                        zscore_clip=self.zscore_clip, min_spread_std=self.min_spread_std,
                        use_dynamic_stop=(dyn_z > 0),
                        dynamic_stop_z=dyn_z,
                        portfolio_stop_loss_pct=p_stop,
                        use_vol_adjust=vol_adj,
                    )

                    trade_log_df, period_pnl = trading.run(ts_str, te_str)

                    if not trade_log_df.empty:
                        state["logs"].append(trade_log_df)

                    slots[slot_idx]["capital"]   = max(0, cap_period + period_pnl)
                    slots[slot_idx]["avail_idx"] = trade_end_idx

            self._export_results(states)
    def _export_results(self, states: dict):
        """將每種參數組合的紀錄匯出為資料庫紀錄"""
        from strategies.db_utils import export_df_to_db
        print("\n✅ 回測完成！正在將交易紀錄寫入 SQLite 資料庫...")
        for params_tuple, state in states.items():
            if state["logs"]:
                full_log_df = __import__('pandas').concat(state["logs"], ignore_index=True)
                
                # 解構參數 (因為不同策略可能有不同參數數量，使用通用的 fallback 機制)
                # ssd_basic 包含: (n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)
                # HDBSCAN 包含: (n, sl, min_cluster_size, ...) 等等
                # 我們統一把原本產生 filename 的邏輯簡化，因為現在 path_key 不用做 Regex
                
                # 自動生出一個 Unique Path Key
                import uuid
                n = params_tuple[0] if len(params_tuple) > 0 else 20
                path_key = f"{self.output_dir.name}/TradeLogs_Top{n}_{uuid.uuid4().hex[:8]}.csv"
                
                # 建構這次網格的參數字典 (使用 getattr 動態讀取自 engine)
                # 注意：其實 df_out 在 simulate_pair 時已經寫入了這些參數，我們在這裡可以讀取 df_out 的第一筆記錄做為參數 (這最準確！)
                # 不過最安全的是傳入 dict，如果沒法完美解析 params_tuple，我們可以傳入 kwargs
                
                # 簡單暴力解：從 self 取出全部屬性當作 params 字典傳下去 (RollingBacktester 本來就有存！)
                grid_params = {}
                for key in dir(self):
                    if not key.startswith('_') and not callable(getattr(self, key)):
                        grid_params[key] = getattr(self, key)
                        
                # 為了確保 n 等參數更新為當前網格的參數：
                grid_params["top_n"] = n
                if len(params_tuple) >= 2: grid_params["stop_loss_pct"] = params_tuple[1]
                if len(params_tuple) >= 3: 
                    # 判斷第三個參數是 z_win 還是 min_cluster_size
                    if hasattr(self, 'zscore_window_list'):
                        grid_params["zscore_window"] = params_tuple[2]
                    else:
                        grid_params["min_cluster_size"] = params_tuple[2]

                success = export_df_to_db(
                    df=full_log_df,
                    strategy_name=getattr(self, "db_method", "Unknown"),
                    params=grid_params,
                    dataset_name=getattr(self, "dataset_name", "Unknown"),
                    path_key=path_key,
                    db_path=getattr(self, "db_path", "results/result.db"),
                    overwrite=True
                )
                
                if success:
                    print(f"  - 已成功寫入 DB: {path_key} (共 {len(full_log_df)} 筆紀錄)")
                else:
                    print(f"  - ⚠️ 寫入 DB 失敗: {path_key}")
                
        print(f"\\n📁 所有交易紀錄已成功寫入資料庫！")

    engine = CustomDTWRollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
