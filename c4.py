from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


import numpy as np
import statsmodels.api as sm

def compute_hurst(ts):
    if len(ts) < 20: return 0.5
    lags = range(2, 20)
    # 確保不會遇到負數或零的警告
    with np.errstate(invalid='ignore', divide='ignore'):
        var_diff = [np.var(ts[lag:] - ts[:-lag]) for lag in lags]
    
    valid = [v > 0 and not np.isnan(v) for v in var_diff]
    if sum(valid) < 5: return 0.5
    
    poly = np.polyfit(np.log(np.array(lags)[valid]), np.log(np.array(var_diff)[valid]), 1)
    return poly[0] / 2.0
    
def compute_half_life(ts):
    z_lag = np.roll(ts, 1)
    z_lag[0] = 0
    z_ret = ts - z_lag
    z_ret[0] = 0
    z_lag2 = sm.add_constant(z_lag)
    try:
        res = sm.OLS(z_ret[1:], z_lag2[1:]).fit()
        hl = -np.log(2) / res.params[1]
        return hl if (hl > 0 and hl < 100) else 15.0
    except:
        return 15.0

def extract_micro_ts_features(ts):
    """提取微觀特徵，並確保回傳值數量與 DataFrame 欄位一致"""
    ticker = ts.name
    try:
        ts_vals = ts.dropna()
        if len(ts_vals) < 30:
            return tuple([ticker] + [np.nan] * 7) # 回傳 8 個元素
            
        # 1. 最近期對數報酬率
        log_ret = np.log(ts_vals / ts_vals.shift(1)).dropna()
        log_ret_latest = log_ret.iloc[-1]
        
        # 2. 波動率 (近似 ATR 概念：High-Low 或純粹標準差)
        atr_latest = ts_vals.rolling(14).std().iloc[-1]
        
        # 3. Z-Score (相對於 60 天均值)
        roll_mean = ts_vals.rolling(60).mean()
        roll_std = ts_vals.rolling(60).std()
        z_score_latest = ((ts_vals - roll_mean) / roll_std).iloc[-1]
        
        # 4. 成交量失衡 (假設目前無 Volume 欄位，先填 0 或結合 VIX)
        vol_imb_latest = 0.0 
        
        # 5. Hurst Exponent
        hurst = compute_hurst(ts_vals.values)
        
        # 6. Half-life (均值回歸半衰期)
        half_life = compute_half_life(ts_vals.values)
        
        # 7. ADF 檢定統計量
        try:
            adf_res = adfuller(ts_vals.values, maxlag=1, regression='c', autolag=None)
            adf_stat = adf_res[0]
        except:
            adf_stat = np.nan
            
        # [修復點] 完整回傳 8 個變數，對應特徵矩陣的欄位
        return ticker, log_ret_latest, atr_latest, z_score_latest, vol_imb_latest, hurst, half_life, adf_stat
        
    except Exception as e:
        # 發生錯誤時也必須回傳 8 個元素，避免 unpack 失敗
        return tuple([ticker] + [np.nan] * 7)

def feature_engineering_pipeline(panel_df, vix_series, n_jobs=-1, n_pca_components=3, scaler_in=None):
    """
    建構橫斷面特徵矩陣，直接準備好輸出給 DBSCAN 使用
    panel_df: 含有 open, high, low, close, volume 的 DataFrame
    vix_series: 此觀測窗口的 VIX 時間序列
    """
    print("啟動特徵工程 Pipeline...")
    
    # 確認 panel_df 是以 [date, ticker] 或是可以直接 pivot
    if not isinstance(panel_df.index, pd.MultiIndex):
        panel_df = panel_df.set_index(['date', 'ticker'])
        
    close_pivot = panel_df['close'].unstack(level='ticker')
    log_returns = np.log(close_pivot / close_pivot.shift(1)).fillna(0)
    
    # A. 將 Close 價格拉平計算全市場因子 (PCA) 與 相關性
    pca = PCA(n_components=n_pca_components)
    market_factors = pca.fit_transform(log_returns)
    reconstructed = pca.inverse_transform(market_factors)
    
    # 提取特有殘差 (PCA Residuals)
    pca_residuals = log_returns - reconstructed
    pca_res_latest = pca_residuals.iloc[-1]
    
    # 動態 Pearson Correlation (與 PCA 第一主成分計算短週期的共同波動性)
    mf_1_series = pd.Series(market_factors[:, 0], index=log_returns.index)
    market_corr = log_returns.apply(lambda x: x.iloc[-20:].corr(mf_1_series.iloc[-20:]))
    
    # B. 分配平行運算：迴圈切割給 CPU 計算各檔股票的微觀高強度特徵
    tickers = close_pivot.columns
    print(f"啟動多執行緒處理 {len(tickers)} 檔股票之時間序列與 ADF 微觀特徵...")
    
    # 將每個 ticker 的歷史 df 提早整理出來給 parallel
    # 避免在 function 裡面重新 filter 導致效率極差
    ticker_dfs = {t: panel_df.xs(t, level='ticker') for t in tickers if t in panel_df.index.get_level_values('ticker')}
    valid_tickers = list(ticker_dfs.keys())

    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_micro_ts_features)(ticker_dfs[t]['close'].rename(t)) for t in valid_tickers
    )
    
    cols = ['ticker', 'Log_Ret', 'ATR', 'Z_Score', 'Vol_Imbalance', 'Hurst', 'Half_life', 'ADF_stat']
    features_df = pd.DataFrame([r for r in results if len(r) == 8], columns=cols).set_index('ticker') # 確保回傳長度相符
    features_df.columns = cols[1:]
    
    # C. 合併市場與總體經濟層面的特徵 (Macro Awareness)
    features_df['PCA_Res'] = pca_res_latest
    features_df['Market_Corr'] = market_corr
    features_df['VIX'] = vix_series.iloc[-1] if not vix_series.empty else np.nan
    
    features_df = features_df.dropna() # 清理計算中斷不穩定的死點
    
    # 1. 剔除 Hurst >= 0.5 的標的 及 確保回歸速度快於半年（約 126 個交易日），避免資金因長期未平倉而產生的機會成本。
    mask = (features_df['Hurst'] < 0.45) & (features_df['Half-life'] < 126)
    features_df = features_df[mask]
    
    # D. 嚴格的特徵標準化機制 (Standardization / Z-Score Scaling)
    if scaler_in is None:
        scaler = StandardScaler()
        fit_mode = True
    else:
        scaler = scaler_in
        fit_mode = False
    feature_columns = ['Log_Ret', 'ATR', 'Z_Score', 'Vol_Imbalance', 'Hurst', 'Half_life', 'ADF_stat', 'PCA_Res', 'Market_Corr', 'VIX']
    
    # 過濾出成功計算出特徵的欄位，防止錯誤
    valid_cols = [c for c in feature_columns if c in features_df.columns]
    if fit_mode:
        std_matrix = scaler.fit_transform(features_df[valid_cols])
    else:
        std_matrix = scaler.transform(features_df[valid_cols])
    
    final_standardized_df = pd.DataFrame(std_matrix, index=features_df.index, columns=valid_cols)
    print(f"特徵矩陣建置完成！維度: {final_standardized_df.shape}")
    
    return final_standardized_df, scaler
