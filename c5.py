from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def dynamic_hdbscan_clustering(features_df, sector_map, min_cluster_size=3):
    """
    HDBSCAN 動態分群 (板塊內層過濾)
    自動標記離群值 (Label = -1) 並輸出品質驗證指標。
    """
    features_df = features_df.copy()
    features_df['sector'] = features_df.index.map(lambda x: sector_map.get(x, 'Unknown'))
    feature_cols = [c for c in features_df.columns if c != 'sector']
    
    cluster_labels = pd.Series(index=features_df.index, dtype=int)
    cluster_labels[:] = -1
    
    global_offset = 0
    all_metrics = []
    
    for sector, group in features_df.groupby('sector'):
        if len(group) < min_cluster_size * 2:
            continue
            
        X = group[feature_cols].values
        
        # 實作: HDBSCAN 演算法
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
        labels = clusterer.fit_predict(X)
        
        # 自動忽略 RuntimeWarning 發生的輪廓計算
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid_mask = labels != -1
            if len(set(labels[valid_mask])) > 1:
                try:
                    sil = silhouette_score(X[valid_mask], labels[valid_mask])
                    ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
                    all_metrics.append({'sector': sector, 'silhouette': sil, 'calinski_harabasz': ch})
                except:
                    pass
            
        new_labels = []
        for l in labels:
            if l == -1:
                new_labels.append(-1)
            else:
                new_labels.append(l + global_offset)
                
        cluster_labels.loc[group.index] = new_labels
        if len(set(labels)) > 1:
            global_offset += max(labels) + 1
            
    if all_metrics:
        avg_sil = np.mean([x['silhouette'] for x in all_metrics])
        avg_ch = np.mean([x['calinski_harabasz'] for x in all_metrics])
        # print(f"  [Cluster Quality] Avg Silhouette: {avg_sil:.3f}, Avg CH Index: {avg_ch:.1f}")
        
    return cluster_labels


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def test_single_coint(a, b, sector, price_window, p_threshold, min_len):
    series_a = price_window.get(a)
    series_b = price_window.get(b)
    
    if series_a is None or series_b is None:
        return None
        
    aligned = pd.concat([series_a, series_b], axis=1, join='inner').dropna()
    if len(aligned) < min_len:
        return None
        
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    
    try:
        x_const = sm.add_constant(x)
        res = sm.OLS(y, x_const).fit()
        beta = res.params.iloc[1]
        
        score, pvalue, _ = coint(y, x, maxlag=1)
        
        # 強制過濾掉不合理 Beta，保護保證金不會因過度不對等爆倉
        if pvalue < p_threshold and 0.5 <= beta <= 2.0:
            return {
                'stock_a': a,
                'stock_b': b,
                'sector': sector,
                'p_value': pvalue,
                'beta': beta
            }
    except:
        pass
        
    return None
        
    aligned = pd.concat([series_a, series_b], axis=1, join='inner').dropna()
    if len(aligned) < min_len: return None
        
    y, x = aligned.iloc[:, 0], aligned.iloc[:, 1]
    
    try:
        x_const = sm.add_constant(x)
        res = sm.OLS(y, x_const).fit()
        beta = res.params.iloc[1]
        score, pvalue, _ = coint(y, x, maxlag=1)
        
        # 強制過濾掉不合理 Beta，保護保證金不會因過度不對等爆倉
        if pvalue < p_threshold and 0.5 <= beta <= 2.0:
            return {'stock_a': a, 'stock_b': b, 'sector': sector, 'p_value': pvalue, 'beta': beta}
    except:
        pass
    return None


def select_pairs_with_hdbscan(price_window, features_df, sector_map, top_ns, coint_pval=COINT_P_VALUE):
    if isinstance(top_ns, int):
        top_ns = [top_ns]
    """
    雙層過濾機制：
    第一層 HDBSCAN 動態群集
    第二層 Engle-Granger Cointegration
    """
    labels = dynamic_hdbscan_clustering(features_df, sector_map)
    unique_clusters = set(labels) - {-1}
    candidates = []
    
    for cid in unique_clusters:
        cluster_tickers = labels[labels == cid].index.tolist()
        if len(cluster_tickers) >= 2:
            s_map = features_df.loc[cluster_tickers[0], 'sector'] if 'sector' in features_df.columns else sector_map.get(cluster_tickers[0])
            for a, b in combinations(cluster_tickers, 2):
                candidates.append((a, b, s_map))
                
    if not candidates:
        return {n: [] for n in top_ns}
        
    print(f"完成排列組合，共產生 {len(candidates)} 組配對。前 5 組配對：")
    for i, c in enumerate(candidates[:5]):
        print(f"  {i+1}: {c[0]} vs {c[1]} (Sector: {c[2]})")
        
    min_len = len(price_window) * 0.8
    norm = price_window / price_window.iloc[0]
    
    coint_results = Parallel(n_jobs=-1, batch_size='auto')(
        delayed(test_single_coint)(a, b, sector, price_window, coint_pval, min_len)
        for a, b, sector in candidates
    )
    # 加入過零率門檻 (一年內至少過零 12 次，即每月平均一次)
    passed = [res for res in coint_results if res is not None and res['zero_cross'] >= 12]
    if not passed:
        return {n: [] for n in top_ns}
        
    print(f"P-value 檢定過濾後，剩餘 {len(passed)} 組配對。前 5 組配對：")
    for i, p in enumerate(passed[:5]):
        print(f"  {i+1}: {p['stock_a']} vs {p['stock_b']} (Beta: {p['beta']:.4f}, P-value: {p['p_value']:.4f})")
        
    for p in passed:
        try:
            # [修正點] 依照文獻對齊：Spread = Price_A - Beta * Price_B
            beta = p['beta']
            a_norm = norm[p['stock_a']]
            b_norm = norm[p['stock_b']]
            
            # 計算共整價差
            spread = a_norm - beta * b_norm
            
            spread_std = spread.std()
            if spread_std < (TRANSACTION_COST * 2): 
                p['ssd'] = np.inf 
                p['zero_cross'] = 0
                continue # 跳過後續計算，直接處理下一組

            # SSD 為價差的平方和
            p['ssd'] = (spread**2).sum()
            
            # 計算過零率 (均值回歸特徵)
            centered = spread - spread.mean()
            zero_crossings = ((centered.shift(1) * centered) < 0).sum()
            p['zero_cross'] = zero_crossings
            
            # 計算半衰期與獲利空間
            p['half_life'] = compute_half_life(spread.values)
            p['profit_space'] = spread.max() - spread.min()
        except:
            p['ssd'] = np.inf
            p['zero_cross'] = 0
            p['half_life'] = np.inf
            p['profit_space'] = 0
            
    # 依照 SSD 排序並選出 Top-N
    df_passed = pd.DataFrame(passed)
    if df_passed.empty:
        return {n: [] for n in top_ns}
        
    # 2. 強行過濾零交叉跟半衰期
    try:
        FORMATION_WINDOW = 252 # 取自全域變數或預設值
    except:
        FORMATION_WINDOW = 252 
        
    df_passed = df_passed[~((df_passed['zero_cross'] < 12) & (df_passed['half_life'] > FORMATION_WINDOW / 2))]
    if df_passed.empty:
        return {n: [] for n in top_ns}
        
    # 3. 綜合評分指標：(獲利空間 / 交易次數)
    df_passed['trade_count'] = df_passed['zero_cross'].replace(0, 1)
    df_passed['profit_per_trade'] = df_passed['profit_space'] / df_passed['trade_count']
    
    df_passed['ssd_rank'] = df_passed['ssd'].rank(ascending=True)
    df_passed['profit_rank'] = df_passed['profit_per_trade'].rank(ascending=False)
    df_passed['final_score'] = df_passed['ssd_rank'] + df_passed['profit_rank']
    
    df_passed = df_passed.sort_values('final_score')
    print(f"SSD 排序完成，前 5 組配對：")
    for i, (_, row) in enumerate(df_passed.head(5).iterrows()):
        print(f"  {i+1}: {row['stock_a']} vs {row['stock_b']} (SSD: {row['ssd']:.4f}, Beta: {row['beta']:.4f})")
        
    return {n: df_passed.head(n).to_dict('records') for n in top_ns}