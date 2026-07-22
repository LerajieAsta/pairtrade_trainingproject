"""
形成期共用特徵萃取層（與分群演算法無關）
======================================================================
把「報酬 PCA 因子載荷」特徵抽成中性純函式，供任何分群策略
（HDBSCAN / Agglomerative / K-means / …）共用，避免各策略互相 import
彼此的內部方法。

方法論依據：
  - Avellaneda & Lee (2010)：報酬相關矩陣的特徵向量（eigenportfolios）
    作為統計套利的共同因子座標
  - Sarmento & Horta (2020)：PCA 降維後的報酬表徵作為分群輸入
  - 因子殘差化（研究框架 #1）：分群前移除市場 + 產業共動，讓表徵建於
    特殊性報酬（`_utils._residualize_returns`）

本模組不做分群、不做排序，只把價格矩陣轉為每檔股票的因子暴露向量。
"""
import numpy as np
from sklearn.decomposition import PCA

from strategies.formation._utils import _residualize_returns


def build_return_pca_loadings(
    price_df,
    pca_n_components: int = 15,
    factor_residual: bool = False,
    sector_mapping: dict = None,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    報酬 PCA 因子載荷特徵。

    步驟：
      1. 對數價格 → 日報酬矩陣 R (T-1 × N)
      2. （可選）因子殘差化：移除市場 + 產業共動，保留特殊性報酬
      3. 逐股標準化（等同對相關矩陣做特徵分解）
      4. PCA 取前 k 主成分，loadings = components.T × sqrt(explained_variance)
         （以 sqrt 特徵值加權，使歐氏距離反映因子重要性）

    參數:
        price_df:          形成窗價格寬表（index=日期, columns=股票）
        pca_n_components:  報酬 PCA 因子數
        factor_residual:   True 時先做因子殘差化（需 sector_mapping 提供產業層）
        sector_mapping:    {ticker: GICS 產業}，殘差化的產業因子用
        random_state:      PCA 隨機種子（sqrt-特徵值加權與 SVD 復現用）
        verbose:           是否印出診斷行

    回傳:
        (loadings [N×k], valid_tickers [長度 N])；有效股票不足 2 檔時回 (空陣列, [])。

    注意：本函式為 HDBSCAN_PCA_Loadings._build_feature_matrix 的逐位元等價抽取，
      任何改動須通過 tools/formation_regression.py 的等價測試。
    """
    log_prices = np.log(price_df)
    tickers = log_prices.columns.tolist()

    # 有效性檢查（run_formation 已 dropna，此處為防禦性過濾）
    valid_tickers = []
    cols = []
    for ticker in tickers:
        series = log_prices[ticker].values
        if len(series) < 30 or not np.all(np.isfinite(series)):
            continue
        valid_tickers.append(ticker)
        cols.append(series)

    if len(valid_tickers) < 2:
        return np.empty((0, 0)), []

    # 日報酬矩陣 (T-1 × N)，逐股標準化 → PCA 等同於對相關矩陣做特徵分解
    R = np.diff(np.column_stack(cols), axis=0)

    # 研究框架 #1：分群前先移除市場（+產業）因子，讓 PCA 建在特殊性報酬上
    if factor_residual:
        sm = sector_mapping or {}
        sec = [sm.get(t.upper(), sm.get(t, "Unknown")) for t in valid_tickers]
        R = _residualize_returns(R, sector_labels=sec)
        if verbose:
            print("  [Formation] 因子殘差化：已移除市場+產業共動（分群建於特殊性報酬）")

    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1e-12
    Rs = (R - mu) / sd
    Rs = np.nan_to_num(Rs, nan=0.0, posinf=0.0, neginf=0.0)  # 防退化欄位造成 SVD 不收斂

    n_factors = max(1, min(pca_n_components, Rs.shape[0] - 1, Rs.shape[1] - 1))
    pca = PCA(n_components=n_factors, random_state=random_state)
    try:
        pca.fit(Rs)
    except np.linalg.LinAlgError:
        # 殘差化可能使部分欄位共線 → 預設 gesdd SVD 不收斂；
        # 加微擾動破除退化 + 改用 randomized solver（對退化矩陣幾乎不會不收斂）。
        rng = np.random.default_rng(random_state)
        pca = PCA(n_components=n_factors, svd_solver="randomized",
                  random_state=random_state)
        pca.fit(Rs + rng.normal(0.0, 1e-6, Rs.shape))

    # loadings (N × k)：每檔股票在各風險因子上的暴露，以 sqrt(特徵值) 加權
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    if verbose:
        print(
            f"  [Formation] PCA loadings：{len(valid_tickers)} 檔 × {n_factors} 因子 | "
            f"累計解釋變異 {pca.explained_variance_ratio_.sum():.1%}"
        )
    return loadings, valid_tickers
