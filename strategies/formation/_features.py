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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from strategies.formation._utils import _residualize_returns
from strategies.formation._fundamentals import (
    CANONICAL_SECTORS, UNKNOWN_SECTOR_IDX, canonicalize_sector,
    load_pit_fundamentals, impute_by_group, winsorize,
)


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


def build_fundamentals_mix_features(
    price_df,
    form_end: str,
    sector_mapping: dict = None,
    pca_n_components: int = 5,
    factor_residual: bool = False,
    fundamentals_parquet_path: str = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
    price_feature_weight: float = 1.0,
    fundamentals_feature_weight: float = 1.0,
    sector_onehot_weight: float = 1.0,
    structural_features: tuple = (),      # 結構性財報特徵欄位（SEC XBRL PIT）
    structural_weight: float = 1.0,
    random_state: int = 42,
    min_tickers: int = 2,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    混合特徵：報酬 PCA 因子載荷 ⊕ 公司基本面（log 市值、盈餘殖利率 1/PE）⊕ GICS 產業 one-hot
    ⊕（可選）結構性財報特徵（ROE/ROA/負債比/毛利率… 見 fetch_sec_fundamentals）。

    三個區塊各自 StandardScaler 後依權重拼接（避免 joint 標準化讓 one-hot 欄位數
    稀釋連續特徵的距離量測）。基本面取自 FMP Point-in-Time：對每個形成窗取
    日期 ≤ form_end 的最近一筆記錄（無前視）。

    為 agglomerative_FMP._build_feature_matrix 的逐位元等價抽取；供任何分群策略共用
    （3×3 消融矩陣的固定特徵）。回傳 (X, valid_tickers)，有效股不足時回 (空, [])。
    """
    sm = sector_mapping or {}

    def _lookup_sector(ticker):
        return sm.get(ticker.upper(), sm.get(ticker, "Unknown"))

    price_loadings, valid_tickers = build_return_pca_loadings(
        price_df=price_df, pca_n_components=pca_n_components,
        factor_residual=factor_residual, sector_mapping=sm,
        random_state=random_state, verbose=verbose,
    )
    if len(valid_tickers) < min_tickers:
        return np.empty((0, 0)), []

    pit_df = load_pit_fundamentals(fundamentals_parquet_path)

    # 取 <= form_end 的最近一筆 PIT 記錄
    target_date = pd.to_datetime(form_end)
    current_fundamentals = {}
    if not pit_df.empty:
        dates = pit_df.index.get_level_values("date").unique()
        valid_dates = dates[dates <= target_date]
        if len(valid_dates) > 0:
            closest_date = valid_dates.max()
            current_fundamentals = pit_df.xs(closest_date, level="date").to_dict("index")

    canonical_sectors = []
    market_caps = np.full(len(valid_tickers), np.nan)
    earnings_yields = np.full(len(valid_tickers), np.nan)
    struct_cols = [c for c in structural_features]
    struct_vals = {c: np.full(len(valid_tickers), np.nan) for c in struct_cols}

    for i, ticker in enumerate(valid_tickers):
        rec = current_fundamentals.get(ticker.upper(), {})
        for c in struct_cols:
            v = rec.get(c)
            if v is not None and pd.notna(v) and np.isfinite(float(v)):
                struct_vals[c][i] = float(v)
        raw_sector = rec.get("industry") if pd.notna(rec.get("industry")) else _lookup_sector(ticker)
        canonical_sectors.append(canonicalize_sector(raw_sector))
        market_cap = rec.get("market_cap")
        pe_ratio = rec.get("pe_ratio")
        if pd.notna(market_cap) and market_cap > 0:
            market_caps[i] = np.log1p(market_cap)
        if pd.notna(pe_ratio) and pe_ratio != 0:
            earnings_yields[i] = 1.0 / pe_ratio

    canonical_sectors = np.array(canonical_sectors)
    n_missing_mc = int(np.isnan(market_caps).sum())
    n_missing_pe = int(np.isnan(earnings_yields).sum())

    market_caps = winsorize(impute_by_group(market_caps, canonical_sectors))
    earnings_yields = winsorize(impute_by_group(earnings_yields, canonical_sectors))

    onehot = np.zeros((len(valid_tickers), len(CANONICAL_SECTORS) + 1))
    sector_index = {s: i for i, s in enumerate(CANONICAL_SECTORS)}
    for i, sector in enumerate(canonical_sectors):
        onehot[i, sector_index.get(sector, UNKNOWN_SECTOR_IDX)] = 1.0

    price_scaled = StandardScaler().fit_transform(price_loadings) * price_feature_weight
    fundamentals_cont = np.column_stack([market_caps, earnings_yields])
    fundamentals_scaled = StandardScaler().fit_transform(fundamentals_cont) * fundamentals_feature_weight
    sector_weighted = onehot * sector_onehot_weight

    blocks = [price_scaled, fundamentals_scaled, sector_weighted]

    # 結構性財報特徵區塊：同樣以產業中位數插補 + winsorize + 標準化 + 權重
    n_struct_missing = {}
    if struct_cols:
        cols_arr = []
        for c in struct_cols:
            v = struct_vals[c]
            n_struct_missing[c] = int(np.isnan(v).sum())
            v = winsorize(impute_by_group(v, canonical_sectors))
            cols_arr.append(v)
        struct_mat = np.column_stack(cols_arr)
        struct_scaled = StandardScaler().fit_transform(struct_mat) * structural_weight
        blocks.append(struct_scaled)

    X = np.hstack(blocks)

    if verbose:
        print(
            f"  [Formation] 混合特徵矩陣：{len(valid_tickers)} 檔 | "
            f"價格因子 {price_loadings.shape[1]} 維 + 基本面 2 維（市值缺失 "
            f"{n_missing_mc}、PE 缺失 {n_missing_pe} 已插補）+ 產業 one-hot {onehot.shape[1]} 維"
            + (f" + 結構性財報 {len(struct_cols)} 維（缺失 "
               + ", ".join(f"{c}:{n}" for c, n in n_struct_missing.items()) + " 已插補）"
               if struct_cols else "")
        )
    return X, valid_tickers


def build_momentum_features(
    price_df,
    horizons_months: tuple = (1, 3, 6, 9, 12),
    include_volatility: bool = True,
    block_pca_components: int = 0,
    random_state: int = 42,
    min_tickers: int = 2,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    多尺度動量特徵（純價格，零額外資料成本）。

    文獻依據：Han, He & Toh (2021), *Pairs Trading via Unsupervised Learning* 以
      48 個 1–48 月動量因子 + 78 個公司特徵分群。本函式在形成窗長度（252 日
      = 12 個月）容許範圍內取多個時間尺度的累積報酬——動量捕捉「不同時間尺度
      上的走勢形狀」，與報酬 PCA 載荷（共同因子暴露）互補。

    參數:
        horizons_months:      動量視窗（月），以 21 交易日 ≈ 1 月換算；
                              超過形成窗長度者自動跳過
        include_volatility:   併入 20/60 日已實現波動與下行波動（可交易性特徵）
        block_pca_components: >0 時對整個特徵區塊做 PCA 降維（避免高維動量
                              在後續拼接中稀釋其他區塊；0 = 不降維）
    回傳 (X [N×d], valid_tickers)；特徵已逐欄標準化。
    """
    log_prices = np.log(price_df)
    tickers = log_prices.columns.tolist()

    valid_tickers, cols = [], []
    for t in tickers:
        s = log_prices[t].values
        if len(s) < 30 or not np.all(np.isfinite(s)):
            continue
        valid_tickers.append(t)
        cols.append(s)
    if len(valid_tickers) < min_tickers:
        return np.empty((0, 0)), []

    LP = np.column_stack(cols)          # (T × N) 對數價格
    T = LP.shape[0]
    feats, names = [], []

    # ── 多尺度動量：各視窗的累積對數報酬 ──────────────────────────────
    for m in horizons_months:
        w = min(int(m * 21), T - 1)      # 視窗上限＝形成窗全長（12 月 ≈ 252 日）
        if w < 5:
            continue
        mom = LP[-1, :] - LP[-1 - w, :]  # log(P_t / P_{t-w})
        if any(n == f"mom{m}m" for n in names):
            continue
        feats.append(mom); names.append(f"mom{m}m")

    # ── 波動 / 可交易性特徵 ───────────────────────────────────────────
    if include_volatility:
        R = np.diff(LP, axis=0)
        for w in (20, 60):
            if w < R.shape[0]:
                feats.append(R[-w:, :].std(axis=0, ddof=1)); names.append(f"vol{w}d")
        # 下行波動（僅負報酬）
        neg = np.where(R < 0, R, 0.0)
        feats.append(neg[-60:, :].std(axis=0, ddof=1) if R.shape[0] >= 60
                     else neg.std(axis=0, ddof=1))
        names.append("downvol60d")

    if not feats:
        return np.empty((0, 0)), []

    X = np.column_stack(feats)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    if block_pca_components and block_pca_components < X.shape[1]:
        k = max(1, min(block_pca_components, X.shape[0] - 1, X.shape[1]))
        X = PCA(n_components=k, random_state=random_state).fit_transform(X)
        if verbose:
            print(f"  [Formation] 動量特徵：{len(valid_tickers)} 檔 × {len(names)} 原始特徵 "
                  f"→ PCA {k} 維（{', '.join(names)}）")
    elif verbose:
        print(f"  [Formation] 動量特徵：{len(valid_tickers)} 檔 × {len(names)} 維 "
              f"（{', '.join(names)}）")
    return X, valid_tickers
