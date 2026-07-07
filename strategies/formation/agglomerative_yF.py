"""
Agglomerative Clustering + Firm Fundamentals 形成期模組
======================================================================

命題：以「報酬 PCA 因子載荷（價格行為）⊕ GICS 產業 one-hot ⊕ log(市值) ⊕
盈餘殖利率(1/PE)（公司基本面）」混合特徵空間做 Agglomerative Clustering
分組，取代單純 GICS 靜態分組或純價格導向的 HDBSCAN 分組。

為何用 Agglomerative 而非 HDBSCAN/DBSCAN：
  密度式分群（HDBSCAN/DBSCAN）在股票特徵空間上容易產生單一巨型群 + 大量
  雜訊點（label=-1）的極端不平衡分布（本專案 HDBSCAN 系列策略已實測平均
  30%+ 標的被判為雜訊排除，見 archive/config_archived_strategies.py）。
  Agglomerative 對每個點都會分配群組（無「雜訊」概念），且改用
  distance_threshold（依每期合併距離的分位數校準）而非固定 n_clusters，
  可避免「強迫合併到固定群數」重現同樣的不平衡問題（固定 n_clusters 或
  ward linkage 在混合連續+one-hot 特徵空間上同樣容易產生一個巨型群）。

基本面資料為單一時點靜態快照（yfinance 今日資料，見
fetch/fundamentals_yfinance.py），非 2000-2025 逐點歷史資料——免費資料源
無法取得歷史逐日基本面（需 WRDS/Compustat 等付費資料商）。這代表每個歷史
形成窗口都使用同一組「現在的」市值/本益比，對早期窗口存在前視偏誤，是
已知且刻意接受的限制（見專案規劃文件）。

實作（與 HDBSCAN_Cluster_SSD_DTW.py 相同的「只換分組方式」組合模式）：
  1. 借用 HDBSCAN_PCA_Loadings._build_feature_matrix() 取得報酬 PCA 因子
     載荷（價格行為特徵），不使用其 HDBSCAN 聚類。
  2. 合併基本面特徵（log 市值、盈餘殖利率、GICS one-hot），區塊分別標準化
     後依權重組合（避免單一 joint StandardScaler 讓 one-hot 欄位數量
     稀釋距離量測——這正是本專案先前已證偽的「特徵未加權原始拼接」失敗
     模式，見 archive/config_archived_strategies.py「HDBSCAN 舊特徵系」）。
  3. AgglomerativeClustering（average linkage、依合併距離分位數校準
     distance_threshold）分群；過小群併入 "Unknown"。
  4. 分群標籤當作 sector_mapping 餵給既有 ssd_rolling.Formation（完全複用
     其 min-SSD 排序 + Engle-Granger ADF + 半衰期 + Hurst 篩選流程）。
  5. 回填真實 GICS 產業、群組 ID、市值、本益比欄位供報表/後續分析使用。
"""

import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from strategies.formation.HDBSCAN_PCA_Loadings import Formation as _PriceFeatureFormation
from strategies.formation.ssd_rolling import Formation as _SSDFormation

# ── GICS 產業別名正規化（跨資料源標籤不一致問題） ─────────────────────────
_SECTOR_CANONICAL_MAP = {
    "Healthcare": "Health Care",
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
}

_CANONICAL_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]

_UNKNOWN_SECTOR_IDX = len(_CANONICAL_SECTORS)  # one-hot 最後一欄

# 基本面資料每個 process 只需載入一次（檔案小，重複載入成本可忽略但無必要）
_fundamentals_cache: dict[tuple, dict] = {}


def _canonicalize_sector(raw: str) -> str:
    if not raw:
        return "Unknown"
    mapped = _SECTOR_CANONICAL_MAP.get(raw, raw)
    return mapped if mapped in _CANONICAL_SECTORS else "Unknown"


def _load_fundamentals(db_path: str, table: str) -> dict:
    cache_key = (db_path, table)
    if cache_key in _fundamentals_cache:
        return _fundamentals_cache[cache_key]

    fundamentals: dict[str, dict] = {}
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(f"SELECT Symbol, MarketCap, TrailingPE FROM {table}").fetchall()
            for symbol, market_cap, trailing_pe in rows:
                fundamentals[str(symbol).upper()] = {"MarketCap": market_cap, "TrailingPE": trailing_pe}
        finally:
            conn.close()
    else:
        print(f"  [Formation] 警告：找不到基本面資料庫 {db_path}，全數標的將回退為全域中位數插補")

    _fundamentals_cache[cache_key] = fundamentals
    return fundamentals


def _impute_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """以群組（正規化後產業）中位數插補缺失值，群組本身無覆蓋則退回全域中位數"""
    values = values.copy()
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return values

    global_median = float(np.nanmedian(values)) if not np.all(nan_mask) else 0.0

    for g in np.unique(groups):
        g_mask = groups == g
        g_nan_mask = g_mask & nan_mask
        if not g_nan_mask.any():
            continue
        g_values = values[g_mask & ~nan_mask]
        fill = float(np.median(g_values)) if len(g_values) > 0 else global_median
        values[g_nan_mask] = fill

    values[np.isnan(values)] = global_median
    return values


def _winsorize(values: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, [lower_pct, upper_pct])
    return np.clip(values, lo, hi)


class Formation:
    """
    Agglomerative（價格 PCA 因子 ⊕ 公司基本面）分組 + SSD Rolling 排序 形成期模組
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,          # 真實 GICS，僅供報表/one-hot 特徵使用
        min_tickers_for_pairing: int = 2,
        # ── 特徵組 ──────────────────────────────────────────────
        pca_n_components: int = 5,
        fundamentals_db_path: str = "./dataset/fundamentals_sp500.db",
        fundamentals_table: str = "Fundamentals",
        price_feature_weight: float = 1.0,
        fundamentals_feature_weight: float = 1.0,
        sector_onehot_weight: float = 1.0,
        umap_random_state: int = 42,
        # ── Agglomerative 分群組 ──────────────────────────────────
        agg_linkage: str = "average",
        agg_threshold_percentile: float = 75.0,
        min_cluster_size: int = 5,
        # ── 排序/共整合組（轉傳 ssd_rolling） ──────────────────────
        adf_pvalue_threshold: float = 0.05,
        trading_window: int = 126,
        **kwargs,
    ):
        self.price_df = price_df
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.real_sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.pca_n_components = pca_n_components
        self.fundamentals_db_path = fundamentals_db_path
        self.fundamentals_table = fundamentals_table
        self.price_feature_weight = price_feature_weight
        self.fundamentals_feature_weight = fundamentals_feature_weight
        self.sector_onehot_weight = sector_onehot_weight
        self.umap_random_state = umap_random_state

        self.agg_linkage = agg_linkage
        self.agg_threshold_percentile = agg_threshold_percentile
        self.min_cluster_size = min_cluster_size

        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.trading_window = trading_window

        self._fundamentals: dict = {}
        self.cluster_labels_: dict = {}
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def _lookup_sector(self, ticker: str) -> str:
        return self.real_sector_mapping.get(ticker.upper(), self.real_sector_mapping.get(ticker, "Unknown"))

    def _lookup_fundamental(self, ticker: str, field: str):
        rec = self._fundamentals.get(ticker.upper(), self._fundamentals.get(ticker))
        return rec.get(field) if rec else None

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        price_former = _PriceFeatureFormation(
            price_df=self.price_df,
            form_start=self.form_start,
            form_end=self.form_end,
            top_n=self.top_n,
            reduce_method="none",
            pca_n_components=self.pca_n_components,
            umap_random_state=self.umap_random_state,
        )
        price_loadings, valid_tickers = price_former._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            return np.empty((0, 0)), []

        self._fundamentals = _load_fundamentals(self.fundamentals_db_path, self.fundamentals_table)

        canonical_sectors = []
        market_caps = np.full(len(valid_tickers), np.nan)
        earnings_yields = np.full(len(valid_tickers), np.nan)

        for i, ticker in enumerate(valid_tickers):
            canonical_sectors.append(_canonicalize_sector(self._lookup_sector(ticker)))

            market_cap = self._lookup_fundamental(ticker, "MarketCap")
            trailing_pe = self._lookup_fundamental(ticker, "TrailingPE")
            if market_cap is not None and market_cap > 0:
                market_caps[i] = np.log1p(market_cap)
            if trailing_pe is not None and trailing_pe != 0:
                earnings_yields[i] = 1.0 / trailing_pe

        canonical_sectors = np.array(canonical_sectors)
        n_missing_mc = int(np.isnan(market_caps).sum())
        n_missing_pe = int(np.isnan(earnings_yields).sum())

        market_caps = _winsorize(_impute_by_group(market_caps, canonical_sectors))
        earnings_yields = _winsorize(_impute_by_group(earnings_yields, canonical_sectors))

        onehot = np.zeros((len(valid_tickers), len(_CANONICAL_SECTORS) + 1))
        sector_index = {s: i for i, s in enumerate(_CANONICAL_SECTORS)}
        for i, sector in enumerate(canonical_sectors):
            onehot[i, sector_index.get(sector, _UNKNOWN_SECTOR_IDX)] = 1.0

        price_scaled = StandardScaler().fit_transform(price_loadings) * self.price_feature_weight
        fundamentals_cont = np.column_stack([market_caps, earnings_yields])
        fundamentals_scaled = StandardScaler().fit_transform(fundamentals_cont) * self.fundamentals_feature_weight
        sector_weighted = onehot * self.sector_onehot_weight

        X = np.hstack([price_scaled, fundamentals_scaled, sector_weighted])

        print(
            f"  [Formation] Agglomerative 特徵矩陣：{len(valid_tickers)} 檔 | "
            f"價格因子 {price_loadings.shape[1]} 維 + 基本面 2 維（市值缺失 "
            f"{n_missing_mc}、PE 缺失 {n_missing_pe} 已插補）+ 產業 one-hot {onehot.shape[1]} 維"
        )
        return X, valid_tickers

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n < 2:
            return np.zeros(n, dtype=int)

        probe = AgglomerativeClustering(n_clusters=1, linkage=self.agg_linkage, compute_distances=True).fit(X)
        distances = probe.distances_
        threshold = float(np.percentile(distances, self.agg_threshold_percentile)) if len(distances) > 0 else 0.0
        if threshold <= 0.0:
            threshold = float(np.max(distances)) if len(distances) > 0 else 1e-6
            if threshold <= 0.0:
                threshold = 1e-6

        final = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage=self.agg_linkage).fit(X)
        return final.labels_

    def run(self) -> pd.DataFrame:
        X, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        labels = self._cluster(X)
        self.cluster_labels_ = {t: int(l) for t, l in zip(valid_tickers, labels)}

        cluster_sizes = pd.Series(labels).value_counts()
        cluster_map = {
            t: (f"Cluster_{l}" if cluster_sizes[l] >= self.min_cluster_size else "Unknown")
            for t, l in zip(valid_tickers, labels)
        }
        n_clusters = len({v for v in cluster_map.values() if v != "Unknown"})
        n_unknown = sum(1 for v in cluster_map.values() if v == "Unknown")
        print(
            f"  [Formation] Agglomerative 分組：{n_clusters} 群 "
            f"（distance_threshold 分位數={self.agg_threshold_percentile}）| "
            f"過小群併入 Unknown {n_unknown}/{len(valid_tickers)} 檔（排除）"
        )

        ssd_form = _SSDFormation(
            price_df=self.price_df[valid_tickers],
            form_start=self.form_start,
            form_end=self.form_end,
            top_n=self.top_n,
            sector_mapping=cluster_map,
            min_tickers_for_pairing=self.min_tickers_for_pairing,
            adf_pvalue_threshold=self.adf_pvalue_threshold,
            trading_window=self.trading_window,
        )
        selected = ssd_form.run()
        if selected.empty:
            self.selected_pairs = selected
            return selected

        selected = selected.copy()
        selected["Sector_A"] = [self._lookup_sector(t) for t in selected["Ticker_A"]]
        selected["Sector_B"] = [self._lookup_sector(t) for t in selected["Ticker_B"]]
        selected["Cluster_ID_A"] = [cluster_map.get(t, "Unknown") for t in selected["Ticker_A"]]
        selected["Cluster_ID_B"] = [cluster_map.get(t, "Unknown") for t in selected["Ticker_B"]]
        selected["MarketCap_A"] = [self._lookup_fundamental(t, "MarketCap") for t in selected["Ticker_A"]]
        selected["MarketCap_B"] = [self._lookup_fundamental(t, "MarketCap") for t in selected["Ticker_B"]]
        selected["TrailingPE_A"] = [self._lookup_fundamental(t, "TrailingPE") for t in selected["Ticker_A"]]
        selected["TrailingPE_B"] = [self._lookup_fundamental(t, "TrailingPE") for t in selected["Ticker_B"]]

        self.selected_pairs = selected
        return selected
