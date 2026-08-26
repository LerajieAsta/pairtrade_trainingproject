"""
通用分群配對形成期組裝器（中性命名，與具體演算法無關）
======================================================================
把「特徵 → 分群 → 群內排序」三個中性層組裝成參數驅動的形成期策略：

    Formation(
        feature_mode   = "fundamentals_mix" | "price_pca",
        cluster_method = "hdbscan" | "agglomerative" | "kmeans" | "gics" | "none",
        ranking_backend= "ssd" | "dtw" | "ssd_dtw_pca",
        ...,
    ).run()

用途：以宣告式參數展開「分群 × 排序」消融矩陣（如 3×3），無需為每個組合
手寫一個策略模組。新增分群法 → _clustering.py 加 backend；新增排序準則 →
_ranking.py 加 backend；新增組合 → config 一行。

K-means 特例：K-means 需預先指定群數。本組裝器以「同期 Agglomerative 的群數」
作為 k（資料驅動、與其他分群量級可比），先跑一次 Agglomerative 取群數再餵給
K-means。
"""
import os

import numpy as np
import pandas as pd

from strategies.formation._features import (
    build_return_pca_loadings, build_fundamentals_mix_features,
    build_momentum_features,
)
from strategies.formation._clustering import cluster, cluster_agglomerative
from strategies.formation._ranking import rank_within_groups


class Formation:
    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        # ── 組裝選擇（形成期三維度：分組 × 排序 × 篩選）───────────────
        feature_mode: str = "fundamentals_mix",
        # "gics"：直接用真實產業分組，不跑分群
        # "none"：完全不分組（全市場單一組），分組維度的零點對照
        cluster_method: str = "agglomerative",
        ranking_backend: str = "ssd",
        # ranking_backend="reversal" 專用（Han, He & Toh 2021 的選對準則）。
        # 預設值即原文設定：過去 1 個月（21 交易日）、1 倍橫斷面 SD。
        # sd_scope 對應原文未明言的歧義（"cross-sectional" 是群內或全體）。
        reversal_lookback: int = 21,
        reversal_sd_mult: float = 1.0,
        reversal_sd_scope: str = "pooled",
        # "coint"（三道）| "adf_only"（僅 ADF，復現許鈞翔 2025）| "none"（零道，純排序消融）
        filter_mode: str = "coint",
        # ── 特徵組 ─────────────────────────────────────────────────
        pca_n_components: int = 5,
        factor_residual: bool = False,
        fundamentals_parquet_path: str = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
        price_feature_weight: float = 1.0,
        fundamentals_feature_weight: float = 1.0,
        sector_onehot_weight: float = 1.0,
        umap_random_state: int = 42,
        # ── 動量特徵組（feature_mode="momentum" / "momentum_mix"）──────
        momentum_horizons: tuple = (1, 3, 6, 9, 12),
        momentum_include_vol: bool = True,
        momentum_block_pca: int = 0,     # >0：動量區塊先 PCA 降維再拼接
        momentum_weight: float = 1.0,
        # ── 結構性財報特徵（SEC XBRL PIT；空 tuple = 停用，維持現行行為）──
        structural_features: tuple = (),
        structural_weight: float = 1.0,
        # 缺失值插補範圍："group"（產業中位數，預設）| "global"（全域中位數）。
        # 命題 1 的特徵擴充實驗須用 "global"，避免產業資訊經插補漏進特徵。
        impute_scope: str = "group",
        # ── 分群組 ─────────────────────────────────────────────────
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int = 2,
        hdbscan_metric: str = "euclidean",
        agg_linkage: str = "average",
        agg_threshold_percentile: float = 75.0,
        min_cluster_size: int = 5,
        # ── 排序組 ─────────────────────────────────────────────────
        adf_pvalue_threshold: float = 0.05,
        trading_window: int = 126,
        dtw_window: int = 15,
        # 正規化口徑："zscore_log"（預設，現行全部策略）| "ggr_index"（GGR 復現）
        normalize_mode: str = "zscore_log",
        **kwargs,
    ):
        self.price_df = price_df
        self.normalize_mode = normalize_mode
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.real_sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.feature_mode = feature_mode
        self.cluster_method = cluster_method
        self.ranking_backend = ranking_backend
        self.reversal_lookback = reversal_lookback
        self.reversal_sd_mult = reversal_sd_mult
        self.reversal_sd_scope = reversal_sd_scope
        self.filter_mode = filter_mode

        self.pca_n_components = pca_n_components
        self.factor_residual = factor_residual
        self.fundamentals_parquet_path = fundamentals_parquet_path
        self.price_feature_weight = price_feature_weight
        self.fundamentals_feature_weight = fundamentals_feature_weight
        self.sector_onehot_weight = sector_onehot_weight
        self.umap_random_state = umap_random_state
        self.momentum_horizons = tuple(momentum_horizons)
        self.momentum_include_vol = momentum_include_vol
        self.momentum_block_pca = int(momentum_block_pca or 0)
        self.momentum_weight = float(momentum_weight)
        self.structural_features = tuple(structural_features)
        self.structural_weight = float(structural_weight)
        self.impute_scope = impute_scope

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric
        self.agg_linkage = agg_linkage
        self.agg_threshold_percentile = agg_threshold_percentile
        self.min_cluster_size = min_cluster_size

        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.trading_window = trading_window
        self.dtw_window = dtw_window

        # {ticker: 群標籤}。ML 分群時為原始整數標籤（−1 = HDBSCAN 雜訊）；
        # cluster_method="gics" 時為產業名、"none" 時為 "All_Market"——
        # 三者都記錄，稽核時分組層的每一個臂才對稱可查。
        self.cluster_labels_: dict = {}
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

        # 分階段存檔（指導教授建議：分組／排序／篩選各自可稽核）。
        # 環境變數 FORMATION_TRACE=1 開啟；預設關閉以維持既有行為與資料量。
        # 開啟後 run_formation 會把 cluster_labels_ 與 stage_trace_ 落地成
        # formation_groups / formation_ranked 兩張表。
        self.collect_trace = os.environ.get("FORMATION_TRACE", "").strip() in ("1", "true", "yes")
        self.stage_trace_: list | None = None

    def _lookup_sector(self, ticker: str) -> str:
        return self.real_sector_mapping.get(
            ticker.upper(), self.real_sector_mapping.get(ticker, "Unknown"))

    # ── 特徵 ───────────────────────────────────────────────────────
    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        if self.feature_mode in ("momentum", "momentum_mix"):
            # 多尺度動量（+ 波動/可交易性）；"momentum_mix" 再併上混合特徵區塊
            mom_X, mom_tickers = build_momentum_features(
                price_df=self.price_df,
                horizons_months=self.momentum_horizons,
                include_volatility=self.momentum_include_vol,
                block_pca_components=self.momentum_block_pca,
                random_state=self.umap_random_state,
                min_tickers=self.min_tickers_for_pairing,
            )
            if self.feature_mode == "momentum" or len(mom_tickers) < self.min_tickers_for_pairing:
                return mom_X * self.momentum_weight, mom_tickers
            base_X, base_tickers = build_fundamentals_mix_features(
                price_df=self.price_df, form_end=self.form_end,
                sector_mapping=self.real_sector_mapping,
                pca_n_components=self.pca_n_components,
                factor_residual=self.factor_residual,
                fundamentals_parquet_path=self.fundamentals_parquet_path,
                price_feature_weight=self.price_feature_weight,
                fundamentals_feature_weight=self.fundamentals_feature_weight,
                sector_onehot_weight=self.sector_onehot_weight,
                structural_features=self.structural_features,
                structural_weight=self.structural_weight,
                impute_scope=self.impute_scope,
                random_state=self.umap_random_state,
                min_tickers=self.min_tickers_for_pairing,
            )
            # 兩區塊的有效股票集可能不同 → 取交集後對齊順序
            common = [t for t in base_tickers if t in set(mom_tickers)]
            if len(common) < self.min_tickers_for_pairing:
                return np.empty((0, 0)), []
            bi = {t: i for i, t in enumerate(base_tickers)}
            mi = {t: i for i, t in enumerate(mom_tickers)}
            X = np.hstack([base_X[[bi[t] for t in common]],
                           mom_X[[mi[t] for t in common]] * self.momentum_weight])
            return X, common

        if self.feature_mode == "price_pca":
            return build_return_pca_loadings(
                price_df=self.price_df,
                pca_n_components=self.pca_n_components,
                factor_residual=self.factor_residual,
                sector_mapping=self.real_sector_mapping,
                random_state=self.umap_random_state,
            )
        # 預設：混合特徵（報酬 PCA ⊕ 基本面 ⊕ 產業）
        return build_fundamentals_mix_features(
            price_df=self.price_df,
            form_end=self.form_end,
            sector_mapping=self.real_sector_mapping,
            pca_n_components=self.pca_n_components,
            factor_residual=self.factor_residual,
            fundamentals_parquet_path=self.fundamentals_parquet_path,
            price_feature_weight=self.price_feature_weight,
            fundamentals_feature_weight=self.fundamentals_feature_weight,
            sector_onehot_weight=self.sector_onehot_weight,
            structural_features=self.structural_features,
            structural_weight=self.structural_weight,
            impute_scope=self.impute_scope,
            random_state=self.umap_random_state,
            min_tickers=self.min_tickers_for_pairing,
        )

    # ── 分群（含 K-means 群數對齊 Agglomerative）────────────────────
    def _cluster(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_method == "kmeans":
            agg_labels = cluster_agglomerative(
                X, linkage=self.agg_linkage,
                threshold_percentile=self.agg_threshold_percentile)
            n_k = len(set(int(l) for l in agg_labels))
            print(f"  [Formation] K-means 群數對齊 Agglomerative：k={n_k}")
            return cluster("kmeans", X, n_clusters=n_k,
                           random_state=self.umap_random_state)
        if self.cluster_method == "hdbscan":
            return cluster("hdbscan", X,
                           min_cluster_size=self.hdbscan_min_cluster_size,
                           min_samples=self.hdbscan_min_samples,
                           metric=self.hdbscan_metric)
        return cluster("agglomerative", X,
                       linkage=self.agg_linkage,
                       threshold_percentile=self.agg_threshold_percentile)

    # ── 主流程 ─────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        # ── 分組：不分組 / GICS 產業（皆不跑分群）或資料驅動分群 ──────
        if self.cluster_method == "none":
            # 全市場單一組——消融矩陣「分組」維度的零點。
            # 分組層在本管線中只做一件事：限制候選池；真正選配對的是排序層與
            # 篩選層。故「不分組」是評價分組價值的必要對照（無此格時，比較的
            # 只是「不同的限制方式」，而非「限制 vs 不限制」）。
            # 註：本模式不建特徵矩陣，故 sector_onehot_weight 等特徵參數無作用。
            # 計算量警告：候選對數為 N(N−1)/2（600 檔 ≈ 18 萬對），遠高於群內比較。
            valid_tickers = [t for t in self.price_df.columns
                             if self.price_df[t].notna().sum() >= 30]
            if len(valid_tickers) < self.min_tickers_for_pairing:
                self.selected_pairs = pd.DataFrame()
                return self.selected_pairs
            cluster_map = {t: "All_Market" for t in valid_tickers}
            # 分組層的零點也要留紀錄：稽核時「對照組沒有分組紀錄」會讓
            # 命題 1 的核心比較（ML 分群 vs GICS vs 不分組）少一邊可查。
            self.cluster_labels_ = dict(cluster_map)
            print(f"  [Formation] 不分組（全市場單一組）：{len(valid_tickers)} 檔 | "
                  f"候選對數 {len(valid_tickers) * (len(valid_tickers) - 1) // 2:,}")
        elif self.cluster_method == "gics":
            valid_tickers = [t for t in self.price_df.columns
                             if self.price_df[t].notna().sum() >= 30]
            if len(valid_tickers) < self.min_tickers_for_pairing:
                self.selected_pairs = pd.DataFrame()
                return self.selected_pairs
            cluster_map = {t: self._lookup_sector(t) for t in valid_tickers}
            self.cluster_labels_ = dict(cluster_map)
            n_groups = len({v for v in cluster_map.values() if v != "Unknown"})
            n_unknown = sum(1 for v in cluster_map.values() if v == "Unknown")
            print(f"  [Formation] GICS 產業分組：{n_groups} 產業 | "
                  f"Unknown {n_unknown}/{len(valid_tickers)} 檔（排除）")
        else:
            X, valid_tickers = self._build_feature_matrix()
            if len(valid_tickers) < self.min_tickers_for_pairing:
                self.selected_pairs = pd.DataFrame()
                return self.selected_pairs

            labels = self._cluster(X)
            self.cluster_labels_ = {t: int(l) for t, l in zip(valid_tickers, labels)}

            # 群標籤 → 分組 map（過小群 / 噪音 併入 Unknown，排序端自動跳過）
            #
            # label = -1 是 HDBSCAN 的**噪音標記**（密度不足、無法歸入任何群），
            # 必須無條件排除。舊實作只看群大小，而噪音點每期中位有 90 檔
            # （佔 21.9%），遠超過 min_cluster_size=5，於是被映射成 "Cluster_-1"
            # 當作一個合法群——92.2% 的期數裡它還是**最大**的一群，光這一桶就
            # 貢獻 C(90,2)=4,005 組候選（該臂全部候選的 46%）。噪音桶依定義是
            # 異質混合物，這使 HDBSCAN 臂有近半在做的事其實是「不分組」。
            cluster_sizes = pd.Series(labels).value_counts()
            cluster_map = {
                t: (f"Cluster_{l}"
                    if int(l) != -1 and cluster_sizes[l] >= self.min_cluster_size
                    else "Unknown")
                for t, l in zip(valid_tickers, labels)
            }
            n_clusters = len({v for v in cluster_map.values() if v != "Unknown"})
            n_unknown = sum(1 for v in cluster_map.values() if v == "Unknown")
            n_noise = sum(1 for l in labels if int(l) == -1)
            print(
                f"  [Formation] {self.cluster_method} 分組：{n_clusters} 群 | "
                f"過小群/噪音併入 Unknown {n_unknown}/{len(valid_tickers)} 檔（排除，"
                f"其中噪音 {n_noise} 檔）"
            )

        # 分階段稽核：排序層與篩選層的逐候選軌跡（cluster_labels_ 已在上方備妥）。
        # 統計量本來就會算，這裡只是不再丟棄；trace=None 時行為與既往完全相同。
        self.stage_trace_ = [] if self.collect_trace else None

        # 群內共整合篩選 + 距離排序（經中性排序層）
        selected = rank_within_groups(
            trace=self.stage_trace_,
            adf_only=(self.filter_mode == "adf_only"),
            method=self.ranking_backend,
            price_df=self.price_df[valid_tickers],
            form_start=self.form_start,
            form_end=self.form_end,
            group_map=cluster_map,
            top_n=self.top_n,
            min_tickers_for_pairing=self.min_tickers_for_pairing,
            adf_pvalue_threshold=self.adf_pvalue_threshold,
            trading_window=self.trading_window,
            dtw_window=self.dtw_window,
            enable_filters=(self.filter_mode != "none"),
            normalize_mode=self.normalize_mode,
            lookback=self.reversal_lookback,
            sd_mult=self.reversal_sd_mult,
            sd_scope=self.reversal_sd_scope,
        )
        if selected.empty:
            self.selected_pairs = selected
            return selected

        # 補回真實 GICS 產業（供 MSR 產業分散與結果分析；Sector 保留群標籤）
        selected = selected.copy()
        selected["Sector_A"] = [self._lookup_sector(t) for t in selected["Ticker_A"]]
        selected["Sector_B"] = [self._lookup_sector(t) for t in selected["Ticker_B"]]
        selected["Cluster_ID_A"] = [cluster_map.get(t, "Unknown") for t in selected["Ticker_A"]]
        selected["Cluster_ID_B"] = [cluster_map.get(t, "Unknown") for t in selected["Ticker_B"]]

        self.selected_pairs = selected
        return selected
