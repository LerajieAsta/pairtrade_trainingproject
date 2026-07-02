"""
HDBSCAN Cluster + SSD-DTW-PCA 分組消融實驗模組
======================================================================

命題：「HDBSCAN 資料驅動聚類」優於「GICS 靜態產業分類」作為配對搜尋空間。

實驗設計（控制變因）：
  對照組：SSD-DTW-PCA (Paper-Fixed) — GICS 產業分組 + SSD-DTW-PCA 排序 + 路徑 B 交易
  實驗組：本模組                     — HDBSCAN  聚類分組 + SSD-DTW-PCA 排序 + 路徑 B 交易
  唯一差異 = 分組方式。其餘（群內共整合篩選、距離排序、spread 定義）完全相同。

實作：組合（composition）兩個既有模組——
  1. HDBSCAN_PCA_Loadings.Formation → 報酬 PCA 因子載荷 + HDBSCAN 聚類，產生 cluster labels
  2. DTW_Cointegration_Paper.Formation → 把 cluster labels 當 sector_mapping 傳入，
     完整沿用其標準化空間 OLS/ADF 篩選、SSD/DTW 距離計算與 PCA 融合排序

  HDBSCAN 噪音點（label = -1）映射為 "Unknown"，DTW 流程會自動跳過——
  等同於把聚類的離群過濾能力也帶進來。

搭配 config：交易端使用 ignore_ols_alpha=True（路徑 B，標準化空間），
與 SSD-DTW-PCA (Paper-Fixed) 交易端完全一致。
"""

import pandas as pd

from strategies.formation.HDBSCAN_PCA_Loadings import Formation as _ClusterFormation
from strategies.formation.DTW_Cointegration_Paper import Formation as _DTWFormation


class Formation:
    """
    HDBSCAN 聚類分組 + SSD-DTW-PCA 排序 形成期模組。

    參數分為兩組：
      聚類組（轉傳 HDBSCAN_PCA_Loadings）：pca_n_components、hdbscan_*、umap_random_state
      排序組（轉傳 DTW_Cointegration_Paper）：method、adf_pvalue_threshold、dtw_window、trading_window
    """

    def __init__(
        self,
        price_df:    pd.DataFrame,
        form_start:  str,
        form_end:    str,
        top_n:                    int   = 20,
        sector_mapping:           dict  = None,   # 僅供結果記錄；分組由 HDBSCAN 決定
        min_tickers_for_pairing:  int   = 2,
        # ── 聚類組 ──────────────────────────────────────────────
        pca_n_components:         int   = 15,
        hdbscan_min_cluster_size: int   = 5,
        hdbscan_min_samples:      int   = 2,
        hdbscan_metric:           str   = "euclidean",
        umap_random_state:        int   = 42,
        # ── 排序組（與 DTW_Cointegration_Paper 一致） ────────────
        method:                   str   = "ssd_dtw_pca",
        adf_pvalue_threshold:     float = 0.01,
        dtw_window:               int   = 15,
        trading_window:           int   = 126,
        **kwargs,
    ):
        self.price_df   = price_df
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.real_sector_mapping     = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        # 聚類器：只用它的特徵萃取 + HDBSCAN，不用它的配對篩選
        self._cluster = _ClusterFormation(
            price_df   = price_df,
            form_start = form_start,
            form_end   = form_end,
            top_n      = top_n,
            reduce_method            = "none",
            pca_n_components         = pca_n_components,
            hdbscan_min_cluster_size = hdbscan_min_cluster_size,
            hdbscan_min_samples      = hdbscan_min_samples,
            hdbscan_metric           = hdbscan_metric,
            umap_random_state        = umap_random_state,
        )

        self.method               = method
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.dtw_window           = dtw_window
        self.trading_window       = trading_window

        self.cluster_labels_: dict         = {}
        self.selected_pairs:  pd.DataFrame = pd.DataFrame()

    def run(self) -> pd.DataFrame:
        # 1. HDBSCAN 聚類（報酬 PCA 因子載荷空間）
        X, valid_tickers = self._cluster._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        labels = self._cluster._hdbscan_cluster(X)
        self.cluster_labels_ = {t: int(l) for t, l in zip(valid_tickers, labels)}

        # 2. 聚類標籤 → 分組 map（噪音 = "Unknown"，DTW 流程自動跳過）
        cluster_map = {
            t: (f"Cluster_{l}" if l != -1 else "Unknown")
            for t, l in self.cluster_labels_.items()
        }
        n_clusters = len(set(l for l in labels if l != -1))
        n_noise    = int((labels == -1).sum())
        print(f"  [Formation] HDBSCAN 分組：{n_clusters} 群 | 噪音 {n_noise}/{len(valid_tickers)} 檔（排除）")

        # 3. 沿用 DTW Paper 流程：群內共整合篩選 + SSD/DTW 距離 + PCA 融合排序
        dtw_form = _DTWFormation(
            price_df                = self.price_df[valid_tickers],
            form_start              = self.form_start,
            form_end                = self.form_end,
            top_n                   = self.top_n,
            sector_mapping          = cluster_map,
            min_tickers_for_pairing = self.min_tickers_for_pairing,
            dtw_window              = self.dtw_window,
            method                  = self.method,
            adf_pvalue_threshold    = self.adf_pvalue_threshold,
            trading_window          = self.trading_window,
        )
        selected = dtw_form.run()
        if selected.empty:
            self.selected_pairs = selected
            return selected

        # 4. 補回真實產業欄位（供 MSR 產業分散與結果分析；Sector 保留群集標籤）
        selected = selected.copy()
        selected["Sector_A"] = [
            self.real_sector_mapping.get(t.upper(), self.real_sector_mapping.get(t, "Unknown"))
            for t in selected["Ticker_A"]
        ]
        selected["Sector_B"] = [
            self.real_sector_mapping.get(t.upper(), self.real_sector_mapping.get(t, "Unknown"))
            for t in selected["Ticker_B"]
        ]

        self.selected_pairs = selected
        return selected
