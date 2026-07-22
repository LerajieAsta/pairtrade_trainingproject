"""
HDBSCAN PCA-Loadings 全市場/跨產業聚類回測系統（PCA 因子載荷特徵版）
======================================================================

與 HDBSCAN_UMAP 的唯一差異：第一層聚類特徵。

  HDBSCAN_UMAP        → 10 維單股統計特徵（波動率、偏態、峰態、Hurst、Beta、價位）
  HDBSCAN_PCA_Loadings → 標準化日報酬的 PCA 因子載荷（factor loadings）

動機（消融實驗）：
  統計特徵描述「單股行為像不像」，但與「兩檔股票價格路徑能否共整合」
  沒有因果關聯（log_price_level 尤其會把價位數字相近的股票聚在一起）。
  文獻標準做法是以報酬率的共同因子暴露作為聚類座標：
    - Avellaneda & Lee (2010)：報酬相關矩陣的特徵向量（eigenportfolios）
    - Sarmento & Horta (2020)：PCA 降維後的報酬表徵 + OPTICS/DBSCAN 分群
  暴露在相同風險因子的股票，才有經濟理由維持長期均衡關係——
  這正是共整合配對的來源。

實作：
  1. 形成期日報酬矩陣 R (T-1 × N)，逐股標準化（等同對相關矩陣做 PCA）
  2. PCA 取前 k 個主成分（k = pca_n_components，預設 15）
  3. 股票 i 的特徵向量 = loadings[i] = components_[:, i] × sqrt(explained_variance_)
     （以 sqrt 特徵值加權，保留因子重要性排序；不再套 RobustScaler，
       避免破壞因子間的相對尺度）
  4. 後續（reduce_method → HDBSCAN → 群內共整合 → Quality Score）完全沿用父類

reduce_method 額外支援 "none"：
  loadings 本身已是低維線性表徵，可直接餵給 HDBSCAN，
  跳過 UMAP 以消除滾動窗口間嵌入不穩定的問題。
"""

import numpy as np

from sklearn.decomposition import PCA

from strategies.formation.HDBSCAN_UMAP import Formation as _UMAPFormation


class Formation(_UMAPFormation):
    """
    HDBSCAN + PCA 因子載荷 配對交易形成期模組

    第一層：標準化報酬 PCA loadings（k 維）→（可選降維）→ HDBSCAN 聚類
    第二層：配對層特徵（10 維）→ 品質評分排序（沿用 HDBSCAN_UMAP）

    建構子簽章與 HDBSCAN_UMAP.Formation 完全相同；
    `pca_n_components` 在本模組中解讀為「報酬 PCA 因子數」。
    """

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        # 特徵萃取邏輯已抽至中性共用層 strategies.formation._features
        # （報酬 PCA 因子載荷 + 因子殘差化），供任何分群策略共用。
        from strategies.formation._features import build_return_pca_loadings
        return build_return_pca_loadings(
            price_df         = self.price_df,
            pca_n_components  = self.pca_n_components,
            factor_residual   = getattr(self, "factor_residual", False),
            sector_mapping    = self.sector_mapping,
            random_state      = self.umap_random_state,
        )

    def _umap_reduce(self, X: np.ndarray) -> np.ndarray:
        # 額外支援 reduce_method="none"：loadings 直接餵 HDBSCAN
        if self.reduce_method == "none":
            return X
        return super()._umap_reduce(X)
