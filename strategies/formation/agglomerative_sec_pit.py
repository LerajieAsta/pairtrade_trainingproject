"""
TODO（中性層，2026-07 重構）：本封存策略仍以舊方式 import HDBSCAN_PCA_Loadings
的內部方法取特徵。復活時應改用中性共用層：
  strategies.formation._features.build_return_pca_loadings
  strategies.formation._clustering.cluster_agglomerative
  strategies.formation._ranking.rank_within_groups
（現役 agglomerative_yF/FMP 已完成此遷移，見 tools/formation_regression.py 等價測試。）

Agglomerative + SEC PIT 基本面 ⊕ Beta 風險先驗 形成期模組（研究框架次步 #5）
======================================================================

命題延伸：在 Agglomerative 分群距離中加入「基本面風險先驗」，讓風險暴露相近
（系統性風險 β、規模、估值相近）的股票更容易被分到同群 → 群內配對更可能有
穩定的長期均衡（共整合）關係。

與 agglomerative_FMP 的差異（唯一新增）：
  agglomerative_FMP  特徵 = 價格 PCA loadings ⊕ log 市值(PIT) ⊕ 盈餘殖利率(PIT) ⊕ GICS one-hot
  本模組（SEC-PIT）  特徵 = 上述 ⊕ **Beta（系統性風險，形成期滾動估計）**

為何只加 Beta、不加槓桿/獲利率：
  SEC EDGAR XBRL 的基本面覆蓋率在本資料集極低——市值/PE 全期僅 ~10%、2009 年
  前近乎 0（XBRL 制度 2009 才普及）。若再從 XBRL 拉槓桿(D/E)、獲利率(ROE/margin)，
  同樣 0%（2009 前）+ 稀疏（之後 ~15%）→ 90%+ 需群組中位數插補 = 讓多數股票在
  該維度上相同 = 引入雜訊而非訊號。相對地 **Beta 由形成期報酬對市場因子回歸
  即得，全覆蓋且天生 point-in-time（僅用形成窗口內資料，無前視）**，是唯一
  高覆蓋、理論上最貼合「市場中性配對」的基本面風險特徵。市值/PE 沿用 PIT
  parquet（有資料處為真 PIT，無資料處以產業中位數插補）。

實作：直接子類 agglomerative_FMP.Formation，只覆寫 __init__（新增
beta_feature_weight）與 _build_feature_matrix（在原特徵後接上獨立加權的 Beta
區塊）。分群、SSD 排序、回填欄位全部沿用父類。

注意：run_formation 以 inspect.signature 過濾建構子參數，故 __init__ 具名列出
所有參數（beta_feature_weight 亦然），不可只靠 **kwargs。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from strategies.formation.agglomerative_FMP import (
    Formation as _FMPFormation,
    _load_fundamentals_parquet, _canonicalize_sector, _impute_by_group, _winsorize,
    _CANONICAL_SECTORS, _UNKNOWN_SECTOR_IDX,
)
from strategies.formation.HDBSCAN_PCA_Loadings import Formation as _PriceFeatureFormation


def _rolling_betas(price_df: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """
    形成期系統性風險 β：對每檔股票，以其對數報酬對「橫斷面平均報酬（市場因子）」
    做單因子回歸的斜率。point-in-time——僅用形成窗口內資料，無前視。
    回傳長度 = len(tickers) 的 β 陣列（缺資料/退化者為 NaN）。
    """
    R = np.diff(np.log(price_df[tickers].values), axis=0)     # (T-1 × N)
    if R.shape[0] < 10 or R.shape[1] < 2:
        return np.full(len(tickers), np.nan)
    f = R.mean(axis=1)                                        # 市場因子
    fc = f - f.mean()
    var_f = float(fc @ fc) + 1e-12
    Rc = R - R.mean(axis=0)
    betas = (Rc.T @ fc) / var_f                               # (N,)
    return np.asarray(betas, dtype=np.float64)


class Formation(_FMPFormation):
    """Agglomerative（價格 PCA ⊕ PIT 基本面 ⊕ Beta 風險先驗）分組 + SSD Rolling 排序。"""

    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        pca_n_components: int = 5,
        fundamentals_parquet_path: str = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet",
        price_feature_weight: float = 1.0,
        fundamentals_feature_weight: float = 1.0,
        sector_onehot_weight: float = 1.0,
        beta_feature_weight: float = 1.0,          # 新增：Beta 風險先驗權重
        umap_random_state: int = 42,
        agg_linkage: str = "average",
        agg_threshold_percentile: float = 75.0,
        min_cluster_size: int = 5,
        adf_pvalue_threshold: float = 0.05,
        trading_window: int = 126,
        **kwargs,
    ):
        super().__init__(
            price_df=price_df, form_start=form_start, form_end=form_end, top_n=top_n,
            sector_mapping=sector_mapping, min_tickers_for_pairing=min_tickers_for_pairing,
            pca_n_components=pca_n_components,
            fundamentals_parquet_path=fundamentals_parquet_path,
            price_feature_weight=price_feature_weight,
            fundamentals_feature_weight=fundamentals_feature_weight,
            sector_onehot_weight=sector_onehot_weight, umap_random_state=umap_random_state,
            agg_linkage=agg_linkage, agg_threshold_percentile=agg_threshold_percentile,
            min_cluster_size=min_cluster_size, adf_pvalue_threshold=adf_pvalue_threshold,
            trading_window=trading_window,
        )
        self.beta_feature_weight = beta_feature_weight

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        price_former = _PriceFeatureFormation(
            price_df=self.price_df, form_start=self.form_start, form_end=self.form_end,
            top_n=self.top_n, reduce_method="none",
            pca_n_components=self.pca_n_components, umap_random_state=self.umap_random_state,
        )
        price_loadings, valid_tickers = price_former._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            return np.empty((0, 0)), []

        pit_df = _load_fundamentals_parquet(self.fundamentals_parquet_path)
        target_date = pd.to_datetime(self.form_end)
        current_fundamentals = {}
        if not pit_df.empty:
            dates = pit_df.index.get_level_values("date").unique()
            valid_dates = dates[dates <= target_date]
            if len(valid_dates) > 0:
                current_fundamentals = pit_df.xs(valid_dates.max(), level="date").to_dict("index")

        canonical_sectors = []
        market_caps = np.full(len(valid_tickers), np.nan)
        earnings_yields = np.full(len(valid_tickers), np.nan)
        for i, ticker in enumerate(valid_tickers):
            rec = current_fundamentals.get(ticker.upper(), {})
            raw_sector = rec.get("industry") if pd.notna(rec.get("industry")) else self._lookup_sector(ticker)
            canonical_sectors.append(_canonicalize_sector(raw_sector))
            mc, pe = rec.get("market_cap"), rec.get("pe_ratio")
            if pd.notna(mc) and mc > 0:
                market_caps[i] = np.log1p(mc)
            if pd.notna(pe) and pe != 0:
                earnings_yields[i] = 1.0 / pe

        canonical_sectors = np.array(canonical_sectors)
        n_missing_mc = int(np.isnan(market_caps).sum())
        n_missing_pe = int(np.isnan(earnings_yields).sum())

        # ── 新增：Beta 風險先驗（PIT，全覆蓋） ──────────────────────
        betas = _rolling_betas(self.price_df, valid_tickers)
        n_missing_beta = int(np.isnan(betas).sum())
        betas = _winsorize(_impute_by_group(betas, canonical_sectors))

        market_caps = _winsorize(_impute_by_group(market_caps, canonical_sectors))
        earnings_yields = _winsorize(_impute_by_group(earnings_yields, canonical_sectors))

        onehot = np.zeros((len(valid_tickers), len(_CANONICAL_SECTORS) + 1))
        sector_index = {s: i for i, s in enumerate(_CANONICAL_SECTORS)}
        for i, sector in enumerate(canonical_sectors):
            onehot[i, sector_index.get(sector, _UNKNOWN_SECTOR_IDX)] = 1.0

        price_scaled = StandardScaler().fit_transform(price_loadings) * self.price_feature_weight
        fundamentals_cont = np.column_stack([market_caps, earnings_yields])
        fundamentals_scaled = StandardScaler().fit_transform(fundamentals_cont) * self.fundamentals_feature_weight
        beta_scaled = StandardScaler().fit_transform(betas.reshape(-1, 1)) * self.beta_feature_weight
        sector_weighted = onehot * self.sector_onehot_weight

        X = np.hstack([price_scaled, fundamentals_scaled, beta_scaled, sector_weighted])

        print(
            f"  [Formation] Agglomerative(SEC-PIT+β) 特徵：{len(valid_tickers)} 檔 | "
            f"價格 {price_loadings.shape[1]} 維 + 基本面 2 維（市值缺 {n_missing_mc}、PE 缺 {n_missing_pe}）"
            f" + Beta 1 維（缺 {n_missing_beta}）+ 產業 one-hot {onehot.shape[1]} 維"
        )
        return X, valid_tickers
