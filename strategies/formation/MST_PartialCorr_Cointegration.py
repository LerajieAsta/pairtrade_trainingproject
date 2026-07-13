"""
MST / 偏相關圖 候選生成器 + 共整合篩選 + SSD-DTW 排序（研究框架次步 #4）
======================================================================

動機（延續 A 段實測結論）：
  A 段消融證明——研究框架的價值集中在「好的候選生成 × 強距離排序器（SSD-DTW）」，
  而非「弱的 quality-score 排序」。本模組把「候選生成」這一環從
  「HDBSCAN 聚類 / GICS 產業」換成「偏相關網路圖」，其餘（共整合篩選 + SSD-DTW
  排序 + 路徑 B 交易）與勝出變體完全相同 → 對命題 1 的公平延伸實驗。

為何用「偏相關」而非普通相關：
  普通相關會把「A、B 只因同時與 C（市場/龍頭）相關」的間接連結也算進來。
  偏相關（精確矩陣的標準化負值）在控制其餘所有股票後，只保留 A–B 的「直接」共動，
  這正是共整合配對的經濟來源（Kenett et al. 2010；Mantegna 1999 的 MST 精神）。
  形成期 T≈252 < N≈350（樣本共變異退化），故以 Ledoit-Wolf 收縮估計精確矩陣。

候選生成（graph_method）：
  "mst"     — 偏相關距離的最小生成樹（N-1 條骨幹邊，最稀疏）。
  "knn"     — 每檔股票連到偏相關最強的 knn_k 個鄰居（union）。
  "mst+knn" — 兩者聯集（預設；MST 保證連通骨幹 + kNN 補強局部強連結）。
  之後僅對「候選邊」做共整合檢定——把 C(N,2)≈6 萬對降到 ~3N≈千級，
  且聚焦於統計上最強的直接連結。

排序：SSD（Z-Score 標準化對數價格距離）或 SSD+DTW 的 PCA 融合 PC1（與勝出路徑一致）。

注意：run_formation 以 inspect.signature 過濾建構子參數，故所有參數須具名列出
（**kwargs 內的具名參數才會保留），不可只靠 **kwargs 吸收。
"""

import numpy as np
import pandas as pd

from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.covariance import LedoitWolf

from strategies.formation._utils import (
    _ols, _adf_stat, _compute_hurst, _cost_viable, _bh_fdr_threshold,
    _residualize_returns,
)
from strategies.formation.HDBSCAN_UMAP import _compute_halflife
from strategies.formation.DTW_Cointegration_Paper import _sakoe_chiba_dtw


class Formation:
    """偏相關圖候選生成 + 共整合篩選 + SSD-DTW 排序 形成期模組。"""

    def __init__(
        self,
        price_df:    pd.DataFrame,
        form_start:  str,
        form_end:    str,
        top_n:                    int   = 20,
        sector_mapping:           dict  = None,
        min_tickers_for_pairing:  int   = 2,
        # ── 候選生成（偏相關圖） ──────────────────────────────
        graph_method:             str   = "mst+knn",   # mst | knn | mst+knn
        knn_k:                    int   = 5,
        partial_corr:             bool  = True,         # True=偏相關；False=普通相關
        pcorr_threshold:          float = 0.0,          # 邊的 |(偏)相關| 下限
        factor_residual:          bool  = True,         # 研究框架 #1：先移市場+產業因子
        # ── 共整合篩選 gate（與 HDBSCAN_UMAP 對齊） ───────────
        adf_max_lags:             int   = 1,
        adf_pvalue_threshold:     float = 0.01,
        min_corr:                 float = 0.50,
        min_zero_crossings:       int   = 5,
        hurst_threshold:          float = 0.5,
        halflife_min:             float = 1.0,
        halflife_max:             float = 60.0,
        # ── 研究框架 #2 / #3 ────────────────────────────────
        use_fdr:                  bool  = False,
        fdr_alpha:                float = 0.05,
        use_cost_filter:          bool  = False,
        roundtrip_cost:           float = 0.0058,
        cost_margin:              float = 1.0,
        # ── 排序 ────────────────────────────────────────────
        method:                   str   = "ssd_dtw_pca",  # ssd | dtw | ssd_dtw_pca
        dtw_window:               int   = 15,
        max_sector_ratio:         float = 0.0,
        umap_random_state:        int   = 42,
        **kwargs,
    ):
        self.price_df   = price_df
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping          = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.graph_method    = graph_method
        self.knn_k           = knn_k
        self.partial_corr    = partial_corr
        self.pcorr_threshold = pcorr_threshold
        self.factor_residual = factor_residual

        self.adf_max_lags         = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.min_corr             = min_corr
        self.min_zero_crossings   = min_zero_crossings
        self.hurst_threshold      = hurst_threshold
        self.halflife_min         = halflife_min
        self.halflife_max         = halflife_max

        self.use_fdr         = use_fdr
        self.fdr_alpha       = fdr_alpha
        self.use_cost_filter = use_cost_filter
        self.roundtrip_cost  = roundtrip_cost
        self.cost_margin     = cost_margin

        self.method            = method
        self.dtw_window        = dtw_window
        self.max_sector_ratio  = max_sector_ratio
        self.umap_random_state = umap_random_state

        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    # ── 候選生成 ──────────────────────────────────────────────
    def _partial_corr_matrix(self, Rs: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf 收縮精確矩陣 → 標準化為偏相關矩陣（對角=1）。"""
        if not self.partial_corr:
            C = np.corrcoef(Rs, rowvar=False)
            return np.nan_to_num(C, nan=0.0)
        lw = LedoitWolf(assume_centered=False).fit(Rs)
        prec = lw.precision_
        d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
        pc = -prec / np.outer(d, d)
        np.fill_diagonal(pc, 1.0)
        return np.nan_to_num(pc, nan=0.0)

    def _candidate_edges(self, P: np.ndarray) -> set:
        """由偏相關矩陣 P 產生候選邊集合（無向，i<j）。距離 = 1 - |P|。"""
        n = P.shape[0]
        W = np.abs(P).copy()
        np.fill_diagonal(W, 0.0)
        D = 1.0 - W                              # 越強（|corr|→1）距離越小
        edges: set = set()

        if self.graph_method in ("mst", "mst+knn"):
            mst = minimum_spanning_tree(D).tocoo()
            for i, j in zip(mst.row, mst.col):
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if W[a, b] >= self.pcorr_threshold:
                    edges.add((a, b))

        if self.graph_method in ("knn", "mst+knn"):
            k = min(self.knn_k, n - 1)
            for i in range(n):
                nbrs = np.argsort(-W[i])          # |corr| 由大到小
                added = 0
                for j in nbrs:
                    j = int(j)
                    if j == i:
                        continue
                    if W[i, j] < self.pcorr_threshold:
                        break
                    a, b = (i, j) if i < j else (j, i)
                    edges.add((a, b))
                    added += 1
                    if added >= k:
                        break
        return edges

    # ── 單對共整合篩選（核心 gate；回傳 record 或 None） ──────────
    def _screen_edge(self, ta, tb, log_prices, norm_df, n_counter):
        log_a = log_prices[ta].values
        log_b = log_prices[tb].values

        corr = float(np.corrcoef(log_a, log_b)[0, 1])
        if not np.isfinite(corr) or corr < self.min_corr:
            return None, False

        al_ab, be_ab, re_ab = _ols(log_a, log_b)
        stat_ab, pval_ab    = _adf_stat(re_ab, self.adf_max_lags)
        al_ba, be_ba, re_ba = _ols(log_b, log_a)
        stat_ba, pval_ba    = _adf_stat(re_ba, self.adf_max_lags)
        n_counter[0] += 1                        # BH-FDR 的 m：做過 ADF 的邊數

        _adf_gate = max(self.adf_pvalue_threshold, 0.10) if self.use_fdr else self.adf_pvalue_threshold
        if min(pval_ab, pval_ba) >= _adf_gate:
            return None, True

        if pval_ab <= pval_ba:
            best_stat, best_pval = stat_ab, pval_ab
            best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
            best_a, best_b = ta, tb
        else:
            best_stat, best_pval = stat_ba, pval_ba
            best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
            best_a, best_b = tb, ta

        halflife = _compute_halflife(best_resid)
        if not (self.halflife_min <= halflife <= self.halflife_max):
            return None, True

        if _compute_hurst(best_resid, already_stationary=True) >= self.hurst_threshold:
            return None, True

        demeaned = best_resid - np.mean(best_resid)
        if int(np.sum(np.diff(np.sign(demeaned)) != 0)) < self.min_zero_crossings:
            return None, True

        spread_std = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0
        if self.use_cost_filter and not _cost_viable(
            spread_std, self.roundtrip_cost, entry_z=2.0, margin=self.cost_margin
        ):
            return None, True

        # SSD / DTW（強排序器，於 Z-Score 標準化對數價格空間）
        na, nb = norm_df[best_a].values, norm_df[best_b].values
        ssd = float(np.sum((na - nb) ** 2))
        dtw = _sakoe_chiba_dtw(na, nb, window=self.dtw_window)

        sec_a = self.sector_mapping.get(best_a.upper(), self.sector_mapping.get(best_a, "Unknown"))
        sec_b = self.sector_mapping.get(best_b.upper(), self.sector_mapping.get(best_b, "Unknown"))
        rec = {
            "Form_Start": self.form_start, "Form_End": self.form_end,
            "Sector": sec_a if sec_a == sec_b else "CrossSector",
            "Sector_A": sec_a, "Sector_B": sec_b,
            "Ticker_A": best_a, "Ticker_B": best_b,
            "ADF_Stat": round(best_stat, 6), "ADF_PValue": round(best_pval, 6),
            "SSD": round(ssd, 6), "DTW_Dist": round(dtw, 6),
            "Hedge_Ratio": round(best_beta, 6), "OLS_Alpha": round(best_alpha, 6),
            "Spread_Mean": round(float(np.mean(best_resid)), 6),
            "Spread_Std": round(spread_std, 6),
            "Correlation": round(corr, 6),
        }
        return rec, True

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method == "dtw":
            return df.sort_values("DTW_Dist")
        if self.method == "ssd":
            return df.sort_values("SSD")
        # ssd_dtw_pca：SSD+DTW 標準化後取 PC1（與勝出路徑一致；退化時回退 SSD）
        if len(df) < 2:
            return df.sort_values("SSD")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        feats = StandardScaler().fit_transform(df[["SSD", "DTW_Dist"]].values)
        pc1 = PCA(n_components=1, random_state=self.umap_random_state).fit_transform(feats).ravel()
        # 讓 PC1 與 SSD 正相關（分數越小越好）
        if np.corrcoef(pc1, df["SSD"].values)[0, 1] < 0:
            pc1 = -pc1
        out = df.copy()
        out["_score"] = pc1
        return out.sort_values("_score")

    def run(self) -> pd.DataFrame:
        log_prices = np.log(self.price_df)
        tickers = [t for t in log_prices.columns
                   if len(log_prices[t]) >= 30 and np.all(np.isfinite(log_prices[t].values))]
        if len(tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        log_prices = log_prices[tickers]
        R = np.diff(log_prices.values, axis=0)            # (T-1 × N)

        if self.factor_residual:
            sec = [self.sector_mapping.get(t.upper(), self.sector_mapping.get(t, "Unknown"))
                   for t in tickers]
            R = _residualize_returns(R, sector_labels=sec)
            print("  [Formation] 因子殘差化：已移除市場+產業共動（圖建於特殊性報酬）")

        mu = R.mean(axis=0); sd = R.std(axis=0, ddof=1); sd[sd < 1e-12] = 1e-12
        Rs = np.nan_to_num((R - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

        P = self._partial_corr_matrix(Rs)
        edges = self._candidate_edges(P)
        kind = "偏相關" if self.partial_corr else "普通相關"
        print(f"  [Formation] {kind}圖（{self.graph_method}）：{len(tickers)} 檔 → {len(edges)} 條候選邊")

        # Z-Score 標準化對數價格（SSD/DTW 用），與 DTW_Cointegration_Paper 一致
        norm_df = (log_prices - log_prices.mean()) / (log_prices.std() + 1e-12)

        records, n_counter = [], [0]
        for a, b in edges:
            rec, _ = self._screen_edge(tickers[a], tickers[b], log_prices, norm_df, n_counter)
            if rec is not None:
                records.append(rec)
        print(f"  [Formation] 共整合篩選：{len(edges)} 邊 → 通過 {len(records)} 對")

        if not records:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 研究框架 #2：BH-FDR（m = 做過 ADF 的邊數）
        if self.use_fdr:
            pvals = [r["ADF_PValue"] for r in records]
            bh_thr = _bh_fdr_threshold(pvals + [1.0] * max(0, n_counter[0] - len(pvals)), self.fdr_alpha)
            before = len(records)
            records = [r for r in records if r["ADF_PValue"] <= bh_thr]
            print(f"  [Formation] BH-FDR（m={n_counter[0]}, α={self.fdr_alpha}）："
                  f"門檻 p≤{bh_thr:.2e} | {before} → {len(records)} 對")
            if not records:
                self.selected_pairs = pd.DataFrame()
                return self.selected_pairs

        eg_df = self._rank(pd.DataFrame(records)).reset_index(drop=True)

        # 產業分散 + Top N
        if self.max_sector_ratio > 0:
            max_per_sec = max(1, int(self.top_n * self.max_sector_ratio))
            sec_exp: dict = {}
            picked = []
            for _, row in eg_df.iterrows():
                sa, sb = row["Sector_A"], row["Sector_B"]
                if sec_exp.get(sa, 0) < max_per_sec and sec_exp.get(sb, 0) < max_per_sec:
                    picked.append(row)
                    sec_exp[sa] = sec_exp.get(sa, 0) + 1
                    if sb != sa:
                        sec_exp[sb] = sec_exp.get(sb, 0) + 1
                if len(picked) >= self.top_n:
                    break
            selected = pd.DataFrame(picked).copy()
        else:
            selected = eg_df.head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)
        selected["Log_Mean_A"] = selected["Ticker_A"].map(log_prices.mean())
        selected["Log_Std_A"]  = selected["Ticker_A"].map(log_prices.std())
        selected["Log_Mean_B"] = selected["Ticker_B"].map(log_prices.mean())
        selected["Log_Std_B"]  = selected["Ticker_B"].map(log_prices.std())

        self.selected_pairs = selected
        return self.selected_pairs
