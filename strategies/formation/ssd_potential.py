# -*- coding: utf-8 -*-
"""以「扣成本後的獲利潛力」排序的形成期（dev/profit_potential/PREREGISTRATION.md）。

與 `ssd_rolling.Formation` 共用一切——分組、標準化對數股價、無截距 OLS 避險比率、價差、
Engle-Granger ADF 篩選（`_adf_stat_batch`，與 SSD 臂同一函式）——只把排序由 SSD 換成

    POT = NZC × (2·sd(S)/D − 2f)，D = 1/σ_A + |β|/σ_B

NZC 為形成期價差（去均值）穿越零軸次數；sd(S)/D 為 signal 避險下每 1 單位 z 的報酬；
2f 為來回成本。通過 ADF 且 POT > 0 者依 POT 由大到小、跨群全域取前 top_n（與 SSD 臂相同的全域取法）。

因排序不單調於 SSD，群內**全部**配對都要做 ADF，故以整群向量化計算取代 SSD 臂的逐列迴圈與提前中止。
輸出欄位與 SSD 臂相同（Hedge_Ratio 四捨五入到 4 位、Spread_Mean／Spread_Std 以未捨入的 β 計算），
另加 POT、NZC、Ret_Per_Z 供診斷。
"""
import numpy as np
import pandas as pd

from strategies.formation._utils import _adf_stat_batch
from strategies.formation.ssd_rolling import Formation as _SSDFormation

#: 單群向量化時一次處理的配對數上限（不分組約 8.4 萬對；20,000 × 252 × 8 bytes ≈ 40 MB）
_CHUNK = 20000


class Formation(_SSDFormation):
    def __init__(self, *args, entry_z: float = 2.0, friction_side: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        if friction_side is None:
            from strategies.config import base_params
            friction_side = float(base_params.get("fee_rate", 0.0)) + float(base_params.get("slippage_rate", 0.0))
        self.entry_z = float(entry_z)
        self.friction_side = float(friction_side)

    def _groups(self, tickers):
        groups = {}
        if self.sector_mapping:
            for t in tickers:
                groups.setdefault(self.sector_mapping.get(t, "Unknown"), []).append(t)
        else:
            groups["All_Market"] = tickers
        return groups

    def compute_ssd(self) -> pd.DataFrame:
        if self.normalized_df.empty:
            self.normalize_prices()
        tickers = self.normalized_df.columns.tolist()
        recs = []
        for sector, g in self._groups(tickers).items():
            if sector == "Unknown" or len(g) < self.min_tickers_for_pairing:
                continue
            V = self.normalized_df[g].values.T                     # (n, T)
            cov = np.cov(V)
            var = np.diag(cov)
            ii, jj = np.triu_indices(len(g), 1)                    # i = X = Ticker_B，j = Y = Ticker_A
            sig = self.std_prices[g].to_numpy()
            for lo in range(0, len(ii), _CHUNK):
                i, j = ii[lo:lo + _CHUNK], jj[lo:lo + _CHUNK]
                vx = var[i]
                beta = np.where(vx > 1e-8, cov[i, j] / np.where(vx > 1e-8, vx, 1.0), 0.0)
                S = V[j] - beta[:, None] * V[i]
                if self.enable_filters:
                    _, pv = _adf_stat_batch(S, max_lags=1)
                    keep = pv < self.adf_pvalue_threshold
                else:
                    keep = np.ones(len(i), dtype=bool)
                mean = S.mean(axis=1)
                sd = S.std(axis=1, ddof=1)
                nzc = (np.diff(np.sign(S - mean[:, None]), axis=1) != 0).sum(axis=1)
                d = 1.0 / sig[j] + np.abs(beta) / sig[i]
                ret_per_z = sd / d
                # 進場 |z| = entry_z、出場 0 的毛捕捉 = entry_z × 每單位 z 的報酬
                pot = nzc * (self.entry_z * ret_per_z - 2.0 * self.friction_side)
                keep &= pot > 0
                ssd_val = ((V[j] - V[i]) ** 2).sum(axis=1)
                for k in np.nonzero(keep)[0]:
                    recs.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": g[j[k]], "Ticker_B": g[i[k]],
                        "SSD": round(float(ssd_val[k]), 6), "Hedge_Ratio": round(float(beta[k]), 4),
                        "Spread_Mean": round(float(mean[k]), 6), "Spread_Std": round(float(sd[k]), 6),
                        "POT": float(pot[k]), "NZC": int(nzc[k]), "Ret_Per_Z": float(ret_per_z[k]),
                    })
        if not recs:
            return pd.DataFrame()
        return (pd.DataFrame(recs).sort_values(["POT", "SSD"], ascending=[False, True])
                .head(self.top_n).reset_index(drop=True))
