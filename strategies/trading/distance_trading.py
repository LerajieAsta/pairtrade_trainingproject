"""
距離基準交易模組（GGR 距離法 / Gatev et al. 2006）
======================================================================

與現行 zscore_trading（回歸基準）的對照——教授指定的消融實驗：

  回歸基準（zscore_trading，路徑 A）：
      spread_t = ln P_A,t − α − β·ln P_B,t         （OLS 共整合殘差）
      交易訊號 z_t = (spread_t − μ_form) / σ_form
      → 依「共整合殘差」的均值回歸交易，對沖比率 β 來自回歸。

  距離基準（本模組，Gatev, Goetzmann & Rouwenhorst 2006）：
      正規化價格（標準化 log-price，以形成期 μ、σ 標準化）：
          P̃_A,t = (ln P_A,t − μ_A) / σ_A,  P̃_B,t 同理
      距離 spread（等權、無回歸，對沖比率固定 = 1）：
          D_t = P̃_A,t − P̃_B,t
      交易訊號 z_t = (D_t − μ_D) / σ_D，其中 μ_D、σ_D 為「形成期距離
      spread」的均值與標準差（GGR 的 2 個歷史標準差開倉門檻即 z_D=2）。
      → 依「兩條正規化價格路徑的距離」發散/收斂交易，不做回歸。

  兩者共用同一套 Z-Score 狀態機（進場 |z|>entry_z、出場 z 穿越 ±exit_z、
  期末強平），唯一差異在 spread 的建構空間（回歸殘差 vs 正規化價格距離）。
  這使得「回歸基準 vs 距離基準」成為乾淨的單變因對照。

Reference:
  Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading:
    Performance of a relative-value arbitrage rule. Review of Financial
    Studies, 19(3), 797–827.

介面與 zscore_trading.Trading 相同（run_trading.py 直接可用）。
"""

import numpy as np
import pandas as pd

from strategies.trading.zscore_trading import Trading as _BaseTrading

# 基底類別 __init__ 明確宣告的參數（過濾 run_trading 傳入的額外 kwargs）
_BASE_INIT_PARAMS = {
    "price_df", "trade_dates", "selected_pairs", "capital_per_pair",
    "fee_rate", "slippage_rate", "stop_loss_pct", "entry_z", "exit_z", "zscore_window",
    "allow_reentry", "zscore_clip", "min_spread_std",
    "use_dynamic_stop", "dynamic_stop_z", "portfolio_stop_loss_pct",
    "use_vol_adjust", "vol_regime_threshold", "hold_to_period_end",
}


class Trading(_BaseTrading):
    """
    GGR 距離法交易策略。繼承 zscore_trading 的完整狀態機，僅覆寫 spread 建構：
      改以「形成期標準化的正規化價格距離」作為交易訊號，對沖比率固定為 1
      （距離法的本質特徵——等權、無回歸）。
    """

    def __init__(self, *args,
                 full_price_df: pd.DataFrame = None,
                 formation_start: str = None,
                 formation_end: str = None,
                 **kwargs):
        base_kwargs = {k: v for k, v in kwargs.items() if k in _BASE_INIT_PARAMS}
        super().__init__(*args, **base_kwargs)
        _clean = lambda df: df.where(df.pct_change().abs() <= 0.50).ffill().bfill() if df is not None else None
        self.full_price_df = _clean(full_price_df.copy() if full_price_df is not None else None)
        self.formation_start = formation_start
        self.formation_end = formation_end
        self._cur_tickers = (None, None)   # 由 _simulate_pair 設定，供 _compute_spread 取形成期相關性

    def _simulate_pair(self, period_start, period_end, sector, ticker_a, ticker_b, *args, **kwargs):
        # 記住本配對，供 _compute_spread 由形成期價格計算距離 spread 的 σ_D
        self._cur_tickers = (ticker_a, ticker_b)
        return super()._simulate_pair(period_start, period_end, sector, ticker_a, ticker_b, *args, **kwargs)

    def _formation_distance_stats(self, log_mean_a, log_std_a, log_mean_b, log_std_b):
        """
        以形成期價格計算距離 spread D = P̃_A − P̃_B 的 (μ_D, σ_D)。
        P̃ 為標準化 log-price（形成期 μ、σ 標準化），故 μ_D ≈ 0，
        σ_D = sqrt(2(1−ρ))（ρ 為形成期 log-price 相關係數）。
        取不到形成期價格時退回 σ_D = sqrt(2(1−0.5))=1.0 的中性值。
        """
        ta, tb = self._cur_tickers
        rho = 0.5
        try:
            if (self.full_price_df is not None and self.formation_start and self.formation_end
                    and ta in self.full_price_df.columns and tb in self.full_price_df.columns):
                fp = self.full_price_df.loc[self.formation_start:self.formation_end, [ta, tb]].dropna()
                if len(fp) > 20:
                    la = (np.log(fp[ta].values) - log_mean_a) / (log_std_a if log_std_a else 1.0)
                    lb = (np.log(fp[tb].values) - log_mean_b) / (log_std_b if log_std_b else 1.0)
                    d = la - lb
                    mu_d = float(np.mean(d))
                    sd_d = float(np.std(d, ddof=1))
                    if np.isfinite(sd_d) and sd_d > 1e-9:
                        return mu_d, sd_d
                    c = np.corrcoef(la, lb)[0, 1]
                    if np.isfinite(c):
                        rho = c
        except Exception:
            pass
        return 0.0, float(np.sqrt(max(2.0 * (1.0 - rho), 1e-6)))

    def _compute_spread(self, price_a, price_b, common_idx, hedge_ratio,
                        form_spread_mean, form_spread_std,
                        log_mean_a, log_std_a, log_mean_b, log_std_b,
                        first_price_a, first_price_b, ols_alpha):
        """
        GGR 距離 spread：正規化價格距離 D_t = P̃_A,t − P̃_B,t（等權、hedge=1），
        z_t = (D_t − μ_D) / σ_D，μ_D、σ_D 取自形成期距離 spread。
        （完全忽略回歸的 ols_alpha 與 hedge_ratio，這正是距離法的定義。）
        """
        _empty = pd.Series(dtype=float)
        lma = log_mean_a if log_mean_a is not None else 0.0
        lmb = log_mean_b if log_mean_b is not None else 0.0
        lsa = log_std_a if log_std_a else 1.0
        lsb = log_std_b if log_std_b else 1.0

        norm_a = (np.log(price_a) - lma) / lsa
        norm_b = (np.log(price_b) - lmb) / lsb
        dist = norm_a - norm_b                      # 距離 spread（等權）

        mu_d, sd_d = self._formation_distance_stats(lma, lsa, lmb, lsb)
        safe_std = max(sd_d, self.min_spread_std)
        zscore = np.clip((dist - mu_d) / safe_std, -self.zscore_clip, self.zscore_clip)
        beta_series = pd.Series(1.0, index=common_idx)   # 距離法對沖比率固定 = 1
        if zscore.isna().all():
            return _empty, _empty
        return zscore, beta_series
