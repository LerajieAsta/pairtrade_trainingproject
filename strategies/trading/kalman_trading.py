"""
時變對沖比率交易模組（Kalman 濾波）
======================================================================
現行 zscore_trading 在形成期估一次 OLS beta，鎖定整段交易期不變。
本模組改以 Kalman 濾波逐日更新 (alpha_t, beta_t)：

    觀測式   P̃_A,t = alpha_t + beta_t · P̃_B,t + eps_t,   eps ~ N(0, R)
    狀態式   (alpha_t, beta_t) = (alpha_{t-1}, beta_{t-1}) + eta_t,  eta ~ N(0, Q)

初始狀態取形成期 OLS 的 (alpha, beta)（即 form_spread_mean 與 hedge_ratio），
R 取形成期殘差變異（form_spread_std^2）。

⚠ 無前視：第 t 日交易所用的 beta 為**事前估計**（僅含 t-1 之前的觀測），
   t 期觀測只用於更新供 t+1 使用。

兩條臂（spread_mode）：
  "beta_only"（B1，單變因）：spread = P̃_A - alpha_form - beta_t · P̃_B，
      標準化沿用形成期的 mu/sigma。唯一變因是 beta，其餘與 zscore_trading 相同。
      已知不一致：mu_form/sigma_form 是以靜態 beta 估的，beta_t 漂移後兩者不再匹配。
  "innovation"（B2，文獻版，Chan 2013）：z_t = e_t / sqrt(S_t)，
      e_t 為量測創新、S_t 為其預測變異。自洽但同時改變了 beta 與標準化。

Q 的指定（q_mode）：
  "est"（預設）：由形成窗切 6 個 42 日子窗各估 beta，取 Var(diff(beta))。逐配對、無自由參數。
  "delta"：Q = kalman_delta * R（固定比例）。

預先註冊見 dev/kalman/PREREGISTRATION.md。

Reference:
  Chan, E. (2013). Algorithmic Trading: Winning Strategies and Their Rationale. Wiley.
  Do, B., Faff, R., & Hamza, K. (2006). A new approach to modeling and estimation
    for pairs trading. Working paper, Monash University.
"""

import numpy as np
import pandas as pd

from strategies.trading.zscore_trading import Trading as _BaseTrading

_BASE_INIT_PARAMS = {
    "price_df", "trade_dates", "selected_pairs", "capital_per_pair",
    "fee_rate", "slippage_rate", "stop_loss_pct", "entry_z", "exit_z", "zscore_window",
    "allow_reentry", "zscore_clip", "min_spread_std",
    "use_dynamic_stop", "dynamic_stop_z", "portfolio_stop_loss_pct",
    "use_vol_adjust", "vol_regime_threshold", "hold_to_period_end",
}


def _kalman_beta(za: np.ndarray, zb: np.ndarray, a0: float, b0: float,
                 R: float, Q: np.ndarray):
    """回傳 (alpha_prior, beta_prior, innovation, innovation_var)，皆為長度 T 的陣列。

    每個 t 的值只依賴 t 之前的觀測（事前估計），故可直接用於 t 期決策。
    """
    T = len(za)
    x = np.array([a0, b0], dtype=np.float64)
    P = Q * 10.0
    a_out = np.empty(T); b_out = np.empty(T)
    e_out = np.empty(T); s_out = np.empty(T)
    for i in range(T):
        P = P + Q                       # 預測
        a_out[i], b_out[i] = x[0], x[1]  # 事前狀態（供第 i 日決策）
        H = np.array([1.0, zb[i]])
        S = float(H @ P @ H) + R
        e = za[i] - float(H @ x)
        e_out[i] = e
        s_out[i] = S
        K = (P @ H) / S
        x = x + K * e                   # 更新（供 i+1）
        P = P - np.outer(K, H @ P)
    return a_out, b_out, e_out, s_out


class Trading(_BaseTrading):
    """Kalman 時變對沖比率。繼承 zscore_trading 的完整狀態機，僅覆寫 spread 建構。"""

    def __init__(self, *args,
                 full_price_df: pd.DataFrame = None,
                 formation_start: str = None,
                 formation_end: str = None,
                 spread_mode: str = "beta_only",
                 q_mode: str = "est",
                 kalman_delta: float = 1e-4,
                 **kwargs):
        base_kwargs = {k: v for k, v in kwargs.items() if k in _BASE_INIT_PARAMS}
        super().__init__(*args, **base_kwargs)
        _clean = lambda df: (df.where(df.pct_change().abs() <= 0.50).ffill().bfill()
                             if df is not None else None)
        self.full_price_df = _clean(full_price_df.copy() if full_price_df is not None else None)
        self.formation_start = formation_start
        self.formation_end = formation_end
        self.spread_mode = spread_mode
        self.q_mode = q_mode
        self.kalman_delta = float(kalman_delta)
        self._cur_tickers = (None, None)

    def _simulate_pair(self, period_start, period_end, sector, ticker_a, ticker_b, *args, **kwargs):
        self._cur_tickers = (ticker_a, ticker_b)
        return super()._simulate_pair(period_start, period_end, sector, ticker_a, ticker_b,
                                      *args, **kwargs)

    def _q_scalar(self, R: float, log_mean_a, log_std_a, log_mean_b, log_std_b) -> float:
        """Q 的對角量值。q_mode="est" 時由形成窗子期的 beta 變異估得。"""
        if self.q_mode != "est":
            return self.kalman_delta * R
        ta, tb = self._cur_tickers
        try:
            if (self.full_price_df is None or not self.formation_start or not self.formation_end
                    or ta not in self.full_price_df.columns or tb not in self.full_price_df.columns):
                return self.kalman_delta * R
            fp = self.full_price_df.loc[self.formation_start:self.formation_end, [ta, tb]].dropna()
            if len(fp) < 200:
                return self.kalman_delta * R
            la = (np.log(fp[ta].values) - log_mean_a) / (log_std_a if log_std_a else 1.0)
            lb = (np.log(fp[tb].values) - log_mean_b) / (log_std_b if log_std_b else 1.0)
            sub = []
            for k in range(6):
                s0, s1 = k * 42, (k + 1) * 42
                if s1 > len(la):
                    break
                v = np.var(lb[s0:s1], ddof=1)
                if v > 1e-10:
                    sub.append(np.cov(la[s0:s1], lb[s0:s1])[0, 1] / v)
            if len(sub) >= 3:
                q = float(np.var(np.diff(sub), ddof=1))
                if np.isfinite(q) and q > 0:
                    return q
        except Exception:
            pass
        return self.kalman_delta * R

    def _compute_spread(self, price_a, price_b, common_idx, hedge_ratio,
                        form_spread_mean, form_spread_std,
                        log_mean_a, log_std_a, log_mean_b, log_std_b,
                        first_price_a, first_price_b, ols_alpha):
        _empty = pd.Series(dtype=float)

        # 標準化空間（與 NOGRP-DTW 等 ignore_ols_alpha 臂一致）
        if first_price_a > 0.0 and log_mean_a is None:
            na = price_a / first_price_a
            nb = price_b / first_price_b
        else:
            lma = log_mean_a if log_mean_a is not None else 0.0
            lmb = log_mean_b if log_mean_b is not None else 0.0
            na = (np.log(price_a) - lma) / (log_std_a if log_std_a else 1.0)
            nb = (np.log(price_b) - lmb) / (log_std_b if log_std_b else 1.0)

        za = np.asarray(na, dtype=np.float64)
        zb = np.asarray(nb, dtype=np.float64)
        if len(za) < 5 or not np.isfinite(za).all() or not np.isfinite(zb).all():
            return _empty, _empty

        R = max(float(form_spread_std) ** 2, 1e-12)
        q = self._q_scalar(R, log_mean_a, log_std_a, log_mean_b, log_std_b)
        Q = np.eye(2) * q

        a_t, b_t, e_t, s_t = _kalman_beta(za, zb, float(form_spread_mean),
                                          float(hedge_ratio), R, Q)

        if self.spread_mode == "innovation":
            z = e_t / np.sqrt(np.maximum(s_t, self.min_spread_std ** 2))
        else:   # beta_only：唯一變因是 beta，標準化沿用形成期 mu/sigma
            spread = za - float(form_spread_mean) - b_t * zb
            z = spread / max(float(form_spread_std), self.min_spread_std)

        z = np.clip(z, -self.zscore_clip, self.zscore_clip)
        zscore = pd.Series(z, index=common_idx)
        beta_series = pd.Series(b_t, index=common_idx)
        if zscore.isna().all():
            return _empty, _empty
        return zscore, beta_series
