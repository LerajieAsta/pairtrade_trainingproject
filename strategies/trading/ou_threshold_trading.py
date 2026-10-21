# -*- coding: utf-8 -*-
"""OU 解析最適門檻：DL-THR 的解析式對照（dev/ou_threshold/PREREGISTRATION.md）。

與 DL-THR 同一份動作選單（SKIP ＋ 進場 z ∈ {1.5, 2.0, 2.5, 3.0} × 出場 z ∈ {0, 0.5}），
但不學習：每個 (配對, 期) 以形成期 z 序列配適 OU，用 Bertram (2010) 的單位時間預期報酬
$\\mu(e,x) = ((e-x) - c_z)/E[T]$ 挑選動作，最大 $\\mu \\le 0$ 即 SKIP。

選定門檻後交給 `zscore_trading` 的同一套狀態機執行——進出場、停損、期末強制平倉、資金配置完全相同，
故「強制每期選 (2.0, 0)」時結果與 Z-Score 逐位元相同（接線驗證 V2）。

只支援標準化對數股價的價差（`_compute_spread` 路徑 B、`zscore_window=0`）與 signal 避險：
成本換算 $c_z$ 依賴 signal 避險下「每單位 Δz 的報酬 = Spread_Std / D」這個恆等式。
"""
import inspect

import numpy as np
import pandas as pd
from scipy.special import erfi

from strategies.trading.zscore_trading import Trading as _BaseTrading, clean_prices

SUPPORTS_HEDGE_MODE = True

_BASE_INIT_PARAMS = set(inspect.signature(_BaseTrading.__init__).parameters) - {"self"}

#: 與 drl_threshold_trading 的 v4 選單相同（不含 SKIP）
MENU = [(ez, xz) for ez in (1.5, 2.0, 2.5, 3.0) for xz in (0.0, 0.5)]
BASELINE = (2.0, 0.0)


def ou_fit(z: np.ndarray):
    """AR(1) $z_t = c + \\varphi z_{t-1} + e$ → (θ, σ, φ)。φ ∉ (0, 1) 時回傳 (None, None, φ)。"""
    x, y = z[:-1], z[1:]
    design = np.column_stack([np.ones_like(x), x])
    (c, phi), *_ = np.linalg.lstsq(design, y, rcond=None)
    if not (0.0 < phi < 1.0):
        return None, None, float(phi)
    resid = y - c - phi * x
    theta = -np.log(phi)
    sigma = float(np.std(resid, ddof=1)) * np.sqrt(2.0 * theta / (1.0 - phi ** 2))
    return float(theta), float(sigma), float(phi)


def bertram_mu(e: float, x: float, theta: float, sigma: float, c_z: float) -> float:
    """Bertram (2010)：多方進場 a=-e、出場 m=-x 的單位時間預期報酬（空方對稱）。"""
    a, m = -e, -x
    k = np.sqrt(theta) / sigma
    et = (np.pi / theta) * (erfi(m * k) - erfi(a * k))
    return float(((e - x) - c_z) / et) if et > 0 else -np.inf


def choose(theta, sigma, c_z):
    """回傳 (動作或 None, {動作: μ})；None = SKIP。"""
    if theta is None:
        return None, {}
    mus = {act: bertram_mu(act[0], act[1], theta, sigma, c_z) for act in MENU}
    best = max(mus, key=mus.get)
    return (best if mus[best] > 0 else None), mus


class Trading(_BaseTrading):
    """以 OU 解析式逐配對選門檻，執行沿用 zscore_trading。"""

    def __init__(self, *args,
                 full_price_df: pd.DataFrame = None,
                 formation_start: str = None,
                 formation_end: str = None,
                 ou_force_baseline: bool = False,
                 **kwargs):
        base_kwargs = {k: v for k, v in kwargs.items() if k in _BASE_INIT_PARAMS}
        super().__init__(*args, **base_kwargs)
        if self.zscore_window != 0:
            raise ValueError("ou_threshold 只定義在 zscore_window=0（形成期參數固定）下")
        if self.hedge_mode != "signal":
            raise ValueError("ou_threshold 的成本換算只對 signal 避險成立")
        self._ou_px = clean_prices(full_price_df)
        self._ou_fs, self._ou_fe = formation_start, formation_end
        self.ou_force_baseline = bool(ou_force_baseline)

    def _formation_z(self, ticker_a, ticker_b, hedge_ratio, form_spread_mean, form_spread_std,
                     log_mean_a, log_std_a, log_mean_b, log_std_b):
        """形成期 z 序列，與 _compute_spread 路徑 B（(ln P − μ)/σ 分支）同一定義。"""
        fp = self._ou_px
        if fp is None or ticker_a not in fp.columns or ticker_b not in fp.columns:
            return None
        fs, fe = pd.Timestamp(self._ou_fs), pd.Timestamp(self._ou_fe)
        form = fp.loc[fs:fe, [ticker_a, ticker_b]]
        form = form[form.index < fe].dropna()
        if len(form) < 60:
            return None
        na = (np.log(form[ticker_a].values) - log_mean_a) / log_std_a
        nb = (np.log(form[ticker_b].values) - log_mean_b) / log_std_b
        return (na - hedge_ratio * nb - form_spread_mean) / max(form_spread_std, self.min_spread_std)

    def _simulate_pair(self, period_start, period_end, sector, ticker_a, ticker_b, pair_rank,
                       hedge_ratio, form_spread_mean, form_spread_std, log_mean_a=None,
                       log_std_a=1.0, log_mean_b=None, log_std_b=1.0, first_price_a=0.0,
                       first_price_b=0.0, ols_alpha=None):
        if ols_alpha is not None or log_mean_a is None:
            raise ValueError("ou_threshold 只支援標準化對數股價的價差（路徑 B）")
        z = self._formation_z(ticker_a, ticker_b, hedge_ratio, form_spread_mean, form_spread_std,
                              log_mean_a, log_std_a, log_mean_b, log_std_b)
        theta = sigma = c_z = np.nan
        if z is None:
            act = None
        else:
            th, sg, _phi = ou_fit(z)
            d = 1.0 / max(abs(log_std_a), self.min_spread_std) + \
                abs(hedge_ratio) / max(abs(log_std_b), self.min_spread_std)
            c_z = 2.0 * self.friction_rate * d / max(form_spread_std, self.min_spread_std)
            act, _ = choose(th, sg, c_z)
            theta = th if th is not None else np.nan
            sigma = sg if sg is not None else np.nan
        if self.ou_force_baseline:
            act = BASELINE

        if act is None:
            self.entry_z, self.exit_z = np.inf, 0.0           # SKIP：整期不進場
        else:
            self.entry_z, self.exit_z = act
        df = super()._simulate_pair(period_start, period_end, sector, ticker_a, ticker_b, pair_rank,
                                    hedge_ratio, form_spread_mean, form_spread_std, log_mean_a,
                                    log_std_a, log_mean_b, log_std_b, first_price_a,
                                    first_price_b, ols_alpha)
        if df.empty:
            return df
        if act is None:
            # 與 DL-THR 同一個 SKIP 標記（prop2_skip_permutation 以此判定）
            df["Status"] = df["Status"].replace("HOLD_CASH", "HOLD_CASH (SKIP)")
        df["Sel_Entry_Z"] = np.nan if act is None else act[0]
        df["Sel_Exit_Z"] = np.nan if act is None else act[1]
        df["OU_Theta"], df["OU_Sigma"], df["OU_Cz"] = theta, sigma, c_z
        return df
