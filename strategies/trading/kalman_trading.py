import numpy as np
import pandas as pd
from strategies.trading.zscore_trading import Trading as BaseTrading


class Trading(BaseTrading):
    """
    Kalman Filter 配對交易策略。
    繼承 BaseTrading 的完整交易迴圈，僅覆寫 _compute_spread 改為卡爾曼濾波動態對沖比率。

    狀態向量 θ = [α, β]（截距、斜率），觀測方程：
        log_p_A[t] = x[t] @ θ[t] + ε[t]，x[t] = [1, log_p_B[t]]
    狀態轉移（隨機漫步）：θ[t] = θ[t-1] + w[t]，Q = delta * I

    對應的正規化新息 e/sqrt(S) 即為天然的 Z-Score，無需額外標準化。
    """
    def __init__(self, *args,
                 kalman_delta: float = 1e-4,
                 kalman_R: float = 1e-2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kalman_delta = kalman_delta
        self.kalman_R = kalman_R

    def _compute_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        common_idx: pd.DatetimeIndex,
        hedge_ratio: float,
        form_spread_mean: float,
        form_spread_std: float,
        log_mean_a,
        log_std_a: float,
        log_mean_b,
        log_std_b: float,
        first_price_a: float,
        first_price_b: float,
        ols_alpha,
    ) -> tuple[pd.Series, pd.Series]:
        """Kalman Filter 動態估計對沖比率，回傳正規化新息為 Z-Score。"""
        log_p_a = np.log(np.maximum(price_a.values, 1e-8))
        log_p_b = np.log(np.maximum(price_b.values, 1e-8))
        n = len(log_p_a)

        # 以形成期 OLS 結果初始化狀態（ols_alpha 可能為 None）
        alpha0 = float(ols_alpha) if ols_alpha is not None else 0.0
        theta = np.array([alpha0, hedge_ratio], dtype=np.float64)
        P = np.eye(2, dtype=np.float64)
        Q = self.kalman_delta * np.eye(2, dtype=np.float64)
        R = self.kalman_R

        zscore_vals = np.empty(n, dtype=np.float32)
        beta_vals = np.empty(n, dtype=np.float64)

        for i in range(n):
            x = np.array([1.0, log_p_b[i]])
            # Predict
            P_pred = P + Q
            # Innovation（預測誤差）
            e = log_p_a[i] - x @ theta
            S = float(x @ P_pred @ x) + R
            S = max(S, 1e-12)
            # Update（卡爾曼增益）
            K = P_pred @ x / S
            theta = theta + K * e
            P = P_pred - np.outer(K, x) @ P_pred
            # 正規化新息即天然 Z-Score
            zscore_vals[i] = e / np.sqrt(S)
            beta_vals[i] = theta[1]

        zscore = pd.Series(zscore_vals, index=common_idx).clip(-self.zscore_clip, self.zscore_clip)
        beta_series = pd.Series(beta_vals, index=common_idx)
        return zscore, beta_series
