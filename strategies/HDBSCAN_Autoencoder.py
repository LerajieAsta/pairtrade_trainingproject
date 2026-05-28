

# ======================================================================
"""
Autoencoder-HDBSCAN 深度表徵配對交易滾動回測系統 (交易明細版)
核心功能：以 HDBSCAN 對形成期特徵向量進行密度分群，從同群內挑選最優配對，
          結合 Engle-Granger OLS Spread 建構 Z-Score 執行配對交易。

改寫基礎：SSD / EG 配對交易滾動回測系統
核心差異：
  - Formation：
      Step 1 → 擷取每支股票的多維特徵向量（動量、波動率、自相關、統計矩）
      Step 2 → UMAP 降維（常開，針對 S&P500 規模最佳化）至低維嵌入空間
      Step 3 → HDBSCAN 密度分群，過濾噪音點（label = -1）
      Step 4 → 同產業 × 同群落 雙重篩選，執行 EG 共整合檢定
              ADF p 值門檻：保守模式 < 1%，積極模式 < 5%
              依 ADF 統計量升序選 top_n 配對，輸出含 ADF_Stat / ADF_PValue
  - Trading：與 EG 版完全相同（OLS Spread + Z-Score）
  - DataProcessor / RollingBacktester：架構沿用，新增 HDBSCAN / ADF p 值參數
"""

import sqlite3
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# HDBSCAN 與 UMAP（需安裝：umap-learn，且支援 sklearn.cluster 原生 HDBSCAN 作為後備）
try:
    import hdbscan
    HDBSCAN_LIB = "hdbscan"
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        HDBSCAN_LIB = "sklearn"
    except ImportError:
        raise ImportError("請先安裝 scikit-learn >= 1.3.0 或 hdbscan：pip install scikit-learn hdbscan")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ umap-learn 未安裝，將跳過 UMAP 降維，直接以原始特徵向量執行 HDBSCAN。")

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 輔助函式
# ══════════════════════════════════════════════════════════════════════════════

def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """簡易 OLS：y = alpha + beta * x + resid，回傳 (alpha, beta, residuals)"""
    n = len(y)
    x_mat = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, y - np.mean(y)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    return alpha, beta, y - alpha - beta * x


def _adf_stat(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """
    ADF 檢定（no constant），同時回傳 (統計量, p 值)。
    失敗時回傳 (0.0, 1.0)，代表無法拒絕單根假設（最差情況）。

    p 值解讀：
      < 0.01 → 保守 (conservative)：強力拒絕單根，共整合顯著
      < 0.05 → 積極 (aggressive) ：一般顯著水準
      ≥ 0.05 → 不顯著，不納入配對池
    """
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, input_size=1, latent_dim=8, hidden_dim=32):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.3)  # 隨機去噪
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, input_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Encoder phase
        _, (hn, _) = self.encoder_lstm(x)
        # hn shape: (1, batch_size, hidden_dim)
        hn = hn[-1] # 取得最後一層的隱狀態
        hn = self.dropout(hn)
        latent = self.encoder_fc(hn) # (batch_size, latent_dim)
        
        # Decoder phase
        dec_input = self.decoder_fc(latent) # (batch_size, hidden_dim)
        dec_input = self.dropout(dec_input)
        dec_input = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch_size, seq_len, hidden_dim)
        
        dec_out, _ = self.decoder_lstm(dec_input) # (batch_size, seq_len, hidden_dim)
        decoded = self.decoder_output(dec_out) # (batch_size, seq_len, input_size)
        decoded = decoded.squeeze(-1) # (batch_size, seq_len)
        
        return latent, decoded

def train_autoencoder(X_train, latent_dim=8, epochs=100, lr=0.01, hidden_dim=32):
    # X_train shape: (n_stocks, seq_len)
    n_stocks, seq_len = X_train.shape
    
    # LSTM 需要 (batch_size, seq_len, input_size) 的 3D 輸入，此處 input_size = 1
    tensor_x = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    
    model = LSTMAutoencoder(seq_len=seq_len, input_size=1, latent_dim=latent_dim, hidden_dim=hidden_dim)
    # 引入 weight_decay 進行 L2 正則化，防止噪聲過擬合
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        latent, decoded = model(tensor_x)
        loss = criterion(decoded, torch.tensor(X_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        latent_features, _ = model(tensor_x)
    return latent_features.numpy()

# ══════════════════════════════════════════════════════════════════════════════
# Class 1：Formation（形成期模組）- HDBSCAN 版本
# ══════════════════════════════════════════════════════════════════════════════

class Formation:
    """
    形成期四步驟：
      1. 特徵萃取  → 每支股票 13 維特徵向量（標準化後）
      2. UMAP 降維 → 常開（S&P500 約 500 支股票，13 維 → umap_n_components 維）
      3. HDBSCAN   → 密度分群，自動決定群數，噪音點（label=-1）排除
      4. 同產業 × 同群落 雙重篩選 → EG 共整合，ADF p 值門檻過濾後依統計量升序選 top_n

    ADF p 值門檻說明：
      adf_pvalue_threshold = 0.01  → 保守 (conservative)：僅接受 1% 顯著水準
      adf_pvalue_threshold = 0.05  → 積極 (aggressive)  ：接受 5% 顯著水準
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,           # 產業分類字典，強制同產業配對
        min_tickers_for_pairing: int = 2,
        # HDBSCAN 參數
        hdbscan_min_cluster_size: int = 3,     # 最小群落大小
        hdbscan_min_samples: int = 1,          # 核心點最小鄰居數（越小 → 越少噪音）
        hdbscan_metric: str = "euclidean",     # 距離度量
        # 降維方法：'umap' 或 'pca'
        reduce_method: str = "umap",
        # UMAP 參數（S&P500 規模下常開）
        umap_n_components: int = 5,            # 降維目標維度
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        # ADF p 值門檻
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.05,   # 0.01=保守 / 0.05=積極
        max_sector_ratio: float = 0.3,
    ):
        self.price_df = price_df.copy()
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.max_sector_ratio = max_sector_ratio

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        self.reduce_method = reduce_method.lower()
        if self.reduce_method == "umap" and not UMAP_AVAILABLE:
            raise RuntimeError("umap-learn 未安裝，請執行：pip install umap-learn")
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors  = umap_n_neighbors
        self.umap_min_dist     = umap_min_dist
        self.umap_random_state = umap_random_state

        self.adf_max_lags           = adf_max_lags
        self.adf_pvalue_threshold   = adf_pvalue_threshold   # 0.01=保守 / 0.05=積極

        self.selected_pairs: pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict = {}     # ticker → cluster_label（供外部查閱）

    # ── Step 1：特徵萃取與標準化 ─────────────────────────────────────────────
    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        log_prices = np.log(self.price_df)
        valid_tickers = []
        ret_rows = []
        for ticker in log_prices.columns:
            series = log_prices[ticker].values
            if len(series) < 30 or not np.all(np.isfinite(series)):
                continue
            ret = np.diff(series)
            ret_rows.append(ret)
            valid_tickers.append(ticker)

        if not ret_rows:
            return np.empty((0, 0)), []

        X = np.vstack(ret_rows)                    # (n_stocks, seq_len)
        X = StandardScaler().fit_transform(X)        # 標準化
        return X, valid_tickers

    # ── Step 2：UMAP 降維（S&P500 規模下常開）────────────────────────────────
    def _umap_reduce(self, X: np.ndarray) -> np.ndarray:
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        n_neigh  = min(self.umap_n_neighbors,  n_stocks - 1)
        if n_comp < 1 or n_neigh < 1:
            return X
        reducer = umap.UMAP(
            n_components  = n_comp,
            n_neighbors   = n_neigh,
            min_dist      = self.umap_min_dist,
            random_state  = self.umap_random_state,
            low_memory    = True,
        )
        return reducer.fit_transform(X)

    # ── Step 2.5：PCA 降維（穩健性對照）─────────────────────────────────────
    def _pca_reduce(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        if n_comp < 1:
            return X
        pca = PCA(n_components=n_comp, random_state=self.umap_random_state)
        return pca.fit_transform(X)

    # ── Step 3：HDBSCAN 分群 ────────────────────────────────────────────────
    def _hdbscan_cluster(self, X: np.ndarray) -> np.ndarray:
        min_cs = min(self.hdbscan_min_cluster_size, max(2, X.shape[0] // 5))
        if HDBSCAN_LIB == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size = min_cs,
                min_samples      = self.hdbscan_min_samples,
                metric           = self.hdbscan_metric,
                core_dist_n_jobs = -1,
            )
        else:
            clusterer = sklearn_HDBSCAN(
                min_cluster_size = min_cs,
                min_samples      = self.hdbscan_min_samples,
                metric           = self.hdbscan_metric,
                n_jobs           = -1,
            )
        clusterer.fit(X)
        return clusterer.labels_   # -1 = 噪音

    # ── Step 4：同產業 × 同群落 雙重篩選 + EG 共整合 ──────────────────────────
    def _cointegration_within_clusters(
        self, tickers: list[str], labels: np.ndarray
    ) -> pd.DataFrame:
        """
        篩選邏輯：
          A. 同產業（sector_mapping 必須提供，Unknown 一律排除）
          B. 同 HDBSCAN 群落（label != -1）
          滿足 A ∩ B 才進行 EG 共整合。

        ADF p 值門檻（adf_pvalue_threshold）：
          0.01 → 保守：僅接受殘差在 1% 水準下顯著定態
          0.05 → 積極：接受殘差在 5% 水準下顯著定態
          p 值 ≥ 門檻者直接排除，不進入配對池。
        """
        log_prices    = np.log(self.price_df[tickers])
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("  [Formation] HDBSCAN 未找到任何有效群落（全為噪音點）。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(f"  [Formation] HDBSCAN 分群結果：{len(unique_labels)} 個群落，"
              f"{noise_count} 個噪音點排除，"
              f"ADF p 值門檻 = {self.adf_pvalue_threshold:.2f} "
              f"({'保守 1%' if self.adf_pvalue_threshold <= 0.01 else '積極 5%'})")

        # 建立 ticker → (sector, cluster_label) 對照表
        ticker_meta: dict[str, tuple[str, int]] = {}
        for t, lbl in zip(tickers, labels):
            sector = self.sector_mapping.get(t.upper(), "Unknown")
            ticker_meta[t] = (sector, int(lbl))

        # 按 (sector, cluster_label) 分組，同時排除 Unknown 與噪音
        group_map: dict[tuple[str, int], list[str]] = {}
        for t, (sec, lbl) in ticker_meta.items():
            if sec == "Unknown" or lbl == -1:
                continue
            group_map.setdefault((sec, lbl), []).append(t)

        # 排除成員數不足的群組
        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}

        if not valid_groups:
            print("  [Formation] 同產業 × 同群落後無有效配對組合。")
            return pd.DataFrame()

        total_group_count = len(valid_groups)
        print(f"  [Formation] 有效 (產業, 群落) 組合：{total_group_count} 組")

        eg_records = []
        passed_count = 0
        rejected_count = 0

        for (sector, cluster_lbl), group_tickers in sorted(valid_groups.items()):
            for i, ta in enumerate(group_tickers):
                log_a = log_prices[ta].values
                for j in range(i + 1, len(group_tickers)):
                    tb    = group_tickers[j]
                    log_b = log_prices[tb].values

                    # EG 正反兩方向，取 p 值較小（共整合較顯著）的方向
                    al_ab, be_ab, re_ab = _ols(log_a, log_b)
                    stat_ab, pval_ab = _adf_stat(re_ab, self.adf_max_lags)

                    al_ba, be_ba, re_ba = _ols(log_b, log_a)
                    stat_ba, pval_ba = _adf_stat(re_ba, self.adf_max_lags)

                    # 選 p 值較小的方向（p 值更小 = 更顯著 = 更傾向定態）
                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ta, tb
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = tb, ta

                    # p 值門檻過濾（核心篩選邏輯）
                    if best_pval >= self.adf_pvalue_threshold:
                        rejected_count += 1
                        continue

                    # Ornstein-Uhlenbeck 均值復歸半衰期過濾
                    dy = np.diff(best_resid)
                    y_lag = best_resid[:-1]
                    n_dy = len(dy)
                    x_mat = np.column_stack([np.ones(n_dy), y_lag])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(x_mat, dy, rcond=None)
                        lambda_val = coeffs[1]
                    except Exception:
                        lambda_val = 0.0

                    if lambda_val >= 0.0:
                        rejected_count += 1
                        continue
                    
                    halflife = -np.log(2) / lambda_val
                    if halflife < 2.0 or halflife > 60.0:
                        rejected_count += 1
                        continue

                    passed_count += 1
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    eg_records.append({
                        "Form_Start":    self.form_start,
                        "Form_End":      self.form_end,
                        "Sector":        sector,
                        "Cluster_Label": cluster_lbl,
                        "Ticker_A":      best_a,
                        "Ticker_B":      best_b,
                        "ADF_Stat":      round(best_stat,   6),
                        "ADF_PValue":    round(best_pval,   6),   # ← 新增 p 值欄位
                        "Hedge_Ratio":   round(best_beta,   6),
                        "OLS_Alpha":     round(best_alpha,  6),
                        "Spread_Mean":   round(spread_mean, 6),
                        "Spread_Std":    round(spread_std,  6),
                    })

        print(f"  [Formation] EG 檢定：{passed_count} 對通過 p < {self.adf_pvalue_threshold}，"
              f"{rejected_count} 對被 p 值門檻排除。")

        if not eg_records:
            return pd.DataFrame()

        return pd.DataFrame(eg_records).sort_values("ADF_Stat").reset_index(drop=True)

    # ── 主流程 ──────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        # Step 1：提取報酬率序列矩陣，並利用 Autoencoder 訓練深度特徵
        X_raw, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 訓練自編碼器，壓縮為 8 維深度特徵
        latent_features = train_autoencoder(X_raw, latent_dim=8, epochs=100, lr=0.01)
        # Step 2：降維 (UMAP 或 PCA)
        if self.reduce_method == "pca":
            X_embed = self._pca_reduce(latent_features)
        else:
            X_embed = self._umap_reduce(latent_features)

        # Step 3：HDBSCAN 分群
        labels = self._hdbscan_cluster(X_embed)
        self.cluster_labels_ = dict(zip(valid_tickers, labels.tolist()))

        # Step 4：同產業 × 同群落 EG 共整合
        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        if getattr(self, "max_sector_ratio", 0) > 0:
            max_pairs_per_sector = max(1, int(self.top_n * self.max_sector_ratio))
            sector_counts = {}
            diversified_records = []
            for _, row in eg_df.iterrows():
                sec = row["Sector"]
                if sec not in sector_counts:
                    sector_counts[sec] = 0
                if sector_counts[sec] < max_pairs_per_sector:
                    diversified_records.append(row)
                    sector_counts[sec] += 1
                if len(diversified_records) >= self.top_n:
                    break
            selected = pd.DataFrame(diversified_records).copy()
        else:
            selected = eg_df.head(self.top_n).copy()
            
        selected["Rank"] = range(1, len(selected) + 1)

        # 附加 log 統計量（供 Trading 期使用）
        log_prices  = np.log(self.price_df)
        mean_prices = log_prices.mean()
        std_prices  = log_prices.std()

        selected["Log_Mean_A"] = selected["Ticker_A"].map(mean_prices)
        selected["Log_Std_A"]  = selected["Ticker_A"].map(std_prices)
        selected["Log_Mean_B"] = selected["Ticker_B"].map(mean_prices)
        selected["Log_Std_B"]  = selected["Ticker_B"].map(std_prices)

        self.selected_pairs = selected
        return self.selected_pairs

# ══════════════════════════════════════════════════════════════════════════════
# Class 2：Trading（交易期模組）- 與 EG 版完全相同
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PairState:
    position: int = 0
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    cooldown_dir: int = 0
    prev_total_pnl: float = 0.0


class Trading:
    def __init__(
        self,
        price_df: pd.DataFrame,
        trade_dates: pd.DatetimeIndex,
        selected_pairs: pd.DataFrame,
        capital_per_pair: float,
        fee_rate: float,
        slippage_rate: float,
        stop_loss_pct: float,
        entry_z: float,
        exit_z: float,
        zscore_window: int,
        allow_reentry: bool = False,
        zscore_clip: float = 10.0,
        min_spread_std: float = 1e-6,
        use_dynamic_stop: bool = False,
        dynamic_stop_z: float = 3.0,
        portfolio_stop_loss_pct: float = 0.10,
        use_vol_adjust: bool = False,
    ):
        self.price_df        = price_df.copy()
        self.trade_dates     = trade_dates
        self.selected_pairs  = selected_pairs
        self.capital_per_pair = capital_per_pair
        self.friction_rate   = fee_rate + slippage_rate
        self.stop_loss_pct   = stop_loss_pct
        self.entry_z         = entry_z
        self.exit_z          = exit_z
        self.zscore_window   = zscore_window
        self.allow_reentry   = allow_reentry
        self.zscore_clip     = zscore_clip
        self.min_spread_std  = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z  = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust  = use_vol_adjust
        self.period_pnl: float = 0.0

    def _execute_entry(self, state, z, p_a, p_b, hedge_ratio):
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)

        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b =  v_b / p_b
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a =  v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a  = p_a
        state.entry_price_b  = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state, current_trade_pnl, stop_loss=False):
        state.realized_pnl += current_trade_pnl
        if stop_loss:
            state.is_stopped = True if not self.allow_reentry else False
            if self.allow_reentry:
                state.cooldown_dir = state.position
        else:
            state.cooldown_dir = state.position
        state.position = 0

    def _simulate_pair(
        self, period_start, period_end, sector, ticker_a, ticker_b, pair_rank,
        hedge_ratio, ols_alpha, form_spread_mean, form_spread_std,
        log_mean_a, log_std_a, log_mean_b, log_std_b,
        cluster_label, cluster_group,
    ) -> pd.DataFrame:

        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns:
            return pd.DataFrame()

        price_a = self.price_df[ticker_a].dropna()
        price_b = self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a    = price_a.loc[common_idx]
        price_b    = price_b.loc[common_idx]

        if len(price_a) < 5:
            return pd.DataFrame()

        log_a = np.log(price_a)
        log_b = np.log(price_b)

        # Z-Score 計算
        if self.zscore_window == 0:
            spread   = log_a - ols_alpha - hedge_ratio * log_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore   = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = pd.Series(hedge_ratio, index=common_idx)
            alpha_series = pd.Series(ols_alpha,   index=common_idx)
        else:
            w = self.zscore_window
            n = len(log_a)
            la_vals, lb_vals = log_a.values, log_b.values
            roll_alpha = np.full(n, np.nan)
            roll_beta  = np.full(n, np.nan)
            roll_mean  = np.full(n, np.nan)
            roll_std   = np.full(n, np.nan)

            for k in range(w - 1, n):
                ya = la_vals[k - w + 1: k + 1]
                xb = lb_vals[k - w + 1: k + 1]
                a_, b_, r_ = _ols(ya, xb)
                roll_alpha[k] = a_
                roll_beta[k]  = b_
                roll_mean[k]  = float(np.mean(r_))
                roll_std[k]   = float(np.std(r_, ddof=1)) if len(r_) > 1 else 0.0

            roll_alpha_s = pd.Series(roll_alpha, index=common_idx)
            roll_beta_s  = pd.Series(roll_beta,  index=common_idx)
            roll_mean_s  = pd.Series(roll_mean,  index=common_idx)
            roll_std_s   = pd.Series(roll_std,   index=common_idx)

            spread     = log_a - roll_alpha_s - roll_beta_s * log_b
            safe_std_s = np.maximum(roll_std_s, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std_s * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std_s
            zscore     = np.clip((spread - roll_mean_s) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = roll_beta_s
            alpha_series = roll_alpha_s

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()

        price_a      = price_a.loc[valid_idx]
        price_b      = price_b.loc[valid_idx]
        zscore       = zscore.loc[valid_idx]
        beta_series  = beta_series.loc[valid_idx]
        alpha_series = alpha_series.loc[valid_idx]

        dates_arr  = valid_idx
        zscore_arr = zscore.values
        pa_arr     = price_a.values
        pb_arr     = price_b.values
        beta_arr   = beta_series.values
        alpha_arr  = alpha_series.values

        base_log = {
            "Period_Start":   period_start,   "Period_End":     period_end,
            "Sector":         sector,          "Cluster_Label":  cluster_label,
            "Pair_Rank":      pair_rank,
            "Ticker_A":       ticker_a,        "Ticker_B":       ticker_b,
            "Log_Mean_A":     log_mean_a,      "Log_Std_A":      log_std_a,
            "Log_Mean_B":     log_mean_b,      "Log_Std_B":      log_std_b,
        }

        state = PairState()
        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_ols_alpha, out_z, out_pos = [], [], [], []
        out_unr, out_rea, out_cum = [], [], []
        out_status, out_tpnl, out_days, out_delta = [], [], [], []

        def _append_row(date, p_a, p_b, c_beta, c_alpha, z_val, pos,
                        unr, rea, cum, status, tpnl, days, delta):
            out_dates.append(date);      out_pa.append(round(p_a, 4));     out_pb.append(round(p_b, 4))
            out_hr.append(round(c_beta, 4)); out_ols_alpha.append(round(c_alpha, 6))
            out_z.append(round(z_val, 4));   out_pos.append(pos)
            out_unr.append(round(unr, 4));   out_rea.append(round(rea, 4)); out_cum.append(round(cum, 4))
            out_status.append(status);   out_tpnl.append(round(tpnl, 4))
            out_days.append(days);        out_delta.append(round(delta, 4))

        for i in range(len(dates_arr)):
            date    = dates_arr[i]
            z       = 0.0 if np.isnan(zscore_arr[i]) else float(zscore_arr[i])
            p_a, p_b = float(pa_arr[i]), float(pb_arr[i])
            c_beta   = float(beta_arr[i])  if not np.isnan(beta_arr[i])  else hedge_ratio
            c_alpha  = float(alpha_arr[i]) if not np.isnan(alpha_arr[i]) else ols_alpha

            unr, tpnl, status = 0.0, 0.0, "HOLD_CASH"

            if state.is_stopped:
                _append_row(date, p_a, p_b, c_beta, c_alpha, z, 0,
                            0.0, state.realized_pnl, state.realized_pnl,
                            "STOPPED", 0.0, 0, 0.0)
                continue

            # 冷卻解除
            if   state.cooldown_dir == -1 and z <= self.exit_z:  state.cooldown_dir = 0
            elif state.cooldown_dir ==  1 and z >= -self.exit_z: state.cooldown_dir = 0

            if state.position != 0:
                state.days_held += 1
                raw_unr  = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                cur_tpnl = raw_unr - state.trade_entry_fee - exit_fee

                is_cap_stop = self.stop_loss_pct > 0 and (-cur_tpnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, cur_tpnl, stop_loss=True)
                    tpnl, status = cur_tpnl, "STOP_LOSS_TRIGGERED"
                elif (state.position == -1 and z <= self.exit_z) or (state.position == 1 and z >= -self.exit_z):
                    self._execute_close(state, cur_tpnl, stop_loss=False)
                    tpnl, status = cur_tpnl, "EXIT"
                else:
                    unr    = raw_unr - state.trade_entry_fee
                    status = "HOLDING"
            else:
                if abs(z) > self.entry_z:
                    entered, unr = self._execute_entry(state, z, p_a, p_b, c_beta)
                    status = ("ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A") if entered else "HOLD_CASH (COOLDOWN)"
                else:
                    status = "HOLD_CASH"

            cum   = state.realized_pnl + unr
            delta = cum - state.prev_total_pnl
            state.prev_total_pnl = cum

            _append_row(date, p_a, p_b, c_beta, c_alpha, z, state.position,
                        unr, state.realized_pnl, cum, status, tpnl, state.days_held, delta)

            if status in ("STOP_LOSS_TRIGGERED", "EXIT"):
                state.days_held = 0

            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    _append_row(
                        dates_arr[j], float(pa_arr[j]), float(pb_arr[j]),
                        float(beta_arr[j]) if not np.isnan(beta_arr[j]) else hedge_ratio,
                        float(alpha_arr[j]) if not np.isnan(alpha_arr[j]) else ols_alpha,
                        0.0 if np.isnan(zscore_arr[j]) else float(zscore_arr[j]),
                        0, 0.0, state.realized_pnl, state.realized_pnl,
                        "STOPPED", 0.0, 0, 0.0
                    )
                break

        # 期末強制平倉
        if state.position != 0 and out_status:
            if out_status[-1] not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                p_a_last, p_b_last = float(pa_arr[-1]), float(pb_arr[-1])
                raw_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                fee_final = (abs(state.shares_a) * p_a_last + abs(state.shares_b) * p_b_last) * self.friction_rate
                final_tpnl = raw_final - state.trade_entry_fee - fee_final
                state.realized_pnl += final_tpnl
                pnl_prev = out_cum[-2] if len(out_cum) > 1 else 0.0

                out_status[-1]     = "PERIOD_END_EXIT"
                out_rea[-1]        = round(state.realized_pnl, 4)
                out_cum[-1]        = round(state.realized_pnl, 4)
                out_unr[-1]        = 0.0
                out_tpnl[-1]       = round(final_tpnl, 4)
                out_delta[-1]      = round(state.realized_pnl - pnl_prev, 4)
                out_days[-1]       = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "OLS_Alpha": out_ols_alpha,
            "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unr, "Realized_PnL": out_rea,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_tpnl, "Days_Held": out_days, "Daily_Delta": out_delta,
        })
        for k, v in base_log.items():
            df_out[k] = v
        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start, period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"], ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                ols_alpha=float(row.get("OLS_Alpha", 0.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A",  1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B",  1.0)),
                cluster_label=int(row.get("Cluster_Label", -1)),
                cluster_group=str(row.get("Sector", "Unknown")),
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # ---- 實作後置投資組合總體止損斷路器 ----
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()
            
            cutoff_date = None
            for date_val, pnl_val in daily_cum_pnl.items():
                if pnl_val / total_cap <= -self.portfolio_stop_loss_pct:
                    cutoff_date = date_val
                    break
            
            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date
                    
                    df_before = df[before_mask]
                    
                    df_at = df[at_mask].copy()
                    final_realized = 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        final_realized = row_at["Cumulative_PnL"]
                        if row_at["Position"] != 0:
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Trade_PnL"]
                        else:
                            if row_at["Status"] not in ("STOPPED", "STOP_LOSS_TRIGGERED", "EXIT"):
                                df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOPPED"
                            final_realized = row_at["Realized_PnL"]
                            
                    df_after = df[after_mask].copy()
                    if not df_after.empty:
                        df_after.loc[df_after.index, "Position"] = 0
                        df_after.loc[df_after.index, "Unrealized_PnL"] = 0.0
                        df_after.loc[df_after.index, "Realized_PnL"] = final_realized
                        df_after.loc[df_after.index, "Cumulative_PnL"] = final_realized
                        df_after.loc[df_after.index, "Status"] = "STOPPED"
                        df_after.loc[df_after.index, "Trade_PnL"] = 0.0
                        df_after.loc[df_after.index, "Daily_Delta"] = 0.0
                        
                    new_dfs.append(pd.concat([df_before, df_at, df_after], ignore_index=True))
                dfs = new_dfs
        # ----------------------------------------------

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily.sum()) if not period_daily.empty else 0.0
        return log_df, self.period_pnl

# ══════════════════════════════════════════════════════════════════════════════
# Class 3：DataProcessor（與原版相同）
# ══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    def __init__(self, db_path: str, table_name: str = "daily_prices"):
        self.db_path, self.table_name = db_path, table_name

    def load_sector_mapping(self, info_table, ticker_col="ticker", sector_col="sector") -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            df   = pd.read_sql_query(f"SELECT {ticker_col}, {sector_col} FROM {info_table}", conn)
            conn.close()
            mapping = {
                str(k).strip().upper(): str(v).strip()
                for k, v in zip(df[ticker_col], df[sector_col])
                if pd.notna(k) and pd.notna(v)
            }
            print(f"✅ 成功載入產業分類表 '{info_table}'，共取得 {len(mapping)} 檔標的分類。")
            return mapping
        except Exception as e:
            print(f"⚠️ 無法載入產業分類表：{e}，退回全市場模式。")
            return {}

    def prepare_backtest_data(self, backtest_start, backtest_end, formation_window):
        conn   = sqlite3.connect(self.db_path)
        raw_df = pd.read_sql_query(
            f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price "
            f"FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC", conn
        )
        conn.close()

        raw_df["date"]  = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]

        price_pivot = (
            raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
            .sort_index()
        )
        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20].ffill(limit=5)
        price_pivot.dropna(axis=1, thresh=int(len(price_pivot) * 0.9), inplace=True)

        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                if is_end and len(str(d_str).strip()) == 7:
                    return dt + pd.offsets.MonthEnd(0)
                return dt
            except Exception:
                return None

        bt_start_ts   = _safe_parse(backtest_start)
        bt_end_ts     = _safe_parse(backtest_end, is_end=True)
        all_dates     = price_pivot.index.tolist()
        start_indices = [i for i, d in enumerate(all_dates) if d >= bt_start_ts] if bt_start_ts else []
        first_idx     = start_indices[0] if start_indices else 0

        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end   = bt_end_ts if bt_end_ts else price_pivot.index[-1]
        price_pivot      = price_pivot.loc[data_slice_start:data_slice_end]

        sliced_dates      = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_idx   = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_idx, formation_window)

# ══════════════════════════════════════════════════════════════════════════════
# Class 4：RollingBacktester（HDBSCAN 版）
# ══════════════════════════════════════════════════════════════════════════════

class RollingBacktester:
    def __init__(
        self,
        top_n_list: list,
        stop_loss_list: list,
        zscore_window_list: list,
        entry_z: float,
        exit_z: float,
        formation_window: int,
        trading_window: int,
        rolling_step: int,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        allow_reentry: bool,
        zscore_clip: float,
        min_spread_std: float,
        min_tickers_for_pairing: int,
        # HDBSCAN 參數
        hdbscan_min_cluster_size: int,
        hdbscan_min_samples: int,
        hdbscan_metric: str,
        # UMAP 參數（常開）
        umap_n_components: int,
        umap_n_neighbors: int,
        umap_min_dist: float,
        umap_random_state: int,
        # EG + ADF p 值參數
        adf_max_lags: int,
        adf_pvalue_threshold: float,   # 0.01=保守 / 0.05=積極
        output_dir: Path,
        reduce_method: str = "umap",
        portfolio_stop_loss_pct_list: list = None,
        max_sector_ratio_list: list = None,
        dynamic_stop_z_list: list = None,
        use_vol_adjust_list: list = None,
    ):
        self.top_n_list = top_n_list
        self.stop_loss_list = stop_loss_list
        self.zscore_window_list = zscore_window_list
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.rolling_step = rolling_step
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing
        
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.adf_max_lags = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.output_dir = output_dir
        self.reduce_method = reduce_method

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

    def run(self, price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping):
        max_concurrent = self.trading_window // self.rolling_step
        states = {}
        for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
            self.top_n_list, self.stop_loss_list, self.zscore_window_list,
            self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
        ):
            states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)] = {
                "logs":  [],
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                          for _ in range(max_concurrent)],
            }

        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始 HDBSCAN Grid Search，共 {len(roll_start_indices)} 期，每期 {len(states)} 種參數組合...")

        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx = trade_start_idx - self.formation_window
            form_end_idx   = trade_start_idx
            trade_end_idx  = min(trade_start_idx + self.trading_window, total_days)

            form_data   = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data  = price_pivot.iloc[trade_start_idx:trade_end_idx]
            valid_cols  = (form_data.isnull().sum() + trade_data.isnull().sum()) == 0
            form_data   = form_data.loc[:, valid_cols]
            trade_dates = trade_data.index

            extended_start = max(0, trade_start_idx - max(self.zscore_window_list))
            extended_data  = price_pivot.iloc[extended_start:trade_end_idx].loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty:
                continue

            ts_str = str(all_dates[trade_start_idx])[:10]
            te_str = str(all_dates[trade_end_idx - 1])[:10]
            fs_str = str(all_dates[form_start_idx])[:10]
            fe_str = str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 第 {roll_idx+1:02d} 期 (交易: {ts_str} ~ {te_str})")

            # ── HDBSCAN 形成期（以最大 top_n * 5 計算一次以提供充足的配對池）
            formation = Formation(
                price_df=form_data,
                form_start=fs_str, form_end=fe_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing,
                hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,
                hdbscan_min_samples=self.hdbscan_min_samples,
                hdbscan_metric=self.hdbscan_metric,
                umap_n_components=self.umap_n_components,
                umap_n_neighbors=self.umap_n_neighbors,
                umap_min_dist=self.umap_min_dist,
                umap_random_state=self.umap_random_state,
                adf_max_lags=self.adf_max_lags,
                adf_pvalue_threshold=self.adf_pvalue_threshold,
                reduce_method=getattr(self, "reduce_method", "umap"),
                max_sector_ratio=0, # 在外部網格進行產業過濾
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty:
                continue

            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                # 產業上限過濾與 top_n 擷取
                if sec_ratio > 0:
                    max_pairs_per_sector = max(1, int(n * sec_ratio))
                    sector_counts = {}
                    diversified_records = []
                    for _, row in max_selected_pairs.iterrows():
                        sec = row["Sector"]
                        if sec not in sector_counts:
                            sector_counts[sec] = 0
                        if sector_counts[sec] < max_pairs_per_sector:
                            diversified_records.append(row)
                            sector_counts[sec] += 1
                        if len(diversified_records) >= n:
                            break
                    selected_pairs = pd.DataFrame(diversified_records).copy()
                else:
                    selected_pairs = max_selected_pairs.head(n).copy()

                if selected_pairs.empty:
                    continue

                selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                state  = states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)]
                slots  = state["slots"]

                free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                slot_idx   = free_slots[0] if free_slots else min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                cap_period   = slots[slot_idx]["capital"]
                cap_per_pair = cap_period / n

                trading = Trading(
                    price_df=extended_data, trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=cap_per_pair,
                    fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl, entry_z=self.entry_z, exit_z=self.exit_z,
                    zscore_window=z_win, allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip, min_spread_std=self.min_spread_std,
                    use_dynamic_stop=(dyn_z > 0),
                    dynamic_stop_z=dyn_z,
                    portfolio_stop_loss_pct=p_stop,
                    use_vol_adjust=vol_adj,
                )

                trade_log_df, period_pnl = trading.run(ts_str, te_str)

                if not trade_log_df.empty:
                    state["logs"].append(trade_log_df)

                slots[slot_idx]["capital"]   = max(0, cap_period + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        self._export_results(states)

    def _export_results(self, states):
        print("\n✅ 回測完成！正在匯出交易紀錄...")
        for (n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj), state in states.items():
            if state["logs"]:
                full_log = pd.concat(state["logs"], ignore_index=True)
                sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                rm_str   = getattr(self, "reduce_method", "umap").upper()
                psl_str = f"PSL{int(p_stop*100)}" if p_stop > 0 else "PSL0"
                msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
                dsz_str = f"DSZ{int(dyn_z)}" if dyn_z > 0 else "DSZ0"
                vol_str  = "VolAdj" if vol_adj else "NoVol"
                filename = f"HDBSCAN_AE_{rm_str}_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
                filepath = self.output_dir / filename
                full_log.to_csv(filepath, index=False)
                print(f"  - 已輸出: {filename} (共 {len(full_log)} 筆紀錄)")
        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════



# ======================================================================
# 原 Main 區塊被自動化註釋解耦如下：
# if __name__ == "__main__":

#     # --- 1. 路徑設定 ---
#     DB_PATH, TABLE_NAME          = r"../data/sp500_Current.db", "Daily_Prices"
#     BACKTEST_START, BACKTEST_END = "2000-01", "2025-12"
#     INFO_TABLE_NAME              = "Constituents"
#     TICKER_COL_NAME, SECTOR_COL_NAME = "Symbol", "GICS_Sector"

#     ALLOW_REENTRY = False
#     OUTPUT_DIR    = Path(r"../results/current/HDBSCAN_AE_NoReEntry")
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     # --- 2. 網格搜尋參數 ---
#     TOP_N_LIST         = [1, 5, 10, 20]
#     STOP_LOSS_LIST     = [0, 0.05, 0.1, 0.15]
#     ZSCORE_WINDOW_LIST = [0, 20, 40, 60]

#     ENTRY_Z, EXIT_Z                                  = 2.0, 0.0
#     FORMATION_WINDOW, TRADING_WINDOW, ROLLING_STEP   = 252, 126, 21

#     # --- 3. HDBSCAN 參數 ---
#     HDBSCAN_MIN_CLUSTER_SIZE = 5        # 最小群落大小（建議 3~10）
#     HDBSCAN_MIN_SAMPLES      = 1        # 核心點最小鄰居數（越小 → 越少噪音點）
#     HDBSCAN_METRIC           = "euclidean"

#     # --- 4. UMAP 參數（S&P500 規模下常開，不可關閉）---
#     UMAP_N_COMPONENTS = 5               # 13 維特徵 → 5 維嵌入（建議 3~8）
#     UMAP_N_NEIGHBORS  = 15              # 局部拓樸鄰居數（建議 10~30）
#     UMAP_MIN_DIST     = 0.1             # 嵌入點最小距離（越小 → 群落越緊密）
#     UMAP_RANDOM_STATE = 42

#     # --- 5. ADF p 值門檻（核心篩選參數）---
#     # 說明：對同產業 × 同群落配對的 OLS 殘差執行 ADF 單根檢定
#     #   保守模式：ADF_PVALUE_THRESHOLD = 0.01  → 僅接受 1% 顯著水準（配對數較少但品質高）
#     #   積極模式：ADF_PVALUE_THRESHOLD = 0.05  → 接受 5% 顯著水準（配對數較多但較寬鬆）
#     ADF_MAX_LAGS           = 1
#     ADF_PVALUE_THRESHOLD   = 0.01       # ← 改為 0.01 即切換至保守模式

#     # --- 6. 交易成本 ---
#     FEE_RATE               = 0.001
#     SLIPPAGE_RATE          = 0.001
#     INITIAL_CAPITAL        = 10_000
#     MIN_TICKERS_FOR_PAIRING = 2
#     ZSCORE_CLIP            = 10.0
#     MIN_SPREAD_STD         = 1e-6

#     # --- 7. 資料前處理 ---
#     processor      = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
#     sector_mapping = processor.load_sector_mapping(INFO_TABLE_NAME, TICKER_COL_NAME, SECTOR_COL_NAME)

#     price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
#         BACKTEST_START, BACKTEST_END, FORMATION_WINDOW
#     )

#     # --- 8. 啟動回測引擎 ---
#     engine = RollingBacktester(
#         top_n_list=TOP_N_LIST, stop_loss_list=STOP_LOSS_LIST, zscore_window_list=ZSCORE_WINDOW_LIST,
#         entry_z=ENTRY_Z, exit_z=EXIT_Z,
#         formation_window=FORMATION_WINDOW, trading_window=TRADING_WINDOW, rolling_step=ROLLING_STEP,
#         fee_rate=FEE_RATE, slippage_rate=SLIPPAGE_RATE, initial_capital=INITIAL_CAPITAL,
#         allow_reentry=ALLOW_REENTRY, zscore_clip=ZSCORE_CLIP, min_spread_std=MIN_SPREAD_STD,
#         min_tickers_for_pairing=MIN_TICKERS_FOR_PAIRING,
#         hdbscan_min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
#         hdbscan_min_samples=HDBSCAN_MIN_SAMPLES,
#         hdbscan_metric=HDBSCAN_METRIC,
#         umap_n_components=UMAP_N_COMPONENTS,
#         umap_n_neighbors=UMAP_N_NEIGHBORS,
#         umap_min_dist=UMAP_MIN_DIST,
#         umap_random_state=UMAP_RANDOM_STATE,
#         adf_max_lags=ADF_MAX_LAGS,
#         adf_pvalue_threshold=ADF_PVALUE_THRESHOLD,
#         output_dir=OUTPUT_DIR,
#     )

#     engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)


# ══════════════════════════════════════════════════════════════════════════════
# 標準化策略進入點接口 (Unified Strategy Entry Point)
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    標準化調用接口，接受外部傳入的價格資料與回測參數，完全解耦資料載入 I/O
    """
    import inspect
    from pathlib import Path
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取 RollingBacktester 的 __init__ 參數列表
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}
    
    # 動態將外部 params 對應並過濾為該策略 RollingBacktester 支援的參數
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    
    # 初始化回測引擎
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    # 執行回測
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
