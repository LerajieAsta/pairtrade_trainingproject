"""
ML Pair Quality 配對交易形成期模組
======================================================================

不模仿 SSD Rolling（strategies/formation/ssd_rolling.py）的排序規則——那樣做
天花板就是「跟 SSD 一樣好，再扣掉模型近似誤差」。而是複用
strategies/trading/drl_threshold_trading.py（DRL-THR v4）已驗證有效的
walk-forward 反事實監督回歸範式，把它往形成期搬一格：不學 SSD 怎麼排序，
而是直接學「這個候選配對在下一個交易期會不會賺錢」，用配對的實際已實現
報酬當標籤訓練排序模型。

背景：HDBSCAN 系列三輪迭代（PCA-Loadings → Cluster 分組 → PCA5 降維修復
維度詛咒）都無法讓年化報酬追上 SSD (Rolling)，瓶頸鎖定在配對品質本身，
不是交易端。此模組把「配對排序」本身變成一個可學習的監督式問題，候選池
仍沿用 SSD Rolling 的同 GICS 產業分組（不用 HDBSCAN——三輪實驗已證明分群
本身是瓶頸來源），只換排序函式，才能乾淨對照「ML 排序 vs SSD 距離排序」
這一個變因。

流程（每個 rolling window，對應 run_formation.py 的一次 `.run()` 呼叫）：
  1. 候選池：同產業內全部配對，先用 SSD 距離初篩，每產業保留
     candidates_per_sector 組（比 SSD 自己篩選用的候選更寬鬆，保留多樣性
     供訓練，而非只留「SSD 最好的」）。
  2. 對候選池逐一計算特徵向量（SSD、DTW 距離、ADF p-value/統計量、半衰期、
     Hurst、hedge ratio、spread 標準差、波動率比、產業內 SSD 相對排名）——
     不當硬性過濾器，而是當連續特徵餵給模型，讓模型自己學這些統計量跟
     真實獲利的關係。
  3. 若 future_price_df 存在（run_formation.py 正常呼叫時一定有——見該檔案
     worker_task 的 kwargs 建構）：用標準 entry_z=2.0/exit_z=0.0 對候選池
     「每一個」候選跑 strategies/trading/zscore_trading.Trading._simulate_pair，
     算出真實已實現報酬當標籤，存進 walk-forward buffer。
  4. Walk-forward 訓練：只用交易期已經在本期交易開始前結束的歷史標籤
     （跟 drl_threshold_trading.py 的 _train_if_ready 邏輯逐字一致）；
     樣本不足時 fallback 回 SSD 距離排序（呼應 DRL-THR「樣本不足時用
     基準動作」的設計原則）。
  5. 足夠樣本時，用 MLP 對候選池全部打分，取預測報酬最高的 top_n。

輸出欄位需求（run_formation.py 的 Formation_Params 硬性合約，供
run_trading.py 讀回執行 _simulate_pair）：Ticker_A, Ticker_B, Hedge_Ratio,
Spread_Mean, Spread_Std, Log_Mean_A, Log_Std_A, Log_Mean_B, Log_Std_B。
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd_dist
import torch
import torch.nn as nn
import torch.optim as optim

from strategies.formation._utils import _compute_hurst, _ols, _adf_stat
from strategies.formation.DTW_Cointegration_Paper import _sakoe_chiba_dtw
from strategies.trading.zscore_trading import Trading as ZTrading

torch.set_num_threads(1)

N_FEAT = 10
_LABEL_CAPITAL = 10_000.0  # 僅供標籤產生用的名目本金，跟真實回測本金無關


class QualityNet(nn.Module):
    def __init__(self, feat_dim: int = N_FEAT, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Formation:
    """學習配對品質的形成期模組：候選池同 SSD Rolling（同產業配對），
    排序改用 walk-forward 訓練出的 MLP 預測報酬，樣本不足時 fallback 回
    SSD 距離排序。
    """

    _shared: dict = {}  # variant_id -> {"net", "opt", "buffer": [(feat, label, trade_end)], "trained_n"}

    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        future_price_df: pd.DataFrame = None,
        variant_id: str = "default",
        candidates_per_sector: int = 60,
        ml_hidden_size: int = 64,
        ml_lr: float = 1e-3,
        ml_train_epochs: int = 40,
        ml_min_train_samples: int = 200,
        trading_window: int = 126,
        dtw_window: int = 15,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.001,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        **kwargs,
    ):
        self.price_df = price_df
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.future_price_df = future_price_df
        self.variant_id = variant_id
        self.candidates_per_sector = candidates_per_sector

        self.hidden = ml_hidden_size
        self.lr = ml_lr
        self.train_epochs = ml_train_epochs
        self.min_samples = ml_min_train_samples

        self.halflife_max = trading_window / 3.0
        self.dtw_window = dtw_window
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    # ── walk-forward 共享狀態（依 variant_id 隔離，避免跨策略變體污染） ──
    def _get_shared(self) -> dict:
        key = self.variant_id
        sh = Formation._shared.get(key)
        if sh is None:
            net = QualityNet(N_FEAT, self.hidden).to(self.device)
            sh = {"net": net, "opt": optim.Adam(net.parameters(), lr=self.lr),
                  "buffer": [], "trained_n": 0}
            Formation._shared[key] = sh
            print(f"  [ML-Pair] New walk-forward quality net ({key}, device={self.device})")
        return sh

    def _train_if_ready(self, sh: dict, trade_start: str) -> int:
        """以「交易期已於本期交易開始前結束」的樣本增量訓練（無前視），
        跟 drl_threshold_trading.py 的 _train_if_ready 邏輯一致。"""
        eligible = [(f, l) for f, l, te in sh["buffer"] if te < trade_start]
        if len(eligible) < self.min_samples or len(eligible) == sh["trained_n"]:
            return len(eligible)

        X = torch.from_numpy(np.stack([e[0] for e in eligible])).to(self.device)
        Y = torch.from_numpy(np.array([e[1] for e in eligible], dtype=np.float32)).to(self.device)
        net, opt = sh["net"], sh["opt"]
        loss_fn = nn.MSELoss()
        net.train()
        n = len(eligible)
        bs = min(4096, n)
        for _ in range(self.train_epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                loss = loss_fn(net(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        sh["trained_n"] = n
        return n

    # ── 候選池：同 SSD Rolling 的同產業分組，SSD 距離初篩 ─────────────
    def _build_candidates(self):
        log_prices = np.log(np.maximum(self.price_df, 1e-8))
        mean_p = log_prices.mean()
        std_p = log_prices.std()
        norm_df = (log_prices - mean_p) / (std_p + 1e-12)

        sector_groups: dict = {}
        for t in norm_df.columns:
            sector = self.sector_mapping.get(t, "Unknown")
            if sector == "Unknown":
                continue
            sector_groups.setdefault(sector, []).append(t)

        candidates = {}
        for sector, sect_tickers in sector_groups.items():
            if len(sect_tickers) < self.min_tickers_for_pairing:
                continue
            norm_vals = norm_df[sect_tickers].values.T
            dmat = ssd_dist.squareform(ssd_dist.pdist(norm_vals, metric="sqeuclidean"))
            pairs = []
            n = len(sect_tickers)
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((sect_tickers[i], sect_tickers[j], float(dmat[i, j])))
            pairs.sort(key=lambda p: p[2])
            candidates[sector] = pairs[: self.candidates_per_sector]

        return candidates, norm_df, mean_p, std_p

    # ── 單一候選配對的特徵向量 ──────────────────────────────────────
    def _pair_features(self, ticker_a: str, ticker_b: str, norm_df: pd.DataFrame,
                        ssd_val: float, sector_rank: int, sector_size: int):
        na = norm_df[ticker_a].values
        nb = norm_df[ticker_b].values

        alpha, beta, resid = _ols(na, nb)
        stat, pval = _adf_stat(resid, max_lags=1)

        dy = np.diff(resid)
        y_lag = resid[:-1]
        denom = float(np.dot(y_lag, y_lag)) + 1e-12
        lam = float(np.dot(y_lag, dy)) / denom
        halflife = (-np.log(2) / lam) if lam < 0 else self.halflife_max * 2
        halflife = float(np.clip(halflife, 1.0, self.halflife_max * 2))

        hurst = _compute_hurst(resid, already_stationary=True)
        dtw_val = _sakoe_chiba_dtw(na, nb, window=self.dtw_window)

        ret_a, ret_b = np.diff(na), np.diff(nb)
        vol_a = float(np.std(ret_a)) + 1e-12
        vol_b = float(np.std(ret_b)) + 1e-12
        spread_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1e-6

        feat = np.array([
            np.log(ssd_val + 1e-6),
            np.log(dtw_val + 1e-6),
            np.clip(pval, 0.0, 1.0),
            np.clip(stat, -10.0, 10.0),
            np.log(halflife),
            hurst,
            np.clip(beta, -5.0, 5.0),
            np.clip(spread_std, 0.0, 10.0),
            np.clip(vol_a / vol_b - 1.0, -3.0, 3.0),
            np.clip(sector_rank / max(sector_size, 1), 0.0, 1.0),
        ], dtype=np.float32)
        feat = np.where(np.isfinite(feat), feat, 0.0).astype(np.float32)

        return feat, float(beta), float(np.mean(resid)), spread_std

    def run(self) -> pd.DataFrame:
        candidates, norm_df, mean_p, std_p = self._build_candidates()
        if not candidates:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        sh = self._get_shared()

        rows = []
        for sector, pairs in candidates.items():
            for rank, (ticker_a, ticker_b, ssd_val) in enumerate(pairs):
                feat, beta, spread_mean, spread_std = self._pair_features(
                    ticker_a, ticker_b, norm_df, ssd_val, rank, len(pairs)
                )
                rows.append({
                    "Sector": sector, "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                    "feat": feat, "Hedge_Ratio": beta,
                    "Spread_Mean": spread_mean, "Spread_Std": spread_std,
                    "ssd_rank": rank,
                })

        if not rows:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # ── 反事實標籤產生（僅供訓練用，不影響本期的最終排序決策） ──────
        n_eligible = 0
        if self.future_price_df is not None and len(self.future_price_df) > 5:
            trade_dates = self.future_price_df.index
            trade_start = trade_dates[0].strftime("%Y-%m-%d")
            trade_end = trade_dates[-1].strftime("%Y-%m-%d")

            sim = ZTrading(
                price_df=self.future_price_df, trade_dates=trade_dates,
                selected_pairs=pd.DataFrame(), capital_per_pair=_LABEL_CAPITAL,
                fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
                stop_loss_pct=0.0, entry_z=self.entry_z, exit_z=self.exit_z,
                zscore_window=0,
            )

            for row in rows:
                ta, tb = row["Ticker_A"], row["Ticker_B"]
                label = 0.0
                if ta in self.future_price_df.columns and tb in self.future_price_df.columns:
                    try:
                        df_log = sim._simulate_pair(
                            period_start=trade_start, period_end=trade_end, sector=row["Sector"],
                            ticker_a=ta, ticker_b=tb, pair_rank=row["ssd_rank"],
                            hedge_ratio=row["Hedge_Ratio"],
                            form_spread_mean=row["Spread_Mean"],
                            form_spread_std=max(row["Spread_Std"], 1e-6),
                            log_mean_a=float(mean_p[ta]), log_std_a=float(std_p[ta]),
                            log_mean_b=float(mean_p[tb]), log_std_b=float(std_p[tb]),
                        )
                        if not df_log.empty:
                            raw_pct = float(df_log["Realized_PnL"].iloc[-1]) / _LABEL_CAPITAL * 100.0
                            # log 壓縮：保留正負號與相對排序（單調變換），但壓低極端報酬
                            # 對 MSE 梯度的主導性——單一實例的已實現報酬肥尾嚴重（曾見過
                            # +314%），直接回歸原始百分比會讓少數極端值主導訓練，模型學到
                            # 的是「追逐罕見暴漲」而非穩健的配對品質。
                            label = float(np.sign(raw_pct) * np.log1p(abs(raw_pct)))
                    except Exception:
                        label = 0.0
                sh["buffer"].append((row["feat"], label, trade_end))

            n_eligible = self._train_if_ready(sh, trade_start)

        # ── 排序：樣本足夠用模型分數，否則 fallback 回 SSD 距離 ─────────
        if n_eligible >= self.min_samples:
            sh["net"].eval()
            with torch.no_grad():
                feats = torch.from_numpy(np.stack([r["feat"] for r in rows])).to(self.device)
                scores = sh["net"](feats).cpu().numpy()
            for r, s in zip(rows, scores):
                r["score"] = float(s)
            rows.sort(key=lambda r: -r["score"])
        else:
            rows.sort(key=lambda r: r["ssd_rank"])

        chosen = rows[: self.top_n]
        if not chosen:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        out = pd.DataFrame([{
            "Ticker_A": r["Ticker_A"], "Ticker_B": r["Ticker_B"], "Sector": r["Sector"],
            "Rank": i + 1,
            "Hedge_Ratio": round(r["Hedge_Ratio"], 4),
            "Spread_Mean": round(r["Spread_Mean"], 6),
            "Spread_Std": round(r["Spread_Std"], 6),
            "Log_Mean_A": float(mean_p[r["Ticker_A"]]), "Log_Std_A": float(std_p[r["Ticker_A"]]),
            "Log_Mean_B": float(mean_p[r["Ticker_B"]]), "Log_Std_B": float(std_p[r["Ticker_B"]]),
        } for i, r in enumerate(chosen)])

        self.selected_pairs = out
        return out
