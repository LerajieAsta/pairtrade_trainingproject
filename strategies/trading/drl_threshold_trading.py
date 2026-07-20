"""
DRL 門檻選擇式交易模組 v4（Kim & Kim 2019 風格）
======================================================================

逐日定位動作空間的三代 DRL（v1 online DQN / v2 修復版 / v3 FQI）已被
系統性證偽：即使修復訓練的計算與統計缺陷，「每天自由決定持倉」的動作
空間仍讓模型對日級噪音計時，OOS 換手費用 6 倍於基準（中位 Sharpe
−1.1 ~ −2.3 vs Z-Score ≈ 0）。失敗根因被隔離在動作空間設計。

v4 徹底縮減動作空間：agent 每配對每期只做「一個」決策——
    動作選單（9）：SKIP（不交易）+ 8 組 (entry_z, exit_z) 門檻
                   ∈ {1.5, 2.0, 2.5, 3.0} × {0.0, 0.5}
選定後交給標準 Z-Score 狀態機執行整個交易期。

結構保證：
  1. 選單包含靜態基準 (2.0, 0.0) → 策略空間 ⊇ Z-Score 基準
  2. 訓練樣本不足（走勢初期）時自動選基準動作 → 早期 ≡ Z-Score
  3. SKIP 讓 agent 可以拒絕壞配對（Z-Score 做不到的選擇性）

學習問題的性質：歷史配對期的「全部 9 個動作的報酬」都可以精確反事實
回算（對交易期價格逐一模擬 9 組門檻）→ 這是全資訊監督回歸，
不是部分回饋的 bandit——無探索問題、樣本效率最高。
  net: MLP(12 維形成期特徵) → 9 個動作的預期報酬
  walk-forward：期 k 決策只用「交易期已於 k 開始前結束」的樣本訓練（無前視）

介面與 v1–v3 相同（run_trading.py 直接可用）。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

torch.set_num_threads(1)

# 動作選單（版本化）：動作統一為 (entry_z, exit_z, max_hold) 三元組；max_hold=0 表示不限。
#   v4（預設）：SKIP + 8 組 (ez, xz)，max_hold 恆 0 —— 與歷史結果完全相容。
#   v5（P4）：SKIP + (ez × xz × mh)，mh ∈ {0, 63} —— 時間維度入選單，agent 可對
#       每配對自選「耐心上限」；63d 依據交易解剖（>63d 未收斂勝率 26-46%）。
def _build_actions(menu_version: int):
    if menu_version >= 5:
        acts = [None] + [(ez, xz, mh)
                         for ez in (1.5, 2.0, 2.5, 3.0)
                         for xz in (0.0, 0.5)
                         for mh in (0, 63)]
    else:
        acts = [None] + [(ez, xz, 0) for ez in (1.5, 2.0, 2.5, 3.0) for xz in (0.0, 0.5)]
    return acts, acts.index((2.0, 0.0, 0))

_ACTIONS_V4, _BASELINE_V4 = _build_actions(4)
N_FEAT = 12


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
    prev_total_pnl: float = 0.0


class ThresholdNet(nn.Module):
    def __init__(self, feat_dim=N_FEAT, hidden=64, n_actions=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _fast_threshold_pnl(z: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                        hedge_ratio: float, capital: float, friction: float,
                        entry_z: float, exit_z: float, max_hold: int = 0) -> float:
    """
    以指定門檻在交易期上模擬 Z-Score 狀態機，回傳期末已實現 PnL。
    與 zscore_trading 相同語意（進場 |z|>ez、出場 z 穿越 ±exit_z、期末強平），
    精確股數會計；用於反事實標籤與正式模擬。
    """
    tw = 1.0 + abs(hedge_ratio)
    st = PairState()
    T = len(z)
    frozen = False
    for i in range(T):
        zi, p_a, p_b = z[i], pa[i], pb[i]
        if st.position != 0:
            st.days_held += 1
            is_exit = (st.position == -1 and zi <= exit_z) or (st.position == 1 and zi >= -exit_z)
            is_time = max_hold > 0 and st.days_held >= max_hold
            if is_exit or is_time or i == T - 1:
                if is_time and not is_exit:
                    frozen = True   # 超時=均衡斷裂：本期凍結，不再進場
                raw = st.shares_a * (p_a - st.entry_price_a) + st.shares_b * (p_b - st.entry_price_b)
                fee = (abs(st.shares_a) * p_a + abs(st.shares_b) * p_b) * friction
                st.realized_pnl += raw - st.trade_entry_fee - fee
                st.position = 0
                st.shares_a = st.shares_b = 0.0
                st.trade_entry_fee = 0.0
                st.days_held = 0
                continue
        elif (not frozen) and abs(zi) > entry_z and i < T - 1:
            v_a = capital / tw
            v_b = capital * abs(hedge_ratio) / tw
            if zi > entry_z:
                st.position, st.shares_a, st.shares_b = -1, -v_a / p_a, v_b / p_b
            else:
                st.position, st.shares_a, st.shares_b = 1, v_a / p_a, -v_b / p_b
            st.entry_price_a, st.entry_price_b = p_a, p_b
            st.trade_entry_fee = (abs(st.shares_a) * p_a + abs(st.shares_b) * p_b) * friction
    return st.realized_pnl


class Trading:
    """DRL Threshold-Selection Trading Strategy Interface for run_trading.py"""

    _shared: dict = {}      # {scope key -> dict(net, opt, buffer, n_trained)}

    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex,
                 selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float,
                 drl_hidden_size: int = 64,
                 drl_lr: float = 1e-3,
                 thr_train_epochs: int = 40,       # 每期增量訓練 epoch 數
                 thr_min_train_samples: int = 200,  # 樣本不足時使用基準動作
                 thr_menu_version: int = 4,         # P4：5 = 選單含 max_hold 時間維度
                 full_price_df: pd.DataFrame = None,
                 formation_start: str = None, formation_end: str = None,
                 variant_id: str = "default", **kwargs):

        _pct_clean = lambda df: df.where(df.pct_change().abs() <= 0.50).ffill().bfill() if df is not None else None
        self.trade_prices = _pct_clean(price_df.copy())
        self.full_price_df = _pct_clean(full_price_df.copy() if full_price_df is not None else None)

        self.trade_dates = trade_dates
        self.capital_per_pair = capital_per_pair
        self.friction_rate = fee_rate + slippage_rate

        self.hidden = drl_hidden_size
        self.lr = drl_lr
        self.train_epochs = thr_train_epochs
        self.min_samples = thr_min_train_samples

        self.formation_start = formation_start
        self.formation_end = formation_end
        self.menu_version = int(thr_menu_version)
        self.actions, self.baseline_idx = _build_actions(self.menu_version)
        self.n_actions = len(self.actions)
        self.variant_id = variant_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 形成期特徵（12 維，標準化 ~[-3,3]） ──────────────────────────────
    @staticmethod
    def _pair_features(z: np.ndarray, log_a: np.ndarray, log_b: np.ndarray,
                       hedge_ratio: float, spread_std: float) -> np.ndarray:
        T = len(z)
        ret_a, ret_b = np.diff(log_a), np.diff(log_b)
        zc = float(np.sum(np.diff(np.sign(z)) != 0)) / max(T, 1)

        # AR1 半衰期
        dy, ylag = np.diff(z), z[:-1]
        denom = float(np.dot(ylag, ylag)) + 1e-12
        lam = float(np.dot(ylag, dy)) / denom
        halflife = -np.log(2) / lam if lam < 0 else 252.0
        halflife = float(np.clip(halflife, 1.0, 252.0))

        z21 = z[-21:] if T >= 21 else z
        va = float(np.std(ret_a)) + 1e-12
        vb = float(np.std(ret_b)) + 1e-12
        corr = float(np.corrcoef(log_a, log_b)[0, 1]) if T > 2 else 0.0

        f = np.array([
            np.clip(z[-1] / 3.0, -3, 3),                          # 期末 z（決定初始偏離方向/幅度）
            np.clip(abs(z[-1]) / 3.0, 0, 3),
            np.clip(zc * 10.0, 0, 3),                             # 零穿越頻率
            np.log(halflife) / 3.0,                               # 均值回歸速度
            np.clip(float(np.std(z21)) / (float(np.std(z)) + 1e-12) - 1.0, -3, 3),   # 近期 z 波動 regime
            np.clip((z[-1] - float(np.mean(z21))) / 3.0, -3, 3),  # 近期 z 趨勢
            np.clip(corr, -1, 1),
            np.clip(va / vb - 1.0, -3, 3),
            np.clip(hedge_ratio - 1.0, -3, 3),
            np.clip(spread_std * 5.0, 0, 3),                      # spread 振幅（費用可行性）
            float(np.mean(np.abs(z) > 2.0)),                      # 形成期 |z|>2 佔比
            np.clip(float(np.max(np.abs(z))) / 5.0, 0, 3),
        ], dtype=np.float32)
        return np.where(np.isfinite(f), f, 0.0)

    def _get_shared(self):
        # 依 variant_id（含策略名稱、Top_n、停損、MSR）區分，避免同一 worker process
        # 生命週期內依序處理的不同策略變體共用同一份網路/緩衝區，互相污染訓練資料。
        key = f"{self.variant_id}|menu_v{self.menu_version}"
        sh = Trading._shared.get(key)
        if sh is None:
            net = ThresholdNet(N_FEAT, self.hidden, self.n_actions).to(self.device)
            sh = {"net": net, "opt": optim.Adam(net.parameters(), lr=self.lr),
                  "buffer": [], "trained_n": 0}
            Trading._shared[key] = sh
            print(f"    [DRL-THR] New walk-forward threshold net ({key}, device={self.device})")
        return sh

    def _train_if_ready(self, sh: dict, trade_start):
        """以「交易期已於本期開始前結束」的樣本增量訓練（無前視）。"""
        eligible = [(f, r) for f, r, te in sh["buffer"] if te < trade_start]
        if len(eligible) < self.min_samples or len(eligible) == sh["trained_n"]:
            return len(eligible)
        X = torch.from_numpy(np.stack([e[0] for e in eligible])).to(self.device)
        Y = torch.from_numpy(np.stack([e[1] for e in eligible])).to(self.device)
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

    # ── 主模擬 ───────────────────────────────────────────────────────────
    def _simulate_pair(self, period_start: str, period_end: str, sector: str,
                       ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float,
                       form_spread_mean: float, form_spread_std: float,
                       log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float,
                       **kwargs) -> pd.DataFrame:

        sh = self._get_shared()

        # 交易期價格與 z 序列（形成期凍結參數，與 zscore 基準同口徑）
        price_a = self.trade_prices[ticker_a].dropna()
        price_b = self.trade_prices[ticker_b].dropna()
        common = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common], price_b.loc[common]
        if len(price_a) < 5:
            return pd.DataFrame()
        valid_idx = common.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()

        sstd = max(float(form_spread_std), 1e-8)

        def z_of(pa: pd.Series, pb: pd.Series) -> np.ndarray:
            na = (np.log(pa.values) - log_mean_a) / log_std_a
            nb = (np.log(pb.values) - log_mean_b) / log_std_b
            return np.clip((na - hedge_ratio * nb - form_spread_mean) / sstd, -10.0, 10.0)

        pa_t = price_a.loc[valid_idx]
        pb_t = price_b.loc[valid_idx]
        z_t = z_of(pa_t, pb_t)
        pa_arr, pb_arr = pa_t.values.astype(float), pb_t.values.astype(float)

        # 形成期特徵
        feats = None
        try:
            form = self.full_price_df.loc[self.formation_start:self.formation_end]
            fa, fb = form[ticker_a].dropna(), form[ticker_b].dropna()
            cf = fa.index.intersection(fb.index)
            if len(cf) > 60:
                fa, fb = fa.loc[cf], fb.loc[cf]
                z_f = z_of(fa, fb)
                feats = self._pair_features(z_f, np.log(fa.values), np.log(fb.values),
                                            hedge_ratio, sstd)
        except Exception:
            pass

        # ── 決策：訓練樣本足夠 → net argmax；否則基準動作 ────────────────
        trade_start_ts = valid_idx[0]
        action_idx = self.baseline_idx
        if feats is not None:
            n_eligible = self._train_if_ready(sh, trade_start_ts)
            if n_eligible >= self.min_samples:
                sh["net"].eval()
                with torch.no_grad():
                    q = sh["net"](torch.from_numpy(feats).unsqueeze(0).to(self.device))
                    action_idx = int(q.argmax().item())

        # ── 反事實標籤：9 個動作在本期的實際報酬（僅供「未來」期訓練） ────
        if feats is not None:
            rets = np.zeros(self.n_actions, dtype=np.float32)
            for ai, act in enumerate(self.actions):
                if act is None:
                    continue
                rets[ai] = _fast_threshold_pnl(z_t, pa_arr, pb_arr, hedge_ratio,
                                               self.capital_per_pair, self.friction_rate,
                                               act[0], act[1], act[2]) / self.capital_per_pair * 100.0
            sh["buffer"].append((feats, rets, valid_idx[-1]))

        # ── 以選定動作執行正式模擬（完整交易紀錄） ────────────────────────
        chosen = self.actions[action_idx]
        st = PairState()
        pair_frozen = False
        out = {k: [] for k in ["dates", "pa", "pb", "hr", "z", "pos", "unreal", "real",
                               "cum", "status", "tpnl", "days", "delta"]}
        T = len(valid_idx)
        for i in range(T):
            zi, p_a, p_b = float(z_t[i]), float(pa_arr[i]), float(pb_arr[i])
            unrealized = 0.0
            closed_pnl = 0.0
            status = "HOLD_CASH (SKIP)" if chosen is None else "HOLD_CASH"

            if chosen is not None:
                ez, xz, mh = chosen
                if st.position != 0:
                    st.days_held += 1
                    raw = st.shares_a * (p_a - st.entry_price_a) + st.shares_b * (p_b - st.entry_price_b)
                    fee = (abs(st.shares_a) * p_a + abs(st.shares_b) * p_b) * self.friction_rate
                    cur_pnl = raw - st.trade_entry_fee - fee
                    is_exit = (st.position == -1 and zi <= xz) or (st.position == 1 and zi >= -xz)
                    is_time = mh > 0 and st.days_held >= mh
                    if is_time and not is_exit:
                        pair_frozen = True
                    if (is_exit or is_time) and i < T - 1:
                        st.realized_pnl += cur_pnl
                        closed_pnl = cur_pnl
                        st.position = 0
                        st.shares_a = st.shares_b = 0.0
                        st.trade_entry_fee = 0.0
                        st.days_held = 0
                        status = "TIME_STOP" if (is_time and not is_exit) else "EXIT"
                    else:
                        unrealized = cur_pnl
                        status = "HOLDING"
                elif (not pair_frozen) and abs(zi) > ez and i < T - 1:
                    tw = 1.0 + abs(hedge_ratio)
                    v_a = self.capital_per_pair / tw
                    v_b = self.capital_per_pair * abs(hedge_ratio) / tw
                    if zi > ez:
                        st.position, st.shares_a, st.shares_b = -1, -v_a / p_a, v_b / p_b
                        status = "ENTER_SHORT_A"
                    else:
                        st.position, st.shares_a, st.shares_b = 1, v_a / p_a, -v_b / p_b
                        status = "ENTER_LONG_A"
                    st.entry_price_a, st.entry_price_b = p_a, p_b
                    st.trade_entry_fee = (abs(st.shares_a) * p_a + abs(st.shares_b) * p_b) * self.friction_rate

            cum = st.realized_pnl + unrealized
            delta = cum - st.prev_total_pnl
            st.prev_total_pnl = cum
            out["dates"].append(valid_idx[i])
            out["pa"].append(round(p_a, 4))
            out["pb"].append(round(p_b, 4))
            out["hr"].append(round(float(hedge_ratio), 4))
            out["z"].append(round(zi, 4))
            out["pos"].append(st.position)
            out["unreal"].append(round(unrealized, 4))
            out["real"].append(round(st.realized_pnl, 4))
            out["cum"].append(round(cum, 4))
            out["status"].append(status)
            out["tpnl"].append(round(closed_pnl, 4))
            out["days"].append(st.days_held)
            out["delta"].append(round(delta, 4))

        # 期末強制平倉
        if st.position != 0 and out["status"]:
            before = out["cum"][-2] if len(out["cum"]) > 1 else 0.0
            p_a, p_b = out["pa"][-1], out["pb"][-1]
            raw = st.shares_a * (p_a - st.entry_price_a) + st.shares_b * (p_b - st.entry_price_b)
            fee = (abs(st.shares_a) * p_a + abs(st.shares_b) * p_b) * self.friction_rate
            closed_pnl = raw - st.trade_entry_fee - fee
            st.realized_pnl += closed_pnl
            out["status"][-1] = "PERIOD_END_EXIT"
            out["real"][-1] = round(st.realized_pnl, 4)
            out["cum"][-1] = round(st.realized_pnl, 4)
            out["unreal"][-1] = 0.0
            out["tpnl"][-1] = round(closed_pnl, 4)
            out["delta"][-1] = round(st.realized_pnl - before, 4)
            out["days"][-1] = st.days_held

        df_out = pd.DataFrame({
            "Date": out["dates"], "Price_A": out["pa"], "Price_B": out["pb"],
            "Hedge_Ratio": out["hr"], "ZScore": out["z"], "Position": out["pos"],
            "Unrealized_PnL": out["unreal"], "Realized_PnL": out["real"],
            "Cumulative_PnL": out["cum"], "Status": out["status"],
            "Trade_PnL": out["tpnl"], "Days_Held": out["days"], "Daily_Delta": out["delta"],
        })
        for k, v in {"Period_Start": period_start, "Period_End": period_end,
                     "Sector": sector, "Pair_Rank": pair_rank,
                     "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                     "Log_Mean_A": log_mean_a, "Log_Std_A": log_std_a,
                     "Log_Mean_B": log_mean_b, "Log_Std_B": log_std_b}.items():
            df_out[k] = v
        return df_out
