"""
RL 門檻選擇式交易模組（部分回饋對照組）
======================================================================

本模組存在的唯一理由，是把 drl_threshold_trading（DL-THR，現役）身上
「深度強化學習」這個標籤該有的東西真的做出來，好回答一個現有文獻沒問過
的問題：**反事實標籤值多少錢？**

DL-THR 每期把 9 個動作的報酬**全部**精確回算後才餵給網路（全資訊監督
回歸），因此無探索問題、樣本效率最高——它其實不是強化學習。本模組維持
完全相同的動作選單、狀態、網路與 walk-forward 切分，只改兩處：

    訓練標籤   9 個動作全部回算    →   只有實際選中的那一個
    決策       恆 argmax          →   ε-greedy

於是它成為真正的部分回饋問題（contextual bandit），與 DL-THR 構成單一
變因對照。任何績效差距只能歸因於「反事實標籤」與「探索成本」。

關於 RL 的形式歸屬（論文須明寫，勿含糊）
--------------------------------------------------------------------
這是 contextual bandit，**沒有 γ、沒有序列信用分配**。原因不是簡化：
每組配對每期只做一次決策，而 12 維狀態全部由形成期視窗算出，選哪個門檻
不會改變下一期的狀態。沒有狀態轉移就沒有東西可以 bootstrap，硬加折扣
因子是假的。它的「RL 性」來自部分回饋與探索／利用權衡（Sutton & Barto
第 2 章），而非 TD 學習。

逐日定位動作空間的三代真 RL（v1 online DQN / v2 修復版 / v3 FQI，
γ=0.99 且有 bootstrapped target）已被系統性證偽，見
archive/trading/ 與 archive/config_archived_strategies.py。

與 DL-THR 的一項結構性差異（不是缺陷，是部分回饋的固有代價）
--------------------------------------------------------------------
DL-THR 保證「訓練樣本不足時自動選基準動作 → 暖身期 ≡ Z-Score 基準」。
本模組**無法**提供這項保證：若暖身期一律選基準，就只會觀測到基準動作的
報酬，網路永遠無從得知其餘八個動作的好壞。部分回饋強迫 agent 從第一期
就得探索，暖身期因此必然偏離基準。這正是本對照要量化的東西之一。

介面與 DL-THR 相同（run_trading.py 直接可用）。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from strategies.trading.drl_threshold_trading import (
    PairState, ThresholdNet, Trading as _DLThreshold,
    _build_actions, _fast_threshold_pnl, N_FEAT,
)

torch.set_num_threads(1)


class Trading:
    """RL (contextual bandit) Threshold-Selection Strategy Interface for run_trading.py"""

    _shared: dict = {}      # {scope key -> dict(net, opt, buffer, trained_n, rng, n_decisions)}

    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex,
                 selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float,
                 drl_hidden_size: int = 64,
                 drl_lr: float = 1e-3,
                 thr_train_epochs: int = 40,
                 thr_min_train_samples: int = 200,
                 thr_menu_version: int = 4,
                 rl_epsilon: float = 0.10,          # 探索率起始值
                 rl_epsilon_final: float = None,    # None = 常數 ε；給值則線性衰減至此
                 rl_epsilon_decay_steps: int = 2000,  # 衰減走完所需的決策次數
                 rl_seed: int = None,               # None = 不固定（與 DL-THR 同慣例）
                 entry_gate: dict = None,
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

        self.eps0 = float(rl_epsilon)
        self.eps_final = None if rl_epsilon_final is None else float(rl_epsilon_final)
        self.eps_steps = max(1, int(rl_epsilon_decay_steps))
        self.seed = rl_seed

        self.formation_start = formation_start
        self.formation_end = formation_end
        self.menu_version = int(thr_menu_version)
        self.actions, self.baseline_idx = _build_actions(self.menu_version)
        self.n_actions = len(self.actions)
        self.variant_id = variant_id
        self.entry_gate = entry_gate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 形成期特徵直接沿用 DL-THR 的實作——單一變因只能落在標籤與決策上，
    # 特徵若各寫一份，日後任一邊改動就會悄悄毀掉這個對照。
    _pair_features = staticmethod(_DLThreshold._pair_features)

    def _epsilon(self, n_decisions: int) -> float:
        """常數 ε，或自 eps0 線性衰減至 eps_final。"""
        if self.eps_final is None:
            return self.eps0
        frac = min(1.0, n_decisions / self.eps_steps)
        return self.eps0 + (self.eps_final - self.eps0) * frac

    def _get_shared(self):
        # ε 併入 scope key：不同探索排程必須各自持有網路與經驗，否則三組
        # 對照會互相餵食彼此的樣本，ε 的效果就量不出來了。
        eps_tag = f"e{self.eps0:g}" + ("" if self.eps_final is None else f"-{self.eps_final:g}")
        key = f"{self.variant_id}|menu_v{self.menu_version}|{eps_tag}"
        sh = Trading._shared.get(key)
        if sh is None:
            net = ThresholdNet(N_FEAT, self.hidden, self.n_actions).to(self.device)
            sh = {"net": net, "opt": optim.Adam(net.parameters(), lr=self.lr),
                  "buffer": [], "trained_n": 0, "n_decisions": 0,
                  "rng": np.random.default_rng(self.seed), "n_explore": 0}
            Trading._shared[key] = sh
            print(f"    [RL-THR] New walk-forward bandit net ({key}, device={self.device})")
        return sh

    def _train_if_ready(self, sh: dict, trade_start):
        """
        以「交易期已於本期開始前結束」的樣本增量訓練（無前視）。

        與 DL-THR 的差別全在損失：每筆樣本只知道**一個**動作的報酬，故僅
        對該動作對應的輸出單元計損失，其餘八個單元不接收梯度。
        """
        eligible = [(f, a, r) for f, a, r, te in sh["buffer"] if te < trade_start]
        if len(eligible) < self.min_samples or len(eligible) == sh["trained_n"]:
            return len(eligible)
        X = torch.from_numpy(np.stack([e[0] for e in eligible])).to(self.device)
        A = torch.tensor([e[1] for e in eligible], dtype=torch.long, device=self.device)
        R = torch.tensor([e[2] for e in eligible], dtype=torch.float32, device=self.device)
        net, opt = sh["net"], sh["opt"]
        loss_fn = nn.MSELoss()
        net.train()
        n = len(eligible)
        bs = min(4096, n)
        for _ in range(self.train_epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                pred = net(X[idx]).gather(1, A[idx].unsqueeze(1)).squeeze(1)
                loss = loss_fn(pred, R[idx])
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

        if self.entry_gate is not None:
            gate_arr = np.array([self.entry_gate.get(ts, True) for ts in valid_idx], dtype=bool)
        else:
            gate_arr = None

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

        # ── 決策：ε-greedy ──────────────────────────────────────────────
        # 貪婪臂：訓練樣本足夠時取 net argmax，否則退回基準動作（未訓練的
        # 網路 argmax 沒有意義）。探索臂：均勻隨機。暖身期同樣要探索——
        # 不探索就只會觀測到基準動作，網路永遠學不到其餘八個。
        trade_start_ts = valid_idx[0]
        action_idx = self.baseline_idx
        explored = False
        if feats is not None:
            n_eligible = self._train_if_ready(sh, trade_start_ts)
            eps = self._epsilon(sh["n_decisions"])
            if sh["rng"].random() < eps:
                action_idx = int(sh["rng"].integers(self.n_actions))
                explored = True
                sh["n_explore"] += 1
            elif n_eligible >= self.min_samples:
                sh["net"].eval()
                with torch.no_grad():
                    q = sh["net"](torch.from_numpy(feats).unsqueeze(0).to(self.device))
                    action_idx = int(q.argmax().item())
            sh["n_decisions"] += 1

        # ── 部分回饋：只觀測選中動作的報酬（僅供「未來」期訓練） ──────────
        # 以 _fast_threshold_pnl 計算，與 DL-THR 產生標籤的路徑完全相同，
        # 差別僅在這裡只跑一個動作、那裡跑九個。SKIP 的報酬恆為 0，不需模擬。
        if feats is not None:
            act = self.actions[action_idx]
            if act is None:
                obs_ret = 0.0
            else:
                obs_ret = _fast_threshold_pnl(
                    z_t, pa_arr, pb_arr, hedge_ratio,
                    self.capital_per_pair, self.friction_rate,
                    act[0], act[1], act[2], allow=gate_arr
                ) / self.capital_per_pair * 100.0
            sh["buffer"].append((feats, action_idx, np.float32(obs_ret), valid_idx[-1]))

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
                elif (not pair_frozen) and (gate_arr is None or gate_arr[i]) and abs(zi) > ez and i < T - 1:
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
