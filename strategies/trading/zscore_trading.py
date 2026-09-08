import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategies.config import INITIAL_CAPITAL

def clean_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """遮蔽單日 50% 以上的跳動並前向遞補 —— **價格清洗的唯一定義**。

    `| _pct.isna()` 不可省（2026-08-28 修正，見 `dev/trading_arch/REVIEW.md` §E）：
    首列的 `pct_change` 是 `NaN`，而 `NaN <= 0.50` 為 `False`，故舊式把**整個
    首列**打成 `NaN`；`ffill` 無前值可用，`bfill` 遂以**第二天**的價回填。
    `zscore_window` 預設 0 時 `extended_start_idx == trade_start_idx`，首列
    就是該期第一個交易日 —— 每期第一天因此拿到隔日的價與 z。

    真正的 >50% 跳空仍被遮蔽：當日以前值遞補，隔日恢復真實價
    （`_pct` 取自原始價格，故跳空隔日的變動幅度不受遮蔽影響）。

    結果快取於來源 `DataFrame` 的 `.attrs`。原本 `distance_trading` 與
    `kalman_trading` 各自持有一份同樣的 lambda，且對**整張 6647×747 全表**
    逐配對重算一次（實測 211 ms × 5,900 次 ≈ 21 分鐘純浪費／變體）。
    以 `.attrs` 而非 `id()` 作鍵，是因為 `id` 在 GC 後會被重用。
    """
    if price_df is None:
        return None

    # pandas 會讓 .attrs 隨 .iloc 切片一起傳播，故快取必須驗明正身：
    # price_pivot 的清洗結果會被它的每一個期間切片繼承，若不檢查，
    # 一個 126 日的切片會拿到整張 6,647 日的表。
    cached = price_df.attrs.get("_cleaned_prices")
    if (cached is not None and len(cached) == len(price_df)
            and cached.index.equals(price_df.index)
            and cached.columns.equals(price_df.columns)):
        return cached

    _pct = price_df.pct_change(fill_method=None).abs()
    out = price_df.where(_pct.le(0.50) | _pct.isna()).ffill().bfill()
    out.attrs.pop("_cleaned_prices", None)     # 不讓結果自帶上一層的快取
    price_df.attrs["_cleaned_prices"] = out
    return out


@dataclass(slots=True)
class PairState:
    """單一配對在模擬過程中的內部狀態 (使用 slots 降低記憶體開銷)"""
    position: int = 0
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    prev_total_pnl: float = 0.0
    cooldown_dir: int = 0

#: 本模組是否真的實作 `hedge_mode`（dev/drl_hedge/PREREGISTRATION.md §四）。
#: `run_trading` 據此決定落庫的 `Hedge_Mode`——未宣告者一律記為 "dollar"。
SUPPORTS_HEDGE_MODE = True


class Trading:
    """負責在交易期 (Trading Period) 模擬配對交易並產生詳細交易紀錄"""
    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float, stop_loss_pct: float, entry_z: float, exit_z: float, zscore_window: int,
                 allow_reentry: bool = False, zscore_clip: float = 10.0, min_spread_std: float = 1e-6,
                 use_dynamic_stop: bool = False, dynamic_stop_z: float = 3.0,
                 portfolio_stop_loss_pct: float = 0.10, use_vol_adjust: bool = False,
                 vol_regime_threshold: float = 0.0,
                 hold_to_period_end: bool = False,
                 max_holding_days: int = 0,
                 entry_gate: dict = None,
                 hedge_mode: str = "signal",
                 execution_lag: int = 0, exec_lag_scope: str = "both"):
        # 清洗結果快取於來源 DataFrame，故逐配對重建 Trading 不會重算（見 clean_prices）
        self.price_df = clean_prices(price_df)
        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair

        self.friction_rate = fee_rate + slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.zscore_window = zscore_window
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust = use_vol_adjust
        # 收斂持有模式（GGR 式）：進場後不做 Z-Score 回歸出場，持有至期末強制平倉
        # （停損機制仍然有效）。用於測試「長持收斂」相對「快進快出均值回歸」的貢獻。
        self.hold_to_period_end = hold_to_period_end
        # 時間停損（P1，2026-07-19 接線）：持倉達 N 天仍未收斂 → 強平並凍結該
        # 配對至期末。依據交易解剖：>63d 未收斂配對勝率 26-46%、期末強平桶
        # ΣPnL 大幅為負——超時即視為均衡斷裂。0 = 停用（預設，維持既有行為）。
        self.max_holding_days = int(max_holding_days or 0)
        # P3 regime 條件化進場（2026-07-19）：entry_gate[date]=False 的日子暫停
        # 「新開倉」（持倉與出場不受影響）。gate 由 run_trading 以 walk-forward
        # 橫斷面分散度分位數預先計算（見 _build_dispersion_gate）。None = 停用。
        self.entry_gate = entry_gate

        # 執行延遲（§F，dev/exec_lag/PREREGISTRATION.md）：決策用 bar i-L 的資訊，
        # 成交仍用 bar i 的收盤價。0 = 停用，所有分支短路，結果逐位元同於接線前。
        self.execution_lag = max(0, int(execution_lag or 0))
        _scope = str(exec_lag_scope or "both").lower()
        if _scope not in ("both", "entry", "exit"):
            raise ValueError(f"exec_lag_scope 只能是 both/entry/exit，收到 {exec_lag_scope!r}")
        self.lag_entry = self.execution_lag if _scope in ("both", "entry") else 0
        self.lag_exit  = self.execution_lag if _scope in ("both", "exit")  else 0

        # 對沖權重口徑（2026-08-28，見 dev/trading_arch/PREREGISTRATION.md）：
        #   "signal"（預設）：v_A : v_B = (1/σ_A) : (β/σ_B) —— 複製訊號 spread
        #   "dollar"        ：v_A : v_B = 1 : |β|            —— 2026-08-28 前的舊行為
        # 保留 "dollar" 是為了能重現舊結果與跑 A/B 對照，不是給人選的參數。
        if hedge_mode not in ("signal", "dollar"):
            raise ValueError(f"hedge_mode 只能是 'signal' 或 'dollar'，收到 {hedge_mode!r}")
        self.hedge_mode = hedge_mode
        self._leg_w: tuple = (1.0, 1.0)   # 逐配對由 _simulate_pair 設定

        # 市場波動政體過濾：vol_regime_threshold > 0 時，年化波動率超過閾值期間暫停新開倉
        self.vol_regime_threshold = vol_regime_threshold
        if vol_regime_threshold > 0:
            _mkt_ret = self.price_df.pct_change().mean(axis=1)
            _mkt_vol = _mkt_ret.rolling(30, min_periods=10).std() * np.sqrt(252)
            self._mkt_vol_dict: dict = _mkt_vol.to_dict()
        else:
            self._mkt_vol_dict: dict = {}

        self.period_pnl: float = 0.0

    def _entry_triggered(self, z: float, z_prev: float) -> bool:
        """
        進場觸發條件。抽成方法只為了讓子類別能替換「何時進場」而不必複製
        _simulate_pair 的 310 行迴圈——本身的邏輯與抽出前完全相同。

        現行（突破式，Gatev 2006）：價差發散到帶外即進場，z_prev 不使用。
        子類別可改寫，例如 zscore_reversion_entry_trading 的
        「發散後收斂回帶內才進場」需要前一日的 z 才判得出穿越方向。
        """
        return abs(z) > self.entry_z

    def _leg_scales(self, ols_alpha, log_std_a: float, log_std_b: float,
                    log_mean_a, first_price_a: float) -> tuple[float, float]:
        """兩腳的**每單位報酬**縮放係數 $(s_A, s_B)$，逐配對於 `_simulate_pair` 求值。

        進場的美元權重為 $v_A:v_B=s_A:s_B\\lvert\\beta\\rvert$，總名目額仍為
        `capital_per_pair`。回傳 (1, 1) 即退回舊的純美元 $1:\\lvert\\beta\\rvert$。

        為何需要它（見 dev/trading_arch/PREREGISTRATION.md）：訊號 spread 定義在
        標準化空間 $\\tilde P_X=(\\ln P_X-\\mu_X)/\\sigma_X$，其變動為
        $\\Delta S=r_A/\\sigma_A-\\beta r_B/\\sigma_B$；而部位損益為
        $V_A r_A+V_B r_B$。要讓兩者成比例就必須 $V_A\\propto1/\\sigma_A$、
        $V_B\\propto\\beta/\\sigma_B$。舊寫法省略了 $1/\\sigma$，在
        $\\sigma_A\\ne\\sigma_B$ 的 40% 配對上交易的不是它自己的訊號。

        逐路徑判定（與 `_compute_spread` 的分支一一對應）：
          · 路徑 A（`ols_alpha` 非 None）：spread 在原始 log-price 空間，
            $\\Delta S=r_A-\\beta r_B$，$1:\\beta$ 本來就正確 → (1, 1)。
          · 路徑 B 的 `P/P_0` 分支：GGR 式指數，維持進場等額 → (1, 1)。
          · 路徑 B 的 $(\\ln P-\\mu)/\\sigma$ 分支：→ $(1/\\sigma_A,\\ 1/\\sigma_B)$。
        """
        if self.hedge_mode != "signal":
            return 1.0, 1.0
        if ols_alpha is not None:
            return 1.0, 1.0
        if first_price_a > 0.0 and log_mean_a is None:
            return 1.0, 1.0
        sa = max(abs(float(log_std_a)) if log_std_a else 1.0, self.min_spread_std)
        sb = max(abs(float(log_std_b)) if log_std_b else 1.0, self.min_spread_std)
        return 1.0 / sa, 1.0 / sb

    def _execute_entry(self, state: PairState, z: float, p_a: float, p_b: float, hedge_ratio: float) -> tuple[bool, float]:
        """處理進場邏輯與資金分配"""
        # _leg_scales 由 _simulate_pair 逐配對設定；(1, 1) 即舊的純美元權重。
        s_a, s_b = self._leg_w
        w_a = s_a
        w_b = s_b * abs(hedge_ratio)
        total_weight = w_a + w_b
        if not (total_weight > 0.0):
            return False, 0.0
        v_a = self.capital_per_pair * (w_a / total_weight)
        v_b = self.capital_per_pair * (w_b / total_weight)

        if z > self.entry_z:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b = v_b / p_b
        elif z < -self.entry_z:
            state.position = +1
            state.shares_a = v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state: PairState, current_trade_pnl: float, stop_loss: bool = False):
        """處理平倉與停損邏輯"""
        state.realized_pnl += current_trade_pnl

        if stop_loss:
            state.is_stopped = True if not self.allow_reentry else False

        state.position = 0
        state.shares_a = 0.0
        state.shares_b = 0.0
        state.entry_price_a = 0.0
        state.entry_price_b = 0.0
        state.trade_entry_fee = 0.0

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
        """
        計算 spread 與 Z-Score。子類別可覆寫此方法替換成不同的 spread 估計方式（如 Kalman Filter）。
        回傳 (zscore, beta_series)，皆以 common_idx 為索引；遇到退化情況時回傳兩個空 Series。
        """
        _empty = pd.Series(dtype=float)

        # ── 路徑 A：OLS log-price 空間（HDBSCAN 系列） ──────────────────────
        if ols_alpha is not None:
            log_p_a = np.log(price_a)
            log_p_b = np.log(price_b)

            if self.zscore_window == 0:
                spread = log_p_a - ols_alpha - hedge_ratio * log_p_b
                safe_std = max(form_spread_std, self.min_spread_std)
                if self.use_vol_adjust:
                    roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                    vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                    adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
                else:
                    adjusted_std = safe_std
                zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
                beta_series = pd.Series(hedge_ratio, index=common_idx)
            else:
                roll_cov = log_p_b.rolling(window=self.zscore_window).cov(log_p_a)
                roll_var = log_p_b.rolling(window=self.zscore_window).var()
                roll_beta = np.where(roll_var > 1e-8, roll_cov / roll_var, 0.0)
                roll_beta = pd.Series(roll_beta, index=common_idx)
                roll_mean_a = log_p_a.rolling(window=self.zscore_window).mean()
                roll_mean_b = log_p_b.rolling(window=self.zscore_window).mean()
                roll_alpha = roll_mean_a - roll_beta * roll_mean_b
                spread = log_p_a - roll_alpha - roll_beta * log_p_b
                roll_var_a = log_p_a.rolling(window=self.zscore_window).var()
                roll_res_var = roll_var_a * (1 - (roll_cov ** 2 / (roll_var * roll_var_a + 1e-12)))
                roll_std = np.sqrt(np.maximum(roll_res_var, 0))
                if (roll_std < self.min_spread_std * 10).mean() > 0.5:
                    return _empty, _empty
                safe_std = np.maximum(roll_std, self.min_spread_std)
                if self.use_vol_adjust:
                    roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                    vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                    adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
                else:
                    adjusted_std = safe_std
                zscore = np.clip(spread / adjusted_std, -self.zscore_clip, self.zscore_clip)
                beta_series = roll_beta

        # ── 路徑 B：標準化 norm_p 空間（SSD / DTW，向下相容） ────────────────
        else:
            if first_price_a > 0.0 and log_mean_a is None:
                norm_p_a = price_a / first_price_a
                norm_p_b = price_b / first_price_b
            else:
                log_p_a = np.log(price_a)
                log_p_b = np.log(price_b)
                norm_p_a = (log_p_a - log_mean_a) / log_std_a
                norm_p_b = (log_p_b - log_mean_b) / log_std_b

            if self.zscore_window == 0:
                spread = norm_p_a - hedge_ratio * norm_p_b
                safe_std = max(form_spread_std, self.min_spread_std)
                if self.use_vol_adjust:
                    roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                    vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                    adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
                else:
                    adjusted_std = safe_std
                zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
                beta_series = pd.Series(hedge_ratio, index=common_idx)
            else:
                roll_cov = norm_p_b.rolling(window=self.zscore_window).cov(norm_p_a)
                roll_var = norm_p_b.rolling(window=self.zscore_window).var()
                roll_beta = np.where(roll_var > 1e-8, roll_cov / roll_var, 0.0)
                roll_beta = pd.Series(roll_beta, index=common_idx)
                roll_mean_a = norm_p_a.rolling(window=self.zscore_window).mean()
                roll_mean_b = norm_p_b.rolling(window=self.zscore_window).mean()
                roll_alpha = roll_mean_a - roll_beta * roll_mean_b
                spread = norm_p_a - roll_alpha - roll_beta * norm_p_b
                roll_var_a = norm_p_a.rolling(window=self.zscore_window).var()
                roll_res_var = roll_var_a * (1 - (roll_cov ** 2 / (roll_var * roll_var_a + 1e-12)))
                roll_std = np.sqrt(np.maximum(roll_res_var, 0))
                if (roll_std < self.min_spread_std * 10).mean() > 0.5:
                    return _empty, _empty
                safe_std = np.maximum(roll_std, self.min_spread_std)
                if self.use_vol_adjust:
                    roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                    vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                    adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
                else:
                    adjusted_std = safe_std
                zscore = np.clip(spread / adjusted_std, -self.zscore_clip, self.zscore_clip)
                beta_series = roll_beta

        return zscore, beta_series

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float,
                       form_spread_mean: float, form_spread_std: float, log_mean_a=None, log_std_a: float = 1.0, log_mean_b=None, log_std_b: float = 1.0,
                       first_price_a: float = 0.0, first_price_b: float = 0.0,
                       ols_alpha: float = None) -> pd.DataFrame:
        """
        ols_alpha: OLS 截距，由 HDBSCAN 等 OLS 形成期傳入。
          - 若非 None：使用原始 log-price 空間計算 spread（與形成期一致）。
          - 若 None：使用標準化 norm_p 空間（SSD / DTW 等舊路徑，保持向下相容）。
        """
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: return pd.DataFrame()

        zscore, beta_series = self._compute_spread(
            price_a, price_b, common_idx,
            hedge_ratio, form_spread_mean, form_spread_std,
            log_mean_a, log_std_a, log_mean_b, log_std_b,
            first_price_a, first_price_b, ols_alpha,
        )
        if zscore.empty: return pd.DataFrame()

        # 兩腳的每單位報酬縮放係數，供 _execute_entry 分配資金。逐配對求一次
        # （σ 來自形成期，整段交易期不變；β 則可能逐日變動，故留在 _execute_entry）。
        self._leg_w = self._leg_scales(ols_alpha, log_std_a, log_std_b,
                                       log_mean_a, first_price_a)

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0: return pd.DataFrame()

        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        zscore = zscore.loc[valid_idx]
        beta_series = beta_series.loc[valid_idx]

        dates_arr = valid_idx
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values
        beta_arr = beta_series.values

        state = PairState()

        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_z, out_pos = [], [], []
        out_unrealized, out_realized, out_cum = [], [], []
        out_status, out_trade_pnl, out_days, out_delta = [], [], [], []

        # §F：決策索引恆為 i-L，成交索引恆為 i。L=0 時 j == i，下方全部退化為
        # 接線前的寫法（無分支、無特例），這正是 P1「逐位元相同」的依據。
        def _zat(k: int) -> float:
            """bar k 的 z；k < 0（延遲期尚無訊號）或 NaN 一律視為 0.0。"""
            if k < 0 or np.isnan(zscore_arr[k]):
                return 0.0
            return zscore_arr[k]

        for i in range(len(dates_arr)):
            date = dates_arr[i]
            z = 0.0 if np.isnan(zscore_arr[i]) else zscore_arr[i]
            p_a, p_b = pa_arr[i], pb_arr[i]

            # 決策讀 bar j_en / j_ex，成交價恆為 p_a / p_b（bar i）。
            j_en, j_ex = i - self.lag_entry, i - self.lag_exit
            z_en, z_ex = _zat(j_en), _zat(j_ex)
            # 前一日 z，僅供 _entry_triggered 的子類別判斷穿越方向；本類別不使用。
            # 首日與 NaN 一律視為 0.0，與 z 本身的處理一致。
            z_prev = _zat(j_en - 1)
            # 資本停損的判定價（bar j_ex）；L=0 時即 p_a/p_b。
            if self.lag_exit == 0:
                pa_d, pb_d = p_a, p_b
            else:
                k = max(j_ex, 0)
                pa_d, pb_d = pa_arr[k], pb_arr[k]

            c_beta = beta_arr[i] if not np.isnan(beta_arr[i]) else hedge_ratio

            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0
            daily_delta = 0.0
            current_status = "HOLD_CASH"

            if state.is_stopped:
                out_dates.append(date)
                out_pa.append(p_a)
                out_pb.append(p_b)
                out_hr.append(float(c_beta))
                out_z.append(float(z))
                out_pos.append(0)
                out_unrealized.append(0.0)
                out_realized.append(float(state.realized_pnl))
                out_cum.append(float(state.realized_pnl))
                out_status.append("STOPPED")
                out_trade_pnl.append(0.0)
                out_days.append(0)
                out_delta.append(0.0)
                continue

            if state.position != 0:
                state.days_held += 1
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate

                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est

                # §F：出場側的三個判定一律讀 bar j_ex；成交仍以 current_trade_pnl
                # （bar i）記帳。L=0 時 decide_trade_pnl 恆等於 current_trade_pnl。
                if self.lag_exit == 0:
                    decide_trade_pnl = current_trade_pnl
                else:
                    raw_d = state.shares_a * (pa_d - state.entry_price_a) + state.shares_b * (pb_d - state.entry_price_b)
                    fee_d = (abs(state.shares_a)*pa_d + abs(state.shares_b)*pb_d) * self.friction_rate
                    decide_trade_pnl = raw_d - state.trade_entry_fee - fee_d
                # 延遲期內（j_ex < 0）尚無可用訊號，不做出場決策
                can_exit = j_ex >= 0

                is_cap_stop = can_exit and self.stop_loss_pct > 0 and (-decide_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = can_exit and self.use_dynamic_stop and self.dynamic_stop_z > 0 and abs(z_ex) > self.dynamic_stop_z
                # P1 時間停損：持倉達 max_holding_days 仍未收斂 → 視為均衡斷裂
                # （延遲下於 bar j_ex 判定，故持有天數同步後移）
                is_time_stop = can_exit and self.max_holding_days > 0 and (state.days_held - self.lag_exit) >= self.max_holding_days

                if is_cap_stop or is_z_stop or is_time_stop:
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "TIME_STOP" if (is_time_stop and not (is_cap_stop or is_z_stop)) else "STOP_LOSS_TRIGGERED"
                else:
                    if self.hold_to_period_end:
                        # 收斂持有模式：不做 Z-Score 出場，持有至期末（PERIOD_END_EXIT 結算）
                        is_exit_short = is_exit_long = False
                    else:
                        is_exit_short = can_exit and (state.position == -1) and (z_ex <= self.exit_z)
                        is_exit_long  = can_exit and (state.position == 1)  and (z_ex >= -self.exit_z)

                    if is_exit_short or is_exit_long:
                        self._execute_close(state, current_trade_pnl, stop_loss=False)
                        closed_trade_pnl = current_trade_pnl
                        current_status = "EXIT"
                    else:
                        unrealized_pnl = current_trade_pnl
                        current_status = "HOLDING"
            else:
                # 市場波動政體過濾：年化波動率超過閾值時暫停新開倉
                in_high_vol = (
                    self.vol_regime_threshold > 0
                    and self._mkt_vol_dict.get(date, 0.0) > self.vol_regime_threshold
                )
                # P3：低分散度 regime 暫停新開倉
                gate_blocked = (
                    self.entry_gate is not None
                    and not self.entry_gate.get(date, True)
                )
                # §F：延遲期內（j_en < 0）尚無可用訊號，不做進場決策
                can_enter = j_en >= 0
                if not in_high_vol and not gate_blocked and can_enter and self._entry_triggered(z_en, z_prev):
                    entered, unrealized_pnl = self._execute_entry(state, z_en, p_a, p_b, c_beta)
                    if entered:
                        current_status = "ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A"
                    else:
                        current_status = "HOLD_CASH (COOLDOWN)"
                elif in_high_vol:
                    current_status = "HOLD_CASH (HIGH_VOL_REGIME)"
                elif gate_blocked and can_enter and self._entry_triggered(z_en, z_prev):
                    current_status = "HOLD_CASH (LOW_DISP_GATE)"
                else:
                    current_status = "HOLD_CASH"

            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl

            out_dates.append(date)
            out_pa.append(p_a)
            out_pb.append(p_b)
            out_hr.append(float(c_beta))
            out_z.append(float(z))
            out_pos.append(state.position)
            out_unrealized.append(float(unrealized_pnl))
            out_realized.append(float(state.realized_pnl))
            out_cum.append(float(cumulative_pnl))
            out_status.append(current_status)
            out_trade_pnl.append(float(closed_trade_pnl))
            out_days.append(state.days_held)
            out_delta.append(float(daily_delta))

            if current_status in ["STOP_LOSS_TRIGGERED", "TIME_STOP", "EXIT"]:
                state.days_held = 0

            if state.is_stopped and i < len(dates_arr) - 1:
                # Bulk-fill remaining STOPPED rows without a Python for-loop
                remain = slice(i + 1, len(dates_arr))
                n_remain = len(dates_arr) - (i + 1)
                final_realized = float(state.realized_pnl)
                out_dates.extend(dates_arr[remain])
                out_pa.extend(pa_arr[remain].tolist())
                out_pb.extend(pb_arr[remain].tolist())
                out_hr.extend(beta_arr[remain].tolist())
                out_z.extend(np.where(np.isnan(zscore_arr[remain]), 0.0, zscore_arr[remain]).tolist())
                out_pos.extend([0] * n_remain)
                out_unrealized.extend([0.0] * n_remain)
                out_realized.extend([final_realized] * n_remain)
                out_cum.extend([final_realized] * n_remain)
                out_status.extend(["STOPPED"] * n_remain)
                out_trade_pnl.extend([0.0] * n_remain)
                out_days.extend([0] * n_remain)
                out_delta.extend([0.0] * n_remain)
                break

        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                pnl_before_last_day = out_cum[-2] if len(out_cum) > 1 else 0.0

                p_a_last, p_b_last = pa_arr[-1], pb_arr[-1]
                raw_unrealized_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                exit_fee = (abs(state.shares_a)*p_a_last + abs(state.shares_b)*p_b_last) * self.friction_rate

                closed_trade_pnl = raw_unrealized_final - state.trade_entry_fee - exit_fee
                state.realized_pnl += closed_trade_pnl
                daily_delta = state.realized_pnl - pnl_before_last_day

                out_status[-1] = "PERIOD_END_EXIT"
                out_realized[-1] = float(state.realized_pnl)
                out_cum[-1] = float(state.realized_pnl)
                out_unrealized[-1] = 0.0
                out_trade_pnl[-1] = float(closed_trade_pnl)
                out_delta[-1] = float(daily_delta)
                out_days[-1] = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unrealized, "Realized_PnL": out_realized,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_trade_pnl, "Days_Held": out_days, "Daily_Delta": out_delta
        })
        _round_cols = ["Price_A", "Price_B", "Hedge_Ratio", "ZScore",
                       "Unrealized_PnL", "Realized_PnL", "Cumulative_PnL",
                       "Trade_PnL", "Daily_Delta"]
        df_out[_round_cols] = df_out[_round_cols].round(4)

        base_log = {
            "Period_Start": period_start, "Period_End": period_end,
            "Sector": sector, "Pair_Rank": pair_rank,
            "Ticker_A": ticker_a, "Ticker_B": ticker_b,
        }
        if first_price_a > 0.0 and (log_mean_a is None or log_mean_a == 0.0):
            base_log["First_Price_A"] = first_price_a
            base_log["First_Price_B"] = first_price_b
        else:
            base_log["Log_Mean_A"] = log_mean_a
            base_log["Log_Std_A"] = log_std_a
            base_log["Log_Mean_B"] = log_mean_b
            base_log["Log_Std_B"] = log_std_b

        for k, v in base_log.items():
            df_out[k] = v

        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """執行該期所有配對的交易模擬"""
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            raw_alpha = row.get("OLS_Alpha", None)
            ols_alpha_val = float(raw_alpha) if raw_alpha is not None and not pd.isna(raw_alpha) else None
            df_pair = self._simulate_pair(
                period_start=period_start,
                period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"],
                ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A", 1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B", 1.0)),
                first_price_a=float(row.get("First_Price_A", 0.0)),
                first_price_b=float(row.get("First_Price_B", 0.0)),
                ols_alpha=ols_alpha_val
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # ── 投資組合總體止損斷路器 ──
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()

            breach = daily_cum_pnl[daily_cum_pnl / total_cap <= -self.portfolio_stop_loss_pct]
            cutoff_date = breach.index[0] if not breach.empty else None

            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date

                    df_before = df[before_mask]

                    df_at = df[at_mask].copy()
                    final_realized = float(df_before["Realized_PnL"].iloc[-1]) if not df_before.empty else 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        if row_at["Position"] != 0:
                            final_realized = row_at["Realized_PnL"] + row_at["Unrealized_PnL"]
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Unrealized_PnL"]
                            df_at.loc[df_at.index, "Realized_PnL"] = final_realized
                            df_at.loc[df_at.index, "Cumulative_PnL"] = final_realized
                            df_at.loc[df_at.index, "Daily_Delta"] = 0.0
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

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily_delta = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily_delta.sum()) if not period_daily_delta.empty else 0.0

        return log_df, self.period_pnl
