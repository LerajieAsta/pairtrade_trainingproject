# -*- coding: utf-8 -*-
"""驗證 dev/drl_hedge/PREREGISTRATION.md 的 P1–P4。

P3 是本檔的重點：**對每一個交易模組**斷言「切換 hedge_mode 必須改變輸出」。
這一層是 §A 與本案兩次漏修都缺的——前兩次都靠人工讀 config 判斷，
而 config 說了不算。

用法：python -m dev.drl_hedge.verify_hedge_modes
"""
import importlib
import inspect
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from strategies.trading.drl_threshold_trading import PairState, _fast_threshold_pnl

SCALE_A, SCALE_B = 0.08, 0.20      # sigma_A != sigma_B，否則兩口徑恰好重合
BETA = 1.3
CAP, FRIC = 10_000.0, 0.0029

MODULES = [
    "strategies.trading.zscore_trading",
    "strategies.trading.distance_trading",
    "strategies.trading.kalman_trading",
    "strategies.trading.zscore_reversion_entry_trading",
    "strategies.trading.drl_threshold_trading",
    "strategies.trading.rl_threshold_trading",
]


def _series(n=140, seed=0):
    """B 為隨機游走；A = B 的 beta 倍再疊一條均值回歸的殘差。

    兩腳的 log 波動刻意不同（SCALE_A / SCALE_B），這是 signal 與 dollar
    會分岔的**必要條件**——sigma_A = sigma_B 時兩個口徑恰好相同，測不出東西。
    """
    rng = np.random.default_rng(seed)
    eb = np.cumsum(rng.normal(0, 1, n)) * 0.01
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.94 * r[i - 1] + rng.normal(0, 1) * 0.4
    la = BETA * eb * SCALE_A / SCALE_B + r * SCALE_A
    return np.exp(la), np.exp(eb), r


def _old_fast_pnl(z, pa, pb, hedge_ratio, capital, friction, ez, xz):
    """2026-09-02 之前的 `_fast_threshold_pnl` 資金配置，逐行照抄。

    P1 要證的是「新程式在 dollar 口徑下與舊程式逐位元相同」，
    故舊公式必須在此獨立存在一份，不能呼叫新程式來驗自己。
    """
    tw = 1.0 + abs(hedge_ratio)
    st = PairState()
    T = len(z)
    for i in range(T):
        zi, p_a, p_b = z[i], pa[i], pb[i]
        if st.position != 0:
            st.days_held += 1
            is_exit = (st.position == -1 and zi <= xz) or (st.position == 1 and zi >= -xz)
            if is_exit or i == T - 1:
                raw = (st.shares_a * (p_a - st.entry_price_a)
                       + st.shares_b * (p_b - st.entry_price_b))
                fee = (abs(st.shares_a) * p_a + abs(st.shares_b) * p_b) * friction
                st.realized_pnl += raw - st.trade_entry_fee - fee
                st.position = 0
                st.shares_a = st.shares_b = 0.0
                st.trade_entry_fee = 0.0
                st.days_held = 0
                continue
        elif abs(zi) > ez and i < T - 1:
            v_a = capital / tw
            v_b = capital * abs(hedge_ratio) / tw
            if zi > ez:
                st.position, st.shares_a, st.shares_b = -1, -v_a / p_a, v_b / p_b
            else:
                st.position, st.shares_a, st.shares_b = 1, v_a / p_a, -v_b / p_b
            st.entry_price_a, st.entry_price_b = p_a, p_b
            st.trade_entry_fee = (abs(st.shares_a) * p_a
                                  + abs(st.shares_b) * p_b) * friction
    return st.realized_pnl


def check_p1(fails):
    print("P1  dollar 口徑 vs 改動前公式（逐行照抄的獨立實作）")
    n_ok = 0
    for seed in range(8):
        pa, pb, z = _series(seed=seed)
        new = _fast_threshold_pnl(z, pa, pb, BETA, CAP, FRIC, 2.0, 0.0,
                                  leg_scales=(1.0, 1.0))
        old = _old_fast_pnl(z, pa, pb, BETA, CAP, FRIC, 2.0, 0.0)
        if abs(new - old) > 1e-12:
            fails.append(f"P1 seed={seed}：新 {new!r} != 舊 {old!r}")
        else:
            n_ok += 1
    print(f"    {n_ok}/8 組 seed 逐位元相同  ->  {'OK' if n_ok == 8 else 'FAIL'}")


def check_p1b(fails):
    pa, pb, z = _series()
    sig = _fast_threshold_pnl(z, pa, pb, BETA, CAP, FRIC, 2.0, 0.0,
                              leg_scales=(1.0 / SCALE_A, 1.0 / SCALE_B))
    dol = _fast_threshold_pnl(z, pa, pb, BETA, CAP, FRIC, 2.0, 0.0,
                              leg_scales=(1.0, 1.0))
    same = abs(sig - dol) < 1e-9
    print(f"P1b signal 與 dollar 必須不同：{sig:.4f} vs {dol:.4f}"
          f"  ->  {'FAIL（P1 是假通過）' if same else 'OK'}")
    if same:
        fails.append("P1b：兩口徑結果相同，代表 leg_scales 根本沒被使用")


def check_p2_p4(fails):
    pa, pb, z = _series()

    # P2a：曝險比 —— 這一項是**恆等式**，可以要求逐位元相等。
    # signal 口徑的定義就是 v_A : v_B = (1/sigma_A) : (beta/sigma_B)。
    print("P2a 兩腳金額曝險比（恆等式，要求逐位元）")
    for tag, (s_a, s_b), want in (
            ("signal", (1.0 / SCALE_A, 1.0 / SCALE_B),
             (1.0 / SCALE_A) / (BETA / SCALE_B)),
            ("dollar", (1.0, 1.0), 1.0 / BETA)):
        tw = s_a + s_b * abs(BETA)
        v_a, v_b = CAP * s_a / tw, CAP * s_b * abs(BETA) / tw
        got = v_a / v_b
        ok = abs(got / want - 1.0) < 1e-12
        print(f"    {tag:<7} v_A/v_B = {got:.12f}  應為 {want:.12f}"
              f"  ->  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"P2a {tag}：曝險比 {got} != {want}")

    # P2b：部位損益與 -dS 的相關。
    #
    # **不能要求恰為 1.0**：部位損益是**價格空間**的線性量
    #   PnL = sh_A * dP_A + sh_B * dP_B
    # 而 spread 是**對數空間**的線性量
    #   S = (ln P_A - mu_A)/sigma_A - beta * (ln P_B - mu_B)/sigma_B
    # 兩者只在一階近似下等價（d ln P ~= dP/P），故長路徑上必然 < 1。
    # 本檔第一版誤設判準為 > 0.999999，signal 實得 0.9986 被判為失敗——
    # 那是判準錯，不是程式錯。改為要求「signal 明顯優於 dollar」。
    print("P2b 部位損益與 -dS 的相關（價格 vs 對數空間，一階近似）")
    got = {}
    for tag, (s_a, s_b) in (("signal", (1.0 / SCALE_A, 1.0 / SCALE_B)),
                            ("dollar", (1.0, 1.0))):
        tw = s_a + s_b * abs(BETA)
        v_a, v_b = CAP * s_a / tw, CAP * s_b * abs(BETA) / tw
        sh_a, sh_b = -v_a / pa[0], v_b / pb[0]          # 空 spread
        pnl = sh_a * (pa - pa[0]) + sh_b * (pb - pb[0])
        got[tag] = float(np.corrcoef(pnl, -(z - z[0]))[0, 1])
        print(f"    {tag:<7} corr = {got[tag]:.10f}")
    ok = got["signal"] > 0.99 and got["signal"] - got["dollar"] > 0.05
    print(f"    signal > 0.99 且領先 dollar > 0.05  ->  {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"P2b：signal {got['signal']:.6f} / dollar {got['dollar']:.6f}")

    print("P4  v_A + v_B = C")
    for tag, (s_a, s_b) in (("signal", (1.0 / SCALE_A, 1.0 / SCALE_B)),
                            ("dollar", (1.0, 1.0))):
        tw = s_a + s_b * abs(BETA)
        tot = CAP * s_a / tw + CAP * s_b * abs(BETA) / tw
        ok = abs(tot / CAP - 1.0) < 1e-12
        print(f"    {tag:<7} 總額/C = {tot / CAP:.15f}  ->  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"P4 {tag}：總名目額 {tot} != {CAP}")


def _run_module(mod, modname, prices, idx, hm):
    kw = dict(price_df=prices, trade_dates=idx, selected_pairs=pd.DataFrame(),
              capital_per_pair=CAP, fee_rate=FRIC, slippage_rate=0.0,
              hedge_mode=hm, full_price_df=prices,
              formation_start=str(idx[0].date()), formation_end=str(idx[-1].date()),
              variant_id=f"verify_{hm}", rl_seed=7,
              stop_loss_pct=0.0, entry_z=2.0, exit_z=0.0, zscore_window=0)
    # 各模組的簽章不同（zscore 沒有 full_price_df、DRL 沒有 entry_z…），
    # 且未必都收 **kwargs——照簽章過濾，與 run_trading 的 _valid_kwargs_keys 同慣例。
    sig = inspect.signature(mod.Trading.__init__)
    if not any(q.kind is inspect.Parameter.VAR_KEYWORD for q in sig.parameters.values()):
        kw = {k: v for k, v in kw.items() if k in sig.parameters}
    t = mod.Trading(**kw)
    df = t._simulate_pair(
        period_start=str(idx[0].date()), period_end=str(idx[-1].date()),
        sector="X", ticker_a="A", ticker_b="B", pair_rank=1,
        hedge_ratio=BETA, form_spread_mean=0.0, form_spread_std=1.0,
        log_mean_a=0.0, log_std_a=SCALE_A, log_mean_b=0.0, log_std_b=SCALE_B,
        first_price_a=0.0, first_price_b=0.0, ols_alpha=None)
    if df is None or df.empty:
        return None
    col = "Cumulative_PnL" if "Cumulative_PnL" in df.columns else df.columns[-1]
    return [round(float(v), 6) for v in df[col].tolist()]


def check_p3(fails):
    print()
    print("P3  逐模組行為測試：切換 hedge_mode 必須改變輸出")
    print("    （宣告 SUPPORTS_HEDGE_MODE=True 卻沒真的用 -> 在此失敗）")
    pa, pb, _ = _series()
    idx = pd.bdate_range("2020-01-01", periods=len(pa))
    prices = pd.DataFrame({"A": pa, "B": pb}, index=idx)
    worst = prices.pct_change().abs().max().max()
    assert worst < 0.50, f"合成價單日變動 {worst:.1%} 會觸發 clean_prices 濾除"

    for modname in MODULES:
        mod = importlib.import_module(modname)
        declared = bool(getattr(mod, "SUPPORTS_HEDGE_MODE", False))
        short = modname.rsplit(".", 1)[-1]
        outs = {}
        err = None
        for hm in ("signal", "dollar"):
            try:
                outs[hm] = _run_module(mod, modname, prices, idx, hm)
            except Exception as exc:               # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                break
        if err:
            print(f"    {short:<32} FAIL  {err}")
            fails.append(f"P3 {short}：模擬失敗 {err}")
        elif outs["signal"] is None or outs["dollar"] is None:
            print(f"    {short:<32} SKIP  無交易，測不出")
            fails.append(f"P3 {short}：合成資料未產生交易，測試無效")
        elif outs["signal"] == outs["dollar"]:
            print(f"    {short:<32} FAIL  兩口徑輸出相同（宣告={declared}）")
            fails.append(f"P3 {short}：切換 hedge_mode 未改變輸出")
        else:
            print(f"    {short:<32} OK    輸出不同（宣告={declared}）")


def main() -> int:
    fails = []
    check_p1(fails)
    check_p1b(fails)
    check_p2_p4(fails)
    check_p3(fails)
    print()
    if fails:
        print(f"FAIL {len(fails)} 項：")
        for f in fails:
            print("   " + f)
        return 1
    print("P1–P4 全部通過。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
