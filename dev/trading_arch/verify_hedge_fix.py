# -*- coding: utf-8 -*-
"""A 修正的驗證：逐條檢查 `PREREGISTRATION.md` §四的 P1–P5。

用法：
    python dev/trading_arch/verify_hedge_fix.py

P1  路徑 A 與 ggr_index 臂的權重逐位不變
P2  進出場判定不受影響（同一組價格 → 同一組 Status 序列）
P3  總名目額仍為 capital_per_pair
P4/P5  執行組合 == 複製組合（本案的成敗）
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from strategies.trading.distance_trading import Trading as DistTrading  # noqa: E402
from strategies.trading.zscore_trading import Trading  # noqa: E402

CAP = 10_000.0
FEE = 0.0029

_fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    if not ok:
        _fails.append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  " + detail) if detail else ""))


def make(cls=Trading, hedge_mode="signal", **extra):
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(7)
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
    b = 50 * np.exp(np.cumsum(rng.normal(0, 0.008, 60)))     # 波動小很多 → σ_B << σ_A
    px = pd.DataFrame({"AAA": a, "BBB": b}, index=idx)
    kw = dict(price_df=px, trade_dates=idx, selected_pairs=pd.DataFrame(),
              capital_per_pair=CAP, fee_rate=FEE, slippage_rate=0.0,
              stop_loss_pct=0.0, entry_z=2.0, exit_z=0.0, zscore_window=0,
              hedge_mode=hedge_mode)
    kw.update(extra)
    return cls(**kw), px


def leg_dollars(t, beta, sa, sb, ols_alpha=None, log_mean_a=0.0, first_price_a=0.0):
    """跑一次 _leg_scales + _execute_entry，回傳兩腳的美元曝險。"""
    from strategies.trading.zscore_trading import PairState
    t._leg_w = t._leg_scales(ols_alpha, sa, sb, log_mean_a, first_price_a)
    st = PairState()
    ok, _ = t._execute_entry(st, z=3.0, p_a=100.0, p_b=50.0, hedge_ratio=beta)
    assert ok
    return abs(st.shares_a) * 100.0, abs(st.shares_b) * 50.0


def main() -> None:
    print("== P1  不該動的路徑逐位不變 ==")
    t_sig, _ = make(hedge_mode="signal")
    t_dol, _ = make(hedge_mode="dollar")
    beta, sa, sb = 0.8, 0.30, 0.12

    # 路徑 A：ols_alpha 非 None
    va_s, vb_s = leg_dollars(t_sig, beta, sa, sb, ols_alpha=0.5)
    va_d, vb_d = leg_dollars(t_dol, beta, sa, sb, ols_alpha=0.5)
    check("路徑 A（OLS log-price）權重不變",
          np.isclose(va_s, va_d) and np.isclose(vb_s, vb_d),
          f"signal=({va_s:.2f}, {vb_s:.2f})  dollar=({va_d:.2f}, {vb_d:.2f})")

    # 路徑 B 的 P/P0 分支：log_mean_a is None 且 first_price_a > 0
    va_s, vb_s = leg_dollars(t_sig, beta, sa, sb, log_mean_a=None, first_price_a=90.0)
    va_d, vb_d = leg_dollars(t_dol, beta, sa, sb, log_mean_a=None, first_price_a=90.0)
    check("路徑 B 的 P/P0 分支權重不變",
          np.isclose(va_s, va_d) and np.isclose(vb_s, vb_d),
          f"({va_s:.2f}, {vb_s:.2f})")

    # distance_trading 的 ggr_index
    tg, _ = make(DistTrading, hedge_mode="signal", normalize_mode="ggr_index",
                 full_price_df=None, formation_start=None, formation_end=None)
    va_g, vb_g = leg_dollars(tg, 1.0, sa, sb)
    check("distance / ggr_index 維持進場等額",
          np.isclose(va_g, vb_g) and np.isclose(va_g, CAP / 2),
          f"({va_g:.2f}, {vb_g:.2f})")

    print("\n== P3  總名目額不變 ==")
    for name, tt, bb in (("zscore signal", t_sig, beta), ("zscore dollar", t_dol, beta)):
        va, vb = leg_dollars(tt, bb, sa, sb)
        check(f"{name}：v_A + v_B == capital_per_pair",
              np.isclose(va + vb, CAP), f"{va + vb:,.4f}")

    print("\n== 權重確實改成 1/σ 口徑 ==")
    va_s, vb_s = leg_dollars(t_sig, beta, sa, sb)
    va_d, vb_d = leg_dollars(t_dol, beta, sa, sb)
    want = (1 / sa) / ((1 / sa) + beta / sb)
    check("signal 模式的 v_A 佔比 == (1/σ_A)/((1/σ_A)+β/σ_B)",
          np.isclose(va_s / CAP, want), f"{va_s / CAP:.6f} vs {want:.6f}")
    check("dollar 模式的 v_A 佔比 == 1/(1+β)",
          np.isclose(va_d / CAP, 1 / (1 + beta)), f"{va_d / CAP:.6f}")

    print("\n== P4/P5  執行組合 == 複製組合 ==")
    # 訊號 spread S = (lnA−μ_A)/σ_A − β(lnB−μ_B)/σ_B
    # 部位損益應與 ΔS 成正比。以隨機報酬檢查比例是否為常數。
    rng = np.random.default_rng(11)
    for beta, sa, sb in ((0.8, 0.30, 0.12), (1.0, 0.10, 0.45), (0.5, 0.22, 0.22)):
        for mode, tt in (("signal", t_sig), ("dollar", t_dol)):
            va, vb = leg_dollars(tt, beta, sa, sb)      # z=3 → 空 A 多 B
            r_a = rng.normal(0, 0.02, 4000)
            r_b = rng.normal(0, 0.02, 4000)
            pnl = -va * r_a + vb * r_b                  # 空 A、多 B
            dS = r_a / sa - beta * r_b / sb             # 訊號 spread 的變動
            # 空頭部位獲利於 spread 收斂（dS < 0），故取 −dS 比對
            rho = float(np.corrcoef(pnl, -dS)[0, 1])
            tag = f"β={beta} σ_A={sa} σ_B={sb}"
            if mode == "signal":
                check(f"signal：損益與 −ΔS 完全共線  {tag}", np.isclose(rho, 1.0, atol=1e-9),
                      f"ρ={rho:.10f}")
            else:
                if not np.isclose(sa, sb):
                    check(f"dollar：σ_A≠σ_B 時不共線  {tag}", rho < 0.9999,
                          f"ρ={rho:.6f}")

    print("\n== P2  進出場判定不受影響 ==")
    # 走完整的 _simulate_pair：z 由 _compute_spread 產生，不經 _leg_w，
    # 故 Status 序列應逐位相同（停損 0% 時）。停損 > 0 時損益會改變觸發點，
    # 屬預期內的差異，此處一併量出來。
    # 刻意造一條「先發散、再收斂」的配對，確保真的會進出場一次以上。
    n = 80
    idx2 = pd.date_range("2020-01-01", periods=n, freq="B")
    bump = np.concatenate([np.zeros(15), np.linspace(0, 0.25, 20),
                           np.linspace(0.25, 0, 20), np.zeros(n - 55)])
    rng2 = np.random.default_rng(3)
    base = np.cumsum(rng2.normal(0, 0.004, n))
    px2 = pd.DataFrame({"AAA": 100 * np.exp(base + bump),
                        "BBB": 50 * np.exp(base)}, index=idx2)

    for sl, expect_same in ((0.0, True), (0.05, False)):
        logs = {}
        for mode in ("signal", "dollar"):
            t = Trading(price_df=px2, trade_dates=idx2, selected_pairs=pd.DataFrame(),
                        capital_per_pair=CAP, fee_rate=FEE, slippage_rate=0.0,
                        stop_loss_pct=sl, entry_z=2.0, exit_z=0.0, zscore_window=0,
                        hedge_mode=mode)
            logs[mode] = t._simulate_pair(
                period_start="2020-01-01", period_end="2020-04-21",
                sector="X", ticker_a="AAA", ticker_b="BBB", pair_rank=1,
                hedge_ratio=0.8, form_spread_mean=0.0, form_spread_std=0.05,
                log_mean_a=4.6, log_std_a=0.30, log_mean_b=3.9, log_std_b=0.12)
        s1, s2 = logs["signal"]["Status"], logs["dollar"]["Status"]
        n_ent = int(s1.str.startswith("ENTER").sum())
        same = s1.equals(s2)
        check(f"SL={sl:.0%}：測試資料確實觸發進場（否則比較無意義）", n_ent > 0,
              f"進場 {n_ent} 次")
        if expect_same:
            check(f"SL={sl:.0%}：Status 序列逐位相同", same, f"進場 {n_ent} 次")
            check(f"SL={sl:.0%}：兩模式損益確實不同（否則測試無效）",
                  not np.isclose(logs["signal"]["Trade_PnL"].sum(),
                                 logs["dollar"]["Trade_PnL"].sum()),
                  f"signal {logs['signal']['Trade_PnL'].sum():.2f} vs "
                  f"dollar {logs['dollar']['Trade_PnL'].sum():.2f}")
        else:
            print(f"  （SL={sl:.0%}：Status 相同 = {same}；停損門檻依損益觸發，"
                  f"差異屬預期）")

    print("\n== 退化守衛 ==")
    va, vb = leg_dollars(t_sig, 0.0, sa, sb)     # β = 0 → 全部在 A 腳
    check("β=0 時全部配置於 A 腳且總額不變",
          np.isclose(va, CAP) and np.isclose(vb, 0.0), f"({va:.2f}, {vb:.2f})")
    va, vb = leg_dollars(t_sig, beta, 0.0, sb)   # σ_A 缺值 → 退回 1.0
    check("σ_A 為 0/None 時不炸且總額不變", np.isclose(va + vb, CAP), f"{va + vb:,.4f}")

    print("\n" + ("全部通過" if not _fails else f"失敗 {len(_fails)} 項：{_fails}"))
    raise SystemExit(1 if _fails else 0)


if __name__ == "__main__":
    main()
