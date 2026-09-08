# -*- coding: utf-8 -*-
"""驗證 §F 執行延遲的語意：決策讀 bar i-L，成交價恆為 bar i。

不是「數字有沒有變」的煙霧測試——那種測試任何寫錯的延遲都會通過。
本檔直接斷言**進出場發生在哪一根 bar、以哪一根 bar 的價格成交**。

用法：python -m dev.exec_lag.verify_lag
"""
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

from strategies.trading.zscore_trading import Trading


def _make(scope: str, lag: int, prices: pd.DataFrame):
    return Trading(
        price_df=prices, trade_dates=prices.index, selected_pairs=pd.DataFrame(),
        capital_per_pair=10_000.0, fee_rate=0.0, slippage_rate=0.0,
        stop_loss_pct=0.0, entry_z=2.0, exit_z=0.0, zscore_window=0,
        min_spread_std=1e-9,
        execution_lag=lag, exec_lag_scope=scope,
    )


def _run(t: Trading, prices: pd.DataFrame) -> pd.DataFrame:
    return t._simulate_pair(
        period_start=str(prices.index[0].date()), period_end=str(prices.index[-1].date()),
        sector="X", ticker_a="A", ticker_b="B", pair_rank=1, hedge_ratio=1.0,
        form_spread_mean=0.0, form_spread_std=1.0,
        log_mean_a=0.0, log_std_a=SCALE, log_mean_b=0.0, log_std_b=1.0,
        first_price_a=0.0, first_price_b=0.0, ols_alpha=None,
    )


#: log 價的縮放。**不可省**：`clean_prices` 會把單日漲跌超過 50% 的 bar 當成
#: 壞資料剔除並 ffill。z 直接當 log 價會使每根 bar 動輒 ±65%，整條序列被前值填平，
#: 測出來的「延遲」其實是清洗造成的假象（本檔第一版就踩了這個坑）。
SCALE = 0.1


def build_prices() -> pd.DataFrame:
    """B 固定；A 的 log 價走一條「發散→收斂→穿越」的折線。

    spread = (ln A − 0)/SCALE − (ln B − 0)/1，故 z 就是下方 `Z_PATH`。
    刻意讓每一根 bar 的 z 都不同，這樣「用 i 還是 i-1 的 z」會產生
    可辨識的日期差，而不是被平台期吃掉。
    """
    idx = pd.bdate_range("2020-01-01", periods=len(Z_PATH))
    df = pd.DataFrame({"A": np.exp(Z_PATH * SCALE), "B": np.ones(len(Z_PATH))}, index=idx)
    worst = df["A"].pct_change().abs().max()
    assert worst < 0.50, f"合成價的單日變動 {worst:.1%} 會觸發 clean_prices 的 50% 濾除"
    return df


Z_PATH = np.array([0.0, 0.5, 1.0, 1.6, 2.4, 2.8, 2.2, 1.4, 0.6, -0.3, -0.8, -0.2,
                   0.3, 0.9, 1.5, 2.1, 2.6, 1.9, 1.1, 0.4], dtype=float)


#: `Price_A` 在 df_out 中存為四位小數，故價格比對只能到 5e-4。
PX_TOL = 5e-4


def summarize(df: pd.DataFrame) -> list:
    """回傳 [(bar 序號, 狀態, 成交價 A)]，只取有動作的 bar。"""
    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        if r["Status"] in ("ENTER_SHORT_A", "ENTER_LONG_A", "EXIT",
                           "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT"):
            out.append((i, r["Status"], float(r["Price_A"])))
    return out


def main() -> int:
    prices = build_prices()
    pa = prices["A"].values
    fails = []

    base = summarize(_run(_make("both", 0, prices), prices))
    ent = summarize(_run(_make("entry", 1, prices), prices))
    ext = summarize(_run(_make("exit", 1, prices), prices))
    both = summarize(_run(_make("both", 1, prices), prices))

    fmt = lambda ev: [(i, st, f"{px:.5f}") for i, st, px in ev]
    for tag, ev in (("L=0 基準", base), ("1E 只進場", ent),
                    ("1X 只出場", ext), ("1B 兩者", both)):
        print(f"{tag:<12} {fmt(ev)}")
    print()

    # ── 測試 1：基準的進場點就是 |z| 首次 > 2 的那一根 ──────────────────
    first = int(np.argmax(np.abs(Z_PATH) > 2.0))
    b_ent = [e for e in base if e[1].startswith("ENTER")]
    if not b_ent or b_ent[0][0] != first:
        fails.append(f"基準進場 bar 應為 {first}，實得 {b_ent[:1]}")
    else:
        print(f"✔ 測試1 基準於 bar {first} 進場（|z| 首次 > 2）")

    # ── 測試 2：只延遲進場 → 進場整整晚一根，且以該根的價成交 ───────────
    e_ent = [e for e in ent if e[1].startswith("ENTER")]
    if not e_ent or e_ent[0][0] != first + 1:
        fails.append(f"1E 進場 bar 應為 {first + 1}，實得 {e_ent[:1]}")
    elif abs(e_ent[0][2] - pa[first + 1]) > PX_TOL:
        fails.append(f"1E 成交價應為 bar {first+1} 的 {pa[first+1]:.6f}，實得 {e_ent[0][2]}")
    else:
        print(f"✔ 測試2 1E 於 bar {first+1} 進場，成交價 = 該根的 {pa[first+1]:.6f}")

    # ── 測試 3：只延遲出場 → 進場不動，出場晚一根 ──────────────────────
    b_ex = [e for e in base if e[1] == "EXIT"]
    x_ex = [e for e in ext if e[1] == "EXIT"]
    x_en = [e for e in ext if e[1].startswith("ENTER")]
    if not x_en or x_en[0][0] != first:
        fails.append(f"1X 不應改變進場 bar（應 {first}），實得 {x_en[:1]}")
    elif not b_ex or not x_ex:
        fails.append(f"基準或 1X 沒有 EXIT 事件（base={b_ex}, 1X={x_ex}）")
    elif x_ex[0][0] != b_ex[0][0] + 1:
        fails.append(f"1X 出場 bar 應為 {b_ex[0][0] + 1}，實得 {x_ex[0][0]}")
    elif abs(x_ex[0][2] - pa[b_ex[0][0] + 1]) > PX_TOL:
        fails.append(f"1X 出場成交價錯：應 {pa[b_ex[0][0]+1]:.6f}，實得 {x_ex[0][2]}")
    else:
        print(f"✔ 測試3 1X 進場不動（bar {first}），出場由 bar {b_ex[0][0]} 移到 {x_ex[0][0]}")

    # ── 測試 4：兩者皆延遲 = 進場與出場各晚一根 ─────────────────────────
    d_en = [e for e in both if e[1].startswith("ENTER")]
    d_ex = [e for e in both if e[1] == "EXIT"]
    if not d_en or d_en[0][0] != first + 1:
        fails.append(f"1B 進場 bar 應為 {first + 1}，實得 {d_en[:1]}")
    elif b_ex and d_ex and d_ex[0][0] != b_ex[0][0] + 1:
        fails.append(f"1B 出場 bar 應為 {b_ex[0][0] + 1}，實得 {d_ex[0][0]}")
    else:
        print(f"✔ 測試4 1B 進出場各晚一根")

    # ── 測試 5：L=0 的三個 scope 全等（scope 在 L=0 時不得有作用） ───────
    for sc in ("entry", "exit", "both"):
        if summarize(_run(_make(sc, 0, prices), prices)) != base:
            fails.append(f"L=0 但 scope={sc} 的結果與基準不同")
    if not any("scope=" in f for f in fails):
        print("✔ 測試5 L=0 時三個 scope 結果全等")

    # ── 測試 6：延遲不得動到成交價的來源（成交價恆為當根 bar 的價） ──────
    for tag, ev in (("1E", ent), ("1X", ext), ("1B", both)):
        for i, st, px in ev:
            if abs(px - pa[i]) > PX_TOL:
                fails.append(f"{tag} bar {i} 的成交價 {px} ≠ 該根的 {pa[i]:.6f}")
    if not any("的成交價" in f for f in fails):
        print("✔ 測試6 三個條件下成交價一律取自當根 bar")

    print()
    if fails:
        print(f"✘ {len(fails)} 項失敗：")
        for f in fails:
            print("   " + f)
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
