# -*- coding: utf-8 -*-
"""strategies/metrics.py 的斷言檢查。

不需要 pytest。秒級完成（純函式部分），對帳部分可加 --recon N 抽驗 N 列。

    python -m tools.check_metrics
    python -m tools.check_metrics --recon 30
"""

import sys
import sqlite3

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from strategies.metrics import (metrics_from_pnl, metrics, traded_notional,
                                breakeven_roundtrip, TRADING_DAYS)
from strategies.config import INITIAL_CAPITAL

_fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    if not ok:
        _fails.append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  " + detail) if detail else ""))


# ── 純函式核心 ──────────────────────────────────────────────────────
def t_pure():
    print("== metrics_from_pnl 純函式 ==")
    idx = pd.date_range("2020-01-01", periods=504, freq="B")

    check("空序列回傳全 0", (metrics_from_pnl(pd.Series(dtype=float)) == 0).all())

    # 零波動：日報酬恆為 0（守衛 std != 0 應讓 Sharpe 回傳 0）。
    # 注意不能用「每日損益恆為 1」來測——日報酬是 損益/前一日權益，
    # 權益成長使日報酬遞減，std != 0，Sharpe 不為 0。
    zero = pd.Series(0.0, index=idx)
    mz = metrics_from_pnl(zero)
    check("零波動 → Sharpe 為 0（守衛 std != 0）", mz["Sharpe_Raw"] == 0.0)
    check("零波動 → MDD 為 0", mz["MDD_Raw"] == 0.0)

    # 常數正損益：單調上升故 MDD 為 0，但日報酬遞減故 Sharpe 有限且為正
    flat = pd.Series(1.0, index=idx)
    m = metrics_from_pnl(flat)
    check("單調上升 → MDD 為 0", m["MDD_Raw"] == 0.0)
    check("常數正損益 → Sharpe 為正有限", 0 < m["Sharpe_Raw"] < np.inf,
          "%.2f" % m["Sharpe_Raw"])
    check("期末權益 = 初始 + 累計", abs(m["Final_Equity"] - (INITIAL_CAPITAL + 504)) < 1e-9)

    # 負值子集為空 → Sortino 走守衛
    check("無負報酬 → Sortino 為 0", m["Sortino_Raw"] == 0.0)

    # MDD 手算：先漲 100 再跌 300
    v = pd.Series([0.0] * 10, index=idx[:10]); v.iloc[1] = 100.0; v.iloc[5] = -300.0
    m2 = metrics_from_pnl(v)
    peak = INITIAL_CAPITAL + 100.0
    expect = (peak - 300.0 - peak) / peak
    check("MDD 手算相符", abs(m2["MDD_Raw"] - expect) < 1e-12,
          "%.6f vs %.6f" % (m2["MDD_Raw"], expect))

    # 年化為月頻，基準是第一個月底權益（非初始資金）——這是 db_utils 的慣例
    eq = INITIAL_CAPITAL + v.cumsum()
    me = eq.resample("ME").last().dropna()
    cum = float(np.prod(1 + me.pct_change().fillna(0)) - 1)
    check("Cum_Ret 為月頻連乘（非期末/初始）",
          abs(m2["Cum_Ret_Raw"] - cum) < 1e-12,
          "月頻 %.8f ｜ 期末/初始 %.8f" % (cum, eq.iloc[-1] / INITIAL_CAPITAL - 1))

    # 尺度不變性：損益與初始資金同倍放大，Sharpe/MDD 不變
    a = metrics_from_pnl(v)
    b = metrics_from_pnl(v * 3, initial_capital=INITIAL_CAPITAL * 3)
    check("Sharpe 對尺度不變", abs(a["Sharpe_Raw"] - b["Sharpe_Raw"]) < 1e-9)
    check("MDD 對尺度不變", abs(a["MDD_Raw"] - b["MDD_Raw"]) < 1e-9)


# ── 對 db 的抽樣對帳 ────────────────────────────────────────────────
def t_recon(n):
    print("\n== 對 strategy_summaries 抽驗 %d 列 ==" % n)
    con = sqlite3.connect("file:results/result.db?mode=ro", uri=True)
    s = pd.read_sql('select _path,"TOP N" t,Cum_Ret_Raw,Ann_Ret_Raw,Sharpe_Raw,'
                    "Sortino_Raw,MDD_Raw,Calmar_Raw,Final_Equity,Avg_Utilization,"
                    "Ann_Ret_Employed,Excess_Ret_RF from strategy_summaries "
                    "order by random() limit ?", con, params=(n,))
    con.close()
    cols = [c for c in s.columns if c not in ("_path", "t")]
    worst = {c: 0.0 for c in cols}
    for _, r in s.iterrows():
        tn = int(str(r.t).replace("Top", "").strip())
        m = metrics(r._path, top_n=tn)
        for c in cols:
            a, b = r[c], m.get(c, np.nan)
            if pd.isna(a) or pd.isna(b):
                continue
            worst[c] = max(worst[c], abs(float(a) - float(b)))
    for c in cols:
        check("%s 對 db 一致" % c, worst[c] < 1e-9, "最大差 %.2e" % worst[c])


# ── 名目額與 break-even ────────────────────────────────────────────
def t_notional():
    print("\n== traded_notional / breakeven_roundtrip ==")
    p = "tiingo/Grid_NOGRP_DTW/TradeLogs_Top1_SL0_ZWin0_MSR0.csv"
    n = traded_notional(p, 1)
    check("名目額為正且有限", np.isfinite(n) and n > 0, f"${n:,.0f}")

    be = breakeven_roundtrip(p, 1)
    check("break-even 落在合理範圍", 0.0 < be < 0.05, "%.4f%%" % (be * 100))

    # 自洽：把費率設為 break-even，淨利應歸零
    fee_paid = n * 0.0029
    net = float(metrics(p)["Final_Equity"]) - INITIAL_CAPITAL
    net_at_be = net + fee_paid - n * (be / 2.0)
    check("費率設為 break-even 時淨利歸零", abs(net_at_be) < 1.0,
          "殘餘 $%.2f" % net_at_be)


def main():
    n = 0
    if "--recon" in sys.argv:
        n = int(sys.argv[sys.argv.index("--recon") + 1])
    t_pure()
    t_notional()
    if n:
        t_recon(n)
    print("\n%s" % ("全部通過" if not _fails else "失敗 %d 項: %s" % (len(_fails), _fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
