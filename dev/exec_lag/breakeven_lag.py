# -*- coding: utf-8 -*-
"""§F 判準：t+1 執行下的 break-even 是否跌破 0.58% 往返成本。

PREREGISTRATION.md §六 的三條敘述規則以此裁決。口徑完全沿用
`analysis/regime_cost_dsr_eval`（同一個 `traded_notional`、同一個
`CURRENT_FEE_SIDE`），否則數字不可與論文表並列。

用法：python -m dev.exec_lag.breakeven_lag
"""
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from strategies.config import INITIAL_CAPITAL
from strategies.metrics import traded_notional

RESULT_DB = "results/result.db"
FEE_SIDE = 0.0029          # 與 regime_cost_dsr_eval.CURRENT_FEE_SIDE 同值
ROUNDTRIP = 2 * FEE_SIDE   # 0.58%
ARMS = ["Grid (NOGRP-SDP)", "Grid (NOGRP-SSD)", "Grid (GICS-SSD)",
        "Grid (NOGRP-DTW)", "Grid (GICS-SSD-FW504)", "Grid (GGR)"]
MAIN4 = ARMS[:4]
_CELL = re.compile(r"(Top\d+_SL\d+_ZWin\d+_MSR\d+)")


def be(path: str, top_n: int) -> float:
    """往返 break-even 成本（比例）。"""
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    eq = con.execute('SELECT Final_Equity FROM strategy_summaries WHERE _path=?',
                     (path,)).fetchone()
    con.close()
    if eq is None:
        return np.nan
    notion = traded_notional(path, top_n)
    if not (notion > 0):
        return np.nan
    return 2.0 * (FEE_SIDE + (float(eq[0]) - INITIAL_CAPITAL) / notion)


def main() -> int:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    d = pd.read_sql(
        'SELECT _path, METHOD, "TOP N" topn, Sharpe_Raw, Final_Equity '
        'FROM strategy_summaries WHERE METHOD IN (%s)'
        % ",".join("?" * len(ARMS)), con, params=ARMS)
    con.close()
    d["Sharpe_Raw"] = pd.to_numeric(d.Sharpe_Raw, errors="coerce")
    d["n"] = d.topn.str.extract(r"(\d+)").astype(int)
    fn = d._path.map(os.path.basename)
    d["cell"] = fn.map(lambda s: _CELL.search(s).group(1) if _CELL.search(s) else None)
    tag = fn.str.extract(r"_LAG(1[EXB])\.csv$")[0]
    d["cond"] = np.where(
        fn.str.fullmatch(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv"), "BASE", tag)
    d = d[d.cond.notna()]

    # 每臂取「同棒口徑下 Sharpe 最高的基準格」，再追蹤同一格在三個條件下的 break-even。
    # 不可各條件各自選最佳——那會多吃一次選擇偏誤，且比較的不是同一個策略。
    print("每臂以『同棒最佳格』定錨，追蹤同一格的往返 break-even%\n")
    hdr = (f"{'臂':<24}{'格':<12}{'同棒':>8}{'1E':>8}{'1X':>8}{'1B':>8}"
           f"{'丙−同棒':>10}   丙是否 > 0.58%")
    print(hdr)
    print("-" * (len(hdr) + 6))
    rows = []
    for m in ARMS:
        g = d[(d.METHOD == m) & (d.cond == "BASE")]
        if g.empty:
            continue
        anchor = g.loc[g.Sharpe_Raw.idxmax()]
        vals = {"BASE": be(anchor._path, anchor.n)}
        for c in ("1E", "1X", "1B"):
            r = d[(d.METHOD == m) & (d.cond == c) & (d.cell == anchor.cell)]
            vals[c] = be(r._path.iloc[0], int(r.n.iloc[0])) if len(r) else np.nan
        cell = anchor.cell.split("_ZWin")[0]
        f = lambda v: f"{v*100:>8.3f}" if np.isfinite(v) else f"{'—':>8}"
        surv = "   ✔ 是" if (np.isfinite(vals["1B"]) and vals["1B"] > ROUNDTRIP) else "   ✘ 否"
        print(f"{m:<24}{cell:<12}{f(vals['BASE'])}{f(vals['1E'])}{f(vals['1X'])}"
              f"{f(vals['1B'])}{(vals['1B']-vals['BASE'])*100:>10.3f}{surv}")
        rows.append({"METHOD": m, **vals})

    R = pd.DataFrame(rows).set_index("METHOD")
    print(f"\n往返成本假設 = {ROUNDTRIP*100:.2f}%（Do & Faff 2012，單邊 0.29%）")

    print("\n" + "=" * 92)
    print("PREREGISTRATION §六 的判準")
    print("=" * 92)
    m4 = R.loc[[m for m in MAIN4 if m in R.index]]
    below = m4[m4["1B"] <= ROUNDTRIP]
    print(f"四條主力臂在丙（t+1 兩端）條件下 break-even 低於 0.58% 的："
          f"{len(below)}/{len(m4)}")
    if len(below) == len(m4):
        print("→ **規則 1**：「扣成本後仍為正」必須撤回，改寫為"
              "「同棒成交下為正、t+1 執行下不成立」。論文層級的修改。")
    elif len(below) == 0:
        drop = (m4["BASE"] - m4["1B"]).median() * 100
        print(f"→ 四臂全數仍高於 0.58%（中位下降 {drop:.3f} pp）。")
        print("   依 §六，年化下降已達 0.44 pp（≥ 0.2 pp）→ **規則 3**："
              "主表並列兩個口徑。")
    else:
        print("→ **規則 3**：部分臂跌破，主表須並列兩個口徑並逐臂標示。")
    print("\n餘裕（break-even − 0.58%，pp）：")
    print(((R - ROUNDTRIP) * 100).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
