# -*- coding: utf-8 -*-
"""
兩年形成期延伸研究 Q-C：新方法計入試驗宇宙後的 DSR（dev/fw504_all/PLAN.md）
======================================================================
§5.1.2 的規則：跑過回測即進入試驗宇宙。19 條新方法（主軸 14＋DL-THR 5）計入後
method 口徑 N = 53 → 72。var_sr 依 regime_cost_dsr_eval 的定義重算（每 METHOD
取其全部格的平均 Sharpe_Raw，橫斷面變異 ÷ 252），排除不屬於試驗的列：
動作空間消融五臂（附錄 D 的明文例外）與附錄 E 的探索列。

先以同一定義重算「49 條現存方法」，核對能否重現 TRIAL_CENSUS 釘死的 var_sr；
再加入 19 條算新值。DSR 表寫到 results/analysis/fw504/，不覆寫正式的
breakeven_dsr.csv。

用法：python -m analysis.fw504_dsr
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import analysis.regime_cost_dsr_eval as R

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOT_TRIALS = ("-DOLLAR)", "-DRL-V1)", "-DRL-V2)", "-DRL-V3)", "EXPLORE")
OUT = "results/analysis/fw504"


def _is_trial(m: str) -> bool:
    return not any(t in m for t in NOT_TRIALS) and m not in R.INCOMPLETE_RUNS


def run():
    con = sqlite3.connect(f"file:{R.RESULT_DB}?mode=ro", uri=True)
    summ = pd.read_sql("SELECT METHOD, Sharpe_Raw FROM strategy_summaries", con)
    con.close()
    trials = summ[summ.METHOD.map(_is_trial)]
    means = trials.groupby("METHOD").Sharpe_Raw.mean()
    is_new = means.index.str.contains("FW504") & (means.index != "Grid (GICS-SSD-FW504)")

    var_old = float(np.var(means[~is_new], ddof=1)) / R.TRADING_DAYS
    var_new = float(np.var(means, ddof=1)) / R.TRADING_DAYS
    n_old_pinned, var_pinned = R.TRIAL_CENSUS["method"]
    n_new = n_old_pinned + int(is_new.sum())
    print(f"現存方法 {int((~is_new).sum())} 條：var_sr 重算 {var_old:.10f}"
          f"（釘死值 {var_pinned:.10f}）")
    print(f"新增 FW504 方法 {int(is_new.sum())} 條 → N {n_old_pinned} → {n_new}，"
          f"var_sr {var_new:.10f}")

    R.TRIAL_CENSUS["method"] = (n_new, var_new)
    R.OUT_DIR = OUT
    os.makedirs(OUT, exist_ok=True)
    methods = [m for m in means.index if "DRL" not in m]
    tbl1, _ = R.run(methods)
    passed = tbl1[tbl1["DSR"] >= 0.95]
    print(f"\n門檻 SR0（年化）＝ {tbl1['門檻SR0'].iloc[0]}；"
          f"{len(tbl1)} 條策略中通過 0.95：{len(passed)}")
    if len(passed):
        print(passed[["策略", "Sharpe", "DSR"]].to_string(index=False))
    return tbl1


if __name__ == "__main__":
    run()
