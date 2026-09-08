# -*- coding: utf-8 -*-
"""`exit_z` 實掃：逐項裁決 PREREGISTRATION.md 的 P1–P5 與 §五 門檻。

P1–P4 由 `strategy_summaries` 直接判定。§五 的門檻（逐日 block bootstrap）
需逐日序列，另由 `gate.py` 執行——本檔先做機制層的裁決，
因為 P3 不成立時門檻檢定就不必跑了。

用法：python -m dev.exit_z.analyze
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

RESULT_DB = "results/result.db"
ARMS = ["Grid (NOGRP-SDP)", "Grid (NOGRP-SSD)", "Grid (GICS-SSD)", "Grid (NOGRP-DTW)"]
XZ = ["25", "50", "75", "100"]
_CELL = re.compile(r"(Top\d+_SL\d+_ZWin\d+_MSR\d+)")


def load() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=180)
    d = pd.read_sql(
        'SELECT _path, METHOD, "TOP N" topn, Ann_Ret_Raw, Sharpe_Raw, Final_Equity, '
        'Entries, Exits, Forced_Closes, Avg_Trade_Days, Avg_Utilization, '
        'Profit_Factor, Win_Rate FROM strategy_summaries WHERE METHOD IN (%s)'
        % ",".join("?" * len(ARMS)), con, params=ARMS)
    con.close()
    for c in d.columns:
        if c not in ("_path", "METHOD", "topn"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    fn = d._path.map(os.path.basename)
    d["cell"] = fn.map(lambda s: _CELL.search(s).group(1) if _CELL.search(s) else None)
    tag = fn.str.extract(r"_XZ(\d+)\.csv$")[0]
    d["xz"] = np.where(
        fn.str.fullmatch(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv"), "0", tag)
    return d[d.xz.notna()].copy()


def main() -> int:
    d = load()
    print(f"讀入 {len(d)} 列：基準格 {(d.xz=='0').sum()}、_XZ {(d.xz!='0').sum()}\n")

    base = d[d.xz == "0"].set_index(["METHOD", "cell"])
    P = []
    for x in XZ:
        g = d[d.xz == x].set_index(["METHOD", "cell"])
        j = g.join(base, rsuffix="_b", how="inner").reset_index()
        j["xz"] = x
        j["d_ann"] = (j.Ann_Ret_Raw - j.Ann_Ret_Raw_b) * 100
        j["d_sharpe"] = j.Sharpe_Raw - j.Sharpe_Raw_b
        j["r_ent"] = j.Entries / j.Entries_b - 1.0
        j["d_days"] = j.Avg_Trade_Days - j.Avg_Trade_Days_b
        P.append(j)
    p = pd.concat(P, ignore_index=True)
    print(f"可配對 {len(p)} 組（臂 × 格 × 值）\n")

    piv = lambda col: (p.pivot_table(index="METHOD", columns="xz", values=col,
                                     aggfunc="median")
                       .reindex(index=[a for a in ARMS], columns=XZ))

    print("=" * 88)
    print("表 1：年化報酬位移（pp，逐格相減後取中位）　exit_z = 0.25 / 0.5 / 0.75 / 1.0")
    print("=" * 88)
    t_ann = piv("d_ann")
    print(t_ann.round(3).to_string())
    print("\nSharpe 位移（中位）:")
    print(piv("d_sharpe").round(4).to_string())

    ok = {}

    # P2：平均持有日單調遞減
    print("\n" + "=" * 88)
    t_days = p.pivot_table(index="METHOD", columns="xz", values="Avg_Trade_Days",
                           aggfunc="median").reindex(index=ARMS, columns=XZ)
    t_days.insert(0, "0", d[d.xz == "0"].groupby("METHOD").Avg_Trade_Days.median()
                  .reindex(ARMS))
    mono = t_days.apply(lambda r: bool((np.diff(r.values) < 0).all()), axis=1)
    ok["P2"] = bool(mono.all())
    print(f"P2  平均持有日單調遞減於 exit_z：{int(mono.sum())}/{len(mono)} 臂  "
          f"→  {'✔ 成立' if ok['P2'] else '✘ 不成立'}")
    print(t_days.round(1).to_string())

    # P3：進場次數上升 —— 本案的賭注
    print("\n" + "=" * 88)
    t_ent = piv("r_ent") * 100
    pos = (t_ent > 0).all(axis=1)
    ok["P3"] = bool((t_ent > 0).all().all())
    print(f"P3  進場次數上升（本案假說的核心）：四臂四值中為正的有 "
          f"{int((t_ent > 0).sum().sum())}/{t_ent.size}  "
          f"→  {'✔ 成立' if ok['P3'] else '✘ 不成立'}")
    print("進場次數相對變動 %:")
    print(t_ent.round(2).to_string())

    # P4：exit_z=1.0 劣於 0.75
    print("\n" + "=" * 88)
    worse = t_ann["100"] < t_ann["75"]
    ok["P4"] = bool(worse.all())
    print(f"P4  exit_z=1.0 劣於 0.75：{int(worse.sum())}/{len(worse)} 臂  "
          f"→  {'✔ 成立' if ok['P4'] else '✘ 不成立'}")

    # P5 / §五 門檻的方向前置：四臂中至少三臂同號為正
    print("\n" + "=" * 88)
    print("§五 門檻的方向前置條件（四臂中至少三臂年化差為正）")
    print("=" * 88)
    any_pass = False
    for x in XZ:
        n_pos = int((t_ann[x] > 0).sum())
        mark = "✔ 可進檢定" if n_pos >= 3 else "✘ 方向即不成立"
        print(f"  exit_z={float(x)/100:<5}  為正 {n_pos}/4 臂   "
              f"合併中位 {p[p.xz==x].d_ann.median():+.3f} pp   {mark}")
        any_pass |= n_pos >= 3
    ok["P5_dir"] = any_pass

    print("\n" + "=" * 88)
    if not any_pass:
        print("裁決：**未過閘**。沒有任何 exit_z 值在四臂中至少三臂為正，")
        print("      方向前置條件即不成立，依 §五 不需執行 bootstrap 檢定。")
        print("      停止：不改 base_params、不進論文主表、照實記錄為否證。")
    else:
        print("裁決：方向前置條件成立 → 須執行 §五 的逐日 block bootstrap 檢定。")
    print("\n預測裁決：" + "、".join(
        f"{k} {'✔' if v else '✘'}" for k, v in ok.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
