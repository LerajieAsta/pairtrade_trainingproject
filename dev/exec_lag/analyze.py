# -*- coding: utf-8 -*-
"""§F 執行延遲消融：逐項裁決 PREREGISTRATION.md 的 P1–P5。

配對方式：以網格格子（Top{n}/SL{x}）把 `_LAG` 列對回同臂的基準格，
逐格相減後取跨格中位。不可直接比兩邊的平均——各臂格數與量級不同。

用法：python -m dev.exec_lag.analyze
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
ARMS = ["Grid (NOGRP-SDP)", "Grid (NOGRP-SSD)", "Grid (GICS-SSD)",
        "Grid (NOGRP-DTW)", "Grid (GICS-SSD-FW504)", "Grid (GGR)"]
COND = {"1E": "只延遲進場", "1X": "只延遲出場", "1B": "兩者皆延遲"}
_CELL = re.compile(r"(Top\d+_SL\d+_ZWin\d+_MSR\d+)")


def load() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    d = pd.read_sql(
        'SELECT _path, METHOD, Ann_Ret_Raw, Sharpe_Raw, Final_Equity, Entries, '
        'Avg_Utilization, Profit_Factor, Forced_Closes, Avg_Trade_Days '
        'FROM strategy_summaries WHERE METHOD IN (%s)'
        % ",".join("?" * len(ARMS)), con, params=ARMS)
    con.close()
    for c in ("Ann_Ret_Raw", "Sharpe_Raw", "Final_Equity", "Entries",
              "Avg_Utilization", "Profit_Factor", "Forced_Closes", "Avg_Trade_Days"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    fn = d._path.map(os.path.basename)
    d["cell"] = fn.map(lambda s: (_CELL.search(s) or [None]).__getitem__(0)
                       if _CELL.search(s) else None)
    tag = fn.str.extract(r"_LAG(1[EXB])\.csv$")[0]
    # 基準格 = 完全沒有後綴的檔名
    d["cond"] = np.where(fn.str.fullmatch(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv"),
                         "BASE", tag)
    return d[d.cond.notna()].copy()


def paired(d: pd.DataFrame) -> pd.DataFrame:
    base = d[d.cond == "BASE"].set_index(["METHOD", "cell"])
    rows = []
    for c in COND:
        g = d[d.cond == c].set_index(["METHOD", "cell"])
        j = g.join(base, rsuffix="_b", how="inner")
        if j.empty:
            continue
        j = j.reset_index()
        j["cond"] = c
        j["d_ann"] = (j.Ann_Ret_Raw - j.Ann_Ret_Raw_b) * 100      # pp
        j["d_sharpe"] = j.Sharpe_Raw - j.Sharpe_Raw_b
        j["d_eq"] = j.Final_Equity - j.Final_Equity_b
        j["r_ent"] = j.Entries / j.Entries_b - 1.0
        rows.append(j)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    d = load()
    n_lag = (d.cond != "BASE").sum()
    print(f"讀入 {len(d)} 列（基準格 {(d.cond=='BASE').sum()}、_LAG {n_lag}）\n")
    if n_lag == 0:
        print("尚無 _LAG 列。")
        return 1

    p = paired(d)
    if p.empty:
        print("無可配對的格子。")
        return 1
    print(f"可配對 {len(p)} 組（臂 × 格 × 條件）\n")

    # ── 表 1：逐臂 × 逐條件的年化位移（pp） ──────────────────────────
    print("=" * 92)
    print("表 1：年化報酬位移（pp，各臂 15 格逐格相減後取中位）")
    print("=" * 92)
    t1 = p.pivot_table(index="METHOD", columns="cond", values="d_ann", aggfunc="median")
    t1 = t1.reindex(index=[a for a in ARMS if a in t1.index],
                    columns=[c for c in COND if c in t1.columns])
    print(t1.round(3).to_string())
    print()
    t1s = p.pivot_table(index="METHOD", columns="cond", values="d_sharpe", aggfunc="median")
    t1s = t1s.reindex(index=t1.index, columns=t1.columns)
    print("Sharpe 位移（中位）:")
    print(t1s.round(4).to_string())

    # ── P2：只延遲進場 → 報酬下降，六臂中至少五臂同號為負 ────────────
    print("\n" + "=" * 92)
    ok = {}
    if "1E" in t1.columns:
        neg = int((t1["1E"] < 0).sum())
        # 判準寫死為「六臂中至少五臂」。臂未跑齊時不得裁決——否則 4/4 全負
        # 會因為分母不足而被判為「不成立」，那是假的否證。
        if len(t1) < len(ARMS):
            print(f"P2  只延遲進場使報酬下降：{neg}/{len(t1)} 臂為負"
                  f"　⏸ 尚有 {len(ARMS)-len(t1)} 臂未完成，判準（六臂中 ≥5）暫不裁決")
        else:
            ok["P2"] = neg >= 5
            print(f"P2  只延遲進場使報酬下降：{neg}/{len(t1)} 臂為負"
                  f"（判準 ≥5）  →  {'✔ 成立' if ok['P2'] else '✘ 不成立'}")
        print(f"    全臂合併中位 {p[p.cond=='1E'].d_ann.median():+.3f} pp")

    # ── P3：只延遲出場 —— 不預測方向，照實報告 ────────────────────────
    if "1X" in t1.columns:
        pos = int((t1["1X"] > 0).sum())
        print(f"\nP3  只延遲出場（跑前不預測方向）：{pos}/{len(t1)} 臂為正、"
              f"{len(t1)-pos} 臂為負")
        print(f"    全臂合併中位 {p[p.cond=='1X'].d_ann.median():+.3f} pp"
              f"　→ 事後判讀見下方裁決")

    # ── P4：可加性 ────────────────────────────────────────────────────
    if all(c in t1.columns for c in ("1E", "1X", "1B")):
        resid = (t1["1B"] - (t1["1E"] + t1["1X"])).abs()
        bound = 0.5 * pd.concat([t1["1E"].abs(), t1["1X"].abs()], axis=1).max(axis=1)
        ok["P4"] = bool((resid <= bound).all())
        print(f"\nP4  丙 ≈ 甲 + 乙（|殘差| ≤ 0.5 × max(|甲|,|乙|)）："
              f"{int((resid <= bound).sum())}/{len(t1)} 臂通過  "
              f"→  {'✔ 成立' if ok['P4'] else '✘ 不成立'}")
        chk = pd.DataFrame({"甲1E": t1["1E"], "乙1X": t1["1X"], "丙1B": t1["1B"],
                            "甲+乙": t1["1E"] + t1["1X"],
                            "殘差": t1["1B"] - (t1["1E"] + t1["1X"]),
                            "容許": bound})
        print(chk.round(3).to_string())

    # ── P5：進場次數變動 < 5% ─────────────────────────────────────────
    worst = p.groupby("cond").r_ent.apply(lambda s: s.abs().max())
    ok["P5"] = bool((worst < 0.05).all())
    print(f"\nP5  進場次數相對變動 < 5%：最大 {worst.max()*100:.2f}%  "
          f"→  {'✔ 成立' if ok['P5'] else '✘ 不成立'}")
    print("    逐條件最大變動：" +
          "、".join(f"{c} {v*100:.2f}%" for c, v in worst.items()))

    # ── 論文層判準（PREREGISTRATION §六） ─────────────────────────────
    print("\n" + "=" * 92)
    print("論文層判準：丙（t+1 兩端）對四條主力臂的年化位移")
    print("=" * 92)
    main4 = ["Grid (NOGRP-SDP)", "Grid (NOGRP-SSD)",
             "Grid (GICS-SSD)", "Grid (NOGRP-DTW)"]
    if "1B" in t1.columns:
        sub = t1.loc[[m for m in main4 if m in t1.index], "1B"]
        print(sub.round(3).to_string())
        drop = -sub.median()
        print(f"\n四臂丙條件年化下降中位：{drop:+.3f} pp")
        if drop < 0.2:
            print("→ 規則 2 適用：以一句話與附表揭露，主表維持同棒口徑")
        else:
            print("→ 位移 ≥ 0.2 pp：須查 break-even 是否跌破 0.58%（規則 1 / 3）")

    # ── 3% 門檻的存續（承前一輪清點） ────────────────────────────────
    print("\n" + "=" * 92)
    print("延伸：t+1 下仍達年化 3% 的配置")
    print("=" * 92)
    b = d[(d.cond == "BASE") & (d.Ann_Ret_Raw >= 0.03)]
    print(f"同棒口徑下本六臂中達標 {len(b)} 格；逐格追蹤其在三個條件下的年化%：")
    print()
    hdr = f"{'臂':<24}{'格':<16}{'同棒':>8}{'1E':>8}{'1X':>8}{'1B':>8}   丙後是否仍達標"
    print(hdr); print("-" * len(hdr))
    for r in b.sort_values("Ann_Ret_Raw", ascending=False).itertuples():
        line = [f"{r.METHOD:<24}{r.cell.split('_ZWin')[0]:<16}{r.Ann_Ret_Raw*100:>8.2f}"]
        vals = {}
        for c in COND:
            g = d[(d.cond == c) & (d.METHOD == r.METHOD) & (d.cell == r.cell)]
            v = float(g.Ann_Ret_Raw.iloc[0]) * 100 if len(g) else np.nan
            vals[c] = v
            line.append(f"{v:>8.2f}" if np.isfinite(v) else f"{'—':>8}")
        keep = np.isfinite(vals.get("1B", np.nan)) and vals["1B"] >= 3.0
        line.append("   ✔ 是" if keep else "   ✘ 否")
        print("".join(line))
    n_keep = sum(
        1 for r in b.itertuples()
        if len(d[(d.cond == "1B") & (d.METHOD == r.METHOD) & (d.cell == r.cell)])
        and float(d[(d.cond == "1B") & (d.METHOD == r.METHOD)
                    & (d.cell == r.cell)].Ann_Ret_Raw.iloc[0]) >= 0.03)
    print()
    print(f"→ t+1 兩端延遲後仍達 3% 的：{n_keep}/{len(b)} 格")

    print("\n" + "=" * 92)
    fails = [k for k, v in ok.items() if not v]
    print("預測裁決：" + ("全部成立" if not fails else f"不成立 → {fails}"))
    print("（P1 已由 dev/exec_lag/verify_lag.py 與 pilot 比對通過；P3 不設方向判準）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
