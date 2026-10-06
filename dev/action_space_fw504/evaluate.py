# -*- coding: utf-8 -*-
"""504 版動作空間消融的評估（判準見同目錄 PREREGISTRATION.md §三）。

重用 `analysis/ablation_action_space.py` 的 H1／H2／H3 函式（同一套結果變數與統計程序：
逐期配對差分、循環區塊自助法 L = 6 期、10,000 次、族內 BH），只覆寫臂的對照表：
自由持倉臂只有 v3，比較族由 6 縮為 2。另以同一批函式算 252 版 v3 在同一組共同期上的數字並列。

    python dev/action_space_fw504/evaluate.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402

import analysis.ablation_action_space as A  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARMS_504 = {"Z-Score": "Grid (GICS-SDP-FW504-DOLLAR)",
            "v4":      "Grid (GICS-SDP-FW504-DRL-DOLLAR)",
            "v3":      "Grid (GICS-SDP-FW504-DRL-V3)"}
ARMS_252 = {"Z-Score": "Grid (GICS-SDP-DOLLAR)",
            "v4":      "Grid (GICS-SDP-DRL-DOLLAR)",
            "v3":      "Grid (GICS-SDP-DRL-V3)"}
OUT = "results/analysis/ablation_fw504"


def run(arms, label):
    A.ARMS = arms
    A.FREE_ARMS = ("v3",)
    A.BASE_ARMS = ("v4", "Z-Score")
    paths = A.arm_paths()
    missing = [k for k in arms if k not in paths]
    if missing:
        raise SystemExit(f"{label} 缺臂：{missing}")
    sets = [set(A.period_table(paths[k]).Period_Start) for k in arms]
    common = sorted(set.intersection(*sets))
    return paths, common, {k: len(s) for k, s in zip(arms, sets)}


def main():
    p504, c504, n504 = run(ARMS_504, "504 版")
    _, c252_all, _ = run(ARMS_252, "252 版")
    # 252 版的五臂共同期由 v1 決定（左緣 11 期 v1 拋例外，不在共同期內）
    A.ARMS = {"v1": "Grid (GICS-SDP-DRL-V1)"}
    p_v1 = A.arm_paths()
    c252 = sorted(set(c252_all) & set(A.period_table(p_v1["v1"]).Period_Start))

    print(f"504 版各臂期數 {n504}；三臂共同期 {len(c504)}（預期 108；< 100 視為執行異常）")
    print(f"252 版五臂共同期 {len(c252)}；兩版共同期相同：{c504 == c252}")
    if len(c504) < 100:
        raise SystemExit("共同期 < 100，先查明再談結果")

    out = {}
    for label, arms, common in (("504", ARMS_504, c504), ("252", ARMS_252, c252)):
        paths, _, _ = run(arms, label)
        out[label] = (A.h1_turnover(paths, common), A.h2_sharpe(paths, common), A.h3_cost(paths, common))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for label, (h1, h2, h3) in out.items():
        print(f"\n================ {label} 日形成期 ================")
        print("H1 換手率（每期每對進場次數；族 = 2）"); print(h1.to_string(index=False))
        print("H2 Sharpe（交集重算）"); print(h2.to_string(index=False))
        print("H3 零成本重算"); print(h3.to_string(index=False))
        h1.to_csv(f"{OUT}_{label}_h1.csv", index=False, encoding="utf-8-sig")
        h2.to_csv(f"{OUT}_{label}_h2.csv", index=False, encoding="utf-8-sig")
        h3.to_csv(f"{OUT}_{label}_h3.csv", index=False, encoding="utf-8-sig")

    h1, h2, h3 = out["504"]
    ok1 = bool(h1["達標"].all()) and len(h1) == 2
    v3s = float(h2.loc[h2.臂 == "v3", "交集Sharpe"].iloc[0])
    v4s = float(h2.loc[h2.臂 == "v4", "交集Sharpe"].iloc[0])
    ok2 = v3s <= v4s
    print("\n== 判準（504 版；PREREGISTRATION.md §三）==")
    print(f"H1：v3 對 v4、對 Z-Score 皆比值 ≥ 2.0 且 BH p < 0.05 → {'重現' if ok1 else '不重現'}")
    print(f"H2：v3 Sharpe {v3s:+.3f} ≤ v4 {v4s:+.3f} → {'重現' if ok2 else '不重現'}")
    r3 = h3.iloc[0]
    print(f"H3：零成本下 v3 相對 v4 的差距縮小 {r3['差距縮小%']}% → {'重現' if r3['H3成立'] else '不重現'}")


if __name__ == "__main__":
    main()
