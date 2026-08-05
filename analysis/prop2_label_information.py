# -*- coding: utf-8 -*-
"""
命題 2 第四項受控對照：反事實標籤值多少錢？
======================================================================

前三項受控對照回答「DL-THR 的增益**不是**來自什麼」——不是拉高門檻
（prop2_exposure_control）、不是篩掉劣質配對（prop2_skip_permutation）、
不是減少曝險。本模組回答一個不同的問題：增益有多少來自**學習演算法**，
又有多少只是因為這個問題碰巧是**全資訊**的？

現役的 DL-THR 名為「DRL」，實際是全資訊監督回歸：每個歷史配對期上 9 個門檻
的報酬都能精確反事實回算，模型拿到完整答案卷，直接以 MSE 回歸學就好，
無探索問題。RL-THR 把答案卷收走——只觀測實際選中的那一格、改用 ε-greedy——
其餘（動作選單、12 維狀態、網路、walk-forward 切分）逐位元相同。

於是兩者相減即為**反事實標籤的價值**（含探索成本，見下）。

口徑
----
一律沿用命題 2 主檢定（proposition2_daily_hac）的設計，否則數字不可互相引用：

  · 抽樣單位 = 時間（逐日），不是參數網格（15 格共用同一條回測路徑 → 偽重複）
  · 聚合 = 15 格等權組合（沒有東西可挑，選擇偏誤自源頭消失）
  · 主檢定 = 循環 block bootstrap（L=126，10,000 次），另附 Newey-West HAC 對照

三組 ε 排程各自檢定；主張以**對 bandit 最有利者**為準——若在最有利條件下
DL-THR 仍顯著較優，「反事實標籤有價值」的結論才保守而站得住。

一項無法分離的混淆（須明寫）
----------------------------
RL-THR 與 DL-THR 的差距同時包含兩個效應：
  (a) 資訊量：每期只累積 1 筆「動作—報酬」觀測，DL-THR 是 9 筆
  (b) 探索成本：ε 抽中時實際下單，虧損如實計入績效
本設計**無法**把兩者分開——不探索就沒有樣本，這正是部分回饋的定義。
掃三組 ε 可界定其取捨曲線，但不能拆解為兩個獨立的數字。

用法：python -m analysis.prop2_label_information
"""
import os
import sys

import numpy as np
import pandas as pd

from analysis.block_bootstrap import BLOCK_L, bootstrap_test
from analysis.proposition2_daily_hac import (
    OUT_DIR, TRADING_DAYS, _grid_cell, baseline_only, load_daily_sids,
    method_paths, newey_west,
)
from strategies.config import INITIAL_CAPITAL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ZSCORE = "Grid (AGG-SSD)"
DLTHR = "Grid (AGG-SSD-DRL)"
RLTHR = {
    "ε=0.05（常數）":        "Grid (AGG-SSD-RLTHR-E05)",
    "ε=0.10（常數）":        "Grid (AGG-SSD-RLTHR-E10)",
    "ε=0.20→0.02（衰減）":   "Grid (AGG-SSD-RLTHR-E20D)",
}


def _cells(methods: list[str]) -> dict[str, dict[str, str]]:
    """{METHOD: {網格格子: strategy_id}}，只取基準格。"""
    meta = method_paths(methods)
    out = {}
    for m, g in meta.groupby("METHOD"):
        out[m] = {_grid_cell(s): s for s in baseline_only(g._path.tolist())}
    return out


def _ew(px: pd.DataFrame, cellmap: dict[str, str], cells: list[str]) -> np.ndarray:
    return px[[cellmap[c] for c in cells]].mean(axis=1).values


def _stats(d: np.ndarray) -> dict:
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    return {
        "年化Δ%": round(mu * TRADING_DAYS / INITIAL_CAPITAL * 100, 3),
        "IR": round(np.sqrt(TRADING_DAYS) * mu / sd, 4) if sd > 0 else np.nan,
        "勝日%": round(float((d > 0).mean()) * 100, 1),
    }


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    methods = [ZSCORE, DLTHR] + list(RLTHR.values())
    cmap = _cells(methods)

    missing = [m for m in methods if not cmap.get(m)]
    if missing:
        sys.exit(f"result.db 查無這些 METHOD 的基準格：{missing}")

    px = load_daily_sids([s for m in methods for s in cmap[m].values()])

    # 三臂必須落在同一組網格格子上，否則等權組合的成分不同、差分無意義
    common = sorted(set.intersection(*(set(cmap[m]) for m in methods)))
    print(f"共同網格格子 {len(common)} 格：{', '.join(common[:5])}"
          f"{' …' if len(common) > 5 else ''}\n")

    zs = _ew(px, cmap[ZSCORE], common)
    dl = _ew(px, cmap[DLTHR], common)

    rows = []
    for label, m in RLTHR.items():
        rl = _ew(px, cmap[m], common)
        for tag, d in (("RL-THR − Z-Score", rl - zs),
                       ("RL-THR − DL-THR", rl - dl)):
            # d 為日損益金額（$）→ bootstrap_test 以預設 capital 換算為年化百分點
            bs = bootstrap_test(d, L=BLOCK_L)
            nw = newey_west(d)
            rows.append({
                "ε 排程": label, "對照": tag, **_stats(d),
                "p(bootstrap)": bs["BB p"],
                "CI下界pp": bs["CI下界"], "CI上界pp": bs["CI上界"],
                "NW t": round(float(nw[0]), 3), "NW p": round(float(nw[1]), 4),
            })

    # DL-THR − Z-Score：主檢定的既有結果，此處重算一次作為量級參照
    d0 = dl - zs
    bs0 = bootstrap_test(d0, L=BLOCK_L)
    nw0 = newey_west(d0)
    rows.insert(0, {
        "ε 排程": "—（參照）", "對照": "DL-THR − Z-Score", **_stats(d0),
        "p(bootstrap)": bs0["BB p"],
        "CI下界pp": bs0["CI下界"], "CI上界pp": bs0["CI上界"],
        "NW t": round(float(nw0[0]), 3), "NW p": round(float(nw0[1]), 4),
    })

    out = pd.DataFrame(rows)
    print("=" * 104)
    print("命題 2 · 受控對照④：反事實標籤的價值（Grid AGG-SSD 配對底，15 格等權，逐日）")
    print("=" * 104)
    print(out.to_string(index=False))

    path = f"{OUT_DIR}/prop2_label_information.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[已存] {path}")

    # ── 判讀 ────────────────────────────────────────────────────────
    best = out[out["對照"] == "RL-THR − DL-THR"].loc[
        lambda d: d["年化Δ%"].idxmax()]
    print("\n" + "─" * 104)
    print(f"對 bandit 最有利的排程：{best['ε 排程']}")
    print(f"  RL-THR − DL-THR：年化 {best['年化Δ%']:+.3f} pp"
          f"  95% CI [{best['CI下界pp']:+.3f}, {best['CI上界pp']:+.3f}]"
          f"  p = {best['p(bootstrap)']:.4f}")
    verdict = ("DL-THR 顯著較優 → 反事實標籤有價值"
               if best["p(bootstrap)"] < 0.05 and best["年化Δ%"] < 0 else
               "未達顯著 → 無法主張反事實標籤帶來可偵測的增益")
    print(f"  判讀：{verdict}")

    vs_zs = out[(out["對照"] == "RL-THR − Z-Score")
                & (out["ε 排程"] == best["ε 排程"])].iloc[0]
    print(f"  同一排程對 Z-Score：年化 {vs_zs['年化Δ%']:+.3f} pp"
          f"  p = {vs_zs['p(bootstrap)']:.4f}"
          f"  → {'仍優於規則型基準' if vs_zs['p(bootstrap)'] < 0.05 and vs_zs['年化Δ%'] > 0 else '未達顯著'}")
    print("─" * 104)
    print("提醒：此差距同時含「資訊量」與「探索成本」兩個效應，本設計無法分離。")


if __name__ == "__main__":
    run()
