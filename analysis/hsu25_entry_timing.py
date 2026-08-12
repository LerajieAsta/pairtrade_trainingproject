# -*- coding: utf-8 -*-
"""
許鈞翔 (2025) 差異定位 · 第五項：進場時點
======================================================================

指導教授質疑本研究的回測結果與前一屆學生（許鈞翔 2025）差異過大。逐項比對
其程式（`ref/original_code/純DTW配對交易-[版本2].ipynb`）與論文第三章後，
共八項差異，本模組隔離其中的**第五項：進場時點**。

    本研究（突破式，Gatev, Goetzmann & Rouwenhorst 2006）
        |z| > entry_z 即進場——價差一發散到帶外就進場，賭它收斂。

    許鈞翔（回歸式）
        if prev > upper and curr <= upper:   signal = -1
        elif prev < lower and curr >= lower:  signal = +1
        價差必須先跑到帶外、再**收斂回帶內**，才在穿越當日進場。

隔離方式
--------
`HSU25 {X} REV` 三條**借用** `HSU25 {X}` 已算好的形成期配對
（config 的 formation_strategy_id_base），交易端換成
`strategies.trading.zscore_reversion_entry_trading`——該類別只覆寫
`_entry_triggered` 與方向判定，部位規模、對沖比率、手續費（單邊 0.29%，
與許鈞翔相同）、四檔止損、出場、政體過濾一律繼承親代。

故 REV − 突破式的差分**完全歸因於進場時點**，其餘七項差異不動。

口徑
----
沿用命題 2 主檢定（proposition2_daily_hac）的設計，否則數字不可互相引用：

  · 抽樣單位 = 時間（逐日），不是參數網格（4 格共用同一組配對 → 偽重複）
  · 聚合 = 四檔止損等權（沒有東西可挑，選擇偏誤自源頭消失）
  · 主檢定 = 循環 block bootstrap（L=126，10,000 次），另附 Newey-West HAC

三種排序（SSD / DTW / SDP）各自檢定，不合併——它們是三組不同的配對底。

讀法上的限制
------------
本檢定只回答「在**本引擎**上、其餘條件相同時，換進場時點值多少」。它**不**
等於「許鈞翔的回測結果減本研究的回測結果」——後者還混著樣本期間、分組、
篩選、取幾對、報酬口徑等其餘七項差異。

用法：python -m analysis.hsu25_entry_timing
"""
import os
import sqlite3
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

# 排序後端 → (突破式 METHOD, 回歸式 METHOD)
ARMS = {
    "SSD": ("HSU25 (SSD)", "HSU25 (SSD-REV)"),
    "DTW": ("HSU25 (DTW)", "HSU25 (DTW-REV)"),
    "SDP": ("HSU25 (SDP)", "HSU25 (SDP-REV)"),
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
    methods = [m for pair in ARMS.values() for m in pair]
    cmap = _cells(methods)

    missing = [m for m in methods if not cmap.get(m)]
    if missing:
        sys.exit(f"result.db 查無這些 METHOD 的基準格：{missing}\n"
                 f"（回歸式三條需先跑：$env:STRATEGIES_SLICE=\"21,23,25\"; python run_trading.py）")

    px = load_daily_sids([s for m in methods for s in cmap[m].values()])

    rows = []
    for tag, (brk_m, rev_m) in ARMS.items():
        # 兩臂必須落在同一組止損格上，否則等權組合的成分不同、差分無意義
        common = sorted(set(cmap[brk_m]) & set(cmap[rev_m]))
        brk = _ew(px, cmap[brk_m], common)
        rev = _ew(px, cmap[rev_m], common)
        d = rev - brk           # 日損益金額（$）；bootstrap_test 內部換算年化 pp
        bs = bootstrap_test(d, L=BLOCK_L)
        nw = newey_west(d)
        rows.append({
            "排序": tag, "共同格數": len(common),
            "突破式年化%": _stats(brk)["年化Δ%"],
            "回歸式年化%": _stats(rev)["年化Δ%"],
            **{f"Δ{k}" if k != "年化Δ%" else k: v for k, v in _stats(d).items()},
            "p(bootstrap)": bs["BB p"],
            "CI下界pp": bs["CI下界"], "CI上界pp": bs["CI上界"],
            "NW t": round(float(nw[0]), 3), "NW p": round(float(nw[1]), 4),
        })

    out = pd.DataFrame(rows)
    print("=" * 118)
    print("許鈞翔 (2025) 差異第五項：進場時點（回歸式 − 突破式，四檔止損等權，逐日）")
    print("=" * 118)
    print(out.to_string(index=False))

    path = f"{OUT_DIR}/hsu25_entry_timing.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[已存] {path}")

    # ── 逐格明細：回歸式進場必然減少訊號數，動用資本口徑會因此改變 ──────
    # 主檢定走承諾資本（Ann_Ret_Raw），與論文其餘各表一致；但兩種進場時點的
    # 曝險本來就不同，故一併列出 REC（動用資本）與利用率，讓「少做幾趟」與
    # 「每趟賺多少」分開看得見。許鈞翔的論文正是報 R^EC。
    import sqlite3
    with sqlite3.connect("file:results/result.db?mode=ro", uri=True) as con:
        det = pd.read_sql(
            "SELECT METHOD, \"STOP LOSS %\" AS 止損, Ann_Ret_Raw AS 年化RCC,"
            " REC_Raw AS 年化REC, Avg_Utilization AS 利用率, Entries AS 進場次數,"
            " Win_Rate AS 勝率, Sharpe_Raw AS Sharpe"
            " FROM strategy_summaries WHERE METHOD LIKE 'HSU25 (%'"
            " ORDER BY METHOD, CAST(REPLACE(\"STOP LOSS %\",'%','') AS INTEGER)", con)
    print("\n" + "=" * 118)
    print("逐格明細（承諾資本 RCC vs 動用資本 REC）")
    print("=" * 118)
    print(det.to_string(index=False))
    det.to_csv(f"{OUT_DIR}/hsu25_entry_timing_cells.csv", index=False, encoding="utf-8-sig")

    # ── 判讀 ────────────────────────────────────────────────────────
    print("\n" + "─" * 118)
    sig = out[out["p(bootstrap)"] < 0.05]
    if sig.empty:
        print("三組排序全部未達 5% 顯著 → 進場時點單獨無法解釋兩份回測的差距，"
              "應往其餘七項（尤以成本 0.2% vs 0.58%、報酬口徑）找。")
    else:
        for _, r in sig.iterrows():
            direction = "回歸式較優" if r["年化Δ%"] > 0 else "突破式較優"
            print(f"{r['排序']}：{direction} {r['年化Δ%']:+.3f} pp"
                  f"  95% CI [{r['CI下界pp']:+.3f}, {r['CI上界pp']:+.3f}]"
                  f"  p = {r['p(bootstrap)']:.4f}")
        print(f"（{len(sig)}/3 組達顯著；未達者見上表）")
    print("─" * 118)
    print("提醒：本檢定只隔離第五項。與許鈞翔實際回測數字的差距仍含其餘七項，不可據此收斂。")


if __name__ == "__main__":
    run()
