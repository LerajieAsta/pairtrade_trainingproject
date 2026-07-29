# -*- coding: utf-8 -*-
"""
命題 1 機制解析：分組 × 篩選 × 產業先驗的因子設計
======================================================================
命題 1 的動機來自 Han, He & Toh (2021) *Pairs Trading via Unsupervised Learning*
（CRSP 全美股、48 動量因子 + 78 公司特徵分群、群內做多低估/做空高估、
**不施加共整合篩選**，並明文指出跨產業發散亦為利潤來源）。

本研究的實作與其有三處結構性差異，各自都可能單獨壓抑該假說，且在
`proposition1_daily_hac` 的 3×3 對照中全被固定住、從未消融：

  (a) 特徵含 12 維 GICS 產業 one-hot（權重 1.0）
  (b) 施加 ADF + 半衰期 + Hurst 共整合篩選
  (c) 消融矩陣的「分組」維度缺「不分組」零點

本模組解析 2×2×2 因子設計（排序固定 SSD）的三個主效果與交互作用。

形成期已觀測到的結構差異（見 run() 輸出表一）：
  - 產業先驗強度形成單調階梯：跨產業比例 0% → 10.7% → 39.6% → 75.3%
  - 候選池飢餓是**分群 × 篩選的交互作用**，非任一單獨因子：
    分群+篩選 11.3–11.5/20，但不分組+篩選 19.4/20（幾乎填滿）

⚠ 混淆警告：各格的期均配對數不同（11.3 ~ 20.0）。「不分組」績效若較佳，
   可能只是名額填得較滿而非配對品質較高。故本模組同時報告 Top1（不受名額
   影響）與全網格等權，兩者結論不一致時以 Top1 為準並明白揭露。

用法：python -m analysis.proposition1_mechanism
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, RESULT_DB, TRADING_DAYS,
    baseline_only, load_daily_sids, method_paths, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FORMATION_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
PRICE_DB = "./dataset/price/sp500_Tiingo.db"

# (標籤, db_method, formation strategy_id, 分組, 篩選, one-hot 權重)
CELLS = [
    ("GICS 產業",      "Grid (GICS-SSD)",          "Grid GICS-SSD_MSR0",          "gics",  "coint", None),
    ("AGG +one-hot",   "Grid (AGG-SSD)",           "Grid AGG-SSD_MSR0",           "agg",   "coint", 1.0),
    ("AGG −one-hot",   "Grid (AGG-SSD-NOSEC)",     "Grid AGG-SSD-NOSEC_MSR0",     "agg",   "coint", 0.0),
    ("不分組",         "Grid (NOGRP-SSD)",         "Grid NOGRP-SSD_MSR0",         "none",  "coint", None),
    ("GICS 產業",      "Grid (GICS-SSD-NF)",       "Grid GICS-SSD-NF_MSR0",       "gics",  "none",  None),
    ("AGG +one-hot",   "Grid (AGG-SSD-NF)",        "Grid AGG-SSD-NF_MSR0",        "agg",   "none",  1.0),
    ("AGG −one-hot",   "Grid (AGG-SSD-NF-NOSEC)",  "Grid AGG-SSD-NF-NOSEC_MSR0",  "agg",   "none",  0.0),
    ("不分組",         "Grid (NOGRP-SSD-NF)",      "Grid NOGRP-SSD-NF_MSR0",      "none",  "none",  None),
]

# 主要對照：每組 (處理, 基準)，皆同篩選層，用來隔離單一因子
CONTRASTS = [
    ("產業先驗：拿掉 one-hot（有篩選）", "Grid (AGG-SSD-NOSEC)",    "Grid (AGG-SSD)"),
    ("產業先驗：拿掉 one-hot（無篩選）", "Grid (AGG-SSD-NF-NOSEC)", "Grid (AGG-SSD-NF)"),
    ("分組：不分組 vs GICS（有篩選）",   "Grid (NOGRP-SSD)",        "Grid (GICS-SSD)"),
    ("分組：不分組 vs GICS（無篩選）",   "Grid (NOGRP-SSD-NF)",     "Grid (GICS-SSD-NF)"),
    ("分組：AGG vs GICS（有篩選）",      "Grid (AGG-SSD)",          "Grid (GICS-SSD)"),
    ("分組：AGG vs GICS（無篩選）",      "Grid (AGG-SSD-NF)",       "Grid (GICS-SSD-NF)"),
    ("篩選：拿掉篩選（GICS）",           "Grid (GICS-SSD-NF)",      "Grid (GICS-SSD)"),
    ("篩選：拿掉篩選（AGG +one-hot）",   "Grid (AGG-SSD-NF)",       "Grid (AGG-SSD)"),
    ("篩選：拿掉篩選（AGG −one-hot）",   "Grid (AGG-SSD-NF-NOSEC)", "Grid (AGG-SSD-NOSEC)"),
    ("篩選：拿掉篩選（不分組）",         "Grid (NOGRP-SSD-NF)",     "Grid (NOGRP-SSD)"),
]


def _formation_stats() -> dict:
    """每個 formation strategy_id 的期均配對數與跨產業比例。"""
    fc = sqlite3.connect(f"file:{FORMATION_DB}?mode=ro", uri=True)
    pc = sqlite3.connect(f"file:{PRICE_DB}?mode=ro", uri=True)
    sec = dict(pd.read_sql("SELECT Symbol,GICS_Sector FROM Constituents", pc).values)
    pc.close()
    out = {}
    for *_, fid, _, _, _ in [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in CELLS]:
        pass
    for c in CELLS:
        fid = c[2]
        d = pd.read_sql("SELECT Ticker_A,Ticker_B,Period_Start FROM formation_pairs "
                        "WHERE strategy_id=?", fc, params=[fid])
        if d.empty:
            out[fid] = (np.nan, np.nan)
            continue
        a, b = d.Ticker_A.map(sec), d.Ticker_B.map(sec)
        cross = ((a != b) & a.notna() & b.notna()).mean()
        out[fid] = (len(d) / d.Period_Start.nunique(), cross)
    fc.close()
    return out


def _series(method: str, top1: bool):
    """等權組合（全 15 格）或單一 Top1/SL0 格的逐日損益。"""
    ids = baseline_only(method_paths([method])._path.tolist())
    if top1:
        ids = [s for s in ids if "Top1_SL0_" in os.path.basename(s)]
    if not ids:
        return None
    px = load_daily_sids(ids)
    cols = [i for i in ids if i in px.columns]
    return px[cols].mean(axis=1) if cols else None


def _ann(s) -> float:
    return float(s.mean()) * TRADING_DAYS / INITIAL_CAPITAL * 100


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    fstats = _formation_stats()

    # ── 表一：各格的形成期結構與績效 ──
    rows = []
    for lbl, m, fid, grp, flt, ohw in CELLS:
        ew, t1 = _series(m, False), _series(m, True)
        pool, cross = fstats.get(fid, (np.nan, np.nan))
        rows.append({
            "分組": lbl, "篩選": flt,
            "one-hot": "—" if ohw is None else f"{ohw:.1f}",
            "期均/20": round(pool, 1) if pool == pool else None,
            "跨產業%": round(cross * 100, 1) if cross == cross else None,
            "全網格等權年化%": round(_ann(ew), 3) if ew is not None else None,
            "Top1/SL0年化%": round(_ann(t1), 3) if t1 is not None else None,
        })
    t1_df = pd.DataFrame(rows)

    # ── 表二：單因子對照（逐日差分 HAC）──
    crows = []
    for name, treat, base in CONTRASTS:
        for scope, top1 in (("全網格等權", False), ("Top1/SL0", True)):
            a, b = _series(treat, top1), _series(base, top1)
            if a is None or b is None:
                continue
            d = (a - b).values
            t, p, _ = newey_west(d)
            crows.append({
                "對照": name, "口徑": scope,
                "年化Δ%": round(float(d.mean()) * TRADING_DAYS / INITIAL_CAPITAL * 100, 3),
                "NW t": round(t, 3), "NW p": round(p, 4),
                "5%顯著": "✔" if p < 0.05 else "✘",
            })
    t2_df = pd.DataFrame(crows)

    # 多重檢定校正：本表為 10 個對照 × 2 種口徑 = 20 個檢定。α=0.05 下純靠運氣
    # 就約有 1 個假陽性（1 − 0.95^20 ≈ 64% 機率至少出現一個），故校正為必要。
    # 於各口徑內分別做 BH（兩口徑非獨立檢定，而是同一組對照的兩種聚合方式）。
    from analysis.proposition1_daily_hac import _bh_adjust
    t2_df["BH校正p"] = np.nan
    for scope in t2_df["口徑"].unique():
        m = t2_df["口徑"] == scope
        t2_df.loc[m, "BH校正p"] = _bh_adjust(t2_df.loc[m, "NW p"].values).round(4)
    t2_df["校正後顯著"] = np.where(t2_df["BH校正p"] < 0.05, "✔", "✘")

    pd.set_option("display.width", 220)
    print("\n" + "=" * 96)
    print("表一：2×2×2 因子設計 —— 形成期結構與績效（排序固定 SSD）")
    print("=" * 96)
    print(t1_df.to_string(index=False))
    print("\n⚠ 期均配對數在各格間不同（11.3~20.0）。「不分組」若較佳可能來自名額填得較滿，")
    print("  而非配對品質。Top1/SL0 只取排名第一的配對，不受名額影響，作為去混淆對照。")

    print("\n" + "=" * 96)
    print("表二：單因子對照（逐日報酬差 + Newey-West HAC）")
    print("=" * 96)
    print(t2_df.pivot(index="對照", columns="口徑",
                      values=["年化Δ%", "NW p", "BH校正p"]).to_string())
    n_raw = int((t2_df["NW p"] < 0.05).sum())
    n_adj = int((t2_df["BH校正p"] < 0.05).sum())
    print(f"\n校正前 5% 顯著 {n_raw}/{len(t2_df)} 個；BH 校正後 {n_adj}/{len(t2_df)} 個。")
    print("（20 個檢定下，α=0.05 純靠運氣約有 64% 機率至少出現一個假陽性。）")

    t1_df.to_csv(f"{OUT_DIR}/prop1_mechanism_cells.csv", index=False, encoding="utf-8-sig")
    t2_df.to_csv(f"{OUT_DIR}/prop1_mechanism_contrasts.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop1_mechanism_{{cells,contrasts}}.csv")


if __name__ == "__main__":
    run()
