# -*- coding: utf-8 -*-
"""
命題 1 主檢定：ML 分群 vs GICS 產業分組（逐日報酬差 + 循環 block bootstrap）
======================================================================
與 `proposition2_daily_hac` 同一套方法論，套用到形成期命題。

抽樣單位為何是「日」而非「參數格」
----------------------------------
舊版以「15 個參數格」為抽樣單位的配對 t 檢定屬偽重複（pseudo-replication）：
那 15 格是同一份資料、同一段期間、同一批配對的 15 種組合設定，有效樣本數
≈ 1 條回測路徑。命題 1 的舊結論「9 組比較中 5 組顯著劣於 GICS」建立在同一個
無效基礎上，故一併重做。改以時間為抽樣單位，並用 block bootstrap 處理
重疊部位造成的自相關（方法與 L=126 的理由見 `analysis.block_bootstrap`）。

設計：3×3 消融矩陣的直接對照
----------------------------
固定**排序準則**與**交易端**，唯一變因為分組方法：

    {HDBSCAN, Agglomerative, K-means} × {SSD, DTW, SSD-DTW-PCA}
        ↕ 對照
    GICS 產業分組            × {SSD, DTW, SSD-DTW-PCA}

共 9 組比較。差分方向 Δr = r_ML − r_GICS，**正值支持命題 1**。

信賴區間取代非劣性檢定
----------------------
雙尾不顯著 ≠ 兩者相當。舊版另做 TOST 式非劣性檢定（δ ∈ {0.25, 0.5, 1.0} pp）
來處理這件事，但那需要事前指定並辯護一個實質等價邊界。95% CI 承載同樣的資訊
且不必選 δ：區間涵蓋 0 表示無法宣稱有差異，而區間**有多寬**直接顯示檢定力——
若兩端都大到具實質意義，就是「檢定力不足」而非「兩者相當」。

多重檢定
--------
9 組比較若各自看 p<0.05，純靠運氣就有約 37% 的機率至少出現一個假陽性
（1 − 0.95^9）。本模組以 Benjamini-Hochberg 控制 FDR，同時報原始 p 與校正後 p。

用法：python -m analysis.proposition1_daily_hac
"""
import os
import sys

import numpy as np
import pandas as pd

from analysis.block_bootstrap import BLOCK_L, bh_adjust, bootstrap_test
from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, TRADING_DAYS, _grid_cell, baseline_only,
    load_daily_sids, method_paths, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLUSTERS = [("HDBSCAN", "HDB"), ("Agglomerative", "AGG"), ("K-means", "KM")]
RANKINGS = [("SSD", "SSD"), ("DTW", "DTW"), ("SSD-DTW-PCA", "SDP")]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    methods = [f"Grid ({c}-{r})" for _, c in CLUSTERS for _, r in RANKINGS]
    methods += [f"Grid (GICS-{r})" for _, r in RANKINGS]
    meta = method_paths(methods)
    m2s = {m: baseline_only(g._path.tolist()) for m, g in meta.groupby("METHOD")}
    px = load_daily_sids([s for v in m2s.values() for s in v])

    rows = []
    for cl_name, cl in CLUSTERS:
        for rk_name, rk in RANKINGS:
            ml_m, gics_m = f"Grid ({cl}-{rk})", f"Grid (GICS-{rk})"
            ml_ids, g_ids = m2s.get(ml_m, []), m2s.get(gics_m, [])
            if not ml_ids or not g_ids:
                print(f"  ⚠ 略過 {ml_m} vs {gics_m}：缺策略")
                continue
            mc = {_grid_cell(s): s for s in ml_ids}
            gc = {_grid_cell(s): s for s in g_ids}
            cells = sorted(set(mc) & set(gc))
            if not cells:
                continue

            d = (px[[mc[c] for c in cells]].mean(axis=1)
                 - px[[gc[c] for c in cells]].mean(axis=1)).values

            res = bootstrap_test(d)
            # HAC 對照欄：不入論文主表，僅供「兩法結論一致」的註腳引用
            _, p_nw, _ = newey_west(d)

            rows.append({"分群": cl_name, "排序": rk_name, "格數": len(cells),
                         **res,
                         "方向": "ML優" if res["年化Δ%"] > 0 else "GICS優",
                         "NW p（對照）": round(p_nw, 4)})

    res = pd.DataFrame(rows)
    if res.empty:
        print("⚠ 無可用資料")
        return

    res["BH校正p"] = bh_adjust(res["BB p"].values).round(4)
    res["5%顯著(校正後)"] = np.where(res["BH校正p"] < 0.05, "✔", "✘")
    res = res.sort_values("年化Δ%", ascending=False)

    pd.set_option("display.width", 260)
    print("\n" + "=" * 104)
    print("命題 1：ML 分群 vs GICS 產業分組（等權組合逐日差分 + block bootstrap）")
    print(f"        Δ = ML − GICS，正值支持命題 1；L={BLOCK_L}，10,000 次重抽")
    print("=" * 104)
    cols = ["分群", "排序", "年化Δ%", "方向", "CI下界", "CI上界",
            "BB p", "BH校正p", "5%顯著(校正後)", "NW p（對照）"]
    print(res[cols].to_string(index=False))

    n_ml, n_gics = int((res.方向 == "ML優").sum()), int((res.方向 == "GICS優").sum())
    n_sig = int((res["BH校正p"] < 0.05).sum())
    widest = res.loc[(res["CI上界"] - res["CI下界"]).idxmax()]
    print("\n" + "=" * 104)
    print(f"總結：{len(res)} 組比較中，方向上 ML 優 {n_ml} 組、GICS 優 {n_gics} 組；"
          f"BH 校正後顯著者 {n_sig} 組。")
    print(f"      最寬區間 {widest['分群']}×{widest['排序']}："
          f"[{widest['CI下界']:+.2f}, {widest['CI上界']:+.2f}] pp——"
          f"區間兩端皆具實質意義即代表檢定力不足，")
    print("      此時「不顯著」不可解讀為「兩者相當」。")
    print("=" * 104)

    res.to_csv(f"{OUT_DIR}/prop1_daily_hac.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop1_daily_hac.csv")


if __name__ == "__main__":
    run()
