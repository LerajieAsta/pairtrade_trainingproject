# -*- coding: utf-8 -*-
"""
命題 1 主檢定：ML 分群 vs GICS 產業分組（逐日報酬差 + Newey-West HAC）
======================================================================
與 `proposition2_daily_hac` 同一套方法論，套用到形成期命題，理由見該檔 docstring：
舊版以「15 個參數格」為抽樣單位的配對 t 檢定屬偽重複（pseudo-replication），
有效樣本數 ≈ 1 條回測路徑。命題 1 的舊結論「9 組比較中 5 組顯著劣於 GICS」
建立在同一個無效基礎上，故一併重做。

設計：3×3 消融矩陣的直接對照
----------------------------
固定**排序準則**與**交易端**，唯一變因為分組方法：

    {HDBSCAN, Agglomerative, K-means} × {SSD, DTW, SSD-DTW-PCA}
        ↕ 對照
    GICS 產業分組            × {SSD, DTW, SSD-DTW-PCA}

共 9 組比較。差分方向 Δr = r_ML − r_GICS，**正值支持命題 1**。

多重檢定校正
------------
9 組比較若各自看 p<0.05，純靠運氣就有約 37% 的機率至少出現一個假陽性
（1 − 0.95^9）。舊版「5 組顯著」的宣稱未經校正，不可直接引用。
本模組以 Benjamini-Hochberg 控制 FDR，同時報原始 p 與校正後 p。

註：命題 2 只有 5 組比較且原始 p 全部 ≤0.0177，BH 校正不改變結論，
    故該處未特別強調；此處 9 組且結論本身就是在數「幾組顯著」，校正為必要。

用法：python -m analysis.proposition1_daily_hac
"""
import os
import sys

import numpy as np
import pandas as pd

from analysis.prop2_block_bootstrap import circular_block_bootstrap_means
from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, TRADING_DAYS, _grid_cell, baseline_only,
    load_daily_sids, method_paths, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLUSTERS = [("HDBSCAN", "HDB"), ("Agglomerative", "AGG"), ("K-means", "KM")]
RANKINGS = [("SSD", "SSD"), ("DTW", "DTW"), ("SSD-DTW-PCA", "SDP")]
LAG_SPECS = ["auto", 63, 126, 252]
BLOCK_L = 126
N_BOOT = 5000
SEED = 20260729

# 非劣性邊界（年化報酬百分點）。參照尺度：GICS 臂等權組合的年化報酬約 1.2–1.7%，
# 故 δ=0.25% 約為其 1/6（嚴格）、δ=1.0% 約為其 2/3（寬鬆到近乎無意義）。
NI_MARGINS = [0.25, 0.5, 1.0]


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg 校正後 p 值（step-up，保單調）。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

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
            t, p, lag = newey_west(d)
            obs = float(d.mean())

            # 無母數對照（block bootstrap，L=126＝一個完整持有期）
            bb = circular_block_bootstrap_means(d - obs, BLOCK_L, N_BOOT, rng)
            p_bb = float((np.abs(bb) >= abs(obs)).mean())

            rec = {"分群": cl_name, "排序": rk_name,
                   "年化Δ%": round(obs * TRADING_DAYS / INITIAL_CAPITAL * 100, 3),
                   "方向": "ML優" if obs > 0 else "GICS優",
                   "NW t": round(t, 3), "NW p": round(p, 4),
                   f"BB p(L={BLOCK_L})": round(p_bb, 4), "落後階": lag}
            for spec in LAG_SPECS:
                if spec != "auto":
                    _, p2, _ = newey_west(d, lags=spec)
                    rec[f"p@lag{spec}"] = round(p2, 4)
            rows.append(rec)

    res = pd.DataFrame(rows)
    if res.empty:
        print("⚠ 無可用資料")
        return

    # ── 非劣性檢定 ────────────────────────────────────────────────
    # 雙尾檢定不顯著 ≠ 兩者相當。要主張「ML 不輸 GICS」須做非劣性檢定：
    #   H0（劣性）：μ_ML − μ_GICS ≤ −δ      H1（非劣）：μ_ML − μ_GICS > −δ
    # 於單尾 95% 信賴下界 > −δ 時拒絕 H0。δ 須事前指定且有實質意義，
    # 此處以年化報酬百分點表示，並附 GICS 臂自身的年化報酬作為尺度參照。
    res["年化SE%"] = (res["年化Δ%"] / res["NW t"]).abs().round(3)
    res["單尾95%下界"] = (res["年化Δ%"] - 1.645 * res["年化SE%"]).round(3)
    for delta in NI_MARGINS:
        res[f"非劣@δ={delta}%"] = np.where(res["單尾95%下界"] > -delta, "✔", "✘")

    res["BH校正p"] = _bh_adjust(res["NW p"].values).round(4)
    res["5%顯著(校正後)"] = np.where(res["BH校正p"] < 0.05, "✔", "✘")

    pd.set_option("display.width", 260)
    print("\n" + "=" * 104)
    print("命題 1：ML 分群 vs GICS 產業分組（等權組合逐日差分 HAC；Δ = ML − GICS，正值支持命題1）")
    print("=" * 104)
    cols = ["分群", "排序", "年化Δ%", "方向", "NW t", "NW p", "BH校正p",
            "5%顯著(校正後)", f"BB p(L={BLOCK_L})"]
    print(res[cols].to_string(index=False))

    print("\n--- 落後階敏感度（原始 p，未校正）")
    print(res[["分群", "排序", "NW p", "p@lag63", "p@lag126", "p@lag252"]].to_string(index=False))

    print("\n" + "=" * 104)
    print("非劣性檢定：H0（劣性）μ_ML − μ_GICS ≤ −δ；單尾 95% 下界 > −δ 方可主張「不輸」")
    print("=" * 104)
    ni_cols = ["分群", "排序", "年化Δ%", "年化SE%", "單尾95%下界"] + \
              [f"非劣@δ={d}%" for d in NI_MARGINS]
    print(res[ni_cols].to_string(index=False))
    for d in NI_MARGINS:
        k = int((res[f"非劣@δ={d}%"] == "✔").sum())
        print(f"  δ={d}%：9 組中 {k} 組可主張非劣")

    n_ml, n_gics = int((res.方向 == "ML優").sum()), int((res.方向 == "GICS優").sum())
    sig = res[res["BH校正p"] < 0.05]
    sig_ml = int((sig.方向 == "ML優").sum())
    sig_gics = int((sig.方向 == "GICS優").sum())
    print("\n" + "=" * 104)
    print(f"總結：{len(res)} 組比較中，方向上 ML 優 {n_ml} 組、GICS 優 {n_gics} 組；")
    print(f"      BH 校正後顯著者 {len(sig)} 組（ML 顯著較優 {sig_ml} 組、GICS 顯著較優 {sig_gics} 組）。")
    print("=" * 104)

    res.to_csv(f"{OUT_DIR}/prop1_daily_hac.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop1_daily_hac.csv")


if __name__ == "__main__":
    run()
