# -*- coding: utf-8 -*-
"""
F09 結構性財報特徵消融的重驗：插補方式是否稀釋了處理效果？
======================================================================
原始 F09（2026-07-24，已封存）結論：「10 維 SEC XBRL 財報比率對分群品質無顯著
貢獻，三分群 Δ 皆在 ±0.22pp 噪音範圍內」。該實驗以 `impute_scope="group"`
（產業中位數插補）執行。

重驗動機
--------
結構性特徵在 2012+ 有 30–50% 缺失，被填成**產業中位數**——而產業資訊**已由
sector_onehot（權重 1.0）編碼**。那些插補值因此是冗餘資訊，使結構性區塊的邊際
貢獻被系統性低估。

註：one-hot 於 BASE / STRUCT 兩臂皆為 1.0，在差分中對消，本身不造成偏誤。
    問題出在**插補值的冗餘**，不是 one-hot。

形成期已證實此機制（2012+ 跨產業配對比例）：

    臂            產業插補 → 全域插補     差
    HDB-BASE        22.3% → 35.4%      +13.1
    HDB-STRUCT      50.5% → 70.3%      +19.8
    AGG-BASE         9.5% → 16.5%       +7.0
    AGG-STRUCT      15.7% → 59.3%      +43.6   ← 最戲劇性
    KM-BASE          9.9% → 15.4%       +5.5
    KM-STRUCT       13.9% → 36.1%      +22.2

插補的影響在 STRUCT 臂遠大於 BASE 臂，正是預測的方向：**在產業插補下，那 10 個
特徵大部分被自己的插補值變成 one-hot 的冗餘，分群幾乎沒動——處理沒有真正施加。**

本模組檢定該形成期差異是否轉化為績效差異。分析限制在 2012+（XBRL 覆蓋率穩定期），
與 `prop1_feature_dimension` 同口徑。

用法：python -m analysis.prop1_f09_reverify
"""
import os
import sys

import numpy as np
import pandas as pd

from analysis.block_bootstrap import bh_adjust as _bh_adjust, bootstrap_test
from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, TRADING_DAYS,
    baseline_only, load_daily_sids, method_paths, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WINDOW_START = "2012-01-01"
CLUSTERS = [("HDB", "hdbscan"), ("AGG", "agglomerative"), ("KM", "kmeans")]


def _series(method: str):
    ids = baseline_only(method_paths([method])._path.tolist())
    if not ids:
        return None
    px = load_daily_sids(ids)
    cols = [i for i in ids if i in px.columns]
    return px[cols].mean(axis=1) if cols else None


def _ann(x) -> float:
    return float(np.asarray(x).mean()) * TRADING_DAYS / INITIAL_CAPITAL * 100


def _diff(a, b, start=WINDOW_START):
    j = pd.concat([b.rename("t"), a.rename("c")], axis=1).fillna(0.0)
    j = j[j.index >= start]
    d = (j["t"] - j["c"]).values
    # 主檢定同 4.1／4.2：block bootstrap；NW 保留為對照欄
    _, p_nw, _ = newey_west(d)
    return _ann(d), p_nw, bootstrap_test(d)["BB p"], len(d)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    need = [f"{pre} ({cs}-{arm})"
            for pre in ("F09", "F09GI") for cs, _ in CLUSTERS for arm in ("BASE", "STRUCT")]
    ser = {m: _series(m) for m in need}
    if any(s is None for s in ser.values()):
        print("⚠ 缺資料：", [m for m, s in ser.items() if s is None])
        return

    # ── 表一：STRUCT − BASE（結構性特徵的處理效果），兩種插補各自計算 ──
    rows = []
    for cs, _ in CLUSTERS:
        for pre, lbl in (("F09", "產業插補（原始）"), ("F09GI", "全域插補（重驗）")):
            b, s = ser[f"{pre} ({cs}-BASE)"], ser[f"{pre} ({cs}-STRUCT)"]
            ann, p_nw, p, n = _diff(b, s)
            rows.append({"分群": cs, "插補": lbl,
                         "BASE 年化%": round(_ann(b[b.index >= WINDOW_START]), 3),
                         "STRUCT 年化%": round(_ann(s[s.index >= WINDOW_START]), 3),
                         "處理效果Δ%": round(ann, 3),
                         "BB p": round(p, 4), "NW p（對照）": round(p_nw, 4)})
    t1 = pd.DataFrame(rows)
    t1["BH校正p"] = _bh_adjust(t1["BB p"].values).round(4)
    t1["校正後顯著"] = np.where(t1["BH校正p"] < 0.05, "✔", "✘")

    # ── 表二：同一臂下「全域插補 − 產業插補」（插補方式本身的效果）──
    rows2 = []
    for cs, _ in CLUSTERS:
        for arm in ("BASE", "STRUCT"):
            g, gi = ser[f"F09 ({cs}-{arm})"], ser[f"F09GI ({cs}-{arm})"]
            ann, p_nw, p, n = _diff(g, gi)
            rows2.append({"臂": f"{cs}-{arm}", "年化Δ%(全域−產業)": round(ann, 3),
                          "BB p": round(p, 4), "NW p（對照）": round(p_nw, 4)})
    t2 = pd.DataFrame(rows2)
    t2["BH校正p"] = _bh_adjust(t2["BB p"].values).round(4)
    t2["校正後顯著"] = np.where(t2["BH校正p"] < 0.05, "✔", "✘")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 104)
    print(f"表一：結構性特徵的處理效果（STRUCT − BASE），{WINDOW_START[:4]}+")
    print("      原始 F09 結論為「三分群 Δ 皆在 ±0.22pp 噪音範圍」——下表逐一重驗")
    print("=" * 104)
    print(t1.to_string(index=False))

    print("\n" + "=" * 104)
    print(f"表二：插補方式本身的效果（全域 − 產業中位數），{WINDOW_START[:4]}+")
    print("=" * 104)
    print(t2.to_string(index=False))

    n_sig = int((t1["BH校正p"] < 0.05).sum())
    print(f"\n表一 BH 校正後顯著 {n_sig}/{len(t1)}；表二 {int((t2['BH校正p'] < 0.05).sum())}/{len(t2)}")

    t1.to_csv(f"{OUT_DIR}/prop1_f09_treatment.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(f"{OUT_DIR}/prop1_f09_impute.csv", index=False, encoding="utf-8-sig")
    print(f"→ {OUT_DIR}/prop1_f09_{{treatment,impute}}.csv")


if __name__ == "__main__":
    run()
