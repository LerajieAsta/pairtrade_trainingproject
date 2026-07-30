# -*- coding: utf-8 -*-
"""
命題 1 殘差之三：特徵維度（Han et al. 的「78 公司特徵」可得子集）
======================================================================
形成期實作差異（`proposition1_mechanism`）與交易機制（`prop1_han_chain`）皆已檢驗
且皆非命題 1 失敗的原因。剩餘兩個殘差為**特徵維度**與**母體範圍**（後者無資料
不可測）。本模組檢驗前者。

本研究原為 7 維連續特徵（5 報酬 PCA + log 市值 + 盈餘殖利率），Han, He & Toh
(2021) 為 48 動量因子 + 78 公司特徵。自既有 SEC companyfacts 快取解出 40 個
Green et al. 風格特徵後，**僅 10 個在 PIT 成分股身分分母下覆蓋率 >70%**
（見 `fetch/fetch_sec_characteristics.py`），連續維度 7 → 17。

單變因對照鏈（每步僅改一項）
----------------------------
    Grid (AGG-SSD)            7 維，one-hot=1.0，產業中位數插補   ← 現行主軸
    Grid (AGG-SSD-NOSEC)      7 維，one-hot=0，  產業中位數插補
    Grid (AGG-SSD-NOSEC-GI)   7 維，one-hot=0，  **全域插補**
    Grid (AGG-SSD-CHARS)      **17 維**，one-hot=0，全域插補

CHARS − NOSEC-GI 即「+10 個公司特徵」的淨效果。若直接拿 CHARS 對比 NOSEC，
會同時改動特徵數與插補方式而無法歸因。

為何插補方式自成一步
--------------------
`impute_by_group` 以**產業中位數**填補缺失，在缺失率 20–30% 時等同為缺資料的股票
額外加上產業標籤——與 `sector_onehot_weight` 是同一種混淆。實測證實：僅把插補
改為全域（不動任何特徵），跨產業配對比例即由 39.6% 升至 57.7%。故它是產業先驗的
第二條隱藏管道，必須單獨計入。

分析期間
--------
SEC XBRL 自 ~2009 起才有資料、覆蓋率至 2012 才穩定。各策略皆以全期執行（與基準
共用期間定義），但**績效比較限制在 2012+**——特徵實際存在的期間。全期數字一併
報出以顯示稀釋程度。

用法：python -m analysis.prop1_feature_dimension
"""
import os
import sys

import numpy as np
import pandas as pd

from analysis.proposition1_daily_hac import _bh_adjust
from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, TRADING_DAYS,
    baseline_only, load_daily_sids, method_paths, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WINDOW_START = "2012-01-01"

CHAIN = [
    ("現行主軸",       "Grid (AGG-SSD)",           "7 維，one-hot=1.0，產業插補"),
    ("拿掉 one-hot",   "Grid (AGG-SSD-NOSEC)",     "one-hot → 0"),
    ("改全域插補",     "Grid (AGG-SSD-NOSEC-GI)",  "產業插補 → 全域插補"),
    ("+10 公司特徵",   "Grid (AGG-SSD-CHARS)",     "7 維 → 17 維連續"),
]
GICS = "Grid (GICS-SSD)"          # 命題 1 的對照基準


def _series(method: str, top1: bool = False):
    ids = baseline_only(method_paths([method])._path.tolist())
    if top1:
        ids = [s for s in ids if "Top1_SL0_" in os.path.basename(s)]
    if not ids:
        return None
    px = load_daily_sids(ids)
    cols = [i for i in ids if i in px.columns]
    return px[cols].mean(axis=1) if cols else None


def _ann(s) -> float:
    return float(np.asarray(s).mean()) * TRADING_DAYS / INITIAL_CAPITAL * 100


def _diff_test(a, b, start=None):
    """b − a 的逐日差分 HAC；start 給定時只取該日之後。"""
    j = pd.concat([b.rename("t"), a.rename("c")], axis=1).fillna(0.0)
    if start:
        j = j[j.index >= start]
    d = (j["t"] - j["c"]).values
    t, p, _ = newey_west(d)
    return _ann(d), t, p, len(d)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    ser = {m: _series(m) for _, m, _ in CHAIN}
    ser[GICS] = _series(GICS)
    missing = [m for m, s in ser.items() if s is None]
    if missing:
        print(f"⚠ 缺資料：{missing}")
        return

    # ── 表一：各步絕對績效（全期 vs 2012+）──
    rows = []
    for tag, m, what in CHAIN:
        s = ser[m]
        rows.append({"步驟": tag, "策略": m, "該步改變": what,
                     "全期年化%": round(_ann(s), 3),
                     f"{WINDOW_START[:4]}+年化%": round(_ann(s[s.index >= WINDOW_START]), 3)})
    g = ser[GICS]
    rows.append({"步驟": "對照", "策略": GICS, "該步改變": "GICS 產業分組",
                 "全期年化%": round(_ann(g), 3),
                 f"{WINDOW_START[:4]}+年化%": round(_ann(g[g.index >= WINDOW_START]), 3)})
    t1 = pd.DataFrame(rows)

    # ── 表二：逐步單變因對照（2012+）──
    crows = []
    for (ta, ma, _), (tb, mb, wb) in zip(CHAIN, CHAIN[1:]):
        ann, t, p, n = _diff_test(ser[ma], ser[mb], WINDOW_START)
        crows.append({"步驟": f"{ta}→{tb}", "改變": wb, "年化Δ%": round(ann, 3),
                      "交易日": n, "NW t": round(t, 3), "NW p": round(p, 4)})
    t2 = pd.DataFrame(crows)
    t2["BH校正p"] = _bh_adjust(t2["NW p"].values).round(4)
    t2["校正後顯著"] = np.where(t2["BH校正p"] < 0.05, "✔", "✘")

    # ── 表三：各步 vs GICS（命題 1 的直接檢定，2012+）──
    grows = []
    for tag, m, _ in CHAIN:
        ann, t, p, n = _diff_test(ser[GICS], ser[m], WINDOW_START)
        grows.append({"ML 設定": tag, "年化Δ%(ML−GICS)": round(ann, 3),
                      "NW t": round(t, 3), "NW p": round(p, 4)})
    t3 = pd.DataFrame(grows)
    t3["BH校正p"] = _bh_adjust(t3["NW p"].values).round(4)
    t3["校正後顯著"] = np.where(t3["BH校正p"] < 0.05, "✔", "✘")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print("表一：特徵維度歸因鏈的絕對績效")
    print("=" * 100)
    print(t1.to_string(index=False))
    print(f"\n（SEC XBRL 自 ~2009 起，覆蓋率至 2012 才穩定；全期數字含大量特徵全缺的年份）")

    print("\n" + "=" * 100)
    print(f"表二：逐步單變因對照（{WINDOW_START[:4]}+，逐日差分 HAC，BH 校正）")
    print("=" * 100)
    print(t2.to_string(index=False))

    print("\n" + "=" * 100)
    print(f"表三：各設定 vs GICS —— 命題 1 的直接檢定（{WINDOW_START[:4]}+）")
    print("=" * 100)
    print(t3.to_string(index=False))

    t1.to_csv(f"{OUT_DIR}/prop1_featdim_cells.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(f"{OUT_DIR}/prop1_featdim_steps.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(f"{OUT_DIR}/prop1_featdim_vs_gics.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop1_featdim_{{cells,steps,vs_gics}}.csv")


if __name__ == "__main__":
    run()
