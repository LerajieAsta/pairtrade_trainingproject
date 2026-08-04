# -*- coding: utf-8 -*-
"""
Han et al. (2021) 交易機制的逐步歸因鏈
======================================================================
`proposition1_mechanism` 的因子設計證實：三處**形成期**實作差異（產業 one-hot、
共整合篩選、缺不分組零點）不足以解釋命題 1 的否定。剩餘兩個殘差為母體範圍
（S&P 500 vs CRSP，無資料不可測）與**交易機制**。本模組檢驗後者。

Han, He & Toh (2021) 的機制與本研究的四項差異，逐步施加以維持單變因：

    起點  Grid (AGG-SSD-NF)   AGG 分群 + SSD 距離 + 無篩選 + OLS-β + z>2 / 126 日
     ②   Grid (HAN2-B1)      β 改 1 等金額（distance_trading；借用同一批配對）
     ③   Grid (HAN3-REV)      選對準則改「群內月報酬發散」（reversal backend）
     ④   Grid (HAN4-MONTHLY)  21 日窗 + 發散即建倉(entry_z=0) + 持有至期末

④ 完成即 Han et al. 的交易端全貌。逐步設計使每一步的效果可獨立歸因——
直接做完整復刻會一次改四個變因，與 SSD (Basic) 的舊錯誤同型。

⚠ 即使 ④ 完美復刻，仍有兩個缺口無法關閉：分群依據為 7 維連續特徵
   （原文 48 動量因子 + 78 公司特徵），母體為 S&P 500（原文 CRSP 全市場）。
   故本鏈只能量化「交易機制解釋了多少差距」，不應預期複製原文的 24.8%。

用法：python -m analysis.prop1_han_chain
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

# (步驟標籤, db_method, 該步改變了什麼)
CHAIN = [
    ("起點", "Grid (AGG-SSD-NF)",   "AGG 分群 + SSD 距離 + 無篩選 + OLS-β + z>2/126日"),
    ("②",   "Grid (HAN2-B1)",      "β 改 1 等金額"),
    ("③",   "Grid (HAN3-REV)",     "選對準則改月報酬發散"),
    ("④",   "Grid (HAN4-MONTHLY)", "21日窗 + 發散即建倉 + 持有至期末"),
]


def _series(method: str, top1: bool):
    """等權組合（15 格）或 Top1/SL0 單格的逐日損益。

    entry_z=0 的變體檔名帶 _EZ0_DSZ0 後綴，故不能用 baseline_only 過濾——
    改以「排除其他實驗維度」的方式取該 METHOD 的全部網格。
    """
    ids = method_paths([method])._path.tolist()
    ids = [s for s in ids
           if not any(k in os.path.basename(s) for k in ("_DYN", "_MHD", "_XZ", "_DG"))]
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

    # ── 表一：鏈上各步的絕對績效 ──
    rows, series = [], {}
    for tag, m, what in CHAIN:
        ew, t1 = _series(m, False), _series(m, True)
        series[m] = (ew, t1)
        rows.append({
            "步驟": tag, "策略": m, "該步改變": what,
            "全網格等權年化%": round(_ann(ew), 3) if ew is not None else None,
            "Top1/SL0年化%": round(_ann(t1), 3) if t1 is not None else None,
        })
    t1_df = pd.DataFrame(rows)

    # ── 表二：逐步單變因對照（相鄰兩步的逐日差分 block bootstrap）──
    crows = []
    for (tag_a, m_a, _), (tag_b, m_b, what_b) in zip(CHAIN, CHAIN[1:]):
        for scope, idx in (("全網格等權", 0), ("Top1/SL0", 1)):
            a, b = series[m_a][idx], series[m_b][idx]
            if a is None or b is None:
                continue
            # 對齊日期（HAN4 期數不同）；未持倉日補 0
            j = pd.concat([b.rename("t"), a.rename("c")], axis=1).fillna(0.0)
            d = (j["t"] - j["c"]).values
            p = bootstrap_test(d)["BB p"]
            _, p_nw, _ = newey_west(d)
            crows.append({
                "步驟": f"{tag_a}→{tag_b}", "改變": what_b, "口徑": scope,
                "年化Δ%": round(float(d.mean()) * TRADING_DAYS / INITIAL_CAPITAL * 100, 3),
                "交易日": len(d), "BB p": round(p, 4), "NW p（對照）": round(p_nw, 4),
            })
    t2_df = pd.DataFrame(crows)
    if not t2_df.empty:
        for scope in t2_df["口徑"].unique():
            m = t2_df["口徑"] == scope
            t2_df.loc[m, "BH校正p"] = _bh_adjust(t2_df.loc[m, "BB p"].values).round(4)
        t2_df["校正後顯著"] = np.where(t2_df["BH校正p"] < 0.05, "✔", "✘")

    # ── 表三：完整復刻 vs 起點（總效果）──
    trows = []
    for scope, idx in (("全網格等權", 0), ("Top1/SL0", 1)):
        a, b = series[CHAIN[0][1]][idx], series[CHAIN[-1][1]][idx]
        if a is None or b is None:
            continue
        j = pd.concat([b.rename("t"), a.rename("c")], axis=1).fillna(0.0)
        d = (j["t"] - j["c"]).values
        p = bootstrap_test(d)["BB p"]
        _, p_nw, _ = newey_west(d)
        trows.append({"對照": "④ 完整復刻 − 起點", "口徑": scope,
                      "年化Δ%": round(float(d.mean()) * TRADING_DAYS / INITIAL_CAPITAL * 100, 3),
                      "BB p": round(p, 4), "NW p（對照）": round(p_nw, 4)})
    t3_df = pd.DataFrame(trows)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 100)
    print("表一：歸因鏈各步的絕對績效")
    print("=" * 100)
    print(t1_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("表二：逐步單變因對照（逐日報酬差 + block bootstrap，BH 校正）")
    print("=" * 100)
    print(t2_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("表三：完整復刻的總效果")
    print("=" * 100)
    print(t3_df.to_string(index=False))
    print("\n⚠ 分群依據仍為 7 維連續特徵（原文 126 維）、母體仍為 S&P 500（原文 CRSP）。")
    print("  本鏈僅量化交易機制的貢獻，不應預期複製原文 24.8% 的績效。")

    t1_df.to_csv(f"{OUT_DIR}/prop1_han_chain_cells.csv", index=False, encoding="utf-8-sig")
    t2_df.to_csv(f"{OUT_DIR}/prop1_han_chain_steps.csv", index=False, encoding="utf-8-sig")
    t3_df.to_csv(f"{OUT_DIR}/prop1_han_chain_total.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop1_han_chain_{{cells,steps,total}}.csv")


if __name__ == "__main__":
    run()
