"""
組合系統檢定：動態分群 + DRL vs 傳統基準
======================================================================

本研究標題宣稱的是「**結合**無監督動態分群**與**深度強化學習」的策略，
但命題 1 與命題 2 各自只檢定了一個成分：

    命題 1   分群 vs GICS          （兩臂皆搭 Z-Score）
    命題 2   DRL  vs Z-Score       （兩臂共用同一批配對）

兩者都沒有回答「整套系統是否優於整套傳統基準」——那才是標題的宣稱，
也是實務上真正要部署的東西。本模組補上這一檢定：

    完整系統   動態分群（252 日視窗、21 日滾動重估）+ DRL 交易端
    傳統基準   GICS 產業分組 + 固定門檻 Z-Score

並將總效果分解為兩個成分的貢獻，說明改善由誰帶來。

分解恆等式（以 Agglomerative 為例）：

    (AGG+DRL) − (GICS+ZS)  =  [(AGG+DRL) − (AGG+ZS)]   ← DRL 成分
                            + [(AGG+ZS)  − (GICS+ZS)]  ← 分群成分

兩個成分分別對應命題 2 與命題 1，故本節不是新的獨立證據，
而是把既有兩命題重新組裝成標題所宣稱的那個比較。

期間口徑以**全期**為主，與 4.1、4.2 兩節的主檢定一致——本節既是那兩節的
重新組裝，用不同視窗會使分解恆等式對不上正文既有的數字。另附 2012 年後
作為對照（該子期間的兩個成分權重明顯不同）。

三個分群法構成一個檢定家族，以 Benjamini-Hochberg 校正——這與命題 2
逐配對底檢定的處理方式相同。

用法：
    python -m analysis.prop3_combined_system
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.block_bootstrap import bootstrap_test  # noqa: E402
from analysis.proposition2_daily_hac import (  # noqa: E402
    INITIAL_CAPITAL, OUT_DIR, baseline_only, load_daily_sids, method_paths,
    newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRADING_DAYS = 252
# None = 全期（主口徑，同 4.1／4.2）；另跑 2012+ 作對照
WINDOWS = [("全期", None), ("2012+", "2012-01-01")]

# (標籤, 分群+DRL, 分群+Z-Score, 傳統基準)
#
# 基準的排序準則必須與該列相同，否則「分群成分」會同時混入分組方法與排序準則
# 兩個變因，分解就不再是單變因的。HDBSCAN 走 SDP（SSD-DTW-PCA）排序，
# 故配 GICS-SDP；其餘兩者走 SSD，配 GICS-SSD。
COMBOS = [
    ("Agglomerative", "Grid (AGG-SSD-DRL)", "Grid (AGG-SSD)", "Grid (GICS-SSD)"),
    ("HDBSCAN",       "Grid (HDB-SDP-DRL)", "Grid (HDB-SDP)", "Grid (GICS-SDP)"),
    ("K-means",       "Grid (KM-SSD-DRL)",  "Grid (KM-SSD)",  "Grid (GICS-SSD)"),
]


def _series(method: str):
    ids = baseline_only(method_paths([method])._path.tolist())
    if not ids:
        return None
    px = load_daily_sids(ids)
    cols = [i for i in ids if i in px.columns]
    return px[cols].mean(axis=1) if cols else None


def _diff(a, b, start=None) -> np.ndarray:
    """a − b 的逐日差分，未持倉日補 0（非遺漏值，是當日無部位）。"""
    j = pd.concat([a.rename("t"), b.rename("c")], axis=1).fillna(0.0)
    if start:
        j = j[j.index >= start]
    return (j["t"] - j["c"]).values


def _stats(d: np.ndarray) -> dict:
    """主檢定同 4.1／4.2：block bootstrap 的 p 值與 95% CI；NW 僅作對照欄。"""
    mu, sd = d.mean(), d.std(ddof=1)
    res = bootstrap_test(d)
    _, p_nw, _ = newey_west(d)
    return {
        "年化Δ%": res["年化Δ%"],
        "IR": round(float(np.sqrt(TRADING_DAYS) * mu / sd), 3) if sd > 0 else np.nan,
        "勝日%": round(float((d > 0).mean() * 100), 1),
        "CI下界": res["CI下界"],
        "CI上界": res["CI上界"],
        "BB p": res["BB p"],
        "NW p（對照）": round(p_nw, 4),
    }


def _bh(ps: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up。"""
    n = len(ps)
    order = np.argsort(ps)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, ps[i] * n / (rank + 1))
        adj[i] = running
    return adj


def _one_window(wlabel, start):
    main_rows, decomp_rows = [], []
    for label, combo_m, cluster_m, base_m in COMBOS:
        combo, cluster, base = _series(combo_m), _series(cluster_m), _series(base_m)
        if combo is None or cluster is None or base is None:
            print(f"  ⚠ 略過 {label}：缺 {combo_m}／{cluster_m}／{base_m}")
            continue

        d_total = _diff(combo, base, start)
        main_rows.append({"期間": wlabel, "分群法": label,
                          "傳統基準": base_m.replace("Grid (", "").replace(")", ""),
                          **_stats(d_total)})

        # 分解：總效果 = DRL 成分 + 分群成分
        d_drl = _diff(combo, cluster, start)   # 同配對底，只換交易端 → 命題 2
        d_clu = _diff(cluster, base, start)    # 同交易端，只換分組   → 命題 1
        s_drl, s_clu = _stats(d_drl), _stats(d_clu)
        decomp_rows.append({
            "期間": wlabel, "分群法": label,
            "總效果": round(float(d_total.mean() * TRADING_DAYS) / INITIAL_CAPITAL * 100, 3),
            "DRL 成分": s_drl["年化Δ%"], "DRL p": s_drl["BB p"],
            "分群成分": s_clu["年化Δ%"], "分群 p": s_clu["BB p"],
        })

    main = pd.DataFrame(main_rows)
    main["BH校正p"] = np.round(_bh(main["BB p"].values), 4)
    main["5%顯著"] = np.where(main["BH校正p"] < 0.05, "✔", "✘")
    decomp = pd.DataFrame(decomp_rows)
    # 恆等式自我檢查：兩成分相加須還原總效果（浮點誤差內）
    resid = (decomp["總效果"] - decomp["DRL 成分"] - decomp["分群成分"]).abs().max()
    assert resid < 0.02, f"{wlabel} 分解不成立，殘差 {resid:.4f}pp"
    return main, decomp, resid


def run():
    mains, decomps = [], []
    w = 100
    for wlabel, start in WINDOWS:
        main, decomp, resid = _one_window(wlabel, start)
        mains.append(main)
        decomps.append(decomp)

        print("=" * w)
        print(f"【{wlabel}】完整系統（動態分群 + DRL） vs 傳統基準"
              f"（GICS 產業分組 + Z-Score，排序準則對齊）")
        print("=" * w)
        print(main.drop(columns="期間").to_string(index=False))
        print()
        print("成分分解（年化百分點）　總效果 = DRL 成分 + 分群成分")
        print(decomp.drop(columns="期間").to_string(index=False))
        print(f"（分解恆等式最大殘差 {resid:.4f} pp）\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.concat(mains).to_csv(f"{OUT_DIR}/prop3_combined_main.csv",
                            index=False, encoding="utf-8-sig")
    pd.concat(decomps).to_csv(f"{OUT_DIR}/prop3_combined_decomp.csv",
                              index=False, encoding="utf-8-sig")
    print(f"→ {OUT_DIR}/prop3_combined_{{main,decomp}}.csv")


if __name__ == "__main__":
    run()
