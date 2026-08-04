# -*- coding: utf-8 -*-
"""
命題 2 曝險對照：DRL 是「交易得更好」還是「交易得更少」？
======================================================================
`drl_behavior.py` 的三項輸出指出一個結構性疑慮：

  1. SKIP 沒有可證實的鑑別力——MWU p = 0.932 / 0.797 / 0.351 / 0.642，無一顯著；
     HDBSCAN 底甚至反向（被 SKIP 的配對平均比留下的更賺：−0.47 vs −1.16）
  2. 但 SKIP 貢獻了總增益的 31–52%（gain_decomp）
  3. 門檻選擇與配對品質無關——Spearman ρ = −0.025 / 0.066 / −0.02，
     而進場門檻中位數 2.20–2.27，高於基準的 2.0

兩條管道都指向同一個機制：**少交易**。底層策略期望值為負時，光是減少曝險
就會機械性地改善績效，不需要任何「學習」。

本模組檢定門檻管道：把 Z-Score 的固定門檻拉到 DRL 的實際中位數（2.2/2.3），
問「同等門檻的笨基準能複製多少增益？」

  對照 A：DRL − ZS(2.0)      已知的表面增益
  對照 B：DRL − ZS(2.2)      決定性檢定——DRL 是否勝過同門檻的笨基準
  對照 C：ZS(2.2) − ZS(2.0)  門檻管道本身值多少（＝曝險縮減的貢獻）

  複製率 = C / A ：表面增益中有多少能被「單純調高門檻」複製

侷限：本模組**只**驗門檻管道。SKIP 管道（佔增益 31–52%）需另做隨機 SKIP 的
bootstrap 虛無分布，那不在本檔範圍內——即使 DRL 勝過 ZS(2.2)，命題 2 也尚未結案。

前置：需先跑
    SENSITIVITY_PARAM=entry_z SENSITIVITY_BASE="<name>" \
    SENSITIVITY_VALUES="2.2,2.3,2.5" python run_trading.py
（沿用既有 formation，只重跑交易端）

用法：python -m analysis.prop2_exposure_control
"""
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

from analysis.block_bootstrap import bootstrap_test
from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, RESULT_DB, TRADING_DAYS,
    load_daily_sids, method_paths, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (配對底, Z-Score db_method, DRL db_method)
BASES = [
    ("Agglomerative",    "Grid (AGG-SSD)",  "Grid (AGG-SSD-DRL)"),
    ("HDBSCAN",          "Grid (HDB-SDP)",  "Grid (HDB-SDP-DRL)"),
    ("K-means",          "Grid (KM-SSD)",   "Grid (KM-SSD-DRL)"),
    ("GICS-SSD（傳統）", "Grid (GICS-SSD)", "Grid (GICS-SSD-DRL)"),
    ("GICS-SDP（傳統）", "Grid (GICS-SDP)", "Grid (GICS-SDP-DRL)"),
]

# DRL 的實際進場門檻中位數（drl_behavior_decisions.csv）
DRL_MEDIAN_Z = {"Agglomerative": 2.22, "HDBSCAN": 2.27, "K-means": 2.20,
                "GICS-SSD（傳統）": 2.21, "GICS-SDP（傳統）": 2.21}

_CELL = re.compile(r"(Top\d+_SL\d+)")
_EZ = re.compile(r"_EZ(\d+)_DSZ(\d+)")


def _cell(sid: str) -> str | None:
    m = _CELL.search(os.path.basename(sid))
    return m.group(1) if m else None


def _entry_z(sid: str) -> float:
    """檔名無 _EZ 後綴 → 預設 2.0（run_trading._log_name 的約定）。"""
    m = _EZ.search(os.path.basename(sid))
    return round(int(m.group(1)) / 10.0, 2) if m else 2.0


def _has_other_suffix(sid: str) -> bool:
    """排除 _DYN/_MHD/_XZ/_DG 等其他實驗變體，只留純 entry_z 變動。"""
    tail = os.path.basename(sid).replace(".csv", "")
    tail = _EZ.sub("", tail)
    return not re.fullmatch(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+", tail)


def _stats(d: np.ndarray) -> dict:
    mu, sd = d.mean(), d.std(ddof=1)
    return {
        "年化Δ%": round(float(mu * TRADING_DAYS) / INITIAL_CAPITAL * 100, 3),
        "IR": round(float(np.sqrt(TRADING_DAYS) * mu / sd), 3) if sd > 0 else np.nan,
    }


def _exposure(methods: list[str]) -> pd.DataFrame:
    """取 Entries / Avg_Utilization，用來驗證『同門檻』是否真的對齊了曝險。"""
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    df = pd.read_sql(
        f"SELECT METHOD,_path,Entries,Avg_Utilization FROM strategy_summaries "
        f"WHERE METHOD IN ({','.join('?' * len(methods))})", con, params=methods)
    con.close()
    return df


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    methods = sorted({m for _, z, d in BASES for m in (z, d)})
    paths = method_paths(methods)
    expo = _exposure(methods)

    # 建 {METHOD: {entry_z: {cell: sid}}}
    reg: dict[str, dict[float, dict[str, str]]] = {}
    for m, g in paths.groupby("METHOD"):
        for sid in g._path:
            if _has_other_suffix(sid):
                continue
            c = _cell(sid)
            if c:
                reg.setdefault(m, {}).setdefault(_entry_z(sid), {})[c] = sid

    all_sids = [s for m in reg for ez in reg[m] for s in reg[m][ez].values()]
    px = load_daily_sids(all_sids)

    rows, expo_rows = [], []

    for base, zs_m, drl_m in BASES:
        zmap, dmap = reg.get(zs_m, {}), reg.get(drl_m, {})
        if 2.0 not in zmap or 2.0 not in dmap:
            print(f"  ⚠ 略過 {base}：缺基準 entry_z=2.0")
            continue
        ez_avail = sorted(z for z in zmap if z != 2.0)
        if not ez_avail:
            print(f"  ⚠ 略過 {base}：尚未跑 entry_z 變體（先跑 run_trading.py）")
            continue

        def ew(sidmap: dict, cells: list[str]) -> pd.Series:
            cols = [sidmap[c] for c in cells if sidmap.get(c) in px.columns]
            return px[cols].mean(axis=1)

        def aligned(cands: list[str], *sidmaps: dict) -> list[str]:
            """只保留「每一臂都有逐日明細」的格子。

            格子清單來自 strategy_summaries，但 px 來自 trade_logs——兩者可能不一致
            （併發寫入競爭會讓某格只剩摘要，見 tools/audit_result_db.py）。若不先取
            交集，ew() 會單方面丟掉缺的那格，於是差分的兩臂平均在不同的籃子上，
            混入籃子成分差異。本對照的整個設計就是要讓兩臂除了進場門檻外完全可比，
            這種不對稱正好打在方法的要害上，所以缺格寧可兩邊一起排除。
            """
            keep = [c for c in cands if all(m.get(c) in px.columns for m in sidmaps)]
            dropped = sorted(set(cands) - set(keep))
            if dropped:
                print(f"  ⚠ {base}：{dropped} 缺逐日明細，兩臂一併排除以維持對齊"
                      f"（{len(cands)} → {len(keep)} 格）")
            return keep

        cells0 = aligned(sorted(set(zmap[2.0]) & set(dmap[2.0])), zmap[2.0], dmap[2.0])
        z0, drl = ew(zmap[2.0], cells0), ew(dmap[2.0], cells0)

        # 對照 A：DRL − ZS(2.0)
        dA = (drl - z0).values
        rA = bootstrap_test(dA)
        pA = rA["BB p"]
        _, pA_nw, _ = newey_west(dA)
        gainA = _stats(dA)["年化Δ%"]
        rows.append({"配對底": base, "對照": "A｜DRL − ZS(2.0)", "entry_z": 2.0,
                     **_stats(dA), "CI下界": rA["CI下界"], "CI上界": rA["CI上界"],
                     "BB p": pA, "NW p（對照）": round(pA_nw, 4),
                     "5%顯著": "✔" if pA < 0.05 else "✘", "複製率%": ""})

        for ez in ez_avail:
            # zmap[2.0] 也列入對齊條件——對照 C 用得到它，B 與 C 共用同一個籃子，
            # 兩者才彼此可比，複製率（C 相對 A 的比值）也才有意義。
            cells = aligned(sorted(set(zmap[ez]) & set(dmap[2.0])),
                            zmap[ez], dmap[2.0], zmap[2.0])
            if not cells:
                continue
            zz = ew(zmap[ez], cells)
            drl_c = ew(dmap[2.0], cells)

            # 對照 B：DRL − ZS(ez)
            dB = (drl_c - zz).values
            rB = bootstrap_test(dB)
            pB = rB["BB p"]
            _, pB_nw, _ = newey_west(dB)
            rows.append({"配對底": base, "對照": f"B｜DRL − ZS({ez})", "entry_z": ez,
                         **_stats(dB), "CI下界": rB["CI下界"], "CI上界": rB["CI上界"],
                         "BB p": pB, "NW p（對照）": round(pB_nw, 4),
                         "5%顯著": "✔" if pB < 0.05 else "✘", "複製率%": ""})

            # 對照 C：ZS(ez) − ZS(2.0)，並算門檻管道的複製率
            dC = (zz - ew(zmap[2.0], cells)).values
            rC = bootstrap_test(dC)
            pC = rC["BB p"]
            _, pC_nw, _ = newey_west(dC)
            gainC = _stats(dC)["年化Δ%"]
            rep = round(gainC / gainA * 100, 1) if gainA else np.nan
            rows.append({"配對底": base, "對照": f"C｜ZS({ez}) − ZS(2.0)", "entry_z": ez,
                         **_stats(dC), "CI下界": rC["CI下界"], "CI上界": rC["CI上界"],
                         "BB p": pC, "NW p（對照）": round(pC_nw, 4),
                         "5%顯著": "✔" if pC < 0.05 else "✘", "複製率%": rep})

        # 曝險對齊驗證
        for label, m_, ez in ([("ZS(2.0)", zs_m, 2.0), ("DRL", drl_m, 2.0)] +
                              [(f"ZS({e})", zs_m, e) for e in ez_avail]):
            sids = set(reg.get(m_, {}).get(ez, {}).values())
            sub = expo[expo._path.isin(sids)]
            if not sub.empty:
                expo_rows.append({
                    "配對底": base, "臂": label,
                    "平均進場次數": round(float(sub.Entries.mean()), 1),
                    "平均利用率": round(float(sub.Avg_Utilization.mean()), 3)})

    res = pd.DataFrame(rows)
    exp_df = pd.DataFrame(expo_rows)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 92)
    print("曝險對照：DRL vs 同門檻的 Z-Score（等權組合逐日差分 block bootstrap）")
    print("=" * 92)
    for base in res.配對底.unique():
        print(f"\n--- {base}（DRL 實際門檻中位數 {DRL_MEDIAN_Z.get(base, '—')}）")
        print(res[res.配對底 == base].drop(columns="配對底").to_string(index=False))

    print("\n" + "=" * 92)
    print("曝險對齊驗證（同門檻是否真的把交易量拉到同一水準？）")
    print("=" * 92)
    if not exp_df.empty:
        print(exp_df.pivot(index="配對底", columns="臂",
                           values="平均進場次數").to_string())

    res.to_csv(f"{OUT_DIR}/prop2_exposure_control.csv", index=False, encoding="utf-8-sig")
    exp_df.to_csv(f"{OUT_DIR}/prop2_exposure_alignment.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop2_exposure_{{control,alignment}}.csv")


if __name__ == "__main__":
    run()
