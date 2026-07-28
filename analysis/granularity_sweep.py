# -*- coding: utf-8 -*-
"""
分群粒度掃描：配對品質由「粒度」還是「演算法」決定？

命題 1 的配對檢定顯示三種 ML 分群皆未優於 GICS 產業分組，且機制指向兩項
結構性代價——跨產業配對與候選池縮減。但「演算法」與「粒度」在該設計中
互相混淆：HDBSCAN / Agglomerative / K-means 產生的群數與群大小本就不同。

本掃描固定演算法為 Agglomerative、固定排序與篩選，僅變動切割門檻
`agg_threshold_percentile` ∈ {50, 60, 75, 90, 95}：

    門檻分位越高 → 合併距離門檻越寬鬆 → 群越少越大 → 候選池越大
                                                  → 越接近 GICS 的分組粒度

若績效隨粒度單調變化，則「粒度」為主因，分群演算法的選擇為次要——
命題 1 的否定即可從「ML 分群無效」精確化為「候選池充足度決定配對品質」。

產生資料：
    SENSITIVITY_PARAM=agg_threshold_percentile SENSITIVITY_BASE="Grid AGG-SSD" \\
    SENSITIVITY_VALUES="50,60,90,95" python run_formation.py && python run_trading.py

用法：python -m analysis.granularity_sweep
"""
import os
import re
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

RESULT_DB = "results/result.db"
FORMATION_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
OUT_DIR = "results/analysis"
GRID = ["TOP N", "STOP LOSS %", "MAX SEC %"]

BASE_METHOD = "Grid (AGG-SSD)"      # 基準（門檻分位 75）
REFERENCE = "Grid (GICS-SSD)"       # 對照：產業分組（粒度上限）
PARAM = "agg_threshold_percentile"


def _level(method: str) -> float:
    """自 db_method 取出門檻分位；基準條目無標註，回傳 75。"""
    m = re.search(rf"{PARAM}=([\d.]+)", method)
    return float(m.group(1)) if m else 75.0


def _formation_id(method: str) -> str:
    """db_method → formation_pairs.strategy_id。"""
    inner = method.replace("Grid (", "Grid ").replace(")", "", 1)
    return f"{inner}_MSR0"


def run():
    con = sqlite3.connect(RESULT_DB)
    summ = pd.read_sql("SELECT * FROM strategy_summaries", con)

    methods = sorted([m for m in summ.METHOD.unique()
                      if m.startswith(BASE_METHOD)], key=_level)
    if len(methods) < 2:
        raise SystemExit("尚無掃描結果——請先執行本模組 docstring 中的產生指令。")

    fcon = sqlite3.connect(FORMATION_DB) if os.path.exists(FORMATION_DB) else None
    gics = summ[summ.METHOD == REFERENCE].set_index(GRID)["Sharpe_Raw"]

    rows = []
    for m in methods + ([REFERENCE] if len(gics) else []):
        g = summ[summ.METHOD == m]
        if g.empty:
            continue
        lvl = "GICS" if m == REFERENCE else f"{_level(m):.0f}"

        pairs_pp, cross = np.nan, np.nan
        if fcon is not None:
            d = pd.read_sql("SELECT Sector_A, Sector_B, Period_Start FROM formation_pairs "
                            "WHERE strategy_id = ?", fcon, params=[_formation_id(m)])
            if len(d):
                pairs_pp = d.groupby("Period_Start").size().mean()
                cross = (d.Sector_A != d.Sector_B).mean() * 100

        # 與 GICS 的配對檢定（同網格逐格）
        p_txt = "—"
        if len(gics) and m != REFERENCE:
            j = pd.concat([gics, g.set_index(GRID)["Sharpe_Raw"]],
                          axis=1, join="inner", keys=["g", "m"]).dropna()
            if len(j) >= 3:
                _, p = stats.ttest_rel(j.m, j.g)
                p_txt = f"{p:.4f}{'*' if p < .05 else ''}"

        rows.append({
            "門檻分位": lvl,
            "期均配對數": round(pairs_pp, 1) if pairs_pp == pairs_pp else "—",
            "跨產業%": round(cross, 1) if cross == cross else "—",
            "最佳年化": f"{g.Ann_Ret_Raw.max() * 100:.2f}%",
            "最佳Sharpe": round(g.Sharpe_Raw.max(), 3),
            "網格均Sharpe": round(g.Sharpe_Raw.mean(), 4),
            "正Sharpe": f"{int((g.Sharpe_Raw > 0).sum())}/{len(g)}",
            "vs GICS p": p_txt,
        })
    if fcon is not None:
        fcon.close()
    con.close()

    tbl = pd.DataFrame(rows)
    print("=" * 96)
    print("分群粒度掃描：Agglomerative 切割門檻 × 配對品質")
    print("  演算法、特徵、排序、篩選、交易端全部固定；唯一變因為切割門檻分位。")
    print("=" * 96)
    print(tbl.to_string(index=False))

    # 粒度（候選池大小）與績效的單調關係
    sw = tbl[tbl.門檻分位 != "GICS"].copy()
    sw = sw[sw.期均配對數 != "—"]
    if len(sw) >= 3:
        x = sw.期均配對數.astype(float)
        print("\n" + "-" * 96)
        print("候選池大小 vs 績效（Spearman 秩相關，n = %d 個粒度層級）" % len(sw))
        print("-" * 96)
        for col in ["最佳Sharpe", "網格均Sharpe"]:
            rho, p = stats.spearmanr(x, sw[col].astype(float))
            verdict = "支持粒度假說" if (rho > 0 and p < .10) else ("方向相反" if rho < 0 else "無關")
            print(f"  期均配對數 vs {col:12s}  ρ = {rho:+.3f}  p = {p:.4f}   {verdict}")
        rho, p = stats.spearmanr(x, sw["跨產業%"].astype(float))
        print(f"  期均配對數 vs 跨產業%      ρ = {rho:+.3f}  p = {p:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tbl.to_csv(f"{OUT_DIR}/granularity_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已存] {OUT_DIR}/granularity_sweep.csv")


if __name__ == "__main__":
    run()
