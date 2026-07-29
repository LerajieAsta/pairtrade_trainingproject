# -*- coding: utf-8 -*-
"""
命題 2 SKIP 管道：DRL 的 SKIP 是選擇技巧，還是隨機也做得到？
======================================================================
`drl_behavior_gain_decomp.csv` 顯示 SKIP 貢獻了 DRL 總增益的 31–52%，但
`drl_behavior_skip_skill.csv` 的 MWU 檢定 p = 0.932 / 0.797 / 0.351 / 0.642
全不顯著——「貢獻很大」與「看不出鑑別力」並存，需要一個直接的虛無分布來裁決。

關鍵在於：**當底層策略期望值為負時，隨機跳過任何一批配對，期望上都會「避開損失」。**
所以「避開了多少損失」這個數字本身不能證明技巧，必須對照「隨機跳過同樣數量」
的分布。

檢定設計（置換檢定 / permutation test）
--------------------------------------
對每個配對底、每個參數格：

  1. 自 DRL 的 trade_logs 找出被 SKIP 的配對期集合 S
     （該配對期全期 Status == "HOLD_CASH (SKIP)"，同 drl_behavior.py 之定義）
  2. 在 Z-Score 端查這些配對期的**反事實損益**（同配對、同期，若照常交易會賺賠多少）
  3. 實際避開損失  A = −Σ pnl_ZS(S)
  4. 虛無分布：自同一組配對期宇宙隨機抽 |S| 個（不放回），重算避開損失，重複 N 次
  5. p = P(虛無 ≥ A)；並把 A 拆成

         A  =  隨機期望（機械成分）  +  超額（選擇技巧成分）

  虛無分布的均值 = |S| × (−平均 pnl)，即「無技巧地跳過同樣數量」所能得到的避損。
  若 A 落在分布中央（p ≈ 0.5），SKIP 就沒有選擇技巧，其貢獻純屬機械效應。

侷限（必須寫進論文）
--------------------
本檢定為**事後重抽**，不重跑回測，因此**不含槽位再配置效應**：實際回測中，
跳過配對 A 會釋放資金給配對 B，而本虛無分布假設「跳過就是不交易」。
故本檢定回答的是「SKIP 的**選股方向**有無技巧」，不是「SKIP 這個動作整體有無價值」。
若 p 不顯著但 DRL 仍勝出（見 prop2_exposure_control），合理推論是增益來自
**槽位週轉**而非**配對篩選**——該推論需重跑回測才能直接驗證。

用法：python -m analysis.prop2_skip_permutation
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

OUT_DIR = "results/analysis"
RESULT_DB = "results/result.db"
CACHE = os.path.join(OUT_DIR, "pairperiod_pnl_mainaxis.parquet")
PAIR_KEY = ["Ticker_A", "Ticker_B", "Period_Start"]
GRID = ["TOP N", "STOP LOSS %", "MAX SEC %"]
N_BOOT = 2000
SEED = 20260729

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASES = [
    ("Agglomerative",    "Grid (AGG-SSD)",  "Grid (AGG-SSD-DRL)"),
    ("HDBSCAN",          "Grid (HDB-SDP)",  "Grid (HDB-SDP-DRL)"),
    ("K-means",          "Grid (KM-SSD)",   "Grid (KM-SSD-DRL)"),
    ("GICS-SSD（傳統）", "Grid (GICS-SSD)", "Grid (GICS-SSD-DRL)"),
    ("GICS-SDP（傳統）", "Grid (GICS-SDP)", "Grid (GICS-SDP-DRL)"),
]


def _baseline_cells(summ: pd.DataFrame, method: str) -> pd.DataFrame:
    """只取無檔名後綴的基準格（排除 entry_z 等變體）。"""
    g = summ[summ.METHOD == method].copy()
    g = g[~g._path.str.contains(r"_EZ\d+|_DYN|_MHD|_XZ|_DG", regex=True, na=False)]
    return g


def load_pairperiod(sids: list[str]) -> pd.DataFrame:
    """
    每個 (strategy_id, 配對, 期) 一列：是否被 SKIP、該配對期總損益。
    trade_logs 3.34 億列 → 聚合結果快取，增量補齊。
    """
    sids = list(dict.fromkeys(sids))
    cached = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    have = set(cached.strategy_id.unique()) if not cached.empty else set()
    missing = [s for s in sids if s not in have]

    if missing:
        print(f"  聚合 {len(missing)} 條 strategy_id 的配對期損益"
              f"（快取已有 {len(have)} 條）…")
        con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
        q = (f"SELECT strategy_id, Ticker_A, Ticker_B, Period_Start, "
             f"MAX(CASE WHEN Status = 'HOLD_CASH (SKIP)' THEN 1 ELSE 0 END) AS skip, "
             f"SUM(Daily_Delta) AS pnl "
             f"FROM trade_logs WHERE strategy_id IN ({','.join('?' * len(missing))}) "
             f"GROUP BY strategy_id, Ticker_A, Ticker_B, Period_Start")
        new = pd.read_sql(q, con, params=missing)
        con.close()
        cached = new if cached.empty else pd.concat([cached, new], ignore_index=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        cached.to_parquet(CACHE)
        print(f"  已快取 → {CACHE}（{len(cached):,} 列）")

    return cached[cached.strategy_id.isin(sids)]


def _permute(pnl: np.ndarray, k: int, actual: float, rng) -> dict:
    """自 pnl 隨機抽 k 個（不放回），回傳虛無分布統計。"""
    n = len(pnl)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = -pnl[rng.choice(n, size=k, replace=False)].sum()
    p = float((draws >= actual).mean())
    return {
        "隨機期望": round(float(draws.mean()), 1),
        "隨機p95": round(float(np.percentile(draws, 95)), 1),
        "實際百分位": round(float((draws < actual).mean()) * 100, 1),
        "技巧成分": round(float(actual - draws.mean()), 1),
        "p值": round(p, 4),
        "5%顯著": "✔" if p < 0.05 else "✘",
    }


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    summ = pd.read_sql("SELECT METHOD,_path," + ",".join(f'"{c}"' for c in GRID) +
                       " FROM strategy_summaries", con)
    con.close()

    # 為每個配對底建立「同參數格」的 (ZS path, DRL path) 配對
    jobs = []
    for base, zs_m, drl_m in BASES:
        z, d = _baseline_cells(summ, zs_m), _baseline_cells(summ, drl_m)
        if z.empty or d.empty:
            print(f"  ⚠ 略過 {base}：缺策略")
            continue
        m = d.merge(z, on=GRID, suffixes=("_drl", "_zs"))
        for _, r in m.iterrows():
            # TOP N / STOP LOSS % 在 DB 中是字串（如 'Top 1' / '5%'），直接串接
            jobs.append((base, r._path_zs, r._path_drl,
                         f"{r['TOP N']}/SL{r['STOP LOSS %']}".replace(" ", "")))

    pp = load_pairperiod([s for j in jobs for s in (j[1], j[2])])
    idx = {s: g for s, g in pp.groupby("strategy_id")}

    rows = []
    for base, zs_p, drl_p, cell in jobs:
        d, z = idx.get(drl_p), idx.get(zs_p)
        if d is None or z is None:
            continue
        j = d[PAIR_KEY + ["skip"]].merge(
            z[PAIR_KEY + ["pnl"]], on=PAIR_KEY, how="inner")
        if j.empty:
            continue
        k = int(j.skip.sum())
        if k < 5 or k >= len(j):
            continue
        actual = float(-j.loc[j.skip == 1, "pnl"].sum())
        rows.append({"配對底": base, "格": cell, "配對期數": len(j), "SKIP 數": k,
                     "SKIP率%": round(k / len(j) * 100, 1),
                     "實際避損$": round(actual, 1),
                     **_permute(j.pnl.values.astype(float), k, actual, rng)})

    res = pd.DataFrame(rows)
    if res.empty:
        print("⚠ 無可用資料")
        return

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print(f"SKIP 選擇技巧的置換檢定（每格 {N_BOOT} 次隨機重抽，不放回）")
    print("H0：SKIP 的配對選擇與隨機無異　H1：SKIP 避開的損失高於隨機")
    print("=" * 100)

    agg = res.groupby("配對底").agg(
        格數=("格", "count"),
        平均SKIP率=("SKIP率%", "mean"),
        實際避損=("實際避損$", "mean"),
        隨機期望=("隨機期望", "mean"),
        技巧成分=("技巧成分", "mean"),
        平均百分位=("實際百分位", "mean"),
        顯著格數=("5%顯著", lambda s: f"{(s == '✔').sum()}/{len(s)}"),
    ).round(1)
    print(agg.to_string())

    print("\n--- 逐格明細（前 12 列）")
    print(res.head(12).to_string(index=False))

    res.to_csv(f"{OUT_DIR}/prop2_skip_permutation.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(f"{OUT_DIR}/prop2_skip_permutation_summary.csv", encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop2_skip_permutation{{,_summary}}.csv")


if __name__ == "__main__":
    run()
