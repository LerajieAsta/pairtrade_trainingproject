# -*- coding: utf-8 -*-
"""
命題 2 統計檢定：DRL 交易端 vs Z-Score 交易端

四項檢定，回答兩個層次的問題：

  相對主張「換交易端有沒有用？」
    一、配對 t 檢定 / Wilcoxon：同一配對底、同一參數網格逐格配對
    二、逐輪穩健性：五輪獨立重訓各自檢定

  絕對主張「這個策略本身賺不賺錢？」
    三、Newey-West HAC t 檢定：H0 平均日報酬 = 0（自相關穩健）
    四、Deflated Sharpe Ratio：校正網格搜尋的多重測試偏誤

配對設計消去兩策略共同承受的市場風險，故相對檢定的檢定力遠高於絕對檢定；
兩者結論不同並不矛盾，分別支撐論文的「方法比較」與「限制」兩節。

用法：python -m analysis.proposition2_stats
"""
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from analysis.regime_cost_dsr_eval import (
    SPEC_MAIN, _trial_specs, deflated_sharpe, load_daily_returns,
)

RESULT_DB = "results/result.db"
VARIANCE_CSV = "results/analysis/drl_variance_runs_mainaxis.csv"
OUT_DIR = "results/analysis"
TRADING_DAYS = 252

# 網格識別欄位：三者組合唯一決定一格
GRID = ["TOP N", "STOP LOSS %", "MAX SEC %"]

# (配對底名稱, Z-Score 策略, DRL 策略)
PAIRS = [
    ("Agglomerative", "Grid (AGG-SSD)", "Grid (AGG-SSD-DRL)"),
    ("HDBSCAN",       "Grid (HDB-SDP)", "Grid (HDB-SDP-DRL)"),
    ("K-means",       "Grid (KM-SSD)",  "Grid (KM-SSD-DRL)"),
]


def newey_west_tstat(r, lags: int | None = None):
    """H0: E[r] = 0，Bartlett kernel HAC 標準誤。lags 預設用 Newey-West 經驗法則。"""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30:
        return np.nan, np.nan, 0
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2 / 9)))
    e = r - r.mean()
    s = (e @ e) / T
    for L in range(1, lags + 1):
        s += 2.0 * (1.0 - L / (lags + 1.0)) * ((e[L:] @ e[:-L]) / T)
    se = np.sqrt(max(s, 1e-18) / T)
    t = r.mean() / se
    return t, 2 * (1 - norm.cdf(abs(t))), lags


def _paired_tests(summ, runs):
    """一、DRL（五輪平均）vs Z-Score，逐格配對。"""
    rows = []
    for name, zs, drl in PAIRS:
        z = summ[summ.METHOD == zs].set_index(GRID)
        d = runs[runs.METHOD == drl].groupby(GRID).agg(
            sh=("Sharpe_Raw", "mean"), ann=("Ann_Ret_Raw", "mean"))
        j = z[["Sharpe_Raw", "Ann_Ret_Raw"]].join(d, how="inner")
        if j.empty:
            continue
        ds = j.sh - j.Sharpe_Raw
        t, pt = stats.ttest_rel(j.sh, j.Sharpe_Raw)
        try:
            _, pw = stats.wilcoxon(j.sh, j.Sharpe_Raw)
        except ValueError:
            pw = np.nan
        rows.append({
            "配對底": name, "n": len(j),
            "ΔSharpe平均": round(ds.mean(), 4),
            "Δ年化pp": round((j.ann - j.Ann_Ret_Raw).mean() * 100, 3),
            "勝格數": f"{int((ds > 0).sum())}/{len(j)}",
            "配對t": round(t, 3), "t檢定p": f"{pt:.2e}",
            "Wilcoxon p": f"{pw:.2e}" if pw == pw else "—",
            "Cohen d": round(ds.mean() / ds.std(ddof=1), 2),
        })
    return pd.DataFrame(rows)


def _per_round(summ, runs):
    """二、五輪各自獨立檢定，確認增益非特定訓練批次所致。"""
    rows = []
    for name, zs, drl in PAIRS:
        z = summ[summ.METHOD == zs].set_index(GRID)["Sharpe_Raw"]
        r = []
        for _, g in runs[runs.METHOD == drl].groupby("run_id"):
            j = pd.concat([z, g.set_index(GRID)["Sharpe_Raw"]], axis=1,
                          join="inner", keys=["z", "d"])
            if j.empty:
                continue
            _, p = stats.ttest_rel(j.d, j.z)
            r.append((int((j.d > j.z).sum()), len(j), p))
        if not r:
            continue
        rows.append({
            "配對底": name,
            "各輪勝格數": "  ".join(f"{a}/{b}" for a, b, _ in r),
            "各輪p值": "  ".join(f"{p:.1e}" for *_, p in r),
            "全輪皆顯著": "✔" if all(p < .05 for *_, p in r) else "✘",
        })
    return pd.DataFrame(rows)


def _absolute_tests(summ, summ_all=None):
    """
    三、四：網格最佳格的 Newey-West 與 Deflated Sharpe。

    summ      ：僅基準格，用來挑「網格最佳配置」並做檢定
    summ_all  ：完整 strategy_summaries，僅供 DSR 計算試驗宇宙（含所有變體與
                封存策略——那才是真實的 researcher degrees of freedom）
    """
    if summ_all is None:
        summ_all = summ
    sel = {}
    for name, zs, drl in PAIRS:
        for tag, m in (("Z-Score", zs), ("DRL", drl)):
            g = summ[summ.METHOD == m]
            if len(g):
                sel[(name, tag)] = (g.loc[g.Sharpe_Raw.idxmax()], g)

    daily = load_daily_returns([b["_path"] for b, _ in sel.values()])

    nw, dsr = [], []
    for (name, tag), (b, g) in sel.items():
        r = daily.get(b["_path"], pd.Series(dtype=float))
        t, p, lags = newey_west_tstat(r.values) if len(r) else (np.nan, np.nan, 0)
        nw.append({"配對底": name, "交易端": tag,
                   "年化": f"{b.Ann_Ret_Raw * 100:.2f}%", "Sharpe": round(b.Sharpe_Raw, 3),
                   "交易日": len(r), "落後階": lags,
                   "NW t": round(t, 3), "NW p": f"{p:.4f}",
                   "5%顯著": "✔" if p < .05 else "✘"})

        # 2026-07-29：N 由 len(g)=15（該策略的參數格數）改為試驗宇宙口徑。
        # 15 格共用同一批配對、僅組合設定不同，不是 15 次獨立試驗；真正的
        # researcher degrees of freedom 是 result.db 裡 87 個相異 METHOD
        # （含試過後封存的負面結果）。N 與 var_sr 取自同一集合（見
        # regime_cost_dsr_eval._trial_specs 的說明）。
        n_tr, var_sr = _trial_specs(summ_all, b["METHOD"])[SPEC_MAIN]
        d = deflated_sharpe(r, n_tr, var_sr)
        dsr.append({"配對底": name, "交易端": tag, "N試驗": n_tr,
                    "SR年化": round(d["SR_ann"], 3), "門檻SR0": round(d["SR0_ann"], 3),
                    "DSR": round(d["DSR"], 3),
                    "判定": "✔" if d["DSR"] >= .95 else "✘"})
    return pd.DataFrame(nw), pd.DataFrame(dsr)


def run():
    if not os.path.exists(VARIANCE_CSV):
        raise SystemExit(f"缺少五輪變異數資料：{VARIANCE_CSV}\n"
                         f"請先執行 DRL_VARIANCE_TAG=mainaxis python -m tools.run_drl_variance")

    con = sqlite3.connect(RESULT_DB)
    summ_all = pd.read_sql("SELECT * FROM strategy_summaries", con)
    con.close()
    runs = pd.read_csv(VARIANCE_CSV)

    # 只保留基準格。entry_z 等交易端變體與基準共用 db_method 與同一組
    # (TOP N, STOP LOSS %, MAX SEC %)，若不濾除，set_index(GRID) 會產生
    # 重複索引（ValueError: cannot handle a non-unique multi-index），
    # 且會把對照組混入配對檢定。
    # 註：DSR 的試驗宇宙仍用完整 summ_all（見 _trial_specs），兩者刻意不同——
    #     檢定要乾淨的基準格，選擇偏誤校正要完整的試驗史。
    summ = summ_all[~summ_all._path.str.contains(
        r"_EZ\d+|_DYN|_MHD|_XZ|_DG", regex=True, na=False)]

    t1 = _paired_tests(summ, runs)
    t2 = _per_round(summ, runs)
    t3, t4 = _absolute_tests(summ, summ_all)

    for title, note, tbl in [
        ("一、配對 t 檢定：DRL vs Z-Score（同一參數網格逐格配對）",
         "H0：兩交易端績效相同。配對設計消去共同市場風險。", t1),
        ("二、逐輪穩健性：五輪獨立重訓各自檢定",
         "確認增益非單一訓練批次的隨機結果。", t2),
        ("三、Newey-West HAC t 檢定（網格最佳格）",
         "H0：平均日報酬 = 0。此為絕對主張，無對照組。", t3),
        ("四、Deflated Sharpe Ratio（Bailey & López de Prado 2014）",
         "校正網格搜尋的多重測試偏誤；門檻 SR0 為純靠運氣的期望最高 Sharpe。", t4),
    ]:
        print("\n" + "=" * 88)
        print(title)
        print("  " + note)
        print("=" * 88)
        print(tbl.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    for nm, tbl in [("paired", t1), ("per_round", t2), ("newey_west", t3), ("dsr", t4)]:
        tbl.to_csv(f"{OUT_DIR}/prop2_{nm}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已存] {OUT_DIR}/prop2_{{paired,per_round,newey_west,dsr}}.csv")


if __name__ == "__main__":
    run()
