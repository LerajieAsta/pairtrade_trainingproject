# -*- coding: utf-8 -*-
"""
DRL 交易端的行為解析：模型學到了什麼？

DRL 每配對每期自 9 個動作中擇一（SKIP 或 8 組 (entry_z, exit_z) 門檻）。
動作本身未落庫，但可由 trade_logs 完整還原：

  SKIP      → 該配對期全期 Status == "HOLD_CASH (SKIP)"
  entry_z   → 該配對期進場列的 min|Z|（首次觸發時最接近門檻）
  exit_z    → 收斂平倉列的 |Z|

四項分析：
  一、決策分布：SKIP 率與門檻選擇，對照靜態基準 (2.0, 0.0)
  二、SKIP 的技巧性：Z-Score 端在「被 DRL 跳過的同一批配對」上的實際損益。
      DRL 與 Z-Score 共用同一形成期配對，故此為直接的反事實對照。
      注意：被跳過者的損益為負不足以證明技巧——配對平均損益本就為負，
      隨機棄權亦會「避開虧損」。判準是其損益是否顯著低於留下者（MWU 檢定）。
  三、門檻選擇與配對屬性：進場門檻是否隨排序名次（配對品質代理）變化。
  四、增益來源分解：將 DRL − Z-Score 的總損益差拆為 SKIP 與門檻選擇兩塊，
      定位增益究竟來自「拒絕交易」還是「調整門檻」。
      ⚠ 此分解為**會計恆等式，不是技巧歸因**——與第二項同一個道理：
        期望值為負時，隨機跳過亦會產生正的「SKIP 貢獻」。
        故不可由「SKIP 佔比 36%」推論「模型學會篩掉爛配對」。
        輸出 CSV 帶 SKIP_性質 欄，以防此表被單獨引用時失去限定條件。

2026-07-29 後續檢定的結論（本模組僅描述行為，不作因果宣稱）：
  - SKIP 的選擇不優於隨機：置換檢定 75 格中僅 7 格顯著（隨機期望 3.75），
    機械成分佔實際避損 42–106%（analysis/prop2_skip_permutation.py）
  - 門檻管道亦非增益來源：把 Z-Score 固定門檻拉到 DRL 的實際中位數 2.2 後，
    DRL 五種配對底仍全部顯著勝出；門檻管道只複製 4.2–20.9% 且不顯著
    （analysis/prop2_exposure_control.py）
  - 「少交易」的直覺不成立：DRL 進場次數是 Z-Score 的 1.6–1.9 倍
    （利用率較低係因持倉更短，非因交易更少）
  三者合計排除了增益的三個常見解釋；殘差與「槽位週轉」一致，惟需重跑回測
  才能直接驗證，現列為後續研究。

用法：python -m analysis.drl_behavior
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

# Windows 主控台預設 cp950，無法輸出 ∈ / − 等數學符號（config.py 亦作同樣處理）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from scipy.stats import mannwhitneyu

RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"
GRID = ["TOP N", "STOP LOSS %", "MAX SEC %"]
PAIR_KEY = ["Ticker_A", "Ticker_B", "Period_Start"]

MENU_ENTRY = np.array([1.5, 2.0, 2.5, 3.0])   # 動作選單的 entry_z 取值
BASELINE_ENTRY = 2.0                           # 靜態基準門檻

# 隨 gain_decomp 一併落檔，確保該表脫離上下文（如被直接引用到論文表格）時
# 仍帶著限定條件——「SKIP 貢獻 X%」極易被誤讀為模型的選擇技巧。
SKIP_CAVEAT = ("會計恆等式，非技巧歸因：底層期望值為負時隨機跳過亦會避損；"
               "置換檢定不顯著（75 格中 7 格，隨機期望 3.75）"
               "→ 見 analysis/prop2_skip_permutation.py")

# (配對底, Z-Score 策略, DRL 策略)
PAIRS = [
    ("Agglomerative", "Grid (AGG-SSD)", "Grid (AGG-SSD-DRL)"),
    ("HDBSCAN",       "Grid (HDB-SDP)", "Grid (HDB-SDP-DRL)"),
    ("K-means",       "Grid (KM-SSD)",  "Grid (KM-SSD-DRL)"),
    ("GICS（傳統）",   "Grid (GICS-SSD)", "Grid (GICS-SSD-DRL)"),
    ("GICS（傳統）",   "Grid (GICS-SDP)", "Grid (GICS-SDP-DRL)"),
]


# 基準格：檔名無後綴。entry_z 等交易端對照變體與基準共用 db_method，
# 若不濾除會被選為「行為解析格」，使本模組解析到對照組而非現役策略。
_BASELINE_CELL = r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv$"


def _behavior_grid(summ, method):
    """
    取 TOP N 最大的一格（同值時取 Sharpe 較高者）。

    行為解析的對象是模型的決策規律而非某組參數的績效，故選配對數最多的格，
    樣本最充足；TOP N = 1 的格因 Pair_Rank 恆為 1，無法做名次相關分析。
    """
    g = summ[(summ.METHOD == method)
             & summ._path.str.contains(_BASELINE_CELL, regex=True, na=False)]
    if g.empty:
        return None
    g = g[g["TOP N"] == g["TOP N"].max()]
    return g.loc[g.Sharpe_Raw.idxmax()]


def _matched_paths(summ, zs_method, drl_method):
    """回傳同一參數格下的 (Z-Score path, DRL path)。"""
    d = _behavior_grid(summ, drl_method)
    if d is None:
        return None, None
    z = summ[(summ.METHOD == zs_method)
             & summ._path.str.contains(_BASELINE_CELL, regex=True, na=False)]
    for col in GRID:
        z = z[z[col] == d[col]]
    return (None if z.empty else z.iloc[0]["_path"]), d["_path"]


def _decisions(con, path):
    """自 trade_logs 還原每個「配對 × 期」的決策。"""
    t = pd.read_sql(
        "SELECT Ticker_A, Ticker_B, Period_Start, Status, ZScore, Pair_Rank, Daily_Delta "
        "FROM trade_logs WHERE strategy_id = ?", con, params=[path])
    if t.empty:
        return pd.DataFrame()
    t["absz"] = t.ZScore.abs()

    g = t.groupby(PAIR_KEY, sort=False)
    out = g.agg(skip=("Status", lambda s: bool((s == "HOLD_CASH (SKIP)").any())),
                rank=("Pair_Rank", "first"),
                pnl=("Daily_Delta", "sum")).reset_index()

    ent = t[t.Status.str.startswith("ENTER", na=False)]
    if not ent.empty:
        ez = ent.groupby(PAIR_KEY).absz.min().rename("entry_z")
        out = out.merge(ez, on=PAIR_KEY, how="left")
    else:
        out["entry_z"] = np.nan

    ex = t[t.Status == "EXIT"]
    if not ex.empty:
        xz = ex.groupby(PAIR_KEY).absz.median().rename("exit_z")
        out = out.merge(xz, on=PAIR_KEY, how="left")
    else:
        out["exit_z"] = np.nan

    # 觀測到的進場 |Z| 貼回選單最近的門檻值
    out["entry_bin"] = np.where(
        out.entry_z.notna(),
        MENU_ENTRY[np.abs(out.entry_z.values[:, None] - MENU_ENTRY).argmin(axis=1)],
        np.nan)
    return out


def run():
    con = sqlite3.connect(RESULT_DB)
    summ = pd.read_sql("SELECT * FROM strategy_summaries", con)

    t1, t2, t3, t4 = [], [], [], []
    for name, zs_m, drl_m in PAIRS:
        zp, dp = _matched_paths(summ, zs_m, drl_m)
        if dp is None:
            continue
        d = _decisions(con, dp)
        if d.empty:
            continue
        traded = d[~d.skip]
        rk = drl_m.split("-")[1].rstrip(")").replace("DRL", "").strip("- ")

        # ── 一、決策分布 ──────────────────────────────────────────
        dist = traded.entry_bin.value_counts(normalize=True).mul(100)
        t1.append({
            "配對底": name, "排序": rk, "配對期數": len(d),
            "SKIP 率": f"{d.skip.mean() * 100:.1f}%",
            "進場門檻中位": round(traded.entry_z.median(), 2),
            "選 1.5": f"{dist.get(1.5, 0):.0f}%", "選 2.0": f"{dist.get(2.0, 0):.0f}%",
            "選 2.5": f"{dist.get(2.5, 0):.0f}%", "選 3.0": f"{dist.get(3.0, 0):.0f}%",
            "偏離基準": f"{(traded.entry_bin != BASELINE_ENTRY).mean() * 100:.0f}%",
        })

        # ── 二、SKIP 的技巧性（反事實：同配對在 Z-Score 端的損益）──
        if zp is not None:
            z = _decisions(con, zp)
            if not z.empty:
                j = d[["Ticker_A", "Ticker_B", "Period_Start", "skip"]].merge(
                    z[["Ticker_A", "Ticker_B", "Period_Start", "pnl"]], on=PAIR_KEY)
                if len(j) and j.skip.any():
                    sk, kp = j[j.skip].pnl, j[~j.skip].pnl
                    _, p = mannwhitneyu(sk, kp, alternative="less") if len(kp) else (0, np.nan)
                    t2.append({
                        "配對底": name, "排序": rk,
                        "SKIP 數": len(sk),
                        "避開之總損益": round(sk.sum(), 1),
                        "SKIP 者虧損比": f"{(sk < 0).mean() * 100:.0f}%",
                        "留下者虧損比": f"{(kp < 0).mean() * 100:.0f}%",
                        "SKIP 均": round(sk.mean(), 2), "留下均": round(kp.mean(), 2),
                        "MWU p": f"{p:.3f}" if p == p else "—",
                    })

        # ── 四、增益來源分解：SKIP vs 門檻選擇 ────────────────────
        # DRL − Z-Score 的總損益差可完全拆為兩塊：
        #   SKIP 貢獻   = −(Z-Score 在被 SKIP 配對上的損益)   ← 避開的部分
        #   門檻貢獻    = (DRL − Z-Score) 在留下的配對上       ← 選門檻賺到的部分
        #
        # ⚠ 這是**會計恆等式，不是技巧歸因**。底層策略期望值為負時，跳過任何
        #   一批配對（含隨機挑選）期望上都會「避開損失」，故「SKIP 貢獻 X%」
        #   不代表模型學會辨識劣質配對。置換檢定顯示該選擇並不優於隨機
        #   （75 格中僅 7 格顯著，隨機期望 3.75 格；機械成分佔實際避損 42–106%）
        #   —— 見 analysis/prop2_skip_permutation.py。
        #   下方輸出加註 SKIP_性質 欄，避免此表脫離上下文被誤讀。
        if zp is not None and not z.empty:
            m = d[["Ticker_A", "Ticker_B", "Period_Start", "skip", "pnl"]].merge(
                z[["Ticker_A", "Ticker_B", "Period_Start", "pnl"]],
                on=PAIR_KEY, suffixes=("_drl", "_zs"))
            if len(m):
                skip_c = -m.loc[m.skip, "pnl_zs"].sum()
                thr_c = (m.loc[~m.skip, "pnl_drl"] - m.loc[~m.skip, "pnl_zs"]).sum()
                tot = skip_c + thr_c
                t4.append({
                    "配對底": name, "排序": rk,
                    "總增益": round(tot, 1),
                    "SKIP 貢獻": round(skip_c, 1),
                    "門檻貢獻": round(thr_c, 1),
                    "SKIP 佔比": f"{skip_c / tot * 100:.0f}%" if tot else "—",
                    "門檻佔比": f"{thr_c / tot * 100:.0f}%" if tot else "—",
                    "SKIP_性質": SKIP_CAVEAT,
                })

        # ── 三、門檻選擇 vs 排序名次 ──────────────────────────────
        v = traded.dropna(subset=["entry_z", "rank"])
        if len(v) > 10:
            q = pd.qcut(v["rank"], min(3, v["rank"].nunique()),
                        labels=False, duplicates="drop")
            by = v.groupby(q).agg(門檻中位=("entry_z", "median"), n=("entry_z", "size"))
            t3.append({
                "配對底": name, "排序": rk,
                "名次前段門檻": round(by.門檻中位.iloc[0], 2),
                "名次後段門檻": round(by.門檻中位.iloc[-1], 2),
                "Spearman ρ": round(v["rank"].corr(v.entry_z, method="spearman"), 3),
            })

    con.close()

    for title, note, rows in [
        ("一、決策分布：模型實際選了什麼",
         f"動作選單 entry_z ∈ {{1.5, 2.0, 2.5, 3.0}}；靜態基準為 {BASELINE_ENTRY}。", t1),
        ("二、SKIP 的技巧性（反事實對照）",
         "被 DRL 跳過的配對，在 Z-Score 端的實際損益。MWU p 檢定 H1：SKIP 者損益低於留下者。", t2),
        ("三、進場門檻 vs 排序名次",
         "名次為配對品質代理（數字小＝距離近）。ρ > 0 代表對較差的配對要求更高門檻。", t3),
        ("四、增益來源分解：SKIP vs 門檻選擇",
         "DRL − Z-Score 的總損益差，拆為「避開被 SKIP 配對」與「在留下的配對上選門檻」兩塊。\n"
         "  ⚠ 此為會計恆等式，非技巧歸因——期望值為負時隨機跳過亦會避損。\n"
         "     SKIP 的選擇性經置換檢定不顯著（prop2_skip_permutation）；\n"
         "     門檻管道亦經同門檻對照排除（prop2_exposure_control，複製率僅 4.2–20.9%）。", t4),
    ]:
        print("\n" + "=" * 92)
        print(title)
        print("  " + note)
        print("=" * 92)
        print(pd.DataFrame(rows).to_string(index=False) if rows else "  （無資料）")

    os.makedirs(OUT_DIR, exist_ok=True)
    for nm, rows in [("decisions", t1), ("skip_skill", t2), ("threshold_rank", t3),
                     ("gain_decomp", t4)]:
        if rows:
            pd.DataFrame(rows).to_csv(f"{OUT_DIR}/drl_behavior_{nm}.csv",
                                      index=False, encoding="utf-8-sig")
    print(f"\n[已存] {OUT_DIR}/drl_behavior_{{decisions,skip_skill,threshold_rank,gain_decomp}}.csv")


if __name__ == "__main__":
    run()
