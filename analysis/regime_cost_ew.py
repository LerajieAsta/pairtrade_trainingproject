"""
Regime 分層 Sharpe 與 break-even 成本（**等權組合口徑**）
========================================================

`regime_cost_dsr_eval.py` 的兩張表以「網格最佳配置」計算。那個口徑用來刻畫
單一策略的風險形狀是可以的，但**不能拿來做兩臂比較**——兩臂各自取最佳格時
選到的格子往往不同（實測 2026-08-12：`Grid (HDB-SDP)` 最佳為 Top3/SL0%，
而 `Grid (HDB-SDP-DRL)` 為 Top1/SL0%），於是「DL-THR 改善了幾格」同時混入
交易端與網格格子兩個變因，且雙方都各自吃了一次 15 選 1 的選擇偏誤。

本模組改以 3.5.5 節規定的主口徑重算同樣兩張表：

  * **15 格等權**：兩臂各自把 15 個參數格的逐日報酬等權平均後才比較
  * **逐格對齊**：只取兩臂都存在的格，故差異的唯一變因是交易端
  * 抽樣口徑與 4.2 節的差分主檢定一致（`load_daily_sids`，單利）

兩份輸出的差別很大，且方向相反，故兩者都要看：

  | | 最佳格 | 等權逐格對齊 |
  | regime 改善格數 | 12 改 / 8 劣 | **25 改 / 0 劣** |
  | DL-THR 提高 break-even | 3/5 | **5/5** |
  | 成本餘裕 | +8.3 ~ +54.2 bps | **−18.9 ~ +34.4 bps** |

增益的兩項宣稱在等權口徑下更乾淨也更強。成本餘裕的方向也如預期：最佳格
10/10 為正（內含 15 選 1 的選擇偏誤），等權口徑下三個 Z-Score 臂已轉負
（AGG −2.5、HDB −6.1、KM −18.9 bps），與全篇「等權絕對報酬約等於零」對得上。

⚠ **上表數字於 2026-08-27 全面更新**，因本檔的名目額計算修正（見 `_breakeven`
的註解）。舊值為最佳格 1.3~17.9 bps、等權 −1.8~+2.4 bps，係低估名目額所致。
更新後餘裕的**全距大幅拉開**：GICS 兩底加 DL-THR 的等權餘裕由 +2.4 升至 +34 bps
（BE 0.92% 對成本 0.58%），K-means 則由 −1.8 惡化至 −18.9 bps。
**「成本餘裕極薄」這個描述已不適用於 GICS 臂**——論文 4.6.2 與 5.3 的對應
段落仍寫著舊值，尚未更新。

註：`breakeven_dsr.csv` 的 `成本餘裕(vs0.58%)` 欄**單位為百分點、非 bps**。

用法：python -m analysis.regime_cost_ew
輸出：results/analysis/regime_sharpe_ew.csv、breakeven_ew.csv

上表「最佳格」欄的數字可由 `regime_cost_dsr_eval.run(methods=[...])` 傳入該八條
策略重現，輸出已存為 `regime_sharpe_main8.csv` / `breakeven_dsr_main8.csv`
（4.6.1、4.6.2 兩處「口徑更正」註腳所引用者）。注意 `regime_cost_dsr_eval.run()`
不帶參數時預設會**濾掉所有 METHOD 含 "DRL" 者**，故其預設輸出
（`regime_sharpe.csv` / `breakeven_dsr.csv`，98 條）本來就不含 DL-THR 臂，
不可用來做兩臂比較——那兩份的用途是 DSR 的試驗宇宙全表。
"""

import os
import sqlite3

import numpy as np
import pandas as pd

from analysis.proposition2_daily_hac import (
    PAIRS, RESULT_DB, baseline_only, _grid_cell, load_daily_sids)
from analysis.regime_cost_dsr_eval import (
    build_market_regimes, CURRENT_FEE_SIDE, TRADING_DAYS, _top_n_int)
from strategies.metrics import traded_notional
from strategies.config import INITIAL_CAPITAL

OUT_DIR = "results/analysis"
CURRENT_RT_FEE = 2 * CURRENT_FEE_SIDE          # 現行往返假設（0.58%）
VOL_LABELS = ["Calm", "Normal", "Turbulent"]
TREND_LABELS = ["Bull", "Bear"]


def _load_meta():
    """{METHOD: [基準格 _path]}，並回傳完整 summary 供 break-even 用。"""
    methods = sorted({m for _, z, d in PAIRS for m in (z, d)})
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    summ = pd.read_sql(
        f"SELECT * FROM strategy_summaries WHERE METHOD IN "
        f"({','.join('?' * len(methods))})", con, params=methods)
    con.close()
    # 只取基準格：entry_z 等變體與基準共用 db_method，混進來會污染等權組合
    m2s = {m: baseline_only(g._path.tolist()) for m, g in summ.groupby("METHOD")}
    return summ, m2s


def _matched_cells(m2s, zs_m, drl_m):
    """兩臂都存在的網格格子，以及各自的 cell → _path 對照。"""
    zc = {_grid_cell(s): s for s in m2s.get(zs_m, [])}
    dc = {_grid_cell(s): s for s in m2s.get(drl_m, [])}
    return sorted(set(zc) & set(dc)), zc, dc


# ── 表 1：regime 分層 Sharpe（等權組合）────────────────────────────
def _regime_sharpe(series: pd.Series, regimes: pd.DataFrame) -> dict:
    j = pd.DataFrame({"ret": series}).join(
        regimes[["vol_regime", "trend_regime"]], how="inner")
    out = {}
    for col, labs in (("vol_regime", VOL_LABELS), ("trend_regime", TREND_LABELS)):
        for lab in labs:
            sub = j[j[col] == lab]["ret"]
            out[lab] = (round(sub.mean() / sub.std(ddof=1) * np.sqrt(TRADING_DAYS), 2)
                        if len(sub) > 20 and sub.std(ddof=1) > 0 else np.nan)
    return out


def regime_table(m2s, regimes):
    px = load_daily_sids([s for v in m2s.values() for s in v])
    rows, imp, dec, eq = [], 0, 0, 0
    for base, zs_m, drl_m in PAIRS:
        cells, zc, dc = _matched_cells(m2s, zs_m, drl_m)
        if not cells:
            print(f"  ⚠ 略過 {base}：兩臂無共同網格格")
            continue
        z = _regime_sharpe(px[[zc[c] for c in cells]].mean(axis=1), regimes)
        d = _regime_sharpe(px[[dc[c] for c in cells]].mean(axis=1), regimes)
        rows.append({"配對底": base, "格數": len(cells), "交易端": "Z-Score", **z})
        rows.append({"配對底": base, "格數": len(cells), "交易端": "DL-THR", **d})
        for lab in VOL_LABELS + TREND_LABELS:
            if d[lab] > z[lab]:
                imp += 1
            elif d[lab] < z[lab]:
                dec += 1
            else:
                eq += 1
    return pd.DataFrame(rows), (imp, dec, eq)


# ── 表 2：break-even 成本（等權組合）──────────────────────────────
def _breakeven(summ, m2s, method, cells_wanted):
    """
    往返 break-even = 2 × (現行單邊費 + Σ淨利 / Σ名目額)。

    成本模型可解析求解：進出場費用 = friction × 名目額，且名目額恰等於每配對
    資金，故 15 格加總後同一恆等式成立（各格資金相同，加總與平均給出同一比值）。
    """
    keep = set(m2s[method])
    g = summ[(summ.METHOD == method) & summ._path.isin(keep)]
    net = notional = 0.0
    n = 0
    for _, r in g.iterrows():
        if _grid_cell(r["_path"]) not in cells_wanted:
            continue
        # 名目額改用 regime_cost_dsr_eval.traded_notional（2026-08-27 修正）。
        # 舊算法在兩處低估 break-even，與該檔 2026-08-26 修正的是同一組錯誤：
        #   其一，cap = INITIAL_CAPITAL / top_n 漏掉並行期數（max_pairs = top_n × 6），
        #        且應用逐日權益而非初始資金；
        #   其二，事件數用 Entries + Exits，漏計停損與強制平倉的出場費
        #        （實測 Entries = Exits + Stop_Losses + Forced_Closes）。
        # 當時只修了 regime_cost_dsr_eval.py，本檔被漏掉——而它產生的
        # breakeven_ew.csv 為論文 4.x 與 5.3「成本餘裕過薄」所引用。
        notional += traded_notional(r["_path"], _top_n_int(r["TOP N"]))
        net += float(r["Final_Equity"]) - INITIAL_CAPITAL
        n += 1
    if notional <= 0:
        return np.nan, n
    return 2.0 * (CURRENT_FEE_SIDE + net / notional) * 100, n


def breakeven_table(summ, m2s):
    rows = []
    for base, zs_m, drl_m in PAIRS:
        cells, _, _ = _matched_cells(m2s, zs_m, drl_m)
        if not cells:
            continue
        cells = set(cells)
        z_be, n = _breakeven(summ, m2s, zs_m, cells)
        d_be, _ = _breakeven(summ, m2s, drl_m, cells)
        rows.append({
            "配對底": base, "格數": n,
            "Z-Score 往返BE%": round(z_be, 3),
            "DL-THR 往返BE%": round(d_be, 3),
            "差(pp)": round(d_be - z_be, 3),
            "Z 餘裕(bps)": round((z_be - CURRENT_RT_FEE * 100) * 100, 1),
            "DL 餘裕(bps)": round((d_be - CURRENT_RT_FEE * 100) * 100, 1),
        })
    return pd.DataFrame(rows)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    summ, m2s = _load_meta()
    regimes = build_market_regimes()

    tbl1, (imp, dec, eq) = regime_table(m2s, regimes)
    tbl2 = breakeven_table(summ, m2s)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 88)
    print("表 1：Regime 分層年化 Sharpe（15 格等權、逐格對齊兩臂）")
    print("=" * 88)
    print(tbl1.to_string(index=False))
    print(f"\n  改善 {imp} / 劣化 {dec} / 持平 {eq}（共 {imp + dec + eq} 格）")
    calm = tbl1["Calm"].dropna()
    print(f"  Calm 全距 {calm.min():.2f} ~ {calm.max():.2f}；為正者 {(calm > 0).sum()} 個")

    print("\n" + "=" * 88)
    print("表 2：往返 break-even 成本（15 格等權、逐格對齊兩臂）")
    print("=" * 88)
    print(tbl2.to_string(index=False))
    up = int((tbl2["差(pp)"] > 0).sum())
    margins = tbl2["Z 餘裕(bps)"].tolist() + tbl2["DL 餘裕(bps)"].tolist()
    print(f"\n  DL-THR 提高 break-even：{up}/{len(tbl2)} 個配對底")
    print(f"  成本餘裕全距 {min(margins):.1f} ~ {max(margins):.1f} bps"
          f"（現行假設往返 {CURRENT_RT_FEE * 100:.2f}%）")

    tbl1.to_csv(f"{OUT_DIR}/regime_sharpe_ew.csv", index=False, encoding="utf-8-sig")
    tbl2.to_csv(f"{OUT_DIR}/breakeven_ew.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已存] {OUT_DIR}/regime_sharpe_ew.csv, breakeven_ew.csv")
    return tbl1, tbl2


if __name__ == "__main__":
    run()
