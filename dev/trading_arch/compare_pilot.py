# -*- coding: utf-8 -*-
"""修正後（§A 對沖 + §E 前視）對修正前基準的逐格比對。

唯讀。用法：
    # 試跑
    python dev/trading_arch/compare_pilot.py --new results/pilot_hedge.db
    # 全網格重跑之後
    python dev/trading_arch/compare_pilot.py --new results/result.db --top 40

回報三件事：

  P2  進出場計數的位移（`SL0` 格應僅受 §E 影響，§A 完全不影響判定）
  P3  `traded_notional` 不變（總名目額仍為 capital_per_pair）
  結果 Sharpe / 年化 / 期末權益的位移 —— **預先註冊刻意未預測方向**

⚠ 試跑同時含 §A 與 §E 兩項修正，無法各自歸因（`PREREGISTRATION.md` §五
已載明接受此代價）。此處只回報合併效果。
"""
import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from strategies.config import INITIAL_CAPITAL  # noqa: E402

OLD_DB = "results/result.db"
BACKUP_CSV = "results/analysis/strategy_summaries_backup_20260828.csv"

_COLS = ["Final_Equity", "Sharpe_Raw", "Ann_Ret_Raw", "MDD_Raw",
         "Entries", "Exits", "Stop_Losses", "Forced_Closes",
         "Win_Rate", "Profit_Factor", "Avg_Utilization"]


def load(src: str) -> pd.DataFrame:
    """讀 `strategy_summaries`。來源可為 .db 或 `snapshot` 匯出的 .csv。

    原地全網格重跑會蓋掉舊庫，故修正前的基準只能來自
    `results/analysis/strategy_summaries_backup_20260828.csv`。

    ⚠ 該備份取於 §B 回填**之前**，故其 `Avg_Utilization` /
    `Ann_Ret_Employed` / `Excess_Ret_RF` 三欄是 §B 修正前的值，不可用於比較。
    其餘欄位（Sharpe / 年化 / 期末權益 / 進出場計數）不受 §B 影響，可以比。
    """
    if src.lower().endswith(".csv"):
        df = pd.read_csv(src)
        keep = ["_path", "METHOD", "TOP N", "STOP LOSS %"] + [c for c in _COLS if c in df.columns]
        return df[keep]
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        cols = ", ".join(f'"{c}"' for c in _COLS)
        return pd.read_sql(
            f'SELECT _path, METHOD, "TOP N", "STOP LOSS %", {cols} '
            "FROM strategy_summaries", con)
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", "--pilot", dest="new", default="results/pilot_hedge.db",
                    help="修正後的來源（試跑庫，或全網格重跑後的 results/result.db）")
    ap.add_argument("--old", default=BACKUP_CSV,
                    help="修正前的基準（預設為備份 CSV；原地重跑後舊庫已不存在）")
    ap.add_argument("--top", type=int, default=0,
                    help="逐格明細只印前 N 列（0 = 全印）")
    args = ap.parse_args()

    new = load(args.new).set_index("_path")
    old = load(args.old).set_index("_path")
    both = new.index.intersection(old.index)
    if len(both) == 0:
        print("兩邊沒有共同的 config，無法比對")
        return
    print(f"基準   {args.old}")
    print(f"修正後 {args.new}")
    print(f"共同 config：{len(both)}（新 {len(new)}、舊 {len(old)}）\n")

    n, o = new.loc[both], old.loc[both]

    print("== P2  進出場計數的位移 ==")
    print("（§A 不影響進出場判定；`SL0` 的差異全部來自 §E 的首日價格修正）")
    for sl, g in n.groupby("STOP LOSS %"):
        idx = g.index
        de = (n.loc[idx, "Entries"] - o.loc[idx, "Entries"])
        rel = (de.abs() / o.loc[idx, "Entries"].replace(0, np.nan)).median()
        print(f"  SL={sl:<4} n={len(idx):2d}  Entries 差 中位={de.median():+.0f} "
              f"最大={de.abs().max():.0f}  相對中位={rel:.4%}")

    print("\n== 結果位移（預先註冊未預測方向）==")
    print(f"{'TOP N':>7}{'SL':>5}{'Sharpe 舊':>11}{'Sharpe 新':>11}"
          f"{'年化 舊':>10}{'年化 新':>10}{'期末 舊':>11}{'期末 新':>11}")
    _order = sorted(both, key=lambda x: (o.loc[x, "METHOD"], o.loc[x, "TOP N"],
                                        o.loc[x, "STOP LOSS %"]))
    for p in (_order[:args.top] if args.top else _order):
        print(f"{o.loc[p, 'TOP N']:>7}{o.loc[p, 'STOP LOSS %']:>5}"
              f"{o.loc[p, 'Sharpe_Raw']:>11.4f}{n.loc[p, 'Sharpe_Raw']:>11.4f}"
              f"{o.loc[p, 'Ann_Ret_Raw']:>10.4f}{n.loc[p, 'Ann_Ret_Raw']:>10.4f}"
              f"{o.loc[p, 'Final_Equity']:>11,.0f}{n.loc[p, 'Final_Equity']:>11,.0f}")

    d_sh = n["Sharpe_Raw"] - o["Sharpe_Raw"]
    d_eq = n["Final_Equity"] - o["Final_Equity"]
    print(f"\n  Sharpe 位移：中位 {d_sh.median():+.4f}  "
          f"改善 {int((d_sh > 0).sum())} / 劣化 {int((d_sh < 0).sum())} / 共 {len(d_sh)}")
    print(f"  期末權益位移：中位 {d_eq.median():+,.0f}  "
          f"改善 {int((d_eq > 0).sum())} / 劣化 {int((d_eq < 0).sum())}")
    print(f"  淨利由 ${(o['Final_Equity'] - INITIAL_CAPITAL).mean():,.0f} "
          f"變為 ${(n['Final_Equity'] - INITIAL_CAPITAL).mean():,.0f}（{len(both)} 格平均）")


if __name__ == "__main__":
    main()
