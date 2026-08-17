# -*- coding: utf-8 -*-
"""
以修正後的邏輯重算既有摘要的交易統計欄位（不重跑回測）
======================================================================
`db_utils.calculate_metrics_from_params` 修正了兩個計數錯誤：

  1. 分組鍵缺 `Period_Start`——交易期有 6 期重疊，同一配對可在多期同時交易，
     只按 ticker 分組會跨期串接：Entries/Exits 高估、Forced_Closes 低估。
     實測 Grid GICS-SSD/Top20/SL0：Entries 32,245→9,598、Forced 1,425→4,744。
  2. 逐筆損益以「部位狀態區段」加總——進場日的 Prev_Pos 為 0，手續費被歸進
     前一段空手區間形成假交易，筆數約兩倍、勝率腰斬（30.5%→58.3%）。
     改直接取引擎於平倉時寫入的 Trade_PnL（已含進出場兩端費用）。

受影響欄位：Entries / Exits / Stop_Losses / Forced_Closes / Win_Rate /
Profit_Factor / Gross_Profit / Gross_Loss。
**報酬類欄位不受影響**（由權益序列算出），本工具不觸碰它們。

實作方式：全部以 SQL 聚合完成，不把 trade_logs 讀進 pandas。
trade_logs 有 1.2 億列，逐 config `SELECT *` 再重算整套指標會耗時數十分鐘，
而我們只需要九個計數欄位；改用視窗函式後每個 config 只掃自己的索引區間。

用法：
    python -m tools.recompute_trade_stats --dry-run   # 只比對
    python -m tools.recompute_trade_stats             # 寫入（跳過 DRL）
    python -m tools.recompute_trade_stats --include-drl

注意：DRL 策略預設跳過——`run_drl_variance.py` 執行期間會反覆覆寫它們的摘要，
此時重算會與之競爭寫入鎖，且結果隨即被蓋掉。待變異數跑完後再加 --include-drl。
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULT_DB = "results/result.db"

# 逐（配對, 期）切分部位序列，取出：
#   · 每列的前一日部位（判斷是否為出場）
#   · 每組的最後一列（判斷期末是否仍持倉 → 強制平倉）
# 分組鍵含 Period_Start 是本次修正的重點。
_SQL = """
WITH t AS (
    SELECT Ticker_A, Ticker_B, Period_Start, Position, Status, Trade_PnL,
           LAG(Position) OVER w AS prev_pos,
           ROW_NUMBER() OVER (PARTITION BY Ticker_A, Ticker_B, Period_Start
                              ORDER BY Date DESC) AS rn_desc
    FROM trade_logs
    WHERE strategy_id = :sid
    WINDOW w AS (PARTITION BY Ticker_A, Ticker_B, Period_Start ORDER BY Date)
)
SELECT
    SUM(CASE WHEN Position <> COALESCE(prev_pos, 0)
              AND COALESCE(prev_pos, 0) <> 0 THEN 1 ELSE 0 END)            AS exits_total,
    SUM(CASE WHEN Position <> COALESCE(prev_pos, 0)
              AND COALESCE(prev_pos, 0) <> 0
              AND (LOWER(COALESCE(Status,'')) LIKE '%stop%'
                OR LOWER(COALESCE(Status,'')) LIKE '%sl%'
                OR COALESCE(Status,'') LIKE '%停損%') THEN 1 ELSE 0 END)    AS n_stop,
    SUM(CASE WHEN rn_desc = 1 AND Position <> 0 THEN 1 ELSE 0 END)          AS n_forced,
    SUM(CASE WHEN Trade_PnL > 0 THEN Trade_PnL ELSE 0 END)                  AS gross_profit,
    SUM(CASE WHEN Trade_PnL < 0 THEN Trade_PnL ELSE 0 END)                  AS gross_loss,
    SUM(CASE WHEN Trade_PnL > 0 THEN 1 ELSE 0 END)                          AS n_win,
    SUM(CASE WHEN Trade_PnL <> 0 THEN 1 ELSE 0 END)                         AS n_trades
FROM t
"""

STAT_COLS = ["Entries", "Exits", "Stop_Losses", "Forced_Closes",
             "Win_Rate", "Profit_Factor", "Gross_Profit", "Gross_Loss"]


def compute(conn, sid):
    r = conn.execute(_SQL, {"sid": sid}).fetchone()
    if r is None or r[6] is None:
        return None
    exits_total, n_stop, n_forced, gp, gl, n_win, n_trades = [x or 0 for x in r]
    return {
        "Entries":       int(exits_total + n_forced),
        "Exits":         int(exits_total - n_stop),
        "Stop_Losses":   int(n_stop),
        "Forced_Closes": int(n_forced),
        "Win_Rate":      float(n_win / n_trades) if n_trades else 0.0,
        "Profit_Factor": float(gp / abs(gl)) if gl else 0.0,
        "Gross_Profit":  float(gp),
        "Gross_Loss":    float(gl),
    }


def main():
    ap = argparse.ArgumentParser(description="重算交易統計欄位")
    ap.add_argument("--include-drl", action="store_true",
                    help="一併重算 DRL 策略（確認 run_drl_variance 已結束再用）")
    ap.add_argument("--dry-run", action="store_true", help="只比對差異，不寫入")
    args = ap.parse_args()

    conn = sqlite3.connect(RESULT_DB, timeout=180)
    conn.execute("PRAGMA busy_timeout=180000")
    rows = pd.read_sql_query(
        'SELECT _path, "METHOD", Entries, Forced_Closes, Win_Rate '
        "FROM strategy_summaries", conn)
    if not args.include_drl:
        skip = rows["METHOD"].str.contains("DRL", na=False)
        print(f"跳過 {int(skip.sum())} 個 DRL 配置（變異數重跑中會被覆寫）")
        rows = rows[~skip]

    print(f"待重算 {len(rows)} 個配置")
    changed, sample = 0, []
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        m = compute(conn, r["_path"])
        if m is None:
            print(f"  ⚠ 無 trade_logs，跳過：{r['_path']}")
            continue
        if int(m["Entries"]) != int(r["Entries"] or 0):
            changed += 1
            if len(sample) < 5:
                sample.append((r["METHOD"], int(r["Entries"] or 0), m["Entries"],
                               int(r["Forced_Closes"] or 0), m["Forced_Closes"],
                               float(r["Win_Rate"] or 0), m["Win_Rate"]))
        if not args.dry_run:
            conn.execute(
                f"UPDATE strategy_summaries SET {', '.join(c + ' = ?' for c in STAT_COLS)} "
                "WHERE _path = ?",
                [m[c] for c in STAT_COLS] + [r["_path"]])
            conn.commit()      # 逐列提交：與背景回測共用 DB 時盡量縮短持鎖時間
        if i % 100 == 0:
            print(f"  …{i}/{len(rows)}")

    conn.close()
    if sample:
        print("\n差異範例（Entries / Forced_Closes / Win_Rate 舊→新）：")
        for m_, e0, e1, f0, f1, w0, w1 in sample:
            print(f"  {m_:<26} {e0:>7,}→{e1:>7,}   {f0:>6,}→{f1:>6,}   {w0:.3f}→{w1:.3f}")
    verb = "將更新" if args.dry_run else "已更新"
    print(f"\n{verb} {changed}/{len(rows)} 個配置")


if __name__ == "__main__":
    main()
