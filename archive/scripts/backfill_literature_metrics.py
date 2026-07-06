"""
回填文獻口徑績效欄位（Avg_Utilization / Ann_Ret_Employed / Excess_Ret_RF）
到 results/result.db 既有的 strategy_summaries 列。

新回測由 db_utils.calculate_metrics_from_params 自動寫入；
本腳本只處理欄位新增前的歷史列（值為 NULL 者）。

用法：python archive/backfill_literature_metrics.py
注意：請在沒有 run_trading.py 執行中時運行（避免 SQLite 寫入鎖）。
"""
import re
import sqlite3

import sys
sys.path.insert(0, ".")
from strategies.config import INITIAL_CAPITAL, CONCURRENT_PERIODS, RF_ANNUAL

DB = "results/result.db"

con = sqlite3.connect(DB)
cur = con.cursor()

# 欄位遷移（冪等）
for col in ("Avg_Utilization", "Ann_Ret_Employed", "Excess_Ret_RF"):
    try:
        cur.execute(f'ALTER TABLE strategy_summaries ADD COLUMN "{col}" REAL;')
    except sqlite3.OperationalError:
        pass
con.commit()

todo = cur.execute(
    'SELECT _path, "TOP N", Final_Equity FROM strategy_summaries WHERE Avg_Utilization IS NULL'
).fetchall()
print(f"待回填: {len(todo)} 列")
if not todo:
    raise SystemExit("無需回填。")

# 單次全表聚合：每策略的交易日數與持倉配對日總數
print("聚合 trade_logs（單次全表掃描，約數分鐘）...")
agg = {
    r[0]: (r[1], r[2])
    for r in cur.execute("""
        SELECT strategy_id, COUNT(DISTINCT Date),
               SUM(CASE WHEN Position != 0 THEN 1 ELSE 0 END)
        FROM trade_logs GROUP BY strategy_id
    """)
}
print(f"聚合完成：{len(agg)} 個 strategy_id")

updates, skipped = [], 0
for path, top_n_str, final_eq in todo:
    m = re.search(r"(\d+)", str(top_n_str or ""))
    if path not in agg or not m:
        skipped += 1
        continue
    n_days, open_days = agg[path]
    top_n = int(m.group(1))
    max_pairs = top_n * CONCURRENT_PERIODS
    if not n_days or max_pairs <= 0:
        skipped += 1
        continue
    years = n_days / 252.0
    final_pnl = (final_eq or INITIAL_CAPITAL) - INITIAL_CAPITAL
    avg_open = (open_days or 0) / n_days
    util = avg_open / max_pairs
    ann_arith = final_pnl / INITIAL_CAPITAL / years if years > 0 else 0.0
    avg_employed = avg_open * (INITIAL_CAPITAL / max_pairs)
    ann_employed = final_pnl / avg_employed / years if (avg_employed > 0 and years > 0) else 0.0
    excess_rf = ann_arith - RF_ANNUAL * util
    updates.append((util, ann_employed, excess_rf, path))

cur.executemany(
    """UPDATE strategy_summaries
       SET Avg_Utilization = ?, Ann_Ret_Employed = ?, Excess_Ret_RF = ?
       WHERE _path = ?""",
    updates,
)
con.commit()
print(f"回填完成: {len(updates)} 列 | 跳過(無交易紀錄): {skipped} 列")
con.close()
