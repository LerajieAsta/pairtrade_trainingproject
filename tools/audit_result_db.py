"""
result.db 完整性稽核：找出「有摘要、無逐日明細」的策略
======================================================================

`strategy_summaries` 有一列、`trade_logs` 卻沒有對應列的策略，會讓分析端讀得到
績效數字卻讀不到日損益序列。等權平均因此少算一格——而且不會報錯。2026-08-03
就是這樣丟掉 Grid (AGG-SSD-NOSEC-GI) 的 Top3_SL0，直到統計結果對不上才發現。

成因是併發寫入競爭（database is locked）。db_utils 現在會在寫入失敗時清除半成品、
run_trading 會把它報成 FAILED，但那只對新的執行有效；既有資料庫仍需本工具檢查。

不用 `SELECT DISTINCT strategy_id FROM trade_logs`——那要掃三億列。改為逐條
EXISTS 探測，每次走 strategy_id 索引。

用法：
    python tools/audit_result_db.py
    python tools/audit_result_db.py --db results/result.db
"""

import argparse
import sqlite3
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def audit(db_path: str) -> list[tuple[str, object]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    summaries = pd.read_sql("SELECT _path, Entries FROM strategy_summaries", con)
    print(f"摘要列 {len(summaries)} 條，逐條探測 trade_logs…")

    missing = []
    for i, (path, entries) in enumerate(zip(summaries._path, summaries.Entries), 1):
        try:
            if float(entries) <= 0:
                continue          # 本來就沒有交易，沒有明細是正常的
        except (TypeError, ValueError):
            continue
        hit = con.execute(
            "SELECT 1 FROM trade_logs WHERE strategy_id = ? LIMIT 1", (path,)
        ).fetchone()
        if hit is None:
            missing.append((path, entries))
            print(f"  ✘ {path}  (Entries={entries})")
        if i % 500 == 0:
            print(f"  …{i}/{len(summaries)}")
    con.close()
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/result.db")
    a = ap.parse_args()

    missing = audit(a.db)
    if not missing:
        print("\n✔ 完整：每一條有交易紀錄的策略都有對應的逐日明細。")
        return 0

    print(f"\n✘ {len(missing)} 條有進場紀錄卻無逐日明細：")
    for path, entries in missing:
        print(f"    {path}  Entries={entries}")
    print("\n補救：以 FORCE_RERUN=1 STRATEGIES_SLICE=<索引> 重跑對應策略。"
          "\n併發寫入競爭是主因，重跑時建議調低 CPU_LIMIT_PCT（例如 0.35）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
