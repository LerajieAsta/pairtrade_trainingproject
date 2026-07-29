# -*- coding: utf-8 -*-
"""
回填 strategy_summaries 的 ENTRY Z / DYN Z NUM（一次性遷移）
======================================================================
背景：兩欄早已存在於 schema，但 `db_utils.export_df_to_db` 的 INSERT 未涵蓋，
故歷來所有列皆為 NULL。交易端變體（entry_z 掃描等）與其基準**共用 db_method**，
兩欄留白時在資料層無法區分，下游只能各自以 `_path` 檔名後綴反推——已知有四處
分析因此誤把對照組當成現役策略（proposition2_stats 崩潰、comparison 去重誤留、
drl_behavior 選錯行為解析格）。

`db_utils.py` 已修為寫入時落庫；本腳本補既有列。

值的來源：`run_trading._log_name` 的檔名約定——
    基準（entry_z=2.0、dynamic_stop_z=0）不加後綴；
    否則加 `_EZ{entry_z*10:.0f}_DSZ{dynamic_stop_z*10:.0f}`。
故由 `_path` 可無損反推兩欄。

用法：
    python -m tools.backfill_entry_z            # 試跑，只報告不寫入
    python -m tools.backfill_entry_z --apply    # 實際寫入
"""
import re
import sqlite3
import sys

RESULT_DB = "results/result.db"
EZ_RE = re.compile(r"_EZ(\d+)_DSZ(\d+)")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_path(path: str) -> tuple[float, float]:
    """_path → (entry_z, dynamic_stop_z)。無後綴者為基準 (2.0, 0.0)。"""
    m = EZ_RE.search(path)
    if not m:
        return 2.0, 0.0
    return int(m.group(1)) / 10.0, int(m.group(2)) / 10.0


def run(apply: bool = False):
    con = sqlite3.connect(RESULT_DB)
    cur = con.cursor()
    rows = cur.execute(
        'SELECT _path, "ENTRY Z", "DYN Z NUM" FROM strategy_summaries').fetchall()

    updates, already, mismatch = [], 0, []
    for path, ez_db, dsz_db in rows:
        ez, dsz = parse_path(path)
        if ez_db is None or dsz_db is None:
            updates.append((ez, dsz, path))
        elif abs(ez_db - ez) > 1e-9 or abs(dsz_db - dsz) > 1e-9:
            mismatch.append((path, ez_db, ez, dsz_db, dsz))
        else:
            already += 1

    print(f"總列數 {len(rows)}")
    print(f"  待回填（NULL）      {len(updates)}")
    print(f"  已有值且一致        {already}")
    print(f"  已有值但與路徑不符  {len(mismatch)}")
    for m in mismatch[:5]:
        print(f"    ⚠ {m[0]}  DB=({m[1]},{m[3]}) 路徑=({m[2]},{m[4]})")

    dist = {}
    for ez, dsz, _ in updates:
        dist[(ez, dsz)] = dist.get((ez, dsz), 0) + 1
    print("  回填值分布：" + "  ".join(
        f"EZ{k[0]}/DSZ{k[1]}×{v}" for k, v in sorted(dist.items())))

    if not apply:
        print("\n（試跑，未寫入。加 --apply 實際執行）")
        con.close()
        return

    cur.executemany(
        'UPDATE strategy_summaries SET "ENTRY Z" = ?, "DYN Z NUM" = ? WHERE _path = ?',
        updates)
    con.commit()
    left = cur.execute(
        'SELECT COUNT(*) FROM strategy_summaries WHERE "ENTRY Z" IS NULL').fetchone()[0]
    con.close()
    print(f"\n已寫入 {len(updates)} 列；剩餘 ENTRY Z 為 NULL 者：{left}")


if __name__ == "__main__":
    run(apply="--apply" in sys.argv)
