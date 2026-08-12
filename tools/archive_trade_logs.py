#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
封存策略的 trade_logs 汰除：先保全逐日序列，再刪明細
======================================================================

`result.db` 的 216GB 裡有 127GB（268M 列、58.8%）屬於**非現役**策略——
70 個已封存的 METHOD、1,180 個網格格。它們的 `strategy_summaries` 摘要列
是研究紀錄（也是 DSR 試驗宇宙 N=104 的來源），必須留；但逐日逐配對的
`trade_logs` 明細，在分析上已無用處。

為什麼刪了檔案卻不會變小
----------------------------------------------------------------------
SQLite 刪列只把頁面還給 freelist，不還給作業系統。要真的縮檔得 VACUUM，
而 VACUUM 需要約與資料庫等量的暫存空間（216GB）——本機沒有。

**刪除的實益是「讓後續寫入重用這些頁面」**：接下來的 run_trading 全量重跑
會寫入兩億多列，不先刪的話 DB 會漲到 300GB+ 而撐爆磁碟；先刪則檔案不再長大。

刪之前一定要做的事
----------------------------------------------------------------------
封存策略的**逐日損益序列**仍被 DSR（92 個策略族）與 regime 分析使用。
明細一刪就再也導不回來，故本腳本強制順序：

    1. 把待刪 strategy_id 的逐日損益補進 parquet 快取（可中斷、可續跑）
    2. 驗證每一條都在快取裡，缺一條就中止
    3. 才開始分批刪除 trade_logs

用法：
    python -m tools.archive_trade_logs --dry-run     # 只看會刪什麼、量多大
    python -m tools.archive_trade_logs --cache-only  # 只補快取，不刪
    python -m tools.archive_trade_logs --execute     # 補快取 → 驗證 → 刪除
"""
import argparse
import os
import sqlite3
import sys
import time

import pandas as pd

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_DB = "results/result.db"
CACHE = "results/analysis/daily_returns_mainaxis.parquet"
BATCH = 20          # 每批刪幾個 strategy_id
CACHE_CHUNK = 40    # 每次聚合幾個 strategy_id


def dead_sids() -> pd.DataFrame:
    """非現役 METHOD 的所有網格格。"""
    from strategies.config import strategies_raw_all
    live = {s["db_method"] for s in strategies_raw_all}
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    sm = pd.read_sql("SELECT _path, METHOD FROM strategy_summaries", con)
    con.close()
    return sm[~sm.METHOD.isin(live)].reset_index(drop=True)


def ensure_cached(sids: list[str]) -> None:
    """把尚未快取的 strategy_id 逐日損益補進 parquet（增量、可中斷）。"""
    cached = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    missing = [s for s in sids if s not in cached.columns]
    if not missing:
        print(f"  快取已完整涵蓋 {len(sids)} 條，無需補齊")
        return
    print(f"  待補 {len(missing)} 條（快取現有 {len(cached.columns)} 條）")

    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    t0 = time.time()
    for i in range(0, len(missing), CACHE_CHUNK):
        blk = missing[i:i + CACHE_CHUNK]
        q = (f"SELECT strategy_id, Date, SUM(Daily_Delta) AS pnl FROM trade_logs "
             f"WHERE strategy_id IN ({','.join('?' * len(blk))}) "
             f"GROUP BY strategy_id, Date")
        raw = pd.read_sql(q, con, params=blk)
        if not raw.empty:
            raw["Date"] = pd.to_datetime(raw["Date"])
            new = raw.pivot(index="Date", columns="strategy_id", values="pnl")
            cached = new if cached.empty else cached.join(new, how="outer")
            cached = cached.sort_index().fillna(0.0)
        # 每批就落地一次：中途斷了也不用重跑前面的
        cached.to_parquet(CACHE)
        done = min(i + CACHE_CHUNK, len(missing))
        el = time.time() - t0
        print(f"    {done}/{len(missing)}  已耗 {el/60:.1f} 分"
              f"  預估剩 {el/done*(len(missing)-done)/60:.1f} 分", flush=True)
    con.close()


def verify(sids: list[str]) -> list[str]:
    """
    回傳「有明細可失去、卻沒進快取」的 strategy_id——這些是不可刪的。

    有一類 sid 永遠聚合不出逐日序列：`trade_logs` 本來就 0 列。成因是
    result.db 曾在高併發下發生 "database is locked"，摘要列寫成功但明細
    整批遺失（見 config.py 的 CPU_LIMIT_PCT 註解）。這種 sid 沒東西可保全、
    也沒東西可刪，不該擋住整批作業，但要列出來讓人知道資料庫有這個缺口。
    """
    cached = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    absent = [s for s in sids if s not in cached.columns]
    if not absent:
        return []
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    risky, empty = [], []
    for s in absent:
        n = con.execute(
            "SELECT COUNT(*) FROM trade_logs WHERE strategy_id=?", (s,)).fetchone()[0]
        (empty if n == 0 else risky).append(s)
    con.close()
    if empty:
        print(f"  ⚠ {len(empty)} 條在 trade_logs 中本來就是 0 列（摘要列存在但明細遺失），"
              f"視為安全：")
        for s in empty:
            print(f"      {s}")
    return risky


def delete(sids: list[str]) -> None:
    con = sqlite3.connect(RESULT_DB, timeout=120.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=OFF;")
    cur = con.cursor()
    t0 = time.time()
    total = 0
    for i in range(0, len(sids), BATCH):
        blk = sids[i:i + BATCH]
        cur.execute(
            f"DELETE FROM trade_logs WHERE strategy_id IN ({','.join('?' * len(blk))})",
            blk)
        total += cur.rowcount
        con.commit()
        done = min(i + BATCH, len(sids))
        el = time.time() - t0
        print(f"    {done}/{len(sids)} 條  已刪 {total:,} 列  耗 {el/60:.1f} 分"
              f"  預估剩 {el/done*(len(sids)-done)/60:.1f} 分", flush=True)
    con.close()
    print(f"\n  合計刪除 {total:,} 列")


def main():
    ap = argparse.ArgumentParser(description="汰除封存策略的 trade_logs 明細")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="只列出將刪除的範圍")
    g.add_argument("--cache-only", action="store_true", help="只補逐日快取，不刪除")
    g.add_argument("--execute", action="store_true", help="補快取 → 驗證 → 刪除")
    args = ap.parse_args()

    dead = dead_sids()
    sids = dead._path.tolist()
    print(f"非現役 METHOD {dead.METHOD.nunique()} 個 / 網格格 {len(sids)} 條")

    if args.dry_run:
        print("\n各 METHOD 的網格格數（前 20）：")
        print(dead.groupby("METHOD").size().sort_values(ascending=False).head(20).to_string())
        left = verify(sids)
        print(f"\n尚未進逐日快取者：{len(left)} 條"
              f"（--execute 會先補齊，補不齊就中止不刪）")
        return

    print("\n[1/3] 補齊逐日快取")
    ensure_cached(sids)

    print("\n[2/3] 驗證")
    left = verify(sids)
    if left:
        sys.exit(f"  ✘ 仍有 {len(left)} 條有明細但未進快取，中止刪除（例：{left[:3]}）")
    print(f"  ✔ {len(sids)} 條的逐日序列全部已保全（或本來就無明細）")

    if args.cache_only:
        print("\n--cache-only：不執行刪除")
        return

    print("\n[3/3] 刪除 trade_logs 明細")
    delete(sids)
    print("\n完成。注意：檔案大小不會變小（頁面進 freelist），"
          "但後續寫入會重用這些空間。")


if __name__ == "__main__":
    main()
