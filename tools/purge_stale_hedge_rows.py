#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""刪除全網格重跑後留下的、無法以現行 config 重現的舊口徑列。

背景見 `dev/trading_arch/REVIEW.md` §A。2026-08-28 的全網格重跑把
`run_trading.py` 由現行 config 展開得出的 627 個配置轉成 `hedge_mode="signal"`，
但 `result.db` 另有 795 列是**現行 config 展開不出來**的歷史配置，仍為舊口徑。

兩種口徑的損益序列不可並列比較（`strategies/returns.py:assert_uniform_hedge_mode`
會擋下），故必須逐類決定去留：

| 類別 | 列數 | 處置 | 理由 |
|:---|---:|:---|:---|
| `EZ` / `DSZ` 參數掃描 | 525 | **補跑** | 論文引用；可用 `SENSITIVITY_PARAM` 重現 |
| `MHD` 時間停損掃描 | 180 | **刪除** | 需還原一次性設定；未進論文主表 |
| 已退役的 METHOD | 60 | **刪除** | config 已無條目，不可能重現 |
| VG 舊檔（無 `_VG` 後綴） | 30 | **刪除** | `_build_filename` 的 `_VG` 後綴在該臂首跑後才加入，本次重跑寫到新檔名而未覆寫舊列，兩者並存；新列已取代之 |

本腳本只做「刪除」那三類。補跑由 `tools/rerun_ez_dsz.py` 負責。

⚠ 刪除會連同 `trade_logs` 的逐筆明細一起消失，無法復原。
   `results/analysis/strategy_summaries_backup_20260828.csv` 留有全部摘要。

用法：
    python tools/purge_stale_hedge_rows.py --dry-run
    python tools/purge_stale_hedge_rows.py --execute
"""
import argparse
import os
import re
import sqlite3
import sys

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULT_DB = "results/result.db"

#: 保留（另行補跑），不在本腳本處理範圍
_KEEP_RE = re.compile(r"_(EZ\d+|DSZ\d+)")
#: 刪除：時間停損掃描
_MHD_RE = re.compile(r"_MHD\d+")


def classify(path: str) -> str:
    fname = path.rsplit("/", 1)[-1]
    subdir = path.rsplit("/", 2)[1] if path.count("/") >= 2 else ""
    if _KEEP_RE.search(fname):
        return "keep_ez_dsz"
    if _MHD_RE.search(fname):
        return "drop_mhd"
    if "_VG" in subdir and "_VG" not in fname:
        return "drop_vg_stale"
    return "drop_retired"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--db", default=RESULT_DB)
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=120.0)
    cur = con.cursor()
    rows = cur.execute(
        'SELECT _path, METHOD FROM strategy_summaries WHERE "Hedge_Mode" IS NULL'
    ).fetchall()

    buckets: dict = {}
    for path, method in rows:
        buckets.setdefault(classify(path), []).append((path, method))

    print(f"舊口徑列共 {len(rows)}：")
    for k in sorted(buckets):
        methods = sorted({m for _, m in buckets[k]})
        print(f"  {k:16s} {len(buckets[k]):4d} 列   METHOD：{', '.join(methods)}")

    to_drop = [p for k, v in buckets.items() if k.startswith("drop_") for p, _ in v]
    print(f"\n本腳本將刪除 {len(to_drop)} 列"
          f"（保留 {len(buckets.get('keep_ez_dsz', []))} 列 EZ/DSZ 供補跑）")

    if args.dry_run:
        print("\n--dry-run：未修改。")
        con.close()
        return 0

    if not to_drop:
        print("沒有要刪的列。")
        con.close()
        return 0

    for i in range(0, len(to_drop), 200):
        chunk = to_drop[i:i + 200]
        q = ",".join("?" * len(chunk))
        cur.execute(f"DELETE FROM strategy_summaries WHERE _path IN ({q})", chunk)
        cur.execute(f"DELETE FROM trade_logs WHERE strategy_id IN ({q})", chunk)
        cur.execute(f"DELETE FROM strategy_pairs WHERE strategy_id IN ({q})", chunk)
        con.commit()
        print(f"  已刪 {min(i + 200, len(to_drop))}/{len(to_drop)}")

    left = cur.execute(
        'SELECT COUNT(*) FROM strategy_summaries WHERE "Hedge_Mode" IS NULL').fetchone()[0]
    total = cur.execute("SELECT COUNT(*) FROM strategy_summaries").fetchone()[0]
    con.close()
    print(f"\n完成。總列 {total}；仍為舊口徑 {left}"
          f"（應等於待補跑的 EZ/DSZ 列數）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
