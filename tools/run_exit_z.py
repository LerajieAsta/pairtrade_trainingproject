#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`exit_z` 出場門檻實掃：四條主力臂 × 四個取值。

設計與判準見 `dev/exit_z/PREREGISTRATION.md`（**跑前定稿**）。

`exit_z=0.0` 為現行基準格，不重跑——既有列就是它。
四個掃描值皆帶 `_XZ{n}` 後綴（n = exit_z × 100），與基準格並存不覆寫。

用法：
    python tools/run_exit_z.py --dry-run
    python tools/run_exit_z.py --execute
"""
import argparse
import os
import sqlite3
import subprocess
import sys
import time

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULT_DB = "results/result.db"

#: 四條主力臂。不含 GGR（動 exit_z 就不再是復現）、
#: 不含 FW504 / TW63（config.py 已列為否證，不是可用策略）。
ARMS = ["Grid NOGRP-SDP", "Grid NOGRP-SSD", "Grid GICS-SSD", "Grid NOGRP-DTW"]
VALUES = ["0.25", "0.5", "0.75", "1.0"]


def _count() -> int:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=120)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM strategy_summaries WHERE _path LIKE '%_XZ%'").fetchone()[0]
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    print(f"{len(ARMS)} 臂 × {len(VALUES)} 值 × 15 格 = "
          f"{len(ARMS) * len(VALUES) * 15} 個配置")
    for a in ARMS:
        print(f"  {a}")
    print(f"exit_z ∈ {VALUES}（0.0 = 既有基準格，不重跑）")

    if args.dry_run:
        print("\n--dry-run：未執行。")
        return 0

    before = _count()
    print(f"\n開跑前既有 _XZ 列：{before}\n")

    t0 = time.time()
    failed = []
    for i, name in enumerate(ARMS, 1):
        env = dict(os.environ)
        env.update({
            "SENSITIVITY_PARAM": "exit_z",
            "SENSITIVITY_BASE": name,
            "SENSITIVITY_VALUES": ",".join(VALUES),
            "FORCE_RERUN": "1",
            "WRITE_TRADE_CSV": "0",
            "PYTHONUTF8": "1",
        })
        print(f"[{i}/{len(ARMS)}] {name}  exit_z ∈ {VALUES}", flush=True)
        r = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", "run_trading.py"],
                           env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            failed.append(name)
            print(f"   X rc={r.returncode}")
            print("   " + "\n   ".join((r.stdout or "").strip().splitlines()[-10:]))
        else:
            print(f"   OK  累計 _XZ 列 {_count()}  ({time.time() - t0:.0f}s)", flush=True)

    after = _count()
    print(f"\n完成 {len(ARMS) - len(failed)}/{len(ARMS)} 臂"
          f"{'；失敗：' + str(failed) if failed else ''}")
    print(f"新增 _XZ 列 {after - before}（總 {after}；預期 240）")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
