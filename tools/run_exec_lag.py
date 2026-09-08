#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""§F 執行延遲消融：六臂 × 三條件的受控重跑。

設計與判準見 `dev/exec_lag/PREREGISTRATION.md`（**跑前定稿**）。

三個條件以 `SENSITIVITY_PARAM=exec_lag` 逐臂展開，取值編碼於一個維度：
    1E = 只延遲進場 / 1X = 只延遲出場 / 1B = 兩者

基準（L=0）不重跑——`result.db` 既有的基準格就是它，且 `verify_lag.py`
的 P1 已證 L=0 與接線前逐位元相同。

三個條件皆帶 `_LAG` 檔名後綴，與基準格並存不覆寫。

用法：
    python tools/run_exec_lag.py --dry-run
    python tools/run_exec_lag.py --execute
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

#: (config 的 name, 對應的 db_method)。選臂理由見 PREREGISTRATION.md §四。
ARMS = [
    ("Grid NOGRP-SDP",      "Grid (NOGRP-SDP)"),
    ("Grid NOGRP-SSD",      "Grid (NOGRP-SSD)"),
    ("Grid GICS-SSD",       "Grid (GICS-SSD)"),
    ("Grid NOGRP-DTW",      "Grid (NOGRP-DTW)"),
    ("Grid GICS-SSD-FW504", "Grid (GICS-SSD-FW504)"),
    ("Grid GGR",            "Grid (GGR)"),
]
CONDITIONS = ["1E", "1X", "1B"]


def _count(where: str) -> int:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM strategy_summaries WHERE {where}").fetchone()[0]
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    print(f"六臂 × 三條件 = {len(ARMS) * len(CONDITIONS)} 批")
    for name, dbm in ARMS:
        print(f"  {name:<24} {dbm}")
    print(f"條件：{CONDITIONS}")

    if args.dry_run:
        print("\n--dry-run：未執行。")
        return 0

    before = _count("_path LIKE '%_LAG%'")
    print(f"\n開跑前既有 _LAG 列：{before}\n")

    t0 = time.time()
    failed = []
    for i, (name, dbm) in enumerate(ARMS, 1):
        env = dict(os.environ)
        env.update({
            "SENSITIVITY_PARAM": "exec_lag",
            "SENSITIVITY_BASE": name,
            "SENSITIVITY_VALUES": ",".join(CONDITIONS),
            "FORCE_RERUN": "1",
            "WRITE_TRADE_CSV": "0",
            "PYTHONUTF8": "1",
        })
        print(f"[{i}/{len(ARMS)}] {name}  條件 {CONDITIONS}", flush=True)
        r = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", "run_trading.py"],
                           env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        n_now = _count("_path LIKE '%_LAG%'")
        if r.returncode != 0:
            failed.append(name)
            print(f"   X rc={r.returncode}")
            print("   " + "\n   ".join((r.stdout or "").strip().splitlines()[-10:]))
        else:
            print(f"   OK  累計 _LAG 列 {n_now}  ({time.time() - t0:.0f}s)", flush=True)

    after = _count("_path LIKE '%_LAG%'")
    print(f"\n完成 {len(ARMS) - len(failed)}/{len(ARMS)} 臂"
          f"{'；失敗：' + str(failed) if failed else ''}")
    print(f"新增 _LAG 列 {after - before}（總 {after}）")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
