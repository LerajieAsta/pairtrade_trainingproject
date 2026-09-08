#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""以新對沖口徑補跑 `EZ` / `DSZ` 參數掃描的 525 個配置。

背景見 `dev/trading_arch/REVIEW.md` §A。全網格重跑只涵蓋現行 config 展開得出的
配置；`entry_z` 與 `dynamic_stop_z` 兩個掃描是歷史上以 `SENSITIVITY_PARAM`
跑出來的，現行展開不含它們，故重跑後仍停在舊口徑。

## 為何不直接 `SENSITIVITY_ALL=1`

那個模式會連 21 個 formation 變體一起跑（需先跑 `run_formation`，數小時），
而且它的 `entry_z` 掃描只涵蓋 `SENSITIVITY_BASES` 的三支主力，與實際存在於
`result.db` 的八支對不起來。本腳本改為**逐臂重放實際存在的取值**，
使補跑產生的 path_key 與原本那 525 個逐一對應。

## 兩個掃描是各自獨立的，不是交叉

實測 `result.db`：`EZ` 掃描時 `DSZ=0`，`DSZ` 掃描時 `EZ=2.0`。
故分兩批跑，各自只設一個 `_list`。

用法：
    python tools/rerun_ez_dsz.py --dry-run     # 只列出要跑什麼
    python tools/rerun_ez_dsz.py --execute
"""
import argparse
import os
import re
import sqlite3
import subprocess
import sys

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULT_DB = "results/result.db"


def plan(db: str = RESULT_DB) -> list:
    """自 `result.db` 讀出仍為舊口徑的 EZ/DSZ 列，還原成 (策略名, 參數, 取值)。

    以資料庫的實況為準而非寫死清單 —— 補跑要覆蓋的就是這些列本身，
    任何寫死的清單都可能與實況分歧。
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            'SELECT _path, METHOD, "ENTRY Z", "DYN Z NUM" '
            'FROM strategy_summaries WHERE "Hedge_Mode" IS NULL').fetchall()
    finally:
        con.close()

    ez, dsz = {}, {}
    for path, method, e, d in rows:
        if not re.search(r"_(EZ\d+|DSZ\d+)", path.rsplit("/", 1)[-1]):
            continue
        # METHOD 形如 "Grid (AGG-SSD)"；config 的 name 是 "Grid AGG-SSD"
        name = method.replace("(", "").replace(")", "")
        if (d or 0) > 0:
            dsz.setdefault(name, set()).add(float(d))
        else:
            ez.setdefault(name, set()).add(float(e))

    jobs = []
    for name, vals in sorted(ez.items()):
        jobs.append(("entry_z", name, sorted(vals)))
    for name, vals in sorted(dsz.items()):
        jobs.append(("dynamic_stop_z", name, sorted(vals)))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    jobs = plan()
    if not jobs:
        print("沒有待補跑的 EZ/DSZ 列。")
        return 0

    n_cfg = sum(len(v) * 15 for _, _, v in jobs)
    print(f"待補跑 {len(jobs)} 批，約 {n_cfg} 個配置：\n")
    for param, name, vals in jobs:
        print(f"  {param:16s} {name:22s} {vals}")

    if args.dry_run:
        print("\n--dry-run：未執行。")
        return 0

    failed = []
    for i, (param, name, vals) in enumerate(jobs, 1):
        env = dict(os.environ)
        env.update({
            "SENSITIVITY_PARAM": param,
            "SENSITIVITY_BASE": name,
            "SENSITIVITY_VALUES": ",".join(str(v) for v in vals),
            "FORCE_RERUN": "1",
            "WRITE_TRADE_CSV": "0",
        })
        print(f"\n[{i}/{len(jobs)}] {param} = {vals}  on  {name}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", "run_trading.py"],
                           env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            failed.append((param, name))
            print(f"  ❌ 失敗（returncode={r.returncode}）")
            print("  " + "\n  ".join(r.stdout.strip().splitlines()[-8:]))

    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    left = con.execute(
        'SELECT COUNT(*) FROM strategy_summaries WHERE "Hedge_Mode" IS NULL').fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM strategy_summaries").fetchone()[0]
    con.close()

    print(f"\n完成 {len(jobs) - len(failed)}/{len(jobs)} 批"
          f"{'；失敗：' + str(failed) if failed else ''}")
    print(f"總列 {total}；仍為舊口徑 {left}（目標 0）")
    return 0 if (left == 0 and not failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
