#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量測 signal 對沖口徑修正後的重訓雜訊（`dev/drl_hedge/PREREGISTRATION.md` §七）。

為何非做不可
------------
`tools/rerun_drl_hedge.py` 量到的前後差（ΔSharpe 中位 +0.0838、109/120 為正）
是**修正效果 + 重訓雜訊**的合體。DRL / RL 皆不固定隨機種子，
單次重跑的差無法歸因。本腳本量的是修正**之後**的雜訊分布——
若 +0.0838 落在雜訊的常見範圍內，就不可宣稱位移。

為何要跑兩次
------------
`run_drl_variance.drl_targets` 依 `trading_module` 選策略，一次只涵蓋一個模組：

    drl_threshold_trading   5 臂（AGG-SSD / GICS-SSD / GICS-SDP / HDB-SDP / KM-SSD）
    rl_threshold_trading    3 臂（RLTHR E05 / E10 / E20D）

兩者依序跑，不並行 —— 兩個引擎同時寫 `result.db` 會互相拖慢，
且各自的 `clear_summaries` 會在對方跑到一半時刪列。

⚠ 跑完後 `result.db` 裡是**最後一輪**的抽樣，不是 `drl_hedge_post.csv` 那一輪。
   這沒有問題（本來就只是分布中的一個抽樣），但下游分析引用的是最後一輪。
"""
import os
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

RUNS = 5
MODULES = [("drl_threshold_trading", "drlhedge"),
           ("rl_threshold_trading", "rlhedge")]


def main() -> int:
    t0 = time.time()
    failed = []
    for module, tag in MODULES:
        print(f"\n{'='*70}\n{module}  (DRL_VARIANCE_TAG={tag}, --runs {RUNS})\n{'='*70}",
              flush=True)
        env = dict(os.environ)
        env.update({"DRL_VARIANCE_TAG": tag, "PYTHONUTF8": "1",
                    "WRITE_TRADE_CSV": "0"})
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "-W", "ignore",
             "-m", "tools.run_drl_variance", "--runs", str(RUNS),
             "--module", module],
            env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        tail = "\n  ".join((r.stdout or "").strip().splitlines()[-25:])
        print("  " + tail, flush=True)
        if r.returncode != 0:
            failed.append(module)
            print(f"  X rc={r.returncode}", flush=True)
            if r.stderr:
                print("  STDERR: " + "\n  ".join(r.stderr.strip().splitlines()[-10:]))
        print(f"  ({time.time() - t0:.0f}s 累計)", flush=True)

    print(f"\n完成 {len(MODULES) - len(failed)}/{len(MODULES)} 個模組"
          f"{'；失敗：' + str(failed) if failed else ''}")
    print("\n下一步：以 results/analysis/drl_variance_runs_{drlhedge,rlhedge}.csv")
    print("        的跨輪分布，判斷 +0.0838 是否超出重訓雜訊；")
    print("        然後重算命題 2 全鏈（proposition2_daily_hac 起 8 支）。")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
