#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""依 `tools/refresh_after_rerun.py` 的 PIPELINE 順序重算，逐支記錄成敗。

為何跑全部 17 支而不是 `dev/drl_hedge/PREREGISTRATION.md` §九 的 8 支：
`refresh_after_rerun.py` 第 61-62 行記著前例——`regime_cost_ew` 曾經
「一直不在本清單，重跑後會靜默留著舊數字」。逐日快取此刻已整份清空
（804 欄 → 0 欄），只跑 8 支會讓其餘分析停在重跑前的數字上，
而那種錯誤不會報錯。順序沿用 PIPELINE（命題 1 先，且
`prop1_han_chain` 必須在 `proposition2_daily_hac` 之後）。

任一支失敗即停——後面的分析多半 import 前面的產出，接著跑只會擴大污染。

SKIP：輸入資料已不存在、非本次重跑所致者。跳過會印出理由，不會靜默略過。
"""
import argparse
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

from tools.refresh_after_rerun import PIPELINE  # noqa: E402

LOG_DIR = "results/logs/pipeline_20260905"

#: 模組 -> 跳過的理由。**只放「輸入資料不存在」這一類**，
#: 不可拿來繞過真正的錯誤。
SKIP = {
    "analysis.granularity_sweep":
        "需要 agg_threshold_percentile= 那批掃描 METHOD，DB 內一支都沒有、"
        "results/analysis/ 也無任何 granularity 輸出。該批資料在 2026-08-13 "
        "result.db 整顆重建時消失，之後未重生成。要復原須先跑該模組 "
        "docstring 的 SENSITIVITY_* 掃描（run_formation.py + run_trading.py）。"
        "與 signal 對沖修正無關。",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="自 PIPELINE 第 N 支起跑（1-based）")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = time.time()
    skipped = []
    for i, (name, mod) in enumerate(PIPELINE, 1):
        if i < args.start:
            continue
        if mod in SKIP:
            print(f"[{i:>2}/{len(PIPELINE)}] {name:<22} {mod}", flush=True)
            print(f"     -- 跳過：{SKIP[mod]}", flush=True)
            skipped.append(mod)
            continue
        log = f"{LOG_DIR}/{i:02d}_{mod.split('.')[-1]}.log"
        print(f"[{i:>2}/{len(PIPELINE)}] {name:<22} {mod}", flush=True)
        t = time.time()
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(
                [sys.executable, "-X", "utf8", "-W", "ignore", "-m", mod],
                env={**os.environ, "PYTHONUTF8": "1"},
                stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"     X 失敗 exit={r.returncode}（{(time.time()-t)/60:.1f} 分），"
                  f"詳見 {log}", flush=True)
            with open(log, encoding="utf-8", errors="replace") as f:
                print("     " + "\n     ".join(f.read().strip().splitlines()[-20:]))
            print(f"\n中止於第 {i} 支。已完成 {i-1}/{len(PIPELINE)}。")
            return 1
        print(f"     完成（{(time.time()-t)/60:.1f} 分，累計 {(time.time()-t0)/60:.1f} 分）",
              flush=True)
    print(f"\n完成 {len(PIPELINE) - len(skipped)}/{len(PIPELINE)} 支，"
          f"共 {(time.time()-t0)/60:.1f} 分。")
    if skipped:
        print(f"跳過（輸入資料不存在，非本次重跑所致）：{skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
