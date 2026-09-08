#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""以修正後的 signal 對沖口徑重跑 DRL / RL 八支臂。

設計與判準見 `dev/drl_hedge/PREREGISTRATION.md`（**跑前定稿**）。
程式面的驗證見 `dev/drl_hedge/verify_hedge_modes.py`（P1–P4 全數通過）。

這是**修正**不是新變體：沿用原 path_key、原地覆寫，不加檔名後綴，
試驗宇宙不變（method N = 53、config N 不增）。

⚠ DRL / RL **不固定隨機種子**。單次重跑的前後差 = 修正效果 + 重訓雜訊，
  兩者無法分離。歸因需搭配 `tools/run_drl_variance` 量測修正後的雜訊分布，
  見預先註冊 §七。**本腳本只負責重跑，不負責歸因。**

用法：
    python tools/rerun_drl_hedge.py --dry-run
    python tools/rerun_drl_hedge.py --execute
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

#: (config 的 name, db_method)。八支臂即 2026-09-01 清點出的 120 列。
ARMS = [
    ("Grid AGG-SSD DRL",        "Grid (AGG-SSD-DRL)"),
    ("Grid GICS-SSD DRL",       "Grid (GICS-SSD-DRL)"),
    ("Grid GICS-SDP DRL",       "Grid (GICS-SDP-DRL)"),
    ("Grid HDB-SDP DRL",        "Grid (HDB-SDP-DRL)"),
    ("Grid KM-SSD DRL",         "Grid (KM-SSD-DRL)"),
    ("Grid AGG-SSD RLTHR E05",  "Grid (AGG-SSD-RLTHR-E05)"),
    ("Grid AGG-SSD RLTHR E10",  "Grid (AGG-SSD-RLTHR-E10)"),
    ("Grid AGG-SSD RLTHR E20D", "Grid (AGG-SSD-RLTHR-E20D)"),
]


def snapshot(tag: str) -> None:
    """把八支臂的現況摘要另存 CSV。

    重跑會原地覆寫，舊值只存在於這份快照裡——沒有它就無法做前後對照，
    而 DRL 的重訓雜訊使「憑印象比較」特別不可靠。
    """
    import pandas as pd
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=180)
    d = pd.read_sql(
        "SELECT * FROM strategy_summaries WHERE METHOD IN (%s)"
        % ",".join("?" * len(ARMS)), con, params=[m for _, m in ARMS])
    con.close()
    out = f"results/analysis/drl_hedge_{tag}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # 不覆寫既有快照：中止後重啟時 tag="pre" 拍到的是混合狀態，
    # 若原地覆寫，真正的修正前基準就永久消失了（2026-09-02 幾乎踩到）。
    if os.path.exists(out):
        out = f"results/analysis/drl_hedge_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"  （既有快照保留不動，改寫新檔）")
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  快照 {len(d)} 列 -> {out}")


def _modes() -> dict:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=180)
    try:
        rows = con.execute(
            'SELECT "Hedge_Mode", COUNT(*) FROM strategy_summaries '
            "WHERE METHOD IN (%s) GROUP BY 1" % ",".join("?" * len(ARMS)),
            [m for _, m in ARMS]).fetchall()
    finally:
        con.close()
    return {(r[0] or "NULL"): r[1] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    print(f"八支臂 × 15 格 = {len(ARMS) * 15} 個配置（原地覆寫，不加後綴）")
    for n, m in ARMS:
        print(f"  {n:<26} {m}")
    print(f"\n重跑前的 Hedge_Mode 分布：{_modes()}")
    print("  （修正前這些列標著 signal 卻以 dollar 執行——正是本次要改正的）")

    if args.dry_run:
        print("\n--dry-run：未執行。")
        return 0

    snapshot("pre")

    t0 = time.time()
    failed = []
    for i, (name, dbm) in enumerate(ARMS, 1):
        env = dict(os.environ)
        env.update({
            "SENSITIVITY_PARAM": "top_n",
            "SENSITIVITY_BASE": name,
            "SENSITIVITY_VALUES": "1,3,5,10,20",
            "FORCE_RERUN": "1",
            "WRITE_TRADE_CSV": "0",
            "PYTHONUTF8": "1",
        })
        print(f"\n[{i}/{len(ARMS)}] {name}", flush=True)
        r = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", "run_trading.py"],
                           env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            failed.append(name)
            print(f"   X rc={r.returncode}")
            print("   " + "\n   ".join((r.stdout or "").strip().splitlines()[-10:]))
        else:
            print(f"   OK  ({time.time() - t0:.0f}s)", flush=True)

    snapshot("post")
    print(f"\n完成 {len(ARMS) - len(failed)}/{len(ARMS)} 臂"
          f"{'；失敗：' + str(failed) if failed else ''}")
    print(f"重跑後的 Hedge_Mode 分布：{_modes()}")
    print("\n下一步（見預先註冊 §七、§九）：")
    print("  1. DRL_VARIANCE_TAG=drlhedge python -m tools.run_drl_variance --runs 5")
    print("     —— 沒有這一步，前後差無法與重訓雜訊分離，不可宣稱位移")
    print("  2. 重算命題 2 全鏈（proposition2_daily_hac 起 8 支）")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
