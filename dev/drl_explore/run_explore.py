# -*- coding: utf-8 -*-
"""
探索性 DRL 變體的執行入口（未預先登記，純探索）
======================================================================
比照 `dev/action_space/candidate_strategies.py` 的接線手法：在
`strategies.config` 第一次被 import 之前設好 BACKTEST_START/END，
再把 `strategies_raw` 換成 `explore_strategies.build()` 的兩條探索臂，
最後呼叫 `run_trading.run_all_trading()`。

用法（於 repo 根目錄執行）：
    $env:BACKTEST_START="2009-07"; $env:BACKTEST_END="2018-12"
    python dev/drl_explore/run_explore.py

可選環境變數：
    EXPLORE_EPISODES=5   煙霧測試用，正式比較不要設
    EXPLORE_TAG=SMOKE    幫 name/sub_dir/db_method 加尾綴，避免與正式列同名
"""
import os
import sys

os.environ.setdefault("BACKTEST_START", "2009-07")
os.environ.setdefault("BACKTEST_END", "2018-12")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import strategies.config as C
from dev.drl_explore import explore_strategies

_explore = explore_strategies.build()
C.strategies_raw_all = C.strategies_raw_all + _explore
C.strategies_raw = _explore

print(f"[explore] 期間 {C.BACKTEST_START} ~ {C.BACKTEST_END}")
print(f"[explore] 將執行 {len(_explore)} 條探索臂：")
for _s in _explore:
    print(f"    {_s['db_method']:<40} {_s['trading_module']}")

import run_trading

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_trading.run_all_trading()
