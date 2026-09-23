# -*- coding: utf-8 -*-
"""
逐日自由持倉 DRL 的探索性變體（**未預先登記，不是論文正式結果**）
======================================================================
不動 `dev/action_space/candidate_strategies.py`（那是附錄 D 已預先登記、
已執行的五臂消融，其輸出被引用於正文——不應被事後修改）。本檔是完全
獨立的探索腳本，借用同一套接線手法（同一個 `run_trading.py`、同一批
GICS-SDP 形成期配對、同一對沖口徑、同一訓練預算），只新增兩條探索臂：

    v3m   FQI + MLP 編碼器（去掉 LSTM 的遞迴，見 drl_fqi_mlp_trading.py）
    v3s   FQI + 單網路 max 目標（去掉 Double-Q，見 drl_fqi_singleq_trading.py）
    v3ms  FQI + MLP 編碼器 + 單網路 max（2x2 設計的第四格，見 drl_fqi_mlp_singleq_trading.py）
    vi    離散表格式價值迭代（無函數逼近、無梯度，見 drl_tabular_vi_trading.py）

比較基準沿用已跑過的官方臂（result.db 既有列，不重跑）：
    Grid (GICS-SDP-DRL-Z-SCORE-DOLLAR)   Z-Score 基準
    Grid (GICS-SDP-DRL-V4-DOLLAR)         DL-THR 基準
    Grid (GICS-SDP-DRL-V3)                FQI／LSTM／Double-Q（正式消融臂）

用法（比照 candidate_strategies.py 的煙霧測試逃生門）：
    $env:BACKTEST_START="2009-07"; $env:BACKTEST_END="2018-12"
    python dev/drl_explore/run_explore.py
"""
import copy
import os

BASE_DB_METHOD = "Grid (GICS-SDP-DRL)"

HEDGE_MODE = "dollar"

ARCHIVED_DRL_PARAMS = {
    "drl_episodes":    150,
    "drl_hidden_size": 256,
    "drl_num_layers":  2,
    "drl_batch_size":  512,
}

_EP_OVERRIDE = os.environ.get("EXPLORE_EPISODES", "").strip()
if _EP_OVERRIDE.isdigit() and int(_EP_OVERRIDE) > 0:
    ARCHIVED_DRL_PARAMS["drl_episodes"] = int(_EP_OVERRIDE)

_TAG = os.environ.get("EXPLORE_TAG", "").strip().upper()
_TAG = f"-{_TAG}" if _TAG else ""

_V4_ONLY = ("thr_train_epochs", "thr_min_train_samples", "thr_menu_version")

GRID_LOCK = {"top_n_list": [1], "stop_loss_list": [0.0]}

VERSIONS = [
    ("v3m",  "dev.drl_explore.drl_fqi_mlp_trading",         "FQI + MLP（去遞迴，探索）"),
    ("v3s",  "dev.drl_explore.drl_fqi_singleq_trading",     "FQI + 單網路 max（去 Double-Q，探索）"),
    ("v3ms", "dev.drl_explore.drl_fqi_mlp_singleq_trading", "FQI + MLP + 單網路 max（2x2 第四格，探索）"),
    ("vi",   "dev.drl_explore.drl_tabular_vi_trading",      "離散表格式價值迭代（無函數逼近，探索）"),
]


def build():
    import strategies.config as C

    tmpl = next(s for s in C.strategies_raw_all if s["db_method"] == BASE_DB_METHOD)
    out = []

    def _clone(suffix: str, module: str, trade_method: str, params_patch: dict):
        s = copy.deepcopy(tmpl)
        s["name"] = f"{tmpl['name']}{suffix}{_TAG}"
        s["trading_module"] = module
        s["sub_dir"] = f"{tmpl['sub_dir']}{suffix}{_TAG}"
        s["db_method"] = f"{tmpl['db_method'][:-1]}{suffix}{_TAG})"
        s["trade_method"] = trade_method
        for k in _V4_ONLY:
            s["params"].pop(k, None)
        s["params"].update(params_patch)
        s["params"]["hedge_mode"] = HEDGE_MODE
        s["params"].update(GRID_LOCK)
        return s

    for tag, module, _desc in VERSIONS:
        out.append(_clone(f"-{tag.upper()}-EXPLORE", module, f"DRL{tag}", dict(ARCHIVED_DRL_PARAMS)))

    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ".")
    for s in build():
        print(f"{s['db_method']:<40} {s['trade_method']:<10} "
              f"hedge={s['params']['hedge_mode']:<7} {s['trading_module']}")
    print("\n本檔未接線；由 run_explore.py 呼叫。")
