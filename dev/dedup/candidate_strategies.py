# -*- coding: utf-8 -*-
"""
跨期去重 —— 網格條目
====================
判準與範圍見同目錄 PREREGISTRATION.md（寫於結果之前）。

預設**不會被執行**：`strategies/config.py` 只在環境變數 `DEDUP_EXTENSION=1`
時把 `strategies_raw` 換成本檔的條目。

    $env:DEDUP_EXTENSION="1"; $env:RESULT_DB_PATH="results/dedup.db"; python run_trading.py

條目**沿用主軸條目的 name / sub_dir / db_method**：形成期配對直接共用
（formation_strategy_id = name + "_MSR0"），結果以檔名 `_DD` 後綴與既有列並存。
對照組（無去重）即 result.db 既有的同格，不重跑。
"""
import copy

GROUPINGS = ("NOGRP", "GICS", "HDB", "AGG", "KM")
RANKINGS = ("SSD", "DTW", "SDP")


def build():
    import strategies.config as C

    by_name = {s["name"]: s for s in C.strategies_raw_all}
    out = []
    for g in GROUPINGS:
        for r in RANKINGS:
            s = copy.deepcopy(by_name[f"Grid {g}-{r}"])
            s["params"]["top_n_list"] = [3, 5, 10, 20]
            s["params"]["stop_loss_list"] = [0.0]
            s["params"]["dedup_across_periods"] = True
            out.append(s)
    return out
