# -*- coding: utf-8 -*-
"""
同一交易期內的同時持倉上限 K —— 網格條目
==========================================
判準與範圍見同目錄 PREREGISTRATION.md（寫於結果之前）。

預設**不會被執行**：`strategies/config.py` 只在環境變數 `MAX_ACTIVE_EXTENSION=1`
時把 `strategies_raw` 換成本檔的條目。

    $env:MAX_ACTIVE_EXTENSION="1"; python run_trading.py

條目**沿用主軸條目的 name / sub_dir / db_method**：形成期配對直接共用
（formation_strategy_id = name + "_MSR0"），結果以檔名 `_MA{K}` 後綴與既有列並存。
展開時 K ≥ top_n 的組合即既有的無上限格（Top 10），`check_trading_completed` 會跳過。
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
            s["params"]["top_n_list"] = [10, 20]
            s["params"]["stop_loss_list"] = [0.0]
            s["params"]["max_active_list"] = [3, 5, 10]
            out.append(s)
    return out
