# -*- coding: utf-8 -*-
"""
延伸研究：論文核心 20 條策略的兩年形成期版本（formation_window = 504）
======================================================================
判準與範圍見同目錄 PLAN.md（寫於結果之前）。

預設**不會被執行**：`strategies/config.py` 只在環境變數 `FW504_EXTENSION=1`
時附加本檔的條目，並把 `strategies_raw` 限縮成它們（與 ACTION_SPACE_ABLATION
同一種掛法）。

    $env:FW504_EXTENSION="1"; python run_formation.py; python run_trading.py

每條只把 `formation_window` 改成 504，其餘參數逐字複製自主軸條目，
避免日後 config 調整時兩邊靜默脫節。GICS-SSD 沿用既有的
`Grid GICS-SSD-FW504`（參數相同），不另建。
"""
import copy

GROUPINGS = [("none", "NOGRP"), ("gics", "GICS"), ("hdbscan", "HDB"),
             ("agglomerative", "AGG"), ("kmeans", "KM")]
RANKINGS = [("ssd", "SSD"), ("dtw", "DTW"), ("ssd_dtw_pca", "SDP")]
DLTHR = [("GICS", "SSD"), ("GICS", "SDP"), ("HDB", "SDP"), ("AGG", "SSD"),
         ("KM", "SSD")]

FW = 504


def build():
    import strategies.config as C

    by_name = {s["name"]: s for s in C.strategies_raw_all}
    out = []

    for _cm, cs in GROUPINGS:
        for _rb, rs in RANKINGS:
            existing = by_name.get(f"Grid {cs}-{rs}-FW504")
            if existing is not None:
                out.append(existing)
                continue
            s = copy.deepcopy(by_name[f"Grid {cs}-{rs}"])
            s["name"] = f"Grid {cs}-{rs}-FW504"
            s["sub_dir"] = f"Grid_{cs}_{rs}_FW504"
            s["db_method"] = f"Grid ({cs}-{rs}-FW504)"
            s["params"]["formation_window"] = FW
            out.append(s)

    for cs, rs in DLTHR:
        s = copy.deepcopy(by_name[f"Grid {cs}-{rs} DRL"])
        s["name"] = f"Grid {cs}-{rs}-FW504 DRL"
        s["formation_strategy_id_base"] = f"Grid {cs}-{rs}-FW504"
        s["sub_dir"] = f"Grid_{cs}_{rs}_FW504_DRL"
        s["db_method"] = f"Grid ({cs}-{rs}-FW504-DRL)"
        s["params"]["formation_window"] = FW
        out.append(s)

    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ".")
    for s in build():
        print(f"{s['db_method']:<30} fw={s['params'].get('formation_window')} "
              f"{s['trading_module'].split('.')[-1]:<22} "
              f"borrow={s.get('formation_strategy_id_base', '-')}")
