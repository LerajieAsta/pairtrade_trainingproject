# -*- coding: utf-8 -*-
"""
動作空間消融的兩年形成期版本：v3（FQI）＋ 兩個基準（v4、Z-Score）
======================================================================
判準見同目錄 PREREGISTRATION.md（寫於結果之前）。

預設**不會被執行**：`strategies/config.py` 只在環境變數 `ABLATION_FW504=1` 時附加本檔的三條臂，
並把 `strategies_raw` 限縮成它們。必須同時設定期間與跳過開關：

    $env:ABLATION_FW504="1"; $env:SKIP_TRUNCATED_FORMATION="1"
    $env:BACKTEST_START="2008-07"; $env:BACKTEST_END="2018-12"
    python run_trading.py

與 252 版（`dev/action_space/candidate_strategies.py`）的唯一差異是形成期配對來源：
模板取附錄 F 的 `Grid GICS-SDP-FW504`／`Grid GICS-SDP-FW504 DRL`（formation_window=504），
v3 的超參數、避險方式、Top1/SL0 網格鎖定一律引用 252 版的常數，不在此另抄一份。
"""
import copy

from dev.action_space.candidate_strategies import (
    ARCHIVED_DRL_PARAMS, GRID_LOCK, HEDGE_MODE, VERSIONS, _V4_ONLY)

TMPL_DRL = "Grid GICS-SDP-FW504 DRL"
TMPL_ZS = "Grid GICS-SDP-FW504"


def build():
    from dev.fw504_all.candidate_strategies import build as build_fw504

    by_name = {s["name"]: s for s in build_fw504()}
    tmpl, zs0 = by_name[TMPL_DRL], by_name[TMPL_ZS]
    out = []

    # v3（只跑最接近 v4 的自由持倉臂；理由見 PREREGISTRATION.md §一）
    tag, module, _ = next(v for v in VERSIONS if v[0] == "v3")
    s = copy.deepcopy(tmpl)
    s["name"] = f"{tmpl['name']}-V3"
    s["trading_module"] = module
    s["sub_dir"] = f"{tmpl['sub_dir']}-V3"
    s["db_method"] = f"{tmpl['db_method'][:-1]}-V3)"
    s["trade_method"] = "DRLv3"
    for k in _V4_ONLY:
        s["params"].pop(k, None)
    s["params"].update(ARCHIVED_DRL_PARAMS)
    s["params"]["hedge_mode"] = HEDGE_MODE
    s["params"].update(GRID_LOCK)
    out.append(s)

    # 同口徑的 v4
    v4 = copy.deepcopy(tmpl)
    v4["name"] += "-DOLLAR"
    v4["sub_dir"] += "_DOLLAR"
    v4["db_method"] = f"{tmpl['db_method'][:-1]}-DOLLAR)"
    v4["params"]["hedge_mode"] = HEDGE_MODE
    v4["params"].update(GRID_LOCK)
    out.append(v4)

    # 同口徑的 Z-Score；明確借用原臂的形成期配對（改名後否則會找不到）
    zs = copy.deepcopy(zs0)
    zs["formation_strategy_id_base"] = zs0["name"]
    zs["name"] += " DOLLAR"
    zs["sub_dir"] += "_DOLLAR"
    zs["db_method"] = f"{zs0['db_method'][:-1]}-DOLLAR)"
    zs["params"]["hedge_mode"] = HEDGE_MODE
    zs["params"].update(GRID_LOCK)
    out.append(zs)
    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ".")
    for s in build():
        print(f"{s['db_method']:<36} {s['trade_method']:<8} hedge={s['params']['hedge_mode']:<7} "
              f"fw={s['params'].get('formation_window')} base={s.get('formation_strategy_id_base')} "
              f"{s['trading_module']}")
