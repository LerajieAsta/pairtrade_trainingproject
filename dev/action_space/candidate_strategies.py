# -*- coding: utf-8 -*-
"""
動作空間消融的候選策略宣告（**未接線，不會被執行**）
======================================================================
用途：若採用改題方向 A（動作空間設計是關鍵約束，見
`thesis/draft/RESTRUCTURE_PLAN.md`），需要把已封存的三代自由持倉 DRL
掛回**現行的配對來源**，才能構成單變因消融：

    Z-Score（固定門檻）  ← 基準
    DRL v1  LSTM-DQN     ← 逐日自由持倉
    DRL v2  修復版         ← 逐日自由持倉（修掉 v1 的計算/統計缺陷）
    DRL v3  FQI           ← 逐日自由持倉（批次擬合 Q）
    DRL v4  門檻選擇       ← 每期單次決策（現行 DL-THR）

**預設不會被執行。** `strategies/config.py` 只在環境變數
`ACTION_SPACE_ABLATION=1` 時才 import 本檔，並把 `build()` 的輸出
附加到 `strategies_raw_all`、同時把 `strategies_raw` 限縮成這五條。
不設該變數時 config 完全不碰本檔——刻意設計成這樣，
避免不小心觸發一次數天的回測。

    $env:ACTION_SPACE_ABLATION="1"
    $env:BACKTEST_START="2009-07"; $env:BACKTEST_END="2018-12"
    python run_trading.py

⚠ 視窗左緣**必須多墊 1.5 年**。`prepare_backtest_data` 只把價格索引向前
回填 252 日，而跳過判定只看**交易期**日期；形成期起點早於索引起點的期
不會被跳過，而是丟例外——**且只有 v1 會丟**（實測 v2/v3/v4/Z-Score 皆正常）。
不墊就會讓 v1 少跑 18 期，五條臂不在同一組期上。
2009-07~2018-12 的共同期為 108，且在 split_half 中點兩側各 54 期。
判準與核對清單見 `thesis/draft/PREREG_action_space.md`。

--------------------------------------------------------------------
可行性（2026-09-09 實測，見 RESTRUCTURE_PLAN.md「四之四」）
--------------------------------------------------------------------
* 介面相容：三支封存模組的 `Trading.__init__` 前六個位置參數與 v4 逐字相同，
  `_simulate_pair` 簽章四支一致且都有 `**kwargs`。三支皆可乾淨 import。
* **成本**：v1 的 `agent_key` 含 ticker，等於**逐配對逐期各訓練一個 agent**
  （log 印 "for period" 是誤導的命名）。實測每 episode 37.3 秒，
  150 episodes → **每對每期 92 分鐘**。全期 295 期單格 = 18.8 天。
* 平行化幫不上：`run_trading` 以**策略設定**為單位平行、CPU 上限 4 worker，
  單一格內的 295 期是循序的。原專案用 H200 GPU，已於 2026-07-06 退租。

→ 只能跑**縮小版**。建議以 `BACKTEST_START`/`BACKTEST_END` 取一段連續子期間
  （config 註明子期間結果與全期並存於 result.db），四條臂跑同一段。

--------------------------------------------------------------------
兩項必須先決定的事
--------------------------------------------------------------------
1. **對沖口徑**。v1–v3 未宣告 `SUPPORTS_HEDGE_MODE`，依 `run_trading` 的
   守衛會被誠實記為 `"dollar"`；而現行 result.db 的 1,620 列全是 `"signal"`。
   直接比較會踩到附錄 B.5.1 剛修好的那個 confound。
   處置：本檔為 v4 與 Z-Score 另建 `dollar` 口徑的對照臂（見 `HEDGE_MODE`），
   讓五條臂在**同一口徑**下比較。

2. **子期間的選擇**。早期窗口（2001–2007）成本最低，但
   `analysis/split_half.py` 顯示前半期幾乎所有策略都好看、後半期轉負——
   在前半期做消融，其結論未必外推。選窗須揭露此點。
"""
import copy
import os

#: 消融的基底：借用哪一條現行策略的形成期配對（零重跑 formation）。
#:
#: 選 GICS-SDP 的理由（2026-09-09 改；原選 GICS-SSD 的理由「效果量最大
#: +0.798pp」是**選擇偏誤**——那個數字是從 15 格裡挑出來的最大值）：
#:
#: 1. SDP := z(SSD) + z(DTW)（§3.2.4），DTW 佔一半權重，且對沖比率
#:    與 DTW 臂同樣採雙向 OLS 取 ADF p 較低者；SSD 臂用的是另一套
#:    （無截距、方向固定）。以 SDP 為底，是**延續**而非取代 DTW 路線。
#: 2. 在 GICS 這一格，SDP 的 15 格等權 Sharpe 是三個配對底中最高的
#:    （SDP 0.076 > DTW 0.068 > SSD 0.043），不是為了方便而選次優。
#: 3. GICS-SDP 已有現成的 v4 臂，且在 §4.2.1 的逐日 HAC + BH 家族中
#:    BH 校正 p = 0.0278（通過）；GICS-SSD 為 0.0504（不過）。
#:
#: 為何不直接用 GICS-DTW：它**沒有 v4 臂**。新建雖只要約 2.6 小時，
#: 但會讓 §4.2.1 從五個配對底變六個，BH 家族重排。實測敏感度：新底
#: 自身 p ≳ 0.06 時，現有的 GICS-SSD 與 K-means 會被推過 0.05，顯著數
#: 由 3/5 掉到 2/6。事前無法得知新底的 p 落在哪邊，故不動這個家族。
BASE_DB_METHOD = "Grid (GICS-SDP-DRL)"

#: 同一配對底的 Z-Score（固定門檻）臂，作為消融的基準組。
BASE_ZSCORE_DB_METHOD = "Grid (GICS-SDP)"

#: 五條臂共用的對沖口徑。設為 "dollar" 是為了與 v1–v3 對齊——
#: 它們沒有實作 signal 口徑，強行宣告只會讓 `Hedge_Mode` 再次說謊。
HEDGE_MODE = "dollar"

#: 封存版 DRL 的超參數，取自 `archive/config_archived_strategies.py`
#: 舊 #11–#13 的宣告。**不要為了省時間改小**——訓練預算是「過度交易」的
#: 替代解釋，改了就無法排除「是不是訓練不足才亂交易」。
ARCHIVED_DRL_PARAMS = {
    "drl_episodes":    150,
    "drl_hidden_size": 256,
    "drl_num_layers":  2,
    "drl_batch_size":  512,
}

#: **只給煙霧測試用的逃生門**。`ABLATION_EPISODES=5` 可把訓練預算調小，
#: 用來在啟動一次數天的正式跑之前，先驗證整條管線會落庫。
#: ⚠ 正式跑**不可**設它——訓練預算是「過度交易」最明顯的替代解釋，
#: 改小就無法排除「是不是訓練不足才亂交易」（見 RESTRUCTURE_PLAN 四之四）。
_EP_OVERRIDE = os.environ.get("ABLATION_EPISODES", "").strip()
if _EP_OVERRIDE.isdigit() and int(_EP_OVERRIDE) > 0:
    ARCHIVED_DRL_PARAMS["drl_episodes"] = int(_EP_OVERRIDE)

#: **煙霧測試／成本探針的命名隔離**。`run_trading` 的完成判定「純以
#: result.db 為準」：`strategy_summaries` 有該 config 的列就視為完成並跳過。
#: 若探針用正式 db_method 落庫，正式跑會被**靜默跳過**，而且跳過的是
#: episodes=2 的垃圾列。設 `ABLATION_TAG=SMOKE` 會把五條臂的
#: name/sub_dir/db_method 全部加尾綴，與正式列永不同名。
#: ⚠ 正式跑不可設它。
_TAG = os.environ.get("ABLATION_TAG", "").strip().upper()
_TAG = f"-{_TAG}" if _TAG else ""

#: v4 專屬、v1–v3 不認得的參數，掛 v1–v3 時要拿掉。
_V4_ONLY = ("thr_train_epochs", "thr_min_train_samples", "thr_menu_version")

#: **把網格鎖成單格**。主軸每條策略會依 top_n_list × stop_loss_list 展開成
#: 15 格；若不鎖，v1 的成本會變成 (1+3+5+10+20) × 3 = 117 倍。
#: 鎖 Top1/SL0 的理由：SL0 無停損，強平事件不被截斷，是觀察「過度交易」
#: 最乾淨的一格（與 analysis/forced_close_followup.py 同一個選擇）。
GRID_LOCK = {"top_n_list": [1], "stop_loss_list": [0.0]}

VERSIONS = [
    ("v1", "archive.trading.drl_lstm_trading",    "LSTM-DQN（原始版）"),
    ("v2", "archive.trading.drl_lstm_v2_trading", "LSTM-DQN（修復版）"),
    ("v3", "archive.trading.drl_fqi_trading",     "FQI（批次擬合 Q）"),
]


def build():
    """
    回傳消融所需的策略宣告清單。

    以現行 `Grid (GICS-SSD-DRL)` 為模板逐條衍生，而非硬抄 51 個 params——
    形成期參數若日後在 config 改動，本檔自動跟上，不會靜默脫節。
    """
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

    # 三代自由持倉 DRL
    for tag, module, _desc in VERSIONS:
        out.append(_clone(f"-{tag.upper()}", module, f"DRL{tag}", dict(ARCHIVED_DRL_PARAMS)))

    # 同口徑的 v4 與 Z-Score 對照臂（不加，五條臂就不在同一個對沖口徑上）
    v4 = copy.deepcopy(tmpl)
    v4["name"] += f"-DOLLAR{_TAG}"
    v4["sub_dir"] += f"_DOLLAR{_TAG}"
    v4["db_method"] = f"{tmpl['db_method'][:-1]}-DOLLAR{_TAG})"
    v4["params"]["hedge_mode"] = HEDGE_MODE
    v4["params"].update(GRID_LOCK)
    out.append(v4)

    zs = next(s for s in C.strategies_raw_all
              if s["db_method"] == BASE_ZSCORE_DB_METHOD)
    zs = copy.deepcopy(zs)
    # Z-Score 臂原本是形成期的**擁有者**（無 formation_strategy_id_base），
    # 改名後會去找不存在的 "Grid GICS-SDP DOLLAR_MSR0" 而整條失敗。
    # 明確宣告借用原臂的配對——與其他四條臂共用同一批配對正是消融的前提。
    zs["formation_strategy_id_base"] = zs["name"]
    zs["name"] += f" DOLLAR{_TAG}"
    zs["sub_dir"] += f"_DOLLAR{_TAG}"
    zs["db_method"] = f"{BASE_ZSCORE_DB_METHOD[:-1]}-DOLLAR{_TAG})"
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
        print(f"{s['db_method']:<34} {s['trade_method']:<8} "
              f"hedge={s['params']['hedge_mode']:<7} {s['trading_module']}")
    print("\n本檔未接線；要執行須手動併入 strategies_raw_all，"
          "並以 BACKTEST_START/BACKTEST_END 限制子期間。")
