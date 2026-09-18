# -*- coding: utf-8 -*-
"""
動作空間消融：H1 換手率、H2 Sharpe、H3 成本歸因
======================================================================
判準寫在 `thesis/draft/PREREG_action_space.md`，**早於資料**（commit 833c4de）。
本模組**只執行**那份預先登記，不新增、不放寬任何判準。

五條臂（唯一變因＝交易端的動作空間，配對來源與對沖口徑完全相同）：

    Z-Score   固定門檻（基準）
    v4        每期單次門檻選擇（基準）
    v1/v2/v3  逐日自由持倉（LSTM-DQN／修復版／FQI）

--------------------------------------------------------------------
硬閘門：`SUCCESS` 不足以證明跑成功
--------------------------------------------------------------------
本專案已經踩過兩次「回報成功但實際沒算」：
  · 探針中 v1 對 27 期「執行」7.4 秒、零筆有效模擬，仍回報 SUCCESS；
  · 槽位洩漏使續傳後第一個計算期零配對，**不印任何錯誤**（1500f68 已修）。
所以七項核對任一不過，本模組**拒絕輸出任何 p 值或效果量**並以非零碼結束。

預先登記第九節原本的第 1、2、4 項是讀 per-arm log 的，但 log 是**單次啟動**
尺度：本次跑跨四次啟動，續傳的期印的是「已完成，跳過（checkpoint）」而非
`Skipping period`，實測 v1/v2 為 90／70，其餘三臂為 175／0——照原口徑第 2 項
必然誤判。故三項改為以 `result.db` 為準的等價檢查（2026-09-17，記於偏離紀錄；
只更嚴、與結果數值無關，且在任何結果產出之前）：

    1  每期的交易日列數完整（截斷或半途失敗的期會少列）
    2  五臂的「視窗內應算期集合」相同（由 formation_pairs ＋ 價格日曆推出）
    4  以**逐期耗時快照**核對 v1/v2 是否真的在訓練（見下）

> ⚠ 第 4 項的證據會消失：`run_trading` 在整條臂跑完後會清除 `.ckpt` 目錄。
> 故耗時證據必須在**跑的過程中**以 `--snapshot` 落檔，事後無法重建。

--------------------------------------------------------------------
估計式
--------------------------------------------------------------------
* **主檢定**：逐期配對差分 `d_k = entries_i[k] − entries_base[k]`，
  循環區塊拔靴（L=6 期＝一個完整重疊週期、B=10,000）。
  **不可用 `block_bootstrap.bootstrap_test`**——它把輸入當日損益並乘 252
  換算年化 %，對「每期進場次數」是錯的單位。此處直接用底層的
  `circular_block_bootstrap_means`。
* **效果量**：總進場數比值 `Σv_i / Σbase`。
  不用逐期比值：實測 v4 與 Z-Score 各有 4/119 期零進場，逐期比值會除以零；
  剔除那些期又恰好剔掉「基準不交易而自由持倉臂在交易」的期，系統性低估效果。
  總數比值也正是 docstring 那個「6 倍」的同口徑。
* **達標**：一條臂須**同時**贏過 v4 與 Z-Score（H1 原文「高於 v4 與 Z-Score」）。

進場的定義取 `Status IN (ENTER_LONG_A, ENTER_SHORT_A, REVERSE_LONG_A,
REVERSE_SHORT_A)`——**反向也是一次進場**。實測可逐位重現 `strategy_summaries`
的 `Entries`：Z-Score 240、v4 238、v3 534（=173 ENTER + 361 REVERSE）。
`REVERSE_*` 只由封存的三支自由持倉模組產生，現行 zscore／drl_threshold
兩支完全不含該字串，故主軸 1,620 列的 `Entries` 不含反向。

--------------------------------------------------------------------
H2 / H3
--------------------------------------------------------------------
* **H2**：各臂落庫的 `Sharpe_Raw` 是用**各自的期集**算的（其餘四臂 119 期、
  v1 108 期），不同基。故以五臂共同期重算為主口徑，落庫值併列為穩健性。
* **H3**：`trade_logs` 只存**淨**損益，無費用欄也無股數，無法逐筆還原。
  改用 `metrics.py` 已審過的解析成本模型：

      零成本損益 = 淨損益 + friction × 名目額
      名目額     = Σ(每筆進場的 capital_per_pair) × 2   （進場費＋出場費）
      capital_per_pair = 當時權益 / (top_n × 並行期數)

  共用的 `metrics.traded_notional` 只數 `Status LIKE 'ENTER%'`，會漏掉
  v3 的 361 筆反向（占其 534 筆進場的 68%），且只給全期值；交集分析需要
  逐期值。故此處自寫逐期版，事件集含 `REVERSE_*`。
  （共用函式的事件集另行修正；主軸零反向，對既有數字是可驗證的 no-op。）

--------------------------------------------------------------------
執行
--------------------------------------------------------------------
    python -m analysis.ablation_action_space              # 正式（過閘門才輸出）
    python -m analysis.ablation_action_space --snapshot   # 跑的過程中落存耗時證據
    python -m analysis.ablation_action_space --dry-run    # 已完成臂乾跑，不寫 CSV
    python -m analysis.ablation_action_space --selftest   # 合成資料自測估計式
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from analysis.block_bootstrap import bh_adjust, circular_block_bootstrap_means  # noqa: E402
from strategies import returns as strategy_returns  # noqa: E402
from strategies.metrics import CURRENT_FEE_SIDE, concurrent_of, metrics_from_returns  # noqa: E402

RESULT_DB = "results/result.db"
FORMATION_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
OUT_DIR = "results/analysis"
SNAPSHOT = f"{OUT_DIR}/ablation_period_durations.csv"
CKPT_ROOT = "results/tiingo/.ckpt"

#: 預先登記 4.2 凍結的視窗與配對底
BACKTEST_START, BACKTEST_END = "2009-07", "2018-12"
FORMATION_SID = "Grid GICS-SDP_MSR0"
FORMATION_WINDOW = 252
TOP_N = 1

#: 預先登記 7：區塊長度＝一個完整重疊週期（126/21 = 6 期）
BLOCK_L_PERIODS = 6
N_BOOT = 10000
SEED = 20260917

#: 預先登記 8 的門檻
RATIO_THRESHOLD = 2.0
ALPHA = 0.05
#: 第 4 項核對的門檻。逐期耗時的兩個數字都對，但取樣母體不同：
#:   · 69.7／71.0 分 = **全歷程**中位數（含 2026-09-10 三臂並行、約 56 分/期 的那段）
#:   · 81 分         = 2026-09-15 重啟後、只有 v1/v2 兩臂並行的子樣本
#: 門檻取 60 分是要擋掉「幾秒就 SUCCESS」那種假成功，不是要卡住正常波動。
MIN_MEDIAN_MINUTES = 60.0

ARMS = {
    "Z-Score": "Grid (GICS-SDP-DOLLAR)",
    "v4":      "Grid (GICS-SDP-DRL-DOLLAR)",
    "v1":      "Grid (GICS-SDP-DRL-V1)",
    "v2":      "Grid (GICS-SDP-DRL-V2)",
    "v3":      "Grid (GICS-SDP-DRL-V3)",
}
FREE_ARMS = ("v1", "v2", "v3")      # 逐日自由持倉
BASE_ARMS = ("v4", "Z-Score")       # 兩個基準
CKPT_ARMS = {"v1": "Grid_GICS-SDP_DRL-V1_Top1_SL0_MSR0",
             "v2": "Grid_GICS-SDP_DRL-V2_Top1_SL0_MSR0"}

ENTRY_STATUSES = ("ENTER_LONG_A", "ENTER_SHORT_A", "REVERSE_LONG_A", "REVERSE_SHORT_A")


class GateFailure(Exception):
    """七項核對未通過。攜帶逐項結果，呼叫端負責輸出並以非零碼結束。"""


# ── 資料存取 ────────────────────────────────────────────────────────
def _con(db=RESULT_DB):
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def arm_paths(result_db=RESULT_DB) -> dict:
    """{臂代號: strategy_id}。未落庫的臂不會出現在回傳值裡。"""
    out = {}
    with _con(result_db) as c:
        for k, method in ARMS.items():
            r = c.execute("SELECT _path FROM strategy_summaries WHERE METHOD = ?",
                          (method,)).fetchone()
            if r:
                out[k] = r[0]
    return out


_CAL_MEMO = None


def price_calendar() -> pd.DatetimeIndex:
    """
    交易日曆。**不走 `strategies.returns.price_calendar`**——它讀 parquet 快取，
    而本機 pyarrow 的 DLL 被應用程式控制原則封鎖（`DLL load failed while
    importing _fs`）。直接查價格庫，無額外相依且與引擎同源。
    """
    global _CAL_MEMO
    if _CAL_MEMO is None:
        import strategies.config as C
        with sqlite3.connect(f"file:{C.DB_PATH}?mode=ro", uri=True) as c:
            d = pd.read_sql(f"SELECT DISTINCT Date FROM {C.TABLE_NAME} ORDER BY Date", c)
        _CAL_MEMO = pd.DatetimeIndex(pd.to_datetime(d.Date))
    return _CAL_MEMO


def expected_periods() -> list:
    """
    視窗內**應該**被計算的期（回傳交易期起點，與 trade_logs.Period_Start 同鍵）。

    重建 `prepare_backtest_data` 的行為而不載入價格矩陣：價格日曆往前回填
    `FORMATION_WINDOW` 個交易日，再取交易期起訖都落在該索引內的期。
    """
    cal = price_calendar()
    lo = cal.searchsorted(pd.Timestamp(f"{BACKTEST_START}-01"))
    start = cal[max(0, lo - FORMATION_WINDOW)]
    end = cal[cal.searchsorted(pd.Timestamp(f"{BACKTEST_END}-31"), side="right") - 1]
    with _con(FORMATION_DB) as c:
        per = pd.read_sql(
            "SELECT DISTINCT Trade_Start, Trade_End FROM formation_pairs "
            "WHERE strategy_id = ? ORDER BY Trade_Start", c, params=(FORMATION_SID,))
    ts, te = pd.to_datetime(per.Trade_Start), pd.to_datetime(per.Trade_End)
    keep = (ts >= start) & (te <= end) & ts.isin(cal) & te.isin(cal)
    return sorted(per.loc[keep, "Trade_Start"])


def left_edge_periods(expected: list) -> set:
    """
    **左緣期**＝形成期起點早於價格索引起點者。

    只有 v1 需要形成期的價格（`_train_shared_agent` 取 formation_start~end 的
    切片），故只有它在這些期會失敗；其餘四臂照常算。這個結構性不對稱是預先
    登記 4.1 在**任何資料存在之前**就揭露的，處置也寫定為「分析取五臂共同期
    交集」。判定只用價格日曆與形成期日期，不涉及任何結果。
    """
    cal = price_calendar()
    lo = cal.searchsorted(pd.Timestamp(f"{BACKTEST_START}-01"))
    idx_start = cal[max(0, lo - FORMATION_WINDOW)]
    with _con(FORMATION_DB) as c:
        per = pd.read_sql("SELECT DISTINCT Period_Start, Trade_Start FROM formation_pairs "
                          "WHERE strategy_id = ?", c, params=(FORMATION_SID,))
    f2t = dict(zip(per.Trade_Start, per.Period_Start))
    return {t for t in expected if pd.Timestamp(f2t[t]) < idx_start}


def period_table(sid: str, result_db=RESULT_DB) -> pd.DataFrame:
    """每期一列：交易日列數、進場次數（含反向）。"""
    q = ",".join("?" * len(ENTRY_STATUSES))
    with _con(result_db) as c:
        return pd.read_sql(
            f"SELECT Period_Start, COUNT(*) AS rows_n, "
            f"SUM(CASE WHEN Status IN ({q}) THEN 1 ELSE 0 END) AS entries "
            f"FROM trade_logs WHERE strategy_id = ? GROUP BY Period_Start "
            f"ORDER BY Period_Start", c, params=(*ENTRY_STATUSES, sid))


def daily_pnl_of(sid: str, periods=None, result_db=RESULT_DB) -> pd.Series:
    """逐日損益；`periods` 給定時只取該期集合（用於交集重算）。"""
    sql = "SELECT Date, SUM(Daily_Delta) AS d FROM trade_logs WHERE strategy_id = ?"
    args = [sid]
    if periods is not None:
        sql += f" AND Period_Start IN ({','.join('?' * len(periods))})"
        args += list(periods)
    with _con(result_db) as c:
        df = pd.read_sql(sql + " GROUP BY Date ORDER BY Date", c, params=args)
    s = pd.Series(df.d.values, index=pd.to_datetime(df.Date)).astype(float)
    return s[~s.index.duplicated()]


def period_notional(sid: str, result_db=RESULT_DB) -> pd.Series:
    """
    逐期名目額（進場＋出場，故 ×2）。

    每筆進場的名目額 = 當時的 `capital_per_pair` = 權益 / (top_n × 並行期數)，
    權益由該臂自己的逐日損益累計（與引擎一致，非初始資金）。
    """
    conc = concurrent_of(sid, result_db)
    max_pairs = max(1, TOP_N * max(1, conc))
    q = ",".join("?" * len(ENTRY_STATUSES))
    with _con(result_db) as c:
        ent = pd.read_sql(
            f"SELECT Date, Period_Start, COUNT(*) AS n FROM trade_logs "
            f"WHERE strategy_id = ? AND Status IN ({q}) GROUP BY Date, Period_Start",
            c, params=(sid, *ENTRY_STATUSES))
    if ent.empty:
        return pd.Series(dtype=float)
    pnl = daily_pnl_of(sid, result_db=result_db)
    eq = strategy_returns.equity_from_pnl(pnl)
    ent["Date"] = pd.to_datetime(ent.Date)
    cap = (eq / max_pairs).reindex(ent.Date).ffill().bfill().values
    ent["notional"] = ent.n.values * cap * 2.0
    return ent.groupby("Period_Start")["notional"].sum()


# ── 七項核對 ────────────────────────────────────────────────────────
def run_gate(paths: dict, expected: list) -> tuple:
    """回傳 (核對表 DataFrame, 共同期 list)。任一項不過則丟 GateFailure。"""
    checks, tables = [], {k: period_table(v) for k, v in paths.items()}
    missing = [k for k in ARMS if k not in paths]

    # 0（前置）：五臂都必須已落庫
    checks.append({"#": 0, "檢查": "五臂皆已落庫",
                   "通過": not missing,
                   "實際": "全部到齊" if not missing else f"缺 {missing}"})

    exp = set(expected)
    # 1：每期的交易日列數完整（截斷／半途失敗的期會少列）
    bad_rows = {k: int((t.rows_n != t.rows_n.mode().iat[0]).sum())
                for k, t in tables.items() if not t.empty}
    checks.append({"#": 1, "檢查": "每期交易日列數完整",
                   "通過": bool(bad_rows) and all(v == 0 for v in bad_rows.values()),
                   "實際": f"列數異常期數 {bad_rows}"})

    # 2：缺的期只能是左緣期，且四條非 v1 臂彼此完全相同
    #    （不能要求五臂集合全等——預先登記 4.1 已揭露 v1 算不了左緣，
    #      並把處置寫定為「取共同期交集」。）
    have = {k: set(t.Period_Start) for k, t in tables.items()}
    miss = {k: sorted(exp - v) for k, v in have.items()}
    left = left_edge_periods(expected)
    only_left = all(set(v) <= left for v in miss.values()) if miss else False
    others = {tuple(v) for k, v in miss.items() if k != "v1"}
    checks.append({"#": 2, "檢查": "缺期僅限左緣，且非 v1 四臂一致",
                   "通過": only_left and len(others) == 1,
                   "實際": {k: len(v) for k, v in miss.items()}})

    # 3：共同期 ≥ 100
    common = sorted(set.intersection(*have.values())) if have else []
    checks.append({"#": 3, "檢查": "五臂共同期 ≥ 100",
                   "通過": len(common) >= 100, "實際": f"{len(common)} 期"})

    # 4：逐期耗時快照證明 v1/v2 真的在訓練
    snap_ok, snap_txt = _check_durations()
    checks.append({"#": 4, "檢查": "v1/v2 逐期耗時 ≥ 60 分",
                   "通過": snap_ok, "實際": snap_txt})

    # 5 / 6：口徑與命名
    with _con() as c:
        hm = {k: c.execute("SELECT Hedge_Mode FROM strategy_summaries WHERE _path = ?",
                           (v,)).fetchone()[0] for k, v in paths.items()}
    checks.append({"#": 5, "檢查": "五臂皆 dollar 口徑",
                   "通過": bool(hm) and set(hm.values()) == {"dollar"}, "實際": hm})
    tagged = [m for m in ARMS.values() if "-PROBE" in m or "-SMOKE" in m]
    checks.append({"#": 6, "檢查": "db_method 不含 PROBE/SMOKE",
                   "通過": not tagged, "實際": tagged or "無"})

    # 7：落庫配對期連續（中段不得缺期）。左緣期不算缺——見 left_edge_periods。
    mid_gaps = {k: [p for p in v if p not in left] for k, v in miss.items()}
    n_mid = {k: len(v) for k, v in mid_gaps.items()}
    checks.append({"#": 7, "檢查": "中段無缺期",
                   "通過": bool(n_mid) and all(v == 0 for v in n_mid.values()),
                   "實際": n_mid})

    df = pd.DataFrame(checks)
    if not df["通過"].all():
        raise GateFailure(df)
    return df, common


def _check_durations() -> tuple:
    """
    第 4 項：耗時證據取自 `--snapshot` 落下的檔。

    `run_trading` 在整條臂跑完後會清除 `.ckpt`，故此證據**必須在跑的過程中**
    擷取，事後無法重建；快照不存在即視為未通過。
    """
    if not os.path.exists(SNAPSHOT):
        return False, f"快照不存在（{SNAPSHOT}）——須於回測進行中以 --snapshot 擷取"
    d = pd.read_csv(SNAPSHOT)
    med = dict(zip(d.臂, d.中位數分鐘))
    ok = all(med.get(a, 0) >= MIN_MEDIAN_MINUTES for a in CKPT_ARMS)
    return ok, {k: round(v, 1) for k, v in med.items()}


def snapshot_durations() -> pd.DataFrame:
    """把 checkpoint 的寫入間隔落檔（回測進行中執行；跑完後證據會消失）。"""
    rows = []
    for arm, d in CKPT_ARMS.items():
        p = os.path.join(CKPT_ROOT, d)
        fs = [os.path.join(p, f) for f in os.listdir(p)] if os.path.isdir(p) else []
        ts = sorted(os.path.getmtime(f) for f in fs if f.endswith(".pkl"))
        # 左緣空期是一瞬間叢發寫入（間隔 ~0 分），中斷則造成數百分鐘的假間隔；
        # 兩者都不是「算一期要多久」，必須排除，否則中位數被系統性拉低。
        gaps = [(b - a) / 60 for a, b in zip(ts, ts[1:])]
        gaps = [g for g in gaps if 1.0 <= g < 600.0]
        if gaps:
            rows.append({"臂": arm, "n": len(gaps),
                         "中位數分鐘": round(float(np.median(gaps)), 1),
                         "最短": round(min(gaps), 1), "最長": round(max(gaps), 1)})
    df = pd.DataFrame(rows)
    if not df.empty:
        os.makedirs(OUT_DIR, exist_ok=True)
        if os.path.exists(SNAPSHOT):
            # 新樣本優先（過濾規則可能修正過，舊值不得覆蓋新值）；舊列只在該臂
            # 已無法取樣時保留——整條臂跑完後 run_trading 會清掉它的 .ckpt。
            old = pd.read_csv(SNAPSHOT)
            df = pd.concat([old[~old["臂"].isin(df["臂"])], df]).sort_values("臂")
        df.to_csv(SNAPSHOT, index=False, encoding="utf-8-sig")
    return df


# ── 估計式 ──────────────────────────────────────────────────────────
def paired_bootstrap(d: np.ndarray, L: int = BLOCK_L_PERIODS,
                     B: int = N_BOOT, seed: int = SEED) -> dict:
    """
    逐期配對差分的循環區塊拔靴。單位即輸入單位（次／期），**不做任何年化**。

    與 `block_bootstrap.bootstrap_test` 的差別只在單位處理：該函式把輸入當
    日損益金額並乘 252／初始資金換算年化 %，用在「每期進場次數」上是錯的。
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < L * 2:
        return {"n": n, "Δ每期": np.nan, "p": np.nan, "CI下界": np.nan, "CI上界": np.nan}
    obs = float(d.mean())
    means = circular_block_bootstrap_means(d, L, B, np.random.default_rng(seed))
    p = float((np.abs(means - obs) >= abs(obs)).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": n, "Δ每期": round(obs, 4), "p": round(p, 4),
            "CI下界": round(float(lo), 4), "CI上界": round(float(hi), 4),
            "獨立區塊數": n // L}


def h1_turnover(paths: dict, common: list) -> pd.DataFrame:
    """H1：逐期配對差分檢定 ＋ 總數比值效果量。主族 3 臂 × 2 基準 = 6 個比較。"""
    ent = {k: period_table(v).set_index("Period_Start").entries.reindex(common).fillna(0)
           for k, v in paths.items()}
    rows = []
    for arm in FREE_ARMS:
        for base in BASE_ARMS:
            r = paired_bootstrap((ent[arm] - ent[base]).values)
            tot_a, tot_b = float(ent[arm].sum()), float(ent[base].sum())
            r.update({"臂": arm, "基準": base,
                      "總進場_臂": int(tot_a), "總進場_基準": int(tot_b),
                      "比值": round(tot_a / tot_b, 3) if tot_b else np.nan})
            rows.append(r)
    df = pd.DataFrame(rows)
    df["BH校正p"] = bh_adjust(df.p.values).round(4)
    df["達標"] = (df.比值 >= RATIO_THRESHOLD) & (df.BH校正p < ALPHA)
    return df[["臂", "基準", "n", "獨立區塊數", "Δ每期", "CI下界", "CI上界",
               "p", "BH校正p", "總進場_臂", "總進場_基準", "比值", "達標"]]


def h2_sharpe(paths: dict, common: list) -> pd.DataFrame:
    """H2：以共同期重算 Sharpe（主口徑），落庫 Sharpe_Raw 併列為穩健性。"""
    rows = []
    with _con() as c:
        stored = {k: c.execute("SELECT Sharpe_Raw FROM strategy_summaries WHERE _path = ?",
                               (v,)).fetchone()[0] for k, v in paths.items()}
    cal = price_calendar()
    for k, sid in paths.items():
        pnl = daily_pnl_of(sid, periods=common).reindex(cal).dropna()
        m = metrics_from_returns(strategy_returns.returns_from_pnl(pnl))
        rows.append({"臂": k, "交集Sharpe": round(float(m.Sharpe_Raw), 3),
                     "交集年化%": round(float(m.Ann_Ret_Raw) * 100, 2),
                     "交集MDD%": round(float(m.MDD_Raw) * 100, 2),
                     "落庫Sharpe(全期集)": round(float(stored[k]), 3),
                     "交易日": len(pnl)})
    df = pd.DataFrame(rows)
    v4 = df.loc[df.臂 == "v4", "交集Sharpe"].iat[0]
    df["不高於v4"] = np.where(df.臂.isin(FREE_ARMS), df.交集Sharpe <= v4, "")
    return df


def h3_cost(paths: dict, common: list) -> pd.DataFrame:
    """H3：零成本重算後，自由持倉臂相對 v4 的 Sharpe 差距是否縮小 ≥50%。"""
    cal = price_calendar()
    net, gross = {}, {}
    for k, sid in paths.items():
        pnl = daily_pnl_of(sid, periods=common)
        noti = period_notional(sid).reindex(common).fillna(0.0)
        fee = float(noti.sum()) * CURRENT_FEE_SIDE
        net[k] = strategy_returns.returns_from_pnl(pnl.reindex(cal).dropna())
        # 摩擦成本按交易日均攤回加：零成本情境只改水準，不改逐日波動結構
        add = fee / max(1, len(net[k]))
        gross[k] = strategy_returns.returns_from_pnl(
            (pnl + add).reindex(cal).dropna())
    rows = []
    for arm in FREE_ARMS:
        dn = float(metrics_from_returns(net[arm]).Sharpe_Raw
                   - metrics_from_returns(net["v4"]).Sharpe_Raw)
        dg = float(metrics_from_returns(gross[arm]).Sharpe_Raw
                   - metrics_from_returns(gross["v4"]).Sharpe_Raw)
        shrink = (1 - abs(dg) / abs(dn)) * 100 if dn else np.nan
        rows.append({"臂": arm, "含成本ΔSharpe(對v4)": round(dn, 3),
                     "零成本ΔSharpe(對v4)": round(dg, 3),
                     "差距縮小%": round(shrink, 1),
                     "H3成立": bool(shrink >= 50) if pd.notna(shrink) else False})
    return pd.DataFrame(rows)


# ── 自測 ────────────────────────────────────────────────────────────
def selftest() -> None:
    """以已知答案驗估計式與 BH。失敗即 assert，不靜默通過。"""
    rng = np.random.default_rng(0)
    n = 108
    # A：真實差為 0 → 不應顯著
    a = paired_bootstrap(rng.normal(0, 1, n))
    assert a["p"] > ALPHA, a
    # B：固定大位移 → 應顯著，且 Δ 應約等於位移
    b = paired_bootstrap(rng.normal(0, 1, n) + 5.0)
    assert b["p"] < ALPHA and abs(b["Δ每期"] - 5.0) < 0.5, b
    # C：樣本不足 → 回傳 NaN 而非硬跑
    assert np.isnan(paired_bootstrap(np.ones(BLOCK_L_PERIODS))["p"])
    # D：獨立區塊數 = n // L
    assert a["獨立區塊數"] == n // BLOCK_L_PERIODS
    # E：BH 對已知輸入（Benjamini-Hochberg step-up，保單調）
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.60])
    # 手算：p*n/i = .006 .024 .078 .0615 .0504 .60，再由右往左取累計最小值
    # （step-up 保單調）→ 第 3、4 項被第 5 項的 .0504 壓下來。
    exp = np.array([0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.60])
    assert np.allclose(bh_adjust(p), exp, atol=1e-4), bh_adjust(p)
    # F：比值口徑——總數比值不受零進場期影響
    e_free = pd.Series([3, 0, 6, 1]); e_base = pd.Series([1, 0, 2, 0])
    assert abs(e_free.sum() / e_base.sum() - 10 / 3) < 1e-9
    print("✔ 自測全數通過：拔靴 p 值、Δ 估計、區塊數、BH 校正、比值口徑")


# ── 主流程 ──────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true", help="合成資料自測估計式")
    ap.add_argument("--snapshot", action="store_true", help="落存逐期耗時證據")
    ap.add_argument("--dry-run", action="store_true",
                    help="以已落庫的臂乾跑管線；跳過閘門、不寫 CSV")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    if args.snapshot:
        df = snapshot_durations()
        print(df.to_string(index=False) if not df.empty else "（無 checkpoint 可取樣）")
        print(f"\n→ {SNAPSHOT}")
        return 0

    paths, expected = arm_paths(), expected_periods()
    print(f"視窗 {BACKTEST_START}~{BACKTEST_END}：應算期 {len(expected)}"
          f"（{expected[0]} ~ {expected[-1]}）")
    print(f"已落庫的臂 {len(paths)}/5：{sorted(paths)}\n")

    if args.dry_run:
        print("=" * 70)
        print("  ⚠ 乾跑（--dry-run）：**非結果**。閘門未執行，臂數可能不全，")
        print("     僅用於驗證 SQL、交集、名目額與輸出路徑是否接得起來。")
        print("=" * 70)
        common = sorted(set.intersection(
            *[set(period_table(v).Period_Start) for v in paths.values()]))
        print(f"\n可用臂的共同期：{len(common)}")
        for k, v in paths.items():
            t = period_table(v)
            print(f"  {k:<8} 期 {len(t):>3}  進場 {int(t.entries.sum()):>4}  "
                  f"每期列數眾數 {int(t.rows_n.mode().iat[0])}  "
                  f"名目額合計 ${period_notional(v).sum():,.0f}")
        if {"v4", "Z-Score"} <= set(paths) and len(common) >= BLOCK_L_PERIODS * 2:
            e4 = period_table(paths["v4"]).set_index("Period_Start").entries.reindex(common).fillna(0)
            ez = period_table(paths["Z-Score"]).set_index("Period_Start").entries.reindex(common).fillna(0)
            print("\n  兩基準互檢（應接近 0、不顯著）：",
                  paired_bootstrap((e4 - ez).values))
        print("\n⚠ 以上為乾跑輸出，未寫入任何 CSV。")
        return 0

    try:
        gate, common = run_gate(paths, expected)
    except GateFailure as e:
        print("✘ 閘門未通過——依預先登記第九節，不得就既有輸出做分析。\n")
        print(e.args[0].to_string(index=False))
        print("\n修正後重跑；本模組不輸出任何 p 值或效果量。")
        return 2

    print(gate.to_string(index=False))
    print(f"\n✔ 七項核對全數通過；共同期 {len(common)} 期"
          f"（{common[0]} ~ {common[-1]}）\n")

    h1, h2, h3 = h1_turnover(paths, common), h2_sharpe(paths, common), h3_cost(paths, common)
    n_pass = sum(h1.groupby("臂").達標.all())
    verdict = ("成立" if n_pass >= 2 else "不成立")

    print("── H1 換手率（主族 6 比較，BH 校正）" + "─" * 30)
    print(h1.to_string(index=False))
    print(f"\n  同時贏過兩個基準的臂：{n_pass}/3 → **H1 {verdict}**"
          f"（判準：≥2 條，比值 ≥{RATIO_THRESHOLD} 且 BH p<{ALPHA}）")
    print("\n── H2 Sharpe（共同期重算）" + "─" * 38)
    print(h2.to_string(index=False))
    print("\n── H3 成本歸因" + "─" * 50)
    print(h3.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, df in (("gate", gate), ("h1_turnover", h1),
                     ("h2_sharpe", h2), ("h3_cost", h3)):
        df.to_csv(f"{OUT_DIR}/ablation_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/ablation_{{gate,h1_turnover,h2_sharpe,h3_cost}}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
