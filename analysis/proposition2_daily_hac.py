# -*- coding: utf-8 -*-
"""
命題 2 主檢定：逐日報酬差 + 循環 block bootstrap
======================================================================
取代 `proposition2_stats._paired_tests` 的 15 格配對 t 檢定作為**主檢定**。

為什麼要換
----------
舊設計以「參數網格」為抽樣單位：top_n ∈ {1,3,5,10,20} × stop_loss ∈ {0,5%,15%}
＝ 15 個觀測，餵給 `ttest_rel`。但這 15 格是**同一份資料、同一段期間、同一批
配對**的 15 種投資組合設定——top_n=10 與 top_n=20 共用 10 個配對，三種停損是
同一批交易的不同出場規則。觀測間相關係數極高，`ttest_rel` 的獨立性假設不成立
（pseudo-replication，有效樣本數 ≈ 1 條回測路徑，而非 15）。

本模組改以**時間**為抽樣單位：對每個配對底取

    Δr_t = r_DRL,t − r_ZScore,t         （逐日，約 6,300 個交易日）

檢定 H0: E[Δr] = 0，並以循環 block bootstrap（L=126）處理重疊部位造成的
自相關，同時輸出雙尾 p 與 95% 信賴區間。方法與 L 的選擇理由見
`analysis.block_bootstrap`。

為什麼不是「逐滾動期 ΔSharpe」
------------------------------
FORWARD_DAYS=126 / rolling_step=21 → 任一時點有 6 個交易期同時在跑
（CONCURRENT_PERIODS=6）。以「期」為單位的話，相鄰 6 期共用 5/6 的日曆時間，
只是把偽重複從參數維度搬到時間維度。逐日序列 + 區塊重抽正是為這種自相關
設計的，不必刪資料、不必降 n。

HAC 的角色
----------
`newey_west` 仍保留在本模組，一則供其他分析模組沿用，二則在主表附一欄
作為「參數法與無母數法結論一致」的對照。它不再是任何一章的主檢定，
故也不再需要 lags ∈ {auto, 63, 126, 252} 的落後階敏感度分析——
換成 bootstrap 後根本沒有落後階要選。

兩種聚合口徑
------------
  A. 等權組合（主口徑）：15 格逐日報酬先等權平均再相減，避免挑格子（selection bias）
  B. 逐格檢定（次口徑）：15 格各自檢定，報「15 格中幾格顯著」與 t 值分布

註：Δr 為兩策略相減，兩臂閒置現金同樣賺 rf，故無風險利率在差分中對消——
    本檢定不受 `Sharpe_Raw` 未扣 rf 的口徑問題影響。

用法：python -m analysis.proposition2_daily_hac
"""
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.block_bootstrap import BLOCK_L, bootstrap_test

# Windows 主控台預設 cp950，無法輸出 U+2212 等符號（config.py 亦作同樣處理）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"
CACHE = os.path.join(OUT_DIR, "daily_returns_mainaxis.parquet")
TRADING_DAYS = 252
INITIAL_CAPITAL = 10000.0   # 同 config.INITIAL_CAPITAL；差分為兩臂相減，縮放不影響 t 值

# (配對底, Z-Score 的 db_method, DRL 的 db_method)
# 含 GICS 兩組傳統配對底——命題 2 宣稱涵蓋五種配對底，檢定就必須涵蓋五種
PAIRS = [
    ("Agglomerative", "Grid (AGG-SSD)",  "Grid (AGG-SSD-DRL)"),
    ("HDBSCAN",       "Grid (HDB-SDP)",  "Grid (HDB-SDP-DRL)"),
    ("K-means",       "Grid (KM-SSD)",   "Grid (KM-SSD-DRL)"),
    ("GICS-SSD（傳統）", "Grid (GICS-SSD)", "Grid (GICS-SSD-DRL)"),
    ("GICS-SDP（傳統）", "Grid (GICS-SDP)", "Grid (GICS-SDP-DRL)"),
]


# ── HAC（對照用，非主檢定）──────────────────────────────────────────
def newey_west(r: np.ndarray, lags=None):
    """H0: E[r]=0。Bartlett kernel HAC 標準誤。回傳 (t, p, lags_used)。"""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30:
        return np.nan, np.nan, 0
    if lags is None or lags == "auto":
        lags = int(np.floor(4 * (T / 100.0) ** (2 / 9)))
    lags = int(min(lags, T - 1))
    e = r - r.mean()
    s = (e @ e) / T
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * ((e[L:] @ e[:-L]) / T)
    # 加權自相關可能使長期變異估計為負（小樣本 Bartlett 仍可能發生）→ 退回白噪音變異
    if s <= 0:
        s = (e @ e) / T
    se = np.sqrt(s / T)
    t = r.mean() / se
    return t, 2 * (1 - norm.cdf(abs(t))), lags


# ── 資料 ────────────────────────────────────────────────────────────
def _grid_cell(sid: str) -> str:
    """strategy_id → 網格格子標籤（Top{N}_SL{X}），供兩臂對齊。"""
    return os.path.basename(sid).replace("TradeLogs_", "").replace(".csv", "") \
                                .replace("_ZWin0_MSR0", "")


# 基準格：TradeLogs_Top{N}_SL{X}_ZWin0_MSR0.csv，無任何後綴。
# entry_z / 動態槽位 / 時間停損等變體會在檔名尾端加 _EZ.. _DYN.. _MHD.. 等後綴，
# 它們與基準共用同一個 db_method，若不排除會混進等權組合，污染主檢定。
_BASELINE_CELL = re.compile(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv$")


def baseline_only(sids: list[str]) -> list[str]:
    return [s for s in sids if _BASELINE_CELL.search(os.path.basename(s))]


def load_daily_sids(sids: list[str], use_cache: bool = True) -> pd.DataFrame:
    """
    回傳 DataFrame(index=Date, columns=strategy_id, values=日損益)。

    trade_logs 有 3.34 億列，逐日聚合很貴，故快取到 parquet 並**增量**補齊：
    只查快取裡沒有的 strategy_id，再與既有欄位合併。加新策略（如 entry_z 變體）
    時不必重掃全表。
    """
    sids = list(dict.fromkeys(sids))
    cached = pd.DataFrame()
    if use_cache and os.path.exists(CACHE):
        cached = pd.read_parquet(CACHE)

    missing = [s for s in sids if s not in cached.columns]
    if missing:
        con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
        print(f"  聚合 {len(missing)} 條新 strategy_id 的逐日損益"
              f"（快取已有 {len(cached.columns)} 條，走 (strategy_id,Date) 索引）…")
        q = (f"SELECT strategy_id, Date, SUM(Daily_Delta) AS pnl FROM trade_logs "
             f"WHERE strategy_id IN ({','.join('?' * len(missing))}) "
             f"GROUP BY strategy_id, Date")
        raw = pd.read_sql(q, con, params=missing)
        con.close()
        if raw.empty:
            print("  ⚠ 查無資料——這些 strategy_id 在 trade_logs 中不存在")
        else:
            raw["Date"] = pd.to_datetime(raw["Date"])
            new = raw.pivot(index="Date", columns="strategy_id", values="pnl")
            cached = new if cached.empty else cached.join(new, how="outer")
            # 未持倉日 = 無損益列 → 補 0（不是遺漏值，是「當日沒有部位」）
            cached = cached.sort_index().fillna(0.0)
            os.makedirs(OUT_DIR, exist_ok=True)
            cached.to_parquet(CACHE)
            print(f"  已快取 → {CACHE}"
                  f"（{cached.shape[0]} 個交易日 × {cached.shape[1]} 條策略）")

    have = [s for s in sids if s in cached.columns]
    return cached[have]


def invalidate_cached_sids(prefixes: list[str], force: bool = False) -> int:
    """
    刪掉快取中符合任一前綴的 strategy_id 欄，回傳刪除欄數。

    快取只以 strategy_id 為鍵，沒有上游資料的指紋，所以「重跑一條既有策略」
    不會讓它失效——load_daily_sids 看到欄位已存在就直接回傳舊序列，分析結果
    會與重跑前逐位元相同而不報任何錯。凡是重跑過 run_trading 的策略，都必須
    先呼叫本函式再做分析。

    ⚠ 守衛：拒絕刪掉「trade_logs 已無明細」的欄。
    2026-08-06 起封存策略的明細已清除（tools/archive_trade_logs.py），逐日序列
    只存在於這份快取裡——刪掉就再也重建不回來。前綴比對很容易誤傷（例如
    "tiingo/Grid_AGG_SSD" 會同時命中 Grid_AGG_SSD_DRL、_NOSEC、_NF…），故本
    函式逐欄確認 trade_logs 仍有列才刪；確定要刪無明細者請傳 force=True。
    """
    if not os.path.exists(CACHE):
        return 0
    cached = pd.read_parquet(CACHE)
    hit = [c for c in cached.columns if any(c.startswith(p) for p in prefixes)]
    if not hit:
        return 0

    if force:
        drop, keep = hit, []
    else:
        con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
        drop, keep = [], []
        for c in hit:
            has = con.execute(
                "SELECT 1 FROM trade_logs WHERE strategy_id=? LIMIT 1", (c,)).fetchone()
            (drop if has else keep).append(c)
        con.close()
    if keep:
        print(f"  ⚠ {len(keep)} 欄命中前綴但 trade_logs 已無明細，保留不刪"
              f"（刪了無法重建）：{keep[:3]}{' …' if len(keep) > 3 else ''}")
    if drop:
        cached.drop(columns=drop).to_parquet(CACHE)
    return len(drop)


def method_paths(methods: list[str]) -> pd.DataFrame:
    """回傳 strategy_summaries 的 (METHOD, _path)，供呼叫端自行挑格子。"""
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    df = pd.read_sql(
        f"SELECT METHOD, _path FROM strategy_summaries WHERE METHOD IN "
        f"({','.join('?' * len(methods))})", con, params=methods)
    con.close()
    return df


def load_daily(methods: list[str], use_cache: bool = True) -> pd.DataFrame:
    """以 METHOD 清單載入（相容舊呼叫端）。"""
    return load_daily_sids(method_paths(methods)._path.tolist(), use_cache)


# ── 檢定 ────────────────────────────────────────────────────────────
def _series_stats(d: np.ndarray) -> dict:
    """
    差分序列的描述統計。

    注意「資訊比率」＝ sqrt(252)·mean(Δr)/std(Δr)，是「做多 DRL、做空 Z-Score」
    這個主動部位的 Sharpe，**不等於**舊檢定的「兩策略 Sharpe 相減」（15 格平均
    ΔSharpe ≈ 0.27–0.30）。兩者是不同統計量，數值不可直接比較；可比較的是
    年化Δ報酬（本表「年化Δ%」對上舊表「Δ年化pp」）。
    """
    mu, sd = d.mean(), d.std(ddof=1)
    return {
        "日均Δ$": round(float(mu), 4),
        "年化Δ%": round(float(mu * TRADING_DAYS) / INITIAL_CAPITAL * 100, 3),
        "資訊比率IR": round(float(np.sqrt(TRADING_DAYS) * mu / sd), 4) if sd > 0 else np.nan,
        "勝日%": round(float((d > 0).mean()) * 100, 1),
    }


def ew_diff_series() -> dict[str, np.ndarray]:
    """
    每個配對底一條「DRL − Z-Score」的等權組合逐日差分序列。

    抽出來供 block bootstrap 等其他檢定共用，確保各檢定吃的是**同一條序列**
    （否則對照結果不可比）。
    """
    methods = sorted({m for _, z, d in PAIRS for m in (z, d)})
    meta = method_paths(methods)
    m2s = {m: baseline_only(g._path.tolist()) for m, g in meta.groupby("METHOD")}
    px = load_daily_sids([s for v in m2s.values() for s in v])

    out = {}
    for base, zs_m, drl_m in PAIRS:
        z_ids, d_ids = m2s.get(zs_m, []), m2s.get(drl_m, [])
        if not z_ids or not d_ids:
            continue
        zc = {_grid_cell(s): s for s in z_ids}
        dc = {_grid_cell(s): s for s in d_ids}
        cells = sorted(set(zc) & set(dc))
        if not cells:
            continue
        out[base] = (px[[dc[c] for c in cells]].mean(axis=1)
                     - px[[zc[c] for c in cells]].mean(axis=1)).values
    return out


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    methods = sorted({m for _, z, d in PAIRS for m in (z, d)})
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    meta = pd.read_sql(
        f"SELECT METHOD, _path FROM strategy_summaries WHERE METHOD IN "
        f"({','.join('?' * len(methods))})", con, params=methods)
    con.close()
    # 只取基準格：entry_z 等變體與基準共用 db_method，混進來會污染等權組合
    m2s = {m: baseline_only(g._path.tolist()) for m, g in meta.groupby("METHOD")}

    px = load_daily(methods)

    ew_rows, cell_rows = [], []

    for base, zs_m, drl_m in PAIRS:
        z_ids, d_ids = m2s.get(zs_m, []), m2s.get(drl_m, [])
        if not z_ids or not d_ids:
            print(f"  ⚠ 略過 {base}：result.db 缺 {zs_m if not z_ids else drl_m}")
            continue

        # 依網格格子對齊兩臂（只用兩邊都有的格）
        zc = {_grid_cell(s): s for s in z_ids}
        dc = {_grid_cell(s): s for s in d_ids}
        cells = sorted(set(zc) & set(dc))

        # ── A. 等權組合（主口徑）
        z_ew = px[[zc[c] for c in cells]].mean(axis=1)
        d_ew = px[[dc[c] for c in cells]].mean(axis=1)
        diff = (d_ew - z_ew).values
        res = bootstrap_test(diff)
        _, p_nw, _ = newey_west(diff)      # 對照欄，見模組 docstring
        ew_rows.append({"配對底": base, "格數": len(cells), "交易日": len(diff),
                        **_series_stats(diff),
                        "CI下界": res["CI下界"], "CI上界": res["CI上界"],
                        "BB p": res["BB p"], "5%顯著": res["顯著"],
                        "NW p（對照）": round(p_nw, 4)})

        # ── B. 逐格檢定（次口徑）：單一參數設定的檢定力
        anns, sig = [], 0
        for c in cells:
            rc = bootstrap_test((px[dc[c]] - px[zc[c]]).values)
            anns.append(rc["年化Δ%"])
            sig += int(rc["BB p"] < 0.05 and rc["年化Δ%"] > 0)
        cell_rows.append({"配對底": base, "格數": len(cells),
                          "正向顯著格數": f"{sig}/{len(cells)}",
                          "年化Δ中位": round(float(np.median(anns)), 3),
                          "年化Δ最小": round(float(np.min(anns)), 3),
                          "年化Δ最大": round(float(np.max(anns)), 3)})

    ew = pd.DataFrame(ew_rows)
    cells_df = pd.DataFrame(cell_rows)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 88)
    print("一、等權組合逐日差分 + block bootstrap（主口徑）")
    print(f"    H0: E[r_DRL − r_ZScore] = 0；L={BLOCK_L}，10,000 次重抽")
    print("=" * 88)
    print(ew.to_string(index=False))

    print("\n" + "=" * 88)
    print("二、逐格檢定（次口徑；15 格各自重抽）——單一參數設定的檢定力")
    print("=" * 88)
    print(cells_df.to_string(index=False))

    ew.to_csv(f"{OUT_DIR}/prop2_daily_hac_ew.csv", index=False, encoding="utf-8-sig")
    cells_df.to_csv(f"{OUT_DIR}/prop2_daily_hac_cells.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop2_daily_hac_{{ew,cells}}.csv")


if __name__ == "__main__":
    run()
