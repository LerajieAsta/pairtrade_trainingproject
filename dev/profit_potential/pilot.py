# -*- coding: utf-8 -*-
"""獲利潛力排序（形成期）的接線驗證（PREREGISTRATION.md §六）。不寫入任何資料庫。

  V1. ranking_backend="ssd" 重跑 GICS-SSD 某期，與 formation DB 既有配對相同；
      另以 "ssd"＋reuse_groups_from 重跑 HDB-SSD，與 DB 逐名次相同（驗證沿用分組的機制）
  V2. "pot"＋reuse_groups_from 的 HDB／AGG／KM 分組，與 formation_groups 存檔相同
      （HDBSCAN 以現行程式重跑與存檔不同——2026-10-21 實測同群關係 9% 不一致——故 POT 臂沿用存檔分組）
  V3. GICS 某期：以逐對、非批次的 `_adf_stat` 獨立重算全部候選的 ADF、NZC、sd(S)/D、POT 與入選名單，與引擎一致
  V4. 同期 POT 入選配對的 sd(S)/D 中位數高於 SSD 入選配對；回報入選數

    python dev/profit_potential/pilot.py
"""
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from strategies import config as C  # noqa: E402
from strategies.formation.cluster_formation import Formation as CF  # noqa: E402
from strategies.formation._utils import _adf_stat  # noqa: E402
from strategies.preprocess_equity import DataProcessor  # noqa: E402

FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
PERIOD_K = 150          # 取第 150 個形成期（約 2012 年），不在任何邊緣


def setup():
    proc = DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, _, _ = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END, C.FORMATION_WINDOW)
    con = sqlite3.connect(f"file:{C.DB_PATH}?mode=ro", uri=True)
    mem = pd.read_sql("SELECT Symbol, start_date, end_date FROM index_memberships", con)
    con.close()
    return pv, list(dates), sector, mem


def form_prices(pv, dates, mem, form_start):
    """與 run_formation.worker_task 相同的切片：形成期 252 日、當期成分股、去缺值欄。"""
    i0 = dates.index(pd.Timestamp(form_start))
    idx = i0 + C.FORMATION_WINDOW
    fp = pv.iloc[i0:idx]
    fe = dates[idx - 1].strftime("%Y-%m-%d")
    act = set(mem[(mem.start_date <= fe) & (mem.end_date.isna() | (mem.end_date >= fe))].Symbol)
    fp = fp[[c for c in fp.columns if c in act]].dropna(axis=1)
    return fp, form_start, fe


def build(name, backend, fp, fs, fe, sector, reuse=None):
    raw = next(s for s in C.strategies_raw_all if s["name"] == name)
    p = dict(raw["params"])
    p.update(ranking_backend=backend, top_n=20)
    if reuse:
        p["reuse_groups_from"] = reuse
    import inspect
    ok = set(inspect.signature(CF.__init__).parameters)
    kw = {k: v for k, v in {"price_df": fp, "form_start": fs, "form_end": fe, "sector_mapping": sector,
                            **p}.items() if k in ok}
    f = CF(**kw)
    return f, f.run()


def main():
    pv, dates, sector, mem = setup()
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    periods = [r[0] for r in con.execute("SELECT DISTINCT Period_Start FROM formation_pairs "
                                         "WHERE strategy_id='Grid GICS-SSD_MSR0' ORDER BY Period_Start")]
    ps = periods[PERIOD_K]
    fp, fs, fe = form_prices(pv, dates, mem, ps)
    print(f"驗證期：形成期 {fs} ～ {fe}，可用標的 {fp.shape[1]}")
    ok = {}

    # V1
    for name, reuse in (("Grid GICS-SSD", None), ("Grid HDB-SSD", "Grid HDB-SSD_MSR0")):
        _, got = build(name, "ssd", fp, fs, fe, sector, reuse)
        db = pd.read_sql("SELECT Ticker_A, Ticker_B, Pair_Rank FROM formation_pairs WHERE strategy_id=? "
                         "AND Period_Start=? ORDER BY Pair_Rank", con, params=(f"{name}_MSR0", ps))
        a = list(zip(got.Ticker_A, got.Ticker_B))
        b = list(zip(db.Ticker_A, db.Ticker_B))
        ok[f"V1 {name}"] = a == b
        print(f"[V1] {name}：重跑 {len(a)} 對 vs DB {len(b)} 對，逐名次相同 = {a == b}")

    # V2
    for g in ("HDB", "AGG", "KM"):
        f2, _ = build(f"Grid {g}-SSD", "pot", fp, fs, fe, sector, f"Grid {g}-SSD_MSR0")
        st = {t: int(l) for t, l in con.execute("SELECT Ticker, Cluster_Label FROM formation_groups "
                                                "WHERE strategy_id=? AND Period_Start=?",
                                                (f"Grid {g}-SSD_MSR0", ps)).fetchall()}
        same = f2.cluster_labels_ == st
        ok[f"V2 {g}"] = same
        print(f"[V2] {g}：POT 分組與存檔相同 = {same}（{len(set(st.values()))} 個標籤）")

    # V3：GICS 獨立重算
    fpot, got = build("Grid GICS-SSD", "pot", fp, fs, fe, sector)
    _, ssd_sel = build("Grid GICS-SSD", "ssd", fp, fs, fe, sector)
    lp = np.log(np.maximum(fp, 1e-8))
    mu, sd_ = lp.mean(), lp.std()
    z = (lp - mu) / (sd_ + 1e-12)
    f2 = 2 * (float(C.base_params["fee_rate"]) + float(C.base_params["slippage_rate"]))
    rows = []
    groups = {}
    for t in z.columns:
        groups.setdefault(sector.get(t, "Unknown"), []).append(t)
    for gname, ts in groups.items():
        if gname == "Unknown" or len(ts) < 2:
            continue
        for ii in range(len(ts)):
            for jj in range(ii + 1, len(ts)):
                b, a = ts[ii], ts[jj]
                x, y = z[b].values, z[a].values
                c = np.cov(np.vstack([x, y]))
                beta = c[0, 1] / c[0, 0] if c[0, 0] > 1e-8 else 0.0
                s = y - beta * x
                if _adf_stat(s, max_lags=1)[1] >= 0.05:
                    continue
                nzc = int((np.diff(np.sign(s - s.mean())) != 0).sum())
                rpz = np.std(s, ddof=1) / (1 / sd_[a] + abs(beta) / sd_[b])
                pot = nzc * (2.0 * rpz - f2)
                if pot > 0:
                    rows.append((a, b, pot))
    ind = sorted(rows, key=lambda r: -r[2])[:20]
    same_list = [(r[0], r[1]) for r in ind] == list(zip(got.Ticker_A, got.Ticker_B))
    maxdiff = max(abs(r[2] - p) for r, p in zip(ind, got.POT)) if same_list else np.inf
    ok["V3"] = same_list and maxdiff < 1e-9
    print(f"[V3] GICS 獨立重算：通過 ADF 且 POT>0 的候選 {len(rows)}；入選名單逐名次相同 = {same_list}；"
          f"POT 最大差 {maxdiff:.1e}")

    # V4
    rz = lambda df: [float(np.std(z[a].values - h * z[b].values, ddof=1) / (1 / sd_[a] + abs(h) / sd_[b]))
                     for a, b, h in zip(df.Ticker_A, df.Ticker_B, df.Hedge_Ratio)]
    m_pot, m_ssd = np.median(rz(got)), np.median(rz(ssd_sel))
    ok["V4"] = m_pot > m_ssd
    print(f"[V4] 每單位 z 的報酬中位數：POT {m_pot:.4%} vs SSD {m_ssd:.4%}；POT 入選 {len(got)} 對 → "
          f"{'通過' if ok['V4'] else '失敗'}")
    print("\n全部通過" if all(ok.values()) else f"\n未通過：{[k for k, v in ok.items() if not v]}")


if __name__ == "__main__":
    main()
