# -*- coding: utf-8 -*-
"""
論文引用數字的單一真相來源
======================================================================
把論文各章引用的每一個數字，改成從資料庫與 config 重算後輸出成一張表。

**為什麼需要這支工具。** 同一個數字先前散落在 `thesis/*.md`、`PROJECT_GUIDE.md`、
`strategies/config.py` 註解與記憶檔四處手抄，且各處抄自不同時期的回測，
彼此已經對不上——實測到的矛盾包括：策略數 17／36／42、DSR 試驗數 87／104／110、
已下市檔數 170／210、交易日數 6,287（實為報酬序列長度，資料期間為 6,539）。
管線一改，這類矛盾只會再生。

**用法。**
    python -m tools.thesis_numbers              # 僅主軸策略（快）
    python -m tools.thesis_numbers --all        # 全部策略
    python -m tools.thesis_numbers --json-only  # 不寫 Markdown

**輸出。**
    results/analysis/thesis_numbers.json   機器可讀（供其他腳本引用）
    results/analysis/thesis_numbers.md     依論文章節分組的對照表

每筆數字都帶 `section` 欄位標明它被論文哪一節引用，使核對成為機械動作而非通讀。

**注意。** formation 重跑期間執行會得到部分結果（期數未滿），表中 `note` 欄會標示。
`formation_ranked`（逐候選 ADF/半衰期/Hurst 軌跡）僅在 `FORMATION_TRACE=1` 時產生；
未產生時，與篩選層通過率有關的欄位會顯示為 None 而非猜測值。
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# 路徑 shim：本工具位於 tools/，將專案根加入 sys.path 並切換 CWD（相對路徑以根為準）
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 主控台預設 cp950，含非 ASCII 的輸出會拋 UnicodeEncodeError
# （snapshot_run.py 即因此在寫檔成功後才崩潰）。這裡先固定為 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import strategies.config as C  # noqa: E402

FUNDAMENTALS_PARQUET = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet"
FORMATION_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"

# 主軸策略：命題 1 的 4 分組 × 3 排序矩陣
MAIN_AXIS = [f"Grid {g}-{r}" for g in ("GICS", "HDB", "AGG", "KM")
             for r in ("SSD", "DTW", "SDP")]

FORMATION_TOP_N = 20   # 形成期選對數固定 20，不隨交易端 top_n_list 變動


def _ro(path: str):
    """唯讀開啟；formation 重跑期間仍可讀（WAL）。找不到檔案回 None。"""
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


class Facts:
    """收集 (section, key, label, value, note) 五元組。"""

    def __init__(self):
        self.rows = []

    def add(self, section, key, label, value, note=""):
        self.rows.append({"section": section, "key": key, "label": label,
                          "value": value, "note": note})

    def to_json(self):
        return {r["key"]: {k: r[k] for k in ("section", "label", "value", "note")}
                for r in self.rows}


# ── 來源一：config ────────────────────────────────────────────────────
def collect_config(f: Facts):
    bp = C.base_params
    n_grid = len(bp["top_n_list"]) * len(bp["stop_loss_list"])
    borrowed = [s for s in C.strategies_raw_all if s.get("formation_strategy_id_base")]

    f.add("3.1.3", "formation_window", "形成期長度（交易日）", C.FORMATION_WINDOW)
    f.add("3.1.3", "trading_window", "交易期長度（交易日）", C.FORWARD_DAYS)
    f.add("3.1.3", "rolling_step", "滾動步長（交易日）", C.rolling_step)
    f.add("3.1.3", "concurrent_periods", "同時重疊期數", C.CONCURRENT_PERIODS,
          f"= {C.FORWARD_DAYS} / {C.rolling_step}")
    f.add("3.1.4", "fee_rate_oneway", "單邊交易成本", bp["fee_rate"],
          "Do & Faff (2012)；一往返 = 2 倍")
    f.add("3.5.5", "n_grid_configs", "每策略參數配置數", n_grid,
          f"top_n {len(bp['top_n_list'])} x stop_loss {len(bp['stop_loss_list'])}")
    f.add("3.3.3", "n_strategies_total", "config 內策略總數", len(C.strategies_raw_all))
    f.add("3.3.3", "n_strategies_formation", "實際計算形成期的策略數",
          len(C.strategies_raw_all) - len(borrowed),
          f"另 {len(borrowed)} 條借用他者配對")
    f.add("3.2.5", "adf_pvalue_threshold", "ADF 通過門檻",
          C._GRID_COMMON["adf_pvalue_threshold"], "EG 臨界值校準（MacKinnon N=2）")
    f.add("3.2.5", "filter_mode", "篩選層設定", C._GRID_COMMON["filter_mode"])
    f.add("3.2.2", "pca_n_components", "報酬主成分數", C._GRID_COMMON["pca_n_components"])
    f.add("3.2.2", "impute_scope", "缺失值插補範圍", C._GRID_COMMON["impute_scope"])
    f.add("3.2.3", "min_cluster_size", "過小群門檻", C._GRID_COMMON["min_cluster_size"])
    f.add("3.2.3", "hdbscan_min_cluster_size", "HDBSCAN 最小群規模",
          C._GRID_COMMON["hdbscan_min_cluster_size"])
    f.add("3.2.3", "agg_threshold_percentile", "Agglomerative 切割分位",
          C._GRID_COMMON["agg_threshold_percentile"])
    f.add("3.2.4", "dtw_window", "Sakoe-Chiba 頻帶寬", C._GRID_COMMON["dtw_window"])
    f.add("3.3.4", "dynamic_slots", "動態槽位資金配置", bp["dynamic_slots"])
    f.add("3.3.4", "slot_percentile", "有效槽位分位數", bp["slot_percentile"])
    f.add("3.3.4", "pair_cap_frac", "單配對資本上限", bp["pair_cap_frac"])


# ── 來源二：價格資料庫 ────────────────────────────────────────────────
def collect_price(f: Facts):
    conn = _ro(C.DB_PATH)
    if conn is None:
        f.add("3.1.1", "price_db", "價格資料庫", None, f"找不到 {C.DB_PATH}")
        return
    q = conn.execute

    lo, hi = f"{C.BACKTEST_START}-01", f"{C.BACKTEST_END}-31"
    n_all = q("SELECT COUNT(DISTINCT Date) FROM Daily_Prices").fetchone()[0]
    n_win = q("SELECT COUNT(DISTINCT Date) FROM Daily_Prices WHERE Date BETWEEN ? AND ?",
              (lo, hi)).fetchone()[0]
    d0, d1 = q("SELECT MIN(Date), MAX(Date) FROM Daily_Prices").fetchone()

    f.add("3.1.1", "trading_days_backtest", "回測區間交易日數", n_win,
          f"{C.BACKTEST_START} ~ {C.BACKTEST_END}")
    f.add("3.1.1", "return_series_days", "報酬序列長度", n_win - C.FORMATION_WINDOW,
          f"= {n_win} - {C.FORMATION_WINDOW}（扣第一個形成窗）")
    f.add("3.1.1", "trading_days_db_all", "資料庫全部交易日數", n_all, f"{d0} ~ {d1}")
    f.add("3.1.1", "n_constituents", "曾為成分股的標的數",
          q("SELECT COUNT(DISTINCT Symbol) FROM Constituents").fetchone()[0])
    f.add("3.1.1", "n_symbols_priced", "有價格資料的標的數",
          q("SELECT COUNT(DISTINCT Symbol) FROM Daily_Prices").fetchone()[0])

    m = pd.read_sql_query("SELECT Symbol, start_date, end_date FROM index_memberships", conn)
    f.add("3.1.2", "n_membership_spans", "成員期間段數", len(m))
    f.add("3.1.2", "n_membership_symbols", "不重複標的數", int(m.Symbol.nunique()))
    f.add("3.1.2", "n_membership_ended", "已有剔除日的段數", int(m.end_date.notna().sum()),
          "「曾被剔出指數」不等同「已下市」")
    conn.close()


# ── 來源三：基本面 parquet ────────────────────────────────────────────
def collect_fundamentals(f: Facts):
    if not os.path.exists(FUNDAMENTALS_PARQUET):
        f.add("3.1.1", "fundamentals", "基本面資料集", None,
              f"找不到 {FUNDAMENTALS_PARQUET}")
        return
    d = pd.read_parquet(FUNDAMENTALS_PARQUET).reset_index()
    d["yr"] = pd.to_datetime(d["date"]).dt.year

    cov = {}
    for yr, g in d.groupby("yr"):
        cov[int(yr)] = {"market_cap": round(100 * g.market_cap.notna().mean(), 1),
                        "pe_ratio": round(100 * g.pe_ratio.notna().mean(), 1)}
    both = 100 * (d.market_cap.notna() & d.pe_ratio.notna()).mean()
    first = min((y for y, v in cov.items() if v["market_cap"] > 0), default=None)

    f.add("3.1.1", "fundamentals_source", "基本面來源", "FMP point-in-time（月頻）")
    f.add("3.1.1", "fundamentals_coverage_by_year", "逐年覆蓋率（%）", cov)
    f.add("3.1.1", "fundamentals_coverage_both", "兩欄同時有值比例（%）", round(both, 1))
    f.add("3.2.2", "fundamentals_first_year", "市值首個有資料年份", first,
          "此前基本面區塊為全零常數，對分群距離無貢獻")


# ── 來源四：形成期資料庫 ──────────────────────────────────────────────
def collect_formation(f: Facts, all_strategies: bool):
    conn = _ro(FORMATION_DB)
    if conn is None:
        f.add("3.3.3", "formation_db", "形成期資料庫", None, f"找不到 {FORMATION_DB}")
        return

    names = [s["name"] for s in C.strategies_raw_all] if all_strategies else MAIN_AXIS
    sids = {n: f"{n}_MSR0" for n in names}

    # 新鮮度判定：`formation_groups` 於重跑前被清空，且每條策略在**跑完時**才整批
    # 合併回主庫。故「有 groups 列」⇔「本次重跑已合併此策略」。
    # 沒有這個判定，重跑期間讀到的 formation_pairs 會是尚未被覆寫的舊資料，
    # 而列數與期數都正常，無從察覺。
    merged = {r[0] for r in conn.execute(
        "SELECT DISTINCT strategy_id FROM formation_groups")}

    pairs, groups = {}, {}
    for name, sid in sids.items():
        d = pd.read_sql_query(
            "SELECT Period_Start, Sector_A, Sector_B FROM formation_pairs "
            "WHERE strategy_id = ?", conn, params=(sid,))
        if d.empty:
            continue
        per = d.groupby("Period_Start").size()
        pairs[name] = {
            "periods": int(per.size),
            "pairs_per_period": round(float(per.mean()), 2),
            "fill_rate_pct": round(100 * float(per.mean()) / FORMATION_TOP_N, 1),
            "cross_sector_pct": round(100 * float((d.Sector_A != d.Sector_B).mean()), 1),
            "fresh": sid in merged,
        }

        g = pd.read_sql_query(
            "SELECT Period_Start, Cluster_Label FROM formation_groups "
            "WHERE strategy_id = ?", conn, params=(sid,))
        if g.empty:
            continue
        g["lab"] = g.Cluster_Label.astype(str)
        n_tot, n_excl, n_grp, sz = [], [], [], []
        for _, gg in g.groupby("Period_Start"):
            vc = gg.lab.value_counts()
            # 排除規則與 cluster_formation 一致：Unknown、噪音(-1)、成員數 < 門檻
            keep = vc[(vc.index != "Unknown") & (vc.index != "-1")
                      & (vc >= C._GRID_COMMON["min_cluster_size"])]
            n_tot.append(len(gg))
            n_excl.append(len(gg) - int(keep.sum()))
            n_grp.append(len(keep))
            sz.append(float(keep.median()) if len(keep) else 0.0)
        groups[name] = {
            "universe_median": int(np.median(n_tot)),
            "excluded_pct": round(100 * float(np.mean(np.array(n_excl) / np.array(n_tot))), 1),
            "n_groups_median": int(np.median(n_grp)),
            "group_size_median": int(np.median(sz)),
        }

    stale = [n for n, v in pairs.items() if not v["fresh"]]
    f.add("_freshness", "formation_stale_strategies", "尚未被本次重跑覆寫的策略",
          stale or "（無，全部為最新）",
          "這些策略的形成期數字仍為前一版管線的產物，不可引用"
          if stale else "")
    f.add("_freshness", "formation_db_mtime", "形成期資料庫最後寫入時間",
          datetime.fromtimestamp(os.path.getmtime(FORMATION_DB)).strftime("%Y-%m-%d %H:%M"))

    expected = None
    if pairs:
        expected = max(v["periods"] for v in pairs.values())
    partial = [n for n, v in pairs.items() if expected and v["periods"] < expected]

    f.add("3.1.3", "n_formation_periods", "形成期期數", expected,
          "取各策略最大值；重跑期間為部分結果" if partial else "")
    f.add("3.2.3", "formation_pairs_by_strategy", "各策略每期配對數與填滿率", pairs,
          f"形成期選對數固定 {FORMATION_TOP_N}")
    f.add("3.2.3", "formation_groups_by_strategy", "各策略分組結構", groups,
          "excluded 含 Unknown、HDBSCAN 噪音、過小群")
    if partial:
        f.add("3.2.3", "_incomplete", "期數未滿的策略", partial,
              "formation 尚在執行；數字為部分結果")

    n_ranked = conn.execute("SELECT COUNT(*) FROM formation_ranked").fetchone()[0]
    f.add("3.2.5", "formation_ranked_rows", "逐候選軌跡列數", int(n_ranked),
          "0 表示本次重跑未開 FORMATION_TRACE=1；篩選層通過率無法自 DB 重算")
    conn.close()


# ── 來源五：回測結果資料庫 ────────────────────────────────────────────
def collect_result(f: Facts):
    conn = _ro(RESULT_DB)
    if conn is None:
        f.add("3.5.5", "result_db", "回測結果資料庫", None,
              f"找不到 {RESULT_DB}（尚未執行 run_trading）")
        return
    q = conn.execute
    n_sum = q("SELECT COUNT(*) FROM strategy_summaries").fetchone()[0]
    n_m = q('SELECT COUNT(DISTINCT "METHOD") FROM strategy_summaries').fetchone()[0]
    f.add("3.5.5", "n_summary_rows", "summary 列數", int(n_sum))
    f.add("3.5.5", "n_distinct_methods", "相異 METHOD 數", int(n_m),
          "DSR 試驗數 N 的候選口徑之一")
    conn.close()


# ── 輸出 ──────────────────────────────────────────────────────────────
def _fmt(v):
    if isinstance(v, dict):
        return "見 JSON"
    if isinstance(v, list):
        return ", ".join(map(str, v)) if v else "（無）"
    return str(v)


def write_markdown(f: Facts, path: str, stamp: str):
    lines = [
        "# 論文引用數字真相表", "",
        f"由 `tools/thesis_numbers.py` 於 {stamp} 自資料庫重算產生。",
        "**論文中的任何數字都應與本表一致**；巢狀結構（逐年覆蓋率、逐策略統計）",
        "完整內容見同目錄的 `thesis_numbers.json`。", "",
    ]
    for sec in sorted({r["section"] for r in f.rows}):
        lines += [f"## {sec}", "", "| 項目 | 值 | 備註 |", "| :--- | ---: | :--- |"]
        for r in [x for x in f.rows if x["section"] == sec]:
            lines.append(f"| {r['label']} | {_fmt(r['value'])} | {r['note']} |")
        lines.append("")

    # 逐策略的兩張表在 Markdown 也展開，這是最常被引用的部分
    by_key = {r["key"]: r["value"] for r in f.rows}
    pairs = by_key.get("formation_pairs_by_strategy") or {}
    groups = by_key.get("formation_groups_by_strategy") or {}
    if pairs:
        lines += ["## 附表：各策略形成期統計", "",
                  "`新鮮` 欄為否者，其數字仍是前一版管線的產物（重跑尚未覆寫），不可引用。", "",
                  "| 策略 | 新鮮 | 期數 | 每期配對數 | 填滿率 | 跨產業配對 | "
                  "母體 | 排除比例 | 群數 | 群大小 |",
                  "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for n in pairs:
            p, g = pairs[n], groups.get(n, {})
            lines.append(
                f"| {n} | {'是' if p['fresh'] else '**否**'} | {p['periods']} | "
                f"{p['pairs_per_period']} | "
                f"{p['fill_rate_pct']}% | {p['cross_sector_pct']}% | "
                f"{g.get('universe_median', '-')} | "
                f"{str(g.get('excluded_pct', '-')) + '%' if g else '-'} | "
                f"{g.get('n_groups_median', '-')} | {g.get('group_size_median', '-')} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="重算論文引用數字並輸出真相表")
    ap.add_argument("--all", action="store_true", help="涵蓋全部策略（預設僅主軸 12 條）")
    ap.add_argument("--json-only", action="store_true", help="不寫 Markdown")
    args = ap.parse_args()

    f = Facts()
    collect_config(f)
    collect_price(f)
    collect_fundamentals(f)
    collect_formation(f, args.all)
    collect_result(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    json_path = os.path.join(OUT_DIR, "thesis_numbers.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": stamp, "facts": f.to_json()},
                  fh, ensure_ascii=False, indent=2, default=str)
    print(f"[thesis_numbers] 已寫入 {json_path}（{len(f.rows)} 項）")

    if not args.json_only:
        md_path = os.path.join(OUT_DIR, "thesis_numbers.md")
        write_markdown(f, md_path, stamp)
        print(f"[thesis_numbers] 已寫入 {md_path}")

    missing = [r["label"] for r in f.rows if r["value"] is None]
    if missing:
        print("[thesis_numbers] 尚無資料：" + "、".join(missing))


if __name__ == "__main__":
    main()
