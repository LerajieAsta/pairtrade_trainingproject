#!/usr/bin/env python
"""
專案狀態總覽 —— 一眼看清「哪些策略有配對、哪些有回測、還缺什麼」。
======================================================================

用法（於專案根目錄）：
    python tools/status.py            # 完整狀態表
    python tools/status.py --brief    # 只列缺漏與建議動作

檢查項目：
  1. 資料層：價格 DB、基本面（yF 快照 / FMP Parquet）是否存在且非 LFS pointer
  2. 形成期：formation DB 中各現役策略的配對列數（0 = 待跑 run_formation）
  3. 交易期：result.db 中各現役策略的網格結果數（0 = 待跑 run_trading）
  4. 產出層：投影片是否落後於筆記本（mtime 比較）
最後輸出「建議動作」清單。
"""
import os
import sys
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

BRIEF = "--brief" in sys.argv


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        head = path.read_bytes()[:60]
        return head.startswith(b"version https://git-lfs")
    except OSError:
        return False


def _db_ok(path: Path) -> str:
    if not path.exists():
        return "缺檔"
    if _is_lfs_pointer(path):
        return "LFS pointer（需重建）"
    size = path.stat().st_size
    human = f"{size/1e6:.0f} MB" if size >= 1e6 else f"{size/1e3:.0f} KB"
    return f"OK（{human}）"


def main() -> None:
    from strategies.config import strategies_raw_all, DB_PATH

    active = [s for s in strategies_raw_all if not s.get("formation_only")]
    formation_only = [s for s in strategies_raw_all if s.get("formation_only")]
    suggestions: list[str] = []

    # ── 1. 資料層 ─────────────────────────────────────────────────────
    price_db = Path(DB_PATH)
    yf_db = Path("dataset/fundamental/fundamentals_sp500.db")
    fmp_pq = Path("dataset/fundamental/sp500_pit_2000_2025_monthly.parquet")
    if not BRIEF:
        print("=" * 72)
        print("【資料層】")
        print(f"  價格 DB       {price_db}  →  {_db_ok(price_db)}")
        print(f"  基本面 yF     {yf_db}  →  {_db_ok(yf_db)}")
        print(f"  基本面 FMP    {fmp_pq}  →  "
              f"{'OK' if fmp_pq.exists() else '缺檔'}")
    if not price_db.exists() or _is_lfs_pointer(price_db):
        suggestions.append("價格 DB 異常：執行 fetch/SP500_Tiingo.py 或還原 LFS")
    if not yf_db.exists() or _is_lfs_pointer(yf_db):
        suggestions.append("yF 基本面缺失：python fetch/fundamentals_yfinance.py")
    if not fmp_pq.exists():
        suggestions.append("FMP 基本面缺失：python fetch/fetch_fmp_fundamentals.py")

    # ── 2. 形成期覆蓋 ─────────────────────────────────────────────────
    form_db = Path("formation_data/formation_pairs_sp500_Tiingo.db")
    form_counts: dict[str, int] = {}
    if form_db.exists() and not _is_lfs_pointer(form_db):
        try:
            with sqlite3.connect(f"file:{form_db}?mode=ro", uri=True) as conn:
                for sid, n in conn.execute(
                    "SELECT strategy_id, COUNT(*) FROM formation_pairs GROUP BY strategy_id"
                ):
                    # strategy_id 形如 "<name>_MSR0"
                    base = sid.rsplit("_MSR", 1)[0]
                    form_counts[base] = form_counts.get(base, 0) + n
        except sqlite3.DatabaseError as exc:
            suggestions.append(f"formation DB 無法讀取（{exc}）")
    else:
        suggestions.append("formation DB 缺失/為 pointer：python run_formation.py 重建")

    # 形成來源：借用配對者看 base 名稱
    def _form_key(s: dict) -> str:
        return s.get("formation_strategy_id_base", s["name"])

    # ── 3. 交易期覆蓋 ─────────────────────────────────────────────────
    result_db = Path("results/result.db")
    trade_counts: dict[str, int] = {}
    if result_db.exists():
        try:
            with sqlite3.connect(f"file:{result_db}?mode=ro", uri=True) as conn:
                for m, n in conn.execute(
                    "SELECT METHOD, COUNT(*) FROM strategy_summaries GROUP BY METHOD"
                ):
                    trade_counts[m] = n
        except sqlite3.DatabaseError as exc:
            suggestions.append(f"result.db 無法讀取（{exc}）")
    else:
        suggestions.append("result.db 不存在：python run_trading.py")

    # ── 4. 逐策略狀態表 ───────────────────────────────────────────────
    if not BRIEF:
        print()
        print("【現役策略覆蓋】  形成期配對列數 ｜ 交易期網格結果數")
        print("-" * 72)
    need_formation, need_trading = [], []
    for s in active:
        fkey = _form_key(s)
        fn = form_counts.get(fkey, 0)
        tn = trade_counts.get(s["db_method"], 0)
        flag = ""
        if fn == 0:
            flag = "← 待形成期"
            need_formation.append(fkey)
        elif tn == 0:
            flag = "← 待交易期"
            need_trading.append(s["name"])
        if not BRIEF:
            borrow = f"（借用 {fkey}）" if "formation_strategy_id_base" in s else ""
            print(f"  {s['name']:<44s} {fn:>6} ｜ {tn:>3}  {flag}{borrow if flag else ''}")
    for s in formation_only:
        fn = form_counts.get(s["name"], 0)
        if fn == 0:
            need_formation.append(s["name"])
        if not BRIEF:
            print(f"  {s['name']:<44s} {fn:>6} ｜  —   (formation-only)")

    if need_formation:
        uniq = sorted(set(need_formation))
        suggestions.append(
            f"形成期缺 {len(uniq)} 個：python run_formation.py（{', '.join(uniq[:4])}"
            + ("…" if len(uniq) > 4 else "") + "）")
    if need_trading:
        suggestions.append(
            f"交易期缺 {len(need_trading)} 個：python run_trading.py（{', '.join(need_trading[:4])}"
            + ("…" if len(need_trading) > 4 else "") + "）")

    # ── 5. 投影片新鮮度 ───────────────────────────────────────────────
    stale_slides = []
    for nb in Path("notebooks").rglob("*.ipynb"):
        rel = nb.relative_to("notebooks").with_suffix(".html")
        html = Path("docs/slides") / rel
        if not html.exists() or html.stat().st_mtime < nb.stat().st_mtime:
            stale_slides.append(str(rel))
    if stale_slides:
        suggestions.append(
            f"投影片落後 {len(stale_slides)} 份：cd notebooks && quarto render")

    # ── 6. 建議動作 ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    if suggestions:
        print("【建議動作】")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")
    else:
        print("【狀態】資料、形成期、交易期、投影片全部就緒 ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
