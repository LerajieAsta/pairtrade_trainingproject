#!/usr/bin/env python
"""
重跑前一鍵歸檔工具 —— 保留舊版結果供比較／回滾，而非就地覆蓋。
======================================================================

用法：
  python snapshot_run.py [tag] [--summary-only] [--with-csv] [--copy]

  tag             歸檔標籤（預設今日日期）。最終檔名：result_<YYYYMMDD>_<tag>.db
  --summary-only  只匯出 strategy_summaries 輕量 CSV，不搬移 result.db（純對照用）
  --with-csv      一併歸檔 results/<dataset>/ 的 Trade Log CSV 目錄
  --copy          複製而非搬移 result.db（較慢、需額外空間；預設搬移＝同碟瞬間）

行為：
  1. 一律先把 strategy_summaries（約 200 列、數十 KB）匯出成
     results/archive/summary_<stamp>.csv —— 日後不必保留整顆 DB 即可比較各版績效。
  2. 非 --summary-only 時，將 result.db 搬移（或複製）至
     results/archive/result_<stamp>.db，並清掉其 -wal/-shm 邊車檔。
     搬移後 result.db 不存在 → 下次 run_trading 會重建全新 DB（全量重跑情境）。

注意：
  - 只想「選擇性重跑」某幾個策略時，請勿用本工具搬走 DB（否則其餘策略也要重算）；
    改用  STRATEGIES_SLICE="i:j" python run_trading.py  ，result.db 會逐 config 覆寫。
  - 續傳與儀表板均以 result.db 為準；CSV 為可選（config.WRITE_TRADE_CSV）。
"""
import os
import sys
import shutil
import sqlite3
from datetime import datetime

RESULTS_DIR = "results"
ARCHIVE_DIR = os.path.join(RESULTS_DIR, "archive")
DB_PATH = os.path.join(RESULTS_DIR, "result.db")


def _dataset_subdir() -> str:
    """由 config 的 DB_PATH 對應 output_root，取得資料集子目錄名（如 'tiingo'）。"""
    try:
        from strategies.config import DB_PATH as CFG_DB, DB_PROFILES
        for prof in DB_PROFILES.values():
            if os.path.normcase(os.path.abspath(prof["db_path"])) == os.path.normcase(os.path.abspath(CFG_DB)):
                return os.path.basename(prof["output_root"].rstrip("/"))
    except Exception:
        pass
    # 退回：results/ 下唯一的資料集目錄（排除保留名）
    reserved = {"archive", "analysis", "logs"}
    for d in sorted(os.listdir(RESULTS_DIR)):
        p = os.path.join(RESULTS_DIR, d)
        if os.path.isdir(p) and d not in reserved:
            return d
    return "tiingo"


def main():
    args = sys.argv[1:]
    summary_only = "--summary-only" in args
    with_csv = "--with-csv" in args
    do_copy = "--copy" in args
    tag_args = [a for a in args if not a.startswith("--")]
    tag = tag_args[0] if tag_args else ""
    stamp = datetime.now().strftime("%Y%m%d") + (f"_{tag}" if tag else "")

    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        print(f"⚠️ 找不到 {DB_PATH}，無可歸檔內容。")
        return

    # 1. 匯出輕量 summary（一律執行）
    summary_csv = os.path.join(ARCHIVE_DIR, f"summary_{stamp}.csv")
    try:
        import pandas as pd
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30.0)
        df = pd.read_sql_query("SELECT * FROM strategy_summaries", con)
        con.close()
        df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 匯出 summary：{summary_csv}（{len(df)} 列）")
    except Exception as e:
        print(f"⚠️ 匯出 summary 失敗：{e}")

    if summary_only:
        print("（--summary-only：未搬移 result.db）")
        return

    # 2. 歸檔 result.db（+ 清邊車檔）
    dest_db = os.path.join(ARCHIVE_DIR, f"result_{stamp}.db")
    if os.path.exists(dest_db):
        print(f"⚠️ {dest_db} 已存在，改用時間戳避免覆蓋。")
        dest_db = os.path.join(ARCHIVE_DIR, f"result_{stamp}_{datetime.now():%H%M%S}.db")
    sz_gb = os.path.getsize(DB_PATH) / 1e9
    if do_copy:
        print(f"📦 複製 result.db（{sz_gb:.1f}GB）→ {dest_db} …")
        shutil.copy2(DB_PATH, dest_db)
    else:
        print(f"📦 搬移 result.db（{sz_gb:.1f}GB）→ {dest_db} …")
        shutil.move(DB_PATH, dest_db)
        for sc in (DB_PATH + "-wal", DB_PATH + "-shm"):
            if os.path.exists(sc):
                os.remove(sc)
    print(f"✅ result.db 已歸檔於 {dest_db}")

    # 3. 可選：歸檔 Trade Log CSV 目錄
    if with_csv:
        ds = _dataset_subdir()
        src = os.path.join(RESULTS_DIR, ds)
        if os.path.isdir(src):
            dest_csv = os.path.join(ARCHIVE_DIR, f"{ds}_{stamp}")
            print(f"📦 搬移 CSV 目錄 {src} → {dest_csv} …")
            shutil.move(src, dest_csv)
            print(f"✅ CSV 已歸檔於 {dest_csv}")
        else:
            print(f"（--with-csv：找不到 {src}，略過）")

    print("\n下一步：")
    if do_copy:
        print("  result.db 為複製歸檔，工作用 DB 仍在原處。若要全量重跑請設 FORCE_RERUN=True。")
    else:
        print("  result.db 已移走 → 直接 `python run_trading.py` 會重建全新 DB（全量重跑）。")
    print("  只想重跑部分策略時改用：STRATEGIES_SLICE=\"i:j\" python run_trading.py")


if __name__ == "__main__":
    main()
