#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回填 `strategy_summaries.Concurrent_Periods`，並修正受它影響的三欄。

背景見 `dev/trading_arch/REVIEW.md` §B。引擎逐策略推導並行期數
（`config.concurrent_periods`），但 `db_utils` 與 `metrics` 兩個讀端
在 2026-08-28 之前都寫死全域 6。兩支臂因此中招：

    Grid (HAN4-MONTHLY)     21/21 → 1 期（讀端當 6）
    Grid (NOGRP-DTW-TW63)   63/21 → 3 期（讀端當 6）

受影響欄位（皆以 `max_pairs = top_n × 並行期數` 為分母／分子）：

    Avg_Utilization    = 日均持倉配對數 / max_pairs
    Ann_Ret_Employed   = 總損益 / (日均動用資金 × 年數)
    Excess_Ret_RF      = 算術年化 − rf × 利用率

`Sharpe_Raw` / `Ann_Ret_Raw` / `MDD_Raw` / `Final_Equity` 由權益序列導出，
**不經 max_pairs，故不受影響**，本工具不動它們。

## 為何不重讀 trade_logs

`result.db` 現為 174 GB，逐策略掃 `trade_logs` 不可行。所幸 `max_pairs`
在三個式子裡都只是**乘法因子**，故修正是精確的封閉解：

    util_new  = util_old  × (6 / conc)          （max_pairs 在分母）
    emp_new   = emp_old   × (conc / 6)          （max_pairs 在分子）
    excess_new = excess_old + rf × (util_old − util_new)
                 （因 ann_ret_arith = excess_old + rf·util_old 與 max_pairs 無關）

`--verify` 會另行以 `trade_logs` 重算數列做抽樣核對（會掃全表，很慢）。
未加該旗標時完全不碰 `trade_logs`。

用法：
    python tools/backfill_concurrency.py --dry-run     # 只報告，不寫入
    python tools/backfill_concurrency.py               # 實際寫入
"""
import argparse
import sqlite3
import sys

sys.path.insert(0, ".")

from strategies.config import (  # noqa: E402
    INITIAL_CAPITAL, RF_ANNUAL, concurrent_periods, strategies_raw)

RESULT_DB = "results/result.db"

#: 讀端在修正前一律假設的並行期數。修正就是把它換成逐策略的真值。
LEGACY_ASSUMED = 6

#: 已自 config 移除、但 result.db 仍有列的臂。
#: 全部為 2026-08-17 前的標準 126/21 設定 —— 明列而非靜默退回預設，
#: 這樣新出現的未知 METHOD 會讓本工具停下來，而不是猜一個數字。
RETIRED = {
    "Grid (AGG-SSD-NOSEC-GI)": 6,   # 2026-08-17 移除（impute_scope="global" 成為基線）
    "F09GI (HDB-BASE)": 6,
    "F09GI (AGG-BASE)": 6,
    "F09GI (KM-BASE)": 6,
}


def build_method_map() -> dict:
    m = {e.get("db_method", e["name"]): concurrent_periods(e.get("params", {}))
         for e in strategies_raw}
    m.update(RETIRED)
    return m


def verify_sample(cur, cmap: dict, n: int = 6) -> None:
    """抽樣以 trade_logs 重算，核對封閉解。會掃全表，很慢。"""
    rows = cur.execute(
        'SELECT _path, METHOD, "TOP N", Final_Equity, Avg_Utilization, '
        "Ann_Ret_Employed, Excess_Ret_RF FROM strategy_summaries "
        f"ORDER BY RANDOM() LIMIT {int(n)}").fetchall()
    print(f"\n抽樣核對（{len(rows)} 列，直接自 trade_logs 重算）：")
    for path, method, top_n_str, final_eq, util, emp, exc in rows:
        n_days, n_open = cur.execute(
            "SELECT COUNT(DISTINCT Date), "
            "SUM(CASE WHEN Position != 0 THEN 1 ELSE 0 END) "
            "FROM trade_logs WHERE strategy_id = ?", (path,)).fetchone()
        if not n_days:
            continue
        top_n = int(str(top_n_str).replace("Top", "").strip())
        max_pairs = max(1, top_n * cmap[method])
        mean_open = (n_open or 0) / n_days
        years = n_days / 252.0
        pnl = final_eq - INITIAL_CAPITAL
        u = mean_open / max_pairs
        avg_emp = mean_open * (INITIAL_CAPITAL / max_pairs)
        e = pnl / avg_emp / years if avg_emp > 0 else 0.0
        x = pnl / INITIAL_CAPITAL / years - RF_ANNUAL * u
        print(f"  {method:<24} ΔUtil={u - util:+.2e}  "
              f"ΔEmp={e - emp:+.2e}  ΔExc={x - exc:+.2e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只報告差異，不寫入")
    ap.add_argument("--verify", action="store_true",
                    help="寫入後抽樣自 trade_logs 重算核對（會掃全表，很慢）")
    ap.add_argument("--db", default=RESULT_DB)
    args = ap.parse_args()

    cmap = build_method_map()
    con = sqlite3.connect(args.db, timeout=60.0)
    cur = con.cursor()

    try:
        cur.execute('ALTER TABLE strategy_summaries ADD COLUMN "Concurrent_Periods" INTEGER;')
        print("已新增 Concurrent_Periods 欄")
    except sqlite3.OperationalError:
        pass   # 欄已存在

    rows = cur.execute(
        'SELECT _path, METHOD, Avg_Utilization, Ann_Ret_Employed, Excess_Ret_RF, '
        '"Concurrent_Periods" FROM strategy_summaries').fetchall()

    unknown = sorted({r[1] for r in rows if r[1] not in cmap})
    if unknown:
        print("以下 METHOD 在 config 與 RETIRED 中皆查無，無法判定並行期數：")
        for m in unknown:
            print(f"  · {m}")
        print("請補進 RETIRED（若為已退役的臂）或確認 config 是否漏了。中止。")
        con.close()
        return 1

    # 冪等性：修正是「把舊值乘上 6/conc」，重跑會再乘一次。
    # 已寫入 Concurrent_Periods 的列即代表修正已套用，只核對不重算。
    done = [r for r in rows if r[5] is not None]
    if done:
        mismatch = [r for r in done if int(r[5]) != cmap[r[1]]]
        print(f"已回填過的列：{len(done)}（跳過，不重複套用修正）")
        if mismatch:
            print(f"⚠ 其中 {len(mismatch)} 列的 Concurrent_Periods 與 config 不符：")
            for r in mismatch[:10]:
                print(f"    {r[1]:<24} 庫內={r[5]}  config={cmap[r[1]]}")
            print("這代表策略設定曾變動而結果未重跑。中止。")
            con.close()
            return 1

    updates, changed = [], []
    for path, method, u_old, emp_old, exc_old, conc_stored in rows:
        conc = cmap[method]
        if conc_stored is not None or conc == LEGACY_ASSUMED:
            updates.append((conc, u_old, emp_old, exc_old, path))
            continue

        scale = LEGACY_ASSUMED / conc            # util 的修正倍數
        u_new = (u_old or 0.0) * scale
        emp_new = (emp_old or 0.0) / scale
        exc_new = (exc_old or 0.0) + RF_ANNUAL * ((u_old or 0.0) - u_new)

        updates.append((conc, u_new, emp_new, exc_new, path))
        changed.append((method, conc, u_old, u_new, emp_old, emp_new, exc_old, exc_new))

    print(f"\n共 {len(rows)} 列；並行期數 ≠ {LEGACY_ASSUMED} 的臂：")
    for m in sorted({r[1] for r in rows if cmap[r[1]] != LEGACY_ASSUMED}):
        print(f"  · {m}  →  {cmap[m]} 期")

    print(f"\n數值有變動的列：{len(changed)}")
    if changed:
        print(f"{'METHOD':<24}{'conc':>5}{'Util 舊':>10}{'Util 新':>10}"
              f"{'Emp 舊':>10}{'Emp 新':>10}{'Exc 舊':>10}{'Exc 新':>10}")
        for c in sorted(changed):
            print(f"{c[0]:<24}{c[1]:>5}{c[2]:>10.4f}{c[3]:>10.4f}"
                  f"{c[4]:>10.4f}{c[5]:>10.4f}{c[6]:>10.4f}{c[7]:>10.4f}")

    # 利用率**可以**略超過 100%：`max_pairs` 是名目槽位數，而期界對齊會讓
    # 少數日子同時跑到 conc+1 期。實測（以形成期窗界在價格日曆上鋪算）：
    #   HAN4-MONTHLY  6,287 個交易日中 13 日重疊 2 期（其餘 1 期）→ 上限 ≈ 1.002
    #   NOGRP-DTW-TW63             13 日重疊 4 期（其餘 3 期）→ 上限 ≈ 1.006
    # 這是 REVIEW.md §D-1「執行期沒有槽位記帳」的另一個徵狀，不是本次的錯。
    # 故門檻取 1.05：足以放行期界溢出，仍能擋下分母整個算錯（那會是 ≥2×）。
    UTIL_CEILING = 1.05
    over = [c for c in changed if c[3] > UTIL_CEILING]
    if over:
        print(f"\n⚠ {len(over)} 列修正後利用率 > {UTIL_CEILING:.0%} —— 分母可能仍然錯。中止，不寫入。")
        for c in over[:10]:
            print(f"    {c[0]:<24} conc={c[1]}  Util {c[2]:.4f} → {c[3]:.4f}")
        con.close()
        return 1

    brim = [c for c in changed if c[3] > 1.0]
    if brim:
        print(f"\n註：{len(brim)} 列利用率略超 100%（最高 {max(c[3] for c in brim):.4f}），"
              f"為期界對齊造成的短暫重疊，非分母錯誤。")

    if args.dry_run:
        print("\n--dry-run：未寫入。")
        con.close()
        return 0

    cur.executemany(
        'UPDATE strategy_summaries SET "Concurrent_Periods" = ?, '
        '"Avg_Utilization" = ?, "Ann_Ret_Employed" = ?, "Excess_Ret_RF" = ? '
        "WHERE _path = ?", updates)
    con.commit()

    n_null = cur.execute(
        'SELECT COUNT(*) FROM strategy_summaries WHERE "Concurrent_Periods" IS NULL'
    ).fetchone()[0]
    n_bad = cur.execute(
        f'SELECT COUNT(*) FROM strategy_summaries WHERE "Avg_Utilization" > {UTIL_CEILING}'
    ).fetchone()[0]
    print(f"\n已寫入 {len(updates)} 列。")
    print(f"  Concurrent_Periods 仍為 NULL：{n_null}（應為 0）")
    print(f"  Avg_Utilization > {UTIL_CEILING:.0%} 的列：{n_bad}（應為 0）")

    if args.verify:
        verify_sample(cur, cmap)

    con.close()
    return 0 if (n_null == 0 and n_bad == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
