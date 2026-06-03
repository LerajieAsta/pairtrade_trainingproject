"""
配對交易滾動回測平行化主控程式
優化版本 — 修正所有已知邏輯錯誤並提升可維護性
"""

import sys
import time
import gc
import os
import re
import argparse
import csv
import sqlite3
import json
import traceback
import threading
import multiprocessing
import concurrent.futures
from pathlib import Path

# ── 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering ──────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from strategies.ssd import DataProcessor

# ── DB Profile 對應表 ────────────────────────────────────────────────────
DB_PROFILES = {
    "sp500_Current": {
        "db_path":     "./data/sp500_Current.db",
        "output_root": "./results/current",
        "label":       "S&P 500 現行成分股 (Current)",
    },
    "sp500_Full": {
        "db_path":     "./data/sp500.db",
        "output_root": "./results/full",
        "label":       "S&P 500 完整歷史成分股 (Full)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════════════════════════════════════

def get_module_mtime(module_path: str):
    """
    取得策略模組實體檔案的修改時間。
    【修正】找不到模組時回傳 None（原回傳 0.0 導致斷點續傳誤判通過）。
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec(module_path)
        if spec and spec.origin:
            return os.path.getmtime(spec.origin)
    except Exception:
        pass
    return None  # 明確表示「無法取得」，由呼叫端決定如何處理


def check_strategy_completed(config: dict, db_path: str) -> bool:
    """
    智慧判定策略是否已完整回測過（斷點續傳）。
    條件（全部滿足才回傳 True）：
      1. 目錄下存在 backtest_completed.json 標記檔
      2. 記錄的 db_path 與 mtime 與目前一致
      3. 記錄的 strategy_module_mtime 與目前一致
      4. 回測網格參數與目前一致
      5. SQLite result.db 中確實存有該策略的數據

    【修正 1】get_module_mtime 回傳 None 時，保守地回傳 False，不誤判通過。
    【修正 2】rel_dir 計算失敗時，保守地回傳 False，不使用 basename 誤匹配。
    """
    output_dir = config["output_dir"]
    mark_path  = os.path.join(output_dir, "backtest_completed.json")

    if not os.path.exists(mark_path):
        return False

    try:
        with open(mark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. 資料庫路徑與 mtime
        if data.get("db_path") != db_path:
            return False
        current_db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        if abs(data.get("db_mtime", -1.0) - current_db_mtime) > 1.0:
            return False

        # 2. 策略模組 mtime
        current_module_mtime = get_module_mtime(config["module"])
        if current_module_mtime is None:
            # 【修正】無法取得模組時間 → 保守地重跑，不誤判
            return False
        recorded_module_mtime = data.get("strategy_module_mtime")
        if recorded_module_mtime is None:
            return False
        if abs(recorded_module_mtime - current_module_mtime) > 1.0:
            return False

        # 3. 網格參數
        if data.get("strategy_params") != config["params"]:
            return False

        # 4. SQLite 資料庫中確實存有數據
        results_dir = Path("results").resolve()
        try:
            rel_dir = Path(output_dir).resolve().relative_to(results_dir).as_posix()
        except ValueError:
            # 【修正】無法計算相對路徑時，保守地重跑，避免 basename 誤匹配
            return False

        db_file = "results/result.db"
        if not os.path.exists(db_file):
            return False

        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_summaries';"
            )
            if not cursor.fetchone():
                return False
            cursor.execute(
                "SELECT count(*) FROM strategy_summaries WHERE _path LIKE ?;",
                (rel_dir + "/%",),
            )
            if cursor.fetchone()[0] == 0:
                return False

        return True

    except Exception:
        return False


def write_completion_mark(output_dir: str, db_path: str, module_path: str, params: dict) -> None:
    """
    回測成功後寫入標記檔，記錄資料庫、策略代碼與參數狀態。
    【修正】module_mtime 為 None 時仍寫入（記錄為 null），讓下次 check 保守重跑。
    """
    mark_path = os.path.join(output_dir, "backtest_completed.json")
    try:
        db_mtime     = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        module_mtime = get_module_mtime(module_path)  # 可能為 None

        info = {
            "db_path":               db_path,
            "db_mtime":              db_mtime,
            "strategy_module_mtime": module_mtime,   # None 會被序列化為 null
            "strategy_params":       params,
            "completed_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(mark_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    except Exception as e:
        sys.stderr.write(f"\n⚠️ 寫入完成標記檔失敗: {e}\n")


# ════════════════════════════════════════════════════════════════════════════
# ProgressAwareStdout — 子行程 stdout/stderr 代理
# ════════════════════════════════════════════════════════════════════════════

class ProgressAwareStdout:
    """
    攔截子行程的 stdout，解析「第 XX 期」文字並即時更新跨行程進度字典。

    【修正】先設定 self.log_file = None 作為哨兵，避免 __getattr__ 在
    open() 失敗時將真實的 OSError 遮蔽為 AttributeError。
    """

    def __init__(self, log_filepath: str, progress_dict, strategy_name: str, total_rolls: int):
        # 先設定哨兵，確保 __getattr__ 在初始化失敗時不遮蔽例外
        self.log_file      = None
        self.progress_dict = progress_dict
        self.strategy_name = strategy_name
        self.total_rolls   = total_rolls
        self.start_time    = time.time()
        self.pattern       = re.compile(r"(?:處理中[：:]|▶)\s*(?:處理中[：:])?\s*(?:第\s*)?(\d+)\s*期")

        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 若 open() 拋出 OSError，哨兵確保外層能收到真實例外
        self.log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> None:
        self.log_file.write(s)

        match = self.pattern.search(s)
        if match:
            try:
                curr_roll = int(match.group(1))
                total     = self.total_rolls if self.total_rolls > 0 else 1
                pct       = min(100, int(curr_roll / total * 100))

                current_info = dict(self.progress_dict.get(self.strategy_name, {}))
                current_info.update({
                    "status":   "RUNNING",
                    "progress": f"{curr_roll}/{total}",
                    "pct":      pct,
                    "msg":      f"正在執行第 {curr_roll:02d}/{total:02d} 期回測",
                    "elapsed":  time.time() - self.start_time,
                })
                self.progress_dict[self.strategy_name] = current_info
            except Exception:
                pass

    def flush(self) -> None:
        self.log_file.flush()

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.close()

    def __getattr__(self, name: str):
        log_file = self.__dict__.get("log_file")
        if log_file is None:
            raise AttributeError(name)
        return getattr(log_file, name)


# ════════════════════════════════════════════════════════════════════════════
# 子行程工作單元
# ════════════════════════════════════════════════════════════════════════════

def worker_task(
    strategy_config: dict,
    price_pivot,
    all_dates,
    total_days: int,
    local_first_trade_idx: int,
    sector_mapping: dict,
    progress_dict,
) -> dict:
    """
    多行程平行運算子任務：在獨立 CPU 行程中執行單一策略的完整回測。
    """
    import importlib

    name        = strategy_config["name"]
    module_path = strategy_config["module"]
    params      = strategy_config["params"]
    output_dir  = strategy_config["output_dir"]
    log_path    = strategy_config["log_path"]

    # 預先計算滾動期數
    trading_window     = params.get("trading_window", 126)
    rolling_step       = params.get("rolling_step", 21)
    roll_start_indices = list(range(local_first_trade_idx, total_days - trading_window + 1, rolling_step))
    total_rolls        = len(roll_start_indices)

    progress_dict[name] = {
        "status":   "RUNNING",
        "progress": f"0/{total_rolls}",
        "pct":      0,
        "msg":      "正在初始化策略模組...",
        "elapsed":  0.0,
    }

    start_time  = time.time()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    progress_stream = ProgressAwareStdout(log_path, progress_dict, name, total_rolls)
    sys.stdout = progress_stream
    sys.stderr = progress_stream

    try:
        module = importlib.import_module(module_path)
        module.run_strategy(
            price_pivot=price_pivot,
            all_dates=all_dates,
            total_days=total_days,
            local_first_trade_idx=local_first_trade_idx,
            sector_mapping=sector_mapping,
            params=params,
            output_dir=output_dir,
        )

        db_path = strategy_config.get("db_path")
        if db_path:
            write_completion_mark(output_dir, db_path, module_path, params)

        elapsed = time.time() - start_time
        progress_dict[name] = {
            "status":   "SUCCESS",
            "progress": "完成",
            "pct":      100,
            "msg":      f"回測成功！共跑完 {total_rolls} 期",
            "elapsed":  elapsed,
        }
        return {"name": name, "status": "SUCCESS", "skipped": False, "elapsed": elapsed, "error": None}

    except Exception as e:
        err_msg = traceback.format_exc()
        sys.stderr.write(f"\n❌ [ERROR] 策略: {name} 執行失敗！\n{err_msg}\n")

        elapsed   = time.time() - start_time
        err_brief = str(e)
        if "No module named" in err_brief:
            err_brief = f"缺少必要套件: {err_brief.split()[-1]}"

        progress_dict[name] = {
            "status":   "FAILED",
            "progress": "失敗",
            "pct":      100,
            "msg":      f"❌ 失敗: {err_brief}",
            "elapsed":  elapsed,
        }
        return {"name": name, "status": "FAILED", "skipped": False, "elapsed": elapsed, "error": str(e)}

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            progress_stream.close()
        except Exception:
            pass
        gc.collect()


# ════════════════════════════════════════════════════════════════════════════
# 終端儀表板
# ════════════════════════════════════════════════════════════════════════════

# 儀表板固定行數（標題區 + 頁尾區，不含策略行）
# 對應：分隔線 + 標題 + 分隔線 + 統計行 + 分隔線 + 策略行*N + 分隔線 + 日誌說明 + 分隔線
_DASHBOARD_FIXED_LINES = 8


def draw_dashboard(
    progress_dict,
    strategies_config: list,
    main_start_time: float,
    output_root: str = "",
) -> None:
    """
    原地渲染終端監控儀表板。
    【修正 1】total_lines 使用常數 _DASHBOARD_FIXED_LINES，與實際 print 行數嚴格一致。
    【修正 2】偵測終端寬度並截斷超長行，防止自動換行使行數估算失準。
    【修正 3】新增 SKIPPED 狀態顯示，與 SUCCESS 區分。
    """
    n_strategies = len(strategies_config)
    total_lines  = n_strategies + _DASHBOARD_FIXED_LINES

    sys.stdout.write(f"\033[{total_lines}A")

    # 取終端實際寬度，但不超過 100，避免窄視窗換行
    try:
        term_width = min(os.get_terminal_size().columns, 100)
    except OSError:
        term_width = 100

    def line(s: str) -> None:
        """截斷至終端寬度並清除行尾殘留字元"""
        # 移除 ANSI 碼後估算可見長度（簡化版）
        visible = re.sub(r"\033\[[^m]*m", "", s)
        if len(visible) > term_width:
            # 保留可見字元在限制內（含 ANSI 碼），粗略截斷
            s = s[:term_width + (len(s) - len(visible))]
        print(f"{s}\033[K")

    line("\033[95m" + "═" * term_width + "\033[0m")
    line("        \033[93;1m🚀 量化交易配對回測多行程即時監控儀表板 (High-Performance Core Engine) 🚀\033[0m")
    line("\033[95m" + "═" * term_width + "\033[0m")

    # 統計各狀態數量
    counts = {"PENDING": 0, "RUNNING": 0, "SUCCESS": 0, "SKIPPED": 0, "FAILED": 0}
    for config in strategies_config:
        info   = progress_dict.get(config["name"], {})
        status = info.get("status", "PENDING")
        if status == "SUCCESS" and "跳過" in info.get("msg", ""):
            counts["SKIPPED"] += 1
        else:
            counts[status] = counts.get(status, 0) + 1

    elapsed = time.time() - main_start_time
    line(
        f"  📊 \033[1m回測進度\033[0m | 總任務: {n_strategies:<2} | "
        f"運行: \033[96m{counts['RUNNING']:<2}\033[0m | "
        f"成功: \033[92m{counts['SUCCESS']:<2}\033[0m | "
        f"跳過: \033[33m{counts['SKIPPED']:<2}\033[0m | "
        f"失敗: \033[91m{counts['FAILED']:<2}\033[0m | "
        f"耗時: {elapsed:.1f}s"
    )
    line("\033[90m" + "─" * term_width + "\033[0m")

    for config in strategies_config:
        name = config["name"]
        info = progress_dict.get(name, {
            "status": "PENDING", "progress": "0/0", "pct": 0,
            "msg": "等待中...", "elapsed": 0.0,
        })

        status       = info["status"]
        pct          = info["pct"]
        prog         = info["progress"]
        msg          = info["msg"]
        task_elapsed = info["elapsed"]

        if status == "PENDING":
            status_str = "\033[90m○ PENDING\033[0m"
        elif status == "RUNNING":
            status_str = "\033[96m● RUNNING\033[0m"
        elif status == "SUCCESS":
            is_skipped = "跳過" in msg
            status_str = "\033[33m⟳ SKIPPED\033[0m" if is_skipped else "\033[92m✓ SUCCESS\033[0m"
        elif status == "FAILED":
            status_str = "\033[91m❌ FAILED \033[0m"
        else:
            status_str = f"\033[37m{status}\033[0m"

        bar_width = 15
        completed = int(bar_width * pct / 100)
        bar       = "\033[94m" + "█" * completed + "\033[90m" + "░" * (bar_width - completed) + "\033[0m"

        if status == "RUNNING" and pct > 0:
            eta_str = f"ETA {task_elapsed / pct * (100 - pct):.0f}s"
        elif status in ("SUCCESS", "SKIPPED"):
            eta_str = "Done"
        elif status == "FAILED":
            eta_str = "Err"
        else:
            eta_str = "---"

        line(
            f"  {status_str:<19} | {name[:30]:<30} | "
            f"{bar} {pct:>3}% ({prog:<5}) | "
            f"{eta_str:<8} | \033[93m{task_elapsed:>5.1f}s\033[0m | "
            f"\033[37m{msg[:25]:<25}\033[0m"
        )

    line("\033[90m" + "─" * term_width + "\033[0m")
    log_dir_desc = f"{output_root}/logs" if output_root else "./results/current/logs"
    line(f"  📁 詳細日誌重導向至: \033[36m{log_dir_desc}/策略名稱.log\033[0m")
    line("\033[95m" + "═" * term_width + "\033[0m")
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════════════
# CLI 參數解析
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配對交易滾動回測平行化主控程式",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--db",
        choices=list(DB_PROFILES.keys()),
        default=None,
        help=(
            "指定資料集 Profile:\n"
            + "\n".join(f"  {k:<20} → {v['output_root']}" for k, v in DB_PROFILES.items())
        ),
    )
    parser.add_argument("--db-path",     default=None, help="直接指定自訂資料庫路徑")
    parser.add_argument("--output-root", default=None, help="直接指定自訂輸出根目錄（需配合 --db-path）")
    parser.add_argument(
        "--workers",
        type=int, default=None,
        help="最大並行行程數（預設為 1，防止 OOM；RAM 充裕時可指定更大值）",
    )
    parser.add_argument(
        "--max-mem-pct",
        type=float, default=85.0,
        help="RAM 使用率上限 (%%)，超過則暫緩啟動新行程（預設 85.0）",
    )
    parser.add_argument(
        "--allow-reentry",
        action="store_true", default=False,
        help="允許停損後再進場（預設 False）",
    )
    parser.add_argument(
        "--use-vol-adjust",
        action="store_true", default=False,
        help="啟用 Z-Score 波動率調節（預設 False）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅印出策略清單與路徑設定，不執行回測",
    )
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 互動選單
# ════════════════════════════════════════════════════════════════════════════

def _safe_input(prompt: str) -> str:
    """統一處理 KeyboardInterrupt，確保中斷行為一致。"""
    try:
        return input(prompt).strip()
    except KeyboardInterrupt:
        print("\n\n⚠️ 使用者中止，程式結束。")
        sys.exit(0)


def select_db_profile_interactive() -> tuple:
    profiles = list(DB_PROFILES.items())
    print("\n請選擇要使用的資料集：\n")
    for i, (_, val) in enumerate(profiles):
        print(f"  [{i + 1}] {val['label']}")
        print(f"       DB : {val['db_path']}")
        print(f"       輸出: {val['output_root']}\n")
    print(f"  [{len(profiles) + 1}] 自訂路徑（手動輸入）")

    while True:
        choice = _safe_input(f"\n請輸入編號 (1-{len(profiles) + 1}, 直接 Enter 預設為 1): ")
        if not choice:
            _, profile = profiles[0]
            print(f"\n✅ 已選擇（預設）：{profile['label']}")
            return profile["db_path"], profile["output_root"]
        try:
            idx = int(choice) - 1
        except ValueError:
            print("❌ 無效輸入，請重新選擇。")
            continue

        if 0 <= idx < len(profiles):
            _, profile = profiles[idx]
            print(f"\n✅ 已選擇：{profile['label']}")
            return profile["db_path"], profile["output_root"]
        elif idx == len(profiles):
            db_path     = _safe_input("請輸入資料庫路徑（如 ./data/my.db）: ")
            output_root = _safe_input("請輸入輸出根目錄（如 ./results/custom）: ")
            return db_path, output_root
        else:
            print("❌ 無效輸入，請重新選擇。")


def select_reentry_interactive() -> bool:
    """
    【優化】只在完全無 CLI 參數的互動模式下才呼叫，
    消除混用 CLI + 互動造成使用者困惑的問題。
    """
    print("\n請選擇是否允許停損後再進場：\n")
    print("  [1] 關閉 (No Re-entry) — 觸發停損後本期不再進場（預設）")
    print("  [2] 開啟 (Allow Re-entry) — 觸發停損後，若信號符合可再次進場")

    while True:
        choice = _safe_input("\n請輸入編號 (1-2, 直接 Enter 預設為 1): ")
        if not choice or choice == "1":
            print("✅ 已選擇（預設）：關閉 (No Re-entry)")
            return False
        if choice == "2":
            print("✅ 已選擇：開啟 (Allow Re-entry)")
            return True
        print("❌ 無效輸入，請輸入 1 或 2。")


def resolve_paths(args: argparse.Namespace) -> tuple:
    """
    解析 DB 路徑與輸出根目錄。
    【修正】--db-path 與 --output-root 必須同時指定或同時省略。
    """
    if bool(args.db_path) != bool(args.output_root):
        print("❌ [參數錯誤] --db-path 與 --output-root 必須同時指定，或同時省略。")
        sys.exit(1)

    if args.db_path and args.output_root:
        return args.db_path, args.output_root

    if args.db:
        profile = DB_PROFILES[args.db]
        return profile["db_path"], profile["output_root"]

    return select_db_profile_interactive()


def _is_fully_interactive(args: argparse.Namespace) -> bool:
    """判斷是否為完全無 CLI 參數的互動模式。"""
    return args.db is None and args.db_path is None


# ════════════════════════════════════════════════════════════════════════════
# 匯入 CSV → SQLite（全部完成後統一執行一次）
# ════════════════════════════════════════════════════════════════════════════

def import_results_to_db(output_root: str, db_file: str = "results/result.db") -> None:
    """
    【修正】移除策略完成時的即時匯入，改為全部策略跑完後統一執行一次，
    避免 delete_after_import=True 與最終整合步驟造成的重複匯入問題。
    """
    print("\n[I/O 匯入] 正在整合所有回測數據至 SQLite 統一資料庫 (result.db)...", flush=True)
    try:
        from strategies.db_utils import import_all_csvs_in_dir
        import_all_csvs_in_dir(results_dir=output_root, db_path=db_file, delete_after_import=True)
        print("✅ [I/O 匯入] SQLite 統一資料庫更新完成！", flush=True)
    except Exception as e:
        print(f"\n⚠️ [警告] 整合資料庫時發生錯誤: {e}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 80, flush=True)
    print("      🚀 配對交易滾動回測平行化控制主程式 (High-Performance Single-I/O Engine) 🚀", flush=True)
    print("=" * 80, flush=True)

    args = parse_args()
    DB_PATH, OUTPUT_ROOT = resolve_paths(args)

    # ── allow_reentry 決策 ──────────────────────────────────────────────
    # 【修正】只有在完全互動模式（無任何 --db / --db-path）且未給 --allow-reentry
    # 時才互動提問，消除「給了 --db 卻靜默套用預設值」的使用者困惑。
    allow_reentry  = args.allow_reentry
    use_vol_adjust = args.use_vol_adjust

    if _is_fully_interactive(args) and not args.allow_reentry:
        allow_reentry = select_reentry_interactive()
    elif not _is_fully_interactive(args) and not args.allow_reentry:
        # 使用 CLI 模式但未指定 --allow-reentry，明確告知套用預設值
        print("ℹ️  [提示] 未指定 --allow-reentry，套用預設值：關閉 (No Re-entry)", flush=True)

    if not os.path.exists(DB_PATH):
        print(f"❌ [錯誤] 找不到資料庫檔案：{DB_PATH}")
        sys.exit(1)

    print(f"\n📁 資料庫路徑 : {DB_PATH}", flush=True)
    print(f"📂 輸出根目錄 : {OUTPUT_ROOT}", flush=True)

    TABLE_NAME = "Daily_Prices"
    INFO_TABLE = "Constituents"
    TICKER_COL = "Symbol"
    SECTOR_COL = "GICS_Sector"

    BACKTEST_START   = "2000-01"
    BACKTEST_END     = "2025-12"
    FORMATION_WINDOW = 252

    # ── 策略網格參數 ─────────────────────────────────────────────────────
    base_params = {
        "entry_z":                      2.0,
        "exit_z":                       0.0,
        "formation_window":             FORMATION_WINDOW,
        "trading_window":               126,
        "rolling_step":                 21,
        "fee_rate":                     0.001,
        "slippage_rate":                0.001,
        "initial_capital":              10000,
        "allow_reentry":                allow_reentry,
        "zscore_clip":                  10.0,
        "min_spread_std":               1e-6,
        "min_tickers_for_pairing":      2,
        "use_vol_adjust":               use_vol_adjust,
        "max_holding_days":             30,
        # 統一網格搜尋參數
        "top_n_list":                   [5, 10, 20],
        "stop_loss_list":               [0, 0.05, 0.15],
        "zscore_window_list":           [0],
        "use_vol_adjust_list":          [use_vol_adjust],
        "portfolio_stop_loss_pct_list": [0.0],
        "max_sector_ratio_list":        [0.0, 0.30, 0.50],
        "dynamic_stop_z_list":          [0.0],
    }

    reentry_suffix = "ReEntry" if allow_reentry else "NoReEntry"

    print(f"🔄 允許再進場 (allow_reentry) : {allow_reentry} (後綴: {reentry_suffix})", flush=True)
    print(f"⚡ 波動率調節 (use_vol_adjust) : {use_vol_adjust}", flush=True)

    # ── 策略定義（共用 HDBSCAN 參數以減少重複）──────────────────────────
    hdbscan_common = {
        "use_dynamic_stop":         True,
        "hdbscan_min_cluster_size": 30,
        "hdbscan_min_samples":      10,
        "hdbscan_metric":           "euclidean",
        "adf_max_lags":             1,
        "adf_pvalue_threshold":     0.01,
    }

    strategies_raw = [
        {
            "name":    "SSD Basic (基本配對距離)",
            "module":  "strategies.ssd_basic",
            "sub_dir": f"SSD_Basic_{reentry_suffix}",
            "params":  base_params,
        },
        {
            "name":    "SSD Rolling (優化殘差配對)",
            "module":  "strategies.ssd",
            "sub_dir": f"SSD_{reentry_suffix}",
            "params":  base_params,
        },
        {
            "name":    "HDBSCAN Clustering + UMAP",
            "module":  "strategies.HDBSCAN",
            "sub_dir": f"HDBSCAN_UMAP_{reentry_suffix}",
            "params":  {
                **base_params,
                **hdbscan_common,
                "reduce_method":     "umap",
                "umap_n_components": 5,
                "umap_n_neighbors":  40,
                "umap_min_dist":     0.01,
                "umap_random_state": 42,
            },
        },
        {
            "name":    "HDBSCAN Clustering + PCA",
            "module":  "strategies.HDBSCAN",
            "sub_dir": f"HDBSCAN_PCA_{reentry_suffix}",
            "params":  {
                **base_params,
                **hdbscan_common,
                "reduce_method":     "pca",
                "umap_n_components": 5,   # PCA 模式下不調用，保留供介面相容
                "umap_n_neighbors":  40,
                "umap_min_dist":     0.1,
                "umap_random_state": 42,
            },
        },
        {
            "name":    "HDBSCAN CrossSector (跨產業聚類)",
            "module":  "strategies.HDBSCAN_CrossSector",
            "sub_dir": f"HDBSCAN_CrossSector_{reentry_suffix}",
            "params":  {
                **base_params,
                **hdbscan_common,
                "reduce_method":     "umap",
                "umap_n_components": 5,
                "umap_n_neighbors":  40,
                "umap_min_dist":     0.01,
                "umap_random_state": 42,
            },
        },
        {
            "name":    "HDBSCAN MultiFactor",
            "module":  "strategies.HDBSCAN_MultiFactor",
            "sub_dir": f"HDBSCAN_MultiFactor_{reentry_suffix}",
            "params":  {**base_params, **hdbscan_common},
        },
        {
            "name":    "DTW Strategy (論文對照組)",
            "module":  "strategies.dtw_strategy",
            "sub_dir": f"DTW_{reentry_suffix}",
            "params":  {
                **base_params,
                "method": "dtw",
                "adf_pvalue_threshold": 0.01,
            },
        },
        {
            "name":    "SSD+DTW PCA Strategy (論文實驗組)",
            "module":  "strategies.dtw_strategy",
            "sub_dir": f"SSD_DTW_PCA_{reentry_suffix}",
            "params":  {
                **base_params,
                "method": "ssd_dtw_pca",
                "adf_pvalue_threshold": 0.01,
            },
        },
    ]

    # ── 組裝完整 config ──────────────────────────────────────────────────
    log_dir = f"{OUTPUT_ROOT}/logs"
    os.makedirs(log_dir, exist_ok=True)

    strategies_config = []
    for raw in strategies_raw:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw["name"])
        strategies_config.append({
            "name":       raw["name"],
            "module":     raw["module"],
            "output_dir": f"{OUTPUT_ROOT}/{raw['sub_dir']}",
            "log_path":   f"{log_dir}/{safe_name}.log",
            "params":     raw["params"],
            "db_path":    DB_PATH,
        })

    for c in strategies_config:
        os.makedirs(c["output_dir"], exist_ok=True)

    # ── Dry-run ───────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n🔍 [Dry-Run 模式] 僅預覽設定，不執行回測：", flush=True)
        print("-" * 80, flush=True)
        for c in strategies_config:
            print(f"  {c['name']:<40} → 輸出: {c['output_dir']}", flush=True)
            print(f"  {'':40}   日誌: {c['log_path']}", flush=True)
        print("-" * 80, flush=True)
        return

    # ── [1-3] 單次 I/O 資料載入 ──────────────────────────────────────────
    print(f"\n[1/5] 正在連結 SQLite 資料庫 '{DB_PATH}'...", flush=True)
    start_io  = time.time()
    processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)

    print(f"[2/5] 正在讀取 GICS 產業分類表 '{INFO_TABLE}'...", flush=True)
    sector_mapping = processor.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)

    print(f"[3/5] 正在載入並清洗歷史價格資料（Pivot 矩陣轉換）...", flush=True)
    try:
        price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
            BACKTEST_START, BACKTEST_END, FORMATION_WINDOW
        )
        io_elapsed = time.time() - start_io
        print(
            f"✅ 數據加載成功！"
            f"歷史天數: {total_days} 天 | "
            f"標的數: {price_pivot.shape[1]} 檔 | "
            f"耗時: {io_elapsed:.2f}s",
            flush=True,
        )
    except Exception as e:
        print(f"❌ [嚴重錯誤] 無法加載回測數據：{e}", flush=True)
        sys.exit(1)

    # ── [4] 斷點續傳篩選 ─────────────────────────────────────────────────
    print(f"\n[4/5] 正在篩選需要執行的策略（斷點續傳檢查）...", flush=True)

    manager       = multiprocessing.Manager()
    progress_dict = manager.dict()
    results       = []
    strategies_to_run = []

    for config in strategies_config:
        if check_strategy_completed(config, DB_PATH):
            progress_dict[config["name"]] = {
                "status":   "SUCCESS",
                "progress": "完成",
                "pct":      100,
                "msg":      "✨ 已跳過 (偵測到已有完整回測結果)",
                "elapsed":  0.0,
            }
            # 【修正】加入 skipped=True，讓 summary.csv 語意清晰
            results.append({
                "name":    config["name"],
                "status":  "SUCCESS",
                "skipped": True,
                "elapsed": 0.0,
                "error":   None,
            })
            print(f"  ⟳ 跳過（已完成）：{config['name']}", flush=True)
        else:
            progress_dict[config["name"]] = {
                "status":   "PENDING",
                "progress": "0/0",
                "pct":      0,
                "msg":      "排隊等待中...",
                "elapsed":  0.0,
            }
            strategies_to_run.append(config)
            print(f"  ● 排入執行：{config['name']}", flush=True)

    # ── 決定並行數（只設定一次）─────────────────────────────────────────
    # 【修正】原程式在兩處設定 max_workers 且互相覆蓋，此版本統一在此決定。
    # 預設值為 1（保守防 OOM）：price_pivot 會被 pickle 複製給每個子行程，
    # RAM 消耗 = N × price_pivot 大小 + 各策略運算，多策略並行極易 OOM。
    # 若 RAM 充裕，請透過 --workers N 明確指定並行數。
    max_workers = args.workers if args.workers else 1
    max_mem_pct = args.max_mem_pct

    # ── 啟動儀表板 ───────────────────────────────────────────────────────
    os.system("")  # Windows: 啟用 ANSI Escape Code 支援

    # 預先佔位，確保儀表板有足夠行數可原地覆寫
    placeholder_lines = len(strategies_config) + _DASHBOARD_FIXED_LINES
    sys.stdout.write("\n" * placeholder_lines)
    sys.stdout.flush()
    time.sleep(0.05)  # 確保佔位輸出已刷入終端 buffer，消除競態

    main_start_time = time.time()
    stop_event      = threading.Event()

    def dashboard_updater() -> None:
        while not stop_event.is_set():
            draw_dashboard(progress_dict, strategies_config, main_start_time, output_root=OUTPUT_ROOT)
            time.sleep(0.3)

    dashboard_thread = threading.Thread(target=dashboard_updater, daemon=True)
    dashboard_thread.start()

    # ── [5] 多行程平行執行 ───────────────────────────────────────────────
    if strategies_to_run:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures  = {}
            task_idx = 0

            while task_idx < len(strategies_to_run) or futures:
                # A. 收割已完成的行程
                done = [f for f in futures if f.done()]
                for f in done:
                    strat_cfg  = futures.pop(f)
                    strat_name = strat_cfg["name"]
                    try:
                        res = f.result()
                        results.append(res)
                    except Exception as exc:
                        results.append({
                            "name":    strat_name,
                            "status":  "FAILED",
                            "skipped": False,
                            "elapsed": 0.0,
                            "error":   str(exc),
                        })
                    gc.collect()

                # B. 提交新任務（未超並行上限且 RAM 充裕）
                while task_idx < len(strategies_to_run) and len(futures) < max_workers:
                    config = strategies_to_run[task_idx]

                    # RAM 動態節流
                    mem_ok = True
                    try:
                        import psutil
                        mem_pct = psutil.virtual_memory().percent
                        if mem_pct >= max_mem_pct:
                            mem_ok = False
                            curr = dict(progress_dict.get(config["name"], {}))
                            curr.update({
                                "status": "PENDING",
                                "msg":    f"⏳ RAM偏高({mem_pct:.1f}%) 暫緩...",
                            })
                            progress_dict[config["name"]] = curr
                    except ImportError:
                        pass

                    if not mem_ok:
                        break  # 等待下一輪迴圈

                    f = executor.submit(
                        worker_task,
                        config,
                        price_pivot,
                        all_dates,
                        total_days,
                        local_first_trade_idx,
                        sector_mapping,
                        progress_dict,
                    )
                    futures[f] = config
                    task_idx  += 1

                time.sleep(0.1 if futures else 1.0)

    # ── 停止儀表板，最終重繪 ─────────────────────────────────────────────
    stop_event.set()
    dashboard_thread.join(timeout=1.0)
    draw_dashboard(progress_dict, strategies_config, main_start_time, output_root=OUTPUT_ROOT)

    # ── 統一匯入 CSV → SQLite（只執行一次，消除重複匯入風險）────────────
    import_results_to_db(output_root=OUTPUT_ROOT)

    # ── 寫入 summary.csv ─────────────────────────────────────────────────
    name_order     = {c["name"]: i for i, c in enumerate(strategies_config)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["name"], 999))

    summary_path = f"{OUTPUT_ROOT}/summary.csv"
    try:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "status", "skipped", "elapsed", "error"])
            writer.writeheader()
            writer.writerows(results_sorted)
        print(f"\n📄 績效摘要已存檔至: {summary_path}", flush=True)
    except Exception as e:
        print(f"\n⚠️ [警告] 無法寫入績效摘要：{e}", flush=True)

    # ── 終端總結報告 ─────────────────────────────────────────────────────
    total_elapsed = time.time() - main_start_time
    print("\n" + "=" * 80, flush=True)
    print("                     📊 [5/5] 回測執行績效總結報告 (Summary) 📊", flush=True)
    print("=" * 80, flush=True)
    print(f" 總耗時: {total_elapsed:.2f} 秒（約 {total_elapsed / 60:.2f} 分鐘）", flush=True)
    print(
        f"\n{'策略名稱':<45} | {'狀態':<10} | {'跳過':<4} | {'耗時(秒)':<10} | 錯誤訊息",
        flush=True,
    )
    print("-" * 90, flush=True)
    for res in results_sorted:
        err     = res.get("error") or "無"
        skipped = "是" if res.get("skipped") else "否"
        print(
            f"{res['name']:<45} | {res['status']:<10} | {skipped:<4} | {res['elapsed']:<10.2f} | {err}",
            flush=True,
        )
    print("=" * 80, flush=True)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()