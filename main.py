import sys
import io
import time
import gc
import os
import re
import argparse
import csv
from pathlib import Path
import concurrent.futures

# 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering (即時刷屏)，不關閉底層 Pipe
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# 從重構後的策略模組中引入數據處理器
from strategies.ssd import DataProcessor

# ── DB Profile 對應表：資料庫 ↔ 輸出目錄 ↔ 顯示名稱 ──────────────────
DB_PROFILES = {
    "sp500_Current": {
        "db_path":    "./data/sp500_Current.db",
        "output_root": "./results/current",
        "label":      "S&P 500 現行成分股 (Current)",
    },
    "sp500_Full": {
        "db_path":    "./data/sp500.db",
        "output_root": "./results/full",
        "label":      "S&P 500 完整歷史成分股 (Full)",
    },
}

def get_module_mtime(module_path):
    """精準取得策略模組實體檔案的修改時間，以便偵測代碼編輯"""
    import importlib.util
    try:
        spec = importlib.util.find_spec(module_path)
        if spec and spec.origin:
            return os.path.getmtime(spec.origin)
    except Exception:
        pass
    return 0.0

def check_strategy_completed(config, db_path):
    """
    智慧判定策略是否已經完整回測過。
    條件：
    1. 目錄下存在 backtest_completed.json 標記檔
    2. 當時使用的 db_path 與其 mtime 與目前一致（防資料庫更新）
    3. 當時使用的 strategy_module_mtime 與目前一致（防策略代碼修改）
    4. 回測的網格參數與目前一致（防參數修改）
    """
    import json
    output_dir = config["output_dir"]
    mark_path = os.path.join(output_dir, "backtest_completed.json")
    if not os.path.exists(mark_path):
        return False
        
    try:
        with open(mark_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 1. 檢查資料庫路徑與其修改時間是否相符
        if data.get("db_path") != db_path:
            return False
        current_db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        if abs(data.get("db_mtime", 0.0) - current_db_mtime) > 1.0: # 容差 1 秒
            return False
            
        # 2. 檢查策略模組代碼是否被修改過
        current_module_mtime = get_module_mtime(config["module"])
        if abs(data.get("strategy_module_mtime", 0.0) - current_module_mtime) > 1.0:
            return False
            
        # 3. 檢查回測網格參數是否一致
        if data.get("strategy_params") != config["params"]:
            return False
            
        return True
    except Exception:
        return False

def write_completion_mark(output_dir, db_path, module_path, params):
    """回測成功後寫入標記檔，記錄資料庫、策略代碼與參數狀態，作為斷點續傳依據"""
    import json
    import time
    mark_path = os.path.join(output_dir, "backtest_completed.json")
    try:
        db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        module_mtime = get_module_mtime(module_path)
        
        info = {
            "db_path": db_path,
            "db_mtime": db_mtime,
            "strategy_module_mtime": module_mtime,
            "strategy_params": params,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(mark_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
    except Exception as e:
        sys.stderr.write(f"\n⚠️ 寫入完成標記檔失敗: {e}\n")

class ProgressAwareStdout:
    def __init__(self, log_filepath, progress_dict, strategy_name, total_rolls):
        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)
        self.progress_dict = progress_dict
        self.strategy_name = strategy_name
        self.total_rolls = total_rolls
        self.start_time = time.time()
        
        # 用於匹配策略中輸出的 "處理中：第 XX 期" 或 "▶ 第 XX 期"
        self.pattern = re.compile(r"(?:處理中[：:]|▶)\s*(?:處理中[：:])?\s*(?:第\s*)?(\d+)\s*期")

    def write(self, s):
        self.log_file.write(s)
        
        match = self.pattern.search(s)
        if match:
            try:
                curr_roll = int(match.group(1))
                total = self.total_rolls if self.total_rolls > 0 else 1
                pct = min(100, int(curr_roll / total * 100))
                
                # 保留原有狀態並更新
                current_info = dict(self.progress_dict.get(self.strategy_name, {}))
                current_info.update({
                    "status": "RUNNING",
                    "progress": f"{curr_roll}/{total}",
                    "pct": pct,
                    "msg": f"正在執行第 {curr_roll:02d}/{total:02d} 期回測",
                    "elapsed": time.time() - self.start_time
                })
                # 重新賦值以觸發 Manager 的跨行程同步
                self.progress_dict[self.strategy_name] = current_info
            except Exception:
                pass

    def flush(self):
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def __getattr__(self, name):
        # 修正：避免 log_file 尚未初始化就發生例外時的無限遞迴 RecursionError
        log_file = self.__dict__.get("log_file")
        if log_file is None:
            raise AttributeError(name)
        return getattr(log_file, name)

def worker_task(strategy_config, price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, progress_dict):
    """
    多行程平行運算子任務：在獨立的 CPU 行程中執行單一策略的完整回測與網格搜尋
    """
    import sys
    import importlib
    import gc
    import time
    
    name = strategy_config["name"]
    module_path = strategy_config["module"]
    params = strategy_config["params"]
    output_dir = strategy_config["output_dir"]
    log_path = strategy_config["log_path"]
    
    # 預先精準計算滾動期數
    trading_window = params.get("trading_window", 126)
    rolling_step = params.get("rolling_step", 21)
    roll_start_indices = list(range(local_first_trade_idx, total_days - trading_window + 1, rolling_step))
    total_rolls = len(roll_start_indices)
    
    # 初始化進度狀態為運行中
    progress_dict[name] = {
        "status": "RUNNING",
        "progress": f"0/{total_rolls}",
        "pct": 0,
        "msg": "正在初始化策略模組...",
        "elapsed": 0.0
    }
    
    start_time = time.time()
    
    # 保存原始的 stdout 與 stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    # 替換 stdout 與 stderr 為我們的 ProgressAwareStdout 代理
    progress_stream = ProgressAwareStdout(log_path, progress_dict, name, total_rolls)
    sys.stdout = progress_stream
    sys.stderr = progress_stream
    
    try:
        # 動態載入策略模組
        module = importlib.import_module(module_path)
        
        # 調用策略標準接口執行回測與網格搜尋
        module.run_strategy(
            price_pivot=price_pivot,
            all_dates=all_dates,
            total_days=total_days,
            local_first_trade_idx=local_first_trade_idx,
            sector_mapping=sector_mapping,
            params=params,
            output_dir=output_dir
        )
        
        # 寫入斷點續傳完畢標記檔
        db_path = strategy_config.get("db_path")
        if db_path:
            write_completion_mark(output_dir, db_path, module_path, params)
            
        elapsed = time.time() - start_time
        # 回報成功狀態
        progress_dict[name] = {
            "status": "SUCCESS",
            "progress": "完成",
            "pct": 100,
            "msg": f"回測成功！共跑完 {total_rolls} 期",
            "elapsed": elapsed
        }
        return {"name": name, "status": "SUCCESS", "elapsed": elapsed, "error": None}
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        sys.stderr.write(f"\n❌ [ERROR] 策略: {name} 執行失敗！錯誤原因:\n{err_msg}\n")
        
        elapsed = time.time() - start_time
        err_brief = str(e)
        if "No module named" in err_brief:
            err_brief = f"缺少必要套件: {err_brief.split()[-1]}"
        progress_dict[name] = {
            "status": "FAILED",
            "progress": "失敗",
            "pct": 100,
            "msg": f"❌ 失敗: {err_brief}",
            "elapsed": elapsed
        }
        return {"name": name, "status": "FAILED", "elapsed": elapsed, "error": str(e)}
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            progress_stream.close()
        except:
            pass
        gc.collect()

def draw_dashboard(progress_dict, strategies_config, main_start_time, output_root=""):
    """
    原地渲染漂亮的監控儀表板 (Premium Terminal UI)
    """
    total_lines = len(strategies_config) + 8
    
    # 修正：移除 is_first_draw 邏輯，永遠上移游標 n 行回到頂部以覆寫（搭配 main 的預先佔位）
    sys.stdout.write(f"\033[{total_lines}A")
    
    width = 100
    print("\033[95m" + "═" * width + "\033[0m\033[K")
    print("        \033[93;1m🚀 量化交易配對回測多行程即時監控儀表板 (High-Performance Core Engine) 🚀\033[0m\033[K")
    print("\033[95m" + "═" * width + "\033[0m\033[K")
    
    # 統計狀態
    total = len(strategies_config)
    pending = 0
    running = 0
    success = 0
    failed = 0
    
    for config in strategies_config:
        name = config["name"]
        status_info = progress_dict.get(name, {})
        status = status_info.get("status", "PENDING")
        if status == "PENDING":
            pending += 1
        elif status == "RUNNING":
            running += 1
        elif status == "SUCCESS":
            success += 1
        elif status == "FAILED":
            failed += 1
            
    elapsed = time.time() - main_start_time
    print(f"  📊 \033[1m回測進度\033[0m | 總任務數: {total:<2} | 運行中: \033[96m{running:<2}\033[0m | 成功: \033[92m{success:<2}\033[0m | 失敗: \033[91m{failed:<2}\033[0m | 總引擎耗時: {elapsed:.1f} 秒\033[K")
    print("\033[90m" + "─" * width + "\033[0m\033[K")
    
    # 印出每個策略
    for config in strategies_config:
        name = config["name"]
        info = progress_dict.get(name, {
            "status": "PENDING",
            "progress": "0/0",
            "pct": 0,
            "msg": "等待中...",
            "elapsed": 0.0
        })
        
        status = info["status"]
        pct = info["pct"]
        prog = info["progress"]
        msg = info["msg"]
        task_elapsed = info["elapsed"]
        
        # 狀態燈號與顏色設定
        if status == "PENDING":
            status_str = "\033[90m○ PENDING\033[0m"
        elif status == "RUNNING":
            status_str = "\033[96m● RUNNING\033[0m"
        elif status == "SUCCESS":
            status_str = "\033[92m✓ SUCCESS\033[0m"
        elif status == "FAILED":
            status_str = "\033[91m❌ FAILED \033[0m"
        else:
            status_str = f"\033[37m{status}\033[0m"
            
        # 渲染進度條
        bar_width = 15
        completed = int(bar_width * pct / 100)
        remaining = bar_width - completed
        bar = "\033[94m" + "█" * completed + "\033[90m" + "░" * remaining + "\033[0m"
        
        # 優化：計算並顯示 ETA 估算
        if status == "RUNNING" and pct > 0:
            eta = task_elapsed / pct * (100 - pct)
            eta_str = f"ETA {eta:.0f}s"
        elif status == "SUCCESS":
            eta_str = "Done"
        elif status == "FAILED":
            eta_str = "Err"
        else:
            eta_str = "---"
            
        # 格式化輸出，對齊寬度
        short_name = name[:30]
        print(f"  {status_str:<19} | {short_name:<30} | {bar} {pct:>3}% ({prog:<5}) | {eta_str:<8} | \033[93m{task_elapsed:>5.1f}s\033[0m | \033[37m{msg[:25]:<25}\033[0m\033[K")
        
    print("\033[90m" + "─" * width + "\033[0m\033[K")
    log_dir_desc = f"{output_root}/logs" if output_root else "./results/current/logs"
    print(f"  📁 詳細策略日誌(stdout/stderr)重導向至: \033[36m{log_dir_desc}/策略名稱.log\033[0m\033[K")
    print("\033[95m" + "═" * width + "\033[0m\033[K")
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(
        description="配對交易滾動回測平行化主控程式",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--db", 
        choices=list(DB_PROFILES.keys()),
        default=None,
        help=(
            "指定資料集 Profile（自動對應輸出目錄）:\n" +
            "\n".join(f"  {k:<20} → {v['output_root']}" for k, v in DB_PROFILES.items())
        )
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="直接指定自訂資料庫路徑（將輸出至 --output-root 所指定位置）"
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="直接指定自訂輸出根目錄（需配合 --db-path 使用）"
    )
    parser.add_argument(
        "--workers",
        type=int, default=None,
        help="最大並行行程數（預設為 CPU 核心數）"
    )
    parser.add_argument(
        "--allow-reentry",
        action="store_true",
        default=False,
        help="允許停損後再進場 (預設為 False，輸出後綴為 _NoReEntry；若啟用則為 _ReEntry)"
    )
    parser.add_argument(
        "--use-vol-adjust",
        action="store_true",
        default=False,
        help="啟用 Z-Score 波動率調節 (預設為 False，輸出後綴為 _NoVolAdj；若啟用則為 _VolAdj)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅印出策略清單與路徑設定，不執行回測"
    )
    return parser.parse_args()

def select_db_profile_interactive():
    """終端互動選單，讓使用者用數字鍵選擇資料集"""
    profiles = list(DB_PROFILES.items())
    
    print("\n請選擇要使用的資料集：\n")
    for i, (key, val) in enumerate(profiles):
        print(f"  [{i+1}] {val['label']}")
        print(f"       DB : {val['db_path']}")
        print(f"       輸出: {val['output_root']}\n")
    print(f"  [{len(profiles)+1}] 自訂路徑（手動輸入）")
    
    while True:
        try:
            choice = input(f"\n請輸入編號 (1-{len(profiles)+1}): ").strip()
            if not choice:
                continue
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                key, profile = profiles[idx]
                print(f"\n✅ 已選擇：{profile['label']}")
                return profile["db_path"], profile["output_root"]
            elif idx == len(profiles):
                # 自訂路徑
                db_path = input("請輸入資料庫路徑（如 ./data/my.db）: ").strip()
                output_root = input("請輸入輸出根目錄（如 ./results/custom）: ").strip()
                return db_path, output_root
        except ValueError:
            print("❌ 無效輸入，請重新選擇。")
        except KeyboardInterrupt:
            # 修正：確保 Ctrl+C 優雅中斷程式，而不被當成無效輸入吞掉
            print("\n\n⚠️ 使用者中止，程式結束。")
            sys.exit(0)

def select_reentry_interactive():
    """終端互動選單，讓使用者用數字鍵選擇是否開啟停損後再進場"""
    print("\n請選擇是否允許停損後再進場 (Allow Re-entry)：\n")
    print("  [1] 關閉 (No Re-entry) - 觸發停損後本期不再進場 (預設)")
    print("  [2] 開啟 (Allow Re-entry) - 觸發停損後，若信號符合可再次進場")
    
    while True:
        try:
            choice = input("\n請輸入編號 (1-2, 直接Enter預設為1): ").strip()
            if not choice:
                print("✅ 已選擇：關閉 (No Re-entry)")
                return False
            idx = int(choice)
            if idx == 1:
                print("✅ 已選擇：關閉 (No Re-entry)")
                return False
            elif idx == 2:
                print("✅ 已選擇：開啟 (Allow Re-entry)")
                return True
            else:
                print("❌ 無效輸入，請輸入 1 或 2。")
        except ValueError:
            print("❌ 無效輸入，請輸入 1 或 2。")
        except KeyboardInterrupt:
            print("\n\n⚠️ 使用者中止，程式結束。")
            sys.exit(0)

def select_vol_adjust_interactive():
    """終端互動選單，讓使用者用數字鍵選擇是否開啟 Z-Score 波動率調節"""
    print("\n請選擇是否開啟 Z-Score 波動率調節 (Vol Adj)：\n")
    print("  [1] 關閉 (No Vol Adj) - 使用基準滾動 Z-Score (預設)")
    print("  [2] 開啟 (Allow Vol Adj) - 引入波動率縮放因子調節開倉門檻")
    
    while True:
        try:
            choice = input("\n請輸入編號 (1-2, 直接Enter預設為1): ").strip()
            if not choice:
                print("✅ 已選擇：關閉 (No Vol Adj)")
                return False
            idx = int(choice)
            if idx == 1:
                print("✅ 已選擇：關閉 (No Vol Adj)")
                return False
            elif idx == 2:
                print("✅ 已選擇：開啟 (Allow Vol Adj)")
                return True
            else:
                print("❌ 無效輸入，請輸入 1 或 2。")
        except ValueError:
            print("❌ 無效輸入，請輸入 1 或 2。")
        except KeyboardInterrupt:
            print("\n\n⚠️ 使用者中止，程式結束。")
            sys.exit(0)

def resolve_paths(args):
    """根據 CLI 參數或互動選單解析 DB 路徑與輸出根目錄"""
    # 修正：對稱性檢查，--db-path 與 --output-root 必須同時指定或省略，防範靜默落入互動模式
    if bool(args.db_path) != bool(args.output_root):
        print("❌ [參數錯誤] --db-path 與 --output-root 必須同時指定，或同時省略。")
        sys.exit(1)
        
    # 優先級 1：同時指定 --db-path 與 --output-root（完全自訂）
    if args.db_path and args.output_root:
        return args.db_path, args.output_root
    
    # 優先級 2：指定 --db Profile Key
    if args.db:
        profile = DB_PROFILES[args.db]
        return profile["db_path"], profile["output_root"]
    
    # 優先級 3：互動選單（無任何 CLI 參數時）
    return select_db_profile_interactive()

def main():
    print("="*80, flush=True)
    print("      🚀 配對交易滾動回測平行化控制主程式 (High-Performance Single-I/O Engine) 🚀", flush=True)
    print("="*80, flush=True)
    
    # 解析參數並取得對應路徑
    args = parse_args()
    DB_PATH, OUTPUT_ROOT = resolve_paths(args)
    
    # 決定是否以互動方式選擇 allow_reentry
    # 當沒有指定 --db 且沒有指定 --db-path 時（代表完全為無參數互動模式），且 CLI 沒有給出 --allow-reentry，我們才提問
    allow_reentry = args.allow_reentry
    if args.db is None and args.db_path is None and not args.allow_reentry:
        allow_reentry = select_reentry_interactive()
        
    # 決定是否以互動方式選擇 use_vol_adjust
    # 當沒有指定 --db 且沒有指定 --db-path 時（代表完全為無參數互動模式），且 CLI 沒有給出 --use-vol-adjust，我們才提問
    use_vol_adjust = args.use_vol_adjust
    if args.db is None and args.db_path is None and not args.use_vol_adjust:
        use_vol_adjust = select_vol_adjust_interactive()
    
    # 優化：提前檢查資料庫檔案是否存在，避免 DataProcessor 拋出模糊的資料連結錯誤
    if not os.path.exists(DB_PATH):
        print(f"❌ [錯誤] 找不到資料庫檔案：{DB_PATH}")
        sys.exit(1)
        
    print(f"\n📁 資料庫路徑 : {DB_PATH}", flush=True)
    print(f"📂 輸出根目錄 : {OUTPUT_ROOT}", flush=True)
    
    TABLE_NAME = "Daily_Prices"
    INFO_TABLE = "Constituents"
    TICKER_COL = "Symbol"
    SECTOR_COL = "GICS_Sector"
    
    BACKTEST_START = "2000-01"
    BACKTEST_END = "2025-12"
    FORMATION_WINDOW = 252  # 統一使用 252 天的形成期視窗
    
    # ── 策略與優化網格參數設定 ──────────────────────────────────────
    base_params = {
        "entry_z": 2.0,
        "exit_z": 0.0,
        "formation_window": FORMATION_WINDOW,
        "trading_window": 126,
        "rolling_step": 21,
        "fee_rate": 0.001,
        "slippage_rate": 0.001,
        "initial_capital": 10000,
        "allow_reentry": allow_reentry,
        "zscore_clip": 10.0,
        "min_spread_std": 1e-6,
        "min_tickers_for_pairing": 2,
        "use_dynamic_stop": False,
        "dynamic_stop_z": 3.0,
        "max_sector_ratio": 0.3,
        "portfolio_stop_loss_pct": 0.10,
        "use_vol_adjust": use_vol_adjust,
        
        # 統一網格搜尋參數：所有策略調用相同參數以進行公平、科學的績效比較
        "top_n_list": [5, 10, 20],              # 統一挑選的最優配對組數 (Top N)
        "stop_loss_list": [0, 0.05, 0.15],     # 統一停損比例限制 (0 代表不停損)
        "zscore_window_list": [0, 20, 60],      # 統一 Z-Score 滾動天數視窗 (0 代表累積視窗)
        "use_vol_adjust_list": [use_vol_adjust] # 網格搜尋參數，用以直接比對原本與波動調節之優劣
    }
    
    # 依據 allow_reentry 的設定動態決定目錄命名後綴
    reentry_suffix = "ReEntry" if base_params.get("allow_reentry", False) else "NoReEntry"
    
    print(f"🔄 允許再進場 (allow_reentry) : {base_params['allow_reentry']} (輸出後綴: {reentry_suffix})", flush=True)
    print(f"⚡ 波動率調節 (use_vol_adjust) : {base_params['use_vol_adjust']}", flush=True)
    
    # 動態產生各策略的 output_dir 與 log_path
    strategies_raw = [
        {
            "name": "SSD Basic (基本配對距離)",
            "module": "strategies.ssd_basic",
            "sub_dir": f"SSD_Basic_{reentry_suffix}",
            "params": base_params                  # 直接套用統一的 base_params
        },
        {
            "name": "SSD Rolling (優化殘差配對)",
            "module": "strategies.ssd",
            "sub_dir": f"SSD_{reentry_suffix}",
            "params": base_params                  # 直接套用統一的 base_params
        },
        {
            "name": "EG Cointegration (Engle-Granger 共整合)",
            "module": "strategies.eg",
            "sub_dir": f"EG_{reentry_suffix}",
            "params": {
                **base_params,
                "exit_buffer": 0.05,                   # 出場緩衝門檻，避免 spreads 微幅波動頻繁平倉
                "adf_max_lags": 1,                     # Engle-Granger 第一階段 ADF 檢定之最大滯後期數
                "p_value_threshold": 0.01              # 共整合 ADF 檢定的 p-value 顯著水準門檻
            }
        },
        {
            "name": "HDBSCAN Clustering + UMAP",
            "module": "strategies.HDBSCAN",
            "sub_dir": f"HDBSCAN_UMAP_{reentry_suffix}",
            "params": {
                **base_params,
                "use_dynamic_stop": True,              # 啟用動態 Z-Score 止損
                "reduce_method": "umap",               # 降維演算法類型。建議: ["umap", "pca"] (umap保留局部非線性拓撲佳, pca速度極快)
                "hdbscan_min_cluster_size": 10,        # HDBSCAN群集最少樣本數。建議: [5, 30] (過大會導致分群極少，過小則群體細碎且多噪聲)
                "hdbscan_min_samples": 5,              # HDBSCAN鄰域核心點樣本數。建議: [1, 10] (控制噪聲判定；越大分群越保守，噪聲越多但群集越純)
                "hdbscan_metric": "euclidean",         # 距離度量指標。建議: ["euclidean", "manhattan", "cosine"]
                "umap_n_components": 8,                # UMAP降維目標空間特徵維度。建議: [3, 15] (過低損失過多特徵，過高易有維度災難阻礙分群)
                "umap_n_neighbors": 20,                # UMAP拓撲計算之鄰近點個數。建議: [5, 50] (較小偏向保留局部微觀細節，較大保留全局宏觀結構)
                "umap_min_dist": 0.1,                  # UMAP低維空間之點最小緊湊距離。建議: [0.001, 0.5] (越小點越扎堆有利於分群，越大越分散)
                "umap_random_state": 42,               # UMAP隨機數種子。建議: 固定整數以確保回測與降維流形結果100%可重複
                "adf_max_lags": 1,                     # 配對篩選時共整合ADF檢定最大滯後期數。建議: [1, 5] (過大會損失自由度，日頻通常用1或2)
                "adf_pvalue_threshold": 0.01           # 配對共整合顯著水準門檻。建議: [0.01, 0.05] (越小對平穩性要求越苛刻，0.01代表99%信心拒絕單根)
            }
        },
        {
            "name": "HDBSCAN Clustering + PCA",
            "module": "strategies.HDBSCAN",
            "sub_dir": f"HDBSCAN_PCA_{reentry_suffix}",
            "params": {
                **base_params,
                "use_dynamic_stop": True,              # 啟用動態 Z-Score 止損
                "reduce_method": "pca",                # 降維演算法類型。建議: ["umap", "pca"] (pca速度極快，主要抓取最大方差方向)
                "hdbscan_min_cluster_size": 10,        # HDBSCAN群集最少樣本數。建議: [5, 30] (控制分群規模，決定最少多少檔股票能成一組群集)
                "hdbscan_min_samples": 5,              # HDBSCAN鄰域核心點樣本數. 建議: [1, 10] (數值越小群集邊緣越寬鬆，數值越大則分群要求越精準)
                "hdbscan_metric": "euclidean",         # 距離度量指標。建議: ["euclidean", "cosine"]
                "umap_n_components": 8,                # UMAP降維目標空間特徵維度 (此PCA模式下不被調用)
                "umap_n_neighbors": 20,                # UMAP拓撲計算之鄰近點個數 (此PCA模式下不被調用)
                "umap_min_dist": 0.1,                  # UMAP低維空間之點最小緊湊距離 (此PCA模式下不被調用)
                "umap_random_state": 42,               # UMAP隨機數種子 (此PCA模式下不被調用)
                "adf_max_lags": 1,                     # 配對篩選時共整合ADF檢定最大滯後期數。建議: [1, 5]
                "adf_pvalue_threshold": 0.01           # 配對共整合顯著水準門檻。建議: [0.01, 0.05] (控制開倉 spreads 平穩度的嚴格程度)
            }
        },
        {
            "name": "HDBSCAN Autoencoder + UMAP",
            "module": "strategies.HDBSCAN_Autoencoder",
            "sub_dir": f"HDBSCAN_AE_UMAP_{reentry_suffix}",
            "params": {
                **base_params,
                "use_dynamic_stop": True,              # 啟用動態 Z-Score 止損
                "reduce_method": "umap",               # 降維演算法類型。建議: ["umap", "pca"] (在自編碼器壓縮後的特徵空間再降維)
                "hdbscan_min_cluster_size": 10,        # HDBSCAN群集最少樣本數。建議: [5, 30]
                "hdbscan_min_samples": 2,              # HDBSCAN鄰域核心點樣本數。建議: [1, 10]
                "hdbscan_metric": "euclidean",         # 距離度量指標。建議: ["euclidean", "manhattan"]
                "umap_n_components": 5,                # UMAP降維目標空間特徵維度。建議: [3, 10] (將AE瓶頸層特徵再進行流形降維)
                "umap_n_neighbors": 15,                # UMAP拓撲計算之鄰近點個數。建議: [5, 30] (控制局部聚類緊密程度)
                "umap_min_dist": 0.1,                  # UMAP低維空間之點最小緊湊距離。建議: [0.001, 0.3]
                "umap_random_state": 42,               # UMAP隨機數種子。確保自編碼器+流形降維結果一致可供穩定對比
                "adf_max_lags": 1,                     # 配對篩選時共整合ADF檢定最大滯後期數。建議: [1, 5]
                "adf_pvalue_threshold": 0.01           # 配對共整合顯著水準門檻。建議: [0.01, 0.05]
            }
        },
        {
            "name": "HDBSCAN Autoencoder + PCA",
            "module": "strategies.HDBSCAN_Autoencoder",
            "sub_dir": f"HDBSCAN_AE_PCA_{reentry_suffix}",
            "params": {
                **base_params,
                "use_dynamic_stop": True,              # 啟用動態 Z-Score 止損
                "reduce_method": "pca",                # 降維演算法類型。建議: ["umap", "pca"] (在自編碼器壓縮後使用PCA進行正交方差最大化降維)
                "hdbscan_min_cluster_size": 10,        # HDBSCAN群集最少樣本數。建議: [5, 30]
                "hdbscan_min_samples": 2,              # HDBSCAN鄰域核心點樣本數。建議: [1, 10]
                "hdbscan_metric": "euclidean",         # 距離度量指標。建議: ["euclidean", "cosine"]
                "umap_n_components": 5,                # UMAP降維目標空間特徵維度 (此PCA模式下不被調用)
                "umap_n_neighbors": 15,                # UMAP拓撲計算之鄰近點個數 (此PCA模式下不被調用)
                "umap_min_dist": 0.1,                  # UMAP低維空間之點最小緊湊距離 (此PCA模式下不被調用)
                "umap_random_state": 42,               # UMAP隨機數種子 (此PCA模式下不被調用)
                "adf_max_lags": 1,                     # 配對篩選時共整合ADF檢定最大滯後期數。建議: [1, 5]
                "adf_pvalue_threshold": 0.01           # 配對共整合顯著水準門檻。建議: [0.01, 0.05]
            }
        },
        {
            "name": "HDBSCAN MultiFactor",
            "module": "strategies.HDBSCAN_MultiFactor",
            "sub_dir": f"HDBSCAN_MultiFactor_{reentry_suffix}",
            "params": {
                **base_params,
                "use_dynamic_stop": True,              # 啟用動態 Z-Score 止損
                "hdbscan_min_cluster_size": 10,        # HDBSCAN群集最少樣本數
                "hdbscan_min_samples": 2,              # HDBSCAN鄰域核心點樣本數
                "hdbscan_metric": "euclidean",         # 距離度量指標
                "adf_max_lags": 1,                     # 配對篩選時共整合ADF檢定最大滯後期數
                "adf_pvalue_threshold": 0.01           # 配對共整合顯著水準門檻
            }
        }
    ]
    
    strategies_config = []
    log_dir = f"{OUTPUT_ROOT}/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    for raw in strategies_raw:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw["name"])
        strategies_config.append({
            "name": raw["name"],
            "module": raw["module"],
            "output_dir": f"{OUTPUT_ROOT}/{raw['sub_dir']}",
            "log_path": f"{log_dir}/{safe_name}.log",
            "params": raw["params"],
            "db_path": DB_PATH
        })
        
    # 優化：統一預先建立所有策略輸出子目錄，防止子行程寫入資料時拋出目錄不存在錯誤
    for c in strategies_config:
        os.makedirs(c["output_dir"], exist_ok=True)
        
    if args.dry_run:
        print("\n🔍 [Dry-Run 模式] 僅預覽設定，不執行回測：", flush=True)
        print("-" * 80, flush=True)
        for c in strategies_config:
            print(f"  策略名稱: {c['name']:<35} -> 輸出: {c['output_dir']}", flush=True)
            print(f"                                         日誌: {c['log_path']}", flush=True)
        print("-" * 80, flush=True)
        return
        
    # ── 1. 單次 I/O 資料載入與格式化 ────────────────────────────────────
    print(f"\n[1/5] [I/O 載入] 正在連結 SQLite 資料庫 '{DB_PATH}'...", flush=True)
    start_io_time = time.time()
    
    processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    
    print(f"[2/5] [I/O 載入] 正在讀取 GICS 產業分類對照表 '{INFO_TABLE}'...", flush=True)
    sector_mapping = processor.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)
    
    print(f"[3/5] [I/O 載入] 正在載入並清洗歷史價格資料 (進行 Pivot 矩陣轉換)...", flush=True)
    try:
        price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
            BACKTEST_START, BACKTEST_END, FORMATION_WINDOW
        )
        io_elapsed = time.time() - start_io_time
        print(f"✅ [I/O 載入完成] 數據加載成功！", flush=True)
        print(f"   - 歷史價格天數: {total_days} 天", flush=True)
        print(f"   - 交易標的數量: {price_pivot.shape[1]} 檔", flush=True)
        print(f"   - I/O 處理耗時 : {io_elapsed:.2f} 秒", flush=True)
        print(f"ℹ️ [效能優勢] 子行程將直接使用已載入並清洗完畢的記憶體數據，免除重複讀取資料庫與 Pivot 矩陣轉換的開銷！", flush=True)
    except Exception as e:
        print(f"❌ [嚴重錯誤] 無法加載回測數據：{e}", flush=True)
        sys.exit(1)
        
    # ── 2. 多行程並行調度與回測執行 ──────────────────────────────────────
    print(f"\n[4/5] [行程平行化] 偵測到需要執行的並行任務數: {len(strategies_config)} 個", flush=True)
    
    # 限制並行進程數以防 OOM
    max_workers = args.workers or min(len(strategies_config), os.cpu_count() or 4)
    print(f"[行程平行化] 正在啟動 ProcessPoolExecutor 進程池 (最大並行數: {max_workers})...", flush=True)
    
    # 初始化跨行程共享 Dict
    import multiprocessing
    import threading
    
    # 在 Windows 上啟用 ANSI 支援以支援彩色刷屏
    os.system("")
    
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    
    # 先將所有策略設定為 PENDING 狀態
    for config in strategies_config:
        progress_dict[config["name"]] = {
            "status": "PENDING",
            "progress": "0/0",
            "pct": 0,
            "msg": "排隊等待中...",
            "elapsed": 0.0
        }
        
    main_start_time = time.time()
    
    # 在控制台開闢空間，預先佔位以防重疊
    sys.stdout.write("\n" * (len(strategies_config) + 8))
    sys.stdout.flush()
    
    # 修正：啟動執行緒前加入極短時間延遲，確保佔位輸出已完整刷入終端 buffer，消除競態
    time.sleep(0.05)
    
    stop_dashboard_event = threading.Event()
    
    def dashboard_updater():
        while not stop_dashboard_event.is_set():
            draw_dashboard(progress_dict, strategies_config, main_start_time, output_root=OUTPUT_ROOT)
            time.sleep(0.3)
            
    dashboard_thread = threading.Thread(target=dashboard_updater, daemon=True)
    dashboard_thread.start()
    
    results = []
    
    # ── 3. 智慧斷點續傳：篩選哪些策略已完整跑完，哪些需要真正執行 ─────────────────
    strategies_to_run = []
    for config in strategies_config:
        if check_strategy_completed(config, DB_PATH):
            # 已經完整跑過了，直接更新進度字典為 SUCCESS，並標記為已跳過
            progress_dict[config["name"]] = {
                "status": "SUCCESS",
                "progress": "完成",
                "pct": 100,
                "msg": "✨ 已跳過 (偵測到已有完整回測結果)",
                "elapsed": 0.0
            }
            results.append({
                "name": config["name"],
                "status": "SUCCESS",
                "elapsed": 0.0,
                "error": None
            })
        else:
            strategies_to_run.append(config)
            
    # 重新計算並行進程數以防 OOM
    if strategies_to_run:
        max_workers = args.workers or min(len(strategies_to_run), os.cpu_count() or 4)
    
    # 使用 ProcessPoolExecutor 分發真正需要執行的任務
    if strategies_to_run:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    worker_task,
                    config,
                    price_pivot,
                    all_dates,
                    total_days,
                    local_first_trade_idx,
                    sector_mapping,
                    progress_dict
                ): config for config in strategies_to_run
            }
            
            # 阻塞等待所有任務執行完畢
            for future in concurrent.futures.as_completed(futures):
                strat_config = futures[future]
                strat_name = strat_config["name"]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    results.append({"name": strat_name, "status": "FAILED", "elapsed": 0.0, "error": str(exc)})
                finally:
                    gc.collect()
                
    # 停止背景渲染
    stop_dashboard_event.set()
    dashboard_thread.join(timeout=1.0)
    # 最後做一次最終重繪，確保所有 SUCCESS/FAILED 百分之百顯示正確
    draw_dashboard(progress_dict, strategies_config, main_start_time, output_root=OUTPUT_ROOT)
    
    # ── 3. 自動寫入 CSV 績效摘要報告 (修正：依照策略定義順序排序) ────────
    name_order = {c["name"]: i for i, c in enumerate(strategies_config)}
    results_sorted = sorted(results, key=lambda r: name_order.get(r["name"], 999))
    
    summary_path = f"{OUTPUT_ROOT}/summary.csv"
    try:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "status", "elapsed", "error"])
            writer.writeheader()
            writer.writerows(results_sorted)
        print(f"\n📄 績效摘要已成功存檔至: {summary_path}", flush=True)
    except Exception as e:
        print(f"\n⚠️ [警告] 無法寫入績效摘要 CSV 檔案：{e}", flush=True)

    # ── 4. 回測績效與耗時總結 (依定義順序排序) ───────────────────────────
    total_elapsed = time.time() - main_start_time
    print("\n" + "="*80, flush=True)
    print("                     📊 [5/5] 回測執行績效總結報告 (Summary) 📊", flush=True)
    print("="*80, flush=True)
    print(f" 所有策略並行回測總耗時 : {total_elapsed:.2f} 秒 (約 {total_elapsed/60:.2f} 分鐘)", flush=True)
    
    
    print(f"{'策略名稱':<45} | {'執行狀態':<10} | {'執行時間 (秒)':<12} | {'錯誤訊息'}", flush=True)
    print("-" * 88, flush=True)
    for res in results_sorted:
        err = res["error"] if res["error"] else "無"
        print(f"{res['name']:<45} | {res['status']:<10} | {res['elapsed']:<12.2f} | {err}", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
