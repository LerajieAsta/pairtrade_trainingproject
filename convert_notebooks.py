import sys
import io
# 強制設置 std streams 為 utf-8 解決 Windows 中文與 emoji 列印編碼錯誤
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import json
import re
from pathlib import Path

def convert_notebook(ipynb_path: Path, output_py_path: Path):
    """
    將 ipynb 檔案轉換為標準化的 py 模組，並解耦主程式與參數
    """
    print(f"正在轉換: {ipynb_path.name} -> {output_py_path.name}")
    
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
        
    code_lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # 確保有換行
            if isinstance(source, list):
                source = "".join(source)
            if source.strip():
                code_lines.append(source)
                
    full_code = "\n\n# " + "="*70 + "\n" + "\n\n".join(code_lines)
    
    # 1. 將 if __name__ == "__main__": 區塊註釋掉或移除
    # 使用正則表達式尋找 if __name__ == "__main__": 的起始位置
    pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
    match = re.search(pattern, full_code)
    
    main_body_commented = ""
    if match:
        start_idx = match.start()
        # 提取 main 區塊前的程式碼
        pre_main_code = full_code[:start_idx]
        main_code = full_code[start_idx:]
        
        # 將 main 區塊的程式碼逐行註釋掉
        commented_lines = []
        for line in main_code.split("\n"):
            commented_lines.append(f"# {line}" if line.strip() else "")
        main_body_commented = "\n".join(commented_lines)
        base_code = pre_main_code
    else:
        base_code = full_code
        
    # 2. 在模組尾端附加一個標準化的進入點函式 run_strategy
    # 此進入點會動態解析 RollingBacktester 所需參數並執行回測
    entry_point_code = """

# ══════════════════════════════════════════════════════════════════════════════
# 標準化策略進入點接口 (Unified Strategy Entry Point)
# ══════════════════════════════════════════════════════════════════════════════
def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    \"\"\"
    標準化調用接口，接受外部傳入的價格資料與回測參數，完全解耦資料載入 I/O
    \"\"\"
    import inspect
    from pathlib import Path
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取 RollingBacktester 的 __init__ 參數列表
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}
    
    # 動態將外部 params 對應並過濾為該策略 RollingBacktester 支援的參數
    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default
            
    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    
    # 初始化回測引擎
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )
    
    # 執行回測
    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")
"""

    final_py_code = base_code + "\n\n" + "# " + "="*70 + "\n# 原 Main 區塊被自動化註釋解耦如下：\n" + main_body_commented + "\n" + entry_point_code
    
    # 寫入目標檔案
    with open(output_py_path, "w", encoding="utf-8") as f:
        f.write(final_py_code)

def main():
    workspace_dir = Path(__file__).parent
    notebooks_dir = workspace_dir / "notebooks"
    strategies_dir = workspace_dir / "strategies"
    
    # 確保策略輸出資料夾存在
    strategies_dir.mkdir(parents=True, exist_ok=True)
    
    # 走訪所有的 .ipynb 檔案進行重構轉換
    ipynb_files = list(notebooks_dir.glob("*.ipynb"))
    if not ipynb_files:
        print("⚠️ 找不到 notebooks/ 資料夾或底下無 .ipynb 檔案！")
        return
        
    for ipynb_file in ipynb_files:
        output_py_name = ipynb_file.stem + ".py"
        output_py_path = strategies_dir / output_py_name
        convert_notebook(ipynb_file, output_py_path)
        
    # 建立 strategies/__init__.py 檔案使其成為標準套件
    init_file = strategies_dir / "__init__.py"
    with open(init_file, "w", encoding="utf-8") as f:
        f.write("# Strategies Package\n")
            
    print("\n🎉 所有交易策略已成功批量轉換、重構並解耦為 strategies/ 下的 py 模組！")

if __name__ == "__main__":
    main()
