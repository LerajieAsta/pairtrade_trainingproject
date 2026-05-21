import json
import sys
import subprocess
import tempfile
from pathlib import Path

# 強制將主腳本的標準輸出與標準錯誤串流設為 UTF-8 編碼，防止 Windows CP950 終端發生 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def run_notebook(notebook_path, reduce_method="umap", is_ae=False):
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist.")
        return False
        
    print(f"\n==================================================")
    print(f"[Test Engine] Preparing to run {notebook_path.name}")
    print(f"   Reduce Method: {reduce_method.upper()}")
    print(f"==================================================")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            code_lines.extend(cell.get("source", []))
            code_lines.append("\n\n")
            
    code_str = "".join(code_lines)
    
    # ── 1. 修改為極簡網格參數，以利於在 10 秒內跑完回測 ──
    code_str = code_str.replace("TOP_N_LIST         = [1, 5, 10, 20]", "TOP_N_LIST         = [1]")
    code_str = code_str.replace("STOP_LOSS_LIST     = [0, 0.05, 0.1, 0.15]", "STOP_LOSS_LIST     = [0.05]")
    code_str = code_str.replace("ZSCORE_WINDOW_LIST = [0, 20, 40, 60]", "ZSCORE_WINDOW_LIST = [20]")
    
    # ── 2. 動態覆寫降維參數 ──
    if "reduce_method=getattr(self, \"reduce_method\", \"umap\")" in code_str:
        # 修改呼叫的預設值，或直接注入
        pass
        
    # ── 3. 動態注入 Main 區塊的參數 ──
    # 我們可以在 main 函數執行前，動態把參數改掉
    # 尋找 engine = RollingBacktester( 這一行
    target_engine_call = "    engine = RollingBacktester(\n"
    if target_engine_call in code_str:
        replacement = (
            f"    engine = RollingBacktester(\n"
            f"        reduce_method=\"{reduce_method}\",\n"
        )
        code_str = code_str.replace(target_engine_call, replacement)
        
    # ── 4. 修改資料庫與輸出目錄為絕對路徑，防止臨時目錄運行時路徑錯亂 ──
    abs_db_path = str(Path("data/SP500_Current.db").resolve()).replace("\\", "/")
    code_str = code_str.replace("../data/sp500_Current.db", abs_db_path)
    
    # ── 5. 將輸出目錄改為絕對路徑 ──
    abs_output_dir = str(Path("results/current/HDBSCAN_NoReEntry").resolve()).replace("\\", "/")
    code_str = code_str.replace("../results/current/HDBSCAN_NoReEntry", abs_output_dir)
    
    abs_ae_output_dir = str(Path("results/current/HDBSCAN_AE_NoReEntry").resolve()).replace("\\", "/")
    code_str = code_str.replace("../results/current/HDBSCAN_AE_NoReEntry", abs_ae_output_dir)
    
    # 創建臨時 Python 檔案
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
        temp_file.write(code_str)
        temp_path = Path(temp_file.name)
        
    try:
        # 強制指定 UTF-8 輸出編碼，徹底消除 Windows CP950 終端上的 UnicodeEncodeError 錯誤
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # 啟動獨立 Python 進程運行回測
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            encoding="utf-8",
            env=env
        )
        
        # 印出 stdout 與 stderr
        if result.stdout:
            print("[STDOUT]")
            print(result.stdout)
        if result.stderr:
            print("[STDERR]", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            
        if result.returncode == 0:
            print(f"[SUCCESS] {notebook_path.name} ({reduce_method.upper()}) executed successfully!")
            return True
        else:
            print(f"[FAILED] {notebook_path.name} ({reduce_method.upper()}) failed with return code {result.returncode}")
            return False
    finally:
        # 刪除臨時檔案
        if temp_path.exists():
            temp_path.unlink()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", action="store_true", help="Run PCA variant of HDBSCAN")
    parser.add_argument("--umap", action="store_true", help="Run UMAP variant of HDBSCAN")
    parser.add_argument("--ae", action="store_true", help="Run Autoencoder model of HDBSCAN")
    args = parser.parse_args()
    
    # 預設執行所有測試
    run_all = not (args.pca or args.umap or args.ae)
    
    success = True
    
    if run_all or args.umap:
        # 執行 HDBSCAN - UMAP (預設)
        res = run_notebook("notebooks/HDBSCAN.ipynb", reduce_method="umap")
        success = success and res
        
    if run_all or args.pca:
        # 執行 HDBSCAN - PCA
        res = run_notebook("notebooks/HDBSCAN.ipynb", reduce_method="pca")
        success = success and res
        
    if run_all or args.ae:
        # 執行 HDBSCAN_Autoencoder (UMAP 預設)
        res = run_notebook("notebooks/HDBSCAN_Autoencoder.ipynb", reduce_method="umap", is_ae=True)
        success = success and res
        
        # 執行 HDBSCAN_Autoencoder (PCA 對照)
        res = run_notebook("notebooks/HDBSCAN_Autoencoder.ipynb", reduce_method="pca", is_ae=True)
        success = success and res
        
    if success:
        print("\n[SUCCESS] All requested backtests completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Some backtests failed. Please check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
