import sys
from pathlib import Path
import os

# 強制 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 將工作區目錄加入 sys.path 以便導入 app.py
sys.path.append(os.getcwd())

# 模擬 streamlit 導入，但因為我們不啟動 UI，只測試邏輯，所以可以 mock 掉 streamlit
class MockStreamlit:
    def __getattr__(self, name):
        # 任何未實現的屬性或方法都回傳一個無害的 Dummy 函數
        def dummy(*args, **kwargs):
            return None
        return dummy
    def cache_data(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def cache_resource(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def fragment(self, func):
        return func

import sys
sys.modules['streamlit'] = MockStreamlit()

# 現在我們可以安全導入 app.py 中的邏輯了！
try:
    import app
    print("✅ Successfully imported app.py!")
    
    # 測試搜尋策略
    strategies = app.scan_strategies("results")
    print(f"📊 Found {len(strategies)} strategy logs in 'results' folder.")
    
    # 印出前幾檔看看解析的 method 欄位是否正確
    hdbscan_count = 0
    ssd_count = 0
    eg_count = 0
    print("\n📋 解析策略檔名範例：")
    for s in strategies:
        dataset, reentry, method, top_n, sl_pct, zwin = app.extract_features_from_path(s)
        if "HDBSCAN" in method:
            hdbscan_count += 1
            if hdbscan_count <= 5:
                print(f"  - 路徑: {s}")
                print(f"    解析為 -> Method: {method}, Top N: {top_n}, Stop Loss: {sl_pct}, Z-Window: {zwin}")
        elif "SSD" in method:
            ssd_count += 1
            if ssd_count <= 2:
                print(f"  - 路徑: {s}")
                print(f"    解析為 -> Method: {method}, Top N: {top_n}, Stop Loss: {sl_pct}, Z-Window: {zwin}")
        elif "EG" in method:
            eg_count += 1
            if eg_count <= 2:
                print(f"  - 路徑: {s}")
                print(f"    解析為 -> Method: {method}, Top N: {top_n}, Stop Loss: {sl_pct}, Z-Window: {zwin}")
                
    print(f"\n📈 各策略大類總數：")
    print(f"  - EG: {eg_count}")
    print(f"  - SSD: {ssd_count}")
    print(f"  - HDBSCAN: {hdbscan_count}")
    
    # 測試對新生成的 4 個 CSV 做核心計算 (calculate_metrics_raw)
    new_csvs = [
        "current/HDBSCAN_NoReEntry/HDBSCAN_UMAP_TradeLogs_Top1_SL5_ZWin20.csv",
        "current/HDBSCAN_NoReEntry/HDBSCAN_PCA_TradeLogs_Top1_SL5_ZWin20.csv",
        "current/HDBSCAN_AE_NoReEntry/HDBSCAN_AE_UMAP_TradeLogs_Top1_SL5_ZWin20.csv",
        "current/HDBSCAN_AE_NoReEntry/HDBSCAN_AE_PCA_TradeLogs_Top1_SL5_ZWin20.csv"
    ]
    
    print("\n🔬 測試加載與計算新策略指標：")
    for csv_path in new_csvs:
        csv_full_path = Path("results") / csv_path
        if csv_full_path.exists():
            print(f"  👉 測試加載 {csv_path} ...")
            metrics = app.calculate_metrics_raw(csv_path)
            if metrics:
                print(f"    [成功] Final Equity: {metrics['Final_Equity']:.2f}, Ann Return: {metrics['Ann_Ret_Raw']*100:.2f}%, Sharpe: {metrics['Sharpe_Raw']:.4f}, MDD: {metrics['MDD_Raw']*100:.2f}%")
                print(f"           RCC: {metrics['RCC_Raw']*100:.2f}%, REC: {metrics['REC_Raw']*100:.2f}%, Entries: {metrics['Entries']}")
            else:
                print(f"    [失敗] 計算指標回傳為空！")
        else:
            print(f"  ❌ 找不到檔案 {csv_path}")
            
except Exception as e:
    print(f"❌ Error occurred during import or testing: {str(e)}")
    import traceback
    traceback.print_exc()
