import os as _os

# ==========================================
# 🛑 全局模式切換開關 (True: 快速開發測試 / False: 論文最終完整回測)
# ==========================================
FAST_TEST_MODE = True

if FAST_TEST_MODE:
    print("🚀 啟動【快速測試模式】: 僅回測 2019-2022 年間之科技板塊。")
    # 縮短時間：涵蓋 2020 疫情崩盤與 2022 升息的壓力測試區間
    START_DATE = '2019-01-01'
    END_DATE = '2022-12-31'
    # 縮小股票池：僅測試單一板塊，運算量大幅減少
    TARGET_SECTOR = 'Information Technology'  
else:
    print("🐢 啟動【完整回測模式】: 回測 2000-2025 年間之全市場 S&P 500。")
    # 論文要求的全樣本時間
    START_DATE = '2000-01-01'
    END_DATE = '2025-12-31'
    # 測試全市場 (設為 None 代表不限制單一產業)
    TARGET_SECTOR = None  

# ==========================================
# 📊 共用策略參數 (兩種模式皆適用)
# ==========================================
DB_PATH           = r'..\data\sp500_data.db'
# 產業動態補齊開關與快取路徑
USE_DYNAMIC_SECTORS = True 
IMPUTED_SECTOR_PATH = r'..\data\imputed_sectors.csv'

# 視窗滾動參數
FORMATION_WINDOW = 252   # 形成期 (約一年)
TRADING_WINDOW = 126     # 交易期 (約半年)
ROLLING_WINDOW = 20      # 滾動步長 (約一個月)
MIN_HISTORY_DAYS  = 200

# 配對與統計檢定參數
MAX_PAIRS_PER_TRANCHE = 3  # 決定每個梯隊要執行的最大配對數 (可設為 1 或 3 或其他數字)
COINT_P_VALUE = 0.01     # 嚴格的共整合 p-value 門檻

# 交易執行與資金控管參數
TRANSACTION_COST = 0.0029 # 雙邊交易手續費 (0.29%)

INITIAL_CAPITAL = 30000  # 初始本金
CAPITAL_TRANCHES = 10       # 滾動視窗資金切割份數

# 確保快取檔案的上一層資料夾存在
os.makedirs(os.path.dirname(IMPUTED_SECTOR_PATH), exist_ok=True)