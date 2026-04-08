# 1. 從資料庫載入原始資料
prices_raw, sector_info = load_data_from_db(DB_PATH, START_DATE, END_DATE)

# 2. 攔截並補齊 Unknown 產業分類 (傳入全域開關與路徑)
sector_info = fix_unknown_sectors(
    sector_info, 
    use_dynamic=USE_DYNAMIC_SECTORS, 
    save_path=IMPUTED_SECTOR_PATH
)

# 3. 執行原有的價格矩陣轉換與 VIX 融合
price_pivot, vix_features = preprocess_prices(prices_raw)
sector_map = sector_info.set_index('ticker')['sector'].to_dict()

# 4. 顯示結果
print(pd.Series(sector_map).value_counts().head(10))