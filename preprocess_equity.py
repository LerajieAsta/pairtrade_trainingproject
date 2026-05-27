import os
import pandas as pd
from pathlib import Path

# 定義策略資料夾路徑 (在 results/current 下)
base_dir = Path(r"d:\Unknown\Papper\Code\results\current")
INITIAL_CAPITAL = 10000.0

best_paths = {
    "SSD_Basic": base_dir / "SSD_Basic_ReEntry" / "TradeLogs_Top5_SL0_ZWin0.csv",
    "SSD_OLS": base_dir / "SSD_ReEntry" / "TradeLogs_Top10_SL0_ZWin0.csv",
    "Engle_Granger": base_dir / "EG_NoReEntry" / "EG_TradeLogs_Top10_SL5.csv",
    "HDBSCAN_Handcrafted": base_dir / "HDBSCAN_UMAP_ReEntry" / "HDBSCAN_UMAP_TradeLogs_Top5_SL0_ZWin0_VolAdj.csv",
    "HDBSCAN_Autoencoder": base_dir / "HDBSCAN_AE_UMAP_ReEntry" / "HDBSCAN_AE_UMAP_TradeLogs_Top5_SL0_ZWin0_VolAdj.csv",
    "HDBSCAN_MultiFactor": base_dir / "HDBSCAN_MultiFactor_ReEntry" / "HDBSCAN_MULTIFACTOR_TradeLogs_Top20_SL0_ZWin0_VolAdj.csv"
}

# 使用篩選出的最佳回測曲線進行資料合併
merged_df = None

print("📈 正在合併最佳策略淨值曲線 (極速版)...")
for name, path in best_paths.items():
    if not path.exists():
        print(f"⚠️ Warning: File {path} not found!")
        continue
        
    print(f"   合併 {name} ({path.name})...")
    
    df = pd.read_csv(path, usecols=["Date", "Daily_Delta"])
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 按 Date 分組，Daily_Delta 加總
    daily = df.groupby("Date")["Daily_Delta"].sum().reset_index()
    daily = daily.sort_values("Date").reset_index(drop=True)
    
    # 計算累加 Equity
    daily[name] = INITIAL_CAPITAL + daily["Daily_Delta"].cumsum()
    daily_equity = daily[["Date", name]]
    
    if merged_df is None:
        merged_df = daily_equity
    else:
        merged_df = pd.merge(merged_df, daily_equity, on="Date", how="outer")

# 填補缺失值 (以向前填補 ffill)
merged_df = merged_df.sort_values("Date").ffill().fillna(INITIAL_CAPITAL)

# 儲存到 notebooks/equity_curves.csv
output_path = Path(r"d:\Unknown\Papper\Code\notebooks\equity_curves.csv")
os.makedirs(output_path.parent, exist_ok=True)
merged_df.to_csv(output_path, index=False)
print(f"\n🎉 成功生成包含最優參數組合的 equity_curves.csv！路徑: {output_path} (大小: {output_path.stat().st_size / 1024:.2f} KB)")
