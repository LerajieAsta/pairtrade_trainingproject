import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# 強制 Windows 終端機使用 UTF-8 輸出並開啟 Line-buffering (即時刷屏)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# =======================================================================
# 智慧型自適應路徑追溯 (Adaptive Path Escalation)
# =======================================================================
notebooks_dir = Path("notebooks")

if not Path("results").exists() and Path("../results").exists():
    print("💡 Auto-Detection: Running inside tohtml/ folder. Path escalated to parent directory.")
    notebooks_dir = Path("../notebooks")

print("🎯 正在優化 Plotly 排版：移除冗餘內部標題以徹底根除重疊，擴張折線圖高度...")

# 確保目錄存在
os.makedirs(notebooks_dir / "iframe_figures", exist_ok=True)

# 載入淨值數據
df_eq = pd.read_csv(notebooks_dir / "equity_curves.csv")
df_eq["Date"] = pd.to_datetime(df_eq["Date"])

# 策略標籤與顏色對照表
strategy_meta = {
    "SSD_Basic": {"label": "經典 SSD (Basic)", "color": "#4ade80"},
    "SSD_OLS": {"label": "進階 SSD (OLS) 🌟", "color": "#60a5fa"},
    "Engle_Granger": {"label": "Engle-Granger 共整合", "color": "#f87171"},
    "HDBSCAN_Handcrafted": {"label": "HDBSCAN (UMAP)", "color": "#fbd38d"},
    "HDBSCAN_Autoencoder": {"label": "HDBSCAN (AE UMAP)", "color": "#c084fc"},
    "HDBSCAN_MultiFactor": {"label": "HDBSCAN (MF)", "color": "#2b6cb0"}
}

has_current = any(col.endswith("_current") for col in df_eq.columns)
has_full = any(col.endswith("_full") for col in df_eq.columns)

fig = go.Figure()
visibility_current = []
visibility_full = []
total_traces = 0

# 1. 繪製 Current 資料集
if has_current:
    for key, meta in strategy_meta.items():
        col = f"{key}_current"
        if col in df_eq.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_eq["Date"],
                    y=df_eq[col],
                    mode="lines",
                    name=meta["label"],
                    line=dict(color=meta["color"], width=2.5),
                    visible=True
                )
            )
            visibility_current.append(True)
            visibility_full.append(False)
            total_traces += 1

# 2. 繪製 Full 資料集
if has_full:
    for key, meta in strategy_meta.items():
        col = f"{key}_full"
        if col in df_eq.columns:
            is_visible = not has_current
            fig.add_trace(
                go.Scatter(
                    x=df_eq["Date"],
                    y=df_eq[col],
                    mode="lines",
                    name=meta["label"],
                    line=dict(color=meta["color"], width=2.5),
                    visible=is_visible
                )
            )
            visibility_current.append(False)
            visibility_full.append(True)
            total_traces += 1

# 設定基本佈局：
# 1. 將 title 設為空字串，徹底消滅重疊病灶並解鎖上方空間。
# 2. 將 t 邊距由 80px 縮小至 50px，配合 y=1.14 使按鈕精緻貼合頂部。
# 3. 寬度 margin.r 依然保持 280px，給圖例留足橫向呼吸空間。
fig.update_layout(
    font_family="Inter, Outfit, sans-serif",
    hovermode="x unified",
    title="", # 徹底消除內部標題，由簡報 Slide 標題統一接管，100% 根除重疊！
    legend=dict(
        orientation="v", 
        yanchor="middle", 
        y=0.5, 
        xanchor="left", 
        x=1.02,
        font=dict(size=12)
    ),
    margin=dict(l=20, r=280, t=50, b=20),
    xaxis_title="時間",
    yaxis_title="帳戶資產淨值 ($)"
)

# 3. 雙通道 Dropdown 選單按鈕 (當前雙資料集皆存在時)
if has_current and has_full:
    print("💡 雙通道數據就緒，正在建立前端互動式 Dropdown 下拉選單...")
    updatemenus = [
        dict(
            type="dropdown",
            direction="down",
            showactive=True,
            active=0,
            x=0.01,
            xanchor="left",
            y=1.14,
            yanchor="top",
            pad={"r": 10, "t": 10},
            buttons=list([
                dict(
                    args=[
                        {"visible": visibility_current}
                    ],
                    label="📊 顯示：當前回測資料集 (Current)",
                    method="update"
                ),
                dict(
                    args=[
                        {"visible": visibility_full}
                    ],
                    label="🌍 顯示：完整歷史資料集 (Full History)",
                    method="update"
                )
            ]),
            font=dict(family="Inter, sans-serif", size=13),
            bgcolor="#ffffff",
            bordercolor="#cbd5e1"
        )
    ]
    fig.update_layout(updatemenus=updatemenus)

# =======================================================================
# 4. 雙向同步寫入 (Dual-Write Asset Synchronization)
# =======================================================================
primary_path = notebooks_dir / "iframe_figures" / "figure_4.html"
os.makedirs(primary_path.parent, exist_ok=True)
fig.write_html(primary_path, include_plotlyjs=True, full_html=True)
print(f"🎉 圖表主要寫入成功: {primary_path}")

# 如果發生路徑追溯，同步寫入當前測試簡報路徑
if notebooks_dir != Path("notebooks"):
    local_path = Path("notebooks") / "iframe_figures" / "figure_4.html"
    os.makedirs(local_path.parent, exist_ok=True)
    fig.write_html(local_path, include_plotlyjs=True, full_html=True)
    print(f"⚡ [雙向同步] 已複製至當前測試簡報路徑: {local_path}")
