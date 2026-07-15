# notebooks/ — 策略筆記本（Quarto revealjs 投影片）

一個策略一個筆記本，形成期與交易期分開放：

```
notebooks/
├── formation/                              # 形成期策略（每形成法一本，共 7 本）
│   ├── ssd_basic.ipynb                     # SSD Basic（基礎原型）
│   ├── ssd_rolling.ipynb                   # SSD Rolling
│   ├── dtw_paper_fixed.ipynb               # DTW
│   ├── ssd_dtw_pca_paper_fixed.ipynb       # SSD-DTW-PCA
│   ├── hdbscan_cluster_pca5.ipynb          # HDBSCAN（分組消融對照組）
│   ├── hdbscan_cluster_pca5_resid.ipynb    # HDBSCAN (殘差)（因子殘差化，命題1 主力）
│   └── agglomerative_fundamentals.ipynb    # Agglomerative Fundamentals（yF 快照 / FMP 逐點，命題1 主力）
│
│   （負面結果 ResidFDR / MST / SEC-PIT Beta 已封存至 archive/notebooks/negative_results/）
├── trading/                                # 交易期策略（每交易模組一本）
│   ├── zscore_trading.ipynb                # Z-Score 狀態機（回歸基準）
│   ├── distance_trading.ipynb              # 距離基準交易（GGR 2006，回歸 vs 距離對照）
│   └── drl_threshold_trading.ipynb         # DRL 門檻選擇模組（DRL THR 系 3 策略）
├── comparison.ipynb                        # 現役策略績效總比較（讀 results/result.db）
├── _quarto.yml                             # revealjs 投影片設定（大字型、zoom、KaTeX）
└── slides.scss                             # 主題（34px root、中文字型、表格縮放）
```

## 渲染投影片

```bash
cd notebooks
quarto render                 # 全部 → ../docs/slides/
quarto render comparison.ipynb
quarto preview formation/ssd_rolling.ipynb
```

- **圖片縮放**：投影片中 **Alt + 點擊** 任意圖片/表格即可放大（reveal.js zoom plugin）
- `comparison.ipynb` 的表格與圖為執行時動態產生：先 `jupyter nbconvert --execute --to notebook --inplace comparison.ipynb`（或 `_quarto.yml` 將 `execute.enabled` 改 true）再 render，即可帶入最新 result.db 數據
- 文獻標註：各筆記本文末「參考文獻」頁標明 `ref/` 內對應 PDF；標 ⚠️ 者為 `ref/` 缺漏、需補充的文獻

舊版單檔筆記本（formation.ipynb / trading.ipynb / *_slides.qmd）已封存至 `archive/notebooks/11507/`。
