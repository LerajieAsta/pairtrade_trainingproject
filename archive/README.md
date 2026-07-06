# Archive 索引

歷史檔案依類型分類封存。**復活策略的方式見 `config_archived_strategies.py` docstring。**

```
archive/
├── config_archived_strategies.py   # 已封存策略的完整 config 條目 + 封存原因與診斷數據
├── notebooks/                      # 舊版探索用筆記本（依開發時期分資料夾）
│   ├── 114/                        #   2025-03~04：最早期原型（pt_step1~3、main 系列）
│   ├── 11505/                      #   2025-05：HDBSCAN/Agglomerative/EG 探索期
│   └── 11506/                      #   2025-06：策略邏輯文件化時期
├── formation/                      # 已封存的形成期模組
│   └── 11506/                      #   HDBSCAN CrossSector/MacroCluster 系列、DTW Pure 等
├── trading/                        # 已封存的交易期模組
│   │                               #   DRL v1–v3（FQI/LSTM，逐日定位動作空間已證偽）、
│   │                               #   Kalman、Pure DTW
├── scripts/                        # 一次性工具腳本
│   │                               #   backfill_literature_metrics、merge_constituents、
│   │                               #   SP500_yf（舊資料抓取）、convert/generate notebooks 等
├── docs/                           # 舊版報告 HTML（1150325~1150527 各週進度）
└── h200/                           # H200 GPU 伺服器相關（2026-07-06 起不再使用）
    │                               #   benchmark_h200、pack_results、setup.sh
```

## 注意事項

- `strategies/formation/` 內仍留有部分非現役模組（HDBSCAN_MultiScale、HDBSCAN_UMAP、
  HDBSCAN_PCA_Loadings、ensemble、ml_pair_quality、ssd_basic），因
  `config_archived_strategies.py` 以模組路徑字串引用它們，復活時直接取消註解即可，
  故未移入 archive。
- 歷史回測結果一律保留在 `results/result.db`，封存不刪數據。
