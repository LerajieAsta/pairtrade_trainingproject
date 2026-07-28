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

- `strategies/formation/` 內仍留有部分非現役模組，因 `config_archived_strategies.py`
  以模組路徑字串引用它們，復活時直接取消註解即可，故未移入 archive：
  - 較早封存：`HDBSCAN_MultiScale`、`HDBSCAN_UMAP`、`HDBSCAN_PCA_Loadings`、`ensemble`、
    `ml_pair_quality`、`ssd_basic`
  - 2026-07-24（commit `2fb47b6`「Archive 14 Grid-superseded strategies」）新增封存
    （不再作為獨立策略入口，但見下方 ⚠️ 例外）：
    `HDBSCAN_Cluster_SSD_DTW`、`agglomerative_yF`、`agglomerative_FMP`、
    `agglomerative_sec_pit`、`MST_PartialCorr_Cointegration`
  - 交易端：`strategies/trading/distance_trading.py`（GGR 2006 距離基準，隨 #2 一併封存）
- ⚠️ **`ssd_rolling.py`／`DTW_Cointegration_Paper.py` 並未真正死亡**：兩者不再是獨立策略入口，
  但 `_ranking.py` 仍在執行期動態 `import` 它們的 `Formation` 類別作為排序引擎（`"ssd"` backend
  → `ssd_rolling.Formation`；`"dtw"`／`"ssd_dtw_pca"` backend → `DTW_Cointegration_Paper.Formation`）。
  現行 17 策略每次跑 formation 都會經過這兩支模組，**不可封存/刪除**。
- `strategies/formation/` 現役的中性組裝層：`cluster_formation.py`（組裝器）+
  `_clustering.py` / `_ranking.py` / `_features.py` / `_fundamentals.py` / `_cointegration.py`
  （各自獨立的分群／排序／特徵／基本面／共整合子模組，單向依賴，無循環引用；`_ranking.py`
  另外委派給仍現役的 `ssd_rolling.py`／`DTW_Cointegration_Paper.py`，見上）。
- 歷史回測結果一律保留在 `results/result.db`，封存不刪數據。
