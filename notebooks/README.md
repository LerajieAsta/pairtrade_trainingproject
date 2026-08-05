# notebooks/ — 策略筆記本（Quarto revealjs 投影片）

收斂至論文主軸「**機器學習分群 + 深度學習交易**」。
形成期一層一本、交易期一端一本，跨策略成果集中於 `main_results.ipynb`。

下列順序即 deck 的呈現順序，與 `_quarto.yml`、`_deck_nav.html`、
`docs/appendix.html` 一致（附錄依形成期四層管線排：分組 → 排序 → 交易端）。
**改動順序請四處同步。**

```
notebooks/
├── thesis/                                 # ★ 論文正文五章
│   ├── ch1_introduction.ipynb              # 緒論：兩個命題與三項方法論目標
│   ├── ch2_literature.ipynb                # 文獻探討
│   ├── ch3_methodology.ipynb               # 研究方法：四層架構、學習法、統計檢定
│   ├── ch4_results.ipynb                   # 實證結果
│   └── ch5_conclusion.ipynb                # 結論：三項方法論貢獻 + 實證貢獻
│
├── main_results.ipynb                      # 兩大命題的實證結果與統計檢定（總覽）
│
├── formation/                              # 附錄・形成期（分組 3 本 + 排序 3 本）
│   ├── hdbscan_cluster_pca5.ipynb          # 分組：HDBSCAN（密度式）
│   ├── agglomerative_fundamentals.ipynb    # 分組：Agglomerative（階層式）
│   ├── kmeans_fundamentals.ipynb           # 分組：K-means（分割式，k 對齊 Agglomerative）
│   ├── ssd_rolling.ipynb                   # 排序：SSD（Gatev 2006）
│   ├── dtw_paper_fixed.ipynb               # 排序：DTW（Sakoe-Chiba）
│   └── ssd_dtw_pca_paper_fixed.ipynb       # 排序：SSD-DTW-PCA 融合
│
├── trading/                                # 附錄・交易期（命題 2 的三個端）
│   ├── zscore_trading.ipynb                # Z-Score 狀態機（規則型基準）
│   ├── drl_threshold_trading.ipynb         # DL-THR：全資訊監督回歸（命題 2 主力）
│   └── rl_threshold_trading.ipynb          # RL-THR：部分回饋 + ε-greedy（受控對照）
│
├── comparison.ipynb                        # 現役策略績效總比較（讀 results/result.db）
├── performance_guide.ipynb                 # 指標與檢定的單一權威定義來源
│
├── _quarto.yml                             # revealjs 設定 + render 順序（明列，非 glob）
├── _deck_nav.html                          # 投影片頁首導覽（含同一套順序）
└── slides.scss                             # 主題（34px root、中文字型、表格縮放）
```

## 用詞約定

| 標準詞 | 指涉 |
| :--- | :--- |
| **分組** | 四層架構的第二層，含 GICS 產業對照組 |
| **分群** | 三種 ML 分群演算法本身（HDBSCAN／Agglomerative／K-means） |
| **群** | cluster 這個物件（群內、群數、過小群） |
| **排序層／排序準則** | 第三層：SSD／DTW／SSD-DTW-PCA |
| **統計篩選層** | 第四層：ADF／OU 半衰期／Hurst 三道檢定 |
| **DL-THR** | 現役交易端。全資訊監督回歸——9 個動作的報酬全部反事實回算，MLP + MSE，決策取 argmax。**不是強化學習** |
| **RL-THR** | 受控對照。同選單同狀態，改為只觀測所選動作 + ε-greedy（contextual bandit） |

`result.db` 的 `METHOD` 欄位沿用歷史字串 `Grid (AGG-SSD-DRL)`、`SSD (DRL)`，
其中 `DRL` 指的是 **DL-THR**；表格為求可追溯不改寫，敘述一律用 DL-THR。

主軸之外的筆記本已移至 `archive/notebooks/`，成果仍保留於 `results/result.db`，
並在 `main_results.ipynb` 的附錄表中列示：

| 封存筆記本 | 對應附錄 |
| :--- | :--- |
| `ssd_basic.ipynb` | A：Gatev (2006) 原型復刻 |
| `hdbscan_cluster_pca5_resid.ipynb` | C：因子殘差化特徵（負面結果） |
| `agglomerative_fmp_dg25.ipynb` | D：regime 條件化進場閘門 |
| `distance_trading.ipynb` | 交易端變體（非現役 TRADE_METHOD） |

更早的負面結果（ResidFDR / MST / SEC-PIT Beta）在
`archive/notebooks/negative_results/`；舊版單檔筆記本在 `archive/notebooks/11507/`。

## 渲染投影片

```bash
cd notebooks && quarto render
```

單本即時預覽：

```bash
quarto preview formation/kmeans_fundamentals.ipynb
```

- **圖片縮放**：投影片中 **Alt + 點擊** 任意圖片/表格即可放大（reveal.js zoom plugin）
- `comparison.ipynb` 的表格與圖為執行時動態產生：先
  `jupyter nbconvert --execute --to notebook --inplace comparison.ipynb`
  （或將 `_quarto.yml` 的 `execute.enabled` 改 true）再 render，即可帶入最新 `result.db` 數據
- 文獻標註：各筆記本「參考文獻」頁標明 `ref/` 內對應 PDF；
  標 ⚠ 者為 `ref/` 缺漏、需補充者（目前僅 MacQueen 1967，見 K-means 筆記本）

## 統計檢定的重跑

`main_results.ipynb` 中命題 2 的假設檢定數據由下列指令產出：

```bash
python -m analysis.proposition2_stats
```
