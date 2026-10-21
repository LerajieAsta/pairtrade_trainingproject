# 預先登記：以「獲利潛力」取代 SSD 排序（形成期）

寫定日期：2026-10-21。**本檔在跑之前定稿。**

---

## 一、問題與動機

現行 SSD 排序選出的是**價差波動最小**的配對（$\mathrm{SSD} = 2(T-1)(1-\rho)$，與 $\sigma_s^2 = 1-\rho^2$ 同序；`dev/IMPASSE.md` 障礙一）。
但進場門檻固定為 $|z| > 2$ 時，每筆可捕捉的報酬與價差的報酬尺度成正比，而成本固定（障礙三）；
DL 權重研究（`dev/dl_weights/RESULTS.md`）也顯示，只有預測波動最高的五分之一配對扣成本後平均為正。
文獻上 Do & Faff (2010) 指出形成期價差的**零軸穿越次數**可預測配對交易獲利；Krauss (2017) 批評 SSD 偏好低波動價差。

本案檢驗：**在同一組通過共整合檢定的候選中，改以「扣成本後的獲利潛力」排序，是否優於 SSD？**

> ⚠ 本案的動機部分來自看過資料後的觀察（DL 權重的波動分位結果）。為此判準納入後半期，且不事後調整分數公式。
> 另，`dev/ou_threshold/RESULTS.md` 顯示形成期的回歸速度被高估約 3 倍（選擇偏誤）——零軸穿越次數同樣來自形成期，
> 也可能被高估；本案檢驗的是**相對排序**是否仍有效。

## 二、排序分數（寫死）

候選與 SSD 臂相同：同一分組、同一標準化對數股價 $\tilde P_X = (\ln P_X - \mu_X)/\sigma_X$、
同一避險比率（SSD 臂的無截距 OLS：$\beta = \mathrm{cov}(\tilde P_A, \tilde P_B)/\mathrm{var}(\tilde P_B)$）、
同一價差 $S = \tilde P_A - \beta \tilde P_B$、同一篩選（Engle-Granger ADF $p < 0.05$，`adf_only`）。差別只在排序：

$$\text{POT} = \mathrm{NZC} \times \Big( 2 \cdot \frac{\mathrm{sd}(S)}{D} - 2f \Big),\qquad D = \frac{1}{\sigma_A} + \frac{|\beta|}{\sigma_B}$$

- $\mathrm{NZC}$：形成期價差（去均值後）穿越零軸的次數；
- $\mathrm{sd}(S)/D$：signal 避險下每 1 單位 $z$ 的報酬（`dev/ou_threshold/PREREGISTRATION.md` §二的同一恆等式），
  $2 \cdot \mathrm{sd}(S)/D$ 即進場於 $|z| = 2$、出場於 0 時每筆的毛捕捉；
- $2f$：來回成本（$f$ = 單邊摩擦率 0.29%，按總名目額）。

**通過 ADF 者依 POT 由大到小取前 20**（top_n = 20，與現行相同）；POT ≤ 0 者不入選（毛捕捉不足以付成本），
若某期某群通過且 POT > 0 者不足 20，就只選到可選的數量。
因排序不再單調於 SSD，**群內全部配對都須做 ADF**（不再提前中止）；2026-10-21 實測不分組每期約 6 秒。

## 三、範圍與唯一變因

| 項目 | 規定 |
|:---|:---|
| 分組 | 不分組、GICS、HDBSCAN、Agglomerative、K-means（各分群法與 SSD 臂同參數；K-means `random_state=42`，皆為確定性） |
| 新臂 | `Grid {NOGRP,GICS,HDB,AGG,KM}-POT`，5 條；形成期 252 日、`filter_mode="adf_only"`，其餘參數逐字複製 SSD 臂 |
| 交易端 | Z-Score，與主軸相同；網格 Top {1,3,5,10,20} × 停損 {0, 5, 15}% = 每臂 15 格，共 **75 格** |
| 對照 | 同分組的 `Grid {G}-SSD`（result.db 既有，不重跑） |
| 寫入 | 形成期配對寫入 formation DB（新 strategy_id `Grid {G}-POT_MSR0`）；交易結果寫入 `results/profit_potential.db`（約 10 GB） |

## 四、判準（跑前寫死）

主要指標：逐臂 ΔSharpe = (15 格 Sharpe 平均)_POT − (15 格 Sharpe 平均)_SSD，n = 5。

**判定「獲利潛力排序較佳」須同時滿足：**
1. ≥ 4/5 臂 ΔSharpe > 0；
2. 5 臂中位 ΔSharpe ≥ +0.05；
3. 後半期（2014-01-02 起）中位 ΔSharpe > 0。

**判定「SSD 較佳」須同時滿足：** ≤ 1/5 為正、中位 ≤ −0.05、後半期中位 < 0。其餘 → 無差異。

另報（不入判準）：15 格等權逐日差分（4.3 同程序，5 組 BH）；**組合市場 β**（殘差報酬價差的教訓）；
入選配對的平均 sd(S)/D、NZC、與 SSD 臂入選配對的重疊率；每期入選數不足 20 的比例；換手率；break-even。

**績效不佳的處置**（同 2026-10-21 使用者指示）：若「獲利潛力排序較佳」不成立，`profit_potential.db` 只留摘要、刪除逐日紀錄。

## 五、試驗宇宙

新的排序準則各為新方法：method N **83 → 88**，config +75（庫外）。跑完後重算 var_sr 與 SR₀，更新 `TRIAL_CENSUS`。

## 六、接線驗證（跑網格前必須全過）

- **V1**：既有條目逐項不變；`ranking_backend="ssd"` 的形成期輸出不受新程式影響（以 GICS-SSD 某期重跑形成期，與 formation DB 既有配對相同）。
- **V2**：POT 臂某期的分組標籤與同分群法的 SSD 臂相同（HDB／AGG／KM 各一期）。
- **V3**：以獨立程式重算某期某群全部候選的 ADF、NZC、sd(S)/D、POT 與入選名單，與引擎輸出一致。
- **V4**：POT 臂入選配對的 sd(S)/D 中位數高於 SSD 臂（分數有作用），且每期入選數不足 20 的比例另行報告。

---

## 附記（2026-10-21，接線驗證之後補記；判準與分數公式未改動）

- **發現：HDBSCAN 分組無法以現行程式重現。** 2012-07-11 形成期，以現行程式重跑 `Grid HDB-SSD`，
  與 formation DB 存檔的同群關係有 1,810/19,900（9%）不一致，入選配對只重疊 15/20；Agglomerative 與 K-means
  則完全一致（0 不一致、20/20）。若 POT 臂重新分群，比較會混入分組差異。
- **處置**：`cluster_formation` 新增預設關閉的 `reuse_groups_from`，HDB／AGG／KM 三條 POT 臂改讀 formation_groups
  中對應 SSD 臂、同一形成期的原始群編號，不重新分群（AGG／KM 原本就相同，沿用不改變結果）。
  §三「各分群法與 SSD 臂同參數」據此改為「沿用 SSD 臂存檔的分組」。
- **V1 通過**：GICS-SSD 重跑與 DB 逐名次相同；HDB-SSD 以 `reuse_groups_from` 重跑亦逐名次相同（驗證沿用機制）。
- **V2 通過**：三條 POT 臂的分組與存檔完全相同。
- **V3 通過**：GICS 該期以逐對 `_adf_stat` 獨立重算，441 個通過 ADF 且 POT > 0 的候選，入選名單逐名次相同，POT 最大差 8.9e-16。
- **V4 通過**：該期每單位 z 的報酬中位數 POT 4.84% vs SSD 1.15%；POT 入選 20 對。
