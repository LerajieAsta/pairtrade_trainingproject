# CONTEXT

本專案的領域詞彙表。**這裡只放「講不清楚就會出錯」的詞**——
程式碼讀得出來的東西不寫，論文章節能查到的不寫。

---

## Selection region（選取區域）

**實際會被交易的那批配對。** 定義是三個條件的交集：

1. 通過形成期的 ADF 篩選（`adf_pass == 1`）
2. 標籤有效（`label_valid` —— 交易窗內曾觸發 `|z| >= entry_z`）
3. 在該期的 SSD 排名前 $k$ 名（$k$ = `top_n`）

**與「候選池」的差別很重要。** 候選池是群內全部配對（每期約 25,000 對，
`NOGRP` 臂達 89,676 對）；選取區域是最終進入交易期的那 $k$ 對。

> **本專案最反覆出現的錯誤，是把候選池上的結論當成選取區域上的結論。**
> 四階梯子在全候選池上的 AUC 由 0.5752 升至 0.6458，統計上極穩健，
> 但在選取區域（$k \le 20$）上完全不轉化——見 `dev/ml_formation/RESULTS.md`。

程式：`dev/ml_formation/selection.py` 的 `selection_region()`。
**篩選與排名是內建的、不可關閉**——這正是這個詞的意思。

### Warm-up（暖身期）

模型分數採前推協定產生，前 36 期（`pi <= 35`）無分數。
`pool.parquet` **沒有 `pi` 欄**，故暖身邊界只有併入 `scores.parquet` 之後才看得見。

⚠ 在 `selection.py` 之前，暖身排除是 `dropna(subset=['score_M1', ...])` 的**副作用**，
沒有名字也沒有人擁有——本輪因此崩潰過一次。
現由 `with_model_scores()` 明確擁有。

---

## Trial universe（試驗宇宙）

Deflated Sharpe Ratio 的 $N$：**為了挑出最終報告的那個策略，總共看過幾個候選。**

兩種口徑，兩者都釘死在 `analysis/regime_cost_dsr_eval.py` 的 `TRIAL_CENSUS`：

| 口徑 | 釘死值 | 含意 |
|:---|---:|:---|
| `method` | 53 | 相異的 `METHOD`，每個代表一次獨立的建模決策 |
| `config` | 1,392 | 全部回測配置（`METHOD` × `top_n` × 停損 × …） |

⚠️ **兩者都是刻意大於實地清點值，不是過期。** 對沖口徑修正（2026-09）之後
`result.db` 只剩 49 個 `METHOD`／1,152 個基準格——MHD 掃描與四支已退役的 METHOD
無法以現行 config 重現而遭刪除。**但那些試驗確實跑過、確實被看過**，
DSR 的 $N$ 計的是「為了挑出最終報告的那個策略，總共看過幾個候選」，
把結果刪掉並不會讓你沒看過它們。維持較大的 $N$ 只會使門檻更嚴。
故 `_trial_specs` 的漂移示警是**預期行為**，不要用「更新常數」消掉它。
（`var_sr` 則相反，已改用修正後的引擎重算——它必須與 Sharpe 同尺度。）

**`method` 是主口徑**：同一 `METHOD` 下的 15 格共用配對、高度相關，
不宜各算一次試驗（偽重複，見論文 5.2.1 其一）。

### 兩件容易搞混的事

**其一，「新增方法」與「新增同方法內的配置」對 DSR 的作用方向相反。**
2026-08-26 第一次清點（44 → 53，新增方法）使 `Grid (NOGRP-DTW)` 的 DSR 由 0.769 降至 0.674；
同日第二次（config 912 → 1392，只新增配置）反而使它升至 0.696。

> 上述三個數字是 **2026-08-26 當時**的，留作方向性的示例，不是現值。
> 對沖口徑修正（2026-09）之後同一支的 DSR 為 **0.738**（$SR_0$ = 0.408）。
> 現值一律以 `results/analysis/breakeven_dsr.csv` 為準。

**其二，`var_sr` 的口徑本身有內在矛盾（未解決）。**
現行定義為「每 `METHOD` 取其**全部參數格的平均** Sharpe，再取橫斷面變異」，
故**在既有方法內新增爛配置會降低門檻**——與 DSR 的懲罰意圖相反。
三種可能口徑各有問題，記於 `dev/IMPASSE.md` §四之二。

---

## Concurrent periods（並行期數）

交易期重疊的層數 —— **逐策略，不是全域常數**：

$$\text{並行期數} = \left\lfloor \frac{\texttt{trading\_window}}{\texttt{rolling\_step}} \right\rfloor$$

多數臂為 126/21 = **6**，但 `Grid HAN4-MONTHLY` 是 21/21 = **1**、
`Grid NOGRP-DTW-TW63` 是 63/21 = **3**。

它決定資金分母：`capital_per_pair = equity / (top_n × 並行期數)`。
凡是利用率、動用資本年化、名目額、break-even，全部經過它。

> **這個量已經錯過三次，全都是同一個形狀：把它當成常數 6。**
> 前兩次是漏乘（`regime_cost_dsr_eval` 2026-08-26、`regime_cost_ew` 2026-08-27），
> 第三次是硬編（`db_utils` 與 `metrics` 的讀端，2026-08-28）。
> 第三次讓 `HAN4-MONTHLY` 的 break-even 報成 **−0.76%**（真值 +0.36%）。

唯一擁有者：`config.concurrent_periods(params)`。
引擎把實際值寫入 `strategy_summaries.Concurrent_Periods`；
讀端一律經 `metrics.concurrent_of(path_key)` 取該值，**取不到就拋錯，不猜**。

⚠ 利用率**可以略超 100%**：`max_pairs` 是名目槽位，執行期沒有記帳，
期界對齊時實際部署會短暫超出（HAN4 於 6,287 個交易日中有 13 日跑 2 期）。
超過 5% 才代表分母錯。詳見 `dev/trading_arch/REVIEW.md` §B、§D-1。

---

## Formation layer（形成期的四層）

`分組 → 篩選 → 排序 → 選取`，宣稱為可獨立替換的四層。

⚠ **實作上這個宣稱只有一半成立。** `strategies/formation/_ranking.py` 名為排序層，
實為三個 `Formation` 建構子的 switch；每個後端各自重做正規化、分組、篩選、選取，
並回傳**不同欄位**的表。詳見架構檢視。

### Ranking backend（排序後端）

`ssd` / `dtw` / `ssd_dtw_pca`（簡寫 SDP）/ `reversal`。

> **`ssd` 排序與相關係數排序是同一個排序。** 在本專案的 z 標準化下
> $\mathrm{SSD} = 2(T-1)(1-\rho)$、$\sigma_s^2 = 1-\rho^2$，
> 實測 5,736 對的 `Spearman(SSD, |β|) = −1.0000`。
> 文獻常見的「距離法 vs 相關係數法」對照，在此實作下**不構成兩個方法**。

---

## Proxy（代理）

以 `capture_frac` 等每筆量估算策略改動的效果，不實跑回測。

$$\text{capture\_frac} = (|z_{\text{進場}}| - |z_{\text{出場}}|) \times \text{unit}, \qquad \text{unit} \propto \sigma_s$$

**代理只能預測「單筆品質」，不能預測「組合報酬」。**
任何改變**進場頻率**或**部署規模**的維度都必須實跑。

已三次獨立確認（`entry_z` 兩次、`trading_window` 一次）。最清楚的一次：
`entry_z` 由 2.0 改 3.0，Profit Factor 上升 2%（代理說對了），
但進場次數 −38%、利用率 −28%，年化因此 −25%。

---

## Pre-registration（預先註冊）

**在跑之前**寫下成功判準，跑完依規則裁決。格式見 `dev/*/PREREGISTRATION.md`（現有 6 份）。

規則：
- 未過閘 → 停在診斷，**不回測、不新增策略**
- **跑了門檻二的回測即進入試驗宇宙，與過閘與否無關**——
  僅停在門檻一、從未回測者才不計入
- 偏離預先註冊要記成「修訂記錄」附加於檔尾，不得改寫原文
