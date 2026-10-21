# 預先登記：OU 解析最適門檻（DL-THR 的解析式對照）

寫定日期：2026-10-21。**本檔在跑之前定稿。**

---

## 一、問題

DL-THR（4.3 節）以監督式迴歸，從 9 個動作（SKIP ＋ 進場 z ∈ {1.5, 2.0, 2.5, 3.0} × 出場 z ∈ {0, 0.5}）
中逐配對逐期選門檻，8 組配對來源方向全正、BH 校正後 5/8 顯著。本案問：

> **同一份選單，若改以 Ornstein-Uhlenbeck 模型的解析式挑選（不學習），能否得到相同的增益？**

若能，DL-THR 的價值來自「依配對調整門檻」本身，而非學習；若不能，學習的價值更明確。

## 二、解析式選擇器（`strategies/trading/ou_threshold_trading.py`）

每個 (配對, 期) 只用形成期資料：

1. **形成期 z 序列**：與交易端同一定義（標準化對數股價的價差，以形成期 `Spread_Mean`／`Spread_Std` 標準化）。
2. **OU 配適**：AR(1) $z_t = c + \varphi z_{t-1} + e_t$（OLS）；$\theta = -\ln\varphi$，
   $\sigma = \mathrm{sd}(e)\sqrt{2\theta/(1-\varphi^2)}$。$\varphi \notin (0, 1)$ 視為無均值回歸 → SKIP。
3. **成本換算為 z 單位**：部位為 signal 避險（$v_A : v_B = 1/\sigma_A : |\beta|/\sigma_B$），每單位 $\Delta z$ 的報酬
   為 $\mathrm{Spread\_Std}/D$，$D = 1/\sigma_A + |\beta|/\sigma_B$；來回成本（進出場各一次、按總名目額）為
   $2f$，故 $c_z = 2f \cdot D / \mathrm{Spread\_Std}$（$f$ = 單邊摩擦率 `fee_rate + slippage_rate`）。
4. **Bertram (2010) 的單位時間預期報酬**：以多方為例，進場 $a = -e$、出場 $m = -x$（$e$、$x$ 為選單的進出場 z），
   $$\mu(e, x) = \frac{(e - x) - c_z}{E[T]},\qquad E[T] = \frac{\pi}{\theta}\Big[\operatorname{Erfi}\big(m\sqrt{\theta}/\sigma\big) - \operatorname{Erfi}\big(a\sqrt{\theta}/\sigma\big)\Big]$$
   空方對稱。挑 $\mu$ 最大的動作；**最大 $\mu \le 0$ 時 SKIP**。
5. 選定後以 `zscore_trading` 的同一套模擬執行（進場 $|z| > e$、出場穿越 $\mp x$、期末強制平倉、停損照格）。

與 DL-THR 的差異只在「挑選方式」（解析式 vs 學習）；選單、z 定義、避險方式、網格相同。
執行引擎不同（本案沿用 `zscore_trading`，DL-THR 有自己的模擬），故另以「強制每期選 (2.0, 0)」
驗證本案的執行與 Z-Score 逐位元相同（V2）。

## 三、範圍

| 項目 | 規定 |
|:---|:---|
| 配對來源 | 4.3 節的 8 組：GICS-SSD、GICS-SDP、HDB-SDP、KM-SSD、AGG-SSD、不分組-SSD／DTW／SDP |
| 網格 | Top {1, 3, 5, 10, 20} × 停損 {0%, 5%, 15%} = 每組 15 格，共 **120 格** |
| 形成期 | 借用各來源既有的配對（零重跑） |
| 寫入 | `results/ou_threshold.db`（約 16 GB；2026-10-21 可用 38.5 GB） |
| 對照 | Z-Score 與 DL-THR 皆取 result.db 既有的同格 |

## 四、判準（跑前寫死；程序同 4.3：15 格等權逐日差分、循環區塊自助法 L = 126、10,000 次、8 組 BH）

**Q1 解析式相對固定門檻（OU − Z-Score）**，判定「有效」須同時滿足：
1. ≥ 6/8 組年化 Δ > 0；
2. 8 組中位年化 Δ > 0；
3. ≥ 2/8 組 BH 校正 $p$ < 0.05 且 Δ > 0。

**Q2 解析式相對 DL-THR（OU − DL-THR）**：
- 「解析式不劣於 DL-THR」：中位年化 Δ ≥ 0，且 0/8 組 BH 校正後顯著為負；
- 「DL-THR 優於解析式」：中位年化 Δ < 0，且 ≥ 2/8 組 BH 校正後顯著為負；
- 其餘 → 不確定。

另報（不入判準）：後半期（2014-01-02 起）Δ；SKIP 比例與動作分布（與 DL-THR 並列）；
$\varphi \notin (0,1)$ 的比例；15 格平均 Sharpe 與 rf 超額（5.1.1 衡量方式）；**組合市場 β**（殘差報酬價差的教訓）。

**績效不佳的處置（使用者 2026-10-21 指示）**：若 Q1 不成立，`ou_threshold.db` 只保留摘要（strategy_summaries），
刪除逐日紀錄（trade_logs）。

## 五、試驗宇宙

解析式選擇器是新的交易端（如同 DL-THR 各臂各為一個 METHOD），8 條各計為新方法：
method N **75 → 83**，config +120（庫外）。跑完後重算 var_sr 與 SR₀ 並更新 `TRIAL_CENSUS`。

## 六、接線驗證（跑網格前必須全過）

- **V1**：既有條目逐項不變（config 比對）。
- **V2**：以測試旗標強制每期選 (2.0, 0)，GICS-SDP 與不分組-DTW 各一格與 result.db 的 Z-Score 同格逐位元相同。
- **V3**：以獨立程式（不同的 AR(1) 估計函式）重算數個配對期的 $\theta$、$\sigma$、$c_z$、各動作的 $\mu$ 與所選動作，
  與引擎記錄一致。
- **V4**：試跑格的 SKIP 比例不為 0% 或 100%，且所選動作不全是同一個。

---

## 附記（2026-10-21，接線驗證之後補記；判準與範圍未改動）

- **V1 通過**：Kalman 交易端（自有同名參數 `spread_mode`）NOGRP-DTW-KAL Top1/SL0 與 result.db 逐位元相同。
- **V2 通過**：強制 (2.0, 0) 時，GICS-SDP、不分組-DTW 的 Top1/SL0 與 Z-Score 同格逐位元相同。
- **V3 通過**：8 個配對期以 `np.polyfit` 獨立重算，|Δθ| ≤ 8.6e-16、|Δσ| ≤ 1.7e-16、|Δc_z| = 0，所選動作全數相符。
- **V4 通過**：不分組-DTW Top3/SL0 的 SKIP 5.1%；所選動作 (1.5, 0) 758、(2.0, 0) 46、(2.5, 0) 22、(3.0, 0) 14。
  **解析式多數期選最低進場門檻**——Bertram 準則是單位時間報酬，成本相對價差小時頻繁交易較優；
  DL-THR 的門檻中位為 2.19–2.37（4.4 節），方向相反。此為預期外的特徵，於結果中報告，不改判準。
