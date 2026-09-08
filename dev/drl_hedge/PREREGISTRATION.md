# 預先註冊：DRL / RL 交易端補上 signal 對沖口徑

寫定日期：2026-09-02。**本檔在跑之前定稿。**
起因：2026-09-01 檢查 `regime_cost_ew` 輸出時發現的漏修。

## 一、事實

`dev/trading_arch/` 的 §A 修正（執行權重由 $1:\beta$ 改為
$(1/\sigma_A):(\beta/\sigma_B)$ 以複製訊號 spread）只改了三個交易模組：
`zscore_trading`、`distance_trading`、`kalman_trading`。

**`drl_threshold_trading` 與 `rl_threshold_trading` 是獨立類別，不繼承
`zscore_trading.Trading`，兩者對 `hedge_mode` / `_leg_scales` 的引用數為 0。**
部位仍為舊口徑：

```python
tw = 1.0 + abs(hedge_ratio)
v_a = capital / tw
v_b = capital * abs(hedge_ratio) / tw      # 1 : β
```

而其 z 由 `z_of()` 建於標準化空間（$\tilde P = (\ln P - \mu)/\sigma$）。
**這正是 §A 指認的訊號／執行空間錯配**，完全適用，只是當時沒改到。

## 二、更嚴重的一半：標籤是錯的，而防護沒攔到

`base_params` 現有 `hedge_mode="signal"`，`db_utils` 以
`params.get("hedge_mode")` 寫入 `Hedge_Mode` 欄。
故 **120 列 DRL / RLTHR 結果標著 `signal`，實際執行 `dollar`**：

| METHOD | 交易端 | 列數 |
|:---|:---|---:|
| `AGG-SSD-DRL`、`GICS-SSD-DRL`、`GICS-SDP-DRL`、`HDB-SDP-DRL`、`KM-SSD-DRL` | DRL | 75 |
| `AGG-SSD-RLTHR-E05/E10/E20D` | RLTHR | 45 |

§A 建的三層防護（寫入端存 `Hedge_Mode`、續傳端比對、讀取端
`assert_uniform_hedge_mode`）**全部沒攔到**——因為它們比對的是**標籤**，
而標籤抄自 params。**防護能防「兩批資料用不同設定跑」，
防不了「引擎根本不讀那個設定」。**

這與 §H 是同一個病灶：從設定推行為，沒追到消費端。

## 三、對命題 2 的影響（必須先講）

命題 2 的設計是「同配對底、同排序，**唯一變因為交易端**」。
現況下 Z-Score 臂用 signal 權重、DL-THR 臂用 dollar 權重 —— **多了第二個變因**。

> **命題 2 的有效狀態只有兩個**：全部 dollar（§A 之前），
> 或全部 signal（本案之後）。
> **2026-08-30 至本次修正之間算出的所有命題 2 數字皆為混淆的，不得引用**，
> 包含 2026-09-01 管線第 6、7、8、9、10、16 步的輸出。

## 四、修法

`_leg_scales` 在此比 `zscore_trading` 簡單：DRL / RL 的 `z_of()` **只有一條路徑**
（標準化空間），無 `ols_alpha` 分支、無 `ggr_index` 分支。故：

```
hedge_mode == "signal"  →  (1/σ_A, 1/σ_B)
否則                    →  (1.0, 1.0)
```

三個落點（`rl_threshold_trading` 由 `drl_threshold_trading` import 第一項）：

| # | 位置 | 用途 |
|--:|:---|:---|
| 1 | `drl._fast_threshold_pnl` | 反事實標籤**與**正式模擬，兩模組共用 |
| 2 | `drl._simulate_pair` 的進場 | 逐日交易紀錄 |
| 3 | `rl._simulate_pair` 的進場 | 同上 |

**落點 1 同時決定訓練標籤**：修正後 9 個動作的反事實報酬會變，
agent 看到的答案卷本身改變。這是本案與 §A 的差別——§A 只改執行，
本案同時改了學習目標。**不可只改落點 2、3。**

### 併修：`Hedge_Mode` 改為由引擎回報

各交易模組宣告模組層常數 `SUPPORTS_HEDGE_MODE`；
`run_trading` 解析有效值：

```
effective = params["hedge_mode"] if module.SUPPORTS_HEDGE_MODE else "dollar"
```

**預設為 `False`**（未宣告即視為不支援）。新模組若忘了實作，
落庫會誠實寫 `dollar`，不會再假冒 `signal`。

> 這只擋「模組沒宣告」，擋不了「宣告了卻沒真的用」。
> 後者由 §五 的 P3 行為測試負責——**那才是真正的防護**。

## 五、可否證的預測

DRL 不固定隨機種子，**單次前後對照無法歸因**。故 P1–P3 一律設計為
**繞開 DRL 訓練的確定性檢定**，只有 P5 涉及重跑。

| # | 預測 | 判準 | 若不成立代表 |
|--:|:---|:---|:---|
| P1 | `hedge_mode="dollar"` 下 `_fast_threshold_pnl` 回傳值**與改動前逐位元相同** | 對真實配對逐一比對，diff = 0 | 改動污染了舊路徑 |
| P2 | signal 模式下部位損益與 $-\Delta S$ 的相關 = **1.0000**；dollar 模式 < 1 | 同 §A 的 P5 檢定，套用於 DRL 的部位建構 | 修正未達成其宣稱目的 |
| P3 | **行為測試**：對每個交易模組，切換 `hedge_mode` 必須改變輸出 | 逐模組斷言；不變即失敗 | 該模組宣告了 `SUPPORTS_HEDGE_MODE` 卻沒真的用 |
| P4 | 每配對總名目額不變（$v_A+v_B=C$） | 相對差 < 1e-9 | 權重正規化寫錯，`traded_notional` 失效 |
| P5 | 反事實標籤矩陣改變 → agent 選的動作分布改變 | 描述性，**不設判準** | —— |

**P3 是本案真正的產出。** 它把「設定是否生效」由人工檢查變成自動斷言，
正是 §H 與本案兩次都缺的那一層。

## 六、不預測績效方向

與 §A 相同：**這是修正，不是待檢驗的處理。**
採納與否不取決於績效往哪邊走。

## 七、重跑與歸因的限制（誠實揭露）

8 支臂 × 15 格 = 120 個配置。**但 DRL / RL 不固定種子**
（`drl_threshold_trading` 的 docstring 自陳：權重初始化與 batch 洗牌皆隨機，
walk-forward 增量訓練再放大路徑依賴）。

故：**單次重跑的前後差 = 修正效果 + 重訓雜訊，兩者無法分離。**

處理方式：
1. 重跑一次，得到修正後的單次值
2. 以 `DRL_VARIANCE_TAG=... python -m tools.run_drl_variance` 對**修正後**的
   同一批臂跑 5 輪，取得重訓雜訊的分布
3. **只有當修正前的值落在修正後 5 輪的全距之外**，才宣稱位移大於雜訊；
   否則只報「位移在重訓雜訊範圍內，不可歸因」

既有變異資料（`results/drl_variance_summary.csv`，2026-07-21）顯示
best-of-15-grid Sharpe 的跨輪全距約 **0.013–0.035**，但那是**舊的 METHOD 集合**
且僅 2–5 輪，不可直接套用到本案的八支臂。**必須重新量。**

## 八、試驗宇宙

**不變。** method $N$ = 53、config $N$ 不增——同一批 METHOD、同一批配置，
只是改正執行口徑。與 §A 的處置一致（修正不是新試驗）。

## 九、跑完必須連帶重算

修正後，命題 2 全鏈的結論可能改變，須重跑：
`proposition2_daily_hac`、`prop2_exposure_control`、`prop2_skip_permutation`、
`prop2_label_information`、`drl_behavior`、`prop3_combined_system`、
`regime_cost_dsr_eval`、`regime_cost_ew`。

---

## 七之補：判準的方向與虛無率（2026-09-04 補寫）

**寫作時點**：DRL 雜訊只跑完第 1、2 輪，第 3–5 輪與全部 RL 輪次尚未產生
（已核對 `drl_variance_runs_drlhedge.csv` 只有 run_id 1、2）。故本補述
**在看到判準所需的資料之前寫成**，不是事後挑閘門。

§七 第 3 點寫的是「修正前的值落在修正後 5 輪的全距之外」。這條規則
少了兩件事，補上：

### 補一、必須帶方向

「落在全距之外」含**低於下界**與**高於上界**兩種。修正若真的有效，
應該是**修正前低於修正後 5 輪的最小值**。落在上界之外是反向證據，
不可與正向合併計數。

### 補二、逐格判準的虛無率是 1/6，不是 0

5 個交換可置換的抽樣把數線切成 6 段，第 6 個抽樣落在最小值以下的機率恰為
$1/6 \approx 16.7\%$（僅需可交換性，不需常態）。故**逐格「落在下界之外」
本身不是證據**，要看的是 120 格中達標的比例是否顯著高於 1/6。

判準（跑前定案）：

| | 門檻 |
|:---|:---|
| 主判準 | 120 格中「修正前 < 修正後 5 輪最小值」的比例，單尾二項檢定對 $p_0 = 1/6$，$\alpha = 0.05$ |
| 方向對照 | 「修正前 > 5 輪最大值」的比例須明顯低於下界比例；若兩者相當，判為雜訊 |
| 分臂 | 8 支臂各自的比例一併列出。若位移只來自 1–2 支臂，只可宣稱那幾支 |

未達標即依 §七 原文報「位移在重訓雜訊範圍內，不可歸因」。

### 補三、`drl_hedge_post.csv` 不計入雜訊分布

+0.0838 是 `drl_hedge_pre_dollar.csv` 對 `drl_hedge_post.csv` 算出來的。
`run_drl_variance` 每輪自行 `clear_summaries` 後重跑，第 1 輪是**另一次抽樣**，
不是 post.csv 那一輪。故 5 輪全部可作為參考分布；
**但 post.csv 不可併入該分布**——它是被檢定的那一點，併入即是雙重取用。
跑完後先核對 post.csv 的值不等於任何一輪，以確認上述理解。

### 附帶（描述性，不作判準）

另列 $\binom{5}{2} = 10$ 組輪間兩兩 ΔSharpe 的分布，與 pre→post 的
ΔSharpe 分布並列。這是描述性的：輪間差是「無位移」的參照形狀，
但它與 pre→post 不同構（前者兩端皆為抽樣，後者一端固定），故不設門檻。

### 七之補的修訂記錄（2026-09-05，跑後）

補三的事實前提**錯誤**：`run_drl_variance` 在 `RUNS_CSV` 不存在時會把
`result.db` 既有列 `harvest(1, ...)` 當第 1 輪，故第 1 輪逐位等於
`drl_hedge_post.csv`（120/120 相等）。補三所要求的核對攔下了它。

處置：剔除第 1 輪，$k = 4$、$p_0 = 1/5$。判準形式不變。
結果見 `RESULTS_VARIANCE.md`。
