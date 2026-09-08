# 交易架構檢視（2026-08-28）

檢視範圍：`run_trading.py` → `strategies/portfolio_manager.py` →
`strategies/trading/*` → `strategies/{returns,metrics,db_utils}.py`，
外加 `result.db` 的 1,392 列實測。

CONTEXT.md 的「架構檢視」指向的是形成期四層；本文是交易端的對應物。

量測腳本置於本目錄，皆為唯讀、可獨立重跑：

| 腳本 | 對應章節 |
|:---|:---|
| `measure_hedge_mismatch.py` | A（修正前的量測；`--db` 可指向試跑庫） |
| `verify_hedge_fix.py` | A（修正後，逐條驗 `PREREGISTRATION.md` 的 P1–P5） |
| `compare_pilot.py` | A + E（試跑對舊庫的逐格比對） |
| `measure_concurrency_bug.py` | B（以修正前備份對照修正後資料庫） |
| `verify_bfill_lookahead.py` | E（最小重現） |
| `verify_clean_prices.py` | E（修正後的端到端驗證） |

**A、B、E 已於 2026-08-28 修正**，各章末附「修正記錄」。
A 另有 `PREREGISTRATION.md`（跑之前定稿）。

### 混庫防護

`strategy_summaries` 新增兩欄，皆由引擎於寫入時落庫：

| 欄 | 意義 | 舊列 |
|:---|:---|:---|
| `Concurrent_Periods` | 引擎實際用的並行期數 | 已由 `tools/backfill_concurrency.py` 回填 |
| `Hedge_Mode` | 對沖權重口徑 | `NULL` = `"dollar"`（修正前） |

`check_trading_completed` 會比對 `Hedge_Mode`：**有列還不夠，口徑也要對得上**。
故全網格重跑中斷後，正確的續跑方式是**不帶 `FORCE_RERUN` 直接再跑一次**——
已用新口徑跑完的格會被跳過，尚未重跑的舊格會被補上。
沒有這道檢查的話，舊口徑的列看起來就是完成的，result.db 會永久停在
「一半 signal、一半 dollar」而無人察覺。

---

## A. 訊號空間 ≠ 執行空間

`ssd_rolling.normalize_prices()` 把每檔標準化成
$\tilde P_X = (\ln P_X - \mu_X)/\sigma_X$。spread、$\beta$、ADF、Hurst、
半衰期、SSD 排序、交易 z —— **全部**定義在這個空間：

$$S_t = \tilde P_{A,t} - \beta \tilde P_{B,t}, \qquad
\Delta S_t = \frac{r_{A,t}}{\sigma_A} - \beta\frac{r_{B,t}}{\sigma_B}$$

但 `zscore_trading.py:86-104` 的建倉是純美元權重：

```python
total_weight = 1.0 + abs(hedge_ratio)
v_a = C * (1.0 / total_weight)          # v_a : v_b = 1 : |β|
v_b = C * (abs(hedge_ratio) / total_weight)
shares = ±v / p
```

$$\text{PnL}_{\text{exec}} \propto r_A - \beta r_B$$

要複製 $\Delta S$ 需要的美元權重是 $(1/\sigma_A) : (\beta/\sigma_B)$。
**B 腳被錯乘了 $\sigma_B/\sigma_A$。**

### 這個比值有多分散

20 萬列 `Formation_Params` 的 $\sigma_A/\sigma_B$：

| 中位 | p10 | p90 | $\lvert\ln(\sigma_A/\sigma_B)\rvert > 0.3$ |
|---:|---:|---:|---:|
| 0.946 | 0.563 | 1.527 | **40.2%** |

### 直接對照兩個組合

`Grid AGG-SSD-NF` Top20/SL0，5,690 筆完成交易，比較「引擎實際持倉」
與「能複製訊號的持倉」的**毛**損益（同一組進出場日期、同一筆總名目額）：

| | 數值 |
|:---|---:|
| 兩者相關 | 0.9216 |
| **符號相反的交易** | **6.34%** |
| $\Delta\lvert z\rvert>1$ 收斂交易：複製組合毛虧損率 | 0.25% |
| $\Delta\lvert z\rvert>1$ 收斂交易：**引擎組合毛虧損率** | **3.08%** |
| 中位相對偏差 $\lvert\text{exec}-\text{rep}\rvert/\lvert\text{rep}\rvert$ | 16.1% |
| $\Delta\lvert z\rvert>1$ 平均毛報酬（複製 → 引擎） | 4.079% → 3.788% |

第三、四列是判準。spread 收斂 1 個標準差以上時，正確複製的組合在毛額上
**不可能**虧錢（0.25% 為浮點與 `asof` 邊界殘留）；引擎的組合虧 3.08%，
高 12 倍。每筆收斂交易平均漏掉 **0.29 pp 毛報酬** —— 對照 break-even
餘裕 0.391–0.924%（`analysis/regime_cost_ew.py`），這不是捨入誤差。

### 哪些臂受影響

| 路徑 | spread 空間 | 執行權重 | 正確？ |
|:---|:---|:---|:---:|
| 路徑 A（`ols_alpha` 非 None，log-price OLS） | $\ln P_A - \alpha - \beta \ln P_B$ | $1:\beta$ | ✅ |
| 路徑 B（`zscore_log`，SSD / DTW / SDP 全系列） | $\tilde P_A - \beta\tilde P_B$ | $1:\beta$ | ❌ 缺 $1/\sigma$ |
| `distance_trading`（`normalize_mode="zscore_log"`，預設） | $\tilde P_A - \tilde P_B$ | $1:1$ | ❌ 缺整個 $\sigma_A/\sigma_B$ |
| `distance_trading`（`normalize_mode="ggr_index"`） | $P_A/P_{A,0} - P_B/P_{B,0}$ | $1:1$ | ✅ GGR 原意 |

### 為何這比「效率損失」嚴重

命題 1 與命題 2 檢定的是「分組／排序準則能不能選出會收斂的 spread」。
被回測的卻是一個 $1:\lvert\beta\rvert$ 美元中性組合的損益。
在 40% 的配對上這兩者不是同一個量。**這不是 bug，是定義選擇** ——
但目前沒有任何地方寫下曾經做過這個選擇。

### 兩條出路（需決策，非修 bug）

1. **改執行以複製訊號**：`_execute_entry` 改用 $(1/\sigma_A):(\beta/\sigma_B)$。
   `Log_Std_A/B` 已在 `Formation_Params` 裡，不需重跑形成期。
   這是 GGR 與 Han 兩份文獻的原意。
2. **改形成期以配合執行**：把排序與統計檢定搬到 $1:\beta$ 美元空間
   （即改用報酬空間的 spread）。代價是全部形成期重跑。

### 修正記錄（2026-08-28）

**採用出路 1（改執行以複製訊號），且判定為修正而非新方法。**
理由與可否證預測見 `PREREGISTRATION.md`（跑之前定稿）。

判定的關鍵：不存在任何一種讀法，能讓 $1:|\beta|$ 美元權重複製路徑 B 的訊號
——除非 $\sigma_A=\sigma_B$，而那在 40% 的配對上不成立。專案內無任何一處
記載曾做過這個選擇，故它是實作錯誤，不是建模決策。
**因此不新增 `METHOD`、不增加試驗宇宙的 $N$。**

實作（`zscore_trading._leg_scales` + `_execute_entry`）：

$$v_A:v_B=\frac{1}{\sigma_A}:\frac{\beta}{\sigma_B},\qquad |v_A|+|v_B|=C$$

新參數 `hedge_mode`：`"signal"`（預設，新）／`"dollar"`（舊行為，供重現與對照）。
z 的計算、`entry_z`、`exit_z`、`_compute_spread` **一律未動**。

`verify_hedge_fix.py` 逐條驗預先註冊的 P1–P5，全數通過：

| 預測 | 結果 |
|:---|:---|
| P1 路徑 A / `P/P0` 分支 / `ggr_index` 權重逐位不變 | ✅ 三者皆 $(5555.56,\ 4444.44)$ 或 $(5000,\ 5000)$，兩模式相同 |
| P2 `SL0` 的 Status 序列逐位相同、損益不同 | ✅ 相同；損益 79.75 vs 71.76 |
| P2′ `SL5` 的 Status 會變 | ✅ 如預期（停損依損益觸發） |
| P3 總名目額仍為 `capital_per_pair` | ✅ 10,000.0000 |
| P4/P5 損益與 $-\Delta S$ 共線 | ✅ signal 模式 $\rho=1.0000000000$（三組 $\beta,\sigma$）；dollar 模式 $\rho=0.84$–$0.91$ |

退化守衛：$\beta=0$ 全額配於 A 腳、$\sigma$ 缺值退回 1.0，總名目額皆不變。

---

## B. `CONCURRENT_PERIODS` 寫死 —— 同一個錯誤的第三份副本

`metrics.py` 的 docstring 自己記了兩次「名目額漏並行因子」
（2026-08-26 `regime_cost_dsr_eval`、2026-08-27 `regime_cost_ew`）。
這是第三次，方向相反，住在為了修前兩次而生的那支模組裡。

引擎**逐策略**推導（`run_trading.py:247`）：

```python
_concurrent = max(1, int(params.get("trading_window", FORWARD_DAYS))
                  // max(1, int(params.get("rolling_step", rolling_step))))
```

讀端寫死全域 6：

| 位置 | 內容 |
|:---|:---|
| `db_utils.py:327` | `max_pairs = top_n_int * CONCURRENT_PERIODS` |
| `metrics.py:221` | `max_pairs = int(top_n) * CONCURRENT_PERIODS` |
| `metrics.py:252` | `conc` 由**預設參數** `trading_window=126, rolling_step=21` 推得 |
| `regime_cost_dsr_eval.py:314` | `traded_notional(br["_path"], top_n)` —— 未傳窗長 |
| `regime_cost_ew.py:147` | `traded_notional(r["_path"], _top_n_int(...))` —— 未傳窗長 |

### 誰中招

`Grid (HAN4-MONTHLY)`：`trading_window=21, rolling_step=21`
（`config.py:779`）→ 引擎的 `_concurrent = 1`，讀端當成 6。

該臂 `entry_z=0.0` + `hold_to_period_end=True`，即每天滿倉，
真實利用率必為 1.0。`result.db` 現存值：

| 欄位 | 現存 | 真值 | 倍數 |
|:---|---:|---:|---:|
| `Avg_Utilization`（Top 1/3/5/10/20，SL0） | **0.167011** | 1.0 | 1/6 |
| `Ann_Ret_Employed`（Top20/SL0） | −12.06% | ≈ −3.31% | 6× |
| `traded_notional`（Top20/SL0） | 752,160 | 4,512,959 | 1/6 |
| **break-even 往返** | **−0.7565%** | **+0.3572%** | −1.11 pp |

`0.167011` 恰為 $1/6$，且五個 `top_n` 完全相同 —— 分母錯的簽名。
現存的 break-even 是**負值**，意即「零成本也照虧到這個程度」，
而 `regime_cost_dsr_eval.run(methods=None)` 涵蓋全部非 DRL 方法，故已進表。

### 修法

`_concurrent` 應該是策略設定的一部分並寫入 `strategy_summaries`，
不是全域常數。`CONCURRENT_PERIODS` 這個名字本身即 bug 來源
—— 它宣稱「並行期數」是全域性質，而實際上是逐策略性質。

### 修正記錄（2026-08-28，已完成）

**中招的是兩支臂，不是一支。** 清點 `strategies_raw` 後另找到
`Grid (NOGRP-DTW-TW63)`（`trading_window=63` → 3 期）。以形成期窗界在
價格日曆上鋪算，兩者的真實重疊如下——確認 1／3／6 期的判定無誤：

| 臂 | 窗長 | 每日重疊期數（日數） |
|:---|---:|:---|
| `Grid (HAN4-MONTHLY)` | 21 日 | 1 期 × 6,274；**2 期 × 13** |
| `Grid (NOGRP-DTW-TW63)` | 63 日 | 3 期 × 6,203；2 期 × 42；**4 期 × 13** |
| 其餘（如 `AGG-SSD-NF`） | 126 日 | 6 期 |

那 13 個重疊日使 `Avg_Utilization` 可略超 100%（HAN4 最高 1.0021）。
**這不是分母錯，是 D-1 的徵狀**：`max_pairs` 只是名目槽位，執行期沒有記帳，
期界對齊時實際部署會短暫超出。回填工具的守衛因此取 1.05 而非 1.0。

改動：

| 檔案 | 改動 |
|:---|:---|
| `config.py` | 新增 `concurrent_periods(params)` —— 唯一擁有者；`CONCURRENT_PERIODS` 降級為「預設值」並加警語 |
| `run_trading.py` | 原地推導改呼叫 helper |
| `db_utils.py` | 改用 helper；新增 `Concurrent_Periods` 欄與其遷移，寫入引擎實際值 |
| `metrics.py` | 新增 `concurrent_of(path_key)` 自 DB 取權威值；`traded_notional` **移除** `trading_window`/`rolling_step` 兩個預設參數 |
| `dashboard.py` | `compute_range_metrics` 的 `* 6` 改為 `concurrent_of(path)` |
| `tools/backfill_concurrency.py` | 新增。回填 1,392 列，修正其中 30 列 |

`metrics.concurrent_of` 在欄位缺失或為 NULL 時**拋錯而非退回預設**——
這個量猜錯不會有任何徵狀，靜默的預設就是前三次的失敗模式。

回填是封閉解（`max_pairs` 在三個式子裡都只是乘法因子），故不必掃 174 GB 的
`trade_logs`；並以 `Concurrent_Periods` 是否已寫入保證冪等。

驗證：

- `python -m tools.check_metrics --recon 8` 全數 PASS，`Avg_Utilization`
  對 `strategy_summaries` 逐位一致
- 受影響的 30 列另以 `metrics()` 自 `trade_logs` 原始重算，與回填值全數相符
  （封閉解與原始重算兩條獨立路徑）
- `python -m tools.check_metrics` 的自洽檢驗「費率設為 break-even 時淨利歸零」
  殘餘 \$0.00

下游重跑（`results/analysis/*.pre20260828.csv` 為修正前備份）：

| 產物 | 變動 |
|:---|:---|
| `breakeven_dsr.csv` | TW63 往返 BE **1.690 → 1.135**（餘裕 1.11 → 0.555 pp）；HAN4 **0.826 → 0.621**（餘裕 0.246 → 0.041 pp） |
| `breakeven_dsr_main8.csv` | **無變動**（八支主力臂皆為 126/21） |
| `breakeven_ew.csv` | 無變動（該表只含 5 個配對底，不含這兩支） |
| `thesis/draft/CONTRIBUTIONS.md` | TW63 的 1.690 已改為 1.135 並附註 |

Sharpe / DSR / `Ann_Ret_Raw` / MDD 全數未動——它們不經 `max_pairs`，
這正是修正應有的作用範圍。

---

## C. `Trading.run()` 是死碼，組合層停損不存在

`worker_task` 逐配對呼叫 `_simulate_pair`（`run_trading.py:420`），
**從不呼叫 `run()`**。全專案 `.run()` 的呼叫端只有 formation 側。

後果：

1. `zscore_trading.py:492-540` 的組合停損斷路器從未執行。
   `portfolio_stop_loss_pct` 預設 0.0 使它目前無害。
2. 那段死碼帶一個真 bug：觸發日設 `Daily_Delta = 0.0` 卻改動
   `Cumulative_PnL`，會把當日（依定義是大跌日）從報酬序列抹掉，
   使 MDD 被低估。參數一旦打開就會中。
3. `archive/scripts/generate_formation_trading_notebooks.py:408`
   仍向讀者宣稱有「PSL 一鍵斬倉」機制。

組合層的東西只能活在跨配對層。要嘛移到 `worker_task`，要嘛刪掉並改文件。

---

## D. 資金配置有四個擁有者，沒有一個管到跨期

### D-1 `active_pairs` 每期清空，跨期槽位帳從未生效

`run_trading.py:460` 的 `pm.active_pairs.clear()` 使每期開頭
`get_available_slots()` 恆等於 `max_pairs`。`allocate_capital` 因此
退化成常數除法 `equity / (top_n × 6)`。

`PortfolioManager` docstring 宣稱「確保任意時點總部署 ≤ current_equity」
—— 成立，但成立的理由是除數保守（最壞情況上界），不是因為有記帳。

### D-2 `dynamic_slots` 的分位數在量一個常數

`concurrency_history` 記的是配置後的 `len(active_pairs)`，
而它每期被 clear，故恆等於 `top_n`：

```
percentile([20, 20, 20, ...], 75) = 20
```

`_DYN75` 的實際行為是「前 8 期（`warmup_obs`）`equity/120`，
之後 `equity/20`」的階梯，不是任何 walk-forward 校準。
既然這些列已與靜態版並存於 `result.db`，引用前須先釐清。

### D-3 權益回填有真前視

`pm.process_closed_trade`（`run_trading.py:453`）在**整期模擬完**
才把該期已實現損益加進權益。但期是重疊的（126/21 → 6 期）：

```
期 i    : Trade_Start = T_i      Trade_End = T_i + 126d
期 i+1  : Trade_Start = T_i + 21d
```

期 $i+1$ 的 `capital_per_pair` 用到了含 $T_i + 126$ 損益的權益 ——
**領先 105 個交易日**。量級小（年化 4% 下每期約 2% 的規模偏差），
但無法在事後宣稱「無前視」。

### D-4 `vol_target_allocation` 繞過 PortfolioManager

`run_trading.py:344-357` 直接寫 `pm.active_pairs`，自帶一套
`min(base*n*w, base*2)`，不經 `allocate_capital`、不檢查 `committed`。
這正是 `returns.py` docstring 花三段在罵的「同一個問題散在多處各自作答」。

### D-5 「哪一條權益曲線」有三個答案

| 出處 | 權益 |
|:---|:---|
| 引擎（`portfolio_manager`） | 逐**期**階梯，於期末一次回填 |
| `returns.py:returns_from_pnl` | 逐**日**累加 |
| `metrics.py:traded_notional` | 逐**日**累加後 `/ max_pairs` |

`returns.py` 宣稱複利分母「與引擎的部位規模一致」。三條線數值接近
（自洽檢驗 −\$9 / −\$2），但「一致」的說法目前不成立。
D-3 修好之後三者才會真的收斂成一條。

---

## E. 每期第一個交易日用的是第二天的價格

`zscore_trading.py:34-35`：

```python
_pct = self.price_df.pct_change().abs()
self.price_df = self.price_df.where(_pct <= 0.50).ffill().bfill()
```

首列的 `pct_change` 是 `NaN`，而 `NaN <= 0.50` → `False`
→ `where` 把**整個首列**打成 `NaN` → `ffill` 無前值可用
→ **`bfill` 以第二天的價回填**。

`zscore_window` 預設 0（`config.py:158`）→
`extended_start_idx == trade_start_idx`（`run_trading.py:377`）
→ 首列就是該期第一個交易日。

**主網格的每一期、每一對，第一個交易日的價格與 z 都是隔日的。**

兩腳同時被替換，故當日→次日的 `Daily_Delta` 為 0；淨效果是
「首日進場拿到次日的成交價」。對形成期結束時已發散的配對
（進場的主要來源）是系統性有利的填單。

同一個 `_clean` lambda 亦出現於 `distance_trading.py:73` 與
`kalman_trading.py:89`，作用於全表，只影響資料首日，無實害。

修法：

```python
self.price_df = self.price_df.where(_pct.le(0.50) | _pct.isna()).ffill().bfill()
```

### 修正記錄（2026-08-28，程式已修，**結果尚未重跑**）

清洗抽成 `zscore_trading.clean_prices()` 這個單一擁有者。原本三份副本
（`zscore_trading.__init__`、`distance_trading` 與 `kalman_trading` 各一個
lambda）帶有同一個前視，改一處會漏兩處——這正是它活到現在的原因。

順帶解掉 §G 的重算浪費：結果快取於來源 `DataFrame` 的 `.attrs`，
逐配對重建 `Trading` 不再重算。以 `.attrs` 而非 `id()` 作鍵，是因為 `id`
在 GC 後會被重用。

⚠ **`.attrs` 會隨 `.iloc` 切片一起傳播**（已於 `verify_clean_prices.py`
測試 4 證實），故快取必須驗明索引與欄位，否則一個 126 日的期間切片會拿到
整張 6,647 日的表。這個陷阱差一點就把修正變成更嚴重的 bug。

`verify_clean_prices.py` 驗五件事，全數通過：

| # | 斷言 | 結果 |
|--:|:---|:---|
| 1 | 首列不再被第二列覆蓋 | 100.0（原值） |
| 2 | 真正的 >50% 跳空仍被遮蔽，隔日恢復真實價 | 40 → 遮成 101，隔日 41 |
| 3 | `Trading.__init__` 這條實際路徑的首列正確 | 100.0 |
| 4 | 期間切片不會拿到全表 | 切片長度 4（全表 8） |
| 5 | 全表快取命中 | 首次 170 ms → 之後 0.016 ms |

**尚未做的事：全網格重跑。** 修正會改變每期第一個交易日的價與 z，
因而改變進場點——依 CONTEXT.md 的 Proxy 規則，位移量必須實跑才知道，
不能估。`result.db` 現有的 1,392 列仍是修正前的結果。

---

## F. 全鏈沒有一處執行延遲

`_simulate_pair` 的迴圈裡，訊號 `z = zscore_arr[i]` 與成交價
`pa_arr[i] / pb_arr[i]` 取自**同一根收盤棒**。
`grep 'shift('` 於 `strategies/trading/` 下 0 個結果
（只有 `run_trading.py` 兩個 regime gate 的建構有 `shift(1)`）。

出場尤其樂觀：z 恰好穿越 `exit_z` 的當日即以該日收盤成交。
對均值回歸策略這是最有利的填單假設。

這是除 A 以外最可能吃掉 0.391–0.924% 成本餘裕的因素，
值得一次 t+1 執行的受控消融。**注意它改變進場頻率與部署規模，
依 CONTEXT.md 的 Proxy 規則必須實跑，不能用 `capture_frac` 估。**

---

## H. 兩支 DRL 臂漏了 `ignore_ols_alpha`（2026-08-28 清點時發現）

`run_trading.py` 的 `ignore_ols_alpha` 註解說明了它為何存在：

> 修正 DTW Paper 的座標系錯位——其 OLS 在**標準化 log-price 空間**擬合，
> 卻因輸出 `OLS_Alpha` 被路徑 A 以**原始 log-price 空間**重建 spread，
> 產生常數 Z-Score 偏移。

清點全部 23 支會產出 `OLS_Alpha` 的 DTW / SDP 臂，**只有兩支沒設這個旗標**：

| 臂 | `ignore_ols_alpha` | 借用的形成期 | 該形成期臂自己的設定 |
|:---|:---|:---|:---|
| `Grid (HDB-SDP-DRL)` | **未設** | `Grid HDB-SDP` | `True` |
| `Grid (GICS-SDP-DRL)` | **未設** | `Grid GICS-SDP` | `True` |
| 其餘 21 支 | `True` | — | — |

兩支都**借用**已設旗標之臂的配對，卻自己不設 → 同一批配對，
Z-Score 臂走路徑 B（正確），DRL 臂走路徑 A（帶常數偏移）。

這正是那個旗標被加進來要修的 bug，只是漏了這兩格。

### 裁決（2026-09-01）：**這是本檢視的誤判，兩支臂不受影響**

原判斷是從 config 的差異推出來的，**沒有追到消費端**。追完之後：

`ignore_ols_alpha` 全庫只有**一個**消費者 —— `run_trading.py:434`，
它只做一件事：把 `ols_alpha` 這個 kwarg 設成 `None`。

而 `drl_threshold_trading.py` 裡 **`ols_alpha` 出現 0 次**。
其 `_simulate_pair` 以 `**kwargs` 吞掉該參數且從不讀取（`kwargs` 在整個
方法體內只出現在簽章那一行），z 由自己的 `z_of()` 硬寫死為標準化空間：

```python
na = (np.log(pa.values) - log_mean_a) / log_std_a
nb = (np.log(pb.values) - log_mean_b) / log_std_b
return np.clip((na - hedge_ratio * nb - form_spread_mean) / sstd, -10.0, 10.0)
```

**那就是路徑 B。** 兩支 DRL 臂本來就走對的空間，旗標對它們是**惰性的**。
故：無 bug、無須重跑、試驗宇宙不變。

而且 `config.py:582`（DRL 疊加區塊的區頭註解）**早就寫明了這件事**：

> 交易端換成 drl_threshold_trading（走標準化空間 z_of，**不讀 OLS_Alpha**）

該註解自 2026-07-23（commit `ec3dc02`）即存在，早於本檢視一個月。
**兩支臂不設旗標是刻意的，不是遺漏。**

### 不補旗標，這是刻意的決定

原本打算「補上旗標作為意圖的文件化」，**改為不補**：
既有註解已經正確且明確地說明了資料流，再加一個惰性旗標反而製造假訊號——
會讓後來的人以為 DRL 端讀它。**惰性設定比缺少設定更難察覺。**

> **教訓**：本條目原本寫著「這正是那個旗標被加進來要修的 bug，只是漏了這兩格」。
> 那句話是從**設定檔的對稱性**推出來的，不是從資料流推出來的——而正確答案
> 就寫在我所標記的那個區塊往上三行的註解裡。
>
> 交易模組各自實作 `_simulate_pair`，config 的旗標是否生效**逐模組而異**。
> §A～§G 皆已追到消費端，唯獨本條沒有。**「設定不對稱」不是缺陷的證據，
> 追到消費端才是。**

---

## G. 較小但該修

| 位置 | 問題 |
|:---|:---|
| `zscore_trading.py:88-98` | $\beta<0$ 時以 `abs()` 定權重、方向卻寫死一多一空 → 對沖方向錯。實測僅 0.07% 配對，但 `zscore_window>0` 的 `roll_beta` 會常態轉負 |
| `run_trading.py:456` | `except Exception: print(...)` 吞掉單一配對失敗，run 仍報 SUCCESS。與 `:502` 特意加的 `_ok` 檢查標準不一致 |
| `distance_trading.py:73`、`kalman_trading.py:89` | 每個配對都把整張 6647×747 全表清洗一次：實測 211 ms × 20 對 × 295 期 ≈ **21 分鐘純浪費／變體** |
| `zscore_trading.py:34` | 期內清洗（126×747）亦每對重建：2.6 ms × 5,900 次／變體 |
| `zscore_trading.py:118-235` | `_compute_spread` 路徑 A / B 為兩份近乎逐字相同的 40 行，唯一差異是 `norm` 怎麼建。`use_vol_adjust` 與 `roll_std` 守衛因此需維護兩次 |
| `run_trading.py:59,83` | gate 快取鍵用 `id(price_pivot)`；id 於 GC 後會被重用。目前 `price_pivot` 活滿 worker 生命週期故無事 |
| `run_trading.py:391-393` | gate 建構寫在逐配對迴圈內 |
| `distance_trading.py:40`、`kalman_trading.py:41` | `_BASE_INIT_PARAMS` 白名單漏了 `entry_gate` 與 `max_holding_days` → 這兩個模組收到也會被靜默丟棄。目前無臂踩到（閘門臂全走 `zscore_trading`），是潛伏而非活躍的 bug |
| `zscore_trading.py:264` | `is_cap_stop` 比較的是**已扣來回費**的 PnL → 實際停損門檻比 `stop_loss_pct` 宣稱的緊。預設 0 故主網格不受影響 |
| `zscore_trading.py:246` | 中段 `NaN` 的 z 被當成 0.0；持倉中遇到會觸發出場（`z >= -exit_z`），而非「無訊號、維持」 |

---

## 處理順序

| # | 項目 | 性質 | 狀態 |
|--:|:---|:---|:---|
| 1 | **B** | 純讀端修正，曾污染 `result.db` 三欄與論文引用的 break-even | ✅ 已修並重跑下游 |
| 2 | **E** | 前視移除 + 清洗收攏成單一擁有者 | ⚠ 程式已修、**結果待重跑** |
| 3 | **A** | 需決策，非修 bug；任一出路都是新 `METHOD` | 待決策 |
| 4 | **C / D** | 架構債：把跨配對層從死掉的 `run()` 提到 `worker_task` | 待排 |
| 5 | **F** | 新實驗（預先註冊），非修正；由 3 的結果決定值不值得 | 待排 |

### 下一步的相依關係

E 的重跑與 A 的決策**應該合併成一次**，不要各跑一輪：兩者都改變進場點，
分兩次跑等於把 1,392 個配置算兩遍（且中間那一版沒有人會引用）。
但 A 一旦成案就是新的 `METHOD`，會進試驗宇宙、改變 DSR 的 $N$——
故順序是「先決策 A，再一起重跑」，而非「先跑 E 再想 A」。
