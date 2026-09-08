# 中止狀態：雜訊量測（2026-09-04）

`tools/run_hedge_variance.py` 於 **DRL 模組第 3/5 輪執行中**被要求暫停，
行程已全數終止（0 個 python 行程）。

## ⚠ `result.db` 缺 48 列，且 summary 與 trade_logs 不一致

`run_drl_variance` 每輪的機制是：先 `clear_summaries()`
（`DELETE FROM strategy_summaries WHERE METHOD IN (...)`），
再跑 `run_trading.py` 逐臂重寫。killed 在重寫途中。

| 臂 | summary | 內容 |
|:---|---:|:---|
| `Grid (AGG-SSD-DRL)` | 15 | 第 3 輪（完整） |
| `Grid (GICS-SSD-DRL)` | **0** | 已刪，未重寫 |
| `Grid (GICS-SDP-DRL)` | **0** | 已刪，未重寫 |
| `Grid (HDB-SDP-DRL)` | **12** | 第 3 輪，缺 3 格 |
| `Grid (KM-SSD-DRL)` | **0** | 已刪，未重寫 |
| 三支 `RLTHR` | 15 各 | 第 1 輪（修正後），本模組未動到 |

**`clear_summaries` 不刪 `trade_logs`。** 故那 48 個 path_key 目前是
「summary 列不存在、trade_logs 仍為第 1 輪資料」的不一致狀態。
`strategies/returns.py` 的快取指紋取自 `strategy_summaries` 的
(Final_Equity, Entries, Gross_Profit)，查無列即失效——不會靜默給錯資料，
但也不能用。

### 使用禁令

**五支 DRL 臂的任何分析都不可執行**，包括命題 2 全鏈。
三支 RLTHR 臂完好，但它們與 DRL 臂並列的分析（如 `prop2_*` 的對照）
同樣不可跑，因為對照組不完整。

## 重啟方式

```
python tools/run_hedge_variance.py
```

第 3 輪會**整輪重跑**（`clear_summaries` 再刪一次、`run_trading.py` 重算），
故上面那 27 列第 3 輪的部分結果會被新值取代——這沒有問題，
它們本來就只是分布中的抽樣。重跑後 DB 即回到完整的 120 列。

已收成的輪次不會重跑：`results/analysis/drl_variance_runs_drlhedge.csv`
已有第 1、2 輪共 150 列，腳本由 `run_id.nunique()` 續跑至第 5 輪。

## 若只想讓 DB 恢復可用、不繼續量雜訊

跑一次 DRL 五臂即可（約 6 小時）：

```
FORCE_RERUN=1 STRATEGIES_SLICE=<drl_targets 給的索引> python run_trading.py
```

但**這樣就沒有雜訊分布**，`dev/drl_hedge/PREREGISTRATION.md` §七 的
歸因要求無法滿足，修正前後的 +0.0838 不可宣稱為位移。

## 中止時的早期讀數（n=2 輪，**不可下結論**）

| | 雜訊（第 2 輪 − 第 1 輪，75 格） | 修正前後（120 格） |
|:---|---:|---:|
| ΔSharpe 中位 | −0.0070 | **+0.0838** |
| 為正 | 33/75（44%） | **109/120（91%）** |
| 標準差 | 0.0735 | — |

方向一致性的差距（44% vs 91%）比中位數更值得注意，但**兩輪只給一次抽樣**，
標準差本身極不穩。五輪才有可用的分布。
且三支 RLTHR 的隨機性更大（多一層 ε-greedy），其雜訊分布尚未量到。

## 快照

| 檔案 | 內容 |
|:---|:---|
| `results/analysis/drl_hedge_pre_dollar.csv` | 修正前 120 列（dollar 口徑）**唯一來源** |
| `results/analysis/drl_hedge_post.csv` | 修正後第 1 輪 120 列 |
| `results/analysis/drl_variance_runs_drlhedge.csv` | 第 1、2 輪 × 75 列 |
| `results/analysis/drl_variance_runs_rlhedge.csv` | **尚未產生**（RL 模組未開始） |

**以上四份都不要動。**

---

## 重啟記錄（2026-09-04）

`python tools/run_hedge_variance.py` 已重新啟動（背景執行，日誌
`logs/run_hedge_variance.log`，該目錄已加入 `.gitignore`）。

從 DRL 模組第 3 輪接續（腳本以 `run_id.nunique()` 判定，第 1、2 輪不重跑）。
第 3 輪的 `clear_summaries` 會再刪一次五支 DRL 臂的 summary 列，
故**上方「缺 48 列」的禁令在本次跑完之前仍然有效**——中途 DB 依舊不可用於分析。

跑完後 `result.db` 回到完整 120 列（內容為 DRL 第 5 輪 + RL 第 5 輪的抽樣）。

---

## 第二次中止（2026-09-05 08:06）

於 **RL 模組第 3/5 輪執行中**被要求暫停。行程由外而內終止
（`run_hedge_variance.py` → `run_drl_variance` → `run_trading.py` → 15 個 worker），
確認 0 個 python 行程。監看器 `b5sceyrzg` 已停。

### 進度

| | 狀態 |
|:---|:---|
| DRL 五臂 × 5 輪 | ✅ **全數完成**，`drl_variance_runs_drlhedge.csv` 375 列（5 × 75） |
| RL 三臂 × 5 輪 | 第 1、2 輪已收成（`..._rlhedge.csv` 90 列 = 2 × 45），**第 3 輪中斷** |

### `result.db` 現況：84 列（缺 36 列），破損處換到 RL 這邊

| 臂 | summary | 內容 |
|:---|---:|:---|
| 五支 `*-DRL` | 15 各（共 75） | DRL 第 5 輪，**完整且與 `trade_logs` 一致** |
| `RLTHR-E05` | **9** | 第 3 輪重寫到一半 |
| `RLTHR-E10` | **0** | 已刪，未重寫 |
| `RLTHR-E20D` | **0** | 已刪，未重寫 |

前一次中止破的是 DRL 臂，這次 DRL 完好、破的是 **RLTHR 三臂**。

### 使用禁令（已更新）

**三支 RLTHR 臂不可用**（summary 缺列、`trade_logs` 仍是第 2 輪，不一致）。
五支 DRL 臂本身完整，但**命題 2 全鏈仍不可跑**——該鏈同時涵蓋 DRL 與 RLTHR。

**雜訊判定也還不能做**：`tools/judge_hedge_variance.py` 需要 120 格到齊，
資料不全時會自行退出（已驗證）。

### 重啟方式

```
mkdir -p logs && python tools/run_hedge_variance.py 2>&1 | tee logs/run_hedge_variance.log
```

DRL 模組會**自動略過**：`run_drl_variance` 以 `range(done_runs+1, runs+1)` 迴圈，
`done_runs = 5` 時迴圈為空、`clear_summaries` 在迴圈**內**故不會執行，
只跑 `aggregate()` 就結束（exit 0）。**DB 裡那 75 列 DRL 不會被刪。**
RL 模組由第 3 輪續跑。

### 剩餘時間（已實測，不再外推）

RL 第 1、2 輪：05:51 → 07:50，**約 1 小時／輪**（DRL 是 5 小時／輪）。
故剩下第 3、4、5 輪約 **3 小時**。先前「RL 約 15 小時」的估計是按 DRL 每臂成本
外推的，實測後推翻——RL 每臂便宜得多。

### 快照（都不要動）

`drl_hedge_pre_dollar.csv`、`drl_hedge_post.csv`、
`drl_variance_runs_drlhedge.csv`（375 列，**已完成，重啟不會覆寫**）、
`drl_variance_runs_rlhedge.csv`（90 列）。
