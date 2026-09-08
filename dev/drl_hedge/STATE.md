# 中止狀態（2026-09-02）

`tools/rerun_drl_hedge.py --execute` 於第 1/8 支臂執行中被使用者要求暫停，
行程已終止。**`result.db` 目前處於混合狀態。**

## 現況

| 範圍 | 列數 | 執行口徑 | `Hedge_Mode` 標籤 |
|:---|---:|:---|:---|
| `Grid (AGG-SSD-DRL)` 的 12 格 | 12 | **signal（已修正）** | signal ✓ |
| `Grid (AGG-SSD-DRL)` 其餘 3 格 | 3 | dollar（未重跑） | signal ✗ |
| 其餘 7 支臂 | 105 | dollar（未重跑） | signal ✗ |

已重跑的 12 個 path_key 列於 `PARTIAL_RERUN.txt`（Top1/3/5/10 × SL0/5/15；
缺 Top20 的三格）。

## ⚠ 這個狀態無法從資料庫本身辨識

新舊列的 `Hedge_Mode` **都是 `signal`**：

- 舊列：`base_params` 寫 signal，但引擎沒實作 → 標籤錯（本案要修的原始 bug）
- 新列：引擎已實作 → 標籤對

`assert_uniform_hedge_mode` 看到的是一片 signal，**不會報錯**。
`PARTIAL_RERUN.txt` 是唯一的區分依據，別刪。

## 重啟方式

```
python tools/rerun_drl_hedge.py --execute
```

`FORCE_RERUN=1` 會把 120 格全部重寫，**已跑的 12 格會再跑一次**——
DRL 不固定種子，故那 12 格會得到與現在不同的新值。這沒有問題：
它們本來就只是分布中的一個抽樣，不是需要保存的東西。

**重啟前不要動 `results/analysis/drl_hedge_pre.csv`**——那是修正前的唯一快照，
120 列都在，重跑會原地覆寫，沒有它就無法做前後對照。

## 暫停期間的使用禁令

**本八支臂的任何分析都不可執行**，包括命題 2 全鏈
（`proposition2_daily_hac`、`prop2_exposure_control`、`prop2_skip_permutation`、
`prop2_label_information`、`drl_behavior`、`prop3_combined_system`、
`regime_cost_dsr_eval`、`regime_cost_ew`）。

現在跑會拿到 12 格新口徑 + 108 格舊口徑的混合，而且**不會有任何錯誤訊息**。

> 這正是 `tools/refresh_after_rerun.py` docstring 描述的那種失敗：
> 「分析結果與重跑前逐位元相同，而且不報任何錯」——只是這次混的是口徑不是快取。

## 過程中的一個判讀錯誤，記錄以免重犯

第一次清點時我以字串比對 `Sharpe_Raw`，得到「94/120 已重跑、八支臂全中」——
但執行器當時只跑到第 1 支臂，這個結果不可能為真。

原因是 `pre` 來自 CSV、`now` 來自 SQLite，浮點數經 CSV 往返後
`str()` 表示不同，**與數值是否改變無關**。改用數值比對（容差 1e-9）
後得到正確答案：12 列，全在第 1 支臂。

**教訓**：跨儲存格式比對數值一律用容差，不要用字串。
而且「結果與已知的執行進度矛盾」本身就是最好的除錯訊號——
先問「這個數字可能為真嗎」，再去找原因。

---

# 重啟紀錄（2026-09-03 16:35）

使用者通知重啟，`tools/rerun_drl_hedge.py --execute` 已重新啟動，
從第 1 支臂開始把 120 格全部重寫（含先前已跑的 12 格）。

## 啟動前做的兩件事

1. **另存 `results/analysis/drl_hedge_pre_dollar.csv`**——這才是真正的修正前
   （120 列全 dollar 口徑）基準。
2. **修掉 `snapshot()` 的原地覆寫**：中止後重啟時 `tag="pre"` 拍到的是
   *混合狀態*，若照原樣覆寫 `drl_hedge_pre.csv`，唯一的修正前基準就沒了。
   現在既有檔案一律保留，改寫成帶時間戳的新檔。

   > 這是「中止—重啟」路徑上的陷阱，不是原始設計的疏失：
   > 一次跑完的情境下 `pre` 確實只會拍一次。**會被中斷的流程，
   > 其冪等性要以「重跑一次」而非「跑一次」為前提來檢查。**

## 前後對照該用哪一份

| 檔案 | 內容 |
|:---|:---|
| `drl_hedge_pre_dollar.csv` | **修正前基準**（120 列 dollar）← 對照用這份 |
| `drl_hedge_pre_20260903_163529.csv` | 重啟當下的混合狀態（12 signal + 108 dollar），僅供存查 |
| `drl_hedge_pre.csv` | 同 `_dollar`，內容相同，保留不動 |
| `drl_hedge_post.csv` | 本次重跑完成後寫出（120 列全 signal） |

`PARTIAL_RERUN.txt` 在本次重跑完成後即失去意義（那 12 格會被新值覆蓋），
可保留作為中止事件的紀錄，但**不要再拿它當口徑判別依據**。
