# 消融回測的斷點續傳操作手冊

適用於 `ACTION_SPACE_ABLATION=1`、視窗 `2009-07~2018-12` 的那一次跑。
起跑時間 2026-09-10 07:57。**預計會遇到週末斷電。**

---

## 一、斷電後怎麼重啟

```bash
cd /c/Clark/YZU/Papper/Code
ACTION_SPACE_ABLATION=1 \
BACKTEST_START=2009-07 BACKTEST_END=2018-12 \
python -u run_trading.py
```

**就這一行，跟第一次完全相同。** 三件事會自動發生：

1. v3（FQI）與 v4（門檻選擇）已完成並落庫，完成判定會直接跳過它們。
2. v1／v2 從 `results/tiingo/.ckpt/` 的逐期 `.pkl` 續算，
   並以「初始資金 + 已實現 PnL 累計」重建權益曲線。
3. Z-Score 臂（尚未跑，只需數秒）會補跑。

### ⚠ 絕對不要做的事

| 不要 | 原因 |
| :--- | :--- |
| `FORCE_RERUN=1` | 會 `shutil.rmtree` 掉 `.ckpt`，數天的計算歸零 |
| 設 `ABLATION_EPISODES` | 訓練預算必須維持 150（預先登記第五節） |
| 設 `ABLATION_TAG` | 會變成另一組 db_method，正式列不會被填上 |
| 改 `BACKTEST_START/END` | 視窗已凍結；改了共同期就不是 108 |
| 刪 `results/tiingo/.ckpt/` | 那就是全部的資產 |

---

## 二、為什麼續傳對這次是安全的

`run_trading.py:314` 是 `use_ckpt = trade_method != "DRL"`。
本來的意思是「DRL 因 walk-forward 訓練狀態暫不套用續傳」，
但五條臂的 `trade_method` 是 `DRLv1`／`DRLv2`／`DRLv3`／`DRL`／`Z-Score`
——**v1／v2／v3 因為字串不等於 `"DRL"` 而意外拿到了續傳**。

意外拿到不等於不安全。逐一查過 agent 快取鍵：

| 臂 | `agent_key` | 跨期是否延續狀態 | 續傳是否等價 |
| :--- | :--- | :--- | :--- |
| v1 | `period_trade_tickerA_tickerB` | 否（每配對每期重訓） | ✔ 等價 |
| v2 | `period_start_trade_start` | 否（每期重訓，期內共享） | ✔ 等價 |
| v3 | `formation_period`／或 `GLOBAL` | **`scope="global"` 時會延續** | ⚠ 見下 |
| v4 | — | `use_ckpt=False`（無續傳） | 23 秒，重跑即可 |

v1／v2 每期都從零重訓，所以「跑完 40 期後斷電、續算第 41 期」與
一次跑完 108 期**結果等價**。這是續傳可信的根據。

> **v3 是唯一有跨期狀態的臂**（`drl_fqi_trading.py:269-270`，
> `scope="global"` 時 `agent_key = "GLOBAL"`，一個 agent 貫穿所有期）。
> 幸好 v3 已於本次跑完（4,604 秒），其 `.ckpt` 目錄已按完成流程清除。
> **若日後 v3 需要重跑，不可依賴續傳，必須整條重跑。**

損失上限：每條臂最多丟掉「正在算的那一期」——v1 約 56 分、v2 約 36 分。

---

## 三、`result.db` 的斷電風險（無法用備份解決）

`strategies/db_utils.py:66` 是 `PRAGMA synchronous=OFF`。
搭配 WAL，這對「行程被殺」是安全的，但對**斷電不是**——
若斷電時剛好有寫入在途，SQLite 有損毀的可能。

**無法備份**：`results/result.db` 為 202 GB，而 C: 只剩 59 GB。

**處置**：`db_utils.py` 已加 env 逃生門 `SQLITE_SYNC`
（預設 `OFF`，行為與過去完全相同；設 `NORMAL` 則在 WAL 下無損毀風險，
最多丟掉最後一次 commit）。預期停電的長跑一律加 `SQLITE_SYNC=NORMAL`。
代價是寫入變慢，但寫入只在收尾發生，整體影響可忽略。

風險其實很窄：**寫入只發生在單一臂全部期跑完的那一刻**
（期間 `conn` 雖開著，但逐期只寫 `.pkl`，不寫 DB；
目前也確實沒有 `-wal` 檔存在）。所以危險窗口是 v1／v2 收尾的那幾分鐘，
不是這三四天。

### 斷電回來後，先驗完整性再繼續

```bash
python -c "import sqlite3;print(sqlite3.connect('results/result.db').execute('PRAGMA quick_check').fetchone()[0])"
```

回 `ok` 才重啟回測。不是 `ok` 就先停下來處理，不要再往裡面寫。

### 另一個靜默陷阱

完成判定「純以 `result.db` 的 `strategy_summaries` 為準」。
若斷電正好發生在 v1 寫入的中途，可能留下**摘要列已寫、明細不全**的狀態，
而重啟時 v1 會被判定為「已完成」而跳過。所以斷電後除了 `quick_check`，
還要核對列數：

```bash
python - <<'PY'
import sqlite3
c=sqlite3.connect("results/result.db")
for m in ("V1","V2","V3","DRL-DOLLAR","DOLLAR"):
    r=c.execute("SELECT _path,Entries FROM strategy_summaries WHERE METHOD LIKE ?",
                (f"%GICS-SDP%{m}%",)).fetchall()
    for p,en in r:
        n=c.execute("SELECT COUNT(*) FROM strategy_pairs WHERE strategy_id=?",(p,)).fetchone()[0]
        print(f"{m:<12} 配對期 {n:>4}  Entries {en}")
PY
```

各臂配對期應為 **119**，v1 應為 **108**（少的 11 期＝左緣例外，
預先登記 4.1 已揭露）。若某臂遠少於此卻有摘要列，就是被截斷的寫入
——須手動刪掉該臂在三張表的列後重跑。

---

## 四、已做的備份

| 內容 | 位置 | 大小 |
| :--- | :--- | ---: |
| 逐期 checkpoint（33 個 `.pkl`） | `backup/ckpt_20260910/` | 482 KB |

各臂應有的配對期數（2026-09-10 實測，非估計）：

| 臂 | 配對期 | 備註 |
| :--- | ---: | :--- |
| v3 FQI | **119** | 已完成落庫 |
| v4 門檻選擇 | **119** | 已完成落庫 |
| v2 修復版 | 119（預期） | 進行中 |
| Z-Score | 119（預期） | 尚未跑 |
| v1 LSTM-DQN | **108**（預期） | 少 11 期＝左緣例外，已於預先登記 4.1 揭露 |

→ **五臂共同期（交集）= 108**，與預先登記一致。

`backup/` 已加入 `.gitignore`。若 `.ckpt` 被誤刪，複製回
`results/tiingo/.ckpt/` 即可續算。


---

## 五、實戰紀錄：2026-09-12 的 48 小時停擺（**不是斷電續傳**）

第一次真正遇到中斷，但走的**不是**第一節那條路——process 從頭到尾沒有死。

| 時間 | 事件 |
| :--- | :--- |
| 09-12 07:34 | v1 寫下第 54 個 checkpoint（`2011-12-08.pkl`） |
| 09-12 07:44 | Kernel-Power 105：電源來源變更 → 改吃電池 |
| ~09-12 07:47 | **DC 待命逾時 180 秒**到期 → 進入 Modern Standby（S0ix） |
| 09-12 19:05 | Kernel-Power 105：市電恢復——**但機器沒有醒** |
| 09-14 07:31 | 人工打開筆電 → worker 立刻恢復滿載，從第 55 期接續 |

**損失 48.7 小時牆鐘（佔 51%），但一筆資料都沒丟。**
54 + 46 個 checkpoint 完好，續算是自動的，不需要任何人工介入。

### 診斷時繞的遠路（留作下次的捷徑）

當下的表象是「6 個 python 行程活著、CPU 98%、儀表板寫著 RUNNING、
但 48 小時沒有任何一期完成」，看起來非常像 worker 掛死。實際排除順序：

| 懷疑 | 實測 | 結論 |
| :--- | :--- | :--- |
| worker 掛死 | 60 秒內 CPU +58.9s | 活著且在算 |
| CPU 降頻 | 17.9 GFLOPS、方案「平衡」 | 正常 |
| 記憶體爆掉／換頁 | 31.4 GB 總量剩 15.2 GB，worker 各 2.2 GB | 無壓力 |
| 資料異常 | 卡住的期 251 列，與已完成者**完全相同**；v1 前一期還是同一對 DTE/NEE、53 分鐘跑完 | 非資料 |
| **電源事件** | **Kernel-Power 105／507** | **✔ 真因** |

→ **下次先查事件記錄檔，不要先查程式。** 一行就夠：
> `Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';StartTime=(Get-Date).AddDays(-5)}`

判別關鍵：**儀表板的 elapsed 會凍結在停擺當下的數值**（它讀的是 worker
回報的 `progress_dict`，不是 `time.time()`），而主行程的重繪執行緒照常輸出。
「log 檔還在長大、但 elapsed 不動」就是整機被掛起的指紋。

### 已做與不做的處置

**已做**：AC 的待命與休眠逾時設為「永不」。
（注意：查下來 AC 待命**本來就**是 `0x7fffff4b`＝永不，所以這一步是補強，
不是這次停擺的原因，也不會單獨解決它。）

**刻意不做**：把 DC（電池）的 180 秒也改掉。
停電時睡著是**保護行為**——不睡就會把電池耗盡然後硬斷電，
那才會真的殺掉 process。現行行為是對的。

**真正的缺口**：市電恢復後沒有任何東西會喚醒機器。
09-12 19:05 電就回來了，卻白白多睡了 36 小時。
若要補，方向是「可喚醒的排程工作」（每日固定時間喚醒一次），
而不是去動電池的睡眠策略。
