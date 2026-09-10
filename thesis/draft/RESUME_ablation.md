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
