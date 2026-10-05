# 預先登記：DL-THR 疊加於「不分組」配對來源

寫定日期：2026-10-05。**本檔在跑之前定稿。**

---

## 一、起因：交易端對照沒有涵蓋論文宣稱的五個來源

論文 3.2 節寫明「本研究建構五個配對來源（不分組、GICS、HDBSCAN、Agglomerative、K-means），
在每一個來源上重複同一組交易端對照」。但 4.3 節的 DL-THR − Z-Score 主檢定實際涵蓋：

| 4.3 的「配對來源」 | 分組 | 排序 |
|:---|:---|:---|
| GICS-SSD | GICS | SSD |
| GICS-SDP | GICS | SDP |
| HDBSCAN | HDBSCAN | SDP |
| K-means | K-means | SSD |
| Agglomerative | Agglomerative | SSD |

**不分組一次也沒有出現，GICS 出現兩次，排序準則從未用 DTW。**
原因是歷史演變（`strategies/config.py` DL-THR 條目的註解）：DL-THR 先疊在三種 ML 分群上
（各取其 Z-Score 3×3 矩陣中的最佳排序——看過結果後才選），再加 GICS 兩條作交叉對照；
不分組於 2026-08-17 才以零點身分併入主網格，未補 DL-THR。

後果：5.1.2 節 Sharpe 最高的三條主軸策略全是不分組（DTW 0.529、SSD 0.455、SDP 0.436），
最佳格帳面年化 +3.641%，高於任何 DL-THR 臂的最佳格（GICS-SDP-DRL +2.66%）——
讀者看到的是「最強的配對來源沒有主要策略的版本」。

## 二、範圍（寫死，不事後增減）

- **臂**：不分組 × {SSD, DTW, SDP} 各疊 DL-THR，共 **3 條新方法**。
  **三種排序全做**，不在不分組內再依結果挑排序（避免重蹈「取最佳排序」的選擇）。
- **網格**：與既有 DL-THR 臂相同——Top {1, 3, 5, 10, 20} × 停損 {0%, 5%, 15%} = 每臂 15 格，共 **45 格**。
- **形成期**：借用 `Grid NOGRP-{SSD,DTW,SDP}` 已算好的配對（`formation_strategy_id_base`），零重跑形成期。
- **交易端**：`strategies.trading.drl_threshold_trading`，超參數與既有 DL-THR 臂完全相同
  （`drl_hidden_size` 64、`thr_train_epochs` 40、`thr_min_train_samples` 200），不調參。
- **價差空間**：DL-THR 無 `OLS_Alpha` 分支、一律走標準化空間；三條 Z-Score 對照組皆設
  `ignore_ols_alpha=True`，亦走標準化空間（2026-10-05 已逐一核對，與既有五組對照相同）。
  故唯一變因是交易端。
- **對照組**：result.db 既有的 `Grid (NOGRP-{SSD,DTW,SDP})` 15 個基準格，不重跑。
- **寫入**：result.db（與既有 DL-THR 臂同庫，使 `analysis/proposition2_daily_hac.py` 可直接加入 `PAIRS`）。
- **不在範圍內**：DTW 排序疊加於其他四種分組（GICS／HDBSCAN／Agglomerative／K-means）。
  本案只補「不分組」這個缺口；DTW 的系統性缺口於論文中揭露。

## 三、檢定（與 4.3 節完全相同的程序）

`analysis/proposition2_daily_hac.py` 的主衡量方式：每個配對來源取 15 個基準格，
逐日報酬先等權平均再相減，Δr_t = r_DL-THR − r_Z-Score；循環區塊自助法
（L = 126、B = 10,000、種子 20260804，`analysis/block_bootstrap.py`），報年化 Δ、IR、勝日%、95% CI、p。

### 多重檢定族

4.3 節的 BH-FDR 族由 5 擴為 **8**（既有 5 ＋ 不分組 3），**以 8 組的 BH 校正 p 作為論文 4.3 表的正式欄**。
**跑前即知且寫明：** 既有 3/5 顯著中，K-means 以 0.0495 通過，族擴大後可能不再通過；
若如此，論文照 8 組族的結果改寫，不保留 5 組族的版本作為主結果（可並列於附註）。

## 四、判準（只針對不分組三組；跑前寫死）

| 判定 | 條件 |
|:---|:---|
| **增益延伸至不分組** | 3 組年化 Δ 點估計**皆 > 0**，且 **≥ 2/3** 的 95% CI 下界 > 0 |
| **增益不延伸** | ≥ 2/3 點估計 ≤ 0，**或**任一組 95% CI 上界 < 0 |
| **不確定** | 以上皆非 |

另報（不入判準）：
- 8 組族的 BH 校正 p 與顯著組數；
- 後半期（2014-01-02 起）Δ（論文 5.1.2 已示範全期指標會選中後半期失效者）；
- 三條新臂的 15 格等權帳面年化、rf 超額年化、資金利用率，以及最佳格（供 5.1.2 對照）；
- SKIP 比例（4.4 節的機械性替代解釋是否同樣存在）。

### 跑前預測

既有五組方向全部為正（+0.239% ～ +0.798%）。預測不分組三組**方向為正、量級在同一範圍**；
但不分組的排名預測力最強（`dev/max_active/RESULTS.md`：配對層排名優勢 NOGRP 最高），
固定門檻在排名有效時可能已接近最佳，故 DL-THR 的增益**可能小於**既有五組。

## 五、穩健性：五輪重訓（比照 4.6.1）

DL-THR 不固定隨機種子。主檢定**只用主跑**（第 1 輪）；另跑第 2–5 輪供跨輪離散度報告，
**寫入獨立的資料庫**（`results/nogrp_dlthr_variance.db`），不覆寫 result.db 的主跑
——`tools/run_drl_variance.py` 的既有機制是每輪覆寫 result.db，本案不沿用該行為。

報告：每臂 15 格的跨輪 Sharpe 標準差與全距中位；以五輪各自計算的等權 Δ 的範圍。
若五輪中有任一輪的 Δ 方向與主跑相反，於論文中明示，且判定降一級（延伸 → 不確定）。

## 六、試驗宇宙

3 條新方法依「跑過回測即進入試驗宇宙」計入：method N **72 → 75**，config **2,280 → 2,325**
（第 2–5 輪重訓是同一批配置的重跑，不另計 config；既有作法亦然——
4.6.1 的重訓每輪覆寫 result.db 的同一批列，清點只看得到一次）。跑完後重算 var_sr 與 SR₀，
更新 `analysis/regime_cost_dsr_eval.py` 的 `TRIAL_CENSUS` 與論文 5.1.2、附錄 F.5、簡報 5.1.2。

## 七、空間與執行（2026-10-05 實測）

| 項目 | 估計 |
|:---|---:|
| 目前可用空間 | 63.1 GB |
| 既有 DL-THR 臂每臂 15 格（`GICS-SSD-DRL`：4,348,890 列 × 約 463 bytes） | 約 2.0 GB |
| 主跑 3 臂 | 約 **6 GB**（result.db） |
| 重訓第 2–5 輪（獨立庫；每輪覆寫同一批 45 格，刪除頁會被重用） | 約 6–7 GB |
| **合計** | **約 13 GB，剩餘約 50 GB** |

- 執行前確認 `git ls-files -v | grep ^S` 仍為 5 筆（形成期庫與 dataset 的 skip-worktree，
  見記憶 `lfs-rehash-fills-disk`），否則 git 檢查會把 9.7 GB 形成期庫複製進 `.git/lfs`。
- DL-THR 的並行上限依 `DRL_MAX_WORKERS`（目前 10）；執行時間以主跑實測，未事先估計。
- 長跑前人工暫停 Windows Update（記憶 `long-run-suspended-by-standby`）。

## 八、接線驗證（跑網格前必須全過）

- **V1**：在 config 加入新條目後，既有 5 條 DL-THR 臂與 3 條不分組 Z-Score 臂的展開名稱、
  檔名、formation_strategy_id 皆不變（不得意外觸發重跑或覆寫）。
- **V2**：試跑 `NOGRP-DTW` DL-THR 的 Top1/SL0 一格至試跑庫：讀到的形成期配對與 Z-Score 對照組同一批
  （逐期 Ticker 對相同），交易期日期集合相同。
- **V3**：試跑格的 SKIP 比例落在既有 DL-THR 臂的範圍內（3–4 成），作為交易端正常運作的檢查；
  若 SKIP 為 0% 或 100%，停止並排查，不開跑網格。

---

## 附記（2026-10-05，接線驗證之後補記；判準與範圍未改動）

- **V1 通過**：`strategies_raw_all` 既有 49 條逐項相同、順序與索引不變；新 3 條的 sub_dir／db_method
  皆不重複；與各自 Z-Score 條目的差異僅 `drl_hidden_size`／`thr_train_epochs`／`thr_min_train_samples`，
  與 GICS DL-THR 條目的組法相同。
- **V2 通過**：`NOGRP-DTW` DL-THR Top1/SL0 讀到 295 個配對期、6,287 個交易日，與 Z-Score 對照組完全一致。
- **V3 通過**：SKIP 比例 12.9%；既有五臂同格為 11.2%–16.9%。
- **更正 §八 V3 的括號數字**：原寫「既有 DL-THR 臂的範圍內（3–4 成）」，但以全期配對期為分母（含前 48 期
  訓練樣本不足、一律採基準動作的暖身期）實測既有五臂 Top1/SL0 為 11–17%。4.4 節的「三至四成」推測為
  排除暖身期後的比例（未驗證）。V3 的實質判準——與既有臂同格同法比較、不為 0% 或 100%——不受影響。
