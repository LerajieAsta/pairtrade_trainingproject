# S&P 500 Pairs Trading 量化回測平台

本專案針對 S&P 500 成分股實作配對交易 (Pairs Trading) 滾動回測系統，支援多種形成期策略（含機器學習配對）、學習式門檻選擇交易期模組（DL-THR）與互動式績效視覺化儀表板。

---

本研究兩大命題：**命題1（形成期）** 機器學習分組能找到比傳統距離／共整合法更高品質的配對；
**命題2（交易期）** 以學習法選擇交易門檻（DL-THR）能比固定門檻 Z-Score 有更好的交易績效。

> **命名說明。** 程式碼與 `result.db` 中此交易端稱 `DRL`，但其學習問題為
> **全資訊監督回歸**（9 個動作的報酬皆可反事實回算，不存在探索／利用取捨），
> 並非強化學習。論文一律稱 **DL-THR**（deep-learning threshold selection）；
> 真正的部分回饋版本另實作為 **RL-THR** 作為受控對照。
> **檔名與 strategy id 維持歷史命名**，以保留與既有回測輸出的可對照性。

> **2026-07-29 命題檢定結果**（純分析層，未重跑回測；`entry_z` 對照組除外）
>
> **⚠ 統計基礎已更換。** 舊版所有 p 值以「15 個參數格」為抽樣單位
> （`top_n` 5 × `stop_loss` 3），但這 15 格共用同一份資料、同一段期間、同一批配對
> （`top_n=10` 與 `top_n=20` 共用 10 個配對；三種停損是同一批交易的不同出場規則），
> 觀測高度相關，`ttest_rel` 的獨立性假設不成立（pseudo-replication，有效樣本數 ≈ 1 條
> 回測路徑）。**現以「時間」為抽樣單位重做**：逐日報酬差 + 循環 block bootstrap
> （`analysis/proposition2_daily_hac.py`，引擎在 `analysis/block_bootstrap.py`）。
> 效果量與舊版幾乎相同（年化 +0.787% vs 舊 0.732pp）——偽重複影響的是**推論正當性**，
> 不是效果量本身。
>
> **2026-08-04 主檢定由 Newey-West HAC 改為 block bootstrap**（L=126＝一個完整交易期，
> 10,000 次重抽），同時輸出 p 值與 95% CI。兩法在全部 14 組對照上結論一致，
> 換掉的理由是可辯護性：bootstrap 無分布假設，也沒有落後階這個研究者自由度
> （落後階敏感度分析因而整組移除）。HAC 保留為各表的對照欄。
>
> **命題2 成立（相對宣稱）**——DL-THR vs 固定門檻 Z-Score，五種配對底逐日差分
> bootstrap 全部顯著（p = 0.0000–0.0060），五個 95% CI 完全落在零的右側
> （最保守下界 +0.20pp）。增益在**傳統 GICS 配對底上最大**（年化 +1.105%），
> 證明門檻選擇的增益與「配對怎麼找到」正交。
> 此處的顯著**不是檢定力僥倖**：點估計中位 +0.787pp 大於該比較自身的 MDE（0.60pp）。
>
> 三項機制對照已排除常見的替代解釋（`prop2_exposure_control.py`、`prop2_skip_permutation.py`）：
> 非源自**拉高門檻**（同門檻 2.2 對照下五支仍顯著）、
> 非源自**篩掉爛配對**（SKIP 置換檢定 75 格僅 6 格顯著，隨機期望 3.75 格）、
> 非源自**減少曝險**（DL-THR 進場次數反而是 Z-Score 的 1.55–1.92 倍）。
> 殘差與「槽位週轉」一致，惟現有資料無法直接驗證，列為後續研究。
>
> ⚠️ **門檻管道並非毫無貢獻**（2026-08-10 重跑後的更正）：舊版稱「門檻管道只複製
> 4.2–20.9% 且不顯著」，此宣稱**已不成立**。HDBSCAN 底的 ZS(2.2)−ZS(2.0) 達
> +0.246pp（p=0.014），複製 **28.4%**；五組的複製率介於 **−8.9% ~ 28.4%**，
> **一組顯著**。DL-THR 在同門檻下仍顯著勝出（+0.619pp，p=0.0052），故結論維持。
>
> **組合系統（實務上要部署的那個檢定）**——動態分群 + DL-THR vs GICS + 固定門檻，
> 排序準則已對齊，六組（3 分群 × 全期／2012+）中**五組於 BH 校正後顯著**
> （+0.63 ~ +1.51pp）。成分分解：**DL-THR 成分六組全部顯著、分群成分六組全部不顯著**
> （p = 0.122 ~ 0.954）。
> **完整系統顯著優於傳統基準，但把功勞歸給哪一半，資料還答不出來。**
> ⚠ 舊版「分群成分侵蝕了改善」的宣稱在資料回補後已不成立（六組全部轉正），不應再引用。
>
> **命題1 未獲支持**（`analysis/proposition1_daily_hac.py`，同一套逐日 bootstrap）——
> 3×3 消融矩陣的 9 組直接對照（固定排序準則與交易端，唯一變因為分組方法），
> **BH-FDR 校正後無一組達統計顯著**（校正後最小 p = 0.455；校正前最小 p = 0.063）。
> 9 組 95% CI 全部涵蓋 0 且寬 1.2–1.9pp（是 GICS 參照臂自身年化績效的數倍），
> 即「檢定力不足」而非「兩者相當」——舊版的非劣性檢定已由 CI 取代。
> 方向上 **ML 優 5 組、GICS 優 4 組**（2026-08-10 基本面資料回補後；回補前為 1:8），
> 但 9 組比較彼此不獨立（每三組共用同一個 GICS 臂），故不對「方向一致性」施加正式檢定。
> **資料條件改變的是方向，不是顯著性。**
>
> ⚠️ **表述須精確**：命題 1 主張「ML 分群能找到更好的配對」，本研究**未能拒絕虛無假設**，
> 故命題 1 不成立。但這**不等於**「ML 顯著較差」——舊版「9 組中 5 組顯著劣於 GICS」的說法
> 建立在已被否定的 n=15 偽重複基礎上，經逐日重做後不再成立，不應引用。
>
> 機制證據（獨立於顯著性）：ML 分群會把 11–25% 的股票跨產業配對，且會壓縮候選池
> （K-means 每期 20 個名額只填滿 8.7 個）。粒度掃描進一步顯示決定配對品質的是
> **候選池充足度**而非分群演算法（`analysis/granularity_sweep.py`）。
>
> **實作原因的因子檢驗**（`analysis/proposition1_mechanism.py`，2×2×2、排序固定 SSD）：
> 命題 1 的動機源自 Han, He & Toh (2021) *Pairs Trading via Unsupervised Learning*
> （CRSP 全美股、48 動量因子 + 78 公司特徵、群內做多低估/做空高估、**不施加共整合篩選**，
> 並明文指出跨產業發散亦為利潤來源）。本研究實作與其有三處差異，皆可能單獨壓抑該假說：
> 特徵含 12 維 GICS 產業 one-hot（權重 1.0）、施加共整合篩選、消融矩陣缺「不分組」零點。
> 三者全部消融後：
>
> | 分組 | 篩選 | 期均/20 | 跨產業% | 全網格等權年化% |
> | :--- | :--- | ---: | ---: | ---: |
> | GICS 產業 | coint | 15.0 | 0.0 | −0.333 |
> | AGG +one-hot | coint | 11.5 | 10.7 | −0.234 |
> | AGG −one-hot | coint | 11.3 | 39.6 | −0.300 |
> | 不分組 | coint | 19.4 | 75.3 | −0.245 |
> | GICS 產業 | none | 20.0 | 0.0 | +0.135 |
> | AGG +one-hot | none | 20.0 | 6.8 | −0.373 |
> | AGG −one-hot | none | 20.0 | 25.3 | +0.226 |
> | 不分組 | none | 20.0 | 44.7 | +0.184 |
>
> 形成期結構如預期改變——**產業先驗強度形成單調階梯**（跨產業比例 0% → 10.7% → 39.6% → 75.3%，
> 由 one-hot 權重與分組方式共同控制），且**候選池飢餓被證實為「分群 × 篩選」的交互作用**
> 而非任一單獨因子（分群+篩選 11.3–11.5/20，但不分組+篩選 19.4/20 幾乎填滿）。
>
> **惟績效未隨之改善**：20 項單因子對照經 BH-FDR 校正後僅 1 項顯著
> （拿掉 one-hot × 無篩選，Top1 口徑 +3.07pp/年，BH p=0.030），且該顯著性**未能在
> 全網格等權口徑複現**（+0.60pp，BH p=0.644）——效果只出現在每期僅持一對、噪音最大的
> 口徑，方向雖與「ML 分群與共整合篩選互斥」的假說一致，但不足以主張。
> 八種設計的等權年化全部落在 −0.37% ~ +0.23% 之間。
>
> **故本研究無法將命題 1 的否定歸因於上述實作選擇。**
>
> **交易機制的逐步歸因鏈**（`analysis/prop1_han_chain.py`）：形成期實作差異被排除後，
> 剩餘殘差之一為交易機制。Han et al. 為「群內配對、選對看月報酬發散、β=1 等金額、
> 發散即建倉、固定持有一個月」；本研究為「選對看歷史距離、OLS-β、等 z>2 建倉、
> 收斂才出場、126 日交易期」。四項差異逐步施加以維持單變因
> （直接完整復刻會一次改四個變因，與舊 SSD (Basic) 對照同型的歸因錯誤）：
>
> | 步驟 | 該步改變 | 全網格等權年化% | Top1/SL0年化% |
> | :--- | :--- | ---: | ---: |
> | 起點 `AGG-SSD-NF` | SSD 距離 + OLS-β + z>2 / 126 日 | −0.373 | −0.909 |
> | ② `HAN2-B1` | β 改 1 等金額 | −0.177 | −0.370 |
> | ③ `HAN3-REV` | 選對準則改月報酬發散 | −0.092 | +0.207 |
> | ④ `HAN4-MONTHLY` | 21 日窗 + 發散即建倉 + 持有至期末 | −0.094 | **+1.369** |
>
> **方向一致但統計上不顯著**：Top1 口徑單調改善（−0.909% → +1.369%，計 +2.28pp），
> 惟逐步對照校正前最小 $p$ = 0.055、BH 校正後無一顯著；總效果（④ − 起點）
> 等權 +0.28pp（$p$=0.923）、Top1 +2.28pp（$p$=0.705）。
> `entry_z`=0 搭配持有至期末使單一配對的日報酬波動極大，檢定力極低。
> **完整復刻 Han et al. 的交易端後，等權年化仍為 −0.094%**，離原文 24.8% 差兩個數量級。
>
> 附帶發現：`HAN3-REV` 的跨產業配對比例為 19.7%，而分群設定完全相同的 `AGG-SSD-NF`
> 僅 6.8%——**排序準則本身亦為產業偏誤的來源**（SSD 距離偏好同產業，因同業歷史價格
> 路徑天然更接近），此點在排序固定為 SSD 的因子設計中無法觀察。
>
> **特徵維度**（`analysis/prop1_feature_dimension.py`）：自既有 SEC companyfacts
> 快取解出 40 個 Green et al. 特徵（**零網路成本**——companyfacts 一次回傳全部概念），
> 12 個通過 70% 覆蓋率門檻（以 PIT 成分股身分為分母），連續維度 **7 → 19**。
> 2012+ 逐步鏈經 BH-FDR **無一顯著**，且兩大效果幾乎抵消（全域插補 +0.481pp、
> +12 特徵 −0.482pp，淨值 −0.001pp）；對 GICS 為 +0.283pp（$p$=0.535），
> **未優於** 7 維版的 +0.455pp。（此處 p 值為 block bootstrap 主檢定值；Newey-West 對照為 0.613。）
>
> ⚠️ 上述兩個中間步驟的**符號不穩定**。初稿在原始價快取缺 279 檔時執行，當時為
> 全域插補 −0.716pp、+10 特徵 +0.685pp；補齊快取後兩者皆反轉，而淨效果與
> 「全部不顯著」不變。不顯著的點估計不應作方向性解讀，此處即是實例。
>
> 補齊快取使 `market_cap` 覆蓋率 10.5% → 39.7%，帶動 7 個評價類特徵大幅上升
> （`bm` 22.4% → 83.4%、`ep` 21.1% → 80.0%），2 個越過門檻；其餘 33 個純會計特徵
> 數值**逐位元不變**——這是補抓只動到該動之處的乾淨佐證。`cfp` 以 69.99% 落選，
> 不調整門檻（那將構成事後選樣）。仍有 5 個評價類特徵卡在 57–70%，其缺口源自
> XBRL 標記本身而非價格資料。
>
> **方法論發現：產業中位數插補是第三條產業資訊管道**（`analysis/prop1_f09_reverify.py`）。
> `impute_by_group` 對缺失值填產業中位數，而結構性特徵在 2012+ 缺失 30–50%，
> 等同為缺資料的股票加上產業標籤——與 `sector_onehot_weight` 同一種混淆，此前未被計入。
> 固定特徵、僅改插補方式，2012+ 跨產業配對比例：
>
> | 臂 | 產業插補 | 全域插補 | 差 |
> | :--- | ---: | ---: | ---: |
> | AGG-BASE | 9.5% | 16.5% | +7.0 |
> | **AGG-STRUCT** | **15.7%** | **59.3%** | **+42.8** |
>
> 影響在 STRUCT 臂（缺失所在）遠大於 BASE 臂。**已封存的 F09 結構性特徵消融因此是在
> 「處理幾乎未施加」下執行**；經全域插補重驗（3 分群 × BASE/STRUCT），其「無貢獻」
> 結論**仍成立**，並由弱 null 升級為**強 null**（處理完整施加、分群大幅改變，
> BH 校正後 0/6 顯著）。新增 `impute_scope` 參數，預設 `"group"` 維持既有策略行為。
>
> **綜合——核心發現：形成期的改良難以驗證，且難在設計、不在資料量。**
> 命題 1 的兩臂持有**不同的配對集合**（特異變異無從消去），命題 2 的兩臂**共用配對**
> （差分即消噪）。兩者標準誤相差 **1.89 倍**，等效於形成期層需 **3.6 倍**樣本；
> 若要求最小可偵測效果（MDE）降至 0.3pp，約需 **14 倍**樣本、逾三個世紀的日資料。
> 對 0.3pp 的真實效果，命題 1 的檢定力僅 **11%**——即使命題 1 為真，本實驗仍有近九成
> 機率報告「不顯著」。**故此一 null 對命題 1 的真偽幾乎不具鑑別力。**
> 唯一的出路是構造**配對設計**（兩臂持同一批標的、僅變動分組），本研究未能做到。
>
> **輔證**：跨產業配對比例在六種設定下自 **0%（GICS）到 69%（不分組）**，
> 涵蓋 one-hot 權重、插補方式、特徵集、分組方式四個維度，
> 而 2012+ 績效全部落在 **−1.84% ~ −0.66%**、未觀察到可偵測的差異
> （跨產業比例與報酬統一以 2012+ 為基準；先前的 75% 係全期數字，基準不一致）。
>
> ⚠️ **此為輔證，不是等價證據。** 六種設定的績效全距為 **1.18pp**，而單一組對照的
> 信賴區間寬度即為 **1.18 ~ 1.91pp**——全部六種設定的離散度尚不及一組對照的區間寬度。
> 舊版「**選哪些配對在極大範圍內不影響報酬**」的宣稱**已撤回**：它與資料一致，
> 但同樣與「本實驗看不見 1pp 以下的效果」一致，兩者無法由該表區分；
> 本研究未施加等價檢定，故不主張之。
>
> 形成期實作差異、交易機制、特徵維度**三者皆已檢驗且皆非命題 1 失敗的原因**。
>
> **絕對績效（限制節）**：改以**全網格等權組合**為口徑後，選擇偏誤從源頭消失，
> 不再需要事後校正。六組配置（3 配對底 × 2 交易端）年化落在 **−0.36% 至 +0.77%**，
> bootstrap 檢定**無一顯著**（p = 0.30–0.97），CI 寬達 2.4–2.9pp。
> 等權組合的年化 Sharpe 僅 **0.09–0.21**，欲達統計顯著約需 **88 年以上**樣本——
> 此為結構性限制而非樣本不足。
> Deflated Sharpe 降為附錄：僅在改報「網格最佳格」時才需要
> （$N$ = **110**、SR0 = **0.408**、六族最高 SR **0.392**，DSR 落在 0.256–0.467，
> 全數不通過；涵蓋全資料庫的 **98 個策略族亦為 0/98**，最高 DSR 0.632。
> N 釘死於 `TRIAL_CENSUS`（清點日 2026-08-11），`_trial_specs` 仍實地清點以偵測漂移）。
> **命題2 與組合系統皆是「A 優於 B」的相對宣稱，不主張策略本身可獲利。**
>
> ⚠️ 本表與命題 2 差分檢定的數字**不可直接相減**：絕對檢定採**複利**口徑
> （`Daily_Delta` / 前一日權益），差分檢定採**單利**口徑（日損益 / 期初資金）。
> Agglomerative 相減得 +0.71pp、差分檢定為 +0.79pp，差距源於口徑而非資料。
>
> **與前行研究差距的受控定位**（`analysis/hsu25_entry_timing.py`）：與許鈞翔（2025）
> 逐項比對辨識出**八項**實作差異，僅隔離其中「**進場時點**」一項施加受控檢定
> （`HSU25 (…-REV)` 三條，借用對應形成期配對、僅覆寫進場觸發）。
> 結果為改採其回歸式進場後三種排序年化**全部下降**（−0.77 ~ −0.91pp，BH 後 **0/3** 顯著）
> ——方向與待解釋的差距**相反**，故此項非差距來源，但差距本身仍未獲解釋。
> 該三條已計入 DSR 試驗宇宙（N 由 107 增至 110）——**試驗數不因結果不利而豁免**。
>
> 詳見 `strategies/config.py` 第 291–300 行註解與 `analysis/` 各模組 docstring。

## 策略清單（`strategies_raw_all`，17 條，皆由 `cluster_formation.py` 中性組裝器動態產生，0-based 索引）

2026-07 中性化重構後，形成期不再是「一策略一支獨立模組」，改由**分群方法 × 群內排序準則**的
3×3 消融矩陣宣告式展開（`strategies/config.py` 第 204 行起）。舊版一策略一模組的獨立策略入口
（`HDBSCAN_Cluster_SSD_DTW.py`、`agglomerative_yF.py`、`agglomerative_FMP.py`、`ssd_basic.py`）
已封存，程式碼保留供復活（見 [archive/README.md](archive/README.md)）。
⚠️ 例外：`ssd_rolling.py`／`DTW_Cointegration_Paper.py` 雖不再是獨立策略入口，但兩者的
`Formation` 類別被新版 `_ranking.py` 動態 import 作為現役排序引擎（每次跑 formation 都會用到），
**並未真正封存**。

| # | 策略 | 分群方法 | 排序準則 | 交易期 | 角色 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 0–8 | `Grid {HDB,AGG,KM}-{SSD,DTW,SDP}` | HDBSCAN／Agglomerative／K-means | SSD／DTW／SSD-DTW-PCA | Z-Score | **命題1** 3×3 分群×排序消融矩陣（ML 分組主力，結果：不支持） |
| 9–11 | `Grid {AGG-SSD,HDB-SDP,KM-SSD} DRL` | 借用 3×3 矩陣內對應格已算好的配對 | 同上 | DL-THR | **命題2** 門檻選擇 vs 固定門檻（ML 配對底 ×3） |
| 12–14 | `Grid GICS-{SSD,DTW,SDP}` | 真實 GICS 產業（不跑分群） | SSD／DTW／SSD-DTW-PCA | Z-Score | **命題1 對照組**（傳統分組基準） |
| 15–16 | `Grid GICS-{SSD,SDP} DRL` | 借用 GICS 格已算好的配對 | 同上 | DL-THR | **命題2 對照組**：傳統配對底 + DL-THR（結果：GICS 底增益最大） |

全部 17 條策略共用單一形成期模組 `strategies.formation.cluster_formation`（`formation_module`），
唯一變因由 `cluster_method` / `ranking_backend` / `filter_mode` 三個參數決定；交易端固定二選一
（`zscore_trading` 或 `drl_threshold_trading`）。實際清單以
`Project/Scripts/python.exe -c "import strategies.config as c; [print(s['name']) for s in c.strategies_raw_all]"`
為準（會隨敏感性分析等環境變數變動）。

切換執行範圍：用環境變數免改檔覆寫（0-based Python 切片，支援逗號複合），如 `STRATEGIES_SLICE="0:9" python run_trading.py` 只跑 3×3 矩陣。

**研究框架沿革**：舊版「#1–#6 研究框架」（因子殘差化、BH-FDR、成本過濾、MST 圖候選、Beta 先驗）
隨其宿主策略（HDBSCAN Cluster、Agglomerative 等）一併封存於 2026-07-24
（`archive/config_archived_strategies.py`，commit `2fb47b6`），程式碼與歷史結果保留可復活。
現行評估層（`analysis/`：regime 分層、break-even 成本表、Deflated Sharpe、`drl_behavior.py` 決策
分解）持續適用於新版 17 策略。

**參數敏感性分析（口試委員要求）**：`config.py` 內建 OFAT 變體產生器。`$env:SENSITIVITY_ALL="1"; python run_formation.py; python run_trading.py` 一次產生 Tier-1 全部變體（`adf_pvalue_threshold`、`pca_n_components`、`beta_feature_weight`、`entry_z`），再 `python -m analysis.sensitivity_report` 看敏感度曲線。

**已封存策略**：完整清單、失敗根因診斷與復活方式見 `archive/config_archived_strategies.py` docstring；歷史回測結果保留於 `results/result.db`。封存分類索引見 [archive/README.md](archive/README.md)。

---

## 專案目錄結構

```text
pairtrade_trainingproject/
├── strategies/
│   ├── config.py                  # 全域參數、3×3 分群×排序 Grid 宣告式展開（17 條現役策略）、敏感性 OFAT 產生器
│   ├── db_utils.py                # SQLite 讀寫工具
│   ├── portfolio_manager.py       # 組合層級資金管理
│   ├── preprocess_equity.py       # 權益曲線前處理
│   ├── formation/
│   │   ├── cluster_formation.py       # ★ 中性組裝器：feature_mode × cluster_method × ranking_backend，17 條現役策略共用
│   │   ├── _clustering.py             # 分群 dispatcher（hdbscan／agglomerative／kmeans；GICS 分組不經此層）
│   │   ├── _ranking.py                # 排序 dispatcher，委派 ssd_rolling／DTW_Cointegration_Paper 的 Formation 類別
│   │   ├── _features.py / _fundamentals.py / _cointegration.py  # 特徵萃取／基本面讀取／ADF+半衰期+Hurst 篩選（各自中性、單向依賴）
│   │   ├── _utils.py                  # 共用統計工具（OLS、ADF、Hurst、_residualize_returns、_bh_fdr_threshold、_cost_viable）
│   │   ├── ssd_rolling.py / DTW_Cointegration_Paper.py  # ⚠️ 非獨立策略入口，但被 `_ranking.py` 動態 import 為現役排序引擎（不可封存）
│   │   ├── HDBSCAN_PCA_Loadings.py / HDBSCAN_Cluster_SSD_DTW.py / agglomerative_yF.py / agglomerative_FMP.py
│   │   │     # 已封存的舊版一策略一模組寫法（2026-07-24 讓位給 cluster_formation.py），程式碼保留供復活
│   │   ├── MST_PartialCorr_Cointegration.py / agglomerative_sec_pit.py / ssd_basic.py  # 已封存策略模組（負面結果，保留供復活）
│   │   ├── ml_pair_quality.py / HDBSCAN_MultiScale.py / HDBSCAN_UMAP.py / ensemble.py  # 已封存（程式碼保留）
│   │   └── __init__.py
│   └── trading/
│       ├── zscore_trading.py          # Z-Score 狀態機（基礎類，三條 Spread 路徑；現役走路徑 B）
│       ├── drl_threshold_trading.py   # DL-THR 門檻選擇模組（#9–11、#15–16）
│       └── distance_trading.py        # ⚠️ GGR 2006 距離基準——config 端已封存，檔案未搬移
├── analysis/                      # 評估層（讀 result.db，不重跑）
│   ├── block_bootstrap.py         # 主檢定引擎：循環區塊重抽 → p 值 + 95% CI（全專案唯一推論來源）
│   ├── regime_cost_dsr_eval.py    # regime 分層 + break-even 成本表 + Deflated Sharpe（DSR 僅供附錄）
│   ├── regime_cost_ew.py         #   同兩表的**等權逐格對齊**版本（4.6.1／4.6.2 用；兩臂比較須用此版）
│   ├── proposition2_stats.py      # 命題2 配對檢定（DL-THR vs 固定門檻，五種配對底）
│   ├── drl_behavior.py            # 還原 DL-THR 決策，解構增益來源
│   └── sensitivity_report.py      # OFAT 參數敏感性報表
├── fetch/
│   ├── SP500_Tiingo.py            # Tiingo API 歷史數據下載
│   ├── sp500_yf_now.py            # yFinance 當日數據更新
│   ├── fundamentals_yfinance.py   # 公司基本面靜態快照（市值、本益比）
│   ├── fetch_fmp_fundamentals.py  # FMP Point-in-Time 基本面 → parquet
│   └── fetch_sec_fundamentals.py  # SEC EDGAR XBRL PIT 基本面（+ Tiingo 原始股價）→ parquet
├── dataset/                       # 資料庫（大檔透過 Git LFS），分 price／fundamental 兩類
│   ├── price/                     # sp500_Tiingo.db（主，DB_PATH 預設）、sp500_yF.db、sp500_Current.db
│   ├── fundamental/               # fundamentals_sp500.db（快照）、sp500_pit_2000_2025_monthly.parquet（PIT，本地產物）、fmp_cache/
│   └── audit_report.csv           # 資料品質審計報告
├── formation_data/
│   └── formation_pairs_sp500_Tiingo.db  # 形成期主合併資料庫（LFS；可用 run_formation.py 完整重建）
├── notebooks/                     # 策略筆記本（Quarto revealjs 投影片；2026-07-27 trim 至論文主軸，見 notebooks/README.md）
│   ├── formation/                 # 形成期 ×6（agglomerative_fundamentals、dtw_paper_fixed、hdbscan_cluster_pca5、kmeans_fundamentals、ssd_dtw_pca_paper_fixed、ssd_rolling）
│   ├── trading/                   # 交易期 ×2（zscore、drl_threshold；distance 已隨其策略封存移除）
│   ├── comparison.ipynb           # 現役策略績效總比較（讀 config + result.db 動態產生）
│   └── main_results.ipynb         # 命題1/2 主軸結果彙整（新增）
├── docs/                          # GitHub Pages：index.html 入口 + slides/（quarto render 產出：comparison + main_results + performance_guide + formation×6 + trading×2）
├── archive/                       # 歷史存檔（分類索引見 archive/README.md）
│   ├── notebooks/ formation/ trading/ scripts/ docs/ h200/
│   └── config_archived_strategies.py  # 已封存策略 config 與完整診斷
├── tools/                         # 輔助工具（從專案根執行）
│   ├── status.py                  #   pt status：資料/形成期/交易期/投影片 狀態總覽 + 建議動作
│   ├── snapshot_run.py            #   全量重跑前歸檔 result.db
│   └── run_drl_variance.py        #   DL-THR 訓練變異數多輪評估
├── dashboard.py                   # Streamlit 績效比對儀表板
├── run_formation.py               # 形成期主程式（多行程平行）
├── run_trading.py                 # 交易期主程式（多行程平行）
├── pt.bat                         # ★ 統一指令入口（pt status / formation / trading / dashboard / slides…）
├── run.bat / setup.bat            # 一鍵啟動 Dashboard／環境初始化（保留相容）
└── requirements.txt               # Python 套件清單
```

---

## 快速啟動

所有日常操作都走統一入口 `pt.bat`（不帶參數顯示完整指令表）：

```bat
pt setup        # 1. 環境初始化（建立 Project/ 虛擬環境 + 安裝套件）
pt status       # 2. 專案狀態總覽——哪些策略缺形成期/交易期數據、附建議動作
pt all          # 3. 形成期 + 交易期一鍵連跑（或分開 pt formation / pt trading）
pt dashboard    # 4. Streamlit 績效儀表板
pt slides       # 5. 渲染全部 Quarto 投影片 → docs/slides/
```

其他：`pt variance N`（DL-THR 變異數 N 輪）、`pt snapshot tag`（重跑前歸檔 result.db）、
`pt fetch-price / fetch-fund / fetch-fmp`（資料下載）。

### 傳統呼叫方式（保留相容）

```bash
python run_formation.py    # 形成期：篩選配對 → formation_data/
python run_trading.py      # 交易期：逐日模擬 → results/result.db
```

兩個主程式均支援：
- **智慧續傳**：完成的策略/期數自動跳過（以 SQLite 資料庫為唯一真相來源；`FORCE_RERUN=True` 可強制全部重算）
- **多行程平行**：每個策略的滾動期獨立平行計算（`spawn` context，避免 CUDA fork 污染）
- **網格搜尋**：自動搜尋 Top N / Stop Loss / MSR 等參數組合
- **免改檔範圍覆寫**：環境變數 `STRATEGIES_SLICE`（如 `"5:7"`、`"0:5,8:12"`）

### 3. 查看結果

```bat
run.bat
```

啟動 Streamlit Dashboard（預設 http://localhost:8501），提供多維篩選、權益曲線對比與逐期 Trade Visualizer。

---

## 核心回測參數（`config.py`）

| 參數 | 值 | 說明 |
| :--- | :--- | :--- |
| `FORMATION_WINDOW` | 252 天 | 形成期長度 |
| `FORWARD_DAYS` | 126 天 | 交易期長度 |
| `rolling_step` | 21 天 | 滾動步長 |
| `entry_z` | 2.0 | Z-Score 進場閾值 |
| `exit_z` | 0.0 | Z-Score 出場閾值（回歸均值） |
| `max_holding_days` | 0（停用） | 時間停損；預設 0 維持既有策略行為（舊文件誤植 30，引擎從未使用該值） |
| `fee_rate` | 0.0029 | 手續費率（單邊；Do & Faff 2012 美股 pairs ~30 bps/邊） |
| `slippage_rate` | 0.0 | 滑點率（已併入 fee_rate；往返成本 = 0.29%×2 = 0.58%） |
| `RF_ANNUAL` | 0.02 | 無風險利率（超額報酬計息） |
| `INITIAL_CAPITAL` | 10,000 | 每配對初始資金 |

---

## 資料範圍與已知限制

**回測期間** 2000-01 至 2025-12（`BACKTEST_START` / `BACKTEST_END` 可用環境變數覆寫）。
成分股採**時點（point-in-time）**認定：`run_formation.py` 每期依 `index_memberships`
的 `start_date` / `end_date`，只保留該形成期結束日**當下真實在 S&P 500 內**的標的，
並納入已下市股，以降低存活者偏誤。

**存活者偏誤未完全消除（量化揭露）**：成分股名單 843 檔，但價格表僅 747 檔，
114 檔完全無價格資料，且缺失與下市狀態相關：

| | 有價格資料 | 無價格資料 |
| :--- | ---: | ---: |
| 未下市 | 608 | 65 |
| **已下市** | **121** | **49** |

即 170 檔已下市成分股中有 **49 檔（29%）無價格資料**；無價格者的已下市比率為 **43%**，
有價格者僅 **17%**——缺失並非隨機，死亡的公司缺資料的機率約為存活者的 2.5 倍。
（另：2000–2026 相異 S&P 500 成分股一般認定在 1,000–1,100 檔，843 本身亦偏低。）

**母體範圍與文獻的差距**：本研究母體為 S&P 500 大型股（每期約 600 檔），而命題 1 的
動機來源 Han, He & Toh (2021) 使用 CRSP 全美股（NYSE+AMEX+Nasdaq，數千檔），
年化 24.8% / Sharpe 2.69。該文主動排除了「獲利來自小型股」的解釋（剔除 NYSE 規模
後 20% 分位僅小幅降低獲利，且選股集中於規模上半部）。本研究的粒度掃描獨立顯示
**候選池充足度是配對品質的關鍵限制**，故母體規模差約 10 倍足以解釋部分績效落差，
方向明確（母體越小、可用配對越少、排序被迫接受更差候選）。此差異在本研究內無法實驗。

**偏誤方向（重要）**：
- 對**絕對績效**：樣本偏向倖存者 → 報酬被**高估**。本研究的絕對結論為「績效不顯著、
  DSR 全數不通過」，在被高估的樣本上仍不顯著，故該結論**偏保守**。
- 對**相對宣稱**（命題 1／命題 2）：各比較臂共用同一個殘缺標的池，偏誤同向抵銷，
  相對比較**基本不受影響**。

**公司特徵僅取得可得子集**：Han, He & Toh (2021) 使用 48 動量因子 + 78 公司特徵；
本研究自 SEC XBRL companyfacts 建出 40 個（PIT 對齊於申報日），僅 **12 個**通過
70% 覆蓋率門檻（分母為 PIT 成分股身分），連續特徵維度 7 → 19。兩層限制：

- **時間**：XBRL 強制申報自 ~2009 起，2000–2008 完全無財報特徵，故該系列實驗
  的分析限制在 2012 年後（覆蓋率穩定期）
- **11 個特徵永遠無法取得**：員工數、Compustat 上市年資、sin 股分類、
  可轉債／擔保債旗標、財報公布日 EPS 意外——XBRL 未標記或需 Compustat 授權

原始價快取的缺口（Tiingo 免費方案每月 500 個不重複代號，774 檔中一度僅取得 495 檔）
**已於 2026-08-03 補齊**，`bm`、`ep` 因而入選。其餘 5 個評價類特徵仍卡在 57–70%
（`cfp` 69.99%、`cashpr` 69.7%、`sp` 59.8%、`dy` 58.0%、`lev` 57.4%），
但其缺口源自 XBRL 標記本身而非價格資料——**補價格已無法再改善**。
門檻不因 `cfp` 僅差約五個 ticker-month 而調整，那將構成事後選樣。

另：companyfacts 快取涵蓋 843 檔中的 **625 檔（74%）**，缺的多為早期下市股
（不在 SEC 現行 ticker→CIK 對照表內），與上述存活者偏誤同源且方向相同。

**成本假設的年代**：`fee_rate` 0.29%/邊取自 Do & Faff (2012)，其估計期為 1962–2009，
本研究套用於 2000–2025。美股摩擦成本長期下降，故此假設偏保守。
Break-even 分析（`analysis/regime_cost_ew.py`，15 格等權、逐格對齊兩臂）顯示往返成本餘裕
僅 **−1.8 ~ +2.4 bps**，且 K-means／GICS-SSD／GICS-SDP 三個 Z-Score 臂**已為負**
——對成本假設高度敏感。
⚠️ 早期版本的 6.5–12.8 bps 取自**網格最佳格**（`regime_cost_dsr_eval.py`），內含 15 選 1
的選擇偏誤，與同篇「等權絕對報酬約等於零」不一致，已更正。

**Sharpe 口徑**：`Sharpe_Raw` 未扣無風險利率（`dashboard.py` 之定義）；rf 超額另見
`Excess_Ret_RF` 欄（扣 `RF_ANNUAL × 平均利用率`）。命題 2 的逐日差分檢定中 rf 於兩臂
相減時對消，不受此口徑影響。

---

## 研究框架 #1–#6 主要結論（已由 2026-07-28 命題檢定取代，見上方提要框）

> ⚠️ 本節為舊版 12+2 策略（Agglomerative 基本面、HDBSCAN Cluster+Resid 等，已於 2026-07-24
> 全數封存）時期的結論，與現行 17 策略 Grid 架構的正式命題檢定結果不一致，僅保留作歷史記錄。
> **目前有效的命題1／命題2結論見本文件開頭的提要框**及 `strategies/config.py` 第 291–300 行。

- **（歷史）最強誠實策略 = Agglomerative 基本面（yF/FMP）**：Top1 年化（動用資金）約 **+3.3~3.4%、Profit Factor ~1.2**，且在多空／高低波動各 regime 皆為正、空頭略優（逆週期分散報酬源）。
- **（歷史）#1 因子殘差化**：配強距離排序器（SSD-DTW）有效——HDBSCAN Cluster+Resid 最佳年化較原版近乎翻倍；#2 BH-FDR／#3 成本過濾能降強制平倉率，但在弱排序器路徑上救不了 Sharpe。
- **（歷史）#4 MST 偏相關圖候選** 與 **#5 Beta 風險先驗**：皆為**負面結果**——過度稀疏或加權主導反而稀釋既有的良好表徵。
- **（歷史）#6 穩健性評估**：所有策略往返 break-even 成本僅 ~0.6–0.67%（餘裕薄，符合 Do & Faff）；校正 best-of-15 選擇偏誤後 **Deflated Sharpe 全部 < 0.95**。

> 具體數值以 `results/result.db` 為準（每次重跑會更新）。完整比較見 [notebooks/comparison.ipynb](notebooks/comparison.ipynb)（讀 config + result.db 動態產生，投影片版 `docs/slides/comparison.html`）；穩健性三表見 `python -m analysis.regime_cost_dsr_eval`。

---

## 策略說明文件

- 形成期排序引擎（公式、參數、文獻標註，6 本）：[notebooks/formation/](notebooks/formation/)
- 交易期策略（Z-Score 狀態機、DL-THR v4，2 本）：[notebooks/trading/](notebooks/trading/)
- 命題1/2 主軸結果彙整：[notebooks/main_results.ipynb](notebooks/main_results.ipynb)
- 績效總比較：[notebooks/comparison.ipynb](notebooks/comparison.ipynb)
- 投影片入口（quarto render 產出）：[docs/index.html](docs/index.html) → `docs/slides/`
- 詳細開發指南：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- 已封存策略完整診斷：[archive/config_archived_strategies.py](archive/config_archived_strategies.py)
