# 一、文獻探討

<br><br>

---

<br><br>

## SSD 基礎配對交易策略 (ssd_basic.py)

* **文獻依據與來源**：
  本策略完全對應並實作了 Gatev, Goetzmann, and Rouwenhorst (2006) 的經典奠基文獻 *「Pairs Trading: Performance of a Relative-Value Arbitrage Rule」* (收錄於 `ref/2006-Pairs Trading Performance of a Relative-Value Arbitrage Rule.pdf`)。此外，對於該規則在不同市場與交易成本下的健壯性，亦與 Do and Faff (2012) *「Are Pairs Trading Profits Robust to Trading Costs?」* (收錄於 `ref/2012 - ARE PAIRS TRADING PROFITS ROBUST TO TRADING COSTS.pdf`) 的實證研究進行了參照。

<br>

* **策略依據與對比分析**：
  * **價格正規化依據**：Gatev et al. (2006) 提出無母數的「最小距離法」(Minimum Distance Method)，為了消除股票價格绝对水準的差異，必須將形成期 (Formation Period) 價格序列轉換為累積總回報指數 (Cumulative Total Returns Index)，即將首日價格歸一化為 1.0。本策略中的 `Formation.normalize_prices()` 函數完全遵循此一邏輯：
    $$I_{i, t} = \frac{P_{i, t}}{P_{i, 0}}$$
  * **平方差之和 (SSD) 依據**：文獻定義走勢相近的股票對是累積總回報指數平方差之和最小者：
    $$SSD_{A, B} = \sum_{t=1}^{T} (I_{A, t} - I_{B, t})^2$$
    本策略在 `Formation.compute_ssd()` 中利用 `scipy.spatial.distance.pdist` 的 `sqeuclidean` 距離，極速計算產業內所有股票對的兩兩 SSD。
  * **避險比例與市值中性**：Gatev et al. (2006) 經典距離法不涉及 any 參數估計（如 OLS 迴歸的 $\beta$），避險比例 (Hedge Ratio) 恆定為 1.0。這等同於在進場時對股票 A 與股票 B 進行 1:1 的等權重市值配對 (Equal dollar weighting)。本策略實作的 `Hedge_Ratio` 恆定為 1.0，完全契合該文獻設計，能有效避免因過度擬合歷史數據而產生的模型風險。

<br><br>

---

<br><br>

## 進階 SSD 配對交易策略 (ssd.py)

* **文獻依據與來源**：
  本策略參照了後續文獻對傳統距離法進行的統計與計量經濟學改進。主要依據包括 Krauss (2015) 的綜述性文獻 *「Statistical arbitrage pairs trading strategies: Review and outlook」* (收錄於 `ref/2015-Statistical arbitrage pairs trading strategies Review and outlook.pdf`)，以及對價格序列非平穩性進行處理的經典方法。

<br>

* **策略依據與對比分析**：
  * **對數價格與 Z-Score 標準化**：相較於經典 SSD 策略直接對原始價格進行歸一化，進階版策略 `ssd.py` 將價格轉換為對數價格，並進行 Z-Score 標準化以消除個股量綱（價格尺度單位）與波動率的差異：
    $$x_{i, t} = \ln(P_{i, t})$$
    $$y_{i, t} = \frac{x_{i, t} - \mu_{i}}{\sigma_{i}}$$
    where：
    * $x_{i, t}$ 為個股 $i$ 在第 $t$ 天的對數價格。
    * $\mu_i$ 與 $\sigma_i$ 分別為個股 $i$ 在形成期對數價格的均值與標準差。
    * $y_{i, t}$ 為個股 $i$ 在第 $t$ 天標準化後的對數價格。
    
    此一改進的文獻依據在於，對數價格的差值代表了連續複利報酬率之差，在金融理論上更具合理性；而 Z-Score 正規化則消除了因個股價格尺度不同而對 SSD 距離計算產生的失真，確保高波動與低波動股票在特徵空間中具有相同的權重。
  * **最小平方法 (OLS, Ordinary Least Squares) 避險比例**：本策略捨棄了經典距離法中 1:1 的硬性限制，在 `Formation.compute_ssd()` 中引入了最小平方法 (OLS) 來估計兩標準化價格之間的動態避險比例：
    $$\beta = \frac{Cov(y_A, y_B)}{Var(y_B)}$$
    這使得資金分配在進場時能依據兩檔股票的相對波動彈性進行市值中性分配（Beta Weighting），避免了當股票 A 波動大於股票 B 時，等額分配資金所導致的淨風險敞口暴露。
  * **交易期參數固定性**：相較於其他計量經濟模型，本策略在交易期內完全固定形成期所估計之避險比率 $\beta_{form}$，以確保策略的低回轉率與執行穩定度。

<br><br>

---

<br><br>

## HDBSCAN 降維密度分群共整合策略 (HDBSCAN.py)

* **文獻依據與來源**：
  本策略結合了機器學習無監督分群與傳統計量共整合方法，其文獻依據源自近年將機器學習應用於配對選擇的前沿研究。主要對應的文獻為 *「Pairs Trading via Unsupervised Learning」* (收錄於 `ref/2021-Pairs Trading via Unsupervised Learning.pdf`)，以及台灣本土研究 *「透過機器學習及標記技術建構配對交易策略」* (收錄於 `ref/2021-透過機器學習及標記技術建構配對交易策略.pdf`)。

<br>

* **策略依據與對比分析**：
  * **多維特徵空間與降維依據**：傳統配對交易在大規模市場中，面臨兩兩配對計算的幾何級數複雜度危機 ($O(M^2)$)，且容易因隨機漂移篩選出走勢相似但本質毫無關聯的「偽配對」(Spurious Pairs)。上述無監督學習文獻提出，應先對股票的歷史時序特徵進行擷取（包含動量、波動率、自相關、統計矩等多維度特徵），再利用 PCA 或 UMAP 進行流行降維。本策略實作的 `Formation._build_feature_matrix()` 完整擷取了 13 維時序金融特徵，並支持 UMAP 降維至低維空間，其依據正是為了保留流形結構的同時降低雜訊。
  * **HDBSCAN 密度分群與噪聲過濾**：相較於 K-Means 需要預先設定分群數，且會將所有噪聲點強行歸類的缺點，文獻更推薦使用基於密度的分群演算法 (DBSCAN/HDBSCAN)。HDBSCAN 能自動識別任意形狀的群落，並將不符合密度特徵的噪聲點標記為 -1 予以排除。本策略的 `Formation._hdbscan_cluster()` 完全契合無監督學習配對交易文獻的這一要求，極大提升了配對池的同質性。
  * **Engle-Granger (EG) 共整合篩選**：分群完成後，策略規定僅在「同產業」且「同群落」內才進行 Engle-Granger 共整合檢定與 ADF 檢定。這在文獻中被證實能顯著提高共整合的通過率，並且篩選出的配對在交易期具有更強的均值復歸動力。

<br><br>

---

<br><br>

## HDBSCAN 多因子特徵空間共整合策略 (HDBSCAN_MultiFactor.py)

* **文獻依據與來源**：
  本策略為無監督學習分群配對交易的進一步昇華，其理論依據除了前述的無監督學習配對交易外，更深入融合了現代資產定價的多因子理論（Multi-Factor Pricing Theory）。主要參照文獻包括 *「In Search of Pairs using Firm Fundamentals」* (收錄於 `ref/2021-In Search of Pairs using Firm Fundamentals.pdf`) 以及台灣金融實證研究如 *「結合共整避險比率與混合獎勵設計之強化學習配對交易模型」* (收錄於 `ref/2025-結合共整避險比率與混合獎勵設計之強化學習配對交易模型_朱羿璁.pdf`)。

<br>

* **策略依據與對比分析**：
  * **金融學多因子特徵空間的建構**：相較於 `HDBSCAN.py` 中 13 維特徵可能存在的統計冗餘，且需依賴降維算法（以降維可能產生的信息折損為代價），本策略直接提取金融學上最具經濟解釋力與統計穩健性的 6 大時序因子（Market Beta、Volatility、Skewness、Kurtosis、Trend Slope、Idiosyncratic Volatility）。文獻指出，Idiosyncratic Volatility (特異波動度) 與 Market Beta 反映了資產結構性的風險暴露；Trend Slope 則描述了資產的長期增長路徑。直接以這 6 大特徵建構特徵空間，在分群上具有極強的金融學邏輯支撐。
  * **無降維密度分群**：本策略在 `Formation.run()` 中將降維方法設為 `none`，跳過降維步驟，直接在 6 維高度結構化的多因子特徵空間中執行 HDBSCAN 分群。這符合近年計量金融文獻中「特徵工程優於無腦降維」的共識，完整保留了多因子空間的幾何結構。
  * **同產業 × 同因子群落共整合**：通過 6 大金融因子密度分群出的股票，代表其在市場系統風險、特異風險、長期趨勢與收益分佈特性上高度同質。在此同質群落內進一步施以 EG 共整合檢定與 O-U 均值復歸半衰期過濾，篩選出的配對在交易期展現出極為穩健且迅速的均值復歸特性，極大程度防範了基本面分歧導致的「配對漂移」風險。

<br><br>

---

<br><br>

# 二、研究方法

<br><br>

---

<br><br>

## 資金管理方式 (Money Management & Capital Allocation)

配對交易作為一種統計套利策略，其生死存亡高度取決於資金管理與風險控管的設計。本研究的回測系統建構了極為嚴密且系統化的「三層資金分配與四道防禦熔斷」資金管理模組：

### 1. 三層資金分配體系
* **第一層：滾動重疊視窗之可用資金槽分配 (Capital Slot Allocation)**：
  在長期的滾動回測中，交易期窗口 (`trading_window`) 與滾動步長 (`rolling_step`) 往往存在重疊（例如每 1 個月滾動一次，但交易期為 6 個月，此時會有 6 個並行的交易期）。為了避免資金在時間軸上產生超額佔用或分配衝突，主引擎 `RollingBacktester` 實作了「資金槽」機制。
  系統計算最大並行數 $Max\_Concurrent = Trading\_Window / Rolling\_Step$。將初始總資金均分為 $Max\_Concurrent$ 個獨立的資金槽（Slots），每個槽分配初始資金：
  $$Capital_{slot} = \frac{Initial\_Capital}{Max\_Concurrent}$$
  每一期滾動視窗開始時，系統會掃描並挑選「目前可用日期（`avail_idx`）最早且小於或等於當期交易開始日」的資金槽，並將該槽當時累積的總資金 `current_period_capital` 全部投入該期交易。期末結算後，將該期的所有盈虧回撥至該資金槽，並更新其可用日期為當期交易結束日。這確保了多個並行交易期之間的資金流動性完全隔離，防止資金超限。
* **第二層：個股配對間的等額資金分配 (Capital Allocation Per Pair)**：
  在某一滾動交易期中，系統經篩選共選定 $N$ 組配對（由 `top_n` 參數決定）。為防範單一配對集中風險，該期所獲配的總資金將被均等分配給這 $N$ 組配對，即每一配對獲配的交易額度為：
  $$Capital_{pair} = \frac{Current\_Period\_Capital}{N}$$
* **第三層：配對內部市值中性資金分配 (Intra-pair Weighting)**：
  * **等權重市值中性 (Dollar Neutrality, 1:1)**：
    適用於 `ssd_basic.py`。不考慮兩股的價格彈性，資金在股票 A 與股票 B 間均分（多空市值各半）：
    $$v_A = 0.5 \times Capital_{pair}, \quad v_B = 0.5 \times Capital_{pair}$$
    股票 A 與 B 的交易股數則分別為：$Shares_A = \pm v_A / P_A$， $Shares_B = \mp v_B / P_B$。
  * **避險比例市值中性 (Beta Neutrality / Hedge Ratio Weighting)**：
    適用於 `ssd.py`、`HDBSCAN.py`、`HDBSCAN_MultiFactor.py`。為了確保在對數價格波動下的市值中性，資金分配必須與避險比例 $\beta$ (Hedge Ratio) 掛鉤。
    設配對避險比例為 $\beta$，則多空總市值權重為 $Total\_Weight = 1.0 + |\beta|$。兩股獲配的交易金額分別為：
    $$v_A = Capital_{pair} \times \frac{1.0}{1.0 + |\beta|}$$
    $$v_B = Capital_{pair} \times \frac{|\beta|}{1.0 + |\beta|}$$
    此一設計保證了當股票 A 上漲 1% 時，其多頭/空頭部位的價值變動，能與股票 B 上漲 $\beta\%$ 時的部位價值變動相抵消，達成了計量經濟學意義上的動態風險中性。

<br>

### 2. 四道風險防禦與熔斷機制（包含進場冷卻）
* **第一道防線：單一配對停損 (Stop Loss Percent)**：
  在每日交易模擬中，系統實時監控每個配對的已實現盈虧與未實現盈虧。若單一配對的累計損失佔其分配資金 $Capital_{pair}$ 的比例達到門檻 `stop_loss_pct` (例如 5% 或 10%)：
  $$\frac{-(Unrealized\_PnL + Realized\_PnL)}{Capital_{pair}} \ge Stop\_Loss\_Pct$$
  則立刻觸發該配對的停損機制，當日以收盤價強行平倉。若設定 `allow_reentry = False`，則該配對在當期交易期剩餘時間內被永久標記為 `STOPPED`，不准重新進場。
* **第二道防線：動態 Z-Score 偏離停損 (Dynamic Z-Score Stop)**：
  配對交易的核心前提是「均值復歸」。若價差 Z-Score 發生異常偏離（例如價格關係破裂，導致 Z-Score 超出合理範疇，如 $|Z_t| > Dynamic\_Stop\_Z = 3.0$），此時極有可能是基本面發生了不可逆的結構性轉變（如並購、暴雷等），均值復歸前提失效。系統將主動介入，觸發 Z-Score 異常停損並平倉，以防止「價差無限發散」帶來毀滅性虧損。
* **第三道防線：停損與平倉後之冷卻期限制 (Cooldown Period / Re-entry Protection)**：
  **本系統在底層代碼中完整實作了基於方向判定的冷卻期機制。**
  當某個配對因為觸發平倉（`EXIT`）或停損（`STOP_LOSS_TRIGGERED`）出場後，若價差沒有回到合理的復歸水準，立刻重新開倉將面臨價格趨勢連續發散的巨大風險。
  為此，系統在平倉當日會將持倉方向記錄在冷卻狀態 `cooldown_dir` 中（例如做多 A 空 B 平倉後記錄為 1，做空 A 多 B 則為 -1）。只要價差 $Z_t$ 依然停留在極端區域：
  * 若 `cooldown_dir == -1` (前次做空 A 多 B)，只要價差 $Z_t > Exit\_Z$，則冷卻持續，**禁止再次開空開倉**。
  * 若 `cooldown_dir == 1` (前次做多 A 空 B)，只要價差 $Z_t < -Exit\_Z$，則冷卻持續，**禁止再次開多開倉**。
  
  只有當 Z-Score 回歸到平倉線以內，即滿足下列解鎖條件時：
  * `cooldown_dir == -1` 且 $Z_t \le Exit\_Z$ 
  * `cooldown_dir == 1` 且 $Z_t \ge -Exit\_Z$
  
  冷卻狀態才會重置為 0，正式解鎖並允許下一次信號進場。這道防線能有效避免在趨勢市中被連續停損打擊。
* **第四道防線：投資組合總體止損斷路器 (Portfolio-level Circuit Breaker)**：
  這是防範系統性風險與極端市場衝擊（黑天鵝事件）的終極熔斷機制。
  在每日模擬結束時，主引擎會統計該期所有配對的總累計虧損。若總累計虧損佔該期總分配資金的比例達到了斷路器門檻 `portfolio_stop_loss_pct` (例如 10%)：
  $$\frac{-\sum_{k=1}^{N} Cumulative\_PnL_k}{Current\_Period\_Capital} \ge Portfolio\_Stop\_Loss\_Pct$$
  則立刻觸發**投資組合熔斷**！當日將該期所有持倉配對強行平倉（交易明細狀態標記為 `PORTFOLIO_STOP_TRIGGERED`），並且在當期交易窗口剩餘的所有日期中，將所有配對狀態強制設定為 `STOPPED`，禁止任何新的開倉與交易。此舉能將極端行情下的最大虧損牢牢鎖定在設定的門檻內，保障帳戶的生存能力。

<br><br>

---

<br><br>

## 各策略說明

本節將統一排版格式，以「**策略架構**」、「**形成期**」、「**交易期**」三大層級對各策略進行深度計量解析，並將數學公式與變數定義進行無縫呈現。所有策略的 `zscore_window` 均固定為 0，即完全採用靜態固定參數模式進行回測。

<br><br>

---

<br><br>

## SSD 基礎配對交易策略 (ssd_basic.py)

* **策略架構**：
  採用經典的無母數距離法，完全以累積總回報指數的幾何距離作為配對標準，交易時採取等額資金中性分配，追求極簡與穩健。

<br>

* **形成期**：
  1. **價格累積歸一化**：對形成期內的價格矩陣進行處理，以形成期首日價格作為分母，將個股價格轉化為累積總回報指數：
     $$I_{i, t} = \frac{P_{i, t}}{P_{i, 0}}$$
     其中：
     * $I_{i, t}$ 為個股 $i$ 在形成期第 $t$ 天的累積總回報指數。
     * $P_{i, t}$ 與 $P_{i, 0}$ 分別為個股 $i$ 在第 $t$ 天與首日的價格。
  2. **計算平方差之和 (SSD)**：在設定的產業分類內，對任意兩股票對 $A$ 與 $B$，計算其累積總回報指數的平方差之和（即平方每日價格差之和，用以衡量走勢相近度）：
     $$SSD_{A, B} = \sum_{t=1}^{T} (I_{A, t} - I_{B, t})^2$$
     其中：
     * $SSD_{A, B}$ 為股票 $A$ 與 $B$ 之間的累積平方差（即兩兩 Euclidean 平方距離）。
     * $T$ 為整個形成期的交易日數。
  3. **配對挑選**：依 $SSD_{A, B}$ 升序排列，篩選出 SSD 最小的前 `top_n` 組配對。
  4. **統計參數錨定**：計算選定配對在形成期內價差 $Spread_t = I_{A, t} - I_{B, t}$ 的均值 $\mu_{spread}$ 與標準差 $\sigma_{spread}$：
     $$\mu_{spread} = \frac{1}{T} \sum_{t=1}^T Spread_t$$
     $$\sigma_{spread} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (Spread_t - \mu_{spread})^2}$$
     避險比率 $Hedge\_Ratio$ 恆定錨定為：
     $$Hedge\_Ratio = 1.0$$

<br>

* **交易期**：
  1. **Z-Score 計算**：在交易期內，每日以形成期首日價格將股票價格正規化，計算即時價差，並利用形成期的固定統計量 $\mu_{spread}$ 與 $\sigma_{spread}$ 標準化為 Z-Score（無滾動動態視窗）：
     $$Spread_t = I_{A, t} - I_{B, t}$$
     $$Z_t = \frac{Spread_t - \mu_{spread}}{\sigma_{spread}}$$
     其中：
     * $Spread_t$ 為第 $t$ 天的即時價差。
     * $Z_t$ 為第 $t$ 天價差標準化後的 Z-Score。
     * $\mu_{spread}$ 與 $\sigma_{spread}$ 分別為形成期計算得到的固定均值與標準差。
  2. **交易信號與冷卻防禦**：
     * **冷卻解除判定**：每日進場前先檢視冷卻狀態，若前次交易為做空 A 多 B (`cooldown_dir == -1`) 且 $Z_t \le Exit\_Z$，或前次為做多 A 空 B (`cooldown_dir == 1`) 且 $Z_t \ge -Exit\_Z$，則冷卻狀態解鎖，重置 `cooldown_dir = 0`。
     * **開空 A 多 B**：當 $Z_t > Entry\_Z$、無持倉且 `cooldown_dir != -1` 時，開倉賣空 A、買入 B。
     * **開多 A 空 B**：當 $Z_t < -Entry\_Z$、無持倉且 `cooldown_dir != 1` 時，開倉買入 A、賣空 B。
     * **平倉與停損信號**：當持倉中且價差復歸至 $|Z_t| \le Exit\_Z$ 時平倉；或觸發單一配對停損（虧損達 `stop_loss_pct`）時強行停損平倉。平倉後記錄當前持倉方向至 `cooldown_dir` 中啟動冷卻保護。
     * **期末平倉**：交易期最後一天（`PERIOD_END_EXIT`）若仍有持倉，則強行平倉結算。

<br><br>

---

<br><br>

## 進階 SSD 配對交易策略 (ssd.py)

* **策略架構**：
  以對數正規化價格消除量綱與波動率偏誤，並以 OLS 估計動量彈性 $\beta$ 作為避險與市值中性權重。

<br>

* **形成期**：
  1. **價格對數 Z-Score 化**：將形成期價格轉為對數價格，並對每支股票的全期數值進行 Z-Score 標準化，消除量綱（價格尺度大小）差異：
     $$x_{i, t} = \ln(P_{i, t})$$
     $$y_{i, t} = \frac{x_{i, t} - \mu_{i}}{\sigma_{i}}$$
     其中：
     * $x_{i, t}$ 為個股 $i$ 在第 $t$ 天的對數價格。
     * $\mu_i$ 與 $\sigma_i$ 分別為個股 $i$ 在形成期對數價格的均值與標準差。
     * $y_{i, t}$ 為個股 $i$ 在第 $t$ 天標準化後的對數價格。
  2. **計算 SSD**：計算標準化對數價格的兩兩平方差之和並排序。
  3. **OLS 避險參數估計**：對排序靠前的配對，以普通最小平方法 (OLS, Ordinary Least Squares) 估計避險比例 $Hedge\_Ratio$ ($\beta$)：
     $$\beta = \frac{Cov(y_A, y_B)}{Var(y_B)}$$
     其中：
     * $\beta$ 為股票 $A$ 相對於股票 $B$ 的對數正規化價格避險比例。
  4. **價差統計錨定**：計算形成期價差 $Spread_t = y_{A, t} - \beta y_{B, t}$ 的均值 $\mu_{spread}$ 與標準差 $\sigma_{spread}$：
     $$\mu_{spread} = \frac{1}{T} \sum_{t=1}^T Spread_t$$
     $$\sigma_{spread} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (Spread_t - \mu_{spread})^2}$$

<br>

* **交易期**：
  1. **即時價差與 Z-Score**：在交易期內，每日計算即時價差，並完全採用形成期估計的固定 $\beta$、$\mu_{spread}$ 與 $\sigma_{spread}$ 計算固定 Z-Score（無滾動動態視窗）：
     $$Spread_t = y_{A, t} - \beta y_{B, t}$$
     $$Z_t = \frac{Spread_t - \mu_{spread}}{\sigma_{spread}}$$
     其中：
     * $Spread_t$ 為第 $t$ 天的即時價差.
     * $Z_t$ 為第 $t$ 天的固定 Z-Score。
  2. **信號與資金管理**：交易開平倉信號與 ZScore 判定一致。進場前先檢視冷卻狀態，符合開倉信號且未處於該方向冷卻期（`cooldown_dir` 限制）時，依避險比例 $\beta$ 進行 $1 : |\beta|$ 的多空資金市值分配。平倉或停損後，將持倉方向記錄至 `cooldown_dir` 中啟動冷卻防禦。

<br><br>

---

<br><br>

## HDBSCAN 降維密度分群共整合策略 (HDBSCAN.py)

* **策略架構**：
  利用高維金融特徵空間表徵個股，經流形降維後使用 HDBSCAN 自動密度分群，再在同產業同分群內進行 EG 共整合檢定與半衰期過濾，選取統計上最具均值復歸特性的配對。

<br>

* **形成期**：
  1. **擷取 13 維金融時序特徵**：為形成期內每支股票的對數價格序列計算以下 13 維特徵，並進行 `StandardScaler` 標準化。這 13 維特徵包括：
     
     * **A. 動量特徵 (Momentum, 4維)**：
       計算 5日、21日、63日、126日的對數累積報酬率：
       $$Ret_{\tau} = \ln(P_T) - \ln(P_{T-\tau})$$
       其中 $\tau \in \{5, 21, 63, 126\}$，分別代表大約 1 周、1 個月、1 季、半年的累積報酬。多尺度的動量能夠刻畫個股的多尺度趨勢軌跡，幫助聚類算法將具有相似價格運動趨勢的個股進行聚類。
       
     * **B. 波動度特徵 (Volatility, 3維)**：
       計算 21日、63日滾動報酬率標準差，以及全形成期日報酬率的總標準差：
       $$Vol_{\tau} = \sqrt{\frac{1}{\tau-1} \sum_{t=T-\tau+1}^T (R_t - \bar{R}_{\tau})^2}$$
       其中 $R_t$ 為日報酬率，$\bar{R}_{\tau}$ 為該窗口內報酬率的均值，$\tau \in \{21, 63, All\}$。這反映了個股的總體風險與波動特性，確保同分群的股票具有相近的價格震盪幅度，防範高風險暴漲股與低風險防禦股的無效配對。
       
     * **C. 自相關特徵 (Autocorrelation, 3維)**：
       計算收益率序列在滯後一階 (Lag-1)、滯後五階 (Lag-5) 及滯後二十一階 (Lag-21) 的 Pearson 自相關係數：
       $$\rho_k = \frac{\sum_{t=k+1}^T (R_t - \bar{R})(R_{t-k} - \bar{R})}{\sum_{t=1}^T (R_t - \bar{R})^2}$$
       其中 $k \in \{1, 5, 21\}$。自相關係數用以捕捉個股日報酬的均值復歸或動量持續特性，確保同分群的資產在微觀交易行為與價格復歸節奏上高度合拍。
       
     * **D. 統計矩與分形特徵 (3維)**：
       * *偏態 (Skewness)*：衡量收益率分佈的非對稱性，描述股票是否存在極端正偏（暴漲）或負偏（暴跌）傾向。
       * *峰態 (Kurtosis)*：衡量收益率分佈的胖尾程度，反映個股承受極端黑天鵝事件的尾部風險。
       * *Hurst 指數近似值*：透過簡化版重標極差 (R/S) 分析進行估計。將形成期收益率劃分為多個不同長度的子區間（全期、1/2全期、1/4全期），在各區間上計算累積偏差的極差並除以標準差，最後對區間長度與 R/S 值取對數進行 OLS 迴歸擬合：
         $$\ln(R/S)_d = H \times \ln(d) + C$$
         擬合直線斜率 $H$ 即為 Hurst 指數。$H \approx 0.5$ 表隨機漫步，$H > 0.5$ 代表強勢持續性（Trend），$H < 0.5$ 代表強烈均值復歸性（Mean-reverting）。聚類 Hurst 指數能直接挑選出具備天然套利體質的資產群落。
  
  2. **流形降維 (UMAP/PCA)**：使用 UMAP 降維算法（或 PCA）將 13 維標準化特徵投影至低維嵌入空間（如 5 維），保留非線性流形結構並排除高維噪聲。
  3. **HDBSCAN 密度分群**：在低維空間中執行 HDBSCAN，自動決定最佳群落數，並將處於稀疏邊緣的噪聲股票標記為 -1 直接剔除。
  4. **產業分群雙重篩選與計量 EG 檢定**：
     規定候選配對必須同屬於某一產業分類且同屬於某一 HDBSCAN 群落（且 label $\ne$ -1）。符合條件的配對，進行雙向 Engle-Granger (EG) 共整合 OLS 迴歸：
     $$\ln(P_{A, t}) = \alpha + \beta \ln(P_{B, t}) + \epsilon_t$$
     其中：
     * $\ln(P_{A, t})$ 與 $\ln(P_{B, t})$ 分別為個股的對數價格。
     * $\beta$ 為共整合避險比率。
     * $\epsilon_t$ 為共整合殘差序列。
     對殘差 $\epsilon_t$ 進行無常數項的 ADF 檢定，取得 ADF 統計量與 p 值。
  5. **均值復歸半衰期過濾**：
     對共整合殘差進行 AR(1) 迴歸，估計均值復歸速度 $\lambda$：
     $$\Delta \epsilon_t = \gamma_0 + \lambda \epsilon_{t-1} + u_t$$
     計算均值復歸半衰期 $Half\_Life$：
     $$Half\_Life = -\frac{\ln(2)}{\lambda}$$
     其中：
     * $\lambda$ 為均值復歸速度（必須為負值才具備均值復歸性）。
     * 規定半衰期必須在 2 天至 60 天之間，排除復歸過快（高頻噪音）或過慢（交易機會過少）的配對。
  6. **配對排序與參數錨定**：通過篩選後，依 ADF 統計量升序排序，選出共整合最顯著的前 `top_n` 組配對。並錨定形成期殘差 $\epsilon_t$ 的固定均值 $\mu_{spread}$ 與標準差 $\sigma_{spread}$：
     $$\mu_{spread} = \frac{1}{T} \sum_{t=1}^T \epsilon_t$$
     $$\sigma_{spread} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (\epsilon_t - \mu_{spread})^2}$$

<br>

* **交易期**：
  1. **價差與 Z-Score**：在交易期內，採用形成期估計的固定 OLS 參數 $\alpha$、$\beta$ 以及殘差統計量 $\mu_{spread}$ 與 $\sigma_{spread}$，計算固定 Z-Score（無滾動動態視窗）：
     $$Spread_t = \ln(P_{A, t}) - \alpha - \beta \ln(P_{B, t})$$
     $$Z_t = \frac{Spread_t - \mu_{spread}}{\sigma_{spread}}$$
  2. **信號與冷卻防禦**：依 Z-Score 信號開平倉。進場前先檢視冷卻狀態，符合開倉信號且未處於該方向冷卻期（`cooldown_dir` 限制）時，按 $\beta$ 比例進行動態市值中性資金分配。平倉或停損後，將持倉方向記錄至 `cooldown_dir` 中啟動冷卻保護。

<br><br>

---

<br><br>

## HDBSCAN 多因子特徵空間共整合策略 (HDBSCAN_MultiFactor.py)

* **策略架構**：
  直接以 6 大金融學系統與特異風險因子作為特徵空間，跳過降維的信息折損，以 HDBSCAN 密度分群尋找基本面與風險高度同質的個股，再通過 EG 共整合與半衰期過濾進行終極篩選。

<br>

* **形成期**：
  1. **萃取 6 大穩健金融因子**：為每支股票的形成期時序數據計算以下 6 大時序特徵，並進行 `StandardScaler` 標準化。這 6 個因子融合了資本資產定價與系統性風險特徵：
     
     * **A. Market Beta (市場貝他)**：
       計算個股收益率 $R_{i, t}$ 與市場等權重日度平均收益率 $R_{m, t}$ 之間的迴歸斜率：
       $$Beta_i = \frac{Cov(R_{i}, R_{m})}{Var(R_{m})}$$
       這是資本資產定價模型 (CAPM) 的核心指標，衡量股票對市場整體波動的敏感度。同群落個股若 Beta 相近，代表在遭遇宏觀市場衝擊時具有等比例、同方向的系統性反應，保證了配對在市場端的天然對沖。
       
     * **B. Volatility (日報酬總波動度)**：
       個股日度收益率在全期內的標準差 $\sigma_i$，用以衡量個股的總體風險暴露。同分群的股票應具有相同的總體不確定性，防範波動懸殊的資產進行無效配對。
       
     * **C. Skewness (偏度)**：
       個股日度收益率分佈的非對稱性：
       $$Skew_i = E\left[ \left( \frac{R_i - \bar{R}_i}{\sigma_i} \right)^3 \right]$$
       刻畫收益分佈的非對稱偏斜特質，聚類相同偏度的股票能確保配對兩側在面臨極端利多或利空時具有對稱性的反應。
       
     * **D. Kurtosis (峰度)**：
       個股日度收益率分佈的峰度：
       $$Kurt_i = E\left[ \left( \frac{R_i - \bar{R}_i}{\sigma_i} \right)^4 \right]$$
       衡量收益分佈的極端肥尾風險。高峰度資產容易出現極端黑天鵝事件，同群落歸類能防範配對兩側因极端的極端噪聲干擾而發生均值復歸軌跡的偏離。
       
     * **E. Trend Slope (價格長期趨勢斜率)**：
       將個股對數價格 $\ln(P_t)$ 相對於時間序列 $t = [0, 1, 2, ..., T]$ 進行 OLS 線性迴歸：
       $$\ln(P_{i, t}) = \alpha_i + Slope_i \times t + \epsilon_{i, t}$$
       其中 $Slope_i$ 即為長期價格對數增長的速率。如果配對的股票長期斜率不一致，其價差便會產生非平穩的結構性趨勢發散（Spurious Drift）。聚類相同趨勢斜率的股票，是確保價差在交易期穩定圍繞恆定均值復歸的基石。
       
     * **F. Idiosyncratic Volatility (特異波動度, IVOL)**：
       個股日度收益率對市場收益率進行 CAPM 迴歸：
       $$R_{i, t} = \alpha_i + \beta_i R_{m, t} + e_{i, t}$$
       計算迴歸殘差項 $e_{i, t}$ 的樣本標準差。這代表了個股無法被市場系統性因子所解釋的「特異風險（個股特有基本面風險）」。配對交易本質上是用特異風險的復歸來賺取 Alpha。聚類 IVOL 相近的股票，能確保配對兩側暴露在等量且可互相抵消的個股不確定性中，鎖定更純粹的統計套利利潤。
  
  2. **直接密度分群**：跳過降維步驟（`reduce_method = "none"`），直接將 6 維因子特徵矩陣送入 HDBSCAN 進行密度分群，過濾標記為 -1 的噪聲股票。這完整保留了多因子定價特徵空間的幾何結構。
  3. **同產業與計量共整合過濾**：在同產業且同因子群落內，進行雙向 Engle-Granger 共整合迴歸、ADF 統計量與 p 值檢定：
     $$\ln(P_{A, t}) = \alpha + \beta \ln(P_{B, t}) + \epsilon_t$$
     並計算 O-U 均值復歸半衰期，進行雙重過濾：
     $$Half\_Life = -\frac{\ln(2)}{\lambda}$$
     篩選半衰期合格的配對，並錨定形成期殘差 $\epsilon_t$ 的均值 $\mu_{spread}$ 與標準差 $\sigma_{spread}$：
     $$\mu_{spread} = \frac{1}{T} \sum_{t=1}^T \epsilon_t$$
     $$\sigma_{spread} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (\epsilon_t - \mu_{spread})^2}$$
     最終選出最優的前 `top_n` 組配對。

<br>

* **交易期**：
  1. **價差計量與 Z-Score**：基於對數價格進行靜態 OLS 價差與 Z-Score 計算：
     $$Spread_t = \ln(P_{A, t}) - \alpha - \beta \ln(P_{B, t})$$
     $$Z_t = \frac{Spread_t - \mu_{spread}}{\sigma_{spread}}$$
  2. **交易執行與冷卻防禦**：基於 Z-Score 執行開平倉，進場前先檢視冷卻狀態，符合開倉信號且未處於該方向冷卻期（`cooldown_dir` 限制）時，按 $\beta$ 比例進行動態市值中性資金分配，平倉或停損後，將持倉方向記錄至 `cooldown_dir` 中啟動冷卻保護。啟用最嚴格的單一配對停損、動態 Z-Score 停損與投資組合總體斷路器熔斷機制。

<br><br>

---

<br><br>

## 各策略對比分析

經過對回測系統架構與代碼底層的專家級全景審查，這四個策略在**形成期、特徵工程、選股檢定、避險計量、資金分配與交易機制**等全流程上，存在著極為清晰且系統化的計量演進關係。以下為你提供最完整的全維度對比解析：

### 1. 全維度核心架構對比表

| 比較維度 | SSD 基礎配對策略 (`ssd_basic.py`) | 進階 SSD 配對策略 (`ssd.py`) | HDBSCAN 共整合策略 (`HDBSCAN.py`) | HDBSCAN 多因子共整合策略 (`HDBSCAN_MultiFactor.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **A. 核心理念** | 經典價格距離最小化 | 考慮對數波動標準化之改進距離法 | 高維流形密度分群與 EG 共整合 | 金融多因子特徵分群與 EG 共整合 |
| **B. 價格基礎** | 原始價格正規化 (累積總回報指數 $I_t$) | 對數價格且 Z-Score 標準化 ($y_t$) | 原始對數價格 ($\ln(P_t)$) | 原始對數價格 ($\ln(P_t)$) |
| **C. 形成期篩選機制** | 產業內兩兩計算幾何距離平方和 (SSD) | 產業內兩兩計算對數 Z-Score 平方和 (SSD) | 13維特徵 + UMAP降維 + HDBSCAN分群 +EG共整合雙重過濾 | 6大金融因子特徵 + HDBSCAN直接分群 + EG共整合雙重過濾 |
| **D. 特徵工程維度** | 無（僅使用價格） | 無（僅使用對數價格） | 13 維金融時序指標（動量、波動、自相關、統計矩、Hurst指數） | 6 大金融系統/特異風險因子（Beta、總波動、偏峰度、趨勢、IVOL） |
| **E. 降維處理** | 無 | 無 | 流形降維（UMAP / PCA 至 5維空間） | **無**（直接在 6維多因子空間分群，無信息折損） |
| **F. 計量共整合檢定** | 無 | 無 | 有（Engle-Granger 雙向 OLS 與 ADF 定態檢定） | 有（Engle-Granger 雙向 OLS 與 ADF 定態檢定） |
| **G. 復歸半衰期過濾** | 無 | 無 | 有（Ornstein-Uhlenbeck 半衰期限制在 2~60 天） | 有（Ornstein-Uhlenbeck 半衰期限制在 2~60 天） |
| **H. 避險比例 $\beta$ 估計** | 恆定為 $1.0$ | 標準化對數價格的 OLS 斜率 $\beta_{form}$ | 原始對數價格的 OLS 斜率 $\beta_{form}$ | 原始對數價格的 OLS 斜率 $\beta_{form}$ |
| **I. 交易價差公式** | $Spread_t = I_{A, t} - I_{B, t}$ | $Spread_t = y_{A, t} - \beta y_{B, t}$ | $Spread_t = \ln(P_{A, t}) - \alpha - \beta \ln(P_{B, t})$ | $Spread_t = \ln(P_{A, t}) - \alpha - \beta \ln(P_{B, t})$ |
| **J. 交易期 Z-Score 分母** | 累積總回報價差的形成期標準差 $\sigma_{spread}$ | 對數正規化價差的形成期標準差 $\sigma_{spread}$ | OLS 迴歸殘差的形成期標準差 $\sigma_{spread}$ | OLS 迴歸殘差的形成期標準差 $\sigma_{spread}$ |
| **K. 交易開平倉信號** | 靜態 Z-Score 閾值判定 | 靜態 Z-Score 閾值判定 | 靜態 Z-Score 閾值判定 | 靜態 Z-Score 閾值判定 |
| **L. 資金分配權重** | **1:1 等額市值中性** (A, B 各占 50%) | **Beta 市值中性分配** ($1 : \vert\beta\vert$ 資金權重) | **Beta 市值中性分配** ($1 : \vert\beta\vert$ 資金權重) | **Beta 市值中性分配** ($1 : \vert\beta\vert$ 資金權重) |
| **M. 交易冷卻保護** | 有 (`cooldown_dir` 進場方向鎖定) | 有 (`cooldown_dir` 進場方向鎖定) | 有 (`cooldown_dir` 進場方向鎖定) | 有 (`cooldown_dir` 進場方向鎖定) |
| **N. 風險控制體系** | 單一配對停損 + 總體熔斷 | 單一配對停損 + Z-Score停損 + 總體熔斷 | 單一配對停損 + Z-Score停損 + 總體熔斷 | 最嚴格單一停損 + Z-Score停損 + 總體熔斷 |

<br>

### 2. 全景架構演進解析

* **第一階段：幾何距離法的起點 (`ssd_basic.py`)**：
  這是配對交易的最經典起點。它假設兩檔走勢相似的股票，價格差會圍繞一個常數波動。其優點在於**完全無參數估計、無過度擬合風險**，計算極其迅速。然而，它忽視了兩股票的波動度差異，且強行規定了 $1:1$ 的避險比例，這在兩股波動彈性不對等時，會暴露顯著的多空市值非對稱風險。
* **第二階段：標準化與彈性避險的引入 (`ssd.py`)**：
  進階版 SSD 透過對數化解決了複利報酬率尺度偏誤，並透過對數價格的 Z-Score 正規化**消除了量綱偏誤**。最重要的是，它首度引入了 OLS $\beta$ 避險比例，使得資金分配能動態適應資產的價格彈性特質（Beta-neutralized）。這顯著提升了資產對沖的效果，但在兩股不具備平穩性（Non-stationarity）與共整合關係時，幾何距離最小化仍可能挑選出「偽配對」。
* **第三階段：無監督流形分群與計量共整合的昇華 (`HDBSCAN.py`)**：
  此策略代表了量化技術的巨大躍升。它不再只用價格，而是為個股建立包含動量、波動、自相關與分形 Hurst 指數的 **13 維高維金融特徵空間**，經流形降維後以 HDBSCAN 自動進行密度分群。這能**自動剔除市場雜訊個股（Noise label = -1）**。在同分群內，施以嚴格的計量經濟學 Engle-Granger 共整合檢定與 O-U 復歸半衰期雙重過濾，確保配對在統計學上具有高度顯著的「均值復歸」特質，徹底防範偽配對發散風險。
* **第四階段：多因子風險特徵空間的終極整合 (`HDBSCAN_MultiFactor.py`)**：
  這是本系統的終極進化版。它擺脫了 `HDBSCAN.py` 中 13 維特徵可能存在的統計冗餘與降維流形信息折損，直接精選了金融定價最核心的 **6 大系統/特異風險因子**（Beta、IVOL、長期 Tend 斜率等）建構特徵空間。這確保了同群資產在**基本面、宏觀曝險與特異风险不確定性上高度同質**。在此高度同質的特徵空間中執行 HDBSCAN 分群與 EG 共整合篩選，能在交易期提供極致穩健的復歸動力，並將多空配對的風險敞口鎖定在微觀的個股 Alpha 區間內，是統計套利的最前沿實踐。

<br><br>

---

<br><br>

# 三、實證結果

<br><br>

---

<br><br>

目前策略回測系統處於架構整合、優化與參數調校階段，具體交易績效與風險分析指標數據暫予略過，以確保實證分析的嚴謹度。後續我們將嚴格依據下列統一之評估指標框架，對這四個策略進行全面且深入的實證對比與表現分析：

<br><br>

---

<br><br>

## 績效表現分析 (Performance Metrics)

* **年化報酬率 (Annualized Return, AR)**：
  評估策略資金的複合增長速度。計算公式為：
  $$AR = \left( \frac{Final\_Wealth}{Initial\_Wealth} \right)^{\frac{252}{Total\_Trading\_Days}} - 1$$
  where：
  * $Final\_Wealth$ 與 $Initial\_Wealth$ 分別為策略全期期末與期初的帳戶總淨值。
  * $Total\_Trading\_Days$ 為策略回測全期的總交易天數。
  * $252$ 為一年的標準交易天數（用於年化轉換）。

<br>

* **年化超額報酬率 (Annualized Excess Return, AER)**：
  策略年化報酬率相較於無風險利率（Risk-Free Rate, $R_f$）的差額，用以衡量策略獲取純粹 Alpha 的能力：
  $$AER = AR - R_f$$
  
  **無風險利率的基準與計算方式**：
  * **基準利率選擇**：本系統預設以**美國 3 個月國庫券 (U.S. 3-Month Treasury Bill) 的年化貼現殖利率**（例如 4.5%）作為全期 $R_f$ 的基準。在台灣市場的實證研究中，可調整為**台灣銀行一年期定期儲蓄存款固定利率**（例如 1.6%）作為機會成本基準。
  * **日度超額報酬日化計算（Backtest 內部實作基準）**：
    在每日模擬中，超額日報酬率的計算方式為：
    $$R_{excess, t} = R_{portfolio, t} - R_{f, daily}$$
    where，日化無風險利率 $R_{f, daily}$ 採用**複利折現日化法**進行計算，以精確消除利息複利效應：
    $$R_{f, daily} = (1 + R_f)^{\frac{1}{252}} - 1$$
    （在部分簡化模式中，亦可採用單利折現日化法：$R_{f, daily} = R_f / 252$）。

<br>

* **累積報酬率 (Cumulative Return, CR)**：
  回測全期策略所獲取的總體回報比率：
  $$CR = \frac{Final\_Wealth - Initial\_Wealth}{Initial\_Wealth}$$

<br><br>

---

<br><br>

## Risk Analysis (風險分析)

* **夏普比率 (Sharpe Ratio, SR)**：
  衡量每一單位總風險所能換取的超額報酬，是評估風險調整後收益的核心指標：
  $$SR = \frac{AR - R_f}{\sigma_{annualized}}$$
  其中：
  * $R_f$ 為年化無風險利率。
  * $\sigma_{annualized}$ 為策略日度報酬率的年化標準差：
    $$\sigma_{annualized} = \sigma_{daily} \times \sqrt{252}$$
    （$\sigma_{daily}$ 為日度收益率的樣本標準差）。

<br>

* **最大回撤 (Maximum Drawdown, MDD)**：
  衡量回測全期內資產淨值從歷史峰值滑落的最大百分比，反映策略可能面臨的最極端下檔風險與壓力承受極限：
  $$MDD = \max_{\tau \le t} \left( \frac{Wealth_{\tau} - Wealth_t}{Wealth_{\tau}} \right)$$
  where：
  * $Wealth_{\tau}$ 為區間 $[0, t]$ 內的資產最高峰值。
  * $Wealth_t$ 為第 $t$ 天的即時資產淨值。

<br>

* **偏態與豐度分析 (Skewness and Kurtosis of Returns)**：
  * **偏態 (Skewness)**：評估日收益率分佈的對稱性：
    $$Skewness = E\left[ \left( \frac{R_t - \mu}{\sigma} \right)^3 \right]$$
    * *正偏態 (Skewness > 0)*：代表策略收益分佈有右側肥尾，多數時候日收益為小幅波動或微虧，但偶爾會出現極大的單日盈利（如成功套利大價差）。這對於統計套利策略是極佳的特徵。
    * *負偏態 (Skewness < 0)*：代表收益分佈左側肥尾，多數時候穩定獲取微小利潤，但偶爾會因價差發散或停損不及而遭受單日巨大虧損（如黑天鵝事件）。
  * **豐度/峰度 (Kurtosis)**：衡量收益率分佈的極端值肥尾特徵：
    $$Kurtosis = E\left[ \left( \frac{R_t - \mu}{\sigma} \right)^4 \right]$$
    * *高超額峰度 (Kurtosis > 3)*：代表收益率分佈呈現顯著的肥尾 (Fat-tailed) 特徵，暗示極端盈虧發生的概率遠高於常態分佈。為此，必須結合嚴格的熔斷機制與單一配對停損，以控制極端尾部風險。
