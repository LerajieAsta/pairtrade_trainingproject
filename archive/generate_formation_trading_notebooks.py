import json
import os

# 定義 notebooks 目錄
notebook_dir = "notebooks"
formation_path = os.path.join(notebook_dir, "formation.ipynb")
trading_path = os.path.join(notebook_dir, "trading.ipynb")

os.makedirs(notebook_dir, exist_ok=True)

# ==================== 1. FORMATION NOTEBOOK CELLS ====================
formation_cells = []

# Cell 0: Title & Overview
formation_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 📐 S&P 500 Pairs Trading 形成期 (Formation Period) 策略邏輯與公式詳解\n",
        "\n",
        "## 📝 概述\n",
        "在配對交易 (Pairs Trading) 中，**形成期 (Formation Period)**（預設 $F = 252$ 天）的核心任務是**篩選候選標的並建立具備統計套利價值的配對組合**。\n",
        "\n",
        "> [!IMPORTANT]\n",
        "> **本文件完全以 `strategies/formation/` 下實際運行的 `.py` 原始碼為準**進行解析，糾正了舊版 Notebook 文檔中的過時描述。\n",
        "\n",
        "### 📂 策略檔案結構對照：\n",
        "- **經典/進階 SSD 距離策略** $\\rightarrow$ `strategies/formation/ssd_basic.py`, `strategies/formation/ssd.py`\n",
        "- **純 DTW 與共整合 DTW 策略** $\\rightarrow$ `strategies/formation/DTW_Pure_Notebook.py`, `strategies/formation/DTW_Cointegration_Paper.py`\n",
        "- **HDBSCAN 密度分群系列策略** $\\rightarrow$ `strategies/formation/HDBSCAN.py`, `HDBSCAN_CrossSector_UMAP.py`, `HDBSCAN_CrossSector_PCA.py`, `HDBSCAN_CrossSector_MultiFactor.py` 等\n",
        "\n",
        "---"
    ]
})

# Cell 1: ssd_basic & ssd formation
formation_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📏 一、 經典 SSD (Basic) 與進階 SSD (OLS) 形成期邏輯\n",
        "\n",
        "在實際的系統程式碼中，**經典 SSD Basic 策略同樣套用了三關統計過濾以防止價差逆勢發散**。兩者在形成期的差異主要在於「價格正規化空間」與「對沖比例是否固定為 1.0」。\n",
        "\n",
        "### 1.1 經典 SSD (Basic) 形成期邏輯 (`ssd_basic.py`)\n",
        "- **價格幾何正規化 (Price Normalization)**：\n",
        "  $$P'_{i, t} = \\frac{P_{i, t}}{P_{i, 0}}$$\n",
        "  以第一天價格 $P_{i,0}$ 為基準將股價歸一化為累積回報指數起點為 1.0。\n",
        "- **幾何歐氏距離平方和 (SSD)**：\n",
        "  $$\\text{SSD}_{A,B} = \\sum_{t=1}^F (P'_{A,t} - P'_{B,t})^2$$\n",
        "  限制在同產業內計算。\n",
        "- **避險比例**：固定為 1.0，即美元中性等市值對沖 ($v_a = 0.5, v_b = 0.5$)。\n",
        "- **三關統計過濾 (已修正)**：程式碼在計算完 SSD 後，會挑選前 $\\text{top\\_n} \\times 15$ 組候選配對進行慢速過濾：\n",
        "  1. **ADF 共整合檢定** ($p < 0.05$)。\n",
        "  2. **Ornstein-Uhlenbeck 半衰期過濾** ($2.0 \\le \\text{Half-Life} \\le 40.0$ 天)。\n",
        "  3. **Hurst 指數篩選** ($\\text{Hurst} < 0.40$)。\n",
        "\n",
        "### 1.2 進階 SSD (OLS) 形成期邏輯 (`ssd.py`)\n",
        "- **對數價格 Z-Score 標準化**：\n",
        "  $$P'_{i, t} = \\frac{\\ln(P_{i, t}) - \\mu_{\\ln(P_i)}}{\\sigma_{\\ln(P_i)}}$$\n",
        "  將收益率波動度與價格規模完全歸一化。\n",
        "- **對沖比例 (Hedge Ratio $\\beta$)**：使用最小二乘法 (OLS) 計算兩者間的避險比例 $\\beta$：\n",
        "  $$\\beta = \\frac{\\text{Cov}(P'_A, P'_B)}{\\text{Var}(P'_B)}$$\n",
        "  殘差為：$\\epsilon_t = P'_{A, t} - \\beta \\cdot P'_{B, t}$。\n",
        "- **三關統計過濾**：同樣套用上述 ADF ($p < 0.05$)、半衰期 ($2.0 \\le HL \\le 40.0$) 與 Hurst ($< 0.40$) 過濾篩選。"
    ]
})

# Cell 2: DTW Pure & Cointegration formation
formation_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## ⏳ 二、 DTW (Dynamic Time Warping) 形成期邏輯\n",
        "\n",
        "動態時間扭曲 (DTW) 能有效應對**兩隻股票在走勢上存在時間滯後 (Lag) 的非同步波動情況**。本系統實作了兩種 DTW 篩選邏輯：\n",
        "\n",
        "### 2.1 純 DTW 距離策略 (`DTW_Pure_Notebook.py`)\n",
        "- **累積總回報指數正規化**：先計算日收益率 $R_{i,t}$，再計算累積回報：\n",
        "  $$P'_{i, t} = \\prod_{\\tau=1}^t (1 + R_{i, \\tau})$$\n",
        "- **無檢定篩選**：直接使用 `dtaidistance` (C優化庫) 計算同產業內所有股票的純 DTW 距離。**完全不套用**任何 ADF、Hurst 或半衰期等統計檢定，最大化保留潛在的扭曲相似配對。依 DTW 距離升序選擇 Top N。\n",
        "\n",
        "### 2.2 許鈞翔 (2025) 論文對齊版共整合 DTW 策略 (`DTW_Cointegration_Paper.py`)\n",
        "- **步驟 1：共整合與統計特徵預選**：\n",
        "  對標準化對數價格進行雙向 OLS 回歸。只保留通過 **ADF 共整合檢定 ($p < 0.01$)**、半衰期在 $[2.0, 40.0]$ 天內、且 $\\text{Hurst} < 0.40$ 的強均值回歸配對。\n",
        "- **步驟 2：帶限制窗口的 Sakoe-Chiba DTW 距離**：\n",
        "  為防止非理性時間扭曲，套用 Sakoe-Chiba 限制窗口 $W$ (預設為 15 天)，計算快速 DTW 距離，使時間對齊限制在合理的領先/落後天數內。\n",
        "- **步驟 3：排序與 PCA 融合 (實驗組)**：\n",
        "  系統支持兩種排序篩選模式：\n",
        "  1. **`dtw` 模式 (對照組)**：依 DTW 距離升序排序選取前 $N$ 對。\n",
        "  2. **`ssd_dtw_pca` 模式 (實驗組)**：將 SSD 與 DTW 距離標準化後，利用主成分分析 (PCA) 提取第一主成分 PC1 得分作為「綜合距離指標」並升序排序，融合了幾何形狀相似性 (SSD) 與時序對齊特徵 (DTW)。"
    ]
})

# Cell 3: DTW formation code snippet
formation_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# DTW_Cointegration_Paper.py 限制窗口的 Sakoe-Chiba DTW 與 PCA 融合篩選\n",
        "import numpy as np\n",
        "from sklearn.decomposition import PCA\n",
        "\n",
        "def compute_sakoe_chiba_dtw(x, y, window=15):\n",
        "    n, m = len(x), len(y)\n",
        "    dp = np.full((n + 1, m + 1), np.inf)\n",
        "    dp[0, 0] = 0.0\n",
        "    \n",
        "    for i in range(1, n + 1):\n",
        "        start_j = max(1, i - window)\n",
        "        end_j = min(m, i + window)\n",
        "        for j in range(start_j, end_j + 1):\n",
        "            cost = (x[i - 1] - y[j - 1]) ** 2\n",
        "            dp[i, j] = cost + min(\n",
        "                dp[i - 1, j],     # 插入\n",
        "                dp[i, j - 1],     # 刪除\n",
        "                dp[i - 1, j - 1]  # 匹配\n",
        "            )\n",
        "    return float(dp[n, m])\n",
        "\n",
        "def fuse_ssd_dtw_via_pca(ssd_list, dtw_list):\n",
        "    # 將 SSD 與 DTW 距離組合成二維特徵矩陣並標準化\n",
        "    data = np.column_stack([ssd_list, dtw_list])\n",
        "    mean = np.mean(data, axis=0)\n",
        "    std = np.std(data, axis=0) + 1e-12\n",
        "    data_scaled = (data - mean) / std\n",
        "    \n",
        "    # 進行 PCA 降維，取得第一主成分 PC1得分\n",
        "    pca = PCA(n_components=1)\n",
        "    pc1_score = pca.fit_transform(data_scaled).squeeze()\n",
        "    \n",
        "    # 確保方向一致：若與原始距離呈負相關，則反轉得分\n",
        "    if pca.components_[0, 0] < 0:\n",
        "        pc1_score = -pc1_score\n",
        "    return pc1_score"
    ]
})

# Cell 4: HDBSCAN formation
formation_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🌐 三、 HDBSCAN 系列密度分群形成期邏輯\n",
        "\n",
        "HDBSCAN 系列策略將股票映射到**特徵空間**進行密度分群，再於同分群內進行配對，排除無規律的隨機噪聲標的。此處區分為「產業內分群」與「全市場跨產業分群」兩大邏輯類型：\n",
        "\n",
        "### 3.1 產業內 HDBSCAN 密度分群 (`HDBSCAN.py`)\n",
        "- **特徵空間 (13維)**：對每檔股票收益率序列提取 13 維時序與統計特徵。\n",
        "- **分群流程**：只在**同產業內**使用 **UMAP** (或 PCA) 降維至 5 維，並透過 HDBSCAN 分群，排除噪音點 ($label = -1$)。配對僅在「同產業且同分群」的股票中產生，進一步進行雙向 OLS 與 Engle-Granger ADF 共整合、半衰期與 Hurst 過濾。\n",
        "\n",
        "### 3.2 全市場跨產業 HDBSCAN 聚類策略 (`HDBSCAN_CrossSector_UMAP.py` / `PCA.py` / `MultiFactor.py`)\n",
        "這是本量化平台的重要進階修正。為克服「同產業配對池過小」且尋找「跨行業的替代性統計關係」，跨產業策略做出了以下優化：\n",
        "- **全市場聚類與跨產業配對 (Cross-Sector)**：\n",
        "  降維與 HDBSCAN 分群是**在全市場（而非各產業內部）**股票上執行。只要被劃分到同一個聚類群落（$label \\ne -1$）的股票即可兩兩配對，即使它們來自**不同的 GICS 板塊**（例如科技與金融）。\n",
        "- **多維因子特徵空間**：\n",
        "  - `HDBSCAN_CrossSector_UMAP.py` 與 `_PCA.py` 使用 13 維時序特徵降維聚類。\n",
        "  - `HDBSCAN_CrossSector_MultiFactor.py` 使用 **6 大金融穩健因子**（市場 Beta、波動率、偏態、峰態、長期價格趨勢斜率、特異波動率），並且**跳過降維步驟**直接進行 HDBSCAN 跨產業分群，以保留最純粹的金融意義。\n",
        "- **更嚴格的 5 道篩選關卡**：\n",
        "  為了防範跨產業隨機配對導致的假性共整合 (Spurious Cointegration)，套用了高達 5 重統計檢定：\n",
        "  1. **皮爾森相關係數 (Correlation Filter)**：要求對數價格之 $\\text{Correlation} \\ge 0.50$。\n",
        "  2. **ADF 共整合檢定**：要求 $p \\text{-value} < \\text{adf\\_pvalue\\_threshold}$ (預設 0.01 或 0.05)。\n",
        "  3. **O-U 半衰期**：價差均值回歸半衰期必須滿足 $\\text{halflife\\_min} \\le HL \\le \\text{halflife\\_max}$ (預設 $2.0 \\le HL \\le 63.0$ 天)。\n",
        "  4. **Hurst 指數**：要求 $\\text{Hurst} < \\text{hurst\\_threshold}$ (預設 0.5)。\n",
        "  5. **均值交叉次數 (Zero Crossings Filter)**：形成期殘差序列跨越其均值的次數必須滿足 $\\text{Zero\\_Crossings} \\ge \\text{min\\_zero\\_crossings}$ (預設 5次)，確認價差回歸的頻繁度。\n",
        "- **Han et al. 2021：Mom1 截面動量差篩選**：\n",
        "  計算個股一個月動量差 $\\text{Mom1} = \\ln(P_t) - \\ln(P_{t-21})$，計算兩者的動量差值 $\\text{Mom1\\_Diff} = |\\text{Mom1}_A - \\text{Mom1}_B|$。差值越大代表近期發生非對稱過度偏離，具有更強的反轉獲利空間。"
    ]
})

# Cell 5: HDBSCAN feature extraction code snippet
formation_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# HDBSCAN_CrossSector_UMAP.py 跨產業配對與 5 道篩選核心邏輯\n",
        "import numpy as np\n",
        "\n",
        "def filter_cross_sector_pairs(log_a, log_b, min_corr=0.50, min_zero_crossings=5):\n",
        "    # 1. Pearson Correlation Filter\n",
        "    corr = np.corrcoef(log_a, log_b)[0, 1]\n",
        "    if corr < min_corr:\n",
        "        return False, corr, 0\n",
        "        \n",
        "    # 2. OLS fitting (A on B) & residual calculation\n",
        "    n_len = len(log_a)\n",
        "    x_mat = np.column_stack([np.ones(n_len), log_b])\n",
        "    coeffs = np.linalg.lstsq(x_mat, log_a, rcond=None)[0]\n",
        "    resid = log_a - coeffs[0] - coeffs[1] * log_b\n",
        "    \n",
        "    # 3. Zero Crossings Filter (穿越均值次數檢測)\n",
        "    mean_val = np.mean(resid)\n",
        "    demeaned = resid - mean_val\n",
        "    # 計算符號變更的次數\n",
        "    zero_crossings = np.sum(np.diff(np.sign(demeaned)) != 0)\n",
        "    if zero_crossings < min_zero_crossings:\n",
        "        return False, corr, zero_crossings\n",
        "        \n",
        "    return True, corr, zero_crossings"
    ]
})

# Cell 6: Summary Table
formation_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 四、 所有策略形成期特徵與配對標準對比\n",
        "\n",
        "| 策略特徵 | 經典 SSD (Basic) | 進階 SSD (OLS) | 純 DTW 距離 | 論文共整合 DTW | HDBSCAN 產業內聚類 | HDBSCAN 跨產業聚類 |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
        "| **跨產業配對** | 否 (限定同產業) | 否 (限定同產業) | 否 (限定同產業) | 否 (限定同產業) | 否 (限定同產業) | **是 (全市場跨板塊聚類)** |\n",
        "| **正規化方法** | $P_{i,t}/P_{i,0}$ | 對數價格 Z-Score | 累積收益指數 | 對數價格 Z-Score | 對數價格 Z-Score | 對數價格 Z-Score |\n",
        "| **特徵維度** | 無 | 無 | 無 | 無 | 13維時序統計特徵 | 13維時序 或 6維金融因子 |\n",
        "| **降維方法** | 無 | 無 | 無 | 無 | UMAP (5維) | UMAP/PCA 或 無 (多因子) |\n",
        "| **過濾檢定** | ADF, 半衰期, Hurst | ADF, 半衰期, Hurst | 無 | ADF, 半衰期, Hurst | ADF, 半衰期, Hurst | **ADF, 半衰期, Hurst, 相關性, 均值穿越次數** |\n",
        "| **動量差過濾** | 否 | 否 | 否 | 否 | 否 | **是 (Mom1 截面動量差)** |\n",
        "| **避險比例 $\\beta$** | 固定為 1.0 | 形成期 OLS 斜率 | 固定為 1.0 | 形成期 OLS 斜率 | 形成期 OLS 斜率 | 形成期 OLS 斜率 |\n",
        "\n",
        "---"
    ]
})

# ==================== 2. TRADING NOTEBOOK CELLS ====================
trading_cells = []

# Cell 0: Title & Overview
trading_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 📈 S&P 500 Pairs Trading 交易期 (Trading Period) 策略邏輯與公式詳解\n",
        "\n",
        "## 📝 概述\n",
        "在配對交易中，**交易期 (Trading Period)**（預設 $T = 126$ 天）的核心任務是**根據即時信號執行交易，並結合高階部位與全域風控機制管理多週期 Slots 權益**。\n",
        "\n",
        "> [!IMPORTANT]\n",
        "> **本文件完全以 `strategies/trading/` 下實際運行的 `.py` 原始碼為準**進行解析，糾正了舊版 Notebook 文檔中的過時描述。\n",
        "\n",
        "### 📂 策略檔案結構對照：\n",
        "- **Z-Score 狀態機交易核心** $\\rightarrow$ `strategies/trading/zscore_trading.py` (包含 SSD 與 HDBSCAN 的交易實作)\n",
        "- **純 DTW 交叉帶內進場交易** $\\rightarrow$ `strategies/trading/pure_dtw_trading.py`\n",
        "- **深度強化學習 LSTM 交易** $\\rightarrow$ `strategies/trading/drl_lstm_trading.py`\n",
        "\n",
        "---"
    ]
})

# Cell 1: ZScore Trading
trading_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🧠 一、 通用 Z-Score 狀態機交易期邏輯 (`zscore_trading.py`)\n",
        "\n",
        "大部分 Z-Score 基礎策略均繼承或使用 `zscore_trading.py` 中實現的狀態機。\n",
        "\n",
        "### 1.1 價差重構與 Z-Score 計算\n",
        "- **路徑 A：OLS 對數空間 (如 HDBSCAN / OLS SSD 策略)**：\n",
        "  當 `ols_alpha is not None`，使用對數價格 $\\ln(P)$ 及形成期傳入的常數項 $\\alpha$、避險比例 $\\beta$：\n",
        "  $$\\text{Spread}_t = \\ln(P_{A, t}) - \\alpha - \\beta \\cdot \\ln(P_{B, t})$$\n",
        "- **路徑 B：標準化價格空間 (如 Basic SSD 策略)**：\n",
        "  當 `ols_alpha is None`，在標準化價格/對數價格空間下計算價差：\n",
        "  $$\\text{Spread}_t = P'_{A, t} - \\beta \\cdot P'_{B, t}$$\n",
        "- **Z-Score 計算 (固定 vs 滾動)**：\n",
        "  - 固定模式 (`zscore_window = 0`)：$Z_t = \\frac{\\text{Spread}_t - \\mu_{form}}{\\sigma_{form}}$。\n",
        "  - 滾動模式 (`zscore_window = W`)：在交易期滾動視窗 $W$ 天內重新進行 OLS 回歸得到動態 $\\alpha_t, \\beta_t$，並以殘差變異數作為標準差中心：\n",
        "    $$\\sigma_{residual, t} = \\sqrt{\\max\\left(\\text{Var}_W(P'_A) - \\beta_t \\cdot \\text{Cov}_W(P'_A, P'_B), 0\\right)}$$\n",
        "    $$Z_t = \\frac{\\text{Spread}_t}{\\sigma_{residual, t}}$$\n",
        "\n",
        "### 1.2 進出場訊號\n",
        "- **建倉 (Entry)**：當 $\\vert Z_t \\vert > \\text{entry\\_z}$ 時。\n",
        "  - $Z_t > \\text{entry\\_z} \\rightarrow$ 空頭建倉 (空 A 多 B)。\n",
        "  - $Z_t < -\\text{entry\\_z} \\rightarrow$ 多頭建倉 (多 A 空 B)。\n",
        "- **平倉 (Exit)**：當 $Z_t$ 回歸至零軸邊界時，即 $\\vert Z_t \\vert \\le \\text{exit\\_z}$。\n",
        "\n",
        "### 1.3 資金部位風險中性配置\n",
        "股票 A 與 B 的市值部位 $v_a, v_b$ 根據避險比例 $\\beta$ 進行加權分配（對沖市場 Beta 風險）：\n",
        "$$\\text{Total Weight} = 1.0 + |\\beta|$$\n",
        "$$v_a = C_{pair} \\times \\frac{1.0}{\\text{Total Weight}}, \\quad v_b = C_{pair} \\times \\frac{|\\beta|}{\\text{Total Weight}}$$\n",
        "*(註：在 `ssd_basic.py` 中因 $\\beta = 1.0$，部位資金分配為等額的 50%/50% 分配)*"
    ]
})

# Cell 2: Pure DTW Trading
trading_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## ⏳ 二、 純 DTW 交叉帶內進場交易邏輯 (`pure_dtw_trading.py`)\n",
        "\n",
        "純 DTW 交易策略繼承自 `Trading` (來自 `zscore_trading.py`)，但在**開倉訊號上做出了重要調整，以減少逆勢建倉風險**。\n",
        "\n",
        "### 2.1 交叉帶內進場條件 (Cross Back Inside the Bands)\n",
        "傳統策略在 Z-Score「突破」臨界線時立刻進場，容易遇到「價差持續發散」的單邊風險。純 DTW 策略要求**價差 Z-Score 必須先突破，並在「回折交叉回歸帶內」時才觸發建倉**：\n",
        "- **空頭建倉 (Short Entry, -1)**：當前一日 $Z_{t-1} > \\text{entry\\_z}$ 且當日 $Z_t \\le \\text{entry\\_z}$ 時。（從上方折返穿過上開倉線）\n",
        "- **多頭建倉 (Long Entry, 1)**：當前一日 $Z_{t-1} < -\\text{entry\\_z}$ 且當日 $Z_t \\ge -\\text{entry\\_z}$ 時。（從下方折返穿過下開倉線）\n",
        "\n",
        "### 2.2 均值平倉條件\n",
        "- **空頭平倉 (Short Exit)**：當 $Z_t \\le \\text{exit\\_z}$ 時。\n",
        "- **多頭平倉 (Long Exit)**：當 $Z_t \\ge -\\text{exit\\_z}$ 時。\n",
        "- **資金配置**：由於純 DTW 策略假設兩隻正規化收益率指數等價，其對沖比例固定為 $1.0$，資金分配為 50%/50% 等權重配置。"
    ]
})

# Cell 3: DRL LSTM Trading
trading_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🤖 三、 深度強化學習 LSTM 交易期邏輯 (`drl_lstm_trading.py`)\n",
        "\n",
        "此策略將配對交易建模為馬可夫決策過程 (MDP)，利用 **DQN + LSTM** 網路，在複雜的多維特徵空間下學習最佳建平倉策略。\n",
        "\n",
        "### 3.1 Gymnasium 環境特徵空間 (5維狀態)\n",
        "在交易期每日，環境提取以下 5 維狀態向量 $S_t$ 作為 Agent 的觀察值：\n",
        "1. **Spread Z-Score**：即時價差標準化值。\n",
        "2. **相對回報率 (Rel_Return)**：兩隻股票日收益率之差：$R_{A,t} - R_{B,t}$。\n",
        "3. **均線距離 (MA_Dist)**：即時價差相對於其 20 日移動平均線的乖離度：$\\text{Spread}_t - \\text{MA}_{20}(\\text{Spread}_t)$。\n",
        "4. **剩餘交易時間比率 (Time-to-Maturity)**：離交易期結束的剩餘時間比例：$\\frac{T - t}{T}$，用以提示臨近強制平倉的風險。\n",
        "5. **動態波動率 (Volatility)**：價差的 20 日滾動標準差，捕捉即時市場風險層級。\n",
        "\n",
        "### 3.2 動作空間與獎勵函數 (Reward Function)\n",
        "- **離散動作空間**：`0` (Flat/平倉或空倉), `1` (Long Spread), `2` (Short Spread)。\n",
        "- **事件型獎勵 (Reward)**：\n",
        "  - **平倉回報**：平倉時根據該筆交易的累計淨收益率給予對應的正/負獎勵：\n",
        "    $$\\text{Reward}_{exit} = \\frac{\\text{Trade PnL}}{\\text{Capital}} \\times 100.0$$\n",
        "  - **交易摩擦懲罰 (Action Penalty)**：開倉時，扣除開倉摩擦成本比例作為懲罰，以抑制無效過度交易：\n",
        "    $$\\text{Reward}_{entry} = -\\frac{\\text{Entry Fee}}{\\text{Capital}} \\times 100.0 \\times 0.5$$\n",
        "\n",
        "### 3.3 LSTM_DQN 模型與時序記憶\n",
        "- **時序序列輸入**：Agent 的輸入並非單日的 5 維特徵，而是**過去 $seq\\_len = 10$ 天的時序特徵序列矩陣** (形狀為 $10 \\times 5$)。\n",
        "- **神經網路**：通過 PyTorch 的 `nn.LSTM` 提取這 10 天特徵的動態時序記憶，取最後一個隱藏狀態 (Last Hidden State) 輸入全連接層，輸出 3 個動作對應的 Q 值。\n",
        "\n",
        "### 3.4 形成期預訓練 + 交易期實時推論\n",
        "- 在每個交易期開始之前，Agent 會先在其**形成期的 252 天歷史數據環境上進行 100 輪 (Episodes) 的 DQN 強化學習訓練**，學習該配對專屬的套利規律。\n",
        "- 進入交易期後，將探索率 $\\epsilon$ 降為 0，模型進行實時 DQN 推論並執行交易。"
    ]
})

# Cell 4: DRL LSTM PyTorch implementation snippet
trading_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# drl_lstm_trading.py 中 Gymnasium 環境與 PyTorch LSTM_DQN 網路實作\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import numpy as np\n",
        "\n",
        "class LSTM_DQN(nn.Module):\n",
        "    def __init__(self, input_dim=5, hidden_dim=64, output_dim=3, num_layers=1):\n",
        "        super(LSTM_DQN, self).__init__()\n",
        "        self.hidden_dim = hidden_dim\n",
        "        self.num_layers = num_layers\n",
        "        # batch_first=True, 輸入形狀為 (batch_size, seq_len, input_dim)\n",
        "        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)\n",
        "        self.fc = nn.Linear(hidden_dim, output_dim)\n",
        "        \n",
        "    def forward(self, x):\n",
        "        # out 形狀: (batch_size, seq_len, hidden_dim)\n",
        "        out, _ = self.lstm(x)\n",
        "        # 取最後一個時間步 (last time-step) 的隱藏狀態\n",
        "        out = out[:, -1, :]\n",
        "        # 輸出 3 個動作的 Q 值\n",
        "        return self.fc(out)\n",
        "\n",
        "def get_trading_env_observation(zscore_series, price_a, price_b, current_step, max_steps):\n",
        "    # 模擬 drl_lstm_trading.py 中的 5 維環境觀察值生成\n",
        "    zscore = zscore_series[current_step]\n",
        "    \n",
        "    # 相對報酬率\n",
        "    ret_a = (price_a[current_step] - price_a[current_step-1]) / price_a[current_step-1] if current_step > 0 else 0.0\n",
        "    ret_b = (price_b[current_step] - price_b[current_step-1]) / price_b[current_step-1] if current_step > 0 else 0.0\n",
        "    rel_return = ret_a - ret_b\n",
        "    \n",
        "    # 均線乖離率 (相對 20日均線)\n",
        "    ma20 = np.mean(zscore_series[max(0, current_step-20):current_step+1])\n",
        "    ma_dist = zscore - ma20\n",
        "    \n",
        "    # 剩餘到期時間比率\n",
        "    time_to_maturity = (max_steps - current_step) / max_steps\n",
        "    \n",
        "    # 波動度\n",
        "    volatility = np.std(zscore_series[max(0, current_step-20):current_step+1]) if current_step > 1 else 1.0\n",
        "    \n",
        "    return np.array([zscore, rel_return, ma_dist, time_to_maturity, volatility], dtype=np.float32)"
    ]
})

# Cell 5: Risk control
trading_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🧠 四、 共享部位管理與六大風控機制\n",
        "\n",
        "本量化系統實作了完善的六大風控防線，防範黑天鵝事件與逆勢單邊發散風險：\n",
        "1. **個股單筆停損 (`SL`)**：當個股未實現虧損比例達到 `stop_loss_pct` 時，觸發強制平倉並凍結該配對。\n",
        "   $$\\text{Loss Ratio} = -\\frac{\\text{Trade PnL}}{C_{pair}} \\ge \\text{stop\\_loss\\_pct}$$\n",
        "2. **部位動態 Z-Score 偏離停損 (`DSZ`)**：當 $\\vert Z_t \\vert > \\text{dynamic\\_stop\\_z}$（如 3.0 或 5.0）時，判斷發生結構性破裂 (Structural Break) 即刻停損。\n",
        "3. **全域投資組合層級最大回撤停損 (`PSL`)**：當總資金虧損達到 `portfolio_stop_loss_pct` (如 10%) 時，觸發 PSL 一鍵斬倉所有持倉配對並重置且凍結所有交易。\n",
        "4. **產業分散化集中度上限 (`MSR`)**：限制單一產業的配對數量上限：$\\max(1, \\lfloor N \\cdot \\text{max\\_sector\\_ratio} \\rfloor)$。\n",
        "5. **方向性建倉冷卻機制 (Cooldown Period)**：平倉後進入冷卻，多頭平倉需等 $Z_t \\ge -\\text{EXIT\\_Z}$，空頭平倉需等 $Z_t \\le \\text{EXIT\\_Z}$ 才能解凍。\n",
        "6. **自適應波動率調節機制 (`VOL ADJ`)**：依近期 20 日波動率放大形成期標準差，防止無序震盪中頻繁建倉：\n",
        "   $$\\sigma_{adjusted} = \\sigma_{formation} \\times \\max\\left(1.0, \\frac{\\sigma_{roll20}}{\\sigma_{formation}}\\right)$$"
    ]
})

# Cell 6: Comparison Table
trading_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 五、 所有策略交易期特徵與參數對比總結\n",
        "\n",
        "| 交易特徵 | 經典 Z-Score 交易 | 純 DTW 交叉交易 | DRL LSTM 強化學習交易 |\n",
        "| :--- | :--- | :--- | :--- |\n",
        "| **對應檔案** | `zscore_trading.py` | `pure_dtw_trading.py` | `drl_lstm_trading.py` |\n",
        "| **開倉條件** | Z-Score 絕對值突破開倉線 ($" + "\\vert Z_t \\vert > \\text{entry\\_z}$" + ") | Z-Score 從帶外折返交叉回歸帶內 | DRL 代理人實時輸出動作 `1` 或 `2` |\n",
        "| **平倉條件** | Z-Score 回歸到退出區間 ($" + "\\vert Z_t \\vert \\le \\text{exit\\_z}$" + ") | Z-Score 回歸至均值中心 | DRL 代理人實時輸出動作 `0` (Flat) |\n",
        "| **部位與對沖** | 依 OLS $\\beta$ 進行風險中性加權配置 | 固定為 1.0 (等權重 50%/50% 分配) | 依 OLS $\\beta$ 進行風險中性加權配置 |\n",
        "| **特徵狀態** | 僅依據單一的 Spread Z-Score 數值 | 僅依據單一的 Spread Z-Score 數值 | 5維狀態空間（Z-Score, 相對回報, 均線距離, 到期剩餘時間, 波動率） |\n",
        "| **時序記憶** | 無 | 前一日與當日 Z-Score 比較 ($Z_{t-1}, Z_t$) | 過去 10 天時序特徵矩陣，經由 LSTM 網路提煉 |\n",
        "| **預訓練機制** | 無 | 無 | 在形成期歷史數據上預訓練 100 輪 (Episodes) |\n",
        "| **部位風控** | 完整支援 (SL, DSZ, PSL, MSR, VOL ADJ) | 完整支援 (SL, DSZ, PSL, MSR, VOL ADJ) | 支援環境內摩擦成本懲罰與強平機制 |\n",
        "\n",
        "---"
    ]
})

# ==================== 3. WRITE TO FILES ====================

def write_notebook(path, cells):
    content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

write_notebook(formation_path, formation_cells)
write_notebook(trading_path, trading_cells)

print("SUCCESS")
