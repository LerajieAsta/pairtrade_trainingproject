import json
import os

nb_path = 'd:/Unknown/Papper/Code/notebooks/pt_step3_1150408.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract code cells from the notebook
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

# We have 10 code cells. 
# code_cells[0]: import subprocess
# code_cells[1]: os, FAST_TEST_MODE
# code_cells[2]: load_data_from_db
# code_cells[3]: prices_raw = ...
# code_cells[4]: extract_micro_ts_features, compute_hurst...
# code_cells[5]: dynamic_hdbscan_clustering...
# code_cells[6]: class PairsTradingEnv...
# code_cells[7]: env_test...
# code_cells[8]: PPO Agent training...
# code_cells[9]: Benchmarking / Plotting...

def c_markdown(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + ("\n" if i < len(src.split("\n"))-1 else "") for i, line in enumerate(src.split("\n"))]
    }

md1 = """## 1. 數據管理與基礎預處理 (Data Pipeline)

**執行功能**：建立 SQLite 資料庫連結並載入 S&P 500 歷史資料；計算對數轉換與均值回歸基礎檢驗；整合 VIX 指數並執行 1 日平移（Shift）以消除前視偏誤（Look-ahead Bias）；修正已下市股票成分以應對生存者偏誤。

**參考文獻**：
- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction.
- Gatev, E., et al. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule.

**參考重點與依據**：
- **Engle-Granger 兩步法**：作為檢驗價差是否具備均值回歸特性的學術標準。
- **生存者偏誤處理**：Gatev (2006) 提出應納入已下市股票資料，避免績效被過度高估。"""

md2 = """## 2. 微觀統計特徵工程與品質過濾

**執行功能**：計算股票個體的 Log Return、ATR、Z-Score、Hurst Exponent 與 Half-life（半衰期）。

**參考文獻**：
- Sarmento, M. S., & Horta, N. (2021). Pairs Trading via Unsupervised Learning.

**參考重點與依據 (硬性篩選門檻)**：
- **Hurst Exponent (H)**：僅保留 $H < 0.45$ 的股票，確保其具備強烈的均值回歸動能。
- **Half-life (HL)**：僅保留 $HL < 126$ 天（約半年）的股票，確保回歸速度足以在有限時間內產生利潤。"""

md3 = """## 3. 動態分群與高獲利配對篩選

**執行功能**：利用 PCA 提取特徵殘差，並執行 HDBSCAN 自動識別相似行為的股票群組；對同群組股票執行 Engle-Granger 共整合檢定與獲利空間校驗。

**參考文獻**：
- Sarmento & Horta (2021). Pairs Trading via Unsupervised Learning.
- Campagnoli, T., et al. (2023). Dynamic pairs trading using clustering and reinforcement learning.

**參考重點與依據 (硬性篩選門檻)**：
- **年化過零率 (Zero-Crossing)**：在形成期內，價差穿越均值的次數必須 $\ge 12$ 次（平均每月至少一次），排除無波動的死水配對。
- **獲利空間校驗**：價差標準差 ($\sigma_{spread}$) 必須 $\ge 2 \times$ 雙邊交易成本 ($2 \times 0.29\%$)，確保獲利能覆蓋摩擦成本。"""

md4 = """## 4. 強化學習 MDP 環境定義

**執行功能**：封裝符合 Gymnasium 標準的 PairsTradingEnv；定義狀態空間（包含 Z-Score、VIX、Half-life）與離散動作空間（Long, Short, Flat）。

**參考文獻**：
- Brockman, G., et al. (2016). OpenAI Gym.
- Lucarelli, G., et al. (2019). A deep reinforcement learning approach for automated cryptocurrency trading.

**參考重點與依據**：
- **混合獎勵設計**：結合增量損益、換倉摩擦、與針對最大回撤（MDD）的平方懲罰，解決金融回饋稀疏問題。"""

md5 = """## 5. 滾動式推進分析與資產管理 (WFO Engine)

**執行功能**：實作 Walk-Forward Optimization，嚴格切分訓練集與樣本外測試集；使用 PPO 演算法訓練代理人，並透過 Daily Portfolio Manager 執行多視窗滾動資金分配。

**參考文獻**：
- Schulman, J., et al. (2017). Proximal policy optimization algorithms.

**參考重點與依據**：
- **WFO 框架**：徹底防範前視偏誤，確保模型在時刻 $t$ 僅使用 $t$ 以前的資料。"""

md6 = """## 6. 基準對標與績效可視化

**執行功能**：與 S&P 500 (SPY) 買入持有策略進行對標分析；繪製累積報酬曲線、最大回撤圖，並計算全期間統計指標。

**參考重點**：
- 遵循元智大學 (YZU) 資管系論文圖表規範與 APA 引用格式。"""

# Fix c4.py logic (code_cells[4])
c4_src = "".join(code_cells[4]['source'])
c4_src = c4_src.replace("features_df['Half-life']", "features_df['Half_life']")
# Assign back
code_cells[4]['source'] = [line + ("\n" if i < len(c4_src.split("\n"))-1 else "") for i, line in enumerate(c4_src.split("\n"))]

# Fix c5.py logic (code_cells[5])
c5_src = "".join(code_cells[5]['source'])
# 1. fix keyerror on zero_cross
c5_src = c5_src.replace(
    "passed = [res for res in coint_results if res is not None and res['zero_cross'] >= 12]",
    "passed = [res for res in coint_results if res is not None]"
)

# 2. Add spread_std calculation and filtering
filter_logic = """    # 依照 SSD 排序並選出 Top-N
    df_passed = pd.DataFrame(passed)
    if df_passed.empty:
        return {n: [] for n in top_ns}
        
    # 硬性篩選門檻：過零率 >= 12 且 spread_std >= 雙邊交易成本(2*0.0029)
    # 取代掉原先模糊的 ~((...<12) & (...>252)) 條件
    df_passed['spread_std'] = df_passed['profit_space'] / 2.0  # 假設預設抓的範圍
    # 我們將真正的 spread_std 計算留在迴圈中並提取
"""

replace_loop = """            spread = a_norm - beta * b_norm
            
            spread_std = spread.std()
            p['spread_std'] = spread_std
            if spread_std < 0.0058: 
                p['ssd'] = np.inf 
                p['zero_cross'] = 0
                continue # 跳過後續計算，直接處理下一組

            # SSD 為價差的平方和"""

c5_src = c5_src.replace(
"""            spread = a_norm - beta * b_norm
            
            spread_std = spread.std()
            if spread_std < (TRANSACTION_COST * 2): 
                p['ssd'] = np.inf 
                p['zero_cross'] = 0
                continue # 跳過後續計算，直接處理下一組

            # SSD 為價差的平方和""", replace_loop)

replace_filter = """    df_passed = df_passed[~((df_passed['zero_cross'] < 12) & (df_passed['half_life'] > FORMATION_WINDOW / 2))]
    if df_passed.empty:
        return {n: [] for n in top_ns}"""

new_filter = """    # 硬性篩選門檻
    df_passed = df_passed[(df_passed['zero_cross'] >= 12) & (df_passed['spread_std'] >= 0.0058)]
    if df_passed.empty:
        return {n: [] for n in top_ns}"""

c5_src = c5_src.replace(replace_filter, new_filter)
code_cells[5]['source'] = [line + ("\n" if i < len(c5_src.split("\n"))-1 else "") for i, line in enumerate(c5_src.split("\n"))]

new_cells = [
    c_markdown(md1),
    code_cells[0],
    code_cells[1],
    code_cells[2],
    code_cells[3],
    c_markdown(md2),
    code_cells[4],
    c_markdown(md3),
    code_cells[5],
    c_markdown(md4),
    code_cells[6],
    code_cells[7],
    c_markdown(md5),
    code_cells[8],
    c_markdown(md6),
    code_cells[9]
]

nb['cells'] = new_cells

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("done")
