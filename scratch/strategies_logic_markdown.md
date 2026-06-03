

<!-- ==================== CELL 0 ==================== -->

# 📊 S&P 500 Pairs Trading 策略交易邏輯與回測系統架構詳解

## 📝 執行摘要與系統概述

本 Jupyter Notebook 旨在**完整且詳細地整理本量化平台內的所有配對交易 (Pairs Trading) 策略交易邏輯與核心架構**。
配對交易是一種經典的**市場中性 (Market Neutral) 統計套利策略**，其基本假設是：若兩隻資產（或其對數組合）在歷史上存在長期的平穩走勢關聯，則當其短期走勢出現異常偏離時，價差必定會在未來收斂。此時，可以通過「買低賣高（多弱空強）」來獲取無風險的統計套利收益。

本平台具備以下**三大核心技術特徵**：
1. **科學化滾動視窗驗證 (Rolling Walk-Forward Backtesting)**：系統採用滾動交疊視窗設計。每一期分為**形成期 (Formation Period)**（預設 $F = 252$ 天）與**交易期 (Trading Period)**（預設 $T = 126$ 天），以 rolling step ($S = 21$ 天) 滾動往前推進。各期獨立篩選配對並進行交易，實現多週期並行的帳戶權益追蹤。
2. **高效能並行網格搜尋引擎 (Parallelized Grid Search)**：利用 Python 多進程 (Multiprocessing) 加速，在主進程中一次性讀取 SQLite 並 Pivot 價格矩陣，子進程共用內存數據，針對 7 組核心網格參數（Top N、Stop Loss、Z-Window、PSL、MSR、DSZ 與 VOL）進行極速搜尋。
3. **實時指標編譯與互動視覺化**：`preprocess_equity.py` 編譯出最優淨值與 RCC/REC 等指標，並將 HTML 表格動態注入分析 Notebook 中，Streamlit Dashboard 則提供多維度 Filter 與下鑽到單一配對的 **Trade Visualizer**，動態還原開平倉與停損軌跡。

---

<!-- ==================== CELL 1 ==================== -->

## 🧠 1. 共享核心量化架構與風控機制

在深入各策略的具體選股與建倉邏輯之前，本章節先詳細說明本平台所有策略所**共用**的基礎統計模型、高階風控機制、績效評估指標與資料庫管線設計。

### 1.1 核心統計模型與數學公式

#### A. 最小二乘法 (OLS) 線性回歸
用於估計配對標的價格之間的長期均衡係數：
$$y_t = \alpha + \beta \cdot x_t + \epsilon_t$$
其中：
- $y_t$ 為標的 A 的價格（或對數價格）
- $x_t$ 為標的 B 的價格（或對數價格）
- $\beta$ 為避險比例 (Hedge Ratio)
- $\alpha$ 為均值偏離常數 (OLS Alpha)
- $\epsilon_t$ 為殘差序列 (Residuals)

#### B. Engle-Granger 共整合單根檢定 (ADF Test)
對 OLS 迴歸產生的殘差序列 $\epsilon_t$ 執行無截距項、無時間趨勢項的簡化 Augmented Dickey-Fuller (ADF) 單根檢定，以驗證殘差是否具備平穩性 (Stationarity)：
$$\Delta \epsilon_t = \gamma \epsilon_{t-1} + \sum_{i=1}^p \delta_i \Delta \epsilon_{t-i} + e_t$$
- **零假設 ($H_0$)**：$\gamma = 0$（存在單根，殘差不平穩，兩者無共整合關係）。
- **對立假設 ($H_1$)**：$\gamma < 0$（殘差平穩，具備共整合關係，價差會均值回歸）。
系統同時計算正反雙向回歸（A對B 與 B對A），並**選擇 ADF p-value 較小（最顯著）的方向**作為交易方向，藉此根除因隨意指定自變量而產生的共整合判定偏差。

#### C. 誤差修正模型 (ECM) 與半衰期 (Half-Life) 估計
為了量化價差從偏離回歸至均值中心的速度，系統對通過共整合檢定的配對建立一階誤差修正模型：
$$\Delta \epsilon_t = a + \lambda \cdot \epsilon_{t-1} + u_t$$
其中 $\lambda$ 為均值回歸速度（必須為負值才具收斂性）。均值回歸的半衰期 (Half-life) 指價差偏離減半所需的時間，計算公式為：
$$\text{Half-Life} = -\frac{\ln(2)}{\lambda}$$
本平台顯式加入**半衰期過濾門檻**，僅保留半衰期在 $[2, 60]$ 交易日之內的配對。若半衰期小於 2 日，可能代表高頻微幅噪聲，交易成本將蠶食利潤；若大於 60 日，則代表回歸過慢，會產生嚴重的資金佔用風險。

### 1.2 高階風控與部位管理系統

本平台具備完善的六大風控防線，確保策略在黑天鵝事件或非對稱極端走勢下不致產生毀滅性虧損：
1. **個股單筆停損 (`SL`)**：當特定交易配對的未實現虧損比例達到或超過 `stop_loss_pct`（以分配給該配對的資金為分母）時，強制執行平倉。若設定不允許重新進場，該配對在該期內將永久凍結 (`is_stopped = True`)。
2. **部位動態 Z-Score 偏離停損 (`DSZ`)**：當價差 Z-Score 絕對值偏離過大，例如 $|Z_t| > \text{dynamic\_stop\_z}$（如 3.0 或 5.0），說明發生了結構性破裂 (Structural Break) 或極端非對稱基本面事件，系統將即刻啟動動態停損出場，而非盲目持有等待均值回歸。
3. **全域投資組合層級最大回撤停損 (`PSL`)**：每日計算併發多個交疊週期累積的 PnL。當總資金虧損達到 `portfolio_stop_loss_pct` (如 10%) 時，觸發 PSL，**一鍵斬倉所有持倉配對**，重置所有插槽部位，並在剩餘回測期內凍結所有交易，以保護核心本金。
4. **產業分散化集中度上限 (`MSR`)**：為了防止配對標的過度集中在某一特定產業（例如金融或科技板塊），系統設定 `max_sector_ratio` 參數。每一期單一產業選入的配對對數上限為 $\max(1, \lfloor N \cdot \text{max\_sector\_ratio} \rfloor)$，實現外部高度多元化。
5. **方向性建倉冷卻機制 (Cooldown Period)**：平倉後（不論是正常平倉或停損且允許重新進場），系統會進入冷卻狀態。
   - 若原先為做多 (Position = 1)，則必須等到 $Z_t \ge -EXIT\_Z$ 才能解除冷卻，再次評估進場。
   - 若原先為做空 (Position = -1)，則必須等到 $Z_t \le EXIT\_Z$ 才能解除冷卻。
6. **自適應波動率調節機制 (`VOL ADJ`)**：若近期市場波動度顯著放大，系統會計算 20 日滾動價差標準差，對形成期基礎標準差進行倍數乘積放大：
   $$\sigma_{adjusted} = \sigma_{formation} \times \max\left(1.0, \frac{\sigma_{roll20}}{\sigma_{formation}}\right)$$
   使 Z-Score 隨波動率自適應收縮，防止在市場無序劇烈動盪中頻繁觸發無效建倉訊號。

### 1.3 核心績效評估指標與公式

- **最終權益值 (Final Equity)**：$\text{Initial Capital} + \text{Cumulative PnL}$
- **年化報酬率 (Annualized Return, CAGR)**：幾何年化公式
  $$\text{Ann\_Ret} = (1 + \text{Cum\_Ret})^{12 / n\_{months}} - 1$$
- **夏普比率 (Sharpe Ratio)**：以 252 交易日進行年化
  $$\text{Sharpe} = \sqrt{252} \times \frac{\text{Mean}(R_{daily})}{\text{Std}(R_{daily})}$$
- **原始資金約束報酬率 (RCC, Return on Capital Constraint)**：以初始配置資金為分母
  $$\text{RCC} = \frac{\text{Final PnL}}{\text{Initial Capital}}$$
- **實際參與資金報酬率 (REC, Return on Engaged Capital)**：以實際發生建倉交易的配對所佔用的總資金為分母，更能反映策略資金的使用效率
  $$\text{REC} = \frac{\text{Final PnL}}{N_{traded} \times C_{pair}}$$
  其中 $C_{pair} = \text{Initial Capital} / N$ 為分配給單一配對的資金限制。

### 1.4 SQLite 數據庫管道 (Database Pipeline) 與 schema 設計

回測引擎產出的 CSV 明細將自動匯入至 SQLite `result.db` 資料庫。資料庫採用 **WAL (Write-Ahead Logging)** 併發讀寫優化模式與 **PRAGMA synchronous = NORMAL** 寫入優化，並包含三張核心資料表：
1. `strategy_summaries`：儲存各組策略參數組合回測的最終績效統計摘要。
2. `trade_logs`：儲存每日交易明細，包含價格、避險比例、Z-Score、持倉部位及 realized/unrealized PnL 等。
3. `strategy_pairs`：儲存各交易期選定配對的形成期引數、對稱產業以及 Log_Mean 等基本資訊，優化 Trade Visualizer 點擊下鑽時的檢索效率。

---

<!-- ==================== CELL 2 ==================== -->

## 距離策略一：經典 SSD (Basic) 距離策略

### 2.1 策略原理與數學特徵

經典 SSD (Basic) 策略基於**價格走勢歐氏距離最小化**原則，是最為傳統與直觀的配對交易策略。其核心數學邏輯如下：

#### A. 價格正規化 (Price Normalization)
由於不同股票的絕對價格級距存在巨大差異，為了在同一維度下比對走勢，SSD 策略在**形成期第一天價格**的基準上對所有股票進行正規化：
$$P'_{i, t} = \frac{P_{i, t}}{P_{i, 0}}$$
其中 $P_{i, 0}$ 為形成期第一天股票 $i$ 的收盤價。正規化後，所有股票在第一天的價格皆為 1.0。

#### B. 歐氏距離平方和 (Sum of Squared Differences, SSD)
計算形成期內所有股票兩兩之間的走勢距離平方和。為了防止非理性跨行業配對，本平台強加**產業對照表過濾**，僅在相同產業板塊內進行兩兩計算：
$$\text{SSD}_{A,B} = \sum_{t=1}^F (P'_{A,t} - P'_{B,t})^2$$
where $F$ 為形成期的總交易日天數。SSD 數值越小，代表兩檔股票在形成期內的走勢高度趋同。系統將 SSD 依據升序排序，選取前 $N$ 對最小者進入交易期。

#### C. 對沖比例與資金分配
- **對沖比例固定**：在 Basic 版中，對沖比例 $\beta$ **固定設定為 1.0**。
- **等權重資金分配**：建倉時，分配給該對配對的資金 $C_{pair}$ 會被**均等平分**給兩隻股票，即股票 A 與股票 B 各分配到 $C_{pair} \times 0.5$ 的市值部位。這種等權重分配隱含著「兩隻股票的波動性與貝他風險對等」的假設。
  $$v_a = C_{pair} \times 0.5, \quad v_b = C_{pair} \times 0.5$$

#### D. 交易價差與 Z-Score
- **即時價差 (Spread)**：
  $$Spread_t = P'_{A,t} - P'_{B,t}$$
- **Z-Score 生成**：
  - 若 `ZSCORE_WINDOW = 0` (固定參數模式)，則使用形成期計算出的價差均值 $\mu_{form}$ 與標準差 $\sigma_{form}$ 作為標準化中心：
    $$Z_t = \frac{Spread_t - \mu_{form}}{\sigma_{form}}$$
  - 若 `ZSCORE_WINDOW > 0` (滾動視窗模式)，則使用交易期過去 $W$ 天的滾動價差均值 $\mu_{roll, t}$ 與滾動價差標準差 $\sigma_{roll, t}$ 作為中心：
    $$Z_t = \frac{Spread_t - \mu_{roll, t}}{\sigma_{roll, t}}$$

### 2.2 Python 核心程式碼實現
以下是 `strategies/ssd_basic.py` 中負責配對形成 (`Formation`) 與交易模擬 (`Trading`) 的完整程式碼：

<!-- ==================== CODE CELL 3 ==================== -->

```python
# 嵌入 ssd_basic.py 的 Formation 與 Trading 實現
"""
SSD 配對交易滾動回測系統 (交易明細基本版)
===========================================

核心功能：基於 SSD (Sum of Squared Differences) 距離指標與 Z-Score 機制進行配對交易回測。
系統採用滾動視窗 (Rolling Walk-Forward) 設計，支援大型參數網格搜索 (Grid Search)，
並對檔案輸出效能進行最佳化（直接串流寫入明細至 CSV 檔案），利於低記憶體耗用下進行大量歷史回測。
"""

import sqlite3
import warnings
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd

# 忽略不必要的警告，確保主控台輸出簡潔
warnings.filterwarnings("ignore")


class Formation:
    """
    配對形成期處理器。
    計算形成期內所有股票兩兩之間的 SSD (歐式距離平方)，並依據產業分類進行篩選，挑選出距離最近的配對組合。
    """
    
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, sector_mapping: dict = None, min_tickers_for_pairing: int = 2):
        """
        初始化配對形成期配置。

        參數:
            price_df (pd.DataFrame): 形成期的歷史價格矩陣（Index 為日期，Columns 為股票代碼）。
            form_start (str): 形成期開始日期。
            form_end (str): 形成期結束日期。
            top_n (int): 每期選取的最佳配對對數，預設為 20 對。
            sector_mapping (dict): 股票代碼對應產業名稱的字典。若無，則採用全市場配對。
            min_tickers_for_pairing (int): 產業內最少必須有多少檔標的才允許進行配對，預設為 2 檔。
        """
        self.price_df = price_df
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        # 初始化儲存正規化價格與最終選定配對的變數
        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.first_day_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """
        將形成期內所有股票的價格進行正規化（除以該期第一天的價格），以消除各股票絕對價格級距的差異。

        回傳:
            pd.DataFrame: 正規化後的價格矩陣。
        """
        self.first_day_prices = self.price_df.iloc[0]
        # 防止第一天價格為 0 導致除以零的錯誤
        safe_first_prices = np.where(self.first_day_prices > 1e-8, self.first_day_prices, 1.0)
        self.normalized_df = self.price_df / safe_first_prices
        return self.normalized_df

    def compute_ssd(self) -> pd.DataFrame:
        """
        在產業分類限制下，計算所有股票兩兩之間的 Sum of Squared Differences (SSD)。

        回傳:
            pd.DataFrame: 包含所有候選配對的 SSD、Hedge Ratio、價差均值與標準差等欄位的 DataFrame。
        """
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        ssd_records = []

        # 依據產業分類進行分組
        sector_groups = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        skipped_unknown_count = 0
        for sector, sector_tickers in sector_groups.items():
            # 跳過未分類 (Unknown) 標的，避免無意義配對
            if sector == "Unknown":
                skipped_unknown_count = len(sector_tickers)
                continue

            # 若該產業內標的數量小於起配門檻，則無法成對，直接跳過
            if len(sector_tickers) < self.min_tickers_for_pairing:
                continue

            # 轉置正規化價格矩陣，以利快速計算距離
            norm_vals = self.normalized_df[sector_tickers].values.T

            # 利用 scipy 的 pdist 高效計算 pairwise 的 squared Euclidean 距離
            ssd_matrix = ssd.pdist(norm_vals, metric='sqeuclidean')

            idx = 0
            for i in range(len(sector_tickers)):
                ticker_b = sector_tickers[i]
                x_val = norm_vals[i]

                for j in range(i + 1, len(sector_tickers)):
                    ticker_a = sector_tickers[j]
                    y_val = norm_vals[j]

                    ssd_value = ssd_matrix[idx]
                    idx += 1

                    # 在基本 SSD 模型中，對沖比例 (Hedge Ratio) 固定設為 1.0
                    beta = 1.0
                    spread = y_val - beta * x_val
                    spread_mean = np.mean(spread)
                    spread_std = np.std(spread, ddof=1) if len(spread) > 1 else 0.0

                    ssd_records.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                        "SSD": round(ssd_value, 6), "Hedge_Ratio": round(beta, 4),
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std": round(spread_std, 6)
                    })

        if skipped_unknown_count > 0:
            print(f"  [Formation] 跳過 {skipped_unknown_count} 支未分類 (Unknown) 股票。")

        if not ssd_records:
            return pd.DataFrame()

        # 將所有候選配對依據 SSD 由小到大排序（SSD 越小代表價格走勢越趨同）
        return pd.DataFrame(ssd_records).sort_values("SSD").reset_index(drop=True)

    def select_pairs(self) -> pd.DataFrame:
        """
        篩選出 SSD 最小的前 N 對配對，並記錄建倉所需的原始期初價格。

        回傳:
            pd.DataFrame: 選定的前 N 對配對及其參數特徵。
        """
        ssd_df = self.compute_ssd()
        if ssd_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 挑選前 Top N 個最佳配對
        selected = ssd_df.head(self.top_n).copy()
        selected["Rank"] = range(1, len(selected) + 1)

        # 記錄形成期第一天價格，用於後續交易期的價格正規化基準
        first_price_a_list, first_price_b_list = [], []
        for _, row in selected.iterrows():
            first_price_a_list.append(self.price_df[row["Ticker_A"]].iloc[0])
            first_price_b_list.append(self.price_df[row["Ticker_B"]].iloc[0])

        selected["First_Price_A"] = first_price_a_list
        selected["First_Price_B"] = first_price_b_list

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        """
        執行完整的配對形成工作流。

        回傳:
            pd.DataFrame: 最終選定的配對清單。
        """
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs


@dataclass(slots=True)
class PairState:
    """
    單一配對在回測交易期中的即時持倉與損益狀態容器。
    使用 slots 以優化記憶體占用與屬性讀寫效能。
    """
    position: int = 0             # 目前持倉部位：0=無持倉, 1=多 A 空 B, -1=空 A 多 B
    shares_a: float = 0.0         # 標的 A 的持有股數（正數為多，負數為空）
    shares_b: float = 0.0         # 標的 B 的持有股數（正數為多，負數為空）
    entry_price_a: float = 0.0    # 標的 A 進場價格
    entry_price_b: float = 0.0    # 標的 B 進場價格
    realized_pnl: float = 0.0     # 該配對已實現的累計損益
    trade_entry_fee: float = 0.0  # 進場交易摩擦成本（手續費+滑價）
    days_held: int = 0            # 當前部位已持有天數
    is_stopped: bool = False      # 該配對是否已觸發永久停損（若不允許重新進場，觸發後該期不再交易）
    cooldown_dir: int = 0         # 冷卻方向限制：避免剛停損出場後立刻在同方向建倉
    prev_total_pnl: float = 0.0   # 前一日的總損益值，用於計算每日 Delta


class Trading:
    """
    配對交易模擬器。
    在指定的交易期內，根據即時的 Z-Score 計算交易訊號，執行開倉、平倉、停損等模擬交易。
    """
    
    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float, stop_loss_pct: float, entry_z: float, exit_z: float, zscore_window: int, allow_reentry: bool = False,
                 zscore_clip: float = 10.0, min_spread_std: float = 1e-6, use_dynamic_stop: bool = False, dynamic_stop_z: float = 3.0,
                 portfolio_stop_loss_pct: float = 0.10, use_vol_adjust: bool = False):
        """
        初始化交易模擬配置。

        參數:
            price_df (pd.DataFrame): 交易期的歷史價格矩陣（包含計算 Z-Score 所需的歷史區間）。
            trade_dates (pd.DatetimeIndex): 實際可進行交易的日期序列。
            selected_pairs (pd.DataFrame): 該期被選定的配對清單。
            capital_per_pair (float): 分配給單一配對的交易資金。
            fee_rate (float): 單邊交易手續費率。
            slippage_rate (float): 單邊交易滑價率。
            stop_loss_pct (float): 配對個損門檻比例（以分配資金為分母）。若 <= 0 則不啟用。
            entry_z (float): 建倉 Z-Score 門檻（絕對值）。
            exit_z (float): 平倉 Z-Score 門檻（接近 0 的臨界值）。
            zscore_window (int): 滾動 Z-Score 的計算視窗大小。若為 0 則使用固定形成期參數。
            allow_reentry (bool): 觸發停損後是否允許在同一期內再度建倉。
            zscore_clip (float): Z-Score 極端值截斷門檻。
            min_spread_std (float): 價差最小標準差限制，防止除以極小值導致訊號暴增。
            use_dynamic_stop (bool): 是否啟用 Z-Score 動態極端值停損機制。
            dynamic_stop_z (float): Z-Score 動態停損門檻（絕對值）。
            portfolio_stop_loss_pct (float): 投資組合層級的最大累積停損門檻。
            use_vol_adjust (bool): 是否啟用波動度調整（調整價差標準差）。
        """
        self.price_df = price_df
        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair
        
        # 合併單邊交易摩擦費率（買賣皆會產生）
        self.friction_rate = fee_rate + slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.zscore_window = zscore_window
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust = use_vol_adjust

        self.period_pnl: float = 0.0

    def _execute_entry(self, state: PairState, z: float, p_a: float, p_b: float) -> tuple[bool, float]:
        """
        執行建倉邏輯，依據 Z-Score 的極值方向進行買多賣空配置。

        參數:
            state (PairState): 配對狀態物件。
            z (float): 當前 Z-Score。
            p_a (float): 標的 A 的價格。
            p_b (float): 標的 B 的價格。

        回傳:
            tuple: (是否建倉成功, 產生的手續費/滑價成本負數)
        """
        # 將資金均分給兩個標的（各 50%）
        v_a = self.capital_per_pair * 0.5
        v_b = self.capital_per_pair * 0.5

        # 價差高估 -> 空 A 多 B (z 為正值且大於開倉線)
        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b = v_b / p_b
        # 價差低估 -> 多 A 空 B (z 為負值且小於負開倉線)
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a = v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        # 計算進場交易摩擦成本
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state: PairState, current_trade_pnl: float, stop_loss: bool = False):
        """
        執行平倉邏輯，結算該筆交易的損益並重置持倉狀態。

        參數:
            state (PairState): 配對狀態物件。
            current_trade_pnl (float): 當前交易淨損益（已扣除摩擦成本）。
            stop_loss (bool): 是否為停損出場。
        """
        state.realized_pnl += current_trade_pnl

        if stop_loss:
            # 若不允許重新進場，則將狀態設為已停損停拍
            state.is_stopped = True if not self.allow_reentry else False
            # 啟用冷卻方向限制，防止立刻進同方向
            if self.allow_reentry:
                state.cooldown_dir = state.position
        else:
            # 正常收斂平倉，亦要冷卻該方向直至回歸均值
            state.cooldown_dir = state.position

        # 重置所有持倉相關狀態欄位
        state.position = 0
        state.shares_a = 0.0
        state.shares_b = 0.0
        state.entry_price_a = 0.0
        state.entry_price_b = 0.0
        state.trade_entry_fee = 0.0

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float,
                       form_spread_mean: float, form_spread_std: float, first_price_a: float, first_price_b: float) -> pd.DataFrame:
        """
        對單一配對在整個交易期間進行日次模擬。

        參數:
            ... 包含形成期產出的統計參數與價格正規化基準。

        回傳:
            pd.DataFrame: 該配對在交易期每日的損益與持倉歷程紀錄。
        """
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: 
            return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: 
            return pd.DataFrame()

        # 進行與形成期一致的價格正規化處理
        norm_p_a = price_a / (first_price_a if first_price_a > 1e-8 else 1.0)
        norm_p_b = price_b / (first_price_b if first_price_b > 1e-8 else 1.0)

        # 依據設定計算每日 Z-Score
        if self.zscore_window == 0:
            # 固定參數模式：使用形成期計算之均值與標準差
            spread = norm_p_a - norm_p_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                # 波動度調整：若近期波動放大，適度調高價差標準差，使訊號收斂更具彈性
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series = pd.Series(1.0, index=common_idx)
        else:
            # 滾動均值模式：以滾動視窗計算動態價差中心與標準差
            roll_mean_a = norm_p_a.rolling(window=self.zscore_window).mean()
            roll_mean_b = norm_p_b.rolling(window=self.zscore_window).mean()
            roll_alpha = roll_mean_a - roll_mean_b
            spread = norm_p_a - roll_alpha - norm_p_b

            roll_std = (norm_p_a - norm_p_b).rolling(window=self.zscore_window).std()
            # 若滾動期間價差近乎為 0（例如長時間停牌），則判定不合適交易，直接跳過
            if (roll_std < self.min_spread_std * 10).mean() > 0.5:
                return pd.DataFrame()

            safe_std = np.maximum(roll_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore = np.clip(spread / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series = pd.Series(1.0, index=common_idx)

        # 僅截取實際交易日期區間的資料
        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0: 
            return pd.DataFrame()

        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        zscore = zscore.loc[valid_idx]
        beta_series = beta_series.loc[valid_idx]

        dates_arr = valid_idx
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values

        base_log = {
            "Period_Start": period_start, "Period_End": period_end,
            "Sector": sector, "Pair_Rank": pair_rank,
            "Ticker_A": ticker_a, "Ticker_B": ticker_b,
            "First_Price_A": first_price_a, "First_Price_B": first_price_b
        }

        state = PairState()

        # 初始化紀錄明細容器
        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_z, out_pos = [], [], []
        out_unrealized, out_realized, out_cum = [], [], []
        out_status, out_trade_pnl, out_days, out_delta = [], [], [], []

        # 進行逐日狀態機模擬
        for i in range(len(dates_arr)):
            date = dates_arr[i]
            z = 0.0 if np.isnan(zscore_arr[i]) else zscore_arr[i]
            p_a, p_b = pa_arr[i], pb_arr[i]

            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0
            daily_delta = 0.0
            current_status = "HOLD_CASH"

            # 情況 1：已觸發永久停損，不再參與交易
            if state.is_stopped:
                out_dates.append(date)
                out_pa.append(round(p_a, 4))
                out_pb.append(round(p_b, 4))
                out_hr.append(1.0)
                out_z.append(round(float(z), 4))
                out_pos.append(0)
                out_unrealized.append(0.0)
                out_realized.append(round(float(state.realized_pnl), 4))
                out_cum.append(round(float(state.realized_pnl), 4))
                out_status.append("STOPPED")
                out_trade_pnl.append(0.0)
                out_days.append(0)
                out_delta.append(0.0)
                continue

            # 情況 2：解除冷卻機制（當 Z-Score 回歸到平倉線內）
            if state.cooldown_dir == -1 and z <= self.exit_z:
                state.cooldown_dir = 0
            elif state.cooldown_dir == 1 and z >= -self.exit_z:
                state.cooldown_dir = 0

            # 情況 3：持倉狀態
            if state.position != 0:
                state.days_held += 1
                # 計算未實現的價差毛損益
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                # 預估平倉所產生之手續費與滑價
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate

                # 交易淨損益 = 價差毛損益 - 進場費用 - 預估出場費用
                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est

                # 檢查個股累積虧損停損
                is_cap_stop = self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                # 檢查 Z-Score 動態偏離停損
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    # 執行停損平倉
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED"
                else:
                    # 正常訊號收斂平倉判斷
                    is_exit_short = (state.position == -1) and (z <= self.exit_z)
                    is_exit_long  = (state.position == 1)  and (z >= -self.exit_z)

                    if is_exit_short or is_exit_long:
                        self._execute_close(state, current_trade_pnl, stop_loss=False)
                        closed_trade_pnl = current_trade_pnl
                        current_status = "EXIT"
                    else:
                        unrealized_pnl = current_trade_pnl
                        current_status = "HOLDING"
            # 情況 4：空倉狀態，尋求建倉機會
            else:
                if abs(z) > self.entry_z:
                    entered, unrealized_pnl = self._execute_entry(state, z, p_a, p_b)
                    if entered:
                        current_status = "ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A"
                    else:
                        current_status = "HOLD_CASH (COOLDOWN)"
                else:
                    current_status = "HOLD_CASH"

            # 計算累計損益與每日損益 Delta
            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl

            # 寫入每日明細紀錄
            out_dates.append(date)
            out_pa.append(round(p_a, 4))
            out_pb.append(round(p_b, 4))
            out_hr.append(1.0)
            out_z.append(round(float(z), 4))
            out_pos.append(state.position)
            out_unrealized.append(round(float(unrealized_pnl), 4))
            out_realized.append(round(float(state.realized_pnl), 4))
            out_cum.append(round(float(cumulative_pnl), 4))
            out_status.append(current_status)
            out_trade_pnl.append(round(float(closed_trade_pnl), 4))
            out_days.append(state.days_held)
            out_delta.append(round(float(daily_delta), 4))

            if current_status in ["STOP_LOSS_TRIGGERED", "EXIT"]:
                state.days_held = 0

            # 若已觸發永久停損，則直接快速填充後續交易日為 STOPPED 狀態，避免多餘計算
            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    rd = dates_arr[j]
                    r_z = 0.0 if np.isnan(zscore_arr[j]) else zscore_arr[j]
                    r_pa, r_pb = pa_arr[j], pb_arr[j]

                    out_dates.append(rd)
                    out_pa.append(round(r_pa, 4))
                    out_pb.append(round(r_pb, 4))
                    out_hr.append(1.0)
                    out_z.append(round(float(r_z), 4))
                    out_pos.append(0)
                    out_unrealized.append(0.0)
                    out_realized.append(round(float(state.realized_pnl), 4))
                    out_cum.append(round(float(state.realized_pnl), 4))
                    out_status.append("STOPPED")
                    out_trade_pnl.append(0.0)
                    out_days.append(0)
                    out_delta.append(0.0)
                break

        # 交易期結束若仍持倉，執行強制平倉結算
        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                pnl_before_last_day = out_cum[-2] if len(out_cum) > 1 else 0.0

                p_a_last, p_b_last = pa_arr[-1], pb_arr[-1]
                raw_unrealized_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                exit_fee = (abs(state.shares_a)*p_a_last + abs(state.shares_b)*p_b_last) * self.friction_rate

                closed_trade_pnl = raw_unrealized_final - state.trade_entry_fee - exit_fee
                state.realized_pnl += closed_trade_pnl
                daily_delta = state.realized_pnl - pnl_before_last_day

                out_status[-1] = "PERIOD_END_EXIT"
                out_realized[-1] = round(state.realized_pnl, 4)
                out_cum[-1] = round(state.realized_pnl, 4)
                out_unrealized[-1] = 0.0
                out_trade_pnl[-1] = round(closed_trade_pnl, 4)
                out_delta[-1] = round(daily_delta, 4)
                out_days[-1] = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unrealized, "Realized_PnL": out_realized,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_trade_pnl, "Days_Held": out_days, "Daily_Delta": out_delta
        })

        for k, v in base_log.items():
            df_out[k] = v

        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """
        平行模擬所有選定的配對，並依據設定判斷是否觸發投資組合層級停損。

        參數:
            period_start (str): 交易期開始時間。
            period_end (str): 交易期結束時間。

        回傳:
            tuple: (合併後的所有配對日交易明細 DataFrame, 該交易期總損益)
        """
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start,
                period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"],
                ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                first_price_a=float(row.get("First_Price_A", 1.0)),
                first_price_b=float(row.get("First_Price_B", 1.0))
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # 投資組合層級最大回撤停損機制 (Portfolio Stop Loss)
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()

            cutoff_date = None
            for date_val, pnl_val in daily_cum_pnl.items():
                # 若總累計虧損比例超過閾值，鎖定該切斷日期
                if pnl_val / total_cap <= -self.portfolio_stop_loss_pct:
                    cutoff_date = date_val
                    break

            # 執行投資組合停損切斷
            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date

                    df_before = df[before_mask]

                    # 當天被切斷：將部位歸零並標註狀態
                    df_at = df[at_mask].copy()
                    final_realized = 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        final_realized = row_at["Cumulative_PnL"]
                        if row_at["Position"] != 0:
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Trade_PnL"]
                        else:
                            if row_at["Status"] not in ("STOPPED", "STOP_LOSS_TRIGGERED", "EXIT"):
                                df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOPPED"
                            final_realized = row_at["Realized_PnL"]

                    # 切斷日之後的所有日期全部重置為空倉停止交易狀態
                    df_after = df[after_mask].copy()
                    if not df_after.empty:
                        df_after.loc[df_after.index, "Position"] = 0
                        df_after.loc[df_after.index, "Unrealized_PnL"] = 0.0
                        df_after.loc[df_after.index, "Realized_PnL"] = final_realized
                        df_after.loc[df_after.index, "Cumulative_PnL"] = final_realized
                        df_after.loc[df_after.index, "Status"] = "STOPPED"
                        df_after.loc[df_after.index, "Trade_PnL"] = 0.0
                        df_after.loc[df_after.index, "Daily_Delta"] = 0.0

                    new_dfs.append(pd.concat([df_before, df_at, df_after], ignore_index=True))
                dfs = new_dfs

        log_df = pd.concat(dfs, ignore_index=True)
        # 統計本期該參數組合產生的每日 Delta 和
        period_daily_delta = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily_delta.sum()) if not period_daily_delta.empty else 0.0

        return log_df, self.period_pnl


class DataProcessor:
    """
    資料庫載入與資料特徵前處理器。
    負責從 SQLite 中載入 Adj_Close，進行數據過濾（如移除停牌或缺值率過高的股票），填補缺失值，解析回測時間等。
    """
    
    def __init__(self, db_path: str, table_name: str = "daily_prices"):
        """
        參數:
            db_path (str): SQLite 資料庫路徑。
            table_name (str): 歷史日收盤價資料表名稱。
        """
        self.db_path, self.table_name = db_path, table_name

    def load_sector_mapping(self, info_table: str, ticker_col: str = "ticker", sector_col: str = "sector") -> dict:
        """
        從指定資料表載入個股對應之產業分類映射字典。

        參數:
            info_table (str): 分類資訊資料表名稱。
            ticker_col (str): 股票代碼欄位名稱。
            sector_col (str): 產業分類欄位名稱。

        回傳:
            dict: {股票代碼: 產業分類名稱} 的映射字典。
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT {ticker_col}, {sector_col} FROM {info_table}", conn)
            conn.close()
            mapping = {}
            for k, v in zip(df[ticker_col], df[sector_col]):
                if pd.notna(k) and pd.notna(v):
                    mapping[str(k).strip().upper()] = str(v).strip()
            print(f"✅ 成功載入產業分類表 '{info_table}'，共取得 {len(mapping)} 檔標的分類。")
            return mapping
        except Exception as e:
            print(f"⚠️ [警告] 無法載入產業分類表 '{info_table}'！錯誤原因：{e}")
            print(f"⚠️ 系統將退回「全市場(All_Market)」跨產業配對模式。")
            return {}

    def prepare_backtest_data(self, backtest_start: str, backtest_end: str, formation_window: int):
        """
        載入並重塑價格矩陣，過濾流動性差與大量缺值的個股。

        參數:
            backtest_start (str): 回測起始日期。
            backtest_end (str): 回測結束日期。
            formation_window (int): 形成期天數（需往前多載入資料以供第一期選配）。

        回傳:
            tuple: (價格 Pivot 矩陣, 時間索引序列, 總天數, 交易啟動本地日期索引位置)
        """
        conn = sqlite3.connect(self.db_path)
        # 合理載入 Adj_Close 或 Close 價格
        raw_df = pd.read_sql_query(f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC", conn)
        conn.close()

        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]

        # 重塑為以日期為 Index，Symbol 為 Columns 的寬表
        price_pivot = raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()

        # 剔除回測區間中缺失值比例大於 20% 的劣質標的，其餘以 forward fill 填補 (限制最多連續填 5 天)
        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20].ffill(limit=5)
        # 進一步確保剩餘股票含有至少 90% 的非空值，防止後續計算報錯
        price_pivot.dropna(axis=1, thresh=int(len(price_pivot) * 0.9), inplace=True)

        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                if is_end and len(str(d_str).strip()) == 7:
                    # 處理如 '2023-12' 自動抓月底日期
                    return dt + pd.offsets.MonthEnd(0)
                return dt
            except Exception:
                return None

        bt_start_ts = _safe_parse(backtest_start)
        bt_end_ts = _safe_parse(backtest_end, is_end=True)
        all_dates = price_pivot.index.tolist()

        start_indices = [i for i, d in enumerate(all_dates) if d >= bt_start_ts] if bt_start_ts else []
        first_idx = start_indices[0] if start_indices else 0

        # 將數據切片起點往前推一個 formation_window，供第一期 Formation 計算
        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end = bt_end_ts if bt_end_ts else price_pivot.index[-1]
        price_pivot = price_pivot.loc[data_slice_start:data_slice_end]

        sliced_dates = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_trade_idx = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_trade_idx, formation_window)


class RollingBacktester:
    """
    滾動回測引擎與網格搜索核心。
    執行滾動視窗 Walk-Forward backtesting，維護交易期併發資金池 (Trading Slots)，
    並以大資料寫入最優方式（串流 Append 追加寫入）降低內存壓力。
    """
    
    def __init__(self, top_n_list: list, stop_loss_list: list, zscore_window_list: list,
                 entry_z: float, exit_z: float, formation_window: int, trading_window: int, rolling_step: int,
                 fee_rate: float, slippage_rate: float, initial_capital: float,
                 allow_reentry: bool, zscore_clip: float, min_spread_std: float,
                 min_tickers_for_pairing: int, output_dir: Path,
                 portfolio_stop_loss_pct_list: list = None,
                 max_sector_ratio_list: list = None,
                 dynamic_stop_z_list: list = None,
                 use_vol_adjust_list: list = None,
                 **kwargs):
        """
        初始化滾動回測與網格搜索引擎。
        各清單型參數 (list) 代表網格搜索中所覆蓋的各候選數值。
        """
        self.top_n_list = top_n_list
        self.stop_loss_list = stop_loss_list
        self.zscore_window_list = zscore_window_list
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.rolling_step = rolling_step
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.output_dir = output_dir

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

        # 動態將其餘 kwargs 寫入類別屬性，相容進階擴充參數
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_csv_path(self, n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj) -> "Path":
        """
        依據參數組合，產生唯一的結果 CSV 檔名與路徑。

        回傳:
            Path: CSV 檔案儲存路徑。
        """
        sl_str  = f"SL{int(sl*100)}"       if sl       > 0 else "SL0"
        psl_str = f"PSL{int(p_stop*100)}"  if p_stop   > 0 else "PSL0"
        msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
        dsz_str = f"DSZ{int(dyn_z)}"       if dyn_z    > 0 else "DSZ0"
        vol_str = "VolAdj" if vol_adj else "NoVol"
        filename = f"TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
        return self.output_dir / filename

    def run(self, price_pivot: pd.DataFrame, all_dates: list, total_days: int, local_first_trade_idx: int, sector_mapping: dict):
        """
        執行多參數網格搜索與滾動回測。

        參數:
            price_pivot (pd.DataFrame): 價格 Pivot 矩陣。
            all_dates (list): 所有日期列表。
            total_days (int): 總交易日天數。
            local_first_trade_idx (int): 第一期可開始交易的本地索引位置。
            sector_mapping (dict): 股票產業對應表。
        """
        # 計算同一交易時間重疊的期數數量 (最大併發插槽數)
        max_concurrent = self.trading_window // self.rolling_step
        states = {}

        # 展開所有網格組合
        all_param_combos = list(itertools.product(
            self.top_n_list, self.stop_loss_list, self.zscore_window_list,
            self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list,
            self.dynamic_stop_z_list, self.use_vol_adjust_list
        ))
        
        # 初始化每一個參數組合的併發 Slots 與寫入狀態
        for combo in all_param_combos:
            n = combo[0]
            states[combo] = {
                "header_written": False,
                "row_count": 0,
                # 將初始資金平分給多個併發 Slot，實現多個交疊週期的獨立複利追蹤
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                          for _ in range(max_concurrent)]
            }

        # 依據 rolling_step 生成滾動視窗的交易啟動日期索引列表
        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始進行 Grid Search，共 {len(roll_start_indices)} 期，每期處理 {len(states)} 種參數組合...")

        # 滾動Walk-Forward主迴圈
        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx, form_end_idx = trade_start_idx - self.formation_window, trade_start_idx
            trade_end_idx = min(trade_start_idx + self.trading_window, total_days)

            form_data_raw = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data_raw = price_pivot.iloc[trade_start_idx:trade_end_idx]

            # 延伸交易數據起點，確保 Z-Score 滾動視窗有足夠的初期歷史數據可供計算
            extended_trade_start_idx = max(0, trade_start_idx - max(self.zscore_window_list))
            extended_trade_data_raw = price_pivot.iloc[extended_trade_start_idx:trade_end_idx]

            # 排除在形成期及交易期中存在 NaN 的不連續交易個股，保證數據完整性
            valid_cols = (form_data_raw.isnull().sum() + extended_trade_data_raw.isnull().sum()) == 0

            form_data = form_data_raw.loc[:, valid_cols]
            trade_data = trade_data_raw.loc[:, valid_cols]
            trade_dates = trade_data.index
            extended_trade_data = extended_trade_data_raw.loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty: 
                continue

            trade_start_str, trade_end_str = str(all_dates[trade_start_idx])[:10], str(all_dates[trade_end_idx - 1])[:10]
            form_start_str, form_end_str = str(all_dates[form_start_idx])[:10], str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 處理中：第 {roll_idx+1:02d} 期 (交易: {trade_start_str} ~ {trade_end_str})")

            # 建立並執行配對篩選，為了產業多元化，在此先挑選最寬廣的對數 (如 Max(Top N) * 5)
            formation = Formation(
                price_df=form_data,
                form_start=form_start_str,
                form_end=form_end_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty: 
                continue

            # 逐一處理網格參數組合
            for combo in all_param_combos:
                n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj = combo

                # 產業分散化篩選機制 (Max Sector Ratio)
                if sec_ratio > 0:
                    max_pairs_per_sector = max(1, int(n * sec_ratio))
                    sector_counts = {}
                    diversified_records = []
                    for _, row in max_selected_pairs.iterrows():
                        sec = row["Sector"]
                        if sec not in sector_counts:
                            sector_counts[sec] = 0
                        if sector_counts[sec] < max_pairs_per_sector:
                            diversified_records.append(row)
                            sector_counts[sec] += 1
                        if len(diversified_records) >= n:
                            break
                    selected_pairs = pd.DataFrame(diversified_records).copy()
                else:
                    selected_pairs = max_selected_pairs.head(n).copy()

                if selected_pairs.empty:
                    continue

                selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                state = states[combo]
                slots = state["slots"]

                # 分配可用交易插槽：優先分配給目前空閒且可用日期小於等於當前開始日期的插槽
                free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                if free_slots:
                    slot_idx = free_slots[0]
                else:
                    # 若無完全空閒插槽，則選擇空閒日期最早釋放者
                    slot_idx = min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                current_period_capital = slots[slot_idx]["capital"]
                current_capital_per_pair = current_period_capital / n

                # 初始化本期模擬器並運行
                trading = Trading(
                    price_df=extended_trade_data,
                    trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=current_capital_per_pair,
                    fee_rate=self.fee_rate,
                    slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl,
                    entry_z=self.entry_z,
                    exit_z=self.exit_z,
                    zscore_window=z_win,
                    allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip,
                    min_spread_std=self.min_spread_std,
                    use_dynamic_stop=(dyn_z > 0),
                    dynamic_stop_z=dyn_z,
                    portfolio_stop_loss_pct=p_stop,
                    use_vol_adjust=vol_adj
                )

                trade_log_df, period_pnl = trading.run(trade_start_str, trade_end_str)

                # 以追加 (a) 模式將明細即時寫入 CSV，保持超低記憶體佔用
                if not trade_log_df.empty:
                    filepath = self._get_csv_path(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)
                    write_header = not
```

<!-- ==================== CODE CELL 4 ==================== -->

```python
 state["header_written"]
                    trade_log_df.to_csv(
                        filepath,
                        mode="w" if write_header else "a",
                        header=write_header,
                        index=False
                    )
                    state["header_written"] = True
                    state["row_count"] += len(trade_log_df)

                # 結算該插槽的期末累計資金，並更新可用日期
                slots[slot_idx]["capital"] = max(0, current_period_capital + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        self._export_results(states, all_param_combos)

    def _export_results(self, states: dict, all_param_combos: list):
        """
        輸出網格搜索最終統計結果摘要至主控台。

        參數:
            states (dict): 儲存各組合狀態資訊的字典。
            all_param_combos (list): 所有參數組合清單。
        """
        print("\n✅ 回測完成！交易紀錄已串流寫入各 CSV 檔案。")
        for combo in all_param_combos:
            n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj = combo
            state = states[combo]
            if state["header_written"]:
                filepath = self._get_csv_path(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)
                print(f"  - 已輸出: {filepath.name} (共 {state['row_count']} 筆紀錄)")

        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")


def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    策略外部啟動入口接口 (定義一)。
    利用 inspect 自動過濾出 RollingBacktester 構造函數所需的合法參數並運行回測。

    參數:
        price_pivot: 價格矩陣。
        all_dates: 日期序列。
        total_days: 總天數。
        local_first_trade_idx: 首期交易位置。
        sector_mapping: 產業分類對應。
        params: 策略總配置字典。
        output_dir: 輸出資料夾路徑。
    """
    import inspect
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 動態解析構造函數簽名，自動過濾無效參數，防止報錯
    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default

    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )

    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")


def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    策略外部啟動入口接口 (定義二，保留原程式重複定義行為)。
    利用 inspect 自動過濾出 RollingBacktester 構造函數所需的合法參數並運行回測。
    """
    import inspect
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default

    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )

    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")

```

<!-- ==================== CELL 5 ==================== -->

## 距離策略二：進階 SSD (OLS) 殘差滾動策略

### 3.1 策略原理與數學特徵

相較於經典 SSD (Basic) 採用的簡單價格差與固定對沖比，**進階 SSD (OLS) 策略引進了對數價格空間與動態 OLS 回歸對沖比**，使其在學術與實務上都更具穩健性。其核心特徵如下：

#### A. 對數價格空間與標準化 (Log Price Space & Z-Score Standardize)
由於金融資產價格分布一般呈對數常態分布 (Log-normal distribution)，進階版將形成期歷史收盤價全部轉換為自然對數空間，以符合收益率加總特性。並對個股進行 Z-Score 標準化：
$$y_{i, t} = \ln(P_{i, t})$$
$$y'_{i, t} = \frac{y_{i, t} - \mu_{ln(P), i}}{\sigma_{ln(P), i}}$$
其中 $\mu_{ln(P), i}$ 與 $\sigma_{ln(P), i}$ 分別為形成期內股票 $i$ 的對數收盤價均值與標準差。

#### B. 動態避險比例 (Dynamic Hedge Ratio via OLS)
- **OLS 回歸避險比**：在形成期內，系統對同產業兩兩股票標準化後的對數價格執行一元線性 OLS 回歸：
  $$y'_{A, t} = \alpha + \beta \cdot y'_{B, t} + \epsilon_t$$
  藉由協方差與方差計算出動態對沖比例 $\beta$ (OLS 斜率係數)：
  $$\beta = \frac{\text{Cov}(y'_A, y'_B)}{\text{Var}(y'_B)}$$
  這克服了 Basic 版中將 $\beta$ 固定為 1.0 的局限性，能客觀反映兩者真實的波動倍數關聯。
- **對數價差歐氏距離平方和**：
  $$\text{SSD}_{A,B} = \sum_{t=1}^F (y'_{A,t} - \beta \cdot y'_{B,t})^2$$
  SSD 越小代表回歸殘差波動越平穩。系統依 SSD 升序選前 $N$ 對最佳配對。

#### C. 市值中性與資金加權分配
建倉時，不再採用Basic版的50/50等權重分配，而是**根據對沖比例 $\beta$ 進行動態資金加權分配**，以實現真實的**市值中性 (Beta-neutral)**：
- 總權重 $W_{total} = 1.0 + |\beta|$
- 標的 A 的資金佔比為 $v_a = C_{pair} \times \frac{1.0}{W_{total}}$
- 標的 B 的資金佔比為 $v_b = C_{pair} \times \frac{|\beta|}{W_{total}}$
這確保了在買入 A 賣出 B 時，空頭市值與多頭市值具備精準的避險對沖比例，極大化地規避了行業或市場的貝他風險。

#### D. 交易期動態滾動 OLS 回歸 Z-Score
- 若 `ZSCORE_WINDOW = 0` (固定參數模式)，價差及 Z-Score 計算使用形成期的靜態 $\alpha$，$\beta$，殘差均值與標準差。
- 若 `ZSCORE_WINDOW > 0` (滾動 OLS 回歸模式)，系統在交易期每日以 $W$ 天的滾動窗口**重新對兩者進行 OLS 回歸**，動態重新擬合當前的 $\alpha_t$ 與 $\beta_t$，並以回歸殘差計算 Z-Score。這使得策略能靈活調適長期對沖關係在交易期發生的緩慢漂移。

### 3.2 Python 核心程式碼實現
以下是 `strategies/ssd.py` 中負責配對形成 (`Formation`) 與交易模擬 (`Trading`) 的完整程式碼：

<!-- ==================== CODE CELL 6 ==================== -->

```python
# 嵌入 ssd.py 的 Formation 與 Trading 實現
"""
SSD 配對交易滾動回測系統 (交易明細進階版 - 動態對沖與對數空間)
============================================================

核心功能：基於對數價格空間中的 SSD (Sum of Squared Differences) 指標與 Z-Score 機制進行配對交易回測。
與基本版 (ssd_basic.py) 不同，本進階版具備以下特徵：
1. 價格正規化在對數空間進行：log_prices = ln(prices)，以符合資產價格呈對數常態分佈之特性。
2. 動態對沖比例 (Hedge Ratio)：在形成期利用協方差與方差計算 OLS 最小二乘法回歸係數 beta = cov(X, Y) / var(X)，而非固定為 1.0。
3. 動態資金分配：建倉時依據對沖比例 beta 動態按權重分配兩隻股票的資金佔比。
4. 滾動 OLS 回歸：在交易期滾動計算 Z-Score 時，若 zscore_window > 0，則利用滾動協方差動態估算 rolling_beta、rolling_alpha 以及殘差標準差，進行高階統計套利模擬。
"""

import sqlite3
import warnings
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd

# 忽略警告資訊
warnings.filterwarnings("ignore")


class Formation:
    """
    配對形成期處理器 (進階版)。
    在對數價格空間中，依據產業分類進行篩選，計算兩兩股票之 SSD，
    並利用 OLS 最小二乘法估算動態對沖比例 (Hedge Ratio)。
    """
    
    def __init__(self, price_df: pd.DataFrame, form_start: str, form_end: str, top_n: int = 20, sector_mapping: dict = None, min_tickers_for_pairing: int = 2):
        """
        初始化配對形成期配置。

        參數:
            price_df (pd.DataFrame): 形成期歷史收盤價矩陣。
            form_start (str): 形成期開始日期。
            form_end (str): 形成期結束日期.
            top_n (int): 選定配對數。
            sector_mapping (dict): 股票產業分類字典。
            min_tickers_for_pairing (int): 最少配對檔數門檻。
        """
        self.price_df = price_df
        self.form_start = form_start
        self.form_end = form_end
        self.top_n = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.normalized_df: pd.DataFrame = pd.DataFrame()
        self.mean_prices: pd.Series = pd.Series(dtype=float)
        self.std_prices: pd.Series = pd.Series(dtype=float)
        self.selected_pairs: pd.DataFrame = pd.DataFrame()

    def normalize_prices(self) -> pd.DataFrame:
        """
        將形成期價格轉換為對數價格空間，並進行標準化 (Z-Score 正規化)。
        正規化公式: normalized_df = (ln(P) - mean(ln(P))) / std(ln(P))

        回傳:
            pd.DataFrame: 對數空間標準化後的價格矩陣。
        """
        # 轉換為自然對數價格
        log_prices = np.log(self.price_df)
        self.mean_prices = log_prices.mean()
        self.std_prices = log_prices.std()
        # 進行標準化
        self.normalized_df = (log_prices - self.mean_prices) / self.std_prices
        return self.normalized_df

    def compute_ssd(self) -> pd.DataFrame:
        """
        在產業限制下，計算所有股票兩兩之間的 SSD 歐式距離，
        並藉由協方差矩陣估計對沖比例 beta (OLS 回歸係數)。

        回傳:
            pd.DataFrame: 包含 SSD、動態 Hedge Ratio、價差統計之候選配對 DataFrame。
        """
        if self.normalized_df.empty:
            self.normalize_prices()

        tickers = self.normalized_df.columns.tolist()
        ssd_records = []

        # 進行產業分組
        sector_groups = {}
        if self.sector_mapping:
            for ticker in tickers:
                sector = self.sector_mapping.get(ticker, "Unknown")
                sector_groups.setdefault(sector, []).append(ticker)
        else:
            sector_groups["All_Market"] = tickers

        skipped_unknown_count = 0
        for sector, sector_tickers in sector_groups.items():
            if sector == "Unknown":
                skipped_unknown_count = len(sector_tickers)
                continue

            if len(sector_tickers) < self.min_tickers_for_pairing:
                continue

            # 轉置標準化對數價格矩陣以計算距離
            norm_vals = self.normalized_df[sector_tickers].values.T

            # 計算 SSD (歐式距離平方)
            ssd_matrix = ssd.pdist(norm_vals, metric='sqeuclidean')

            # 計算協方差矩陣，用於 OLS 估算 beta
            cov_matrix = np.cov(norm_vals)
            var_diag = np.diag(cov_matrix)

            idx = 0
            for i in range(len(sector_tickers)):
                ticker_b = sector_tickers[i]
                x_val = norm_vals[i]
                var_x = var_diag[i]

                for j in range(i + 1, len(sector_tickers)):
                    ticker_a = sector_tickers[j]
                    y_val = norm_vals[j]

                    ssd_value = ssd_matrix[idx]
                    idx += 1

                    # 估計 OLS 回歸係數 beta (Hedge Ratio)
                    # Y = beta * X + alpha + residual -> beta = cov(X,Y) / var(X)
                    cov_xy = cov_matrix[i, j]
                    beta = cov_xy / var_x if var_x > 1e-8 else 0.0

                    # 計算價差價列
                    spread = y_val - beta * x_val
                    spread_mean = np.mean(spread)
                    spread_std = np.std(spread, ddof=1) if len(spread) > 1 else 0.0

                    ssd_records.append({
                        "Form_Start": self.form_start, "Form_End": self.form_end,
                        "Sector": sector, "Ticker_A": ticker_a, "Ticker_B": ticker_b,
                        "SSD": round(ssd_value, 6), "Hedge_Ratio": round(beta, 4),
                        "Spread_Mean": round(spread_mean, 6),
                        "Spread_Std": round(spread_std, 6)
                    })

        if skipped_unknown_count > 0:
            print(f"  [Formation] 跳過 {skipped_unknown_count} 支未分類 (Unknown) 股票。")

        if not ssd_records:
            return pd.DataFrame()

        return pd.DataFrame(ssd_records).sort_values("SSD").reset_index(drop=True)

    def select_pairs(self) -> pd.DataFrame:
        """
        挑選 SSD 最小的前 N 對配對，並記錄對數空間建倉所需的均值與標準差。

        回傳:
            pd.DataFrame: 選定的前 N 對配對及其參數。
        """
        ssd_df = self.compute_ssd()
        if ssd_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        selected = ssd_df.head(self.top_n).copy()
        selected["Rank"] = range(1, len(selected) + 1)

        # 記錄形成期對數價格的 mean 與 std，以便交易期做相同的正規化
        mean_a_list, std_a_list, mean_b_list, std_b_list = [], [], [], []
        for _, row in selected.iterrows():
            mean_a_list.append(self.mean_prices[row["Ticker_A"]])
            std_a_list.append(self.std_prices[row["Ticker_A"]])
            mean_b_list.append(self.mean_prices[row["Ticker_B"]])
            std_b_list.append(self.std_prices[row["Ticker_B"]])

        selected["Log_Mean_A"] = mean_a_list
        selected["Log_Std_A"] = std_a_list
        selected["Log_Mean_B"] = mean_b_list
        selected["Log_Std_B"] = std_b_list

        self.selected_pairs = selected
        return self.selected_pairs

    def run(self) -> pd.DataFrame:
        """
        執行完整的配對形成工作流。
        """
        self.normalize_prices()
        self.select_pairs()
        return self.selected_pairs


@dataclass(slots=True)
class PairState:
    """
    單一配對在交易期中的持倉狀態與損益追蹤容器。
    """
    position: int = 0
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    cooldown_dir: int = 0
    prev_total_pnl: float = 0.0


class Trading:
    """
    配對交易模擬器 (進階版)。
    依據動態對沖比例進行資金加權分配，並支援滾動式動態 OLS 回歸計算 Z-Score。
    """
    
    def __init__(self, price_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, selected_pairs: pd.DataFrame, capital_per_pair: float,
                 fee_rate: float, slippage_rate: float, stop_loss_pct: float, entry_z: float, exit_z: float, zscore_window: int, allow_reentry: bool = False,
                 zscore_clip: float = 10.0, min_spread_std: float = 1e-6, use_dynamic_stop: bool = False, dynamic_stop_z: float = 3.0,
                 portfolio_stop_loss_pct: float = 0.10, use_vol_adjust: bool = False):
        """
        初始化交易模擬配置。
        """
        self.price_df = price_df
        self.trade_dates = trade_dates
        self.selected_pairs = selected_pairs
        self.capital_per_pair = capital_per_pair
        
        self.friction_rate = fee_rate + slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.zscore_window = zscore_window
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust = use_vol_adjust

        self.period_pnl: float = 0.0

    def _execute_entry(self, state: PairState, z: float, p_a: float, p_b: float, hedge_ratio: float) -> tuple[bool, float]:
        """
        依據對沖比例 (Hedge Ratio / Beta)，以市值中性與資金加權比例開倉。
        配對資金分配公式：
            總權重 = 1.0 + |beta|
            標的 A 權重 = 1.0 / 總權重，配得資金 v_a = 總資金 * 標的 A 權重
            標的 B 權重 = |beta| / 總權重，配得資金 v_b = 總資金 * 標的 B 權重
        """
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)

        # 價差高估 -> 空 A 多 B (買入價值 v_b 的 B，賣空價值 v_a 的 A)
        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b = v_b / p_b
        # 價差低估 -> 多 A 空 B (買入價值 v_a 的 A，賣空價值 v_b 的 B)
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a = v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a = p_a
        state.entry_price_b = p_b
        # 計算交易手續費摩擦
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state: PairState, current_trade_pnl: float, stop_loss: bool = False):
        """
        平倉交易，更新損益並清空持倉狀態。
        """
        state.realized_pnl += current_trade_pnl

        if stop_loss:
            state.is_stopped = True if not self.allow_reentry else False
            if self.allow_reentry:
                state.cooldown_dir = state.position
        else:
            state.cooldown_dir = state.position

        state.position = 0
        state.shares_a = 0.0
        state.shares_b = 0.0
        state.entry_price_a = 0.0
        state.entry_price_b = 0.0
        state.trade_entry_fee = 0.0

    def _simulate_pair(self, period_start: str, period_end: str, sector: str, ticker_a: str, ticker_b: str, pair_rank: int, hedge_ratio: float,
                       form_spread_mean: float, form_spread_std: float, log_mean_a: float, log_std_a: float, log_mean_b: float, log_std_b: float) -> pd.DataFrame:
        """
        在交易期中，執行單一配對的逐日模擬。
        
        若 zscore_window > 0，則利用滾動 OLS 公式：
            roll_beta = roll_cov(X, Y) / roll_var(X)
            roll_alpha = mean(Y) - roll_beta * mean(X)
            residual_variance = roll_var(Y) - roll_beta * roll_cov(X, Y)
            roll_std = sqrt(residual_variance)
            zscore = (Y - roll_alpha - roll_beta * X) / roll_std
        """
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns: 
            return pd.DataFrame()

        price_a, price_b = self.price_df[ticker_a].dropna(), self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a, price_b = price_a.loc[common_idx], price_b.loc[common_idx]

        if len(price_a) < 5: 
            return pd.DataFrame()

        # 轉換為對數空間並採用形成期的 Mean/Std 做標準化
        log_p_a = np.log(price_a)
        log_p_b = np.log(price_b)

        norm_p_a = (log_p_a - log_mean_a) / log_std_a
        norm_p_b = (log_p_b - log_mean_b) / log_std_b

        # 依設定計算 Z-Score 與 Beta
        if self.zscore_window == 0:
            # 固定參數模式
            spread = norm_p_a - hedge_ratio * norm_p_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series = pd.Series(hedge_ratio, index=common_idx)
        else:
            # 滾動 OLS 回歸模式：動態估計滾動對沖比例與殘差波動
            roll_cov = norm_p_b.rolling(window=self.zscore_window).cov(norm_p_a)
            roll_var = norm_p_b.rolling(window=self.zscore_window).var()

            # 滾動對沖比例 beta = cov(X, Y) / var(X)
            roll_beta = np.where(roll_var > 1e-8, roll_cov / roll_var, 0.0)
            roll_beta = pd.Series(roll_beta, index=common_idx)

            roll_mean_a = norm_p_a.rolling(window=self.zscore_window).mean()
            roll_mean_b = norm_p_b.rolling(window=self.zscore_window).mean()
            # 滾動截距項 alpha = mean(Y) - beta * mean(X)
            roll_alpha = roll_mean_a - roll_beta * roll_mean_b

            # 價差價列 (OLS 殘差)
            spread = norm_p_a - roll_alpha - roll_beta * norm_p_b

            # 計算殘差波動度 (殘差方差 = var(Y) - beta * cov(X,Y))
            roll_var_a = norm_p_a.rolling(window=self.zscore_window).var()
            roll_res_var = roll_var_a - roll_beta * roll_cov
            roll_std = np.sqrt(np.maximum(roll_res_var, 0))

            if (roll_std < self.min_spread_std * 10).mean() > 0.5:
                return pd.DataFrame()

            safe_std = np.maximum(roll_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            # 標準化殘差即為動態 Z-Score
            zscore = np.clip(spread / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series = roll_beta

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0: 
            return pd.DataFrame()

        price_a = price_a.loc[valid_idx]
        price_b = price_b.loc[valid_idx]
        zscore = zscore.loc[valid_idx]
        beta_series = beta_series.loc[valid_idx]

        dates_arr = valid_idx
        zscore_arr = zscore.values
        pa_arr = price_a.values
        pb_arr = price_b.values
        beta_arr = beta_series.values

        base_log = {
            "Period_Start": period_start, "Period_End": period_end,
            "Sector": sector, "Pair_Rank": pair_rank,
            "Ticker_A": ticker_a, "Ticker_B": ticker_b,
            "Log_Mean_A": log_mean_a, "Log_Std_A": log_std_a,
            "Log_Mean_B": log_mean_b, "Log_Std_B": log_std_b
        }

        state = PairState()

        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_z, out_pos = [], [], []
        out_unrealized, out_realized, out_cum = [], [], []
        out_status, out_trade_pnl, out_days, out_delta = [], [], [], []

        # 逐日運行狀態機模擬
        for i in range(len(dates_arr)):
            date = dates_arr[i]
            z = 0.0 if np.isnan(zscore_arr[i]) else zscore_arr[i]
            p_a, p_b = pa_arr[i], pb_arr[i]
            # 當日對沖比例
            c_beta = beta_arr[i] if not np.isnan(beta_arr[i]) else hedge_ratio

            unrealized_pnl = 0.0
            closed_trade_pnl = 0.0
            daily_delta = 0.0
            current_status = "HOLD_CASH"

            if state.is_stopped:
                out_dates.append(date)
                out_pa.append(round(p_a, 4))
                out_pb.append(round(p_b, 4))
                out_hr.append(round(float(c_beta), 4))
                out_z.append(round(float(z), 4))
                out_pos.append(0)
                out_unrealized.append(0.0)
                out_realized.append(round(float(state.realized_pnl), 4))
                out_cum.append(round(float(state.realized_pnl), 4))
                out_status.append("STOPPED")
                out_trade_pnl.append(0.0)
                out_days.append(0)
                out_delta.append(0.0)
                continue

            if state.cooldown_dir == -1 and z <= self.exit_z:
                state.cooldown_dir = 0
            elif state.cooldown_dir == 1 and z >= -self.exit_z:
                state.cooldown_dir = 0

            # 持倉部位
            if state.position != 0:
                state.days_held += 1
                raw_unrealized = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee_est = (abs(state.shares_a)*p_a + abs(state.shares_b)*p_b) * self.friction_rate

                current_trade_pnl = raw_unrealized - state.trade_entry_fee - exit_fee_est

                # 檢查個股虧損比例停損 (Stop Loss)
                is_cap_stop = self.stop_loss_pct > 0 and (-current_trade_pnl / self.capital_per_pair) >= self.stop_loss_pct
                # 檢查 Z-Score 極端值動態停損
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, current_trade_pnl, stop_loss=True)
                    closed_trade_pnl = current_trade_pnl
                    current_status = "STOP_LOSS_TRIGGERED"
                else:
                    is_exit_short = (state.position == -1) and (z <= self.exit_z)
                    is_exit_long  = (state.position == 1)  and (z >= -self.exit_z)

                    if is_exit_short or is_exit_long:
                        self._execute_close(state, current_trade_pnl, stop_loss=False)
                        closed_trade_pnl = current_trade_pnl
                        current_status = "EXIT"
                    else:
                        unrealized_pnl = current_trade_pnl
                        current_status = "HOLDING"
            # 空倉尋找建倉機會
            else:
                if abs(z) > self.entry_z:
                    # 傳入當前的對沖比例進行加權建倉
                    entered, unrealized_pnl = self._execute_entry(state, z, p_a, p_b, c_beta)
                    if entered:
                        current_status = "ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A"
                    else:
                        current_status = "HOLD_CASH (COOLDOWN)"
                else:
                    current_status = "HOLD_CASH"

            cumulative_pnl = state.realized_pnl + unrealized_pnl
            daily_delta = cumulative_pnl - state.prev_total_pnl
            state.prev_total_pnl = cumulative_pnl

            out_dates.append(date)
            out_pa.append(round(p_a, 4))
            out_pb.append(round(p_b, 4))
            out_hr.append(round(float(c_beta), 4))
            out_z.append(round(float(z), 4))
            out_pos.append(state.position)
            out_unrealized.append(round(float(unrealized_pnl), 4))
            out_realized.append(round(float(state.realized_pnl), 4))
            out_cum.append(round(float(cumulative_pnl), 4))
            out_status.append(current_status)
            out_trade_pnl.append(round(float(closed_trade_pnl), 4))
            out_days.append(state.days_held)
            out_delta.append(round(float(daily_delta), 4))

            if current_status in ["STOP_LOSS_TRIGGERED", "EXIT"]:
                state.days_held = 0

            # 若已觸發永久停損，則快速填充剩餘日期
            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    rd = dates_arr[j]
                    r_z = 0.0 if np.isnan(zscore_arr[j]) else zscore_arr[j]
                    r_pa, r_pb = pa_arr[j], pb_arr[j]
                    r_beta = beta_arr[j] if not np.isnan(beta_arr[j]) else hedge_ratio

                    out_dates.append(rd)
                    out_pa.append(round(r_pa, 4))
                    out_pb.append(round(r_pb, 4))
                    out_hr.append(round(float(r_beta), 4))
                    out_z.append(round(float(r_z), 4))
                    out_pos.append(0)
                    out_unrealized.append(0.0)
                    out_realized.append(round(float(state.realized_pnl), 4))
                    out_cum.append(round(float(state.realized_pnl), 4))
                    out_status.append("STOPPED")
                    out_trade_pnl.append(0.0)
                    out_days.append(0)
                    out_delta.append(0.0)
                break

        # 交易期末強制平倉
        if state.position != 0 and out_status:
            last_status = out_status[-1]
            if last_status not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                pnl_before_last_day = out_cum[-2] if len(out_cum) > 1 else 0.0

                p_a_last, p_b_last = pa_arr[-1], pb_arr[-1]
                raw_unrealized_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                exit_fee = (abs(state.shares_a)*p_a_last + abs(state.shares_b)*p_b_last) * self.friction_rate

                closed_trade_pnl = raw_unrealized_final - state.trade_entry_fee - exit_fee
                state.realized_pnl += closed_trade_pnl
                daily_delta = state.realized_pnl - pnl_before_last_day

                out_status[-1] = "PERIOD_END_EXIT"
                out_realized[-1] = round(state.realized_pnl, 4)
                out_cum[-1] = round(state.realized_pnl, 4)
                out_unrealized[-1] = 0.0
                out_trade_pnl[-1] = round(closed_trade_pnl, 4)
                out_delta[-1] = round(daily_delta, 4)
                out_days[-1] = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unrealized, "Realized_PnL": out_realized,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_trade_pnl, "Days_Held": out_days, "Daily_Delta": out_delta
        })

        for k, v in base_log.items():
            df_out[k] = v

        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """
        平行模擬所有選定的配對，並依據設定判定是否觸發投資組合層級的最大累計停損。
        """
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start,
                period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"],
                ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A", 1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B", 1.0))
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # 投資組合層級停損處理機制 (Portfolio Stop Loss)
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()

            cutoff_date = None
            for date_val, pnl_val in daily_cum_pnl.items():
                if pnl_val / total_cap <= -self.portfolio_stop_loss_pct:
                    cutoff_date = date_val
                    break

            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date

                    df_before = df[before_mask]

                    df_at = df[at_mask].copy()
                    final_realized = 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        final_realized = row_at["Cumulative_PnL"]
                        if row_at["Position"] != 0:
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Trade_PnL"]
                        else:
                            if row_at["Status"] not in ("STOPPED", "STOP_LOSS_TRIGGERED", "EXIT"):
                                df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOPPED"
                            final_realized = row_at["Realized_PnL"]

                    df_after = df[after_mask].copy()
                    if not df_after.empty:
                        df_after.loc[df_after.index, "Position"] = 0
                        df_after.loc[df_after.index, "Unrealized_PnL"] = 0.0
                        df_after.loc[df_after.index, "Realized_PnL"] = final_realized
                        df_after.loc[df_after.index, "Cumulative_PnL"] = final_realized
                        df_after.loc[df_after.index, "Status"] = "STOPPED"
                        df_after.loc[df_after.index, "Trade_PnL"] = 0.0
                        df_after.loc[df_after.index, "Daily_Delta"] = 0.0

                    new_dfs.append(pd.concat([df_before, df_at, df_after], ignore_index=True))
                dfs = new_dfs

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily_delta = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily_delta.sum()) if not period_daily_delta.empty else 0.0

        return log_df, self.period_pnl


class DataProcessor:
    """
    價格資料載入與過濾前處理器。
    """
    
    def __init__(self, db_path: str, table_name: str = "daily_prices"):
        self.db_path, self.table_name = db_path, table_name

    def load_sector_mapping(self, info_table: str, ticker_col: str = "ticker", sector_col: str = "sector") -> dict:
        """
        從 SQLite 中載入產業分類表。
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT {ticker_col}, {sector_col} FROM {info_table}", conn)
            conn.close()
            mapping = {}
            for k, v in zip(df[ticker_col], df[sector_col]):
                if pd.notna(k) and pd.notna(v):
                    mapping[str(k).strip().upper()] = str(v).strip()
            print(f"✅ 成功載入產業分類表 '{info_table}'，共取得 {len(mapping)} 檔標的分類。")
            return mapping
        except Exception as e:
            print(f"⚠️ [警告] 無法載入產業分類表 '{info_table}'！錯誤原因：{e}")
            print(f"⚠️ 系統將退回「全市場(All_Market)」跨產業配對模式。")
            return {}

    def prepare_backtest_data(self, backtest_start: str, backtest_end: str, formation_window: int):
        """
        載入、過濾缺值率大於 20% 的股票，重塑 Pivot 價格表。
        """
        conn = sqlite3.connect(self.db_path)
        raw_df = pd.read_sql_query(f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC", conn)
        conn.close()

        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]

        price_pivot = raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()

        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20].ffill(limit=5)

        price_pivot.dropna(axis=1, thresh=int(len(price_pivot) * 0.9), inplace=True)

        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                if is_end and len(str(d_str).strip()) == 7:
                    return dt + pd.offsets.MonthEnd(0)
                return dt
            except Exception:
                return None

        bt_start_ts = _safe_parse(backtest_start)
        bt_end_ts = _safe_parse(backtest_end, is_end=True)
        all_dates = price_pivot.index.tolist()

        start_indices = [i for i, d in enumerate(all_dates) if d >= bt_start_ts] if bt_start_ts else []
        first_idx = start_indices[0] if start_indices else 0

        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end = bt_end_ts if bt_end_ts else price_pivot.index[-1]
        price_pivot = price_pivot.loc[data_slice_start:data_slice_end]

        sliced_dates = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_trade_idx = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_trade_idx, formation_window)


class RollingBacktester:
    """
    滾動回測引擎與網格搜索核心。
    實作 Walk-Forward validation 機制與併發回測 Slot 機制，採用 Append 串流 CSV 追加寫入。
    """
    
    def __init__(self, top_n_list: list, stop_loss_list: list, zscore_window_list: list,
                 entry_z: float, exit_z: float, formation_window: int, trading_window: int, rolling_step: int,
                 fee_rate: float, slippage_rate: float, initial_capital: float,
                 allow_reentry: bool, zscore_clip: float, min_spread_std: float,
                 min_tickers_for_pairing: int, output_dir: Path,
                 portfolio_stop_loss_pct_list: list = None,
                 max_sector_ratio_list: list = None,
                 dynamic_stop_z_list: list = None,
                 use_vol_adjust_list: list = None,
                 **kwargs):
        """
        初始化滾動回測參數網格。
        """
        self.top_n_list = top_n_list
        self.stop_loss_list = stop_loss_list
        self.zscore_window_list = zscore_window_list
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.rolling_step = rolling_step
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing
        self.output_dir = output_dir

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_csv_path(self, n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj) -> "Path":
        """
        產生回測結果之唯一的輸出 CSV 檔名與路徑。
        """
        sl_str  = f"SL{int(sl*100)}"         if sl       > 0 else "SL0"
        psl_str = f"PSL{int(p_stop*100)}"    if p_stop   > 0 else "PSL0"
        msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
        dsz_str = f"DSZ{int(dyn_z)}"         if dyn_z    > 0 else "DSZ0"
        vol_str = "VolAdj" if vol_adj else "NoVol"
        filename = f"TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
        return self.output_dir / filename

    def run(self, price_pivot: pd.DataFrame, all_dates: list, total_days: int, local_first_trade_idx: int, sector_mapping: dict):
        """
        執行滾動 Walk-Forward 與多參數併發回測。
        """
        max_concurrent = self.trading_window // self.rolling_step
        states = {}

        all_param_combos = list(itertools.product(
            self.top_n_list, self.stop_loss_list, self.zscore_window_list,
            self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list,
            self.dynamic_stop_z_list, self.use_vol_adjust_list
        ))
        
        for combo in all_param_combos:
            states[combo] = {
                "header_written": False,
                "row_count": 0,
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                          for _ in range(max_concurrent)]
            }

        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始進行 Grid Search，共 {len(roll_start_indices)} 期，每期處理 {len(states)} 種參數組合...")

        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx, form_end_idx = trade_start_idx - self.formation_window, trade_start_idx
            trade_end_idx = min(trade_start_idx + self.trading_window, total_days)

            form_data_raw = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data_raw = price_pivot.iloc[trade_start_idx:trade_end_idx]

            extended_trade_start_idx = max(0, trade_start_idx - max(self.zscore_window_list))
            extended_trade_data_raw = price_pivot.iloc[extended_trade_start_idx:trade_end_idx]

            valid_cols = (form_data_raw.isnull().sum() + extended_trade_data_raw.isnull().sum()) == 0

            form_data = form_data_raw.loc[:, valid_cols]
            trade_data = trade_data_raw.loc[:, valid_cols]
            trade_dates = trade_data.index
            extended_trade_data = extended_trade_data_raw.loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty: 
                continue

            trade_start_str, trade_end_str = str(all_dates[trade_start_idx])[:10], str(all_dates[trade_end_idx - 1])[:10]
            form_start_str, form_end_str = str(all_dates[form_start_idx])[:10], str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 處理中：第 {roll_idx+1:02d} 期 (交易: {trade_start_str} ~ {trade_end_str})")

            formation = Formation(
                price_df=form_data,
                form_start=form_start_str,
                form_end=form_end_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty: 
                continue

            for combo in all_param_combos:
                n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj = combo

                # 產業多元分散化篩選
                if sec_ratio > 0:
                    max_pairs_per_sector = max(1, int(n * sec_ratio))
                    sector_counts = {}
                    diversified_records = []
                    for _, row in max_selected_pairs.iterrows():
                        sec = row["Sector"]
                        if sec not in sector_counts:
                            sector_counts[sec] = 0
                        if sector_counts[sec] < max_pairs_per_sector:
                            diversified_records.append(row)
                            sector_counts[sec] += 1
                        if len(diversified_records) >= n:
                            break
                    selected_pairs = pd.DataFrame(diversified_records).copy()
                else:
                    selected_pairs = max_selected_pairs.head(n).copy()

                if selected_pairs.empty:
                    continue

                selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                state = states[combo]
                slots = state["slots"]

                # 動態分配獨立資金併發插槽
                free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                if free_slots:
                    slot_idx = free_slots[0]
                else:
                    slot_idx = min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                current_period_capital = slots[slot_idx]["capital"]
                current_capital_per_pair = current_period_capital / n

                trading = Trading(
                    price_df=extended_trade_data,
                    trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=current_capital_per_pair,
                    fee_rate=self.fee_rate,
                    slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl,
                    entry_z=self.entry_z,
                    exit_z=self.exit_z,
                    zscore_window=z_win,
                    allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip,
                    min_spread_std=self.min_spread_std,
                    use_dynamic_stop=(dyn_z > 0),
                    dynamic_stop_z=dyn_z,
                    portfolio_stop_loss_pct=p_stop,
                    use_vol_adjust=vol_adj
                )

                trade_log_df, period_pnl = trading.run(trade_start_str, trade_end_str)

                # 以追加 (a) 模式將明細寫入結果 CSV
                if not trade_log_df.empty:
                    filepath = self._get_csv_path(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)
                    write_header = not state["header_written"]
                    trade_log_df.to_csv(
                        filepath,
                        mode="w" if write_header else "a",
                        header=write_header,
                        index=False
                    )
                    state["header_written"] = True
                    state["row_count"] += len(trade_log_df)

                slots[slot_idx]["capital"] = max(0, current_period_capital + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        self._export_results(states, all_param_combos)

    def _export_results(self, states: dict, all_param_combos: list):
        """
        輸出網格搜索最終統計結果摘要。
        """
        print("\n✅ 回測完成！交易紀錄已串流寫入各 CSV 檔案。")
        for combo in all_param_combos:
            n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj = combo
            state = states[combo]
            if state["header_written"]:
                filepath = self._get_csv_path(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)
                print(f"  - 已輸出: {filepath.name} (共 {state['row_count']} 筆紀錄)")

        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")


def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    策略外部啟動入口 (定義一)。
    過濾 RollingBacktester 構造函數的有效參數並調用。
    """
    import inspect
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default

    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )

    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mappi
```

<!-- ==================== CODE CELL 7 ==================== -->

```python
ng)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")


def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    策略外部啟動入口 (定義二，保留重複定義行為)。
    過濾 RollingBacktester 構造函數的有效參數並調用。
    """
    import inspect
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default

    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )

    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")

```

<!-- ==================== CELL 8 ==================== -->

## 分群策略一：HDBSCAN + UMAP/PCA 密度分群策略

### 4.1 策略原理與數學特徵

傳統距離策略僅對個股價格進行比對，缺乏對個股內在統計特徵與非線性交互作用的捕捉。**本策略引入無監督機器學習，將特徵工程、維度壓縮與密度分群相結合**，建立起高效的選股過濾網絡。其核心步驟如下：

#### A. 13 維核心時序特徵工程 (Feature Engineering)
對形成期內每支股票的對數日收益率 $R_t = P_t - P_{t-1}$，擷取 13 個維度的核心統計與動量特徵：
1. **週動量 (5日收益率)**：$R_{5} = P_t - P_{t-5}$
2. **月動量 (21日收益率)**：$R_{21} = P_t - P_{t-21}$
3. **季動量 (63日收益率)**：$R_{63} = P_t - P_{t-63}$
4. **半年動量 (126日收益率)**：$R_{126} = P_t - P_{t-126}$
5. **短期波動度 (21日標準差)**：$\sigma_{21}$
6. **中期波動度 (63日標準差)**：$\sigma_{63}$
7. **長期總波動度 (全期標準差)**：$\sigma_{all}$
8. **日自相關 (Lag 1自相關)**：$\rho_1 = \text{Corr}(R_t, R_{t-1})$
9. **週自相關 (Lag 5自相關)**：$\rho_5 = \text{Corr}(R_t, R_{t-5})$
10. **月自相關 (Lag 21自相關)**：$\rho_{21} = \text{Corr}(R_t, R_{t-21})$
11. **收益偏度 (Skewness)**：衡量收益率分佈偏斜程度，反映肥尾極端收益概率。
12. **收益峰度 (Kurtosis)**：衡量收益率分佈尖銳程度與厚尾風險。
13. **Hurst 指數近似值**：透過 R/S 分析法進行雙對數線性擬合：
    $$\ln(R/S) = H \cdot \ln(n) + c$$
    若 $H < 0.5$ 代表序列具備顯著的均值回歸性；$H > 0.5$ 代表趨勢持續性；$H=0.5$ 代表隨機漫步。

所有個股特徵向量進行 Z-Score 標準化，得到特徵矩陣 $X$。

#### B. UMAP / PCA 維度壓縮 (Dimension Reduction)
- **流形降維 (UMAP)**：為克服高維特徵下的「維度災難」，運用 UMAP (Uniform Manifold Approximation and Projection) 非線性流形拓撲學算法，將 13 維特徵投影至 5 維嵌入空間，在保留全局集群結構的同時最大化保留局部鄰域關係。
- **主成分分析 (PCA)**：系統同時保留了傳統的線性 PCA 降維機制，作為穩健性對照。

#### C. HDBSCAN 空間密度聚類 (Density-Based Clustering)
對降維後的特徵空間執行 HDBSCAN 層級密度分群。不同於 K-Means 需要預設群落數並會強制將所有點歸類，HDBSCAN **自動識別高密度群落，並將分佈孤立、無顯著共性特徵的離群點標記為噪音 (Label = -1) 予以剔除**。

#### D. 群落內 EG 共整合與 ECM 半衰期雙重篩選
只在「**同產業 × 同聚類有效群落**」的雙重交集約束下進行兩兩股票配對。隨後對配對進行雙向 OLS 回歸與 ADF 單根檢定，並計算 ECM 均值收斂半衰期。僅保留 $p$-value 小於門檻且半衰期在 $[2, 60]$ 天之內的優質配對。這大幅提升了配對的統計顯著性，防止因大量隨機無意義配對引發的多重比較謬誤 (Data Snooping Bias)。

### 4.2 Python 核心程式碼實現
以下是 `strategies/HDBSCAN.py` 中特徵計算、降維、HDBSCAN 分群、 Engle-Granger 篩選與交易模擬的完整實現：

<!-- ==================== CODE CELL 9 ==================== -->

```python
# 嵌入 HDBSCAN.py 的完整實現
"""
HDBSCAN 分群配對交易滾動回測系統 (交易明細版)
=============================================

核心功能：本模組採用機器學習、流形降維與非監督式密度分群演算法，為配對交易提供先進的標的篩選機制：
1. 特徵工程 (Feature Engineering)：對每隻股票在形成期內之對數價格，擷取包含動量（5、21、63、126日收益率）、
   波動度（21、63日及全期標準差）、自相關性（Lag 1、5、21）、偏態、峰態，以及 Hurst 指數近似值（R/S 分析）等 13 維統計特徵向量。
2. 降維 (Dimensionality Reduction)：運用 UMAP 流形降維或 PCA 主成分分析，將特徵矩陣投影至低維度嵌入空間，克服高維距離計算的「維度災難」。
3. 密度分群 (Density-Based Clustering)：使用 HDBSCAN 演算法對降維後的空間進行密度分群，過濾走勢孤立的噪音點 (label = -1)，將走勢特徵相近的個股自動歸入相同群落。
4. 共整合檢定 (Engle-Granger Cointegration Test)：在「同群落 x 同產業」雙重約束下，進行兩兩標的之 OLS 回歸。
   對回歸殘差執行 ADF 檢定，過濾非共整合配對，並利用誤差修正模型 (ECM) 計算價差均值回歸的半衰期 (Half-life)，僅保留半衰期在合理區間 [2, 60] 日之配對。
5. 滾動回測與交易模擬：在交易期內，支援固定參數或每日動態滾動 OLS 回歸估算對沖比例，產生 Z-Score 進出場及停損訊號。
"""

import sqlite3
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 動態偵測並載入可用之 HDBSCAN 庫（ sklearn >= 1.3.0 或 獨立的 hdbscan 庫）
try:
    import hdbscan
    HDBSCAN_LIB = "hdbscan"
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        HDBSCAN_LIB = "sklearn"
    except ImportError:
        raise ImportError("請先安裝 scikit-learn >= 1.3.0 或 hdbscan：pip install scikit-learn hdbscan")

# 動態偵測 UMAP 降維庫是否可用
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ umap-learn 未安裝，將跳過 UMAP 降維，直接以原始特徵向量執行 HDBSCAN。")

from sklearn.preprocessing import StandardScaler

# 忽略不必要的警告資訊
warnings.filterwarnings("ignore")


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    執行快速一元線性最小二乘法 (OLS) 回歸： Y = alpha + beta * X + residual

    參數:
        y (np.ndarray): 因變量序列 (標的 A 的對數價格)。
        x (np.ndarray): 自變量序列 (標的 B 的對數價格)。

    回傳:
        tuple: (截距項 alpha, 斜率項 beta, 殘差序列 residual)
    """
    n = len(y)
    # 構造自變量矩陣（加入截距常數項）
    x_mat = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        # 若回歸矩陣奇異無法求解，則退回均值價差
        return 0.0, 0.0, y - np.mean(y)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    return alpha, beta, y - alpha - beta * x


def _adf_stat(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """
    對回歸殘差執行無截距、無時間趨勢項的簡化 ADF (Augmented Dickey-Fuller) 單根檢定，判斷價差平穩性。

    參數:
        resid (np.ndarray): OLS 回歸殘差序列。
        max_lags (int): ADF 檢定的滯後階數，預設為 1。

    回傳:
        tuple: (ADF 統計量 t-stat, p-value 值)
    """
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        # regression="n" 代表無常數項、無時間趨勢項的隨機漫步差分回歸
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0


def _extract_features(log_price: np.ndarray) -> np.ndarray:
    """
    從個股對數歷史價格中，計算 13 維核心特徵向量，反映個股之動量、波動率、自相關度與長期記憶性。

    參數:
        log_price (np.ndarray): 個股對數價格序列。

    回傳:
        np.ndarray: 長度為 13 的特徵向量。
    """
    ret = np.diff(log_price)
    n   = len(ret)

    # 1. 計算多個週期的累積收益率 (5、21、63、126 日，對應約週、月、季、半年)
    def safe_ret(window):
        if n >= window:
            return float(log_price[-1] - log_price[-window])
        return 0.0

    # 2. 計算多個週期的滾動歷史波動率 (21、63 日)
    def roll_vol(window):
        if n >= window:
            return float(np.std(ret[-window:], ddof=1))
        return float(np.std(ret, ddof=1)) if n > 1 else 0.0

    # 3. 計算特定滯後階數的收益率自相關係數 (Lag 1、5、21)
    def autocorr(lag):
        if n <= lag:
            return 0.0
        x1, x2 = ret[:-lag], ret[lag:]
        if len(x1) < 2:
            return 0.0
        try:
            return float(np.corrcoef(x1, x2)[0, 1])
        except Exception:
            return 0.0

    # 4. 估算 Hurst 指數 (長期記憶與均值回歸性指標)
    # 利用 R/S (Rescaled Range) 分析，在 log(n) 與 log(R/S) 上進行線性擬合的斜率
    def hurst_approx():
        if n < 20:
            return 0.5
        rs_list = []
        for seg_len in [n // 4, n // 2, n]:
            if seg_len < 4:
                continue
            seg = ret[:seg_len]
            mean_seg = np.mean(seg)
            deviate  = np.cumsum(seg - mean_seg)
            rs = (np.max(deviate) - np.min(deviate)) / (np.std(seg, ddof=1) + 1e-8)
            rs_list.append((np.log(seg_len), np.log(rs + 1e-8)))
        if len(rs_list) < 2:
            return 0.5
        xs, ys = zip(*rs_list)
        try:
            h = float(np.polyfit(xs, ys, 1)[0])
        except Exception:
            h = 0.5
        return np.clip(h, 0.0, 1.0)

    # 5. 全期基礎統計量：波動率、偏態、峰態
    vol_all  = float(np.std(ret, ddof=1)) if n > 1 else 0.0
    skew_all = float(pd.Series(ret).skew()) if n > 2 else 0.0
    kurt_all = float(pd.Series(ret).kurt()) if n > 3 else 0.0

    features = np.array([
        safe_ret(5),   safe_ret(21),  safe_ret(63),  safe_ret(126),
        roll_vol(21),  roll_vol(63),  vol_all,
        autocorr(1),   autocorr(5),   autocorr(21),
        skew_all,      kurt_all,      hurst_approx(),
    ], dtype=np.float64)

    # 清洗可能存在的無效數值
    features = np.where(np.isfinite(features), features, 0.0)
    return features


class Formation:
    """
    配對形成期機器學習分群篩選器。
    執行：特徵矩陣構建 -> Z-Score標準化 -> 降維 (UMAP/PCA) -> HDBSCAN 聚類 -> 群內 EG 共整合檢定與半衰期過濾。
    """
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        hdbscan_min_cluster_size: int = 3,
        hdbscan_min_samples: int = 1,
        hdbscan_metric: str = "euclidean",
        reduce_method: str = "umap",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.05,
        max_sector_ratio: float = 0.3,
    ):
        """
        初始化機器學習形成期參數配置。
        """
        self.price_df = price_df
        self.max_sector_ratio = max_sector_ratio
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        # HDBSCAN 聚類參數
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        # 降維方法參數
        self.reduce_method = reduce_method.lower()
        if self.reduce_method == "umap" and not UMAP_AVAILABLE:
            raise RuntimeError("umap-learn 未安裝，請執行：pip install umap-learn")
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors  = umap_n_neighbors
        self.umap_min_dist     = umap_min_dist
        self.umap_random_state = umap_random_state

        # Engle-Granger 單根檢定參數
        self.adf_max_lags           = adf_max_lags
        self.adf_pvalue_threshold   = adf_pvalue_threshold

        self.selected_pairs: pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict = {}

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        """
        對所有合格標的，計算 13 維特徵，並執行 Z-Score 標準化，消除量綱影響。

        回傳:
            tuple: (標準化特徵矩陣 X [n_samples, 13], 股票代碼列表 tickers)
        """
        log_prices = np.log(self.price_df)
        tickers    = log_prices.columns.tolist()

        feat_rows, valid_tickers = [], []
        for ticker in tickers:
            series = log_prices[ticker].values
            # 排除停牌時間長或資料不齊的股票
            if len(series) < 30 or not np.all(np.isfinite(series)):
                continue
            feat_rows.append(_extract_features(series))
            valid_tickers.append(ticker)

        if not feat_rows:
            return np.empty((0, 0)), []

        X = np.vstack(feat_rows)
        # 進行標準差與均值標準化
        X = StandardScaler().fit_transform(X)
        return X, valid_tickers

    def _umap_reduce(self, X: np.ndarray) -> np.ndarray:
        """
        利用 UMAP (Uniform Manifold Approximation and Projection) 將高維特徵矩陣進行非線性流形拓撲降維。
        """
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        n_neigh  = min(self.umap_n_neighbors,  n_stocks - 1)
        if n_comp < 1 or n_neigh < 1:
            return X
        reducer = umap.UMAP(
            n_components  = n_comp,
            n_neighbors   = n_neigh,
            min_dist      = self.umap_min_dist,
            random_state  = self.umap_random_state,
            low_memory    = True,
        )
        return reducer.fit_transform(X)

    def _pca_reduce(self, X: np.ndarray) -> np.ndarray:
        """
        利用 PCA (主成分分析) 進行線性降維，保留最大方差方向。
        """
        from sklearn.decomposition import PCA
        n_stocks = X.shape[0]
        n_comp   = min(self.umap_n_components, n_stocks - 1)
        if n_comp < 1:
            return X
        pca = PCA(n_components=n_comp, random_state=self.umap_random_state)
        return pca.fit_transform(X)

    def _hdbscan_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        對降維後的嵌入特徵空間執行 HDBSCAN 空間層級密度分群。
        """
        min_cs = min(self.hdbscan_min_cluster_size, max(2, X.shape[0] // 5))
        if HDBSCAN_LIB == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size = min_cs,
                min_samples      = self.hdbscan_min_samples,
                metric           = self.hdbscan_metric,
                core_dist_n_jobs = -1,
            )
        else:
            clusterer = sklearn_HDBSCAN(
                min_cluster_size = min_cs,
                min_samples      = self.hdbscan_min_samples,
                metric           = self.hdbscan_metric,
                n_jobs           = -1,
            )
        clusterer.fit(X)
        return clusterer.labels_

    def _cointegration_within_clusters(self, tickers: list[str], labels: np.ndarray) -> pd.DataFrame:
        """
        在相同的「產業分群 (Sector) x 密度群落 (HDBSCAN Cluster)」內進行兩兩配對檢定，
        運用 Engle-Granger 雙向 OLS 尋找共整合配對，並根據誤差修正模型判定均值回歸的半衰期門檻。

        參數:
            tickers (list): 合格股票代碼。
            labels (np.ndarray): HDBSCAN 分群標籤（-1 表示密度不足的雜訊點）。

        回傳:
            pd.DataFrame: 通過共整合檢定並按 ADF 統計量升序排序的配對集。
        """
        log_prices    = np.log(self.price_df[tickers])
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # 過濾噪音標籤 -1，只關注有效凝聚的群落
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("  [Formation] HDBSCAN 未找到任何有效群落（全為噪音點）。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(f"  [Formation] HDBSCAN 分群結果：{len(unique_labels)} 個群落，"
              f"{noise_count} 個噪音點排除，"
              f"ADF p 值門檻 = {self.adf_pvalue_threshold:.2f} "
              f"({'保守 1%' if self.adf_pvalue_threshold <= 0.01 else '積極 5%'})")

        # 結合產業別與聚類群落 label
        ticker_meta: dict[str, tuple[str, int]] = {}
        for t, lbl in zip(tickers, labels):
            sector = self.sector_mapping.get(t.upper(), "Unknown")
            ticker_meta[t] = (sector, int(lbl))

        # 映射 (產業, 群落id) -> 股票代碼清單
        group_map: dict[tuple[str, int], list[str]] = {}
        for t, (sec, lbl) in ticker_meta.items():
            if sec == "Unknown" or lbl == -1:
                continue
            group_map.setdefault((sec, lbl), []).append(t)

        # 篩選出滿足配對門檻的有效子群
        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}

        if not valid_groups:
            print("  [Formation] 同產業 × 同群落後無有效配對組合。")
            return pd.DataFrame()

        total_group_count = len(valid_groups)
        print(f"  [Formation] 有效 (產業, 群落) 組合：{total_group_count} 組")

        eg_records = []
        passed_count = 0
        rejected_count = 0

        # 在子群內部兩兩進行 Engle-Granger Cointegration 檢定
        for (sector, cluster_lbl), group_tickers in sorted(valid_groups.items()):
            for i, ta in enumerate(group_tickers):
                log_a = log_prices[ta].values
                for j in range(i + 1, len(group_tickers)):
                    tb    = group_tickers[j]
                    log_b = log_prices[tb].values

                    # 方向一：A 對 B 進行線性回歸，取得殘差檢定單根
                    al_ab, be_ab, re_ab = _ols(log_a, log_b)
                    stat_ab, pval_ab = _adf_stat(re_ab, self.adf_max_lags)

                    # 方向二：B 對 A 進行線性回歸
                    al_ba, be_ba, re_ba = _ols(log_b, log_a)
                    stat_ba, pval_ba = _adf_stat(re_ba, self.adf_max_lags)

                    # 選擇 p-value 較顯著（即平穩均值回歸概率最高）的回歸方向
                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ta, tb
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = tb, ta

                    # 超過 p-value 顯著性閾值（如 5%），剔除該配對
                    if best_pval >= self.adf_pvalue_threshold:
                        rejected_count += 1
                        continue

                    # 計算均值回歸的半衰期 (Half-Life)
                    # 誤差修正模型形式: d(Resid_t) = lambda * Resid_{t-1} + e_t
                    # 回歸係數 lambda 代表回歸速度，須為負值才具收斂性
                    dy = np.diff(best_resid)
                    y_lag = best_resid[:-1]
                    n_dy = len(dy)
                    x_mat = np.column_stack([np.ones(n_dy), y_lag])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(x_mat, dy, rcond=None)
                        lambda_val = coeffs[1]
                    except Exception:
                        lambda_val = 0.0

                    if lambda_val >= 0.0:
                        rejected_count += 1
                        continue

                    # 半衰期公式: Half-Life = -ln(2) / lambda
                    halflife = -np.log(2) / lambda_val
                    # 限制回歸速度：回歸過快（<2日，可能存在高頻微噪）或過慢（>60日，資金占用長）皆予以排除
                    if halflife < 2.0 or halflife > 60.0:
                        rejected_count += 1
                        continue

                    passed_count += 1
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    eg_records.append({
                        "Form_Start":    self.form_start,
                        "Form_End":      self.form_end,
                        "Sector":        sector,
                        "Cluster_Label": cluster_lbl,
                        "Ticker_A":      best_a,
                        "Ticker_B":      best_b,
                        "ADF_Stat":      round(best_stat,   6),
                        "ADF_PValue":    round(best_pval,   6),
                        "Hedge_Ratio":   round(best_beta,   6),
                        "OLS_Alpha":     round(best_alpha,  6),
                        "Spread_Mean":   round(spread_mean, 6),
                        "Spread_Std":    round(spread_std,  6),
                    })

        print(f"  [Formation] EG 檢定：{passed_count} 對通過 p < {self.adf_pvalue_threshold}，"
              f"{rejected_count} 對被 p 值門檻/半衰期排除。")

        if not eg_records:
            return pd.DataFrame()

        # ADF 統計量越小（負得越多），說明平穩性與均值回歸越強烈，故依此排序
        return pd.DataFrame(eg_records).sort_values("ADF_Stat").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """
        執行完整的 HDBSCAN-EG 聚類與共整合配對選定流程。
        """
        X, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 進行降維
        if self.reduce_method == "pca":
            X_embed = self._pca_reduce(X)
        else:
            X_embed = self._umap_reduce(X)

        # 進行聚類
        labels = self._hdbscan_cluster(X_embed)
        self.cluster_labels_ = dict(zip(valid_tickers, labels.tolist()))

        # 群內共整合檢定
        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 執行產業別數量上限分散化 (Max Sector Ratio)
        if getattr(self, "max_sector_ratio", 0) > 0:
            max_pairs_per_sector = max(1, int(self.top_n * self.max_sector_ratio))
            sector_counts = {}
            diversified_records = []
            for _, row in eg_df.iterrows():
                sec = row["Sector"]
                if sec not in sector_counts:
                    sector_counts[sec] = 0
                if sector_counts[sec] < max_pairs_per_sector:
                    diversified_records.append(row)
                    sector_counts[sec] += 1
                if len(diversified_records) >= self.top_n:
                    break
            selected = pd.DataFrame(diversified_records).copy()
        else:
            selected = eg_df.head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        # 映射對數空間基礎正規化參數
        log_prices  = np.log(self.price_df)
        mean_prices = log_prices.mean()
        std_prices  = log_prices.std()

        selected["Log_Mean_A"] = selected["Ticker_A"].map(mean_prices)
        selected["Log_Std_A"]  = selected["Ticker_A"].map(std_prices)
        selected["Log_Mean_B"] = selected["Ticker_B"].map(mean_prices)
        selected["Log_Std_B"]  = selected["Ticker_B"].map(std_prices)

        self.selected_pairs = selected
        return self.selected_pairs


@dataclass(slots=True)
class PairState:
    """
    持倉狀態與交易明細統計。
    """
    position: int = 0
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    cooldown_dir: int = 0
    prev_total_pnl: float = 0.0


class Trading:
    """
    Engle-Granger OLS 價差配對交易模擬器。
    計算每日 Z-Score，支援滾動視窗的動態多期 OLS 重新回歸擬合。
    """
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        trade_dates: pd.DatetimeIndex,
        selected_pairs: pd.DataFrame,
        capital_per_pair: float,
        fee_rate: float,
        slippage_rate: float,
        stop_loss_pct: float,
        entry_z: float,
        exit_z: float,
        zscore_window: int,
        allow_reentry: bool = False,
        zscore_clip: float = 10.0,
        min_spread_std: float = 1e-6,
        use_dynamic_stop: bool = False,
        dynamic_stop_z: float = 3.0,
        portfolio_stop_loss_pct: float = 0.10,
        use_vol_adjust: bool = False,
    ):
        """
        初始化交易配置。
        """
        self.price_df        = price_df
        self.trade_dates     = trade_dates
        self.selected_pairs  = selected_pairs
        self.capital_per_pair = capital_per_pair
        self.friction_rate   = fee_rate + slippage_rate
        self.stop_loss_pct   = stop_loss_pct
        self.entry_z         = entry_z
        self.exit_z          = exit_z
        self.zscore_window   = zscore_window
        self.allow_reentry   = allow_reentry
        self.zscore_clip     = zscore_clip
        self.min_spread_std  = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z  = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust  = use_vol_adjust
        self.period_pnl: float = 0.0

    def _execute_entry(self, state, z, p_a, p_b, hedge_ratio):
        """
        根據對沖比例進行資金比重建倉。
        """
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)

        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b =  v_b / p_b
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a =  v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a  = p_a
        state.entry_price_b  = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state, current_trade_pnl, stop_loss=False):
        """
        平倉。
        """
        state.realized_pnl += current_trade_pnl
        if stop_loss:
            state.is_stopped = True if not self.allow_reentry else False
            if self.allow_reentry:
                state.cooldown_dir = state.position
        else:
            state.cooldown_dir = state.position
        state.position = 0

    def _simulate_pair(
        self, period_start, period_end, sector, ticker_a, ticker_b, pair_rank,
        hedge_ratio, ols_alpha, form_spread_mean, form_spread_std,
        log_mean_a, log_std_a, log_mean_b, log_std_b,
        cluster_label, cluster_group,
    ) -> pd.DataFrame:
        """
        對單一對數價格共整合配對進行逐日模擬。
        支援每日重新做滾動線性回歸 (Rolling OLS) 擬合截距項與對沖比例。
        """
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns:
            return pd.DataFrame()

        price_a = self.price_df[ticker_a].dropna()
        price_b = self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a    = price_a.loc[common_idx]
        price_b    = price_b.loc[common_idx]

        if len(price_a) < 5:
            return pd.DataFrame()

        log_a = np.log(price_a)
        log_b = np.log(price_b)

        # 根據是否啟用滾動視窗計算 Z-Score
        if self.zscore_window == 0:
            # 固定參數模式：使用形成期估算之對沖比例與截距
            spread   = log_a - ols_alpha - hedge_ratio * log_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore   = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = pd.Series(hedge_ratio, index=common_idx)
            alpha_series = pd.Series(ols_alpha,   index=common_idx)
        else:
            # 滾動線性回歸模式：每日以大小為 w 的視窗重新估算 OLS 參數
            w = self.zscore_window
            n = len(log_a)
            la_vals, lb_vals = log_a.values, log_b.values
            roll_alpha = np.full(n, np.nan)
            roll_beta  = np.full(n, np.nan)
            roll_mean  = np.full(n, np.nan)
            roll_std   = np.full(n, np.nan)

            for k in range(w - 1, n):
                ya = la_vals[k - w + 1: k + 1]
                xb = lb_vals[k - w + 1: k + 1]
                a_, b_, r_ = _ols(ya, xb)
                roll_alpha[k] = a_
                roll_beta[k]  = b_
                roll_mean[k]  = float(np.mean(r_))
                roll_std[k]   = float(np.std(r_, ddof=1)) if len(r_) > 1 else 0.0

            roll_alpha_s = pd.Series(roll_alpha, index=common_idx)
            roll_beta_s  = pd.Series(roll_beta,  index=common_idx)
            roll_mean_s  = pd.Series(roll_mean,  index=common_idx)
            roll_std_s   = pd.Series(roll_std,   index=common_idx)

            spread     = log_a - roll_alpha_s - roll_beta_s * log_b
            safe_std_s = np.maximum(roll_std_s, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std_s * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std_s
            zscore     = np.clip((spread - roll_mean_s) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = roll_beta_s
            alpha_series = roll_alpha_s

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()

        price_a      = price_a.loc[valid_idx]
        price_b      = price_b.loc[valid_idx]
        zscore       = zscore.loc[valid_idx]
        beta_series  = beta_series.loc[valid_idx]
        alpha_series = alpha_series.loc[valid_idx]

        dates_arr  = valid_idx
        zscore_arr = zscore.values
        pa_arr     = price_a.values
        pb_arr     = price_b.values
        beta_arr   = beta_series.values
        alpha_arr  = alpha_series.values

        base_log = {
            "Period_Start":   period_start,   "Period_End":     period_end,
            "Sector":         sector,          "Cluster_Label":  cluster_label,
            "Pair_Rank":      pair_rank,
            "Ticker_A":       ticker_a,        "Ticker_B":       ticker_b,
            "Log_Mean_A":     log_mean_a,      "Log_Std_A":      log_std_a,
            "Log_Mean_B":     log_mean_b,      "Log_Std_B":      log_std_b,
        }

        state = PairState()
        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_ols_alpha, out_z, out_pos = [], [], [], []
        out_unr, out_rea, out_cum = [], [], []
        out_status, out_tpnl, out_days, out_delta = [], [], [], []

        def _append_row(date, p_a, p_b, c_beta, c_alpha, z_val, pos,
                        unr, rea, cum, status, tpnl, days, delta):
            out_dates.append(date);      out_pa.append(round(p_a, 4));     out_pb.append(round(p_b, 4))
            out_hr.append(round(c_beta, 4)); out_ols_alpha.append(round(c_alpha, 6))
            out_z.append(round(z_val, 4));   out_pos.append(pos)
            out_unr.append(round(unr, 4));   out_rea.append(round(rea, 4)); out_cum.append(round(cum, 4))
            out_status.append(status);   out_tpnl.append(round(tpnl, 4))
            out_days.append(days);        out_delta.append(round(delta, 4))

        # 每日持倉訊號模擬
        for i in range(len(dates_arr)):
            date    = dates_arr[i]
            z       = 0.0 if np.isnan(zscore_arr[i]) else float(zscore_arr[i])
            p_a, p_b = float(pa_arr[i]), float(pb_arr[i])
            c_beta   = float(beta_arr[i])  if not np.isnan(beta_arr[i])  else hedge_ratio
            c_alpha  = float(alpha_arr[i]) if not np.isnan(alpha_arr[i]) else ols_alpha

            unr, tpnl, status = 0.0, 0.0, "HOLD_CASH"

            if state.is_stopped:
                _append_row(date, p_a, p_b, c_beta, c_alpha, z, 0,
                            0.0, state.realized_pnl, state.realized_pnl,
                            "STOPPED", 0.0, 0, 0.0)
                continue

            if   state.cooldown_dir == -1 and z <= self.exit_z:  state.cooldown_dir = 0
            elif state.cooldown_dir ==  1 and z >= -self.exit_z: state.cooldown_dir = 0

            # 持倉期
            if state.position != 0:
                state.days_held += 1
                raw_unr  = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                cur_tpnl = raw_unr - state.trade_entry_fee - exit_fee

                # 停損檢定
                is_cap_stop = self.stop_loss_pct > 0 and (-cur_tpnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, cur_tpnl, stop_loss=True)
                    tpnl, status = cur_tpnl, "STOP_LOSS_TRIGGERED"
                elif (state.position == -1 and z <= self.exit_z) or (state.position == 1 and z >= -self.exit_z):
                    self._execute_close(state, cur_tpnl, stop_loss=False)
                    tpnl, status = cur_tpnl, "EXIT"
                else:
                    unr    = raw_unr - state.trade_entry_fee
                    status = "HOLDING"
            # 空倉期
            else:
                if abs(z) > self.entry_z:
                    entered, unr = self._execute_entry(state, z, p_a, p_b, c_beta)
                    status = ("ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A") if entered else "HOLD_CASH (COOLDOWN)"
                else:
                    status = "HOLD_CASH"

            cum   = state.realized_pnl + unr
            delta = cum - state.prev_total_pnl
            state.prev_total_pnl = cum

            _append_row(date, p_a, p_b, c_beta, c_alpha, z, state.position,
                        unr, state.realized_pnl, cum, status, tpnl, state.days_held, delta)

            if status in ("STOP_LOSS_TRIGGERED", "EXIT"):
                state.days_held = 0

            # 若觸發永久停損，則提前填充後續交易日
            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    _append_row(
                        dates_arr[j], float(pa_arr[j]), float(pb_arr[j]),
                        float(beta_arr[j]) if not np.isnan(beta_arr[j]) else hedge_ratio,
                        float(alpha_arr[j]) if not np.isnan(alpha_arr[j]) else ols_alpha,
                        0.0 if np.isnan(zscore_arr[j]) else float(zscore_arr[j]),
                        0, 0.0, state.realized_pnl, state.realized_pnl,
                        "STOPPED", 0.0, 0, 0.0
                    )
                break

        # 交易期結束強制平倉
        if state.position != 0 and out_status:
            if out_status[-1] not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                p_a_last, p_b_last = float(pa_arr[-1]), float(pb_arr[-1])
                raw_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                fee_final = (abs(state.shares_a) * p_a_last + abs(state.shares_b) * p_b_last) * self.friction_rate
                final_tpnl = raw_final - state.trade_entry_fee - fee_final
                state.realized_pnl += final_tpnl
                pnl_prev = out_cum[-2] if len(out_cum) > 1 else 0.0

                out_status[-1]     = "PERIOD_END_EXIT"
                out_rea[-1]        = round(state.realized_pnl, 4)
                out_cum[-1]        = round(state.realized_pnl, 4)
                out_unr[-1]        = 0.0
                out_tpnl[-1]       = round(final_tpnl, 4)
                out_delta[-1]      = round(state.realized_pnl - pnl_prev, 4)
                out_days[-1]       = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "OLS_Alpha": out_ols_alpha,
            "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unr, "Realized_PnL": out_rea,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_tpnl, "Days_Held": out_days, "Daily_Delta": out_delta,
        })
        for k, v in base_log.items():
            df_out[k] = v
        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """
        平行模擬所有選定的配對，並依據設定判定是否觸發投資組合層級最大累計停損。
        """
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start, period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"], ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                ols_alpha=float(row.get("OLS_Alpha", 0.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A",  1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B",  1.0)),
                cluster_label=int(row.get("Cluster_Label", -1)),
                cluster_group=str(row.get("Sector", "Unknown")),
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # 投資組合層級停損處理機制 (Portfolio Stop Loss)
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()

            cutoff_date = None
            for date_val, pnl_val in daily_cum_pnl.items():
                if pnl_val / total_cap <= -self.portfolio_stop_loss_pct:
                    cutoff_date = date_val
                    break

            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date

                    df_before = df[before_mask]

                    df_at = df[at_mask].copy()
                    final_realized = 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        final_realized = row_at["Cumulative_PnL"]
                        if row_at["Position"] != 0:
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Trade_PnL"]
                        else:
                            if row_at["Status"] not in ("STOPPED", "STOP_LOSS_TRIGGERED", "EXIT"):
                                df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOPPED"
                            final_realized = row_at["Realized_PnL"]

                    df_after = df[after_mask].copy()
                    if not df_after.empty:
                        df_after.loc[df_after.index, "Position"] = 0
                        df_after.loc[df_after.index, "Unrealized_PnL"] = 0.0
                        df_after.loc[df_after.index, "Realized_PnL"] = final_realized
                        df_after.loc[df_after.index, "Cumulative_PnL"] = final_realized
                        df_after.loc[df_after.index, "Status"] = "STOPPED"
                        df_after.loc[df_after.index, "Trade_PnL"] = 0.0
                        df_after.loc[df_after.index, "Daily_Delta"] = 0.0

                    new_dfs.append(pd.concat([df_before, df_at, df_after], ignore_index=True))
                dfs = new_dfs

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily.sum()) if not period_daily.empty else 0.0
        return log_df, self.period_pnl


class DataProcessor:
    """
    價格資料載入與Pivot重塑前處理器。
    """
    
    def __init__(self, db_path: str, table_name: str = "daily_prices"):
        self.db_path, self.table_name = db_path, table_name

    def load_sector_mapping(self, info_table, ticker_col="ticker", sector_col="sector") -> dict:
        """
        載入產業對應清單。
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df   = pd.read_sql_query(f"SELECT {ticker_col}, {sector_col} FROM {info_table}", conn)
            conn.close()
            mapping = {
                str(k).strip().upper(): str(v).strip()
                for k, v in zip(df[ticker_col], df[sector_col])
                if pd.notna(k) and pd.notna(v)
            }
            print(f"✅ 成功載入產業分類表 '{info_table}'，共取得 {len(mapping)} 檔標的分類。")
            return mapping
        except Exception as e:
            print(f"⚠️ 無法載入產業分類表：{e}，退回全市場模式。")
            return {}

    def prepare_backtest_data(self, backtest_start, backtest_end, formation_window):
        """
        讀取 SQLite、重塑 Pivot、 forward fill 並剔除大量空值個股，為回測做準備。
        """
        conn   = sqlite3.connect(self.db_path)
        raw_df = pd.read_sql_query(
            f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price "
            f"FROM {self.table_name} WHERE COALESCE(Adj_Close, Close) IS NOT NULL ORDER BY Date ASC", conn
        )
        conn.close()

        raw_df["date"]  = pd.to_datetime(raw_df["date"])
        raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")
        raw_df.dropna(subset=["price"], inplace=True)
        raw_df = raw_df[raw_df["price"] > 0]

        price_pivot = (
            raw_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
            .sort_index()
        )
        price_pivot = price_pivot.loc[:, price_pivot.isnull().mean() < 0.20].ffill(limit=5)
        price_pivot.dropna(axis=1, thresh=int(len(price_pivot) * 0.9), inplace=True)

        def _safe_parse(d_str, is_end=False):
            if not d_str: return None
            try:
                dt = pd.to_datetime(str(d_str).strip())
                if is_end and len(str(d_str).strip()) == 7:
                    return dt + pd.offsets.MonthEnd(0)
                return dt
            except Exception:
                return None

        bt_start_ts   = _safe_parse(backtest_start)
        bt_end_ts     = _safe_parse(backtest_end, is_end=True)
        all_dates     = price_pivot.index.tolist()
        start_indices = [i for i, d in enumerate(all_dates) if d >= bt_start_ts] if bt_start_ts else []
        first_idx     = start_indices[0] if start_indices else 0

        data_slice_start = all_dates[max(0, first_idx - formation_window)] if bt_start_ts else price_pivot.index[0]
        data_slice_end   = bt_end_ts if bt_end_ts else price_pivot.index[-1]
        price_pivot      = price_pivot.loc[data_slice_start:data_slice_end]

        sliced_dates      = price_pivot.index.tolist()
        new_start_indices = [i for i, d in enumerate(sliced_dates) if d >= bt_start_ts] if bt_start_ts else []
        local_first_idx   = new_start_indices[0] if new_start_indices else formation_window

        return price_pivot, sliced_dates, len(price_pivot), max(local_first_idx, formation_window)


class RollingBacktester:
    """
    HDBSCAN 密度聚類與滾動共整合回測引擎。
    執行 walk-forward grid search，追蹤插槽資金複利狀況，並在回測完成後統一寫入。
    """
    
    def __init__(
        self,
        top_n_list: list,
        stop_loss_list: list,
        zscore_window_list: list,
        entry_z: float,
        exit_z: float,
        formation_window: int,
        trading_window: int,
        rolling_step: int,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        allow_reentry: bool,
        zscore_clip: float,
        min_spread_std: float,
        min_tickers_for_pairing: int,
        hdbscan_min_cluster_size: int,
        hdbscan_min_samples: int,
        hdbscan_metric: str,
        umap_n_components: int,
        umap_n_neighbors: int,
        umap_min_dist: float,
        umap_random_state: int,
        adf_max_lags: int,
        adf_pvalue_threshold: float,
        output_dir: Path,
        reduce_method: str = "umap",
        portfolio_stop_loss_pct_list: list = None,
```

<!-- ==================== CODE CELL 10 ==================== -->

```python

        max_sector_ratio_list: list = None,
        dynamic_stop_z_list: list = None,
        use_vol_adjust_list: list = None,
        **kwargs
    ):
        """
        初始化 HDBSCAN 回測引擎參數。
        """
        self.top_n_list = top_n_list
        self.stop_loss_list = stop_loss_list
        self.zscore_window_list = zscore_window_list
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.rolling_step = rolling_step
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing

        # 密度分群與降維特徵
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.adf_max_lags = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.output_dir = output_dir
        self.reduce_method = reduce_method

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

        for k, v in kwargs.items():
            setattr(self, k, v)

    def run(self, price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping):
        """
        對所有網格組合進行滾動回測模擬。
        """
        max_concurrent = self.trading_window // self.rolling_step
        states = {}
        for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
            self.top_n_list, self.stop_loss_list, self.zscore_window_list,
            self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
        ):
            states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)] = {
                "logs":  [],
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                          for _ in range(max_concurrent)],
            }

        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始 HDBSCAN Grid Search，共 {len(roll_start_indices)} 期，每期 {len(states)} 種參數組合...")

        # 滾動回測迴圈
        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx = trade_start_idx - self.formation_window
            form_end_idx   = trade_start_idx
            trade_end_idx  = min(trade_start_idx + self.trading_window, total_days)

            form_data   = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data  = price_pivot.iloc[trade_start_idx:trade_end_idx]
            valid_cols  = (form_data.isnull().sum() + trade_data.isnull().sum()) == 0
            form_data   = form_data.loc[:, valid_cols]
            trade_dates = trade_data.index

            # 為交易期滾動 OLS 提供初期延伸緩衝數據
            extended_start = max(0, trade_start_idx - max(self.zscore_window_list))
            extended_data  = price_pivot.iloc[extended_start:trade_end_idx].loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty:
                continue

            ts_str = str(all_dates[trade_start_idx])[:10]
            te_str = str(all_dates[trade_end_idx - 1])[:10]
            fs_str = str(all_dates[form_start_idx])[:10]
            fe_str = str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 第 {roll_idx+1:02d} 期 (交易: {ts_str} ~ {te_str})")

            # 建立並執行分群與共整合選配
            formation = Formation(
                price_df=form_data,
                form_start=fs_str, form_end=fe_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing,
                hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,
                hdbscan_min_samples=self.hdbscan_min_samples,
                hdbscan_metric=self.hdbscan_metric,
                umap_n_components=self.umap_n_components,
                umap_n_neighbors=self.umap_n_neighbors,
                umap_min_dist=self.umap_min_dist,
                umap_random_state=self.umap_random_state,
                adf_max_lags=self.adf_max_lags,
                adf_pvalue_threshold=self.adf_pvalue_threshold,
                reduce_method=getattr(self, "reduce_method", "umap"),
                max_sector_ratio=0, # 先保留完整群落後續再依網格參數分散
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty:
                continue

            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                # 產業比例分散篩選
                if sec_ratio > 0:
                    max_pairs_per_sector = max(1, int(n * sec_ratio))
                    sector_counts = {}
                    diversified_records = []
                    for _, row in max_selected_pairs.iterrows():
                        sec = row["Sector"]
                        if sec not in sector_counts:
                            sector_counts[sec] = 0
                        if sector_counts[sec] < max_pairs_per_sector:
                            diversified_records.append(row)
                            sector_counts[sec] += 1
                        if len(diversified_records) >= n:
                            break
                    selected_pairs = pd.DataFrame(diversified_records).copy()
                else:
                    selected_pairs = max_selected_pairs.head(n).copy()

                if selected_pairs.empty:
                    continue

                selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                state  = states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)]
                slots  = state["slots"]

                # 分配重疊週期的獨立複利 Slot 插槽
                free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                slot_idx   = free_slots[0] if free_slots else min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                cap_period   = slots[slot_idx]["capital"]
                cap_per_pair = cap_period / n

                # 開始本期交易模擬
                trading = Trading(
                    price_df=extended_data, trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=cap_per_pair,
                    fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl, entry_z=self.entry_z, exit_z=self.exit_z,
                    zscore_window=z_win, allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip, min_spread_std=self.min_spread_std,
                    use_dynamic_stop=(dyn_z > 0),
                    dynamic_stop_z=dyn_z,
                    portfolio_stop_loss_pct=p_stop,
                    use_vol_adjust=vol_adj,
                )

                trade_log_df, period_pnl = trading.run(ts_str, te_str)

                if not trade_log_df.empty:
                    state["logs"].append(trade_log_df)

                # 更新該插槽可用時間與資金餘額
                slots[slot_idx]["capital"]   = max(0, cap_period + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        self._export_results(states)

    def _export_results(self, states):
        """
        將各參數組合累積的明細合併，並匯出為單一 CSV 檔案。
        """
        print("\n✅ 回測完成！正在匯出交易紀錄...")
        for (n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj), state in states.items():
            if state["logs"]:
                full_log = pd.concat(state["logs"], ignore_index=True)
                sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                rm_str   = getattr(self, "reduce_method", "umap").upper()
                psl_str = f"PSL{int(p_stop*100)}" if p_stop > 0 else "PSL0"
                msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
                dsz_str = f"DSZ{int(dyn_z)}" if dyn_z > 0 else "DSZ0"
                vol_str  = "VolAdj" if vol_adj else "NoVol"
                
                filename = f"HDBSCAN_{rm_str}_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
                filepath = self.output_dir / filename
                full_log.to_csv(filepath, index=False)
                print(f"  - 已輸出: {filename} (共 {len(full_log)} 筆紀錄)")
                
        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")


def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    外部主程式啟動入口。
    過濾 RollingBacktester 構造函數簽名所需的有效參數並調用。
    """
    import inspect
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default

    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )

    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")

```

<!-- ==================== CELL 11 ==================== -->

## 分群策略二：HDBSCAN + PyTorch 深度神經自編碼器 (AE) 特徵提取策略

### 5.1 策略原理與數學特徵

在標準 HDBSCAN 策略中，使用 UMAP 降維雖然強大，但 UMAP 本質上是基於局部近鄰圖的拓撲映射，屬於非參數方法，難以進行深層非線性層級特徵交互。**本策略引入深度表徵學習，使用 PyTorch 搭建深度前饋自編碼器 (MLP Autoencoder) 神經網絡**，進行特徵的深度非線性高度壓縮。其核心原理如下：

#### A. 自編碼器神經網絡架構 (MLP Autoencoder Network Structure)
自編碼器是一種無監督深度學習模型，由**編碼器 (Encoder)** 與**解碼器 (Decoder)** 兩部分組成。其目標是將輸入特徵 $X$ 投影至低維度的**瓶頸隱層 (Bottleneck Latent Space)** $Z$，再透過解碼器重構為 $\hat{X}$，強迫網絡濾除雜訊，學習到個股在時間序列上最具本質特徵的低維表徵嵌入：
- **輸入維度**：$D_{in} = 13$（個股 13 維統計與動量特徵）
- **編碼器網絡 (Encoder)**：
  $$H_{enc} = \tanh(W_{enc1} \cdot X + b_{enc1})$$
  $$Z = W_{enc2} \cdot H_{enc} + b_{enc2}$$
  其中第一層使用 64 個神經元配以 $\tanh$ 非線性激活函數，第二層將其壓縮至 $D_{latent} = 8$ 維空間。
- **解碼器網絡 (Decoder)**：
  $$H_{dec} = \tanh(W_{dec1} \cdot Z + b_{dec1})$$
  $$\hat{X} = W_{dec2} \cdot H_{dec} + b_{dec2}$$
  解碼器接收 8 維潛在變量 $Z$，經由 64 個神經元重構出原始的 13 維特徵 $\hat{X}$。

#### B. 損失函數與權重優化 (Loss Function & Optimization)
- **均方誤差重構損失 (MSE Loss)**：
  $$\mathcal{L}_{MSE}(X, \hat{X}) = \frac{1}{N_{stocks} \cdot 13} \sum_{i=1}^{N_{stocks}} \sum_{j=1}^{13} (x_{i,j} - \hat{x}_{i,j})^2$$
- **優化器**：採用 Adam 優化器，初始學習率 $lr = 0.01$，進行 100 次迭代 (epochs) 的全批次梯度下降訓練，以更新網絡的權重矩陣 $W$ 與偏置項 $b$。

#### C. 表徵提取與密度分群交易
訓練完成後，切換模型至 `model.eval()` 評估模式。關閉梯度計算，將原始特徵 $X$ 輸入至編碼器中，擷取隱層輸出 $Z \in \mathbb{R}^{N_{stocks} \times 8}$ 作為個股的深度特徵嵌入。隨後，直接對此 8 維深度表徵空間執行 UMAP/PCA 降維，再使用 HDBSCAN 進行密度分群，最終進行 Engle-Granger 共整合檢定與滾動 OLS 價差交易。這代表了機器學習與量化金融在特徵選股領域的最前沿結合。

### 5.2 Python 核心程式碼實現
以下是自編碼器在 PyTorch 下的模型架構、訓練迴圈與 HDBSCAN 配對形成的完整核心程式碼：

<!-- ==================== CODE CELL 12 ==================== -->

```python
# 嵌入 MLPAutoencoder、train_autoencoder 函數與 Formation 的 Python 程式碼
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
from statsmodels.tsa.stattools import adfuller

class MLPAutoencoder(nn.Module):
    """
    PyTorch 深度自編碼器模型
    用於將 13 維時序統計特徵壓縮為 8 維潜在表徵空間
    """
    def __init__(self, input_dim, latent_dim=8):
        super(MLPAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded

def train_autoencoder(X_train, latent_dim=8, epochs=100, lr=0.01):
    """
    自編碼器訓練流程
    """
    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    input_dim = X_train.shape[1]
    model = MLPAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        latent, decoded = model(tensor_x)
        loss = criterion(decoded, tensor_x)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        latent_features, _ = model(tensor_x)
    return latent_features.numpy()

```

<!-- ==================== CELL 13 ==================== -->

## 分群策略三：HDBSCAN + 金融多因子特徵空間策略

### 6.1 策略原理與數學特徵

前述分群策略的特徵向量全由單隻股票自身的時序統計量構成，未考量個股相對於全市場大盤的動態關聯與系統性風險特徵。**本策略引進多因子特徵空間模型，建構包含六大時序金融因子的穩健空間**。其核心原理如下：

#### A. 六大時序多因子特徵空間建構 (6 Financial Factors Space)
在形成期內，系統首先計算全市場所有合格個股在每日的收益率均值，將其作為「大盤指數收益率」代表 $R_{m, t}$。接著，對每支個股計算與大盤及時間趨勢相關聯 of 6 維特徵：
1. **系統性貝他風險 (Systematic Beta)**：個股對大盤收益率敏感係數：
   $$\beta_{market} = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$
2. **歷史總波動率 (Historical Volatility)**：個股收益率的標準差 $\sigma_i$，代表其總風險。
3. **收益率偏度 (Skewness)**：衡量收益率分佈的不對稱性，偏負代表肥尾暴跌機率高。
4. **收益率峰度 (Kurtosis)**：衡量極端極大/極小收益的頻率（尾部風險）。
5. **長期價格趨勢斜率 (Log Price Long-term Slope)**：個股對數收盤價相對於時間索引 $t \in [1, F]$ 的一元 OLS 回歸斜率。反映個股在形成期內的長期牛熊趨勢偏向。
6. **特異波動率 (Idiosyncratic Volatility)**：個股收益率對大盤收益率進行線性回歸的殘差標準差：
   $$R_{i, t} = \alpha_i + \beta_i R_{m, t} + \epsilon_{i, t}$$
   $$\sigma_{idiosyncratic} = \text{Std}(\epsilon_i)$$
   這代表個股無法被大盤指數解釋的特異風險成分，是尋求統計套利配對極為關鍵的特徵指標。

#### B. 跳過降維壓縮 (Skip Dimension Reduction / None Method)
- **黃金特徵空間**：由於六大金融特徵空間的維度僅有 6 維，恰好處於歐氏距離計算最不容易發生維度失效的「黃金維度區間」（一般大於10維才需要壓縮）。
- **顯式設定 `reduce_method = 'none'`**：策略跳過了 PCA 與 UMAP 壓縮步驟，直接將 Z-Score 標準化後的 6 維多因子特徵矩陣輸入至 HDBSCAN 密度聚類算法中。這保留了最為完整且具備明確金融學含義的原始因子結構，避免了降維過程中的特徵失真。

#### C. Engle-Granger 共整合與交易期模擬
在分群 label 標註完成後，於同產業且同因子密度群落內進行配對，利用 ADF p值與半衰期進行過濾。交易期內則依據市值中性與資金加權比例，執行 Z-Score 價差交易與滾動回歸。這是多因子量化選股與配對交易的絕佳交叉融合實踐。

### 6.2 Python 核心程式碼實現
以下是 `strategies/HDBSCAN_MultiFactor.py` 中金融因子計算、直接聚類、共整合篩選與交易模擬的完整實現：

<!-- ==================== CODE CELL 14 ==================== -->

```python
# 嵌入 HDBSCAN_MultiFactor.py 的完整實現
"""
HDBSCAN 分群配對交易滾動回測系統 (交易明細版) - 時序多因子特徵空間版
========================================================================

核心功能：本模組使用多因子量化特徵空間與密度聚類演算法 (HDBSCAN) 進行配對交易標的選配。
相較於標準版 (HDBSCAN.py) 進行多維度時序統計特徵擷取後使用 UMAP 降維，本版具備以下特徵：
1. 時序六大金融因子特徵空間：擷取個股與市場大盤之間的動態關聯與風險屬性，建構 6 維度的穩健金融特徵空間：
   - Beta (系統性風險係數)：個股相對於市場大盤收益率之敏感度 beta = cov(ticker, market) / var(market)
   - Volatility (歷史波動度)：個股收益率的標準差
   - Skewness (收益率偏態)：衡量收益率分佈的不對稱性
   - Kurtosis (收益率峰態)：衡量極端尾部風險
   - Slope (對數價格長期走勢斜率)：個股對數收盤價相對於時間索引的線性趨勢斜率
   - Idiosyncratic Volatility (特異波動度)：個股相對於市場進行 OLS 線性回歸之殘差波動度，代表無法被大盤解釋的特異風險
2. 跳過維度壓縮 (Skip Dimension Reduction)：由於特徵空間僅有 6 維，處於最適合距離計算的黃金區間，
   因此顯式設定 `reduce_method = "none"`，跳過 PCA/UMAP 降維步驟，直接在原始金融特徵空間中執行 HDBSCAN 密度聚類。
3. 群內 EG 共整合檢定與誤差修正模型：與標準版一致，對凝聚群落內的配對進行 Engle-Granger 共整合檢定與半衰期過濾。
"""

import sqlite3
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 偵測並載入可用之 HDBSCAN 聚類庫
try:
    import hdbscan
    HDBSCAN_LIB = "hdbscan"
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
        HDBSCAN_LIB = "sklearn"
    except ImportError:
        raise ImportError("請先安裝 scikit-learn >= 1.3.0 或 hdbscan：pip install scikit-learn hdbscan")

from sklearn.preprocessing import StandardScaler

# 忽略不必要的警告
warnings.filterwarnings("ignore")


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    執行一元線性最小二乘法回歸 (OLS)： Y = alpha + beta * X + residual
    """
    n = len(y)
    x_mat = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, y - np.mean(y)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    return alpha, beta, y - alpha - beta * x


def _adf_stat(resid: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """
    對回歸殘差執行無常數項的 ADF 單根平穩性檢定。
    """
    if len(resid) < max_lags + 5:
        return 0.0, 1.0
    try:
        result = adfuller(resid, maxlag=max_lags, regression="n", autolag=None)
        return float(result[0]), float(result[1])
    except Exception:
        return 0.0, 1.0


class Formation:
    """
    時序多因子特徵空間形成期處理器。
    執行：六大金融特徵工程 -> Z-Score 標準化 -> 直接 HDBSCAN 分群 -> 群內共整合與半衰期篩選。
    """
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        form_start: str,
        form_end: str,
        top_n: int = 20,
        sector_mapping: dict = None,
        min_tickers_for_pairing: int = 2,
        hdbscan_min_cluster_size: int = 3,
        hdbscan_min_samples: int = 1,
        hdbscan_metric: str = "euclidean",
        reduce_method: str = "none",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.05,
        max_sector_ratio: float = 0.3,
    ):
        """
        初始化多因子聚類形成期配置。
        """
        self.price_df = price_df
        self.form_start = form_start
        self.form_end   = form_end
        self.top_n      = top_n
        self.sector_mapping = sector_mapping or {}
        self.min_tickers_for_pairing = min_tickers_for_pairing

        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples
        self.hdbscan_metric           = hdbscan_metric

        # 固定設定為不降維，直接使用多因子特徵空間
        self.reduce_method = "none"
        self.adf_max_lags           = adf_max_lags
        self.adf_pvalue_threshold   = adf_pvalue_threshold
        self.max_sector_ratio       = max_sector_ratio

        self.selected_pairs: pd.DataFrame = pd.DataFrame()
        self.cluster_labels_: dict = {}

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        """
        計算 6 大金融特徵，構建時序多因子特徵空間矩陣，並進行 Z-Score 標準化。

        回傳:
            tuple: (標準化後的特徵矩陣 X [n_samples, 6], 股票代碼 tickers)
        """
        log_prices = np.log(self.price_df)
        tickers    = log_prices.columns.tolist()

        # 計算對數日收益率
        returns_df = log_prices.diff().dropna()
        if returns_df.empty or len(returns_df.columns) < 2:
            return np.empty((0, 0)), []

        # 估算大盤指數日收益率（各個股日收益率之橫截面均值）
        market_returns = returns_df.mean(axis=1).values
        feat_rows, valid_tickers = [], []
        t_indices = np.arange(len(log_prices))

        for ticker in tickers:
            prices = log_prices[ticker].values
            if len(prices) < 30 or not np.all(np.isfinite(prices)):
                continue

            ticker_ret = returns_df[ticker].values

            # 1. 系統性貝他風險 beta = cov(R_i, R_m) / var(R_m)
            cov_mat = np.cov(ticker_ret, market_returns)
            beta = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12) if cov_mat[1, 1] > 1e-12 else 0.0

            # 2. 歷史總波動率 (Volatility)
            vol = np.std(ticker_ret, ddof=1) if len(ticker_ret) > 1 else 0.0

            # 3. 收益率偏態 (Skewness)
            skew = float(pd.Series(ticker_ret).skew())

            # 4. 收益率峰態 (Kurtosis)
            kurt = float(pd.Series(ticker_ret).kurt())

            # 5. 長期趨勢斜率 (對數收盤價相對於時間趨勢的最小二乘回歸斜率)
            try:
                x_mat = np.column_stack([np.ones(len(prices)), t_indices])
                coeffs, _, _, _ = np.linalg.lstsq(x_mat, prices, rcond=None)
                slope = float(coeffs[1])
            except Exception:
                slope = 0.0

            # 6. 特異波動率 (Idiosyncratic Volatility)：收益率對大盤收益率線性回歸之殘差標準差
            try:
                alpha, beta_val, resid = _ols(ticker_ret, market_returns)
                idio_vol = np.std(resid, ddof=1) if len(resid) > 1 else 0.0
            except Exception:
                idio_vol = vol

            # 組裝 6 大金融因子
            feats = np.array([beta, vol, skew, kurt, slope, idio_vol], dtype=np.float64)
            feats = np.where(np.isfinite(feats), feats, 0.0)

            feat_rows.append(feats)
            valid_tickers.append(ticker)

        if not feat_rows:
            return np.empty((0, 0)), []

        X = np.vstack(feat_rows)
        # Z-Score 標準化
        X = StandardScaler().fit_transform(X)
        return X, valid_tickers

    def _hdbscan_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        使用 HDBSCAN 在 6 維多因子特徵空間中執行層級密度分群。
        """
        min_cs = min(self.hdbscan_min_cluster_size, max(2, X.shape[0] // 5))
        if HDBSCAN_LIB == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size = min_cs,
                min_samples      = self.hdbscan_min_samples,
                metric           = self.hdbscan_metric,
                core_dist_n_jobs = -1,
            )
        else:
            clusterer = sklearn_HDBSCAN(
                min_cluster_size = min_cs,
                min_samples      = self.hdbscan_min_samples,
                metric           = self.hdbscan_metric,
                n_jobs           = -1,
            )
        clusterer.fit(X)
        return clusterer.labels_

    def _cointegration_within_clusters(self, tickers: list[str], labels: np.ndarray) -> pd.DataFrame:
        """
        在相同的「產業分群 (Sector) x 多因子聚類群落 (Cluster Label)」內進行兩兩配對之 Engle-Granger Cointegration 檢定與 ECM 半衰期過濾。
        """
        log_prices = np.log(self.price_df[tickers])
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            print("  [Formation] HDBSCAN 未找到任何有效群落（全為噪音點）。")
            return pd.DataFrame()

        noise_count = int(np.sum(labels == -1))
        print(f"  [Formation] HDBSCAN 分群結果：{len(unique_labels)} 個群落，"
              f"{noise_count} 個噪音點排除，"
              f"ADF p 值門檻 = {self.adf_pvalue_threshold:.2f} "
              f"({'保守 1%' if self.adf_pvalue_threshold <= 0.01 else '積極 5%'})")

        ticker_meta: dict[str, tuple[str, int]] = {}
        for t, lbl in zip(tickers, labels):
            sector = self.sector_mapping.get(t.upper(), "Unknown")
            ticker_meta[t] = (sector, int(lbl))

        group_map: dict[tuple[str, int], list[str]] = {}
        for t, (sec, lbl) in ticker_meta.items():
            if sec == "Unknown" or lbl == -1:
                continue
            group_map.setdefault((sec, lbl), []).append(t)

        valid_groups = {k: v for k, v in group_map.items() if len(v) >= self.min_tickers_for_pairing}
        if not valid_groups:
            print("  [Formation] 同產業 × 同群落後無有效配對組合。")
            return pd.DataFrame()

        eg_records = []
        passed_count = 0
        rejected_count = 0

        for (sector, cluster_lbl), group_tickers in sorted(valid_groups.items()):
            for i, ta in enumerate(group_tickers):
                log_a = log_prices[ta].values
                for j in range(i + 1, len(group_tickers)):
                    tb    = group_tickers[j]
                    log_b = log_prices[tb].values

                    # 雙向最小二乘回歸
                    al_ab, be_ab, re_ab = _ols(log_a, log_b)
                    stat_ab, pval_ab = _adf_stat(re_ab, self.adf_max_lags)

                    al_ba, be_ba, re_ba = _ols(log_b, log_a)
                    stat_ba, pval_ba = _adf_stat(re_ba, self.adf_max_lags)

                    # 選擇顯著性高者
                    if pval_ab <= pval_ba:
                        best_stat, best_pval = stat_ab, pval_ab
                        best_alpha, best_beta, best_resid = al_ab, be_ab, re_ab
                        best_a, best_b = ta, tb
                    else:
                        best_stat, best_pval = stat_ba, pval_ba
                        best_alpha, best_beta, best_resid = al_ba, be_ba, re_ba
                        best_a, best_b = tb, ta

                    # 篩選 ADF 統計量顯著度 p-value
                    if best_pval >= self.adf_pvalue_threshold:
                        rejected_count += 1
                        continue

                    # 計算價差均值收斂半衰期 (Half-Life)
                    dy = np.diff(best_resid)
                    y_lag = best_resid[:-1]
                    n_dy = len(dy)
                    x_mat = np.column_stack([np.ones(n_dy), y_lag])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(x_mat, dy, rcond=None)
                        lambda_val = coeffs[1]
                    except Exception:
                        lambda_val = 0.0

                    if lambda_val >= 0.0:
                        rejected_count += 1
                        continue

                    halflife = -np.log(2) / lambda_val
                    # 半衰期過濾限制在合理的 2 至 60 交易日內
                    if halflife < 2.0 or halflife > 60.0:
                        rejected_count += 1
                        continue

                    passed_count += 1
                    spread_mean = float(np.mean(best_resid))
                    spread_std  = float(np.std(best_resid, ddof=1)) if len(best_resid) > 1 else 0.0

                    eg_records.append({
                        "Form_Start":    self.form_start,
                        "Form_End":      self.form_end,
                        "Sector":        sector,
                        "Cluster_Label": cluster_lbl,
                        "Ticker_A":      best_a,
                        "Ticker_B":      best_b,
                        "ADF_Stat":      round(best_stat,   6),
                        "ADF_PValue":    round(best_pval,   6),
                        "Hedge_Ratio":   round(best_beta,   6),
                        "OLS_Alpha":     round(best_alpha,  6),
                        "Spread_Mean":   round(spread_mean, 6),
                        "Spread_Std":    round(spread_std,  6),
                    })

        print(f"  [Formation] EG 檢定：{passed_count} 對通過 p < {self.adf_pvalue_threshold}，"
              f"{rejected_count} 對被 p 值門檻/半衰期排除。")

        if not eg_records:
            return pd.DataFrame()

        return pd.DataFrame(eg_records).sort_values("ADF_Stat").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """
        執行多因子聚類配對篩選工作流。
        """
        X, valid_tickers = self._build_feature_matrix()
        if len(valid_tickers) < self.min_tickers_for_pairing:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 顯式直接使用多因子原始特徵矩陣進行密度分群，跳過 UMAP/PCA 壓縮步驟
        X_embed = X

        labels = self._hdbscan_cluster(X_embed)
        self.cluster_labels_ = dict(zip(valid_tickers, labels.tolist()))

        eg_df = self._cointegration_within_clusters(valid_tickers, labels)
        if eg_df.empty:
            self.selected_pairs = pd.DataFrame()
            return self.selected_pairs

        # 分散化處理
        if getattr(self, "max_sector_ratio", 0) > 0:
            max_pairs_per_sector = max(1, int(self.top_n * self.max_sector_ratio))
            sector_counts = {}
            diversified_records = []
            for _, row in eg_df.iterrows():
                sec = row["Sector"]
                if sec not in sector_counts:
                    sector_counts[sec] = 0
                if sector_counts[sec] < max_pairs_per_sector:
                    diversified_records.append(row)
                    sector_counts[sec] += 1
                if len(diversified_records) >= self.top_n:
                    break
            selected = pd.DataFrame(diversified_records).copy()
        else:
            selected = eg_df.head(self.top_n).copy()

        selected["Rank"] = range(1, len(selected) + 1)

        # 標準化基礎映射
        log_prices  = np.log(self.price_df)
        mean_prices = log_prices.mean()
        std_prices  = log_prices.std()

        selected["Log_Mean_A"] = selected["Ticker_A"].map(mean_prices)
        selected["Log_Std_A"]  = selected["Ticker_A"].map(std_prices)
        selected["Log_Mean_B"] = selected["Ticker_B"].map(mean_prices)
        selected["Log_Std_B"]  = selected["Ticker_B"].map(std_prices)

        self.selected_pairs = selected
        return self.selected_pairs


@dataclass(slots=True)
class PairState:
    """
    持倉狀態與損益追蹤。
    """
    position: int = 0
    shares_a: float = 0.0
    shares_b: float = 0.0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    realized_pnl: float = 0.0
    trade_entry_fee: float = 0.0
    days_held: int = 0
    is_stopped: bool = False
    cooldown_dir: int = 0
    prev_total_pnl: float = 0.0


class Trading:
    """
    配對交易模擬器，計算每日 Z-Score 以生成進出場訊號，支援滾動 OLS 擬合。
    """
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        trade_dates: pd.DatetimeIndex,
        selected_pairs: pd.DataFrame,
        capital_per_pair: float,
        fee_rate: float,
        slippage_rate: float,
        stop_loss_pct: float,
        entry_z: float,
        exit_z: float,
        zscore_window: int,
        allow_reentry: bool = False,
        zscore_clip: float = 10.0,
        min_spread_std: float = 1e-6,
        use_dynamic_stop: bool = False,
        dynamic_stop_z: float = 3.0,
        portfolio_stop_loss_pct: float = 0.10,
        use_vol_adjust: bool = False,
    ):
        """
        初始化交易。
        """
        self.price_df        = price_df
        self.trade_dates     = trade_dates
        self.selected_pairs  = selected_pairs
        self.capital_per_pair = capital_per_pair
        self.friction_rate   = fee_rate + slippage_rate
        self.stop_loss_pct   = stop_loss_pct
        self.entry_z         = entry_z
        self.exit_z          = exit_z
        self.zscore_window   = zscore_window
        self.allow_reentry   = allow_reentry
        self.zscore_clip     = zscore_clip
        self.min_spread_std  = min_spread_std
        self.use_dynamic_stop = use_dynamic_stop
        self.dynamic_stop_z  = dynamic_stop_z
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.use_vol_adjust  = use_vol_adjust
        self.period_pnl: float = 0.0

    def _execute_entry(self, state, z, p_a, p_b, hedge_ratio):
        """
        市值中性與資金加權比重建倉。
        """
        total_weight = 1.0 + abs(hedge_ratio)
        v_a = self.capital_per_pair * (1.0 / total_weight)
        v_b = self.capital_per_pair * (abs(hedge_ratio) / total_weight)

        if z > self.entry_z and state.cooldown_dir != -1:
            state.position = -1
            state.shares_a = -v_a / p_a
            state.shares_b =  v_b / p_b
        elif z < -self.entry_z and state.cooldown_dir != 1:
            state.position = +1
            state.shares_a =  v_a / p_a
            state.shares_b = -v_b / p_b
        else:
            return False, 0.0

        state.entry_price_a  = p_a
        state.entry_price_b  = p_b
        state.trade_entry_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
        state.days_held = 0
        return True, -state.trade_entry_fee

    def _execute_close(self, state, current_trade_pnl, stop_loss=False):
        """
        平倉。
        """
        state.realized_pnl += current_trade_pnl
        if stop_loss:
            state.is_stopped = True if not self.allow_reentry else False
            if self.allow_reentry:
                state.cooldown_dir = state.position
        else:
            state.cooldown_dir = state.position
        state.position = 0

    def _simulate_pair(
        self, period_start, period_end, sector, ticker_a, ticker_b, pair_rank,
        hedge_ratio, ols_alpha, form_spread_mean, form_spread_std,
        log_mean_a, log_std_a, log_mean_b, log_std_b,
        cluster_label, cluster_group,
    ) -> pd.DataFrame:
        """
        單一配對逐日交易期狀態機模擬。
        """
        if ticker_a not in self.price_df.columns or ticker_b not in self.price_df.columns:
            return pd.DataFrame()

        price_a = self.price_df[ticker_a].dropna()
        price_b = self.price_df[ticker_b].dropna()
        common_idx = price_a.index.intersection(price_b.index)
        price_a    = price_a.loc[common_idx]
        price_b    = price_b.loc[common_idx]

        if len(price_a) < 5:
            return pd.DataFrame()

        log_a = np.log(price_a)
        log_b = np.log(price_b)

        if self.zscore_window == 0:
            # 固定參數模式
            spread   = log_a - ols_alpha - hedge_ratio * log_b
            safe_std = max(form_spread_std, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std
            zscore   = np.clip((spread - form_spread_mean) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = pd.Series(hedge_ratio, index=common_idx)
            alpha_series = pd.Series(ols_alpha,   index=common_idx)
        else:
            # 滾動回歸模式：每日在滾動視窗上重新擬合 OLS 殘差 Z-Score
            w = self.zscore_window
            n = len(log_a)
            la_vals, lb_vals = log_a.values, log_b.values
            roll_alpha = np.full(n, np.nan)
            roll_beta  = np.full(n, np.nan)
            roll_mean  = np.full(n, np.nan)
            roll_std   = np.full(n, np.nan)

            for k in range(w - 1, n):
                ya = la_vals[k - w + 1: k + 1]
                xb = lb_vals[k - w + 1: k + 1]
                a_, b_, r_ = _ols(ya, xb)
                roll_alpha[k] = a_
                roll_beta[k]  = b_
                roll_mean[k]  = float(np.mean(r_))
                roll_std[k]   = float(np.std(r_, ddof=1)) if len(r_) > 1 else 0.0

            roll_alpha_s = pd.Series(roll_alpha, index=common_idx)
            roll_beta_s  = pd.Series(roll_beta,  index=common_idx)
            roll_mean_s  = pd.Series(roll_mean,  index=common_idx)
            roll_std_s   = pd.Series(roll_std,   index=common_idx)

            spread     = log_a - roll_alpha_s - roll_beta_s * log_b
            safe_std_s = np.maximum(roll_std_s, self.min_spread_std)
            if getattr(self, "use_vol_adjust", False):
                roll20_std = spread.rolling(window=20, min_periods=1).std().fillna(form_spread_std)
                vol_factor = np.maximum(1.0, roll20_std / form_spread_std)
                adjusted_std = np.maximum(safe_std_s * vol_factor, self.min_spread_std)
            else:
                adjusted_std = safe_std_s
            zscore     = np.clip((spread - roll_mean_s) / adjusted_std, -self.zscore_clip, self.zscore_clip)
            beta_series  = roll_beta_s
            alpha_series = roll_alpha_s

        valid_idx = common_idx.intersection(self.trade_dates)
        if len(valid_idx) == 0:
            return pd.DataFrame()

        price_a      = price_a.loc[valid_idx]
        price_b      = price_b.loc[valid_idx]
        zscore       = zscore.loc[valid_idx]
        beta_series  = beta_series.loc[valid_idx]
        alpha_series = alpha_series.loc[valid_idx]

        dates_arr  = valid_idx
        zscore_arr = zscore.values
        pa_arr     = price_a.values
        pb_arr     = price_b.values
        beta_arr   = beta_series.values
        alpha_arr  = alpha_series.values

        base_log = {
            "Period_Start":   period_start,   "Period_End":     period_end,
            "Sector":         sector,          "Cluster_Label":  cluster_label,
            "Pair_Rank":      pair_rank,
            "Ticker_A":       ticker_a,        "Ticker_B":       ticker_b,
            "Log_Mean_A":     log_mean_a,      "Log_Std_A":      log_std_a,
            "Log_Mean_B":     log_mean_b,      "Log_Std_B":      log_std_b,
        }

        state = PairState()
        out_dates, out_pa, out_pb = [], [], []
        out_hr, out_ols_alpha, out_z, out_pos = [], [], [], []
        out_unr, out_rea, out_cum = [], [], []
        out_status, out_tpnl, out_days, out_delta = [], [], [], []

        def _append_row(date, p_a, p_b, c_beta, c_alpha, z_val, pos,
                        unr, rea, cum, status, tpnl, days, delta):
            out_dates.append(date);      out_pa.append(round(p_a, 4));     out_pb.append(round(p_b, 4))
            out_hr.append(round(c_beta, 4)); out_ols_alpha.append(round(c_alpha, 6))
            out_z.append(round(z_val, 4));   out_pos.append(pos)
            out_unr.append(round(unr, 4));   out_rea.append(round(rea, 4)); out_cum.append(round(cum, 4))
            out_status.append(status);   out_tpnl.append(round(tpnl, 4))
            out_days.append(days);        out_delta.append(round(delta, 4))

        for i in range(len(dates_arr)):
            date    = dates_arr[i]
            z       = 0.0 if np.isnan(zscore_arr[i]) else float(zscore_arr[i])
            p_a, p_b = float(pa_arr[i]), float(pb_arr[i])
            c_beta   = float(beta_arr[i])  if not np.isnan(beta_arr[i])  else hedge_ratio
            c_alpha  = float(alpha_arr[i]) if not np.isnan(alpha_arr[i]) else ols_alpha

            unr, tpnl, status = 0.0, 0.0, "HOLD_CASH"

            if state.is_stopped:
                _append_row(date, p_a, p_b, c_beta, c_alpha, z, 0,
                            0.0, state.realized_pnl, state.realized_pnl,
                            "STOPPED", 0.0, 0, 0.0)
                continue

            if   state.cooldown_dir == -1 and z <= self.exit_z:  state.cooldown_dir = 0
            elif state.cooldown_dir ==  1 and z >= -self.exit_z: state.cooldown_dir = 0

            # 持倉
            if state.position != 0:
                state.days_held += 1
                raw_unr  = state.shares_a * (p_a - state.entry_price_a) + state.shares_b * (p_b - state.entry_price_b)
                exit_fee = (abs(state.shares_a) * p_a + abs(state.shares_b) * p_b) * self.friction_rate
                cur_tpnl = raw_unr - state.trade_entry_fee - exit_fee

                is_cap_stop = self.stop_loss_pct > 0 and (-cur_tpnl / self.capital_per_pair) >= self.stop_loss_pct
                is_z_stop = self.use_dynamic_stop and abs(z) > self.dynamic_stop_z

                if is_cap_stop or is_z_stop:
                    self._execute_close(state, cur_tpnl, stop_loss=True)
                    tpnl, status = cur_tpnl, "STOP_LOSS_TRIGGERED"
                elif (state.position == -1 and z <= self.exit_z) or (state.position == 1 and z >= -self.exit_z):
                    self._execute_close(state, cur_tpnl, stop_loss=False)
                    tpnl, status = cur_tpnl, "EXIT"
                else:
                    unr    = raw_unr - state.trade_entry_fee
                    status = "HOLDING"
            # 空倉
            else:
                if abs(z) > self.entry_z:
                    entered, unr = self._execute_entry(state, z, p_a, p_b, c_beta)
                    status = ("ENTER_SHORT_A" if state.position == -1 else "ENTER_LONG_A") if entered else "HOLD_CASH (COOLDOWN)"
                else:
                    status = "HOLD_CASH"

            cum   = state.realized_pnl + unr
            delta = cum - state.prev_total_pnl
            state.prev_total_pnl = cum

            _append_row(date, p_a, p_b, c_beta, c_alpha, z, state.position,
                        unr, state.realized_pnl, cum, status, tpnl, state.days_held, delta)

            if status in ("STOP_LOSS_TRIGGERED", "EXIT"):
                state.days_held = 0

            # 永久停損提前填充
            if state.is_stopped and i < len(dates_arr) - 1:
                for j in range(i + 1, len(dates_arr)):
                    _append_row(
                        dates_arr[j], float(pa_arr[j]), float(pb_arr[j]),
                        float(beta_arr[j]) if not np.isnan(beta_arr[j]) else hedge_ratio,
                        float(alpha_arr[j]) if not np.isnan(alpha_arr[j]) else ols_alpha,
                        0.0 if np.isnan(zscore_arr[j]) else float(zscore_arr[j]),
                        0, 0.0, state.realized_pnl, state.realized_pnl,
                        "STOPPED", 0.0, 0, 0.0
                    )
                break

        # 交易期末強制平倉
        if state.position != 0 and out_status:
            if out_status[-1] not in ("EXIT", "STOP_LOSS_TRIGGERED", "PERIOD_END_EXIT", "STOPPED"):
                p_a_last, p_b_last = float(pa_arr[-1]), float(pb_arr[-1])
                raw_final = state.shares_a * (p_a_last - state.entry_price_a) + state.shares_b * (p_b_last - state.entry_price_b)
                fee_final = (abs(state.shares_a) * p_a_last + abs(state.shares_b) * p_b_last) * self.friction_rate
                final_tpnl = raw_final - state.trade_entry_fee - fee_final
                state.realized_pnl += final_tpnl
                pnl_prev = out_cum[-2] if len(out_cum) > 1 else 0.0

                out_status[-1]     = "PERIOD_END_EXIT"
                out_rea[-1]        = round(state.realized_pnl, 4)
                out_cum[-1]        = round(state.realized_pnl, 4)
                out_unr[-1]        = 0.0
                out_tpnl[-1]       = round(final_tpnl, 4)
                out_delta[-1]      = round(state.realized_pnl - pnl_prev, 4)
                out_days[-1]       = state.days_held

        if not out_dates:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "Date": out_dates, "Price_A": out_pa, "Price_B": out_pb,
            "Hedge_Ratio": out_hr, "OLS_Alpha": out_ols_alpha,
            "ZScore": out_z, "Position": out_pos,
            "Unrealized_PnL": out_unr, "Realized_PnL": out_rea,
            "Cumulative_PnL": out_cum, "Status": out_status,
            "Trade_PnL": out_tpnl, "Days_Held": out_days, "Daily_Delta": out_delta,
        })
        for k, v in base_log.items():
            df_out[k] = v
        return df_out

    def run(self, period_start: str, period_end: str) -> tuple:
        """
        平行模擬所有選定的配對，並依據設定判定是否觸發投資組合層級最大累計停損。
        """
        dfs = []
        for _, row in self.selected_pairs.iterrows():
            df_pair = self._simulate_pair(
                period_start=period_start, period_end=period_end,
                sector=row.get("Sector", "Unknown"),
                ticker_a=row["Ticker_A"], ticker_b=row["Ticker_B"],
                pair_rank=row["Rank"],
                hedge_ratio=float(row.get("Hedge_Ratio", 1.0)),
                ols_alpha=float(row.get("OLS_Alpha", 0.0)),
                form_spread_mean=float(row.get("Spread_Mean", 0.0)),
                form_spread_std=float(row.get("Spread_Std", 1.0)),
                log_mean_a=float(row.get("Log_Mean_A", 0.0)),
                log_std_a=float(row.get("Log_Std_A",  1.0)),
                log_mean_b=float(row.get("Log_Mean_B", 0.0)),
                log_std_b=float(row.get("Log_Std_B",  1.0)),
                cluster_label=int(row.get("Cluster_Label", -1)),
                cluster_group=str(row.get("Sector", "Unknown")),
            )
            if not df_pair.empty:
                dfs.append(df_pair)

        if not dfs:
            return pd.DataFrame(), 0.0

        # 投資組合層級最大回撤停損機制 (Portfolio Stop Loss)
        if getattr(self, "portfolio_stop_loss_pct", 0) > 0:
            temp_df = pd.concat(dfs, ignore_index=True)
            total_cap = self.capital_per_pair * len(dfs)
            daily_cum_pnl = temp_df.groupby("Date")["Cumulative_PnL"].sum()

            cutoff_date = None
            for date_val, pnl_val in daily_cum_pnl.items():
                if pnl_val / total_cap <= -self.portfolio_stop_loss_pct:
                    cutoff_date = date_val
                    break

            if cutoff_date is not None:
                new_dfs = []
                for df in dfs:
                    df = df.copy()
                    before_mask = df["Date"] < cutoff_date
                    at_mask = df["Date"] == cutoff_date
                    after_mask = df["Date"] > cutoff_date

                    df_before = df[before_mask]

                    df_at = df[at_mask].copy()
                    final_realized = 0.0
                    if not df_at.empty:
                        row_at = df_at.iloc[0]
                        final_realized = row_at["Cumulative_PnL"]
                        if row_at["Position"] != 0:
                            df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOP_TRIGGERED"
                            df_at.loc[df_at.index, "Position"] = 0
                            df_at.loc[df_at.index, "Unrealized_PnL"] = 0.0
                            df_at.loc[df_at.index, "Trade_PnL"] = row_at["Trade_PnL"]
                        else:
                            if row_at["Status"] not in ("STOPPED", "STOP_LOSS_TRIGGERED", "EXIT"):
                                df_at.loc[df_at.index, "Status"] = "PORTFOLIO_STOPPED"
                            final_realized = row_at["Realized_PnL"]

                    df_after = df[after_mask].copy()
                    if not df_after.empty:
                        df_after.loc[df_after.index, "Position"] = 0
                        df_after.loc[df_after.index, "Unrealized_PnL"] = 0.0
                        df_after.loc[df_after.index, "Realized_PnL"] = final_realized
                        df_after.loc[df_after.index, "Cumulative_PnL"] = final_realized
                        df_after.loc[df_after.index, "Status"] = "STOPPED"
                        df_after.loc[df_after.index, "Trade_PnL"] = 0.0
                        df_after.loc[df_after.index, "Daily_Delta"] = 0.0

                    new_dfs.append(pd.concat([df_before, df_at, df_after], ignore_index=True))
                dfs = new_dfs

        log_df = pd.concat(dfs, ignore_index=True)
        period_daily = log_df.groupby("Date")["Daily_Delta"].sum()
        self.period_pnl = float(period_daily.sum()) if not period_daily.empty else 0.0
        return log_df, self.period_pnl


class RollingBacktester:
    """
    時序多因子 HDBSCAN 回測引擎。
    執行滾動 Walk-Forward 網格搜索，採用最終聚合導出 CSV。
    """
    
    def __init__(
        self,
        top_n_list: list,
        stop_loss_list: list,
        zscore_window_list: list,
        entry_z: float,
        exit_z: float,
        formation_window: int,
        trading_window: int,
        rolling_step: int,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        allow_reentry: bool,
        zscore_clip: float,
        min_spread_std: float,
        min_tickers_for_pairing: int,
        hdbscan_min_cluster_size: int,
        hdbscan_min_samples: int,
        hdbscan_metric: str,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        adf_max_lags: int = 1,
        adf_pvalue_threshold: float = 0.05,
        output_dir: Path = None,
        reduce_method: str = "none",
        portfolio_stop_loss_pct_list: list = None,
        max_sector_ratio_list: list = None,
        dynamic_stop_z_list: list = None,
        use_vol_adjust_list: list = None,
    ):
        """
        初始化多因子回測器。
        """
        self.top_n_list = top_n_list
        self.stop_loss_list = stop_loss_list
        self.zscore_window_list = zscore_window_list
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.rolling_step = rolling_step
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.allow_reentry = allow_reentry
        self.zscore_clip = zscore_clip
        self.min_spread_std = min_spread_std
        self.min_tickers_for_pairing = min_tickers_for_pairing

        # 密度聚類特徵
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_metric = hdbscan_metric
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.adf_max_lags = adf_max_lags
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.output_dir = output_dir
        self.reduce_method = reduce_method

        self.portfolio_stop_loss_pct_list = portfolio_stop_loss_pct_list or [0.0]
        self.max_sector_ratio_list = max_sector_ratio_list or [0.0]
        self.dynamic_stop_z_list = dynamic_stop_z_list or [0.0]
        self.use_vol_adjust_list = use_vol_adjust_list or [False]

    def run(self, price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping):
        """
        運行網格搜索。
        """
        max_concurrent = self.trading_window // self.rolling_step
        states = {}
        for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
            self.top_n_list, self.stop_loss_list, self.zscore_window_list,
            self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
        ):
            states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)] = {
                "logs":  [],
                "slots": [{"avail_idx": 0, "capital": self.initial_capital / max_concurrent}
                          for _ in range(max_concurrent)],
            }

        roll_start_indices = list(range(local_first_trade_idx, total_days - self.trading_window + 1, self.rolling_step))
        print(f"\n🚀 開始 HDBSCAN MultiFactor Grid Search，共 {len(roll_start_indices)} 期，每期 {len(states)} 種參數組合...")

        for roll_idx, trade_start_idx in enumerate(roll_start_indices):
            form_start_idx = trade_start_idx - self.formation_window
            form_end_idx   = trade_start_idx
            trade_end_idx  = min(trade_start_idx + self.trading_window, total_days)

            form_data   = price_pivot.iloc[form_start_idx:form_end_idx]
            trade_data  = price_pivot.iloc[trade_start_idx:trade_end_idx]
            valid_cols  = (form_data.isnull().sum() + trade_data.isnull().sum()) == 0
            form_data   = form_data.loc[:, valid_cols]
            trade_dates = trade_data.index

            extended_start = max(0, trade_start_idx - max(self.zscore_window_list))
            extended_data  = price_pivot.iloc[extended_start:trade_end_idx].loc[:, valid_cols]

            if form_data.shape[1] < 2 or trade_data.empty:
                continue

            ts_str = str(all_dates[trade_start_idx])[:10]
            te_str = str(all_dates[trade_end_idx - 1])[:10]
            fs_str = str(all_dates[form_start_idx])[:10]
            fe_str = str(all_dates[form_end_idx - 1])[:10]
            print(f"  ▶ 第 {roll_idx+1:02d} 期 (交易: {ts_str} ~ {te_str})")

            formation = Formation(
                price_df=form_data,
                form_start=fs_str, form_end=fe_str,
                top_n=max(self.top_n_list) * 5,
                sector_mapping=sector_mapping,
                min_tickers_for_pairing=self.min_tickers_for_pairing,
                hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,
                hdbscan_min_samples=self.hdbscan_min_samples,
                hdbscan_metric=self.hdbscan_metric,
                adf_max_lags=self.adf_max_lags,
                adf_pvalue_threshold=self.adf_pvalue_threshold,
                max_sector_ratio=0,
            )
            max_selected_pairs = formation.run()

            if max_selected_pairs.empty:
                continue

            for n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj in itertools.product(
                self.top_n_list, self.stop_loss_list, self.zscore_window_list,
                self.portfolio_stop_loss_pct_list, self.max_sector_ratio_list, self.dynamic_stop_z_list, self.use_vol_adjust_list
            ):
                if sec_ratio > 0:
                    max_pairs_per_sector = max(1, int(n * sec_ratio))
                    sector_counts = {}
                    diversified_records = []
                    for _, row in max_selected_pairs.iterrows():
                        sec = row["Sector"]
                        if sec not in sector_counts:
                            sector_counts[sec] = 0
                        if sector_counts[sec] < max_pairs_per_sector:
                            diversified_records.append(row)
                            sector_counts[sec] += 1
                        if len(diversified_records) >= n:
                            break
                    selected_pairs = pd.DataFrame(diversified_records).copy()
                else:
                    selected_pairs = max_selected_pairs.head(n).copy()

                if selected_pairs.empty:
                    continue

                selected_pairs["Rank"] = range(1, len(selected_pairs) + 1)
                state  = states[(n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj)]
                slots  = state["slots"]

                free_slots = [i for i, s in enumerate(slots) if s["avail_idx"] <= trade_start_idx]
                slot_idx   = free_slots[0] if free_slots else min(range(max_concurrent), key=lambda i: slots[i]["avail_idx"])

                cap_period   = slots[slot_idx]["capital"]
                cap_per_pair = cap_period / n

                trading = Trading(
                    price_df=extended_data, trade_dates=trade_dates,
                    selected_pairs=selected_pairs,
                    capital_per_pair=cap_per_pair,
                    fee_rate=self.fee_rate, slippage_rate=self.slippage_rate,
                    stop_loss_pct=sl, entry_z=self.entry_z, exit_z=self.exit_z,
                    zscore_window=z_win, allow_reentry=self.allow_reentry,
                    zscore_clip=self.zscore_clip, min_spread_std=self.min_spread_std,
                    use_dynamic_stop=(dyn_z > 0),
                    dynamic_stop_z=dyn_z,
                    portfolio_stop_loss_pct=p_stop,
                    use_vol_adjust=vol_adj,
                )

                trade_log_df, period_pnl = trading.run(ts_str, te_str)

                if not trade_log_df.empty:
                    state["logs"].append(trade_log_df)

                slots[slot_
```

<!-- ==================== CODE CELL 15 ==================== -->

```python
idx]["capital"]   = max(0, cap_period + period_pnl)
                slots[slot_idx]["avail_idx"] = trade_end_idx

        self._export_results(states)

    def _export_results(self, states):
        """
        匯出回測統計與明細。
        """
        print("\n✅ 回測完成！正在匯出交易紀錄...")
        for (n, sl, z_win, p_stop, sec_ratio, dyn_z, vol_adj), state in states.items():
            if state["logs"]:
                full_log = pd.concat(state["logs"], ignore_index=True)
                sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"
                rm_str   = "MULTIFACTOR"
                psl_str = f"PSL{int(p_stop*100)}" if p_stop > 0 else "PSL0"
                msr_str = f"MSR{int(sec_ratio*100)}" if sec_ratio > 0 else "MSR0"
                dsz_str = f"DSZ{int(dyn_z)}" if dyn_z > 0 else "DSZ0"
                vol_str  = "VolAdj" if vol_adj else "NoVol"
                
                filename = f"HDBSCAN_{rm_str}_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}_{psl_str}_{msr_str}_{dsz_str}_{vol_str}.csv"
                filepath = self.output_dir / filename
                full_log.to_csv(filepath, index=False)
                print(f"  - 已輸出: {filename} (共 {len(full_log)} 筆紀錄)")
                
        print(f"\n📁 所有交易紀錄已成功儲存至: {self.output_dir}")


def run_strategy(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping, params, output_dir):
    """
    外部啟動入口接口。
    """
    import inspect
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_sig = inspect.signature(RollingBacktester.__init__)
    valid_params = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ('self', 'output_dir'):
            continue
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param.default is not inspect.Parameter.empty:
            valid_params[param_name] = param.default

    print(f"[{Path(__file__).stem.upper()}] 正在初始化 RollingBacktester...")
    engine = RollingBacktester(
        output_dir=out_dir,
        **valid_params
    )

    print(f"[{Path(__file__).stem.upper()}] 正在啟動滾動回測...")
    engine.run(price_pivot, all_dates, total_days, local_first_trade_idx, sector_mapping)
    print(f"[{Path(__file__).stem.upper()}] 回測執行完畢。")

```