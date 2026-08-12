"""
回歸式進場（reversion entry）——學長 `純DTW配對交易-[版本2].ipynb` 的進場時點。

本研究現役的 Z-Score 交易端採 **突破式進場**（Gatev, Goetzmann & Rouwenhorst 2006）：
價差一發散到 ±entry_z 帶外就進場，賭它會收斂。學長的程式反過來——

    if prev > upper and curr <= upper:  df.at[t, 'signal1'] = -1
    elif prev < lower and curr >= lower: df.at[t, 'signal1'] = 1

價差必須先跑到帶外、再**收斂回帶內**，才在穿越當日進場。等於多等一個「已經開始
回歸」的確認訊號。這是老師指出的兩份回測八項差異中的第五項，單獨隔離出來測。

兩者的取捨：
  - 突破式吃得到完整的收斂幅度，但會被持續發散的價差套牢（本研究靠停損處理）。
  - 回歸式犧牲掉帶外那一段報酬，換取「不接下墜的刀」；代價是若價差直接一路發散
    到期末不回頭，這一趟就完全不進場——訊號數會比突破式少。

除了進場時點，其餘一切（部位規模、對沖比率、手續費、停損、出場、政體過濾）
都直接沿用親代 `zscore_trading.Trading`，確保差異可完全歸因於進場時點。
"""
from strategies.trading.zscore_trading import Trading as _Breakout, PairState


class Trading(_Breakout):
    """進場時點改為「發散後收斂回帶內」，其餘與 zscore_trading.Trading 相同。"""

    # 最近一次觸發是穿越哪一條帶：+1 = 由上帶外收斂回來（放空價差），
    # -1 = 由下帶外收斂回來（做多價差），0 = 未觸發。
    # 由 _entry_triggered 寫入、緊接著在同一次迴圈由 _execute_entry 讀取。
    #
    # 刻意宣告為類別屬性而不覆寫 __init__：run_trading 以 inspect 檢查建構子
    # 是否有 **kwargs 來決定要不要過濾參數（run_trading.py:363），若這裡寫成
    # __init__(*args, **kwargs)，它會把 full_price_df 等親代不收的參數也一併
    # 灌進來而爆 TypeError。不覆寫 __init__ 即完整繼承親代簽章。
    _entry_dir: int = 0

    def _entry_triggered(self, z: float, z_prev: float) -> bool:
        """前一日在帶外、今日回到帶內 → 進場。純突破（連續兩日皆在帶外）不進場。"""
        up = self.entry_z
        if z_prev > up and z <= up:
            self._entry_dir = +1
            return True
        if z_prev < -up and z >= -up:
            self._entry_dir = -1
            return True
        self._entry_dir = 0
        return False

    def _execute_entry(self, state: PairState, z: float, p_a: float, p_b: float,
                       hedge_ratio: float) -> tuple[bool, float]:
        """
        進場當日 z 已回到帶內，親代以 |z| > entry_z 判方向會判不出來（會回傳
        False）。方向其實由「穿越了哪一條帶」決定，已由 _entry_triggered 記在
        _entry_dir，故此處代入該側的帶外代表值，讓親代的資金配置、對沖比率、
        手續費計算原封不動地沿用。
        """
        if self._entry_dir == 0:
            return False, 0.0
        z_eff = (self.entry_z + 1e-9) * self._entry_dir
        return super()._execute_entry(state, z_eff, p_a, p_b, hedge_ratio)

    def _simulate_pair(self, *args, **kwargs):
        # 每對重置，避免上一對的殘留方向影響本對的第一次進場判斷。
        self._entry_dir = 0
        return super()._simulate_pair(*args, **kwargs)
