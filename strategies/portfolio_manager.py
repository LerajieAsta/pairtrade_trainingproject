class PortfolioManager:
    """
    組合層資金配置。

    兩種模式：
      靜態槽位（預設，向下相容）：
          capital_per_pair = current_equity / max_pairs
          max_pairs = top_n × 重疊期數 —— 理論上限，實際同時持倉常遠低於此，
          造成 40%+ 資金長期閒置、帳面年化被稀釋（動用資本年化 >> 帳面年化）。

      動態槽位（dynamic_slots=True，方案 A）：
          以 walk-forward「歷史同時承諾配對數」的分位數（預設 75 百分位）
          作為有效槽位數 target_slots：
              capital_per_pair = current_equity / target_slots
          護欄：
            1. 暖身期（觀測 < warmup_obs 次）沿用靜態 max_pairs —— 無前視；
            2. 單配對上限 pair_cap_frac × equity（預設 15%），防過度集中；
            3. 總承諾資本 ≤ current_equity —— 超額候選排隊（放棄本期進場）。
          歷史觀測只記「配置完成後的同時承諾數」，均為過去資訊，無前視。
    """

    def __init__(self, strategy_id: str, initial_capital: float = 10000.0, max_pairs: int = 10,
                 dynamic_slots: bool = False, slot_percentile: float = 75.0,
                 pair_cap_frac: float = 0.15, warmup_obs: int = 8):
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.max_pairs = max_pairs
        self.active_pairs = {}  # dict of (ticker_a, ticker_b) -> allocated_capital
        self.history = []

        # ── 動態槽位（方案 A）────────────────────────────────────────
        self.dynamic_slots = dynamic_slots
        self.slot_percentile = slot_percentile
        self.pair_cap_frac = pair_cap_frac
        self.warmup_obs = warmup_obs
        self.concurrency_history: list[int] = []   # 每次配置後的同時承諾配對數

    def get_available_slots(self) -> int:
        return max(0, self.max_pairs - len(self.active_pairs))

    def _target_slots(self) -> int:
        """動態模式的有效槽位數（walk-forward 分位數；暖身期回退 max_pairs）。"""
        if not self.dynamic_slots or len(self.concurrency_history) < self.warmup_obs:
            return self.max_pairs
        import numpy as np
        q = float(np.percentile(self.concurrency_history, self.slot_percentile))
        # 至少 1、至多 max_pairs；取 ceil 保守（多一槽 = 單配對資金少一點）
        import math
        return int(min(self.max_pairs, max(1, math.ceil(q))))

    def allocate_capital(self, candidate_pairs: list) -> dict:
        """
        Allocates capital to candidate pairs based on available slots.
        candidate_pairs: list of tuples (Ticker_A, Ticker_B)
        Returns a dict: {(Ticker_A, Ticker_B): allocated_amount}
        """
        allocations = {}
        slots = self.get_available_slots()

        if slots == 0 or self.current_equity <= 0:
            if self.dynamic_slots:
                self.concurrency_history.append(len(self.active_pairs))
            return allocations

        if self.dynamic_slots:
            target = self._target_slots()
            # 護欄 2：單配對上限 pair_cap_frac × equity
            capital_per_pair = min(self.current_equity / target,
                                   self.current_equity * self.pair_cap_frac)
            committed = sum(self.active_pairs.values())
            for pair in candidate_pairs[:slots]:
                # 護欄 3：總承諾 ≤ 權益，超額排隊
                if committed + capital_per_pair > self.current_equity + 1e-9:
                    break
                allocations[pair] = capital_per_pair
                self.active_pairs[pair] = capital_per_pair
                committed += capital_per_pair
            # 記錄本次配置後的同時承諾數（walk-forward 觀測）
            self.concurrency_history.append(len(self.active_pairs))
            return allocations

        # ── 靜態模式（原始行為，完全不變）─────────────────────────────
        capital_per_pair = self.current_equity / self.max_pairs

        for pair in candidate_pairs[:slots]:
            allocations[pair] = capital_per_pair
            self.active_pairs[pair] = capital_per_pair

        return allocations

    def process_closed_trade(self, pair: tuple, realized_pnl: float):
        """
        Updates the portfolio equity when a trade is closed.
        """
        if pair in self.active_pairs:
            del self.active_pairs[pair]
        self.current_equity += realized_pnl
        self.history.append({
            "pair": pair,
            "pnl": realized_pnl,
            "equity_after": self.current_equity
        })
