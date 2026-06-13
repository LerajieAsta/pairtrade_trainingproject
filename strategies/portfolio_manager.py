class PortfolioManager:
    def __init__(self, strategy_id: str, initial_capital: float = 10000.0, max_pairs: int = 10):
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.max_pairs = max_pairs
        self.active_pairs = {} # dict of (ticker_a, ticker_b) -> allocated_capital
        self.history = []

    def get_available_slots(self) -> int:
        return max(0, self.max_pairs - len(self.active_pairs))

    def allocate_capital(self, candidate_pairs: list) -> dict:
        """
        Allocates capital to candidate pairs based on available slots.
        candidate_pairs: list of tuples (Ticker_A, Ticker_B)
        Returns a dict: {(Ticker_A, Ticker_B): allocated_amount}
        """
        allocations = {}
        slots = self.get_available_slots()
        
        if slots == 0 or self.current_equity <= 0:
            return allocations
            
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
