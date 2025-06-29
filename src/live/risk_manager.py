from src.live.portfolio_manager import LivePortfolioManager


class RiskManager:
    def __init__(self, portfolio_manager: LivePortfolioManager, max_drawdown_pct: float):
        self.portfolio_manager = portfolio_manager
        self.max_drawdown_pct = max_drawdown_pct
        self.initial_equity = None
        self.max_equity_so_far = 0.0
    
    def update_state(self, current_equity: float):
        pass
    
    def is_risk_threshold_exceeded(self) -> bool:
        return False
