from src.live.portfolio_manager import LivePortfolioManager


class RiskManager:
    """
    Supervisa el riesgo del portfolio de forma independiente, principalmente el drawdown.
    Actúa como un freno de emergencia si se superan los umbrales de riesgo.
    """
    def __init__(self, portfolio_manager: LivePortfolioManager, max_drawdown_pct: float):
        """
        Inicializa el gestor de riesgo.

        Args:
            portfolio_manager: El gestor del portfolio para obtener el estado actual.
            max_drawdown_pct: El drawdown máximo permitido como un flotante (ej. 0.2 para 20%).
        """
        self.portfolio_manager = portfolio_manager
        self.max_drawdown_pct = max_drawdown_pct
        self.initial_equity = 0.0
        self.max_equity_so_far = 0.0
        self.is_initialized = False
        print(f"RiskManager inicializado con un max_drawdown_pct de {max_drawdown_pct:.2%}")

    def update_state(self, current_equity: float):
        """
        Actualiza el estado del RiskManager con el equity más reciente.
        Este método debe ser llamado en cada ciclo del bot.
        """
        if not self.is_initialized:
            # La primera vez que se llama, se establece el equity base.
            self.initial_equity = current_equity
            self.max_equity_so_far = current_equity
            self.is_initialized = True
            print(f"RiskManager: Equity inicial establecido en {current_equity:.2f} USDT.")
            return

        # Actualizar el equity máximo alcanzado hasta ahora.
        if current_equity > self.max_equity_so_far:
            self.max_equity_so_far = current_equity

    def is_risk_threshold_exceeded(self) -> bool:
        """
        Verifica si el drawdown actual ha superado el umbral máximo permitido.

        Returns:
            True si el riesgo es demasiado alto, False en caso contrario.
        """
        if not self.is_initialized:
            return False

        current_equity = self.portfolio_manager.balance # Asumimos que balance es el equity actual
        
        # Evitar división por cero
        if self.max_equity_so_far == 0:
            return False

        # Calcular el drawdown actual
        drawdown = (self.max_equity_so_far - current_equity) / self.max_equity_so_far

        if drawdown > self.max_drawdown_pct:
            print(f"🚨 ALERTA DE RIESGO: Drawdown ({drawdown:.2%}) ha superado el umbral de {self.max_drawdown_pct:.2%}")
            return True

        return False
