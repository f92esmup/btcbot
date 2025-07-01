from src.entorno.base_portfolio import BasePortfolio


class RiskManager:
    """
    Supervisa el riesgo del portfolio de forma independiente, principalmente el drawdown.
    Actúa como un freno de emergencia si se superan los umbrales de riesgo.
    """
    def __init__(self, portfolio_manager: BasePortfolio, risk_config: dict):
        """
        Inicializa el gestor de riesgo.

        Args:
            portfolio_manager: El gestor del portfolio para obtener el estado actual.
            risk_config: Diccionario con la configuración de riesgo.
        """
        self.portfolio_manager = portfolio_manager
        self.max_drawdown_pct = risk_config['max_drawdown_configurado_cuenta']
        self.max_consecutive_losses = risk_config['max_consecutive_losses']
        self.consecutive_losses_counter = 0
        self.initial_equity = 0.0
        self.max_equity_so_far = 0.0
        self.is_initialized = False
        print(f"RiskManager inicializado con un max_drawdown_pct de {self.max_drawdown_pct:.2%}")
        print(f"Contador de pérdidas consecutivas inicializado (máximo permitido: {self.max_consecutive_losses})")

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

    def register_trade(self, pnl_realizado: float):
        """
        Registra el resultado de una operación cerrada y actualiza el contador de pérdidas consecutivas.
        
        Args:
            pnl_realizado: El PnL realizado de la operación (positivo para ganancias, negativo para pérdidas).
        """
        if pnl_realizado > 0:
            # Ganancia: reiniciar el contador de pérdidas consecutivas
            self.consecutive_losses_counter = 0
        else:
            # Pérdida o trade sin ganancia: incrementar el contador
            self.consecutive_losses_counter += 1
        
        print(f"Racha de pérdidas consecutivas actualizada: {self.consecutive_losses_counter}")

    def is_risk_threshold_exceeded(self) -> bool:
        """
        Verifica si el drawdown actual ha superado el umbral máximo permitido.

        Returns:
            True si el riesgo es demasiado alto, False en caso contrario.
        """
        # Verificar pérdidas consecutivas
        if self.consecutive_losses_counter >= self.max_consecutive_losses:
            print(f"🚨 ALERTA DE RIESGO: Se ha alcanzado el número máximo de pérdidas consecutivas ({self.consecutive_losses_counter}/{self.max_consecutive_losses})")
            return True
            
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
