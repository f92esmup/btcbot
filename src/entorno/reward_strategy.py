"""
Módulo para definir estrategias de cálculo de recompensas.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseRewardStrategy(ABC):
    """
    Clase base abstracta para las estrategias de cálculo de recompensas.
    """
    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config

    @abstractmethod
    def calculate_reward(self, portfolio, trade_ejecutado: bool, pnl_realizado: float, equity_anterior: float) -> float:
        """
        Calcula la recompensa para un paso del entorno.

        Args:
            portfolio: La instancia del portfolio con el estado actual.
            trade_ejecutado (bool): Si se ejecutó un trade en este paso.
            pnl_realizado (float): PNL realizado si se cerró una posición.
            equity_anterior (float): Equity del paso anterior.

        Returns:
            float: La recompensa calculada.
        """
        pass

class EquityChangeRewardStrategy(BaseRewardStrategy):
    """
    Estrategia de recompensa basada en el cambio del equity y el PNL de operaciones cerradas.
    """
    def calculate_reward(self, portfolio, trade_ejecutado: bool, pnl_realizado: float, equity_anterior: float) -> float:
        """
        Calcula la recompensa híbrida.
        """
        # Componente por paso: cambio en equity
        recompensa_paso = (portfolio.equity - equity_anterior) / equity_anterior if equity_anterior > 0 else 0.0
        recompensa_paso *= self.config.get('peso_recompensa_paso', 1.0)

        # Componente por cierre de operación
        recompensa_cierre = 0.0
        if trade_ejecutado and pnl_realizado != 0.0 and portfolio.historial_trades:
            ultimo_trade = portfolio.historial_trades[-1]
            roe_operacion = ultimo_trade['roe']
            
            # La lógica de log1p ha sido eliminada. La normalización de recompensas se maneja en el agente.
            recompensa_cierre = roe_operacion
            
            recompensa_cierre *= self.config.get('peso_recompensa_cierre', 1.0)

        return float(recompensa_paso + recompensa_cierre)