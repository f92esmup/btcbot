"""
Módulo para definir estrategias de cálculo de recompensas.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.analysis.metrics import FinancialMetrics

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
        recompensa_paso *= self.config.peso_recompensa_paso

        # Componente por cierre de operación
        recompensa_cierre = 0.0
        if trade_ejecutado and pnl_realizado != 0.0 and portfolio.historial_trades:
            ultimo_trade = portfolio.historial_trades[-1]
            roe_operacion = ultimo_trade['roe']
            
            # La lógica de log1p ha sido eliminada. La normalización de recompensas se maneja en el agente.
            recompensa_cierre = roe_operacion
            
            recompensa_cierre *= self.config.peso_recompensa_cierre

        return float(recompensa_paso + recompensa_cierre)


class SortinoRewardStrategy(BaseRewardStrategy):
    """
    Estrategia de recompensa basada en el Sortino Ratio diferencial.
    
    Esta estrategia calcula la recompensa como la diferencia entre el Sortino Ratio
    actual y el Sortino Ratio del período anterior, incentivando mejoras en el
    ratio riesgo-retorno ajustado por downside risk.
    """
    
    def calculate_reward(self, portfolio, trade_ejecutado: bool, pnl_realizado: float, equity_anterior: float) -> float:
        """
        Calcula la recompensa basada en la diferencia del Sortino Ratio.
        
        Args:
            portfolio: La instancia del portfolio con el estado actual.
            trade_ejecutado (bool): Si se ejecutó un trade en este paso.
            pnl_realizado (float): PNL realizado si se cerró una posición.
            equity_anterior (float): Equity del paso anterior.
            
        Returns:
            float: La recompensa diferencial basada en Sortino Ratio.
        """
        # Obtener la serie de equity del portfolio
        equity_series = portfolio.get_equity_series()
        
        # Calcular los retornos porcentuales
        returns = equity_series.pct_change().dropna()
        
        # Manejar casos borde: si hay menos de 2 retornos, no se puede calcular desviación estándar
        if len(returns) < 2:
            return 0.0
        
        # Calcular el Sortino Ratio actual (con todos los retornos)
        sortino_actual = FinancialMetrics.calculate_sortino_ratio(returns.tolist())
        
        # Calcular el Sortino Ratio anterior (sin el último retorno)
        sortino_anterior = FinancialMetrics.calculate_sortino_ratio(returns.iloc[:-1].tolist())
        
        # Manejar valores infinitos y NaN, convirtiendo a números manejables
        sortino_actual_clean = np.nan_to_num(sortino_actual, nan=0.0, posinf=100.0, neginf=-100.0)
        sortino_anterior_clean = np.nan_to_num(sortino_anterior, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Calcular la recompensa diferencial
        recompensa = sortino_actual_clean - sortino_anterior_clean
        
        return float(recompensa)