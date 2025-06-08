"""
Financial metrics calculation utilities for btcbot project.

This module provides the FinancialMetrics class with static methods for calculating
various financial performance metrics.
"""

import numpy as np
from typing import List


class FinancialMetrics:
    """
    A class containing static methods for calculating financial performance metrics.
    
    This class provides utilities for calculating various financial metrics such as
    maximum drawdown, Sharpe ratio, and Sortino ratio.
    """
    
    @staticmethod
    def calculate_max_drawdown(equity_series: List[float]) -> float:
        """
        Calcula el máximo drawdown de una serie de equity.
        
        Args:
            equity_series: Lista de valores de equity
            
        Returns:
            float: Máximo drawdown como porcentaje (0.0 - 1.0)
        """
        if len(equity_series) == 0:
            return 0.0
        
        equity_array = np.array(equity_series)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = np.max(drawdown)
        
        return max_drawdown

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calcula el Sharpe Ratio de una serie de retornos.
        
        Args:
            returns: Lista de retornos
            risk_free_rate: Tasa libre de riesgo (anualizada)
            
        Returns:
            float: Sharpe Ratio
        """
        if len(returns) <= 1:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calcula el Sortino Ratio de una serie de retornos.
        
        Args:
            returns: Lista de retornos
            risk_free_rate: Tasa libre de riesgo (anualizada)
            
        Returns:
            float: Sortino Ratio
        """
        if len(returns) <= 1:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        # Solo considerar retornos negativos para la desviación estándar
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            # No hay retornos negativos, se considera muy bueno
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        sortino_ratio = np.mean(excess_returns) / downside_deviation
        return sortino_ratio
