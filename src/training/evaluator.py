"""
Agent evaluator module for btcbot.

This module contains the AgentEvaluator class which encapsulates
the logic for evaluating the performance of trading agents.
"""

import numpy as np
import torch
from typing import Dict, List
from src.agente.agent import TransformerSACAgent
from src.entorno.environment import FuturesTradingEnv
from src.analysis.metrics import FinancialMetrics


class AgentEvaluator:
    """
    Clase para encapsular la lógica de evaluación del agente.
    
    Esta clase se encarga de evaluar el rendimiento del agente de trading
    calculando métricas financieras avanzadas como máximo drawdown,
    Sharpe ratio y Sortino ratio.
    """
    
    def __init__(self, metrics_calculator: FinancialMetrics = None):
        """
        Inicializa el evaluador con una instancia del calculador de métricas.
        
        Args:
            metrics_calculator: Instancia de FinancialMetrics para calcular métricas
                              (opcional, se creará una nueva si no se proporciona)
        """
        if metrics_calculator is None:
            self.metrics_calculator = FinancialMetrics()
        else:
            self.metrics_calculator = metrics_calculator
    
    def evaluate(self, agent: TransformerSACAgent, env: FuturesTradingEnv, 
                num_episodes: int) -> Dict[str, float]:
        """
        Evalúa el rendimiento del agente.
        
        Args:
            agent: Agente a evaluar
            env: Entorno de trading
            num_episodes: Número de episodios de evaluación
            
        Returns:
            Dict[str, float]: Métricas de evaluación
        """
        agent.eval_mode()
        
        episode_returns = []
        episode_profits = []
        episode_lengths = []
        successful_trades = 0
        total_trades = 0
        
        # Para métricas financieras avanzadas
        all_equity_values = []  # Para calcular max drawdown
        episode_equity_series = []  # Equity por episodio
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0
            episode_length = 0
            initial_balance = env.balance_actual
            initial_equity = env.equity_actual
            
            # Track equity durante el episodio
            episode_equity_track = [initial_equity]
            
            done = False
            while not done:
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                
                # Registrar equity en cada paso
                episode_equity_track.append(env.equity_actual)
                
                # Contar trades
                if 'trade_executed' in info and info['trade_executed']:
                    total_trades += 1
                    if reward > 0:
                        successful_trades += 1
            
            final_balance = env.balance_actual
            final_equity = env.equity_actual
            profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
            
            episode_returns.append(episode_return)
            episode_profits.append(profit_pct)
            episode_lengths.append(episode_length)
            
            # Guardar series de equity
            episode_equity_series.extend(episode_equity_track)
            all_equity_values.append(final_equity)
        
        agent.train_mode()
        
        # Calcular métricas financieras avanzadas usando la instancia almacenada
        max_drawdown = self.metrics_calculator.calculate_max_drawdown(episode_equity_series)
        sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(episode_profits)
        sortino_ratio = self.metrics_calculator.calculate_sortino_ratio(episode_profits)
        
        metrics = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_profit_pct': np.mean(episode_profits),
            'std_profit_pct': np.std(episode_profits),
            'mean_episode_length': np.mean(episode_lengths),
            'win_rate': successful_trades / max(total_trades, 1),
            'total_trades': total_trades / num_episodes,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
        
        return metrics
