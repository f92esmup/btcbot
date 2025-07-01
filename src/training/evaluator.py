"""
Agent evaluator module for btcbot.

This module contains the AgentEvaluator class which encapsulates
the logic for evaluating the performance of trading agents.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from src.agente.agent import TransformerSACAgent
from src.entorno.environment import FuturesTradingEnv
from src.analysis.metrics import FinancialMetrics
from src.utils.observation_parser import parse_observation


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
    
    def get_trades_dataframe(self, trades_history: list) -> pd.DataFrame:
        """
        Convierte el historial de trades a formato DataFrame para análisis.
        
        Args:
            trades_history: Lista de diccionarios con información de trades
            
        Returns:
            pd.DataFrame: DataFrame con los trades para análisis
        """
        if not trades_history:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'tipo', 'precio_entrada', 'precio_salida', 'tamaño_activo',
                'margen_usado', 'pnl_abs', 'roe', 'pasos_duracion', 'paso_cierre'
            ])
        
        # Convert trades list to DataFrame
        df_trades = pd.DataFrame(trades_history)
        
        # Add derived metrics
        if len(df_trades) > 0:
            # Calculate profit/loss flag
            df_trades['is_profitable'] = df_trades['pnl_abs'] > 0
            
            # Calculate price change percentage
            df_trades['price_change_pct'] = (
                (df_trades['precio_salida'] - df_trades['precio_entrada']) / 
                df_trades['precio_entrada'] * 100
            )
            
            # Add trade direction flag (1 for LARGO, -1 for CORTO)
            df_trades['direccion_num'] = df_trades['tipo'].map({'LARGO': 1, 'CORTO': -1})
            
            # Calculate effective ROE considering direction
            df_trades['roe_efectivo'] = df_trades['roe'] * df_trades['direccion_num']
        
        return df_trades
    
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
        all_trades_data = []  # Todos los trades para análisis
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0
            episode_length = 0
            initial_balance = env.portfolio.balance_actual
            initial_equity = env.portfolio.equity_actual
            
            # Track equity durante el episodio
            episode_equity_track = [initial_equity]
            
            # Store initial trade count to calculate episode trades
            initial_trade_count = len(env.portfolio.historial_trades)
            
            done = False
            while not done:
                # Parse observation and select action
                market_data, portfolio_data = parse_observation(obs, env.config_entorno, agent.device)
                action = agent.select_action(market_data, portfolio_data, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                
                # Registrar equity en cada paso
                episode_equity_track.append(env.portfolio.equity_actual)
                
                # Contar trades
                if 'trade_ejecutado' in info and info['trade_ejecutado']:
                    total_trades += 1
                    if reward > 0:
                        successful_trades += 1
            
            final_balance = env.portfolio.balance_actual
            final_equity = env.portfolio.equity_actual
            profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
            
            episode_returns.append(episode_return)
            episode_profits.append(profit_pct)
            episode_lengths.append(episode_length)
            
            # Guardar series de equity
            episode_equity_series.extend(episode_equity_track)
            all_equity_values.append(final_equity)
            
            # Episode Summary Analysis - Extract trades from this episode
            episode_trades = env.portfolio.historial_trades[initial_trade_count:]
            if episode_trades:
                all_trades_data.extend(episode_trades)
        
        agent.train_mode()
        
        # Episode Summary Analysis using environment histories
        episode_summary = self._calculate_episode_summary(
            env.portfolio.historial_trades, 
            env.historial_equity,
            all_trades_data
        )
        
        # Calcular métricas financieras avanzadas usando la instancia almacenada
        max_drawdown = self.metrics_calculator.calculate_max_drawdown(episode_equity_series)
        sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(episode_profits)
        sortino_ratio = self.metrics_calculator.calculate_sortino_ratio(episode_profits)
        
        # Métricas básicas
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
        
        # Combinar con análisis de resumen de episodio
        metrics.update(episode_summary)
        
        return metrics
    
    def _calculate_episode_summary(self, historial_trades: list, 
                                 historial_equity: list, 
                                 all_trades_data: list) -> Dict[str, float]:
        """
        Calcula estadísticas de resumen de episodio usando los historiales del entorno.
        
        Args:
            historial_trades: Historial completo de trades del entorno
            historial_equity: Historial completo de equity del entorno
            all_trades_data: Todos los trades recopilados durante la evaluación
            
        Returns:
            Dict[str, float]: Métricas de resumen de episodio
        """
        summary = {}
        
        # Análisis de equity
        if historial_equity:
            equity_series = pd.Series(historial_equity)
            summary.update({
                'final_equity': equity_series.iloc[-1] if len(equity_series) > 0 else 0.0,
                'max_equity_reached': equity_series.max(),
                'min_equity_reached': equity_series.min(),
                'equity_volatility': equity_series.std(),
                'equity_trend': (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0] if len(equity_series) > 1 and equity_series.iloc[0] != 0 else 0.0
            })
        
        # Análisis detallado de trades usando DataFrame
        if all_trades_data:
            trades_df = self.get_trades_dataframe(all_trades_data)
            
            if len(trades_df) > 0:
                # Métricas de trading
                summary.update({
                    'total_trades_analysis': len(trades_df),
                    'profitable_trades': trades_df['is_profitable'].sum(),
                    'losing_trades': (~trades_df['is_profitable']).sum(),
                    'win_rate_analysis': trades_df['is_profitable'].mean(),
                    
                    # Métricas de ROE
                    'mean_roe': trades_df['roe'].mean(),
                    'std_roe': trades_df['roe'].std(),
                    'max_roe': trades_df['roe'].max(),
                    'min_roe': trades_df['roe'].min(),
                    
                    # Métricas de PNL
                    'mean_pnl_abs': trades_df['pnl_abs'].mean(),
                    'std_pnl_abs': trades_df['pnl_abs'].std(),
                    'total_pnl_abs': trades_df['pnl_abs'].sum(),
                    
                    # Métricas de duración
                    'mean_trade_duration': trades_df['pasos_duracion'].mean(),
                    'std_trade_duration': trades_df['pasos_duracion'].std(),
                    'max_trade_duration': trades_df['pasos_duracion'].max(),
                    
                    # Métricas de margen
                    'mean_margin_used': trades_df['margen_usado'].mean(),
                    'total_margin_used': trades_df['margen_usado'].sum(),
                    
                    # Análisis por dirección
                    'long_trades_count': (trades_df['tipo'] == 'LARGO').sum(),
                    'short_trades_count': (trades_df['tipo'] == 'CORTO').sum(),
                    'long_trades_win_rate': trades_df[trades_df['tipo'] == 'LARGO']['is_profitable'].mean() if (trades_df['tipo'] == 'LARGO').any() else 0.0,
                    'short_trades_win_rate': trades_df[trades_df['tipo'] == 'CORTO']['is_profitable'].mean() if (trades_df['tipo'] == 'CORTO').any() else 0.0,
                })
                
                # Análisis de consecutividad
                trades_df_sorted = trades_df.sort_values('paso_cierre')
                if len(trades_df_sorted) > 1:
                    # Consecutive wins/losses
                    consecutive_analysis = self._analyze_consecutive_trades(trades_df_sorted)
                    summary.update(consecutive_analysis)
        
        # Si no hay trades, llenar con valores por defecto
        if not all_trades_data:
            summary.update({
                'total_trades_analysis': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate_analysis': 0.0,
                'mean_roe': 0.0,
                'std_roe': 0.0,
                'max_roe': 0.0,
                'min_roe': 0.0,
                'mean_pnl_abs': 0.0,
                'std_pnl_abs': 0.0,
                'total_pnl_abs': 0.0,
                'mean_trade_duration': 0.0,
                'std_trade_duration': 0.0,
                'max_trade_duration': 0.0,
                'mean_margin_used': 0.0,
                'total_margin_used': 0.0,
                'long_trades_count': 0,
                'short_trades_count': 0,
                'long_trades_win_rate': 0.0,
                'short_trades_win_rate': 0.0,
            })
        
        return summary
    
    def _analyze_consecutive_trades(self, trades_df_sorted: pd.DataFrame) -> Dict[str, float]:
        """
        Analiza patrones de trades consecutivos.
        
        Args:
            trades_df_sorted: DataFrame de trades ordenado por paso de cierre
            
        Returns:
            Dict[str, float]: Métricas de consecutividad
        """
        if len(trades_df_sorted) < 2:
            return {
                'max_consecutive_wins': 0.0,
                'max_consecutive_losses': 0.0,
                'current_streak': 0.0
            }
        
        # Analizar rachas consecutivas
        is_profitable = trades_df_sorted['is_profitable'].values
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for profitable in is_profitable:
            if profitable:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Racha actual
        current_streak = current_wins if current_wins > 0 else -current_losses
        
        return {
            'max_consecutive_wins': float(max_wins),
            'max_consecutive_losses': float(max_losses),
            'current_streak': float(current_streak)
        }
