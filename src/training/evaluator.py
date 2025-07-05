"""
Agent evaluator module for btcbot.

This module contains the AgentEvaluator class which implements
a comprehensive final backtesting engine for trading agents,
executing a single complete backtest over the entire available dataset
and returning detailed performance metrics along with the equity curve.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from src.agente.agent import TransformerSACAgent
from src.entorno.environment import FuturesTradingEnv
from src.analysis.metrics import FinancialMetrics
from src.agente.observation_parser import parse_observation


class AgentEvaluator:
    """
    Motor de backtesting final y exhaustivo para evaluar agentes de trading.
    
    Esta clase se encarga de ejecutar un único y completo backtest sobre todo 
    el dataset disponible en el entorno, calculando métricas financieras 
    avanzadas como máximo drawdown, Sharpe ratio, Sortino ratio y devolviendo 
    la curva de equity completa del backtest.
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
    
    def evaluate(self, agent: TransformerSACAgent, env: FuturesTradingEnv) -> Tuple[Dict[str, float], List[float], List[float]]:
        """
        Realiza un backtest final y exhaustivo del agente sobre todo el dataset disponible.
        
        Args:
            agent: Agente a evaluar
            env: Entorno de trading
            
        Returns:
            Tuple[Dict[str, float], List[float], List[float]]: 
                - Diccionario con métricas finales de rendimiento
                - Lista con la serie temporal completa del equity
                - Lista con los PnL absolutos de todos los trades
        """
        agent.eval_mode()
        
        # Inicializar el entorno para el backtest completo
        obs, _ = env.reset()
        
        # Variables para el tracking del backtest
        total_return = 0
        step_count = 0
        successful_trades = 0
        total_trades = 0
        
        # Valores iniciales
        initial_balance = env.portfolio.balance
        initial_equity = env.portfolio.equity
        
        # Serie temporal del equity para la curva completa
        equity_curve = [initial_equity]
        
        # Tracking de retornos por paso para métricas financieras
        step_returns = []
        
        # Ejecutar el backtest completo en un solo episodio
        done = False
        while not done:
            # Parse observation and select action deterministically
            market_data, portfolio_data = parse_observation(obs, env.config_entorno, agent.config, agent.device)
            action = agent.select_action(market_data, portfolio_data, deterministic=True)
            
            # Equity antes del paso para calcular retorno
            prev_equity = env.portfolio.equity
            
            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Actualizar métricas
            total_return += reward
            step_count += 1
            
            # Calcular retorno del paso
            current_equity = env.portfolio.equity
            if prev_equity > 0:
                step_return = (current_equity - prev_equity) / prev_equity
                step_returns.append(step_return)
            
            # Registrar equity en cada paso
            equity_curve.append(current_equity)
            
            # Contar trades ejecutados
            if 'trade_ejecutado' in info and info['trade_ejecutado']:
                total_trades += 1
                if reward > 0:
                    successful_trades += 1
        
        agent.train_mode()
        
        # Calcular métricas finales del backtest
        final_balance = env.portfolio.balance
        final_equity = env.portfolio.equity
        
        # Calcular el rendimiento total
        total_profit_pct = ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0.0
        
        # Calcular métricas financieras avanzadas usando la serie de equity
        max_drawdown = self.metrics_calculator.calculate_max_drawdown(equity_curve)
        
        # Para Sharpe y Sortino, usar retornos por paso
        sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(step_returns) if step_returns else 0.0
        sortino_ratio = self.metrics_calculator.calculate_sortino_ratio(step_returns) if step_returns else 0.0
        
        # Calcular estadísticas de los retornos
        returns_mean = np.mean(step_returns) if step_returns else 0.0
        returns_std = np.std(step_returns) if step_returns else 0.0
        
        # Calcular métricas de trading detalladas
        trades_summary = self._calculate_backtest_summary(
            env.portfolio.historial_trades, 
            env.portfolio.historial_equity
        )
        
        # Construir el diccionario de métricas finales con valores por defecto
        final_metrics = {
            # Métricas básicas del backtest
            'total_return': total_return,
            'total_profit_pct': total_profit_pct,
            'total_steps': step_count,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            
            # Métricas de retornos
            'mean_return': returns_mean,
            'std_return': returns_std,
            'mean_step_return': returns_mean,
            'std_step_return': returns_std,
            'volatility': returns_std * np.sqrt(252) if returns_std > 0 else 0.0,  # Anualizada
            
            # Métricas de riesgo
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            
            # Métricas de trading
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'win_rate': successful_trades / max(total_trades, 1),
            
            # Métricas de equity
            'max_equity': max(equity_curve) if equity_curve else initial_equity,
            'min_equity': min(equity_curve) if equity_curve else initial_equity,
            'equity_volatility': np.std(equity_curve) if len(equity_curve) > 1 else 0.0,
        }
        
        # Añadir métricas que podrían no existir si no hay trades
        final_metrics.update({
            'mean_profit_pct': final_metrics.get('total_profit_pct', 0.0), # Usar el total como fallback
            'std_profit_pct': np.std([t['roe'] for t in env.portfolio.historial_trades]) if env.portfolio.historial_trades else 0.0,
            'mean_episode_length': step_count # Para un solo backtest, es la longitud total
        })

        # Combinar con análisis detallado de trades
        final_metrics.update(trades_summary)
        
        # Extraer lista de PnL para análisis de distribución
        trade_pnl_list = self.get_trade_pnl_list(env.portfolio.historial_trades)
        
        return final_metrics, equity_curve, trade_pnl_list
    
    def _calculate_backtest_summary(self, historial_trades: list, 
                                   historial_equity: list) -> Dict[str, float]:
        """
        Calcula estadísticas de resumen del backtest final usando los historiales del entorno.
        
        Args:
            historial_trades: Historial completo de trades del entorno
            historial_equity: Historial completo de equity del entorno
            
        Returns:
            Dict[str, float]: Métricas detalladas del backtest
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
        if historial_trades:
            trades_df = self.get_trades_dataframe(historial_trades)
            
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
        if not historial_trades:
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
    
    def get_trade_pnl_list(self, historial_trades: list) -> list:
        """
        Extrae una lista de PnL absolutos de todos los trades para análisis de distribución.
        
        Args:
            historial_trades: Lista de diccionarios con información de trades
            
        Returns:
            list: Lista con los PnL absolutos de todos los trades
        """
        if not historial_trades:
            return []
        
        # Extraer solo los valores de pnl_abs
        pnl_list = [trade.get('pnl_abs', 0.0) for trade in historial_trades]
        return pnl_list
