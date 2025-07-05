"""
TensorBoard logging utilities for btcbot project.

This module provides the TensorboardLogger class for centralized TensorBoard logging operations.
"""

from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter
import torch
import logging
import numpy as np
import os


class TensorboardLogger:
    """
    Clase para manejar todas las operaciones de logging de TensorBoard en local.
    Escribe logs únicamente en el sistema de archivos local.
    """
    
    def __init__(self, log_dir: str, run_id: str = None) -> None:
        """
        Inicializa el logger de TensorBoard para operación local.
        
        Args:
            log_dir: Directorio base para los logs de TensorBoard.
            run_id: ID único del entrenamiento, usado como subdirectorio. 
                   Si es None, se usa log_dir directamente.
        """
        self.logger = logging.getLogger(__name__)
        
        if run_id is not None:
            # Crear la ruta completa combinando log_dir con run_id
            self.full_log_path = os.path.join(log_dir, run_id)
        else:
            # Usar log_dir directamente (útil para evaluaciones)
            self.full_log_path = log_dir
        
        # Crear el directorio si no existe
        os.makedirs(self.full_log_path, exist_ok=True)
        
        # Inicializar el SummaryWriter con la ruta completa
        self.writer = SummaryWriter(log_dir=self.full_log_path)
        
        self.logger.info(f"TensorBoard configurado para logging local en: {self.full_log_path}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            hparams: Dictionary containing hyperparameters to log
        """
        # Log hyperparameters as text for now (can be linked to metrics later)
        self.writer.add_text('Hyperparameters', str(hparams), 0)
    
    def log_step_metrics(self, step: int, losses: Dict[str, float], alpha: float) -> None:
        """
        Log training step metrics (losses and alpha value).
        
        Args:
            step: Current learning step
            losses: Dictionary containing loss values (actor_loss, critic_1_loss, critic_2_loss, alpha_loss)
            alpha: Current alpha value
        """
        # Group all loss metrics in a single TensorBoard graph for better comparability
        loss_dict = {
            'Actor': losses['actor_loss'],
            'Critic_1': losses['critic_1_loss'],
            'Critic_2': losses['critic_2_loss'],
            'Alpha': losses['alpha_loss']
        }
        self.writer.add_scalars('Losses/Step', loss_dict, step)
        
        # Log alpha value with cleaner tag hierarchy
        self.writer.add_scalar('Agent/Alpha', alpha, step)
    
    def log_episode_metrics(self, episode: int, episode_return: float, profit_pct: float, 
                          episode_length: int, trade_metrics: Dict[str, Any], 
                          env_metrics: Dict[str, Any]) -> None:
        """
        Log episode-level metrics organized under Performance/Train hierarchy.
        
        Args:
            episode: Episode number
            episode_return: Total return for the episode
            profit_pct: Profit percentage for the episode
            episode_length: Length of the episode in steps
            trade_metrics: Dictionary containing trade-related metrics
            env_metrics: Dictionary containing environment-related metrics
        """
        # Performance/Train/Episode - General episode metrics
        self.writer.add_scalar('Performance/Train/Episode/Return', episode_return, episode)
        self.writer.add_scalar('Performance/Train/Episode/Profit_Pct', profit_pct, episode)
        self.writer.add_scalar('Performance/Train/Episode/Length', episode_length, episode)
        
        # Performance/Train/Trading - Trading activity metrics
        # Group trade counts in a single chart for better comparison
        trade_counts = {}
        if trade_metrics.get('trades_count') is not None:
            trade_counts['Total'] = trade_metrics['trades_count']
        if trade_metrics.get('winning_trades') is not None:
            trade_counts['Winning'] = trade_metrics['winning_trades']
        if trade_metrics.get('long_trades') is not None:
            trade_counts['Long'] = trade_metrics['long_trades']
        if trade_metrics.get('short_trades') is not None:
            trade_counts['Short'] = trade_metrics['short_trades']
        
        if trade_counts:
            self.writer.add_scalars('Performance/Train/Trading/Counts', trade_counts, episode)
        
        # Individual trading metrics
        if trade_metrics.get('total_pnl_realized_abs') is not None:
            self.writer.add_scalar('Performance/Train/Trading/Total_PnL_Realized', trade_metrics['total_pnl_realized_abs'], episode)
        if trade_metrics.get('total_roe_realizado') is not None:
            self.writer.add_scalar('Performance/Train/Trading/Total_ROE_Realizado', trade_metrics['total_roe_realizado'], episode)
        if trade_metrics.get('avg_margin_used') is not None:
            self.writer.add_scalar('Performance/Train/Trading/Avg_Margin_Used', trade_metrics['avg_margin_used'], episode)
        
        # Performance/Train/Portfolio - Portfolio health metrics
        if env_metrics.get('drawdown_episode') is not None:
            self.writer.add_scalar('Performance/Train/Portfolio/Drawdown_Pct', env_metrics['drawdown_episode'] * 100, episode)
        if env_metrics.get('final_balance') is not None:
            self.writer.add_scalar('Performance/Train/Portfolio/Equity', env_metrics['final_balance'], episode)
    
    def log_per_trade_metrics(self, trade_counter: int, trade_data: Dict[str, Any]) -> None:
        """
        Log individual trade metrics.
        
        Args:
            trade_counter: Global trade counter
            trade_data: Dictionary containing trade information
        """
        if 'roe' in trade_data:
            self.writer.add_scalar('Per_Trade/PNL_Realized_ROE', trade_data['roe'], trade_counter)
        if 'pnl_abs' in trade_data:
            self.writer.add_scalar('Per_Trade/PNL_Realized_Absolute', trade_data['pnl_abs'], trade_counter)
        if 'pasos_duracion' in trade_data:
            self.writer.add_scalar('Per_Trade/Duration_Steps', trade_data['pasos_duracion'], trade_counter)
        if 'margen_usado' in trade_data:
            self.writer.add_scalar('Per_Trade/Margin_Used', trade_data['margen_usado'], trade_counter)
        if 'tipo' in trade_data:
            direction_val = 1 if trade_data['tipo'] == 'LARGO' else (-1 if trade_data['tipo'] == 'CORTO' else 0)
            self.writer.add_scalar('Per_Trade/Direction', direction_val, trade_counter)
    
    def log_evaluation_metrics(self, episode: int, eval_metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics organized under Performance/Evaluation hierarchy.
        
        Args:
            episode: Current episode/step number
            eval_metrics: Dictionary containing evaluation metrics
        """
        # Performance/Evaluation/Key_Metrics - Core performance indicators
        key_metrics = {}
        if eval_metrics.get('mean_return') is not None:
            key_metrics['Mean_Return'] = eval_metrics['mean_return']
        if eval_metrics.get('sharpe_ratio') is not None:
            key_metrics['Sharpe_Ratio'] = eval_metrics['sharpe_ratio']
        if eval_metrics.get('sortino_ratio') is not None:
            key_metrics['Sortino_Ratio'] = eval_metrics['sortino_ratio']
        
        if key_metrics:
            self.writer.add_scalars('Performance/Evaluation/Key_Metrics', key_metrics, episode)
        
        # Performance/Evaluation/Risk - Risk assessment metrics
        risk_metrics = {}
        if eval_metrics.get('max_drawdown') is not None:
            risk_metrics['Max_Drawdown_Pct'] = eval_metrics['max_drawdown'] * 100
        if eval_metrics.get('std_return') is not None:
            risk_metrics['Volatility'] = eval_metrics['std_return']
        
        if risk_metrics:
            self.writer.add_scalars('Performance/Evaluation/Risk', risk_metrics, episode)
        
        # Performance/Evaluation/Trading - Trading activity metrics
        trading_metrics = {}
        if eval_metrics.get('win_rate') is not None:
            trading_metrics['Win_Rate_Pct'] = eval_metrics['win_rate'] * 100
        if eval_metrics.get('total_trades') is not None:
            trading_metrics['Avg_Trades'] = eval_metrics['total_trades']
        
        if trading_metrics:
            self.writer.add_scalars('Performance/Evaluation/Trading', trading_metrics, episode)
        
        # Additional individual metrics under Performance/Evaluation
        if eval_metrics.get('mean_profit_pct') is not None:
            self.writer.add_scalar('Performance/Evaluation/Mean_Profit_Pct', eval_metrics['mean_profit_pct'], episode)
        if eval_metrics.get('std_profit_pct') is not None:
            self.writer.add_scalar('Performance/Evaluation/Std_Profit_Pct', eval_metrics['std_profit_pct'], episode)
        if eval_metrics.get('mean_episode_length') is not None:
            self.writer.add_scalar('Performance/Evaluation/Mean_Episode_Length', eval_metrics['mean_episode_length'], episode)
    
    def log_evaluation_summary(self, hparams: dict, final_metrics: dict, equity_curve: list, trade_pnl_list: list) -> None:
        """
        Registra un resumen completo de la evaluación final en TensorBoard.
        
        Este método es el punto de entrada principal para registrar todos los resultados
        de la evaluación final, incluyendo curva de equity, distribución de PnL,
        métricas finales y la vinculación con hiperparámetros.
        
        Args:
            hparams: Diccionario con los hiperparámetros del experimento
            final_metrics: Diccionario con todas las métricas finales calculadas
            equity_curve: Lista con la serie temporal completa del equity
            trade_pnl_list: Lista con los PnL de todos los trades ejecutados
        """
        if self.writer is None:
            self.logger.warning("TensorBoard writer no está inicializado. No se pueden registrar métricas.")
            return
        
        # 1. Registrar la curva de equity como serie temporal
        if equity_curve:
            for step, equity_value in enumerate(equity_curve):
                self.writer.add_scalar('Evaluation/Final_Equity_Curve', equity_value, step)
        
        # 2. Registrar la distribución de PnL de trades como histograma
        if trade_pnl_list:
            # Convertir a numpy array para el histograma
            trade_pnl_array = np.array(trade_pnl_list)
            self.writer.add_histogram('Evaluation/Trade_PNL_Distribution', trade_pnl_array, 0)
        
        # 3. Crear un resumen de texto con las métricas finales
        metrics_summary = self._format_metrics_summary(final_metrics)
        self.writer.add_text('Evaluation/Final_Summary_Metrics', metrics_summary, 0)
        
        # 4. Registrar la vinculación entre hiperparámetros y métricas finales
        # Esto es crucial para comparar experimentos en TensorBoard
        self.writer.add_hparams(hparams, final_metrics)
        
        # 5. Asegurar que todos los datos se escriban
        self.writer.flush()
        
        self.logger.info("✅ Resumen de evaluación final registrado en TensorBoard.")
    
    def _format_metrics_summary(self, metrics: dict) -> str:
        """
        Formatea el diccionario de métricas en un texto legible para TensorBoard.
        
        Args:
            metrics: Diccionario con las métricas a formatear
            
        Returns:
            str: Texto formateado con las métricas
        """
        lines = ["## 📊 Resumen de Evaluación Final", ""]
        
        # Agrupar métricas por categorías
        basic_metrics = ['total_return', 'total_profit_pct', 'total_steps', 'initial_equity', 'final_equity']
        risk_metrics = ['max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'volatility']
        trading_metrics = ['total_trades', 'successful_trades', 'win_rate', 'mean_roe', 'total_pnl_abs']
        
        # Métricas básicas
        lines.append("### 🎯 Métricas Básicas")
        for metric in basic_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    lines.append(f"- **{metric}**: {value}")
        lines.append("")
        
        # Métricas de riesgo
        lines.append("### ⚠️ Métricas de Riesgo")
        for metric in risk_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    lines.append(f"- **{metric}**: {value}")
        lines.append("")
        
        # Métricas de trading
        lines.append("### 💰 Métricas de Trading")
        for metric in trading_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    lines.append(f"- **{metric}**: {value}")
        lines.append("")
        
        # Otras métricas no categorizadas
        categorized = set(basic_metrics + risk_metrics + trading_metrics)
        other_metrics = {k: v for k, v in metrics.items() if k not in categorized}
        
        if other_metrics:
            lines.append("### 📈 Otras Métricas")
            for metric, value in other_metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{metric}**: {value:.4f}")
                else:
                    lines.append(f"- **{metric}**: {value}")
        
        return "\n".join(lines)

    def close(self) -> None:
        """Cierra el writer de TensorBoard y limpia los recursos."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            self.logger.info("TensorBoard writer cerrado y logs enviados.")
    
    def log_agent_distributions(self, episode: int, actions: list, q_values: list) -> None:
        """
        Log agent behavior distributions (actions and Q-values).
        
        Args:
            episode: Current episode number
            actions: List of action tensors from the episode
            q_values: List of Q-value tensors from the episode
        """
        if actions:
            # Concatenate all actions and convert to numpy for histogram
            all_actions = np.concatenate([action.cpu().numpy().flatten() for action in actions])
            self.writer.add_histogram('Distributions/Actions', all_actions, episode)
        
        if q_values:
            # Concatenate all Q-values and convert to numpy for histogram
            all_q_values = np.concatenate([q_val.cpu().numpy().flatten() for q_val in q_values])
            self.writer.add_histogram('Distributions/Q_Values', all_q_values, episode)
    
    def log_buffer_stats(self, episode: int, buffer_size: int, buffer_capacity: int) -> None:
        """
        Log replay buffer statistics.
        
        Args:
            episode: Current episode number
            buffer_size: Current number of experiences in buffer
            buffer_capacity: Maximum capacity of the buffer
        """
        # Calculate fill percentage
        fill_percentage = (buffer_size / buffer_capacity) * 100 if buffer_capacity > 0 else 0
        
        # Log buffer metrics
        self.writer.add_scalar('Agent/Buffer/Fill_Percentage', fill_percentage, episode)
        self.writer.add_scalar('Agent/Buffer/Size', buffer_size, episode)
