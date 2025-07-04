"""
TensorBoard logging utilities for btcbot project.

This module provides the TensorboardLogger class for centralized TensorBoard logging operations.
"""

from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
import os


class TensorboardLogger:
    """
    Clase para manejar todas las operaciones de logging de TensorBoard en local.
    Escribe logs únicamente en el sistema de archivos local.
    """
    
    def __init__(self, log_dir: str, run_id: str) -> None:
        """
        Inicializa el logger de TensorBoard para operación local.
        
        Args:
            log_dir: Directorio base para los logs de TensorBoard.
            run_id: ID único del entrenamiento, usado como subdirectorio.
        """
        self.logger = logging.getLogger(__name__)
        
        # Crear la ruta completa combinando log_dir con run_id
        self.full_log_path = os.path.join(log_dir, run_id)
        
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
        self.writer.add_scalar('Agent/Actor_Loss_step', losses['actor_loss'], step)
        self.writer.add_scalar('Agent/Critic_1_Loss_step', losses['critic_1_loss'], step)
        self.writer.add_scalar('Agent/Critic_2_Loss_step', losses['critic_2_loss'], step)
        self.writer.add_scalar('Agent/Alpha_Loss_step', losses['alpha_loss'], step)
        self.writer.add_scalar('Agent/Alpha_Value_step', alpha, step)
    
    def log_episode_metrics(self, episode: int, episode_return: float, profit_pct: float, 
                          episode_length: int, trade_metrics: Dict[str, Any], 
                          env_metrics: Dict[str, Any]) -> None:
        """
        Log episode-level metrics.
        
        Args:
            episode: Episode number
            episode_return: Total return for the episode
            profit_pct: Profit percentage for the episode
            episode_length: Length of the episode in steps
            trade_metrics: Dictionary containing trade-related metrics
            env_metrics: Dictionary containing environment-related metrics
        """
        # Basic episode metrics
        self.writer.add_scalar('Episode_Metrics/Return', episode_return, episode)
        self.writer.add_scalar('Episode_Metrics/Profit_Percentage_Initial_Capital', profit_pct, episode)
        self.writer.add_scalar('Episode_Metrics/Length', episode_length, episode)
        
        # Trade metrics
        if trade_metrics.get('trades_count') is not None:
            self.writer.add_scalar('Trading_Episode/Total_Trades_Executed', trade_metrics['trades_count'], episode)
        if trade_metrics.get('winning_trades') is not None:
            self.writer.add_scalar('Trading_Episode/Number_Winning_Trades', trade_metrics['winning_trades'], episode)
        if trade_metrics.get('long_trades') is not None:
            self.writer.add_scalar('Trading_Episode/Number_Long_Trades', trade_metrics['long_trades'], episode)
        if trade_metrics.get('short_trades') is not None:
            self.writer.add_scalar('Trading_Episode/Number_Short_Trades', trade_metrics['short_trades'], episode)
        if trade_metrics.get('total_pnl_realized_abs') is not None:
            self.writer.add_scalar('Trading_Episode/Total_PNL_Realized_Absolute', trade_metrics['total_pnl_realized_abs'], episode)
        if trade_metrics.get('total_roe_realizado') is not None:
            self.writer.add_scalar('Trading_Episode/Total_ROE_Realizado', trade_metrics['total_roe_realizado'], episode)
        if trade_metrics.get('avg_margin_used') is not None:
            self.writer.add_scalar('Trading_Episode/Average_Margin_Used_per_Trade', trade_metrics['avg_margin_used'], episode)
        
        # Environment metrics
        if env_metrics.get('drawdown_episode') is not None:
            self.writer.add_scalar('Environment_Episode/Drawdown_Percentage', env_metrics['drawdown_episode'] * 100, episode)
        if env_metrics.get('final_balance') is not None:
            self.writer.add_scalar('Environment_Episode/Final_Balance', env_metrics['final_balance'], episode)
        if env_metrics.get('initial_balance') is not None:
            self.writer.add_scalar('Environment_Episode/Final_Equity', env_metrics['final_balance'], episode)  # Using final_balance as equity for now
    
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
        Log evaluation metrics.
        
        Args:
            episode: Current episode/step number
            eval_metrics: Dictionary containing evaluation metrics
        """
        # Basic evaluation metrics
        if 'mean_return' in eval_metrics:
            self.writer.add_scalar('Evaluation/Mean_Return', eval_metrics['mean_return'], episode)
        if 'std_return' in eval_metrics:
            self.writer.add_scalar('Evaluation/Std_Return', eval_metrics['std_return'], episode)
        if 'mean_profit_pct' in eval_metrics:
            self.writer.add_scalar('Evaluation/Mean_Profit_Pct', eval_metrics['mean_profit_pct'], episode)
        if 'std_profit_pct' in eval_metrics:
            self.writer.add_scalar('Evaluation/Std_Profit_Pct', eval_metrics['std_profit_pct'], episode)
        if 'mean_episode_length' in eval_metrics:
            self.writer.add_scalar('Evaluation/Mean_Episode_Length', eval_metrics['mean_episode_length'], episode)
        if 'win_rate' in eval_metrics:
            self.writer.add_scalar('Evaluation/Win_Rate', eval_metrics['win_rate'] * 100, episode)
        if 'total_trades' in eval_metrics:
            self.writer.add_scalar('Evaluation/Avg_Trades_per_Episode', eval_metrics['total_trades'], episode)
        
        # Financial metrics
        if 'max_drawdown' in eval_metrics:
            self.writer.add_scalar('Evaluation/Max_Drawdown_Pct', eval_metrics['max_drawdown'] * 100, episode)
        if 'sharpe_ratio' in eval_metrics:
            self.writer.add_scalar('Evaluation/Sharpe_Ratio', eval_metrics['sharpe_ratio'], episode)
        if 'sortino_ratio' in eval_metrics:
            self.writer.add_scalar('Evaluation/Sortino_Ratio', eval_metrics['sortino_ratio'], episode)
    
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
