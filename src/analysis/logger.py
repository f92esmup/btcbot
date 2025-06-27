"""
TensorBoard logging utilities for btcbot project.

This module provides the TensorboardLogger class for centralized TensorBoard logging operations.
"""

from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import logging
from ..configuration.config import config

# Importar aiplatform para la integración con Vertex AI
try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None


class TensorboardLogger:
    """
    Clase para manejar todas las operaciones de logging de TensorBoard.
    Soporta logging local y en Vertex AI TensorBoard.
    """
    
    def __init__(self, log_dir: str, run_id: str) -> None:
        """
        Inicializa el logger de TensorBoard.
        
        Args:
            log_dir: Directorio para logs locales o None si es modo GCP.
            run_id: ID único del entrenamiento, usado como nombre del run en TensorBoard.
        """
        self.writer = None
        self.storage_mode = config.storage_mode
        self.logger = logging.getLogger(__name__)

        if self.storage_mode == "gcp":
            self._init_vertex_ai_writer(run_id)
        else:
            self._init_local_writer(log_dir)

    def _init_local_writer(self, log_dir: str):
        """Inicializa el writer para logging local."""
        self.logger.info(f"Configurando TensorBoard para logging local en: {log_dir}")
        self.writer = SummaryWriter(log_dir=log_dir)

    def _init_vertex_ai_writer(self, run_id: str):
        """Inicializa el writer para logging en Vertex AI TensorBoard."""
        if aiplatform is None:
            self.logger.error("Librería 'google-cloud-aiplatform' no encontrada. No se puede loggear en Vertex AI.")
            return

        project = config.project_id
        location = config.tensorboard_location
        instance_name = config.tensorboard_instance_name
        experiment_name = config.tensorboard_experiment_name

        if not all([project, location, instance_name]):
            self.logger.error("Configuración incompleta para Vertex AI TensorBoard en config.yaml.")
            return

        try:
            # Inicializar cliente de AI Platform
            aiplatform.init(project=project, location=location)
            
            self.logger.info("Configurando TensorBoard para logging en Vertex AI...")
            self.logger.info(f"  - Proyecto: {project}")
            self.logger.info(f"  - Instancia: {instance_name}")
            self.logger.info(f"  - Experimento: {experiment_name}")
            self.logger.info(f"  - Run: {run_id}")

            # Crear el writer para el experimento y el run específicos
            # Usamos una estructura de directorio que Vertex AI puede interpretar correctamente
            log_dir_vertex = (
                f"projects/{project}/locations/{location}/tensorboards/{instance_name}"
                f"/experiments/{experiment_name}/runs/{run_id}"
            )
            self.writer = SummaryWriter(log_dir=log_dir_vertex)
            self.logger.info("✅ Conexión con Vertex AI TensorBoard establecida.")

        except Exception as e:
            self.logger.error(f"Error al inicializar Vertex AI TensorBoard Writer: {e}")
            self.writer = None  # Asegurar que no se intente usar un writer fallido
    
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
        if 'trades_count' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Total_Trades_Executed', trade_metrics['trades_count'], episode)
        if 'winning_trades' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Number_Winning_Trades', trade_metrics['winning_trades'], episode)
        if 'long_trades' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Number_Long_Trades', trade_metrics['long_trades'], episode)
        if 'short_trades' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Number_Short_Trades', trade_metrics['short_trades'], episode)
        if 'total_pnl_realized_abs' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Total_PNL_Realized_Absolute', trade_metrics['total_pnl_realized_abs'], episode)
        if 'total_roe_realized' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Total_ROE_Realized', trade_metrics['total_roe_realized'], episode)
        if 'avg_margin_used' in trade_metrics:
            self.writer.add_scalar('Trading_Episode/Average_Margin_Used_per_Trade', trade_metrics['avg_margin_used'], episode)
        
        # Environment metrics
        if 'drawdown_episode' in env_metrics:
            self.writer.add_scalar('Environment_Episode/Drawdown_Percentage', env_metrics['drawdown_episode'] * 100, episode)
        if 'final_balance' in env_metrics:
            self.writer.add_scalar('Environment_Episode/Final_Balance', env_metrics['final_balance'], episode)
        if 'initial_balance' in env_metrics:
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
    
    def close(self) -> None:
        """Cierra el writer de TensorBoard y limpia los recursos."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            self.logger.info("TensorBoard writer cerrado y logs enviados.")
