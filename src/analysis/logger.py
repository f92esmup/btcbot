"""
TensorBoard logging utilities for btcbot project.

This module provides the TensorboardLogger class for centralized TensorBoard logging operations.
"""

from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """
    A class for handling all TensorBoard logging operations.
    
    This class centralizes all TensorBoard logging functionality to keep logging
    logic organized and reusable across different parts of the application.
    """
    
    def __init__(self, log_dir: str) -> None:
        """
        Initialize the TensorBoard logger.
        
        Args:
            log_dir: Directory path where TensorBoard logs will be stored
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
    
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
        """
        Close the TensorBoard writer and clean up resources.
        """
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            self.writer = None
