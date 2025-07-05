"""
Trainer class for centralizing training logic.
Handles the main training loop with dependency injection.
"""
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from src.agente.replay_buffer import ReplayBuffer
from src.agente.observation_parser import parse_observation, parse_observation_batch
from src.configuration.constants import (
    KEY_SEED, KEY_BATCH_SIZE, KEY_MIN_BUFFER_FOR_LEARNING, 
    KEY_REPLAY_BUFFER_CAPACITY, KEY_REPLAY_BUFFER_SIZE,
    KEY_EVAL_FREQUENCY, KEY_SAVE_FREQUENCY, KEY_TENSORBOARD_DIR,
    KEY_STORAGE_MODE, STORAGE_MODE_GCP,
    STATUS_LEARNING, STATUS_COLLECTING
)


class Trainer:
    """
    Centralized trainer class that orchestrates the training process.
    Uses dependency injection for all components.
    """
    
    def __init__(self, agent, env, evaluator, logger, checkpoint_manager, config_manager, training_run_id, trainer_config, logger_console):
        """
        Initialize trainer with all dependencies.
        
        Args:
            agent: The trading agent to train
            env: The trading environment
            evaluator: AgentEvaluator instance for periodic evaluation
            logger: TensorboardLogger instance for metrics logging
            checkpoint_manager: CheckpointManager instance for model operations
            config_manager: ConfigManager instance for configuration operations
            training_run_id: The ID of the training run (used for saving models/checkpoints)
            trainer_config: Configuration dict with training parameters
            logger_console: Console logger for status messages
        """
        self.agent = agent
        self.env = env
        self.evaluator = evaluator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.config_manager = config_manager
        self.training_run_id = training_run_id
        self.config = trainer_config
        self.logger_console = logger_console
        
        # Training state
        self.best_eval_return = float('-inf')
        self.best_model_path = None
        self.global_trade_counter = 0
        self.learning_started = False
        
        # Initialize ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            capacity=trainer_config.get(KEY_REPLAY_BUFFER_CAPACITY, trainer_config.get(KEY_REPLAY_BUFFER_SIZE, 100000)),
            observation_shape=env.observation_space.shape,
            action_dim=env.action_space.shape[0]
        )

    def train(self, start_episode: int, total_episodes: int):
        """
        Main training loop.
        
        Args:
            start_episode: Episode number to start from (for resuming)
            total_episodes: Total number of episodes to train
        """
        self.logger_console.info(f"Iniciando entrenamiento por {total_episodes} episodios (desde episodio {start_episode})...")
        self.logger_console.info(f"Usando semilla base {self.config[KEY_SEED]} para generación de seeds específicos por episodio")
        
        # Training metrics storage
        episode_returns = []
        episode_profits = []
        episode_lengths = []
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        alpha_values = []
        
        start_time = time.time()
        
        for episode in range(start_episode, total_episodes):
            episode_start_time = time.time()
            
            # Calculate episode-specific seed
            current_episode_seed = self.config[KEY_SEED] + episode
            
            # Reset environment with specific seed
            obs, _ = self.env.reset(seed=current_episode_seed)
            episode_return = 0
            episode_length = 0
            initial_balance = self.env.portfolio.balance
            initial_equity = self.env.portfolio.equity
            max_equity_episode = self.env.portfolio.equity
            
            # Episode trading metrics
            episode_trades_count = 0
            episode_long_trades = 0
            episode_short_trades = 0
            episode_winning_trades = 0
            episode_total_pnl_realized_abs = 0.0
            episode_total_roe_realized = 0.0
            episode_margins_used = []
            
            num_trades_inicio_episodio = len(self.env.portfolio.historial_trades)
            
            # Episode loss tracking
            ep_actor_losses = []
            ep_critic1_losses = []
            ep_critic2_losses = []
            ep_alpha_losses = []
            
            done = False
            while not done:
                # Parse observation and select action
                market_data, portfolio_data = parse_observation(obs, self.env.config_entorno, self.agent.config, self.agent.device)
                action = self.agent.select_action(market_data, portfolio_data, deterministic=False)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)
                
                # Train if enough experiences accumulated
                if self.replay_buffer.can_sample(self.config[KEY_BATCH_SIZE]) and len(self.replay_buffer) >= self.config[KEY_MIN_BUFFER_FOR_LEARNING]:
                    # Log when learning starts for the first time
                    if not self.learning_started:
                        self.logger_console.info(f"🎯 INICIANDO APRENDIZAJE: Buffer alcanzó {len(self.replay_buffer)} experiencias (mínimo: {self.config[KEY_MIN_BUFFER_FOR_LEARNING]})")
                        self.learning_started = True
                    
                    # Sample batch from replay buffer
                    batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminated, batch_truncated = self.replay_buffer.sample(
                        self.config[KEY_BATCH_SIZE], self.agent.device
                    )
                    
                    # Parse batch observations (vectorized)
                    batch_market_data, batch_portfolio_data = parse_observation_batch(
                        batch_obs, self.env.config_entorno, self.agent.config, len(self.env.column_names)
                    )
                    batch_next_market_data, batch_next_portfolio_data = parse_observation_batch(
                        batch_next_obs, self.env.config_entorno, self.agent.config, len(self.env.column_names)
                    )
                    
                    # Learn from batch
                    losses = self.agent.learn(
                        batch_market_data, batch_portfolio_data, batch_actions,
                        batch_rewards, batch_next_market_data, batch_next_portfolio_data,
                        batch_terminated, batch_truncated
                    )
                    
                    if losses:
                        ep_actor_losses.append(losses['actor_loss'])
                        ep_critic1_losses.append(losses['critic_1_loss'])
                        ep_critic2_losses.append(losses['critic_2_loss'])
                        ep_alpha_losses.append(losses['alpha_loss'])
                        
                        # Log step metrics (solo si hay logger)
                        if self.logger:
                            self.logger.log_step_metrics(self.agent.learning_steps, losses, self.agent.alpha.item())
                
                obs = next_obs
                episode_return += reward
                episode_length += 1
                
                # Update max equity for episode drawdown
                max_equity_episode = max(max_equity_episode, self.env.portfolio.equity)
                
                # Track individual trades
                if info.get('trade_ejecutado') and info.get('pnl_realizado', 0) != 0:
                    self.global_trade_counter += 1
                    episode_trades_count += 1
                    
                    ultimo_trade = self.env.portfolio.historial_trades[-1]
                    
                    pnl_abs_trade = ultimo_trade.get('pnl_abs', 0.0)
                    roe_trade = ultimo_trade.get('roe', 0.0)
                    margen_usado_trade = ultimo_trade.get('margen_usado', 0.0)
                    
                    episode_total_pnl_realized_abs += pnl_abs_trade
                    episode_total_roe_realized += roe_trade
                    if margen_usado_trade > 0:
                        episode_margins_used.append(margen_usado_trade)
                    
                    if pnl_abs_trade > 0:
                        episode_winning_trades += 1
                    
                    if ultimo_trade['tipo'] == 'LARGO':
                        episode_long_trades += 1
                    elif ultimo_trade['tipo'] == 'CORTO':
                        episode_short_trades += 1
                    
                    # Log per-trade metrics (solo si hay logger)
                    trade_data = {
                        'roe': roe_trade,
                        'pnl_abs': pnl_abs_trade,
                        'pasos_duracion': ultimo_trade.get('pasos_duracion', 0),
                        'margen_usado': margen_usado_trade,
                        'tipo': ultimo_trade['tipo']
                    }
                    if self.logger:
                        self.logger.log_per_trade_metrics(self.global_trade_counter, trade_data)
            
            # Calculate episode profit
            final_balance = self.env.portfolio.balance
            profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
            
            # Store metrics
            episode_returns.append(episode_return)
            episode_profits.append(profit_pct)
            episode_lengths.append(episode_length)
            
            # Log average losses to TensorBoard
            if ep_actor_losses:
                actor_losses.append(np.mean(ep_actor_losses))
                critic_losses.append((np.mean(ep_critic1_losses) + np.mean(ep_critic2_losses)) / 2)
                alpha_losses.append(np.mean(ep_alpha_losses))
                alpha_values.append(self.agent.alpha.item())
            
            # Prepare metrics for logging
            drawdown_episode = (max_equity_episode - self.env.portfolio.equity) / max_equity_episode if max_equity_episode > 0 else 0
            win_rate_episode = (episode_winning_trades / episode_trades_count) * 100 if episode_trades_count > 0 else 0
            avg_roe_episode = (episode_total_roe_realized / episode_trades_count) if episode_trades_count > 0 else 0.0
            avg_margin_used_episode = np.mean(episode_margins_used) if episode_margins_used else 0.0
            
            # Organize metrics in dictionaries
            trade_metrics = {
                'trades_count': episode_trades_count,
                'winning_trades': episode_winning_trades,
                'long_trades': episode_long_trades,
                'short_trades': episode_short_trades,
                'total_pnl_realized_abs': episode_total_pnl_realized_abs,
                'total_roe_realized': episode_total_roe_realized,
                'avg_margin_used': avg_margin_used_episode
            }
            
            env_metrics = {
                'drawdown_episode': drawdown_episode,
                'final_balance': self.env.portfolio.balance,
                'initial_balance': initial_balance
            }
            
            # Log episode metrics (solo si hay logger)
            if self.logger:
                self.logger.log_episode_metrics(episode, episode_return, profit_pct, episode_length, trade_metrics, env_metrics)
            
            # Log agent-specific metrics if we have losses
            if ep_actor_losses:
                agent_episode_metrics = {
                    'mean_actor_loss': np.mean(ep_actor_losses),
                    'mean_critic_loss': (np.mean(ep_critic1_losses) + np.mean(ep_critic2_losses)) / 2,
                    'mean_alpha_loss': np.mean(ep_alpha_losses),
                    'alpha_value_at_end': self.agent.alpha.item()
                }
                
                # Use individual metric logging for agent metrics not covered by high-level methods (solo si hay logger)
                if self.logger:
                    self.logger.writer.add_scalar('Agent_Episode/Mean_Actor_Loss', agent_episode_metrics['mean_actor_loss'], episode)
                    self.logger.writer.add_scalar('Agent_Episode/Mean_Critic_Loss', agent_episode_metrics['mean_critic_loss'], episode)
                    self.logger.writer.add_scalar('Agent_Episode/Mean_Alpha_Loss', agent_episode_metrics['mean_alpha_loss'], episode)
                    self.logger.writer.add_scalar('Agent_Episode/Alpha_Value_at_End', agent_episode_metrics['alpha_value_at_end'], episode)
            
            # Additional metrics (solo si hay logger)
            if self.logger:
                self.logger.writer.add_scalar('Environment_Episode/Max_Equity_Reached', max_equity_episode, episode)
                self.logger.writer.add_scalar('Trading_Episode/Win_Rate_Percentage', win_rate_episode, episode)
                self.logger.writer.add_scalar('Trading_Episode/Average_ROE_per_Trade', avg_roe_episode, episode)
                self.logger.writer.add_scalar('Agent_Stats/Replay_Buffer_Size', len(self.replay_buffer), episode)
            
            episode_time = time.time() - episode_start_time
            
            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_profit = np.mean(episode_profits[-10:])
                buffer_size = len(self.replay_buffer)
                learning_status = STATUS_LEARNING if buffer_size >= self.config[KEY_MIN_BUFFER_FOR_LEARNING] else f"{STATUS_COLLECTING} ({buffer_size}/{self.config[KEY_MIN_BUFFER_FOR_LEARNING]})"
                
                self.logger_console.info(f"Episodio {episode + 1}/{total_episodes} | "
                           f"Return: {episode_return:.2f} | "
                           f"Profit: {profit_pct:.2f}% | "
                           f"Avg Return (10): {avg_return:.2f} | "
                           f"Avg Profit (10): {avg_profit:.2f}% | "
                           f"Buffer: {buffer_size} | "
                           f"Status: {learning_status} | "
                           f"Time: {episode_time:.1f}s")
                
                if alpha_values:
                    self.logger_console.info(f"Alpha: {alpha_values[-1]:.4f} | "
                               f"Actor Loss: {actor_losses[-1]:.4f} | "
                               f"Critic Loss: {critic_losses[-1]:.4f}")
            
            # Periodic evaluation (solo si hay evaluator)
            if (episode + 1) % self.config[KEY_EVAL_FREQUENCY] == 0 and self.evaluator:
                self.logger_console.info(f"\n=== Evaluación en episodio {episode + 1} ===")
                
                # Use AgentEvaluator for evaluation
                eval_metrics, _, _ = self.evaluator.evaluate(self.agent, self.env)
                
                self.logger_console.info(f"Métricas de evaluación:")
                self.logger_console.info(f"  - Return promedio: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
                self.logger_console.info(f"  - Profit promedio: {eval_metrics['mean_profit_pct']:.2f}% ± {eval_metrics['std_profit_pct']:.2f}%")
                self.logger_console.info(f"  - Longitud promedio: {eval_metrics['mean_episode_length']:.1f}")
                self.logger_console.info(f"  - Win rate: {eval_metrics['win_rate']:.2%}")
                self.logger_console.info(f"  - Trades por episodio: {eval_metrics['total_trades']:.1f}")
                self.logger_console.info(f"  - Máximo Drawdown: {eval_metrics['max_drawdown'] * 100:.2f}%")
                self.logger_console.info(f"  - Sharpe Ratio: {eval_metrics['sharpe_ratio']:.4f}")
                self.logger_console.info(f"  - Sortino Ratio: {eval_metrics['sortino_ratio']:.4f}")
                
                # Log evaluation metrics to TensorBoard using high-level method (solo si hay logger)
                if self.logger:
                    self.logger.log_evaluation_metrics(episode + 1, eval_metrics)
                
                # Save best model using CheckpointManager (solo si hay checkpoint_manager)
                if self.checkpoint_manager and eval_metrics['mean_return'] > self.best_eval_return:
                    self.best_eval_return = eval_metrics['mean_return']
                    self.best_model_path = self.checkpoint_manager.save_best_model(self.training_run_id, self.agent)
                    self.logger_console.info(f"  - Nuevo mejor modelo guardado: {self.best_model_path}")
            
            # Periodic checkpoint saving (solo si hay checkpoint_manager)
            if (episode + 1) % self.config[KEY_SAVE_FREQUENCY] == 0 and self.checkpoint_manager:
                self.checkpoint_manager.save_agent_checkpoint(self.training_run_id, self.agent, episode + 1)
        
        # Final save (solo si hay checkpoint_manager)
        final_model_path = None
        if self.checkpoint_manager:
            final_model_path = self.checkpoint_manager.save_final_model(self.training_run_id, self.agent)
        
        total_time = time.time() - start_time
        self.logger_console.info(f"\n=== Entrenamiento Completado ===")
        self.logger_console.info(f"Tiempo total: {total_time/3600:.2f} horas")
        self.logger_console.info(f"Return promedio: {np.mean(episode_returns):.2f}")
        self.logger_console.info(f"Profit promedio: {np.mean(episode_profits):.2f}%")
        self.logger_console.info(f"Mejor evaluación: {self.best_eval_return:.2f}")
        if final_model_path:
            self.logger_console.info(f"Modelo final guardado: {final_model_path}")
        
        # === SUBIDA DE LOGS DE TENSORBOARD A GCS ===
        # Verificar si debemos subir logs de TensorBoard a GCS
        if (self.checkpoint_manager and 
            self.config.get(KEY_STORAGE_MODE) == STORAGE_MODE_GCP and 
            self.config.get(KEY_TENSORBOARD_DIR) and 
            self.training_run_id):
            
            # Construir la ruta completa al directorio de logs del run actual
            local_tensorboard_run_dir = f"{self.config[KEY_TENSORBOARD_DIR]}/{self.training_run_id}"
            
            self.logger_console.info(f"\n📤 === SUBIENDO LOGS DE TENSORBOARD A GCS ===")
            self.logger_console.info(f"Directorio local de logs: {local_tensorboard_run_dir}")
            
            try:
                # Llamar al método de subida del CheckpointManager
                self.checkpoint_manager.upload_tensorboard_logs(
                    local_log_dir=local_tensorboard_run_dir,
                    training_run_id=self.training_run_id
                )
                self.logger_console.info("✅ Subida de logs de TensorBoard completada")
            except Exception as e:
                self.logger_console.error(f"❌ Error durante la subida de logs de TensorBoard: {e}")
        
        # Close TensorBoard writer (solo si hay logger)
        if self.logger:
            self.logger.close()
