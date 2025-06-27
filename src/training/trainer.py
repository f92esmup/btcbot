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


class Trainer:
    """
    Centralized trainer class that orchestrates the training process.
    Uses dependency injection for all components.
    """
    
    def __init__(self, agent, env, evaluator, logger, run_manager, trainer_config, logger_console):
        """
        Initialize trainer with all dependencies.
        
        Args:
            agent: The trading agent to train
            env: The trading environment
            evaluator: AgentEvaluator instance for periodic evaluation
            logger: TensorboardLogger instance for metrics logging
            run_manager: RunManager instance for file operations
            trainer_config: Configuration dict with training parameters
            logger_console: Console logger for status messages
        """
        self.agent = agent
        self.env = env
        self.evaluator = evaluator
        self.logger = logger
        self.run_manager = run_manager
        self.config = trainer_config
        self.logger_console = logger_console
        
        # Training state
        self.best_eval_return = float('-inf')
        self.global_trade_counter = 0
        self.learning_started = False
        
        # Initialize ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            capacity=trainer_config.get('replay_buffer_capacity', trainer_config.get('replay_buffer_size', 100000)),
            observation_shape=env.observation_space.shape,
            action_dim=env.action_space.shape[0]
        )
        
    def _parse_observation(self, observation: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse observation from environment into market and portfolio tensors.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (market_data, portfolio_data) tensors
        """
        # Environment observation is concatenated: [market_data_flat, portfolio_data]
        ventana_size = self.env.config_entorno['ventana_observacion_size']
        num_features_mercado = len(self.env.column_names)
        market_features_total = ventana_size * num_features_mercado
        
        # Split observation
        market_data_flat = observation[:market_features_total]
        portfolio_data_flat = observation[market_features_total:]
        
        # Reshape market data to (sequence_length, num_features)
        market_data = market_data_flat.reshape(ventana_size, num_features_mercado)
        
        # Convert to tensors and add batch dimension
        market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(self.agent.device)
        portfolio_tensor = torch.FloatTensor(portfolio_data_flat).unsqueeze(0).to(self.agent.device)
        
        return market_tensor, portfolio_tensor
    
    def _parse_observation_batch(self, observations_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized parsing of a batch of observations.
        
        Args:
            observations_batch: Tensor of shape (batch_size, total_features)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (market_data, portfolio_data) tensors
            - market_data: shape (batch_size, ventana_size, num_features_mercado)
            - portfolio_data: shape (batch_size, portfolio_features)
        """
        # Calculate dimensions
        ventana_size = self.env.config_entorno['ventana_observacion_size']
        num_features_mercado = len(self.env.column_names)
        market_features_total = ventana_size * num_features_mercado
        
        # Use tensor slicing to separate market and portfolio data (vectorized)
        market_data_flat = observations_batch[:, :market_features_total]
        portfolio_data = observations_batch[:, market_features_total:]
        
        # Reshape market data to 3D tensor (vectorized)
        batch_size = observations_batch.shape[0]
        market_data = market_data_flat.view(batch_size, ventana_size, num_features_mercado)
        
        return market_data, portfolio_data

    def train(self, start_episode: int, total_episodes: int):
        """
        Main training loop.
        
        Args:
            start_episode: Episode number to start from (for resuming)
            total_episodes: Total number of episodes to train
        """
        self.logger_console.info(f"Iniciando entrenamiento por {total_episodes} episodios (desde episodio {start_episode})...")
        self.logger_console.info(f"Usando semilla base {self.config['seed']} para generación de seeds específicos por episodio")
        
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
            current_episode_seed = self.config['seed'] + episode
            
            # Reset environment with specific seed
            obs, _ = self.env.reset(seed=current_episode_seed)
            episode_return = 0
            episode_length = 0
            initial_balance = self.env.balance_actual
            initial_equity = self.env.equity_actual
            max_equity_episode = self.env.equity_actual
            
            # Episode trading metrics
            episode_trades_count = 0
            episode_long_trades = 0
            episode_short_trades = 0
            episode_winning_trades = 0
            episode_total_pnl_realized_abs = 0.0
            episode_total_roe_realized = 0.0
            episode_margins_used = []
            
            num_trades_inicio_episodio = len(self.env.historial_trades)
            
            # Episode loss tracking
            ep_actor_losses = []
            ep_critic1_losses = []
            ep_critic2_losses = []
            ep_alpha_losses = []
            
            done = False
            while not done:
                # Parse observation and select action
                market_data, portfolio_data = self._parse_observation(obs)
                action = self.agent.select_action(market_data, portfolio_data, deterministic=False)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)
                
                # Train if enough experiences accumulated
                if self.replay_buffer.can_sample(self.config['batch_size']) and len(self.replay_buffer) >= self.config['min_buffer_for_learning']:
                    # Log when learning starts for the first time
                    if not self.learning_started:
                        self.logger_console.info(f"🎯 INICIANDO APRENDIZAJE: Buffer alcanzó {len(self.replay_buffer)} experiencias (mínimo: {self.config['min_buffer_for_learning']})")
                        self.learning_started = True
                    
                    # Sample batch from replay buffer
                    batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminated, batch_truncated = self.replay_buffer.sample(
                        self.config['batch_size'], self.agent.device
                    )
                    
                    # Parse batch observations (vectorized)
                    batch_market_data, batch_portfolio_data = self._parse_observation_batch(batch_obs)
                    batch_next_market_data, batch_next_portfolio_data = self._parse_observation_batch(batch_next_obs)
                    
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
                max_equity_episode = max(max_equity_episode, self.env.equity_actual)
                
                # Track individual trades
                if info.get('trade_ejecutado') and info.get('pnl_realizado', 0) != 0:
                    self.global_trade_counter += 1
                    episode_trades_count += 1
                    
                    ultimo_trade = self.env.historial_trades[-1]
                    
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
            final_balance = self.env.balance_actual
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
            drawdown_episode = (max_equity_episode - self.env.equity_actual) / max_equity_episode if max_equity_episode > 0 else 0
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
                'final_balance': self.env.balance_actual,
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
                learning_status = "LEARNING" if buffer_size >= self.config['min_buffer_for_learning'] else f"COLLECTING ({buffer_size}/{self.config['min_buffer_for_learning']})"
                
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
            if (episode + 1) % self.config['eval_frequency'] == 0 and self.evaluator:
                self.logger_console.info(f"\n=== Evaluación en episodio {episode + 1} ===")
                
                # Use AgentEvaluator for evaluation
                eval_metrics = self.evaluator.evaluate(self.agent, self.env, self.config['eval_episodes'])
                
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
                
                # Save best model using RunManager (solo si hay run_manager)
                if self.run_manager and eval_metrics['mean_return'] > self.best_eval_return:
                    self.best_eval_return = eval_metrics['mean_return']
                    best_model_path = self.run_manager.save_best_model(self.agent)
                    self.logger_console.info(f"  - Nuevo mejor modelo guardado: {best_model_path}")
            
            # Periodic checkpoint saving (solo si hay run_manager)
            if (episode + 1) % self.config['save_frequency'] == 0 and self.run_manager:
                self.run_manager.save_agent_checkpoint(self.agent, episode + 1)
        
        # Final save (solo si hay run_manager)
        final_model_path = None
        if self.run_manager:
            final_model_path = self.run_manager.save_final_model(self.agent)
        
        total_time = time.time() - start_time
        self.logger_console.info(f"\n=== Entrenamiento Completado ===")
        self.logger_console.info(f"Tiempo total: {total_time/3600:.2f} horas")
        self.logger_console.info(f"Return promedio: {np.mean(episode_returns):.2f}")
        self.logger_console.info(f"Profit promedio: {np.mean(episode_profits):.2f}%")
        self.logger_console.info(f"Mejor evaluación: {self.best_eval_return:.2f}")
        if final_model_path:
            self.logger_console.info(f"Modelo final guardado: {final_model_path}")
        
        # Close TensorBoard writer (solo si hay logger)
        if self.logger:
            self.logger.close()
