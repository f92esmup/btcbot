"""
Trainer class for centralizing training logic.
Handles the main training loop with dependency injection.
"""
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


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
                # Select action
                action = self.agent.select_action(obs, deterministic=False)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.agent.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)
                
                # Train if enough experiences accumulated
                if len(self.agent.replay_buffer) >= self.config['min_buffer_for_learning']:
                    # Log when learning starts for the first time
                    if not self.learning_started:
                        self.logger_console.info(f"🎯 INICIANDO APRENDIZAJE: Buffer alcanzó {len(self.agent.replay_buffer)} experiencias (mínimo: {self.config['min_buffer_for_learning']})")
                        self.learning_started = True
                    
                    losses = self.agent.learn()
                    if losses:
                        ep_actor_losses.append(losses['actor_loss'])
                        ep_critic1_losses.append(losses['critic_1_loss'])
                        ep_critic2_losses.append(losses['critic_2_loss'])
                        ep_alpha_losses.append(losses['alpha_loss'])
                        
                        # Log step metrics
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
                    
                    # Log per-trade metrics
                    trade_data = {
                        'roe': roe_trade,
                        'pnl_abs': pnl_abs_trade,
                        'pasos_duracion': ultimo_trade.get('pasos_duracion', 0),
                        'margen_usado': margen_usado_trade,
                        'tipo': ultimo_trade['tipo']
                    }
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
            
            # Log episode metrics
            self.logger.log_episode_metrics(episode, episode_return, profit_pct, episode_length, trade_metrics, env_metrics)
            
            # Log agent-specific metrics if we have losses
            if ep_actor_losses:
                self.logger.writer.add_scalar('Agent_Episode/Mean_Actor_Loss', np.mean(ep_actor_losses), episode)
                self.logger.writer.add_scalar('Agent_Episode/Mean_Critic_Loss', (np.mean(ep_critic1_losses) + np.mean(ep_critic2_losses)) / 2, episode)
                self.logger.writer.add_scalar('Agent_Episode/Mean_Alpha_Loss', np.mean(ep_alpha_losses), episode)
                self.logger.writer.add_scalar('Agent_Episode/Alpha_Value_at_End', self.agent.alpha.item(), episode)
            
            # Additional metrics
            self.logger.writer.add_scalar('Environment_Episode/Max_Equity_Reached', max_equity_episode, episode)
            self.logger.writer.add_scalar('Trading_Episode/Win_Rate_Percentage', win_rate_episode, episode)
            self.logger.writer.add_scalar('Trading_Episode/Average_ROE_per_Trade', avg_roe_episode, episode)
            self.logger.writer.add_scalar('Agent_Stats/Replay_Buffer_Size', len(self.agent.replay_buffer), episode)
            
            episode_time = time.time() - episode_start_time
            
            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_profit = np.mean(episode_profits[-10:])
                buffer_size = len(self.agent.replay_buffer)
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
            
            # Periodic evaluation
            if (episode + 1) % self.config['eval_frequency'] == 0:
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
                
                # Log evaluation metrics to TensorBoard
                self.logger.writer.add_scalar('Evaluation/Mean_Return', eval_metrics['mean_return'], episode + 1)
                self.logger.writer.add_scalar('Evaluation/Mean_Profit_Pct', eval_metrics['mean_profit_pct'], episode + 1)
                self.logger.writer.add_scalar('Evaluation/Win_Rate', eval_metrics['win_rate'], episode + 1)
                self.logger.writer.add_scalar('Evaluation/Max_Drawdown', eval_metrics['max_drawdown'], episode + 1)
                self.logger.writer.add_scalar('Evaluation/Sharpe_Ratio', eval_metrics['sharpe_ratio'], episode + 1)
                self.logger.writer.add_scalar('Evaluation/Sortino_Ratio', eval_metrics['sortino_ratio'], episode + 1)
                
                # Save best model using RunManager
                if eval_metrics['mean_return'] > self.best_eval_return:
                    self.best_eval_return = eval_metrics['mean_return']
                    best_model_path = self.run_manager.save_best_model(self.agent)
                    self.logger_console.info(f"  - Nuevo mejor modelo guardado: {best_model_path}")
            
            # Periodic checkpoint saving
            if (episode + 1) % self.config['save_frequency'] == 0:
                self.run_manager.save_agent_checkpoint(self.agent, episode + 1)
        
        # Final save
        final_model_path = self.run_manager.save_final_model(self.agent)
        
        total_time = time.time() - start_time
        self.logger_console.info(f"\n=== Entrenamiento Completado ===")
        self.logger_console.info(f"Tiempo total: {total_time/3600:.2f} horas")
        self.logger_console.info(f"Return promedio: {np.mean(episode_returns):.2f}")
        self.logger_console.info(f"Profit promedio: {np.mean(episode_profits):.2f}%")
        self.logger_console.info(f"Mejor evaluación: {self.best_eval_return:.2f}")
        self.logger_console.info(f"Modelo final guardado: {final_model_path}")
        
        # Close TensorBoard writer
        self.logger.close()
        
        # Sync TensorBoard logs to GCS if configured
        if self.config.get('storage_mode') == "gcp" and self.config.get('tensorboard_dir'):
            try:
                self.logger_console.info(f"Sincronizando logs de TensorBoard a GCS...")
                gcs_utils = self.config.get('gcs_utils')
                if gcs_utils:
                    tensorboard_prefix = f"{self.config['run_id']}/tensorboard"
                    gcs_utils.upload_directory_to_gcs(
                        local_directory_path=str(self.config['tensorboard_dir']),
                        gcs_prefix=tensorboard_prefix
                    )
                    self.logger_console.info(f"Logs de TensorBoard sincronizados exitosamente a gs://{self.config['gcs_bucket_name']}/{tensorboard_prefix}")
            except Exception as e:
                self.logger_console.error(f"Error al sincronizar logs de TensorBoard a GCS: {e}")
                self.logger_console.warning("El entrenamiento se completó correctamente, pero los logs no se pudieron sincronizar")
