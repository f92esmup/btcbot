"""
Script principal de entrenamiento del bot de trading de Bitcoin.
Orquesta la adquisición de datos, cálculo de indicadores y entrenamiento del modelo.
"""

import argparse
import sys
import logging
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import re
from torch.utils.tensorboard import SummaryWriter

from src.data.Adquisicion import Adquisicion
from src.data.indicadores import Indicadores
from src.data.normalization import Normalization
from src.entorno.environment import FuturesTradingEnv
from src.agente.agent import TransformerSACAgent
from src.configuration.config import config
from src.configuration.gcs_utils import GCSUtils


def calculate_max_drawdown(equity_series: list) -> float:
    """
    Calcula el máximo drawdown de una serie de equity.
    
    Args:
        equity_series: Lista de valores de equity
        
    Returns:
        float: Máximo drawdown como porcentaje (0.0 - 1.0)
    """
    if len(equity_series) == 0:
        return 0.0
    
    equity_array = np.array(equity_series)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown = np.max(drawdown)
    
    return max_drawdown


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """
    Calcula el Sharpe Ratio de una serie de retornos.
    
    Args:
        returns: Lista de retornos
        risk_free_rate: Tasa libre de riesgo (anualizada)
        
    Returns:
        float: Sharpe Ratio
    """
    if len(returns) <= 1:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio


def calculate_sortino_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """
    Calcula el Sortino Ratio de una serie de retornos.
    
    Args:
        returns: Lista de retornos
        risk_free_rate: Tasa libre de riesgo (anualizada)
        
    Returns:
        float: Sortino Ratio
    """
    if len(returns) <= 1:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Solo considerar retornos negativos para la desviación estándar
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        # No hay retornos negativos, se considera muy bueno
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(negative_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    sortino_ratio = np.mean(excess_returns) / downside_deviation
    return sortino_ratio


def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            #logging.FileHandler('trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Bot de trading de Bitcoin')
    
    # Argumentos requeridos
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Símbolo del par de trading (ej: BTCUSDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Intervalo de tiempo para las velas'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Fecha de inicio en formato YYYY-MM-DD'
    )
    
    # Argumentos opcionales para entrenamiento
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Número de episodios de entrenamiento (default: 1000)'
    )
    
    parser.add_argument(
        '--eval-frequency',
        type=int,
        default=50,
        help='Frecuencia de evaluación en episodios (default: 50)'
    )
    
    parser.add_argument(
        '--save-frequency',
        type=int,
        default=100,
        help='Frecuencia de guardado en episodios (default: 100)'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Deshabilitar CUDA aunque esté disponible'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=5,
        help='Número de episodios para evaluación (default: 5)'
    )
    
    return parser.parse_args()


def setup_device(no_cuda: bool = False) -> torch.device:
    """
    Configura el device para el entrenamiento.
    
    Args:
        no_cuda (bool): Si True, fuerza el uso de CPU
        
    Returns:
        torch.device: Device configurado
    """
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        # Configurar para mejor rendimiento
        torch.backends.cudnn.benchmark = True
    
    return device


def create_trading_environment(dataframe: Any, logger) -> FuturesTradingEnv:
    """
    Crea el entorno de trading con los datos procesados.
    
    Args:
        dataframe: DataFrame con datos normalizados
        logger: Logger para mensajes
        
    Returns:
        FuturesTradingEnv: Entorno configurado
    """
    logger.info("Creando entorno de trading...")
    
    # Cargar el price_scaler directamente desde donde se guardó
    from src.data.normalization import Normalization
    
    logger.info("Cargando price_scaler desde almacenamiento...")
    try:
        # Cargar el price_scaler que se guardó durante la normalización
        price_scaler = Normalization.load_price_scaler()
        
        # Obtener información del rango para logging
        if hasattr(price_scaler, 'data_min_') and hasattr(price_scaler, 'data_max_'):
            close_min = price_scaler.data_min_[0]
            close_max = price_scaler.data_max_[0]
            logger.info(f"Price scaler cargado exitosamente - Rango Close: {close_min:.2f} - {close_max:.2f}")
        else:
            logger.info("Price scaler cargado exitosamente")
        
    except Exception as e:
        logger.error(f"Error crítico al cargar price_scaler: {e}")
        logger.error("No se puede continuar sin el price_scaler. Deteniendo ejecución.")
        raise RuntimeError(f"Fallo al cargar price_scaler: {e}")
    
    env = FuturesTradingEnv(
        data_df=dataframe,
        price_scaler=price_scaler
    )
    
    logger.info(f"Entorno creado:")
    logger.info(f"  - Balance inicial: ${config.capital_inicial:,.2f}")
    logger.info(f"  - Apalancamiento: {config.apalancamiento}x")
    logger.info(f"  - Ventana observación: {config.ventana_observacion_size}")
    logger.info(f"  - Espacio de observación: {env.observation_space}")
    logger.info(f"  - Espacio de acción: {env.action_space}")
    
    return env


def create_sac_agent(env: FuturesTradingEnv, device: torch.device, logger) -> TransformerSACAgent:
    """
    Crea el agente SAC con arquitectura Transformer.
    
    Args:
        env: Entorno de trading
        device: Device para el entrenamiento
        logger: Logger para mensajes
        
    Returns:
        TransformerSACAgent: Agente configurado
    """
    logger.info("Creando agente SAC con Transformer...")
    
    # Obtener parámetros del entorno
    observation_space_shape = env.observation_space.shape
    action_space_shape = env.action_space.shape
    
    # Calcular características de mercado y portfolio
    ventana_size = config.ventana_observacion_size
    num_features_mercado = len(env.data_df.columns)
    market_features = num_features_mercado
    portfolio_features = 4  # tipo_posicion, pnl_roe, pasos_posicion, precio_entrada
    sequence_length = ventana_size
    
    agent = TransformerSACAgent(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        market_features=market_features,
        portfolio_features=portfolio_features,
        sequence_length=sequence_length,
        device=device
    )
    
    # Contar parámetros del modelo
    total_params = sum(p.numel() for p in agent.actor.parameters())
    trainable_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    
    logger.info(f"Agente SAC creado:")
    logger.info(f"  - Parámetros totales: {total_params:,}")
    logger.info(f"  - Parámetros entrenables: {trainable_params:,}")
    logger.info(f"  - Market features: {market_features}")
    logger.info(f"  - Portfolio features: {portfolio_features}")
    logger.info(f"  - Sequence length: {sequence_length}")
    logger.info(f"  - Gamma: {config.gamma}")
    logger.info(f"  - Tau: {config.tau}")
    logger.info(f"  - Alpha inicial: {config.initial_log_alpha}")
    logger.info(f"  - Learning rates: Actor={config.actor_learning_rate}, Critic={config.critic_learning_rate}")
    
    return agent


def evaluate_agent(agent: TransformerSACAgent, env: FuturesTradingEnv, 
                  num_episodes: int, logger, writer: SummaryWriter = None, global_step: int = None) -> Dict[str, float]:
    """
    Evalúa el rendimiento del agente.
    
    Args:
        agent: Agente a evaluar
        env: Entorno de trading
        num_episodes: Número de episodios de evaluación
        logger: Logger para mensajes
        writer: TensorBoard SummaryWriter (opcional)
        global_step: Paso global para TensorBoard (opcional)
        
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
    
    # Calcular métricas financieras avanzadas
    max_drawdown = calculate_max_drawdown(episode_equity_series)
    sharpe_ratio = calculate_sharpe_ratio(episode_profits)
    sortino_ratio = calculate_sortino_ratio(episode_profits)
    
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
    
    # Log básico de métricas
    logger.info(f"Métricas de evaluación avanzadas:")
    logger.info(f"  - Máximo Drawdown: {max_drawdown * 100:.2f}%")
    logger.info(f"  - Sharpe Ratio: {sharpe_ratio:.4f}")
    logger.info(f"  - Sortino Ratio: {sortino_ratio:.4f}")
    
    # Log to TensorBoard if writer provided
    if writer and global_step is not None:
        writer.add_scalar('Evaluation/Mean_Return', metrics['mean_return'], global_step)
        writer.add_scalar('Evaluation/Std_Return', metrics['std_return'], global_step)
        writer.add_scalar('Evaluation/Mean_Profit_Pct', metrics['mean_profit_pct'], global_step)
        writer.add_scalar('Evaluation/Std_Profit_Pct', metrics['std_profit_pct'], global_step)
        writer.add_scalar('Evaluation/Mean_Episode_Length', metrics['mean_episode_length'], global_step)
        writer.add_scalar('Evaluation/Win_Rate', metrics['win_rate'] * 100, global_step)
        writer.add_scalar('Evaluation/Avg_Trades_per_Episode', metrics['total_trades'], global_step)
        
        # Nuevas métricas financieras en TensorBoard
        writer.add_scalar('Evaluation/Max_Drawdown_Pct', metrics['max_drawdown'] * 100, global_step)
        writer.add_scalar('Evaluation/Sharpe_Ratio', metrics['sharpe_ratio'], global_step)
        writer.add_scalar('Evaluation/Sortino_Ratio', metrics['sortino_ratio'], global_step)
    
    return metrics


def train_agent(agent: TransformerSACAgent, env: FuturesTradingEnv, 
               num_episodes: int, eval_frequency: int, eval_episodes: int,
               save_frequency: int, logger, writer: SummaryWriter,
               model_prefix: str = "models", log_dir: Path = None,
               start_episode: int = 0, model_save_base_prefix_arg: str = None):
    """
    Entrena el agente SAC.
    
    Args:
        agent: Agente a entrenar
        env: Entorno de trading
        num_episodes: Número de episodios de entrenamiento
        eval_frequency: Frecuencia de evaluación
        eval_episodes: Episodios para evaluación
        save_frequency: Frecuencia de guardado
        logger: Logger para mensajes
        writer: TensorBoard SummaryWriter
        model_prefix: Prefijo para nombres de modelo (usado en GCS)
        log_dir: Directorio de logs de TensorBoard para sync a GCS
        start_episode: Episodio inicial (para resumir entrenamiento)
        model_save_base_prefix_arg: Prefijo base para guardar checkpoints
    """
    logger.info(f"Iniciando entrenamiento por {num_episodes} episodios (desde episodio {start_episode})...")
    
    # Determinar el prefijo base para guardado de checkpoints
    checkpoint_save_prefix = model_save_base_prefix_arg or model_prefix
    
    # Métricas de entrenamiento
    episode_returns = []
    episode_profits = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    alpha_losses = []
    alpha_values = []
    
    # Control de inicio de aprendizaje
    learning_started = False
    
    # Crear directorio para modelos (solo si no es GCS)
    if config.storage_mode != "gcp":
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
    
    # Variables para tracking
    best_eval_return = float('-inf')
    start_time = time.time()
    global_trade_counter = 0
    
    for episode in range(start_episode, num_episodes):
        episode_start_time = time.time()
        
        # Reset del entorno
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        initial_balance = env.balance_actual
        initial_equity = env.equity_actual
        max_equity_episode = env.equity_actual
        
        # Métricas de trading para el episodio actual
        episode_trades_count = 0
        episode_long_trades = 0
        episode_short_trades = 0
        episode_winning_trades = 0
        episode_total_pnl_realized_abs = 0.0
        episode_total_roe_realized = 0.0
        episode_margins_used = []
        
        num_trades_inicio_episodio = len(env.historial_trades)
        
        # Variables para losses del episodio
        ep_actor_losses = []
        ep_critic1_losses = []
        ep_critic2_losses = []
        ep_alpha_losses = []
        
        done = False
        while not done:
            # Seleccionar acción
            action = agent.select_action(obs, deterministic=False)
            
            # Ejecutar acción en el entorno
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Almacenar experiencia
            agent.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)
            
            # Entrenar solo si hay suficientes experiencias acumuladas
            if len(agent.replay_buffer) >= config.min_buffer_for_learning:
                # Log especial cuando inicie el aprendizaje por primera vez
                if not learning_started:
                    logger.info(f"🎯 INICIANDO APRENDIZAJE: Buffer alcanzó {len(agent.replay_buffer)} experiencias (mínimo: {config.min_buffer_for_learning})")
                    learning_started = True
                
                losses = agent.learn()
                if losses:
                    ep_actor_losses.append(losses['actor_loss'])
                    ep_critic1_losses.append(losses['critic_1_loss'])
                    ep_critic2_losses.append(losses['critic_2_loss'])
                    ep_alpha_losses.append(losses['alpha_loss'])
                    
                    # Log losses por cada paso de aprendizaje
                    writer.add_scalar('Agent/Actor_Loss_step', losses['actor_loss'], agent.learning_steps)
                    writer.add_scalar('Agent/Critic_1_Loss_step', losses['critic_1_loss'], agent.learning_steps)
                    writer.add_scalar('Agent/Critic_2_Loss_step', losses['critic_2_loss'], agent.learning_steps)
                    writer.add_scalar('Agent/Alpha_Loss_step', losses['alpha_loss'], agent.learning_steps)
                    writer.add_scalar('Agent/Alpha_Value_step', agent.alpha.item(), agent.learning_steps)
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
            
            # Actualizar max equity para drawdown del episodio
            max_equity_episode = max(max_equity_episode, env.equity_actual)
            
            # Tracking de trades individuales
            if info.get('trade_ejecutado') and info.get('pnl_realizado', 0) != 0:
                global_trade_counter += 1
                episode_trades_count += 1
                
                ultimo_trade = env.historial_trades[-1]
                
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
                
                # Log métricas por trade individual
                writer.add_scalar('Per_Trade/PNL_Realized_ROE', roe_trade, global_trade_counter)
                writer.add_scalar('Per_Trade/PNL_Realized_Absolute', pnl_abs_trade, global_trade_counter)
                writer.add_scalar('Per_Trade/Duration_Steps', ultimo_trade.get('pasos_duracion', 0), global_trade_counter)
                writer.add_scalar('Per_Trade/Margin_Used', margen_usado_trade, global_trade_counter)
                
                direction_val = 1 if ultimo_trade['tipo'] == 'LARGO' else (-1 if ultimo_trade['tipo'] == 'CORTO' else 0)
                writer.add_scalar('Per_Trade/Direction', direction_val, global_trade_counter)
        
        # Calcular profit del episodio
        final_balance = env.balance_actual
        profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
        
        # Guardar métricas
        episode_returns.append(episode_return)
        episode_profits.append(profit_pct)
        episode_lengths.append(episode_length)
        
        # Log losses medias del episodio a TensorBoard
        if ep_actor_losses:
            actor_losses.append(np.mean(ep_actor_losses))
            # Promedio de ambos critic losses
            critic_losses.append((np.mean(ep_critic1_losses) + np.mean(ep_critic2_losses)) / 2)
            alpha_losses.append(np.mean(ep_alpha_losses))
            alpha_values.append(agent.alpha.item())
            
            # TensorBoard logging por episodio - Agent metrics
            writer.add_scalar('Agent_Episode/Mean_Actor_Loss', np.mean(ep_actor_losses), episode)
            writer.add_scalar('Agent_Episode/Mean_Critic_Loss', (np.mean(ep_critic1_losses) + np.mean(ep_critic2_losses)) / 2, episode)
            writer.add_scalar('Agent_Episode/Mean_Alpha_Loss', np.mean(ep_alpha_losses), episode)
            writer.add_scalar('Agent_Episode/Alpha_Value_at_End', agent.alpha.item(), episode)
        
        # TensorBoard logging por episodio - Episode metrics
        writer.add_scalar('Episode_Metrics/Return', episode_return, episode)
        writer.add_scalar('Episode_Metrics/Profit_Percentage_Initial_Capital', profit_pct, episode)
        writer.add_scalar('Episode_Metrics/Length', episode_length, episode)
        
        # TensorBoard logging por episodio - Environment metrics
        writer.add_scalar('Environment_Episode/Final_Balance', env.balance_actual, episode)
        writer.add_scalar('Environment_Episode/Final_Equity', env.equity_actual, episode)
        writer.add_scalar('Environment_Episode/Max_Equity_Reached', max_equity_episode, episode)
        drawdown_episode = (max_equity_episode - env.equity_actual) / max_equity_episode if max_equity_episode > 0 else 0
        writer.add_scalar('Environment_Episode/Drawdown_Percentage', drawdown_episode * 100, episode)
        
        # TensorBoard logging por episodio - Trading metrics
        writer.add_scalar('Trading_Episode/Total_Trades_Executed', episode_trades_count, episode)
        writer.add_scalar('Trading_Episode/Number_Long_Trades', episode_long_trades, episode)
        writer.add_scalar('Trading_Episode/Number_Short_Trades', episode_short_trades, episode)
        win_rate_episode = (episode_winning_trades / episode_trades_count) * 100 if episode_trades_count > 0 else 0
        writer.add_scalar('Trading_Episode/Win_Rate_Percentage', win_rate_episode, episode)
        
        avg_roe_episode = (episode_total_roe_realized / episode_trades_count) if episode_trades_count > 0 else 0.0
        writer.add_scalar('Trading_Episode/Average_ROE_per_Trade', avg_roe_episode, episode)
        writer.add_scalar('Trading_Episode/Total_PNL_Realized_Absolute', episode_total_pnl_realized_abs, episode)
        avg_margin_used_episode = np.mean(episode_margins_used) if episode_margins_used else 0.0
        writer.add_scalar('Trading_Episode/Average_Margin_Used_per_Trade', avg_margin_used_episode, episode)
        
        # Buffer size
        writer.add_scalar('Agent_Stats/Replay_Buffer_Size', len(agent.replay_buffer), episode)
        
        episode_time = time.time() - episode_start_time
        
        # Log progreso cada 10 episodios
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_profit = np.mean(episode_profits[-10:])
            buffer_size = len(agent.replay_buffer)
            learning_status = "LEARNING" if buffer_size >= config.min_buffer_for_learning else f"COLLECTING ({buffer_size}/{config.min_buffer_for_learning})"
            
            logger.info(f"Episodio {episode + 1}/{num_episodes} | "
                       f"Return: {episode_return:.2f} | "
                       f"Profit: {profit_pct:.2f}% | "
                       f"Avg Return (10): {avg_return:.2f} | "
                       f"Avg Profit (10): {avg_profit:.2f}% | "
                       f"Buffer: {buffer_size} | "
                       f"Status: {learning_status} | "
                       f"Time: {episode_time:.1f}s")
            
            if alpha_values:
                logger.info(f"Alpha: {alpha_values[-1]:.4f} | "
                           f"Actor Loss: {actor_losses[-1]:.4f} | "
                           f"Critic Loss: {critic_losses[-1]:.4f}")
        
        # Evaluación periódica
        if (episode + 1) % eval_frequency == 0:
            logger.info(f"\n=== Evaluación en episodio {episode + 1} ===")
            eval_metrics = evaluate_agent(agent, env, eval_episodes, logger, writer, episode + 1)
            
            logger.info(f"Métricas de evaluación:")
            logger.info(f"  - Return promedio: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
            logger.info(f"  - Profit promedio: {eval_metrics['mean_profit_pct']:.2f}% ± {eval_metrics['std_profit_pct']:.2f}%")
            logger.info(f"  - Longitud promedio: {eval_metrics['mean_episode_length']:.1f}")
            logger.info(f"  - Win rate: {eval_metrics['win_rate']:.2%}")
            logger.info(f"  - Trades por episodio: {eval_metrics['total_trades']:.1f}")
            logger.info(f"  - Máximo Drawdown: {eval_metrics['max_drawdown'] * 100:.2f}%")
            logger.info(f"  - Sharpe Ratio: {eval_metrics['sharpe_ratio']:.4f}")
            logger.info(f"  - Sortino Ratio: {eval_metrics['sortino_ratio']:.4f}")
            
            # Guardar mejor modelo
            if eval_metrics['mean_return'] > best_eval_return:
                best_eval_return = eval_metrics['mean_return']
                if config.storage_mode == "gcp":
                    best_model_path = f"{model_prefix}/best_model.pth"
                else:
                    best_model_path = models_dir / "best_model.pth"
                agent.save(best_model_path)
                logger.info(f"  - Nuevo mejor modelo guardado: {best_model_path}")
        
        # Guardado periódico
        if (episode + 1) % save_frequency == 0:
            if config.storage_mode == "gcp":
                checkpoint_path = f"{checkpoint_save_prefix}/checkpoint_episode_{episode + 1}"
                agent.save_models(checkpoint_path)
            else:
                checkpoint_path = models_dir / f"checkpoint_episode_{episode + 1}"
                agent.save_models(str(checkpoint_path))
            logger.info(f"Checkpoint guardado: {checkpoint_path}")
    
    # Guardado final
    if config.storage_mode == "gcp":
        final_model_path = f"{model_prefix}/final_model.pth"
    else:
        final_model_path = models_dir / "final_model.pth"
    agent.save(final_model_path)
    
    total_time = time.time() - start_time
    logger.info(f"\n=== Entrenamiento Completado ===")
    logger.info(f"Tiempo total: {total_time/3600:.2f} horas")
    logger.info(f"Return promedio: {np.mean(episode_returns):.2f}")
    logger.info(f"Profit promedio: {np.mean(episode_profits):.2f}%")
    logger.info(f"Mejor evaluación: {best_eval_return:.2f}")
    logger.info(f"Modelo final guardado: {final_model_path}")
    
    # Cerrar TensorBoard writer
    writer.close()
    
    # Sincronizar logs de TensorBoard a GCS si está configurado
    if config.storage_mode == "gcp" and log_dir:
        try:
            logger.info(f"Sincronizando logs de TensorBoard a GCS...")
            gcs_utils = GCSUtils()
            tensorboard_prefix = f"tensorboard_logs/{log_dir.name}"
            gcs_utils.upload_directory_to_gcs(
                local_directory_path=str(log_dir),
                gcs_prefix=tensorboard_prefix
            )
            logger.info(f"Logs de TensorBoard sincronizados exitosamente a gs://{config.gcs_bucket_name}/{tensorboard_prefix}")
        except Exception as e:
            logger.error(f"Error al sincronizar logs de TensorBoard a GCS: {e}")
            logger.warning("El entrenamiento se completó correctamente, pero los logs no se pudieron sincronizar")


def validate_start_date(date_string: str) -> bool:
    """
    Valida que la fecha de inicio tenga el formato correcto.
    
    Args:
        date_string (str): Fecha en formato YYYY-MM-DD
        
    Returns:
        bool: True si es válida, False en caso contrario
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def find_latest_checkpoint_info(checkpoint_search_base: str, logger: logging.Logger) -> Optional[Tuple[str, int]]:
    """
    Orquestar la búsqueda del último checkpoint, ya sea localmente o en GCS.
    
    Args:
        checkpoint_search_base (str): Si es local, es la ruta al directorio base de modelos.
                                    Si es GCS, es el prefijo de la carpeta de checkpoints en GCS.
        logger (logging.Logger): Logger para mensajes
    
    Returns:
        Optional[Tuple[str, int]]: Tupla con el prefijo completo para cargar el checkpoint
                                 y el número de episodio, o None si no se encuentra ningún checkpoint
    """
    logger.info(f"Buscando checkpoints en: {checkpoint_search_base}")
    
    try:
        if config.storage_mode == "gcp":
            # Modo GCS: usar GCSUtils para buscar
            from src.configuration.gcs_utils import gcs_utils
            return gcs_utils.find_latest_checkpoint_gcs_info(checkpoint_search_base)
            
        else:
            # Modo local: buscar en directorio local
            checkpoint_dir = Path(checkpoint_search_base)
            
            if not checkpoint_dir.exists():
                logger.info(f"Directorio de checkpoints no existe: {checkpoint_search_base}")
                return None
            
            # Buscar archivos de metadata de checkpoint usando glob
            metadata_files = list(checkpoint_dir.glob("checkpoint_episode_*_metadata.pkl"))
            
            if not metadata_files:
                logger.info(f"No se encontraron checkpoints en: {checkpoint_search_base}")
                return None
            
            # Patrón regex para extraer el nombre base del checkpoint y el número de episodio
            metadata_pattern = re.compile(r"(checkpoint_episode_(\d+))_metadata\.pkl$")
            
            latest_episode_number = -1
            latest_checkpoint_base_name = None
            
            # Iterar sobre los archivos encontrados
            for metadata_file in metadata_files:
                filename = metadata_file.name
                match = metadata_pattern.search(filename)
                
                if match:
                    checkpoint_base_name = match.group(1)  # "checkpoint_episode_123"
                    episode_number = int(match.group(2))    # 123
                    
                    logger.debug(f"Encontrado checkpoint local: {checkpoint_base_name} (episodio {episode_number})")
                    
                    if episode_number > latest_episode_number:
                        latest_episode_number = episode_number
                        latest_checkpoint_base_name = checkpoint_base_name
            
            if latest_checkpoint_base_name is None:
                logger.info("No se encontraron checkpoints válidos localmente")
                return None
            
            # Construir el prefijo completo del checkpoint más reciente
            latest_checkpoint_full_prefix = str(checkpoint_dir / latest_checkpoint_base_name)
            
            logger.info(f"Checkpoint local más reciente: {latest_checkpoint_base_name} (episodio {latest_episode_number})")
            logger.info(f"Prefijo completo: {latest_checkpoint_full_prefix}")
            
            return (latest_checkpoint_full_prefix, latest_episode_number)
            
    except Exception as e:
        logger.error(f"Error al buscar checkpoints: {e}")
        return None


def main():
    """Función principal del script."""
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== Iniciando Bot de Trading de Bitcoin ===")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Validar fecha de inicio
    if not validate_start_date(args.start_date):
        logger.error(f"Formato de fecha inválido: {args.start_date}. Use YYYY-MM-DD")
        sys.exit(1)
    
    logger.info(f"Parámetros: Symbol={args.symbol}, Interval={args.interval}, Start Date={args.start_date}")

    # Inicializar TensorBoard Writer
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_name = f"SAC_Transformer_{args.symbol}_{args.interval}_{current_time}"
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs se guardarán en: {log_dir}")

    # Registrar Hiperparámetros
    hparams = {
        'symbol': args.symbol,
        'interval': args.interval,
        'start_date': args.start_date,
        'episodes': args.episodes,
        'eval_frequency': args.eval_frequency,
        'save_frequency': args.save_frequency,
        'actor_lr': config.actor_learning_rate,
        'critic_lr': config.critic_learning_rate,
        'alpha_lr': config.alpha_learning_rate,
        'gamma': config.gamma,
        'tau': config.tau,
        'batch_size': config.batch_size,
        'buffer_size': config.replay_buffer_size,
        'd_model': config.d_model,
        'n_head': config.n_head,
        'num_encoder_layers': config.num_encoder_layers,
        'ventana_observacion': config.ventana_observacion_size,
        'capital_inicial': config.capital_inicial,
        'apalancamiento': config.apalancamiento
    }
    # Log hyperparameters as text for now (we'll link to metrics later)
    writer.add_text('Hyperparameters', str(hparams), 0)
    
    try:
        # 1. Adquisición de datos
        logger.info("=== FASE 1: Adquisición de Datos ===")
        adquisicion = Adquisicion(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date
        )
        
        # Ejecutar proceso de adquisición
        dataframe = adquisicion.main()

        logger.info(f"Datos adquiridos exitosamente:")
        logger.info(f"  - Forma del DataFrame: {dataframe.shape}")
        logger.info(f"  - Rango temporal: {dataframe.index.min()} a {dataframe.index.max()}")
        logger.info(f"  - Columnas: {list(dataframe.columns)}")
        logger.info(f"  - Memoria utilizada: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar estadísticas básicas
        logger.info("Estadísticas básicas del DataFrame:")
        logger.info(f"\n{dataframe.describe()}")
        
        # 2. Cálculo de Indicadores Técnicos
        logger.info("=== FASE 2: Cálculo de Indicadores Técnicos ===")
        indicadores = Indicadores(dataframe)
        
        # Ejecutar proceso de cálculo de indicadores
        dataframe_with_indicators = indicadores.main()
        
        logger.info(f"Indicadores calculados exitosamente:")
        logger.info(f"  - Forma del DataFrame: {dataframe_with_indicators.shape}")
        logger.info(f"  - Columnas totales: {len(dataframe_with_indicators.columns)}")
        logger.info(f"  - Nuevas columnas de indicadores: {len(dataframe_with_indicators.columns) - len(dataframe.columns)}")
        logger.info(f"  - Memoria utilizada: {dataframe_with_indicators.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar las nuevas columnas
        original_columns = set(dataframe.columns)
        new_columns = [col for col in dataframe_with_indicators.columns if col not in original_columns]
        if new_columns:
            logger.info(f"  - Indicadores añadidos: {new_columns}")
        
        # Actualizar referencia al dataframe
        dataframe = dataframe_with_indicators
        
        # 3. Normalización de Datos
        logger.info("=== FASE 3: Normalización de Datos ===")
        normalization = Normalization(dataframe)
        
        # Ejecutar proceso de normalización
        normalized_dataframe, scaler = normalization.main()
        
        logger.info(f"Normalización completada exitosamente:")
        logger.info(f"  - Forma del DataFrame normalizado: {normalized_dataframe.shape}")
        logger.info(f"  - Rango de valores: [{normalized_dataframe.min().min():.6f}, {normalized_dataframe.max().max():.6f}]")
        logger.info(f"  - Scaler guardado en: {normalization.scaler_path}")
        logger.info(f"  - Memoria utilizada: {normalized_dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar información del scaler
        feature_info = normalization.get_feature_info()
        logger.info(f"  - Características normalizadas: {feature_info['num_features']}")
        logger.info(f"  - Tipo de scaler: {feature_info['scaler_type']}")
        logger.info(f"  - Rango de normalización: {feature_info['feature_range']}")
        
        # Actualizar referencia al dataframe
        dataframe = normalized_dataframe
        
        # 4. Entrenamiento del Modelo SAC
        logger.info("=== FASE 4: Entrenamiento del Modelo SAC ===")
        
        # Configurar device
        device = setup_device(args.no_cuda)
        logger.info(f"Usando device: {device}")
        
        # Crear entorno de trading
        env = create_trading_environment(dataframe, logger)
        
        # Crear y entrenar agente
        agent = create_sac_agent(env, device, logger)
        
        # Variables para resumir entrenamiento
        start_episode = 0
        model_save_base_prefix = f"models/{args.symbol}_{args.interval}_{current_time}"
        
        # Buscar checkpoint previo para resumir entrenamiento
        logger.info("\n=== BÚSQUEDA DE CHECKPOINT PARA RESUMIR ENTRENAMIENTO ===")
        
        if config.storage_mode == "gcp":
            checkpoint_search_base = "models"  # Buscar en la carpeta de modelos en GCS
        else:
            checkpoint_search_base = "models"  # Buscar en directorio local models/
        
        checkpoint_info = find_latest_checkpoint_info(checkpoint_search_base, logger)
        
        if checkpoint_info:
            checkpoint_prefix, latest_episode = checkpoint_info
            
            logger.info(f"Se encontró checkpoint del episodio {latest_episode}")
            logger.info(f"Ubicación: {checkpoint_prefix}")
            
            # Preguntar al usuario si desea resumir
            response = input(f"\n¿Desea resumir el entrenamiento desde el episodio {latest_episode}? (y/n): ").strip().lower()
            
            if response in ['y', 'yes', 'sí', 'si', 's']:
                try:
                    logger.info(f"Cargando checkpoint desde: {checkpoint_prefix}")
                    agent.load_models(checkpoint_prefix)
                    
                    start_episode = latest_episode
                    model_save_base_prefix = checkpoint_prefix.rsplit('_episode_', 1)[0] if '_episode_' in checkpoint_prefix else model_save_base_prefix
                    
                    logger.info(f"✅ Checkpoint cargado exitosamente")
                    logger.info(f"  - Episodio inicial: {start_episode}")
                    logger.info(f"  - Total steps: {agent.total_steps}")
                    logger.info(f"  - Learning steps: {agent.learning_steps}")
                    logger.info(f"  - Prefijo de guardado: {model_save_base_prefix}")
                    
                except Exception as e:
                    logger.error(f"❌ Error al cargar checkpoint: {e}")
                    logger.info("Continuando con entrenamiento desde cero...")
                    start_episode = 0
            else:
                logger.info("Iniciando entrenamiento desde cero (no se cargó checkpoint)")
        else:
            logger.info("No se encontraron checkpoints previos. Iniciando entrenamiento desde cero.")
        
        logger.info(f"\n=== CONFIGURACIÓN FINAL DE ENTRENAMIENTO ===")
        logger.info(f"  - Episodio inicial: {start_episode}")
        logger.info(f"  - Episodios totales: {args.episodes}")
        logger.info(f"  - Episodios por entrenar: {args.episodes - start_episode}")
        logger.info(f"  - Prefijo de guardado: {model_save_base_prefix}")
        
        # Verificar que queden episodios por entrenar
        if start_episode >= args.episodes:
            logger.warning(f"El checkpoint ya alcanzó o superó el número de episodios objetivo ({args.episodes})")
            logger.info("No hay episodios adicionales para entrenar. Terminando...")
            return
        
        # Ejecutar entrenamiento
        train_agent(
            agent=agent,
            env=env,
            num_episodes=args.episodes,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes,
            save_frequency=args.save_frequency,
            logger=logger,
            writer=writer,
            model_prefix=f"models/{args.symbol}_{args.interval}_{current_time}",
            log_dir=log_dir,
            start_episode=start_episode,
            model_save_base_prefix_arg=model_save_base_prefix
        )
        
        logger.info("=== Proceso Completado Exitosamente ===")
        
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        writer.close()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        logger.exception("Detalles del error:")
        writer.close()
        sys.exit(1)
    
    finally:
        # Asegurar que el writer se cierre
        if 'writer' in locals():
            writer.close()


if __name__ == "__main__":
    main()