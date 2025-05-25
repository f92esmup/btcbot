#!/usr/bin/env python3
"""
Script para evaluar un agente de RL previamente entrenado
en el entorno de trading simulado.
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import uuid
import time
from typing import Dict, Any, List

# Añadir el directorio raíz del proyecto al path de Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.environments.trading_env import TradingEnvironment
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger, get_madrid_timestamp_str, get_madrid_timestamp
from src.utils.bigquery_utils import stream_data_to_bigquery
from src.callbacks.bigquery_evaluation_schema import EVALUATION_LOG_SCHEMA, EVALUATION_STEP_SCHEMA
from dotenv import load_dotenv
from google.cloud import bigquery

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logger = setup_logger("EvaluateRLAgent")


def parse_arguments():
    """
    Parsea los argumentos de la línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Evalúa un agente de RL para trading")
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/config.yaml",
        help="Ruta al archivo de configuración centralizada"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip", # No se usa el sufijo gs//
        help="Ruta en GCS al modelo entrenado para evaluar (formato: path/to/model.zip)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Número de episodios para evaluar"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Ruta donde guardar los resultados de evaluación"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Desactivar el uso de GPU incluso si está disponible"
    )
    
    parser.add_argument(
        "--no-bigquery",
        action="store_true",
        help="Desactivar el guardado de resultados en BigQuery"
    )
    
    return parser.parse_args()


def calculate_trading_metrics(episode_data: Dict[str, List]) -> Dict[str, Any]:
    """
    Calcula métricas de trading detalladas a partir de los datos del episodio.
    
    Args:
        episode_data: Datos recopilados durante el episodio
        
    Returns:
        Diccionario con métricas calculadas
    """
    try:
        # Convertir a arrays para facilitar cálculos
        rewards = np.array(episode_data['rewards'])
        actions = np.array(episode_data['actions'])
        positions = np.array(episode_data['position'])
        portfolio_values = np.array(episode_data['portfolio_value'])
        prices = np.array(episode_data['market_price'])
        
        # Métricas básicas
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_percent': 0.0,
            'max_drawdown_percent': 0.0,
            'max_profit_percent': 0.0,
            'sharpe_ratio': 0.0,
            'avg_position_duration_steps': 0.0,
            'max_position_duration_steps': 0,
            'long_positions_count': 0,
            'short_positions_count': 0,
        }
        
        if len(positions) == 0:
            return metrics
            
        # Detectar cambios de posición (trades)
        position_changes = np.diff(positions, prepend=positions[0])
        trade_indices = np.where(position_changes != 0)[0]
        
        if len(trade_indices) > 0:
            metrics['total_trades'] = len(trade_indices)
            
            # Analizar duración de posiciones
            durations = []
            current_pos_start = 0
            
            for i, change_idx in enumerate(trade_indices):
                if i > 0:  # No contar la posición inicial como trade
                    duration = change_idx - current_pos_start
                    durations.append(duration)
                current_pos_start = change_idx
            
            # Duración de la última posición si termina en posición
            if len(positions) > 0 and positions[-1] != 0:
                durations.append(len(positions) - current_pos_start)
            
            if durations:
                metrics['avg_position_duration_steps'] = float(np.mean(durations))
                metrics['max_position_duration_steps'] = int(np.max(durations))
            
            # Contar tipos de posiciones
            for pos in positions:
                if pos > 0:
                    metrics['long_positions_count'] += 1
                elif pos < 0:
                    metrics['short_positions_count'] += 1
        
        # Calcular drawdown y profit máximos
        if len(portfolio_values) > 0:
            initial_value = portfolio_values[0]
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max * 100
            profits = (portfolio_values - initial_value) / initial_value * 100
            
            metrics['max_drawdown_percent'] = float(np.min(drawdowns))
            metrics['max_profit_percent'] = float(np.max(profits))
        
        # Calcular Sharpe ratio simplificado
        if len(rewards) > 1:
            mean_return = np.mean(rewards)
            std_return = np.std(rewards)
            if std_return > 0:
                metrics['sharpe_ratio'] = float(mean_return / std_return)
        
        # Calcular win rate basado en rewards positivos en trades
        if len(rewards) > 0:
            positive_rewards = np.sum(rewards > 0)
            negative_rewards = np.sum(rewards < 0)
            metrics['winning_trades'] = int(positive_rewards)
            metrics['losing_trades'] = int(negative_rewards)
            
            if (positive_rewards + negative_rewards) > 0:
                metrics['win_rate_percent'] = float(positive_rewards / (positive_rewards + negative_rewards) * 100)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculando métricas de trading: {e}", exc_info=True)
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_percent': 0.0,
            'max_drawdown_percent': 0.0,
            'max_profit_percent': 0.0,
            'sharpe_ratio': 0.0,
            'avg_position_duration_steps': 0.0,
            'max_position_duration_steps': 0,
            'long_positions_count': 0,
            'short_positions_count': 0,
        }
def visualize_episode_results(episode_data, output_dir, episode_idx, save_files=False):
    """
    Visualiza los resultados de un episodio y calcula métricas.
    
    Args:
        episode_data: Datos recopilados durante un episodio de evaluación
        output_dir: Directorio donde guardar las visualizaciones (solo si save_files=True)
        episode_idx: Índice del episodio
        save_files: Si guardar archivos locales (CSV y gráficos)
    """
    # Construir un DataFrame para facilitar los cálculos
    df = pd.DataFrame(episode_data)
    
    # Solo crear archivos si se solicita explícitamente
    if save_files:
        # Crear el directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Figura 1: PnL acumulado y valor del portafolio
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['portfolio_value'], label='Valor del Portafolio ($)')
        plt.title(f'Evaluación del Agente RL - Episodio {episode_idx+1}')
        plt.ylabel('Valor ($)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df['position'], label='Posición')
        plt.plot(df['actions'], label='Señal de Acción', linestyle='--', alpha=0.7)
        plt.xlabel('Paso Temporal')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        # Guardar la figura
        plt.tight_layout()
        plt.savefig(f"{output_dir}/episodio_{episode_idx+1}_resultados.png")
        plt.close()
        
        # También guardar los datos numéricos
        df.to_csv(f"{output_dir}/episodio_{episode_idx+1}_datos.csv", index=False)
        
        logger.info(f"Archivos guardados para episodio {episode_idx+1} en {output_dir}")
    
    # Imprimir estadísticas de rendimiento
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    logger.info(f"Episodio {episode_idx+1} - Retorno Total: {total_return:.2f}%")
    logger.info(f"Valor Inicial: ${initial_value:.2f}, Valor Final: ${final_value:.2f}")
    
    # Calcular métricas de trading adicionales
    trading_metrics = calculate_trading_metrics(episode_data)
    
    return {
        'episode': episode_idx + 1,
        'total_return': total_return,
        'initial_value': initial_value,
        'final_value': final_value,
        'avg_reward': df['rewards'].mean(),
        'cumulative_reward': df['rewards'].sum(),
        **trading_metrics
    }


def evaluate_agent(agent_manager, env, num_episodes=1, output_dir="results", 
                  save_to_bigquery=True, config_manager=None, model_path=None):
    """
    Evalúa un agente en el entorno especificado.
    
    Args:
        agent_manager: Instancia de RLAgentManager con el modelo cargado
        env: Entorno de evaluación
        num_episodes: Número de episodios para evaluar
        output_dir: Directorio donde guardar los resultados
        save_to_bigquery: Si guardar los resultados en BigQuery
        config_manager: ConfigManager para obtener configuración de BigQuery
        model_path: Ruta del modelo evaluado
    
    Returns:
        Resumen de resultados de evaluación
    """
    logger.info(f"Evaluando agente durante {num_episodes} episodios...")
    
    # Configurar BigQuery si está habilitado
    bq_client = None
    gcp_project_id = None
    bigquery_dataset_id = None
    evaluation_id = str(uuid.uuid4())
    
    if save_to_bigquery and config_manager:
        try:
            gcp_project_id = config_manager.get_env_variable('GCP_PROJECT_ID')
            bigquery_dataset_id = os.environ.get('BIGQUERY_LOG_DATASET_ID')
            
            if gcp_project_id and bigquery_dataset_id:
                bq_client = bigquery.Client(project=gcp_project_id)
                logger.info(f"BigQuery habilitado para evaluación. Proyecto: {gcp_project_id}, Dataset: {bigquery_dataset_id}")
            else:
                logger.warning("Configuración de BigQuery incompleta. Se saltará el logging a BigQuery.")
                save_to_bigquery = False
        except Exception as e:
            logger.error(f"Error configurando BigQuery: {e}. Se saltará el logging a BigQuery.")
            save_to_bigquery = False
    
    all_episode_stats = []
    
    # Crear directorio de salida independientemente del método de guardado
    os.makedirs(output_dir, exist_ok=True)
    
    for episode_idx in range(num_episodes):
        logger.info(f"Iniciando episodio {episode_idx+1}/{num_episodes}")
        
        observation, info = env.reset()
        done = False
        truncated = False
        episode_data = {
            'rewards': [],
            'actions': [],
            'position': [],
            'portfolio_value': [],
            'market_price': []
        }
        
        episode_start_time = get_madrid_timestamp()
        
        # Bucle principal del episodio
        while not (done or truncated):
            action = agent_manager.predict_action(observation, deterministic=True)
            
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Recopilar datos para visualización
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(float(action[0]))
            episode_data['position'].append(info.get('position_side', 0))
            episode_data['portfolio_value'].append(info.get('current_equity', 0))
            episode_data['market_price'].append(info.get('market_price', 0))
            
            observation = next_observation
        
        episode_end_time = get_madrid_timestamp()
        
        # Visualizar y guardar resultados del episodio
        # Solo guardar archivos locales si BigQuery no está habilitado
        save_local_files = not save_to_bigquery
        episode_stats = visualize_episode_results(episode_data, output_dir, episode_idx, save_files=save_local_files)
        all_episode_stats.append(episode_stats)
        
        # Guardar en BigQuery si está habilitado
        if save_to_bigquery and bq_client:
            try:
                # Preparar datos para BigQuery
                bq_record = {
                    'evaluation_id': evaluation_id,
                    'model_path': model_path or 'unknown',
                    'timestamp_evaluation': episode_end_time.isoformat(),
                    'config_path': config_manager.config_path if config_manager else 'unknown',
                    'num_episodes': num_episodes,
                    'episode_number': episode_idx + 1,
                    'total_return_percent': episode_stats['total_return'],
                    'initial_value': episode_stats['initial_value'],
                    'final_value': episode_stats['final_value'],
                    'avg_reward_per_step': episode_stats['avg_reward'],
                    'cumulative_reward': episode_stats['cumulative_reward'],
                    'episode_length_steps': len(episode_data['rewards']),
                    'total_trades': episode_stats.get('total_trades', 0),
                    'winning_trades': episode_stats.get('winning_trades', 0),
                    'losing_trades': episode_stats.get('losing_trades', 0),
                    'win_rate_percent': episode_stats.get('win_rate_percent', 0.0),
                    'total_pnl': episode_stats['final_value'] - episode_stats['initial_value'],
                    'max_drawdown_percent': episode_stats.get('max_drawdown_percent', 0.0),
                    'max_profit_percent': episode_stats.get('max_profit_percent', 0.0),
                    'sharpe_ratio': episode_stats.get('sharpe_ratio', 0.0),
                    'avg_position_duration_steps': episode_stats.get('avg_position_duration_steps', 0.0),
                    'max_position_duration_steps': episode_stats.get('max_position_duration_steps', 0),
                    'long_positions_count': episode_stats.get('long_positions_count', 0),
                    'short_positions_count': episode_stats.get('short_positions_count', 0),
                    'market_conditions': 'evaluation',  # Podría ser más específico
                    'start_timestamp': episode_start_time.isoformat(),
                    'end_timestamp': episode_end_time.isoformat(),
                    'device_used': agent_manager.device if hasattr(agent_manager, 'device') else 'unknown',
                    'environment_type': 'TradingEnvironment',
                    'notes': f'Evaluation episode {episode_idx + 1} of {num_episodes}'
                }
                
                # Crear tabla con fecha actual
                current_date = get_madrid_timestamp().strftime('%Y%m%d')
                table_id = f"evaluacion_{current_date}"
                
                # Enviar a BigQuery
                success = stream_data_to_bigquery(
                    project_id=gcp_project_id,
                    dataset_id=bigquery_dataset_id,
                    table_id=table_id,
                    rows_to_insert=[bq_record],
                    client=bq_client,
                    schema=EVALUATION_LOG_SCHEMA
                )
                
                if success:
                    logger.info(f"Datos del episodio {episode_idx+1} guardados en BigQuery tabla {table_id}")
                else:
                    logger.error(f"Error guardando datos del episodio {episode_idx+1} en BigQuery")
                    
            except Exception as e:
                logger.error(f"Error enviando datos a BigQuery para episodio {episode_idx+1}: {e}", exc_info=True)
    
    # Crear un resumen de todos los episodios (solo si no se usa BigQuery como método principal)
    if not save_to_bigquery:
        summary_df = pd.DataFrame(all_episode_stats)
        summary_df.to_csv(f"{output_dir}/resumen_evaluacion.csv", index=False)
        logger.info(f"Resumen de evaluación guardado en {output_dir}/resumen_evaluacion.csv")
    else:
        summary_df = pd.DataFrame(all_episode_stats)  # Para cálculos, sin guardar archivo
    
    # Imprimir estadísticas de rendimiento promedio
    avg_return = summary_df['total_return'].mean()
    avg_reward = summary_df['avg_reward'].mean()
    
    logger.info(f"=== Resumen de Evaluación ({num_episodes} episodios) ===")
    logger.info(f"Retorno Promedio: {avg_return:.2f}%")
    logger.info(f"Recompensa Promedio por Paso: {avg_reward:.4f}")
    
    if 'total_trades' in summary_df.columns:
        avg_trades = summary_df['total_trades'].mean()
        avg_win_rate = summary_df['win_rate_percent'].mean()
        logger.info(f"Trades Promedio por Episodio: {avg_trades:.1f}")
        logger.info(f"Tasa de Acierto Promedio: {avg_win_rate:.1f}%")
    
    return summary_df


def main():
    """
    Función principal para evaluar un agente de RL.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Cargar la configuración centralizada
    config_manager = ConfigManager(config_path=args.config)
    agent_config = config_manager.get_agent_config()
    
    # Verificar que GCS_BUCKET_NAME esté configurado
    gcs_bucket_name = config_manager.get_env_variable("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        logger.error("Error: GCS_BUCKET_NAME no está configurado en las variables de entorno.")
        sys.exit(1)
    
    # Verificar configuración de BigQuery
    gcp_project_id = config_manager.get_env_variable('GCP_PROJECT_ID')
    bigquery_dataset_id = os.environ.get('BIGQUERY_LOG_DATASET_ID')
    
    if not gcp_project_id or not bigquery_dataset_id:
        logger.warning("Configuración de BigQuery incompleta. Los resultados solo se guardarán localmente.")
        logger.warning("Para habilitar BigQuery, configure GCP_PROJECT_ID y BIGQUERY_LOG_DATASET_ID")
    else:
        logger.info(f"BigQuery configurado. Los resultados se guardarán en {gcp_project_id}.{bigquery_dataset_id}")
    
    # Actualizar la configuración si se solicita no usar GPU
    if args.no_gpu:
        agent_config["use_gpu"] = False
        logger.info("Uso de GPU desactivado por argumento de línea de comandos")
    
    # Crear el administrador del agente con la configuración centralizada
    agent_manager = RLAgentManager(config_path=args.config)
    
    # Si se desactivó la GPU por argumento, aplicar la configuración al administrador
    if args.no_gpu:
        agent_manager.config["use_gpu"] = False
        agent_manager.device = "cpu"
    
    # Cargar el modelo entrenado desde GCS
    logger.info(f"Cargando modelo desde GCS: {args.model_path}")
    agent_manager.setup_agent(
        load_model=True,
        model_path=args.model_path
    )
    
    # Configurar el entorno de evaluación (modo determinístico)
    # Usar el agent manager para configurar el entorno correctamente
    eval_env = agent_manager.setup_environment(is_eval=True)
    
    # Evaluar el agente
    logger.info("Iniciando evaluación del agente...")
    logger.info(f"Modelo: {args.model_path}")
    logger.info(f"Episodios: {args.episodes}")
    logger.info(f"Directorio de salida: {args.output_path}")
    
    # Determinar si usar BigQuery
    use_bigquery = not args.no_bigquery and gcp_project_id and bigquery_dataset_id
    if args.no_bigquery:
        logger.info("BigQuery desactivado por argumento de línea de comandos")
    
    evaluate_agent(
        agent_manager=agent_manager,
        env=eval_env,
        num_episodes=args.episodes,
        output_dir=args.output_path,
        save_to_bigquery=use_bigquery,
        config_manager=config_manager,
        model_path=args.model_path
    )
    
    logger.info("Evaluación completada.")
    logger.info(f"Resultados guardados en: {args.output_path}")
    
    if use_bigquery:
        current_date = get_madrid_timestamp().strftime('%Y%m%d')
        logger.info(f"Datos también disponibles en BigQuery: {gcp_project_id}.{bigquery_dataset_id}.evaluacion_{current_date}")
        logger.info("Para análisis posterior, consulte los archivos CSV generados y la tabla de BigQuery.")
    else:
        logger.info("Para análisis posterior, consulte los archivos CSV generados.")


if __name__ == "__main__":
    main()
