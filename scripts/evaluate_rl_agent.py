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

# Añadir el directorio raíz del proyecto al path de Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.environments.trading_env import TradingEnvironment
from src.utils.config import ConfigManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvaluateRLAgent")


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
        required=True,
        help="Ruta al modelo entrenado para evaluar"
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
    
    return parser.parse_args()


def visualize_episode_results(episode_data, output_dir, episode_idx):
    """
    Visualiza los resultados de un episodio.
    
    Args:
        episode_data: Datos recopilados durante un episodio de evaluación
        output_dir: Directorio donde guardar las visualizaciones
        episode_idx: Índice del episodio
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Construir un DataFrame para facilitar la visualización
    df = pd.DataFrame(episode_data)
    
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
    
    # Imprimir estadísticas de rendimiento
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    logger.info(f"Episodio {episode_idx+1} - Retorno Total: {total_return:.2f}%")
    logger.info(f"Valor Inicial: ${initial_value:.2f}, Valor Final: ${final_value:.2f}")
    
    return {
        'episode': episode_idx + 1,
        'total_return': total_return,
        'initial_value': initial_value,
        'final_value': final_value,
        'avg_reward': df['rewards'].mean(),
        'cumulative_reward': df['rewards'].sum()
    }


def evaluate_agent(agent_manager, env, num_episodes=1, output_dir="results"):
    """
    Evalúa un agente en el entorno especificado.
    
    Args:
        agent_manager: Instancia de RLAgentManager con el modelo cargado
        env: Entorno de evaluación
        num_episodes: Número de episodios para evaluar
        output_dir: Directorio donde guardar los resultados
    
    Returns:
        Resumen de resultados de evaluación
    """
    logger.info(f"Evaluando agente durante {num_episodes} episodios...")
    
    all_episode_stats = []
    
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
        
        # Bucle principal del episodio
        while not (done or truncated):
            action = agent_manager.predict_action(observation, deterministic=True)
            
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Recopilar datos para visualización
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(float(action[0]))
            episode_data['position'].append(info.get('position', 0))
            episode_data['portfolio_value'].append(info.get('portfolio_value', 0))
            episode_data['market_price'].append(info.get('current_price', 0))
            
            observation = next_observation
        
        # Visualizar y guardar resultados del episodio
        episode_stats = visualize_episode_results(episode_data, output_dir, episode_idx)
        all_episode_stats.append(episode_stats)
    
    # Crear un resumen de todos los episodios
    summary_df = pd.DataFrame(all_episode_stats)
    summary_df.to_csv(f"{output_dir}/resumen_evaluacion.csv", index=False)
    
    # Imprimir estadísticas de rendimiento promedio
    avg_return = summary_df['total_return'].mean()
    avg_reward = summary_df['avg_reward'].mean()
    
    logger.info(f"=== Resumen de Evaluación ({num_episodes} episodios) ===")
    logger.info(f"Retorno Promedio: {avg_return:.2f}%")
    logger.info(f"Recompensa Promedio por Paso: {avg_reward:.4f}")
    
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
    
    # Cargar el modelo entrenado
    agent_manager.setup_agent(
        load_model=True,
        model_path=args.model_path
    )
    
    # Configurar el entorno de evaluación (modo determinístico)
    # Pasar directamente la ruta de configuración y establecer el modo de renderización
    eval_env = TradingEnvironment(config_path=args.config, render_mode='human')
    
    # Evaluar el agente
    evaluate_agent(
        agent_manager=agent_manager,
        env=eval_env,
        num_episodes=args.episodes,
        output_dir=args.output_path
    )
    
    logger.info("Evaluación completada.")


if __name__ == "__main__":
    main()
