"""
Script de trial para optimización de hiperparámetros con Vertex AI Hypertune.
Ejecuta un ciclo de vida completo pero abreviado: cargar datos -> entrenar (corto) -> evaluar -> reportar métrica.
"""

import os
import sys
import logging
import tempfile
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

# Importar hypertune para reportar métricas a Vertex AI
try:
    import hypertune
except ImportError:
    print("Warning: hypertune library not found. Install with: pip install cloudml-hypertune")
    hypertune = None

from src.data.artifact_manager import ArtifactManager
from src.entorno.factory import create_trading_environment
from src.agente.factory import create_sac_agent
from src.agente.replay_buffer import PrioritizedReplayBuffer
from src.agente.observation_parser import parse_observation
from src.utils.observation_builder import ObservationBuilder
from src.utils.system import setup_logging, set_seed, setup_device
from src.utils.cli import parse_hypertune_arguments
from src.training import AgentEvaluator
from src.configuration.config_manager import ConfigManager
from src.configuration.secret_utils import SecretManagerUtils
from src.configuration import AppConfig, EnvironmentConfig, AgentConfig
from src.configuration.constants import (
    CONFIG_PATH_DEFAULT, STORAGE_MODE_GCP, DEFAULT_NETWORK_INTERFACE,
    FILE_PRICE_SCALER
)


def setup_agent_config_from_args(args) -> AgentConfig:
    """
    Crea una configuración de agente basada en los argumentos de línea de comandos.
    
    Args:
        args: Argumentos parseados de la CLI
        
    Returns:
        AgentConfig: Configuración del agente con los hiperparámetros especificados
    """
    # Cargar configuración base desde el archivo
    try:
        base_config = AppConfig.from_yaml_file('src/configuration/config.yaml')
        agent_config = base_config.agent
    except FileNotFoundError:
        logging.error("❌ No se encontró el archivo 'src/configuration/config.yaml'. Abortando.")
        sys.exit(1)
    
    # Sobrescribir con los hiperparámetros del trial
    agent_config.hiperparametros_sac.actor_learning_rate = args.actor_learning_rate
    agent_config.hiperparametros_sac.critic_learning_rate = args.critic_learning_rate
    agent_config.hiperparametros_sac.alpha_learning_rate = args.alpha_learning_rate
    agent_config.batch_size = args.batch_size
    agent_config.hiperparametros_sac.tau = args.tau
    agent_config.hiperparametros_sac.per_alpha = args.per_alpha
    agent_config.hiperparametros_sac.per_beta = args.per_beta
    
    return agent_config


def simple_training_loop(agent, env, episodes: int, min_buffer_size: int, device: torch.device, logger: logging.Logger):
    """
    Ejecuta un bucle de entrenamiento simplificado para el trial.
    Replica la lógica esencial del Trainer pero de forma más ligera.
    
    Args:
        agent: El agente SAC
        env: El entorno de trading
        episodes: Número de episodios a entrenar
        min_buffer_size: Tamaño mínimo del buffer para empezar a aprender
        device: Dispositivo de torch
        logger: Logger para mensajes
    """
    logger.info(f"🚀 Iniciando entrenamiento simplificado para {episodes} episodios")
    
    # Crear replay buffer con la capacidad correcta desde la configuración del agente
    replay_buffer = PrioritizedReplayBuffer(
        capacity=agent.config.replay_buffer_size,
        observation_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        alpha=agent.config.hiperparametros_sac.per_alpha,
        beta=agent.config.hiperparametros_sac.per_beta
    )
    
    learning_started = False
    
    for episode in range(episodes):
        # Reset environment con semilla específica del episodio
        current_episode_seed = 42 + episode  # Semilla base + episodio
        obs, _ = env.reset(seed=current_episode_seed)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Parse observation y seleccionar acción
            market_data, portfolio_data = parse_observation(obs, env.config_entorno, agent.config, device)
            action = agent.select_action(market_data, portfolio_data, deterministic=False)
            
            # Ejecutar paso en el entorno
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Almacenar experiencia en el replay buffer
            replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)
            
            # Aprender si el buffer tiene suficientes muestras
            if replay_buffer.can_sample(agent.config.batch_size) and len(replay_buffer) >= min_buffer_size:
                # Log cuando inicia el aprendizaje por primera vez
                if not learning_started:
                    logger.info(f"🎯 INICIANDO APRENDIZAJE: Buffer alcanzó {len(replay_buffer)} experiencias (mínimo: {min_buffer_size})")
                    learning_started = True
                
                # Muestrear batch del replay buffer priorizado
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminated, batch_truncated, tree_indices, is_weights = replay_buffer.sample(
                    agent.config.batch_size, device
                )
                
                # Parse batch observations
                from src.agente.observation_parser import parse_observation_batch
                batch_market_data, batch_portfolio_data = parse_observation_batch(
                    batch_obs, env.config_entorno, agent.config, len(env.column_names)
                )
                batch_next_market_data, batch_next_portfolio_data = parse_observation_batch(
                    batch_next_obs, env.config_entorno, agent.config, len(env.column_names)
                )
                
                # Aprender del batch con los datos parseados correctamente
                result = agent.learn(
                    batch_market_data, batch_portfolio_data, batch_actions,
                    batch_rewards, batch_next_market_data, batch_next_portfolio_data,
                    batch_terminated, batch_truncated, is_weights
                )
                
                # Actualizar prioridades si hay TD errors disponibles
                if result and len(result) > 1:
                    losses, q_values = result
                    if 'td_errors' in losses:
                        replay_buffer.update_priorities(tree_indices, losses['td_errors'])
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        # Log progreso cada 50 episodios
        if (episode + 1) % 50 == 0:
            logger.info(f"Episodio {episode + 1}/{episodes} - Reward: {episode_reward:.4f} - Steps: {steps}")
    
    logger.info(f"✅ Entrenamiento completado. Total episodios: {episodes}")


def main():
    """Función principal del script de trial para Hypertune."""
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 === TRIAL DE HYPERTUNE INICIADO ===")
    
    # Inicializar Hypertune
    hpt = None
    if hypertune:
        hpt = hypertune.HyperTune()
        logger.info("✅ Hypertune inicializado correctamente")
    else:
        logger.warning("⚠️ Hypertune no disponible - ejecutando en modo de desarrollo")
    
    # Parsear argumentos
    args = parse_hypertune_arguments()
    logger.info(f"📋 Argumentos del trial:")
    logger.info(f"  - Train Data Run ID: {args.train_data_run_id}")
    logger.info(f"  - Eval Data Run ID: {args.eval_data_run_id}")
    logger.info(f"  - Episodes: {args.episodes}")
    logger.info(f"  - Hiperparámetros:")
    logger.info(f"    * Actor LR: {args.actor_learning_rate}")
    logger.info(f"    * Critic LR: {args.critic_learning_rate}")
    logger.info(f"    * Alpha LR: {args.alpha_learning_rate}")
    logger.info(f"    * Batch Size: {args.batch_size}")
    logger.info(f"    * Tau: {args.tau}")
    logger.info(f"    * PER Alpha: {args.per_alpha}")
    logger.info(f"    * PER Beta: {args.per_beta}")
    
    # Configurar dispositivo y semilla
    device = setup_device()
    set_seed(42, logger)
    
    try:
        # === CARGAR CONFIGURACIÓN BASE ===
        logger.info("📊 Cargando configuración base...")
        base_config = AppConfig.from_yaml_file('src/configuration/config.yaml')
        
        # === CARGAR DATOS ===
        logger.info("📂 Cargando datos de entrenamiento y evaluación...")
        
        # Preparar configuración GCP si es necesario
        gcp_config_for_load = None
        if base_config.normalization.storage_mode == STORAGE_MODE_GCP:
            gcp_config_for_load = base_config.gcp.model_dump()
        
        # Instanciar ArtifactManager
        artifact_manager = ArtifactManager(
            storage_mode=base_config.normalization.storage_mode,
            gcp_config=gcp_config_for_load
        )
        
        # Cargar artefactos de entrenamiento (dataframe, scaler, price_scaler)
        logger.info(f"📈 Cargando artefactos de entrenamiento: {args.train_data_run_id}")
        train_df = artifact_manager.load_dataframe(args.train_data_run_id)
        scaler = artifact_manager.load_scaler(args.train_data_run_id)
        price_scaler = artifact_manager.load_price_scaler(args.train_data_run_id)
        
        # Cargar solo el dataframe de evaluación
        logger.info(f"📉 Cargando dataframe de evaluación: {args.eval_data_run_id}")
        eval_df = artifact_manager.load_dataframe(args.eval_data_run_id)
        
        logger.info(f"✅ Datos cargados - Train: {len(train_df)} filas, Eval: {len(eval_df)} filas")
        
        # === CREAR CONFIGURACIÓN DEL AGENTE ===
        agent_config = setup_agent_config_from_args(args)
        
        # === CREAR ENTORNO DE ENTRENAMIENTO ===
        logger.info("🏗️ Creando entorno de entrenamiento...")
        
        # La factoría se encarga de crear el ObservationBuilder internamente.
        # Preparamos la configuración completa del run como espera la factoría.
        full_run_config_dict = {'config': base_config.model_dump()}

        train_env = create_trading_environment(
            dataframe=train_df,
            logger=logger,
            price_scaler=price_scaler,
            scaler=scaler,
            env_config=base_config.environment,
            run_config=full_run_config_dict
        )
        
        # === CREAR AGENTE ===
        logger.info("🤖 Creando agente SAC...")
        agent = create_sac_agent(
            env=train_env,  # Pasar env completo, no observation_space/action_space por separado
            device=device,
            logger=logger,
            agent_config=agent_config,
            is_distributed=False
        )
        
        # === ENTRENAMIENTO SIMPLIFICADO ===
        logger.info("🎯 Iniciando fase de entrenamiento...")
        simple_training_loop(
            agent=agent,
            env=train_env,
            episodes=args.episodes,
            min_buffer_size=agent_config.min_buffer_for_learning,
            device=device,
            logger=logger
        )
        
        # === CREAR ENTORNO DE EVALUACIÓN ===
        logger.info("🔍 Creando entorno de evaluación...")
        
        # La factoría se encarga de crear el ObservationBuilder internamente.
        eval_env = create_trading_environment(
            dataframe=eval_df,
            logger=logger,
            price_scaler=price_scaler, # Usar el mismo price_scaler del entrenamiento
            scaler=scaler, # Usar el mismo scaler del entrenamiento
            env_config=base_config.environment,
            run_config=full_run_config_dict
        )
        
        # === EVALUACIÓN ===
        logger.info("📊 Iniciando evaluación...")
        evaluator = AgentEvaluator()  # Usar constructor por defecto
        
        # Ejecutar evaluación (método actualizado sin argumentos adicionales)
        metrics, equity_curve, trades_pnl = evaluator.evaluate(agent=agent, env=eval_env)
        
        # === REPORTAR MÉTRICA ===
        sortino_ratio = metrics.get('sortino_ratio', 0.0)
        logger.info(f"📈 Sortino Ratio obtenido: {sortino_ratio:.6f}")
        
        # Reportar a Hypertune con los argumentos correctos
        if hpt:
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='sortino_ratio',
                metric_value=sortino_ratio,
                global_step=args.episodes
            )
            logger.info("✅ Métrica reportada a Vertex AI Hypertune")
        else:
            logger.info("📝 Métrica calculada (modo desarrollo - no reportada a Hypertune)")
        
        # Log resumen de métricas adicionales
        logger.info("📋 Resumen de métricas:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  - {metric_name}: {value:.6f}")
        
        logger.info("🎉 === TRIAL DE HYPERTUNE COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        logger.error(f"❌ Error durante el trial: {str(e)}")
        logger.exception("Detalles del error:")
        
        # Reportar métrica de fallo si es posible
        if hpt:
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='sortino_ratio',
                metric_value=-999.0,  # Valor que indica fallo
                global_step=args.episodes
            )
        
        sys.exit(1)


if __name__ == "__main__":
    main()
