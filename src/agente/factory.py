"""
Factory module for creating SAC agent instances.
Centralizes the creation logic for TransformerSACAgent to improve modularity.
"""

import torch
import logging
from typing import Any

from .agent import TransformerSACAgent
from .networks import ActorNetwork, CriticNetwork
from ..entorno.environment import FuturesTradingEnv
from ..configuration import AgentConfig


def create_sac_agent(
    env: FuturesTradingEnv, 
    device: torch.device, 
    logger: logging.Logger, 
    agent_config: AgentConfig, 
    is_distributed: bool = False
) -> TransformerSACAgent:
    """
    Crea el agente SAC con arquitectura Transformer.
    
    Args:
        env: Entorno de trading
        device: Device para el entrenamiento
        logger: Logger para mensajes
        agent_config: Configuración del agente (objeto Pydantic)
        is_distributed: Si el entrenamiento es distribuido
        
    Returns:
        TransformerSACAgent: Agente configurado
    """
    logger.info("Creando agente SAC con Transformer...")
    
    # Obtener parámetros del entorno
    observation_space_shape = env.observation_space.shape
    action_space_shape = env.action_space.shape
    
    # Calcular características de mercado y portfolio
    ventana_size = env.config_entorno.ventana_observacion_size
    num_features_mercado = len(env.column_names)
    market_features = num_features_mercado
    # Leer portfolio_features desde la configuración del agente
    portfolio_features = agent_config.architecture.portfolio_features
    sequence_length = ventana_size
    action_dim = action_space_shape[0]
    
    # Extraer configuraciones
    transformer_config = agent_config.transformer
    mlp_heads_config = agent_config.mlp_heads
    mlp_hidden_dims = mlp_heads_config.hidden_dims
    
    # Crear las redes neuronales (Inversión de Dependencias)
    logger.info("Creando redes neuronales...")
    
    # Red del Actor
    actor = ActorNetwork(
        market_features=market_features,
        portfolio_features=portfolio_features,
        transformer_config=transformer_config,
        mlp_hidden_dims=mlp_hidden_dims,
        action_dim=action_dim,
        agent_config=agent_config
    )
    logger.info("✅ ActorNetwork creado.")

    # Redes de los Críticos
    critic_1 = CriticNetwork(
        market_features=market_features,
        portfolio_features=portfolio_features,
        action_dim=action_dim,
        transformer_config=transformer_config,
        mlp_hidden_dims=mlp_hidden_dims,
        agent_config=agent_config
    )
    
    critic_2 = CriticNetwork(
        market_features=market_features,
        portfolio_features=portfolio_features,
        action_dim=action_dim,
        transformer_config=transformer_config,
        mlp_hidden_dims=mlp_hidden_dims,
        agent_config=agent_config
    )
    logger.info("✅ CriticNetworks creados.")
    
    # Crear el agente con las redes inyectadas (Dependency Injection)
    agent = TransformerSACAgent(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        config_override=agent_config,
        device=device,
        is_distributed=is_distributed
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
    logger.info(f"  - Gamma: {agent_config.hiperparametros_sac.gamma}")
    logger.info(f"  - Tau: {agent_config.hiperparametros_sac.tau}")
    logger.info(f"  - Alpha inicial: {agent_config.hiperparametros_sac.initial_log_alpha}")
    logger.info(f"  - Learning rates: Actor={agent_config.hiperparametros_sac.actor_learning_rate}, Critic={agent_config.hiperparametros_sac.critic_learning_rate}")
    logger.info(f"  - Entrenamiento distribuido: {'Sí' if is_distributed else 'No'}")
    
    return agent
