"""
Módulo para centralizar la lógica de parseo de observaciones del entorno.
Ubicado en el módulo del agente para mejorar la cohesión.
"""
import torch
import numpy as np
from typing import Tuple, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..configuration.config_model import EnvironmentConfig, AgentConfig


def parse_observation(
    observation: np.ndarray, 
    env_config: Union[Dict, "EnvironmentConfig"], 
    agent_config: Union[Dict, "AgentConfig"],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parsea la observación del entorno en tensores de mercado y de portfolio.

    Args:
        observation (np.ndarray): Vector de observación crudo del entorno.
        env_config (Union[Dict, EnvironmentConfig]): Configuración del entorno.
        agent_config (Union[Dict, AgentConfig]): Configuración del agente.
        device (torch.device): Dispositivo (CPU/GPU) al que mover los tensores.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Una tupla conteniendo (market_data, portfolio_data).
            - market_data: Tensor de forma (1, ventana_size, num_features_mercado)
            - portfolio_data: Tensor de forma (1, portfolio_features)
    """
    # Extraer dimensiones desde la configuración (compatible con dict y objetos Pydantic)
    if isinstance(env_config, dict):
        ventana_size = env_config['ventana_observacion_size']
    else:
        ventana_size = env_config.ventana_observacion_size
    
    if isinstance(agent_config, dict):
        portfolio_features = agent_config['architecture']['portfolio_features']
    else:
        portfolio_features = agent_config.architecture.portfolio_features

    # El número total de features en la observación es (ventana * features_mercado) + features_portfolio
    # Podemos deducir el número de features de mercado
    market_features_total = observation.shape[0] - portfolio_features
    num_features_mercado = market_features_total // ventana_size

    # Separar los datos de mercado y de portfolio
    market_data_flat = observation[:market_features_total]
    portfolio_data_flat = observation[market_features_total:]

    # Reestructurar los datos de mercado a su forma 3D
    market_data = market_data_flat.reshape(ventana_size, num_features_mercado)

    # Convertir a tensores de PyTorch, añadir dimensión de batch y mover al dispositivo correcto
    market_tensor = torch.from_numpy(market_data).float().unsqueeze(0).to(device)
    portfolio_tensor = torch.from_numpy(portfolio_data_flat).float().unsqueeze(0).to(device)

    return market_tensor, portfolio_tensor


def parse_observation_batch(
    observations_batch: torch.Tensor,
    env_config: Union[Dict, "EnvironmentConfig"],
    agent_config: Union[Dict, "AgentConfig"],
    num_market_features: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parsea un lote de observaciones del entorno en tensores de mercado y de portfolio.

    Args:
        observations_batch (torch.Tensor): Lote de observaciones con forma (batch_size, observation_size).
        env_config (Union[Dict, EnvironmentConfig]): Configuración del entorno.
        agent_config (Union[Dict, AgentConfig]): Configuración del agente.
        num_market_features (int): Número de características de mercado por timestep.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Una tupla conteniendo (market_data, portfolio_data).
            - market_data: Tensor de forma (batch_size, ventana_size, num_market_features)
            - portfolio_data: Tensor de forma (batch_size, portfolio_features)
    """
    # Extraer dimensiones desde la configuración (compatible con dict y objetos Pydantic)
    if isinstance(env_config, dict):
        ventana_size = env_config['ventana_observacion_size']
    else:
        ventana_size = env_config.ventana_observacion_size
    
    if isinstance(agent_config, dict):
        portfolio_features = agent_config['architecture']['portfolio_features']
    else:
        portfolio_features = agent_config.architecture.portfolio_features

    batch_size = observations_batch.shape[0]
    
    # Calcular el número total de features de mercado
    market_features_total = ventana_size * num_market_features
    
    # Dividir el lote de observaciones en market_data_flat y portfolio_data
    market_data_flat = observations_batch[:, :market_features_total]
    portfolio_data = observations_batch[:, market_features_total:]
    
    # Reestructurar market_data_flat a su forma 3D
    market_data = market_data_flat.view(batch_size, ventana_size, num_market_features)
    
    return market_data, portfolio_data
