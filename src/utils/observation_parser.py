"""
Módulo para centralizar la lógica de parseo de observaciones del entorno.
"""
import torch
import numpy as np
from typing import Tuple, Dict


def parse_observation(observation: np.ndarray, env_config: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parsea la observación del entorno en tensores de mercado y de portfolio.

    Args:
        observation (np.ndarray): Vector de observación crudo del entorno.
        env_config (Dict): Diccionario de configuración del entorno, que debe contener
                           'ventana_observacion_size' y 'architecture.portfolio_features'.
        device (torch.device): Dispositivo (CPU/GPU) al que mover los tensores.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Una tupla conteniendo (market_data, portfolio_data).
            - market_data: Tensor de forma (1, ventana_size, num_features_mercado)
            - portfolio_data: Tensor de forma (1, portfolio_features)
    """
    # Extraer dimensiones desde la configuración
    ventana_size = env_config['ventana_observacion_size']
    
    # La configuración del agente ahora está anidada
    agent_arch_config = env_config.get('architecture', {})
    portfolio_features = agent_arch_config.get('portfolio_features', 4)

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
