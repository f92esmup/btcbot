"""
Módulo para la normalización centralizada del estado del portfolio.

Este módulo contiene la lógica unificada para convertir el estado del portfolio
en un vector de características normalizado, asegurando consistencia entre
el entorno de simulación y el modo en vivo.
"""

import numpy as np
from typing import Dict, Any

from .base_portfolio import TipoOperacion


def get_normalized_portfolio_features(
    portfolio_state: Dict[str, Any], 
    env_config: Dict[str, Any], 
    price_scaler
) -> np.ndarray:
    """
    Obtiene las características del portafolio normalizadas de forma centralizada.
    
    Esta función centraliza la lógica de normalización del estado del portfolio,
    eliminando la duplicación entre el entorno de simulación y el modo en vivo.
    
    Args:
        portfolio_state (Dict[str, Any]): Estado actual del portfolio
        env_config (Dict[str, Any]): Configuración del entorno
        price_scaler: Scaler ajustado para normalizar precios
        
    Returns:
        np.ndarray: Vector de características del portfolio normalizado
        
    Raises:
        ValueError: Si la configuración o el estado del portfolio son inválidos
    """
    # Validar parámetros de entrada
    if not isinstance(portfolio_state, dict):
        raise ValueError("portfolio_state debe ser un diccionario")
    
    if not isinstance(env_config, (dict, object)):
        raise ValueError("env_config debe ser un diccionario o un objeto Pydantic")
    
    if price_scaler is None:
        raise ValueError("price_scaler no puede ser None")
    
    # Validar que las claves necesarias estén presentes
    required_keys = ['tipo', 'pnl_no_realizado_roe', 'pasos_en_posicion', 'precio_entrada']
    for key in required_keys:
        if key not in portfolio_state:
            raise ValueError(f"Clave requerida '{key}' no encontrada en portfolio_state")
    
    
    
    # 1. Normalizar tipo de posición
    tipo_posicion = portfolio_state['tipo']
    
    # Manejar tanto enums como strings para compatibilidad
    if hasattr(tipo_posicion, 'name'):
        # Es un enum
        tipo_nombre = tipo_posicion.name
    elif isinstance(tipo_posicion, str):
        # Es un string
        tipo_nombre = tipo_posicion
    else:
        # Es el enum directamente
        if tipo_posicion == TipoOperacion.LARGO:
            tipo_nombre = 'LARGO'
        elif tipo_posicion == TipoOperacion.NEUTRAL:
            tipo_nombre = 'NEUTRAL'
        else:  # TipoOperacion.CORTO
            tipo_nombre = 'CORTO'
    
    if tipo_nombre == 'LARGO':
        tipo_posicion_norm = 1.0
    elif tipo_nombre == 'NEUTRAL':
        tipo_posicion_norm = 0.5
    else:  # 'CORTO'
        tipo_posicion_norm = 0.0
    
    # 2. Normalizar PNL ROE con clipping
    min_roe = env_config['min_clip_pnl_roe']
    max_roe = env_config['max_clip_pnl_roe']
    pnl_roe_clipped = np.clip(portfolio_state['pnl_no_realizado_roe'], min_roe, max_roe)
    
    if max_roe > min_roe:
        pnl_roe_norm = (pnl_roe_clipped - min_roe) / (max_roe - min_roe)
    else:
        pnl_roe_norm = 0.5
    
    # 3. Normalizar pasos en posición
    pasos_norm = min(1.0, portfolio_state['pasos_en_posicion'] / env_config['max_pasos_en_posicion'])
    
    # 4. Normalizar precio de entrada
    if tipo_nombre != 'NEUTRAL' and portfolio_state['precio_entrada'] > 0:
        # Usar el price_scaler para normalizar el precio de entrada
        precio_entrada_scaled = price_scaler.transform([[portfolio_state['precio_entrada']]])[0][0]
        precio_entrada_norm = np.clip(precio_entrada_scaled, 0.0, 1.0)
    else:
        precio_entrada_norm = 0.5  # Valor neutral
    
    return np.array([tipo_posicion_norm, pnl_roe_norm, pasos_norm, precio_entrada_norm], dtype=np.float32)
