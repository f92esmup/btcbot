"""
Factory module for creating trading environment instances.
Centralizes the creation logic for FuturesTradingEnv to improve modularity.
"""

from typing import Any
import logging

from .environment import FuturesTradingEnv
from .portfolio import Portfolio
from ..configuration import EnvironmentConfig


def create_trading_environment(
    dataframe: Any, 
    logger: logging.Logger, 
    price_scaler: Any, 
    scaler: Any, 
    env_config: EnvironmentConfig, 
    run_config: dict
) -> FuturesTradingEnv:
    """
    Crea el entorno de trading con los datos procesados.
    
    Args:
        dataframe: DataFrame con datos normalizados
        logger: Logger para mensajes
        price_scaler: Price scaler ya cargado
        scaler: Scaler principal para normalización de características
        env_config: Configuración del entorno (objeto Pydantic)
        run_config: Configuración completa del run
        
    Returns:
        FuturesTradingEnv: Entorno configurado
    """
    from ..utils.observation_builder import ObservationBuilder
    logger.info("Creando entorno de trading...")
    
    # Obtener información del rango para logging
    if hasattr(price_scaler, 'data_min_') and hasattr(price_scaler, 'data_max_'):
        close_min = price_scaler.data_min_[0]
        close_max = price_scaler.data_max_[0]
        logger.info(f"Price scaler cargado exitosamente - Rango Close: {close_min:.2f} - {close_max:.2f}")
    else:
        logger.info("Price scaler cargado exitosamente")
    
    # Crear ObservationBuilder centralizado
    observation_builder = ObservationBuilder(
        scaler=scaler,
        price_scaler=price_scaler,
        run_config=run_config
    )
    logger.info("✅ ObservationBuilder creado y configurado")
    
    sim_portfolio = Portfolio(env_config)

    env = FuturesTradingEnv(
        data_df=dataframe,
        price_scaler=price_scaler,
        env_config=env_config, # Inyección de configuración
        portfolio=sim_portfolio,
        observation_builder=observation_builder  # Inyección del constructor de observaciones
    )
    
    logger.info(f"Entorno creado:")
    logger.info(f"  - Balance inicial: ${env_config.capital_inicial:,.2f}")
    logger.info(f"  - Apalancamiento: {env_config.apalancamiento}x")
    logger.info(f"  - Ventana observación: {env_config.ventana_observacion_size}")
    logger.info(f"  - Espacio de observación: {env.observation_space}")
    logger.info(f"  - Espacio de acción: {env.action_space}")
    
    return env
