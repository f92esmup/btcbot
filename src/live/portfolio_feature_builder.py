import logging
import numpy as np
from typing import Dict, Optional

from src.utils.logging_utils import setup_logger

logger = setup_logger('PortfolioFeatureBuilder')

def build_live_portfolio_features(
    account_info: Dict, 
    account_balance_info: Dict, 
    position_info: Dict, 
    env_config: Dict, 
    initial_equity_config: float,
    last_step_equity: float
) -> np.ndarray:
    """
    Construye características de cartera normalizadas para usar en el modelo.
    
    Args:
        account_info: Información de la cuenta desde Binance
        account_balance_info: Información del balance por asset desde Binance
        position_info: Información de la posición desde Binance
        env_config: Configuración del entorno desde ConfigManager
        initial_equity_config: Equity inicial configurado
        last_step_equity: Equity del último ciclo para calcular retorno
        
    Returns:
        Array numpy de características de cartera normalizadas
    """
    try:
        # 1. Extraer datos relevantes
        # Total wallet balance (equity)
        current_equity = float(account_info.get('totalWalletBalance', 0))
        
        # Posición actual
        current_position_size = float(position_info.get('positionAmt', 0))
        entry_price = float(position_info.get('entryPrice', 0))
        unrealized_pnl = float(position_info.get('unRealizedProfit', 0))
        leverage = float(position_info.get('leverage', env_config.get('leverage', 1.0)))
        
        # Precio actual
        mark_price = float(position_info.get('markPrice', 0))
        
        # 2. Calcular características
        # Posición normalizada (-1 a +1, donde 0 es neutral)
        if current_position_size != 0:
            # Calcular el tamaño nocional de la posición
            position_notional = abs(current_position_size * mark_price)
            # Normalizar respecto al tamaño máximo permitido
            max_position_notional = current_equity * float(env_config.get('position_size_pct_equity', 0.05)) * leverage
            normalized_position_size = (position_notional / max_position_notional) * (1 if current_position_size > 0 else -1)
        else:
            normalized_position_size = 0
        
        # Retorno de equity respecto al inicial (logarítmico)
        log_return_from_initial = np.log(current_equity / initial_equity_config)
        
        # Retorno de equity respecto al último ciclo (logarítmico)
        if last_step_equity > 0:
            log_return_from_last_step = np.log(current_equity / last_step_equity)
        else:
            log_return_from_last_step = 0
            logger.warning("El equity del último paso es 0 o negativo, usando 0 para el retorno")
        
        # PnL no realizado normalizado por equity
        unrealized_pnl_normalized = unrealized_pnl / current_equity if current_equity > 0 else 0
        
        # % de Equity usado como margen
        used_margin = float(account_info.get('totalPositionInitialMargin', 0))
        margin_ratio = used_margin / current_equity if current_equity > 0 else 0
        
        # 3. Construir array de características
        # [position_size_normalized, log_return_from_initial, log_return_from_last_step, unrealized_pnl_normalized, margin_ratio]
        portfolio_features = np.array([
            normalized_position_size,
            log_return_from_initial,
            log_return_from_last_step,
            unrealized_pnl_normalized,
            margin_ratio
        ], dtype=np.float32)
        
        logger.info(f"Características de cartera: pos={normalized_position_size:.3f}, ret_init={log_return_from_initial:.3f}, ret_last={log_return_from_last_step:.3f}, pnl={unrealized_pnl_normalized:.3f}, margin={margin_ratio:.3f}")
        return portfolio_features
        
    except Exception as e:
        logger.error(f"Error construyendo características de cartera: {e}", exc_info=True)
        # En caso de error, devolver un array de ceros
        return np.zeros(5, dtype=np.float32)
