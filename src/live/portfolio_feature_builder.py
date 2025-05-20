# src/live/portfolio_feature_builder.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from src.utils.config import ConfigManager

from src.utils.logging_utils import setup_logger
logger = setup_logger("PortfolioFeatureBuilder")

class PortfolioFeatureBuilder:
    """
    Construye características de cartera para el modelo en modo de trading en vivo.
    Estas características son compatibles con el formato esperado por el modelo entrenado.
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        # Cargar configuraciones relevantes
        env_config = self.config_manager.get_environment_config()
        preprocessing_config = self.config_manager.get_preprocessing_config()
        
        # Obtener parámetros de normalización para features de cartera
        portfolio_norm_config = env_config.get('portfolio_features_normalization', {})
        self.max_steps_in_position = portfolio_norm_config.get('max_steps_in_position', 288)
        
        # Parámetros de trading
        self.leverage = float(env_config.get('leverage', 10.0))
        self.position_size_pct_equity = env_config.get('position_size_pct_equity', 0.05)
        self.taker_fee_rate = env_config.get('taker_fee_rate', 0.0004)
        
        # Longitud de la secuencia para el modelo
        self.sequence_length_L = preprocessing_config.get('sequence_length_L', 96)
        
        # Inicializar el contador de pasos en posición actual
        self.steps_in_current_position = 0  # 0 significa sin posición
        self.current_position_direction = 0  # 0: neutral, 1: long, -1: short
        
        logger.info(f"PortfolioFeatureBuilder inicializado: max_steps_in_position={self.max_steps_in_position}, "
                   f"leverage={self.leverage}x, position_size={self.position_size_pct_equity*100}% equity")
    
    def build_portfolio_features(self, 
                                position_data: Dict[str, Any], 
                                account_data: Dict[str, Any],
                                current_price: float) -> np.ndarray:
        """
        Construye el vector de características de cartera para una predicción.
        
        Args:
            position_data: Datos de la posición actual de Binance (get_position_risk)
            account_data: Datos de la cuenta de Binance (get_account_info)
            current_price: Precio actual del activo
            
        Returns:
            Características de la cartera en formato numpy para el modelo
        """
        try:
            # 1. Extraer información relevante de posición y cuenta
            position_amount = float(position_data.get('positionAmt', '0'))
            entry_price = float(position_data.get('entryPrice', '0'))
            unrealized_pnl = float(position_data.get('unRealizedProfit', '0'))
            # Primero convertir a float y luego a int para manejar valores como '10.0'
            current_leverage = int(float(position_data.get('leverage', str(self.leverage))))
            
            # Información de la cuenta
            total_wallet_balance = float(account_data.get('totalWalletBalance', '0'))
            total_margin_balance = float(account_data.get('totalMarginBalance', '0'))
            total_pnl = float(account_data.get('totalUnrealizedProfit', '0'))
            available_balance = float(account_data.get('availableBalance', '0'))
            
            # 2. Determinar la dirección de la posición
            position_direction = 0  # Neutral (sin posición)
            if abs(position_amount) > 1e-8:  # Si hay posición significativa
                position_direction = 1 if position_amount > 0 else -1  # 1: Long, -1: Short
            
            # 3. Actualizar el contador de pasos en posición
            if position_direction == 0:
                # Sin posición, resetear contador
                self.steps_in_current_position = 0
                self.current_position_direction = 0
            elif position_direction != self.current_position_direction:
                # Cambio de dirección, resetear contador y actualizar dirección
                self.steps_in_current_position = 1
                self.current_position_direction = position_direction
            else:
                # Continuación de posición existente, incrementar contador
                self.steps_in_current_position += 1
            
            # 4. Calcular características normalizadas para la cartera
            
            # a. Tamaño relativo de posición (normalizado por equity y leverage)
            # Para que sea consistente con el simulador: posSize = posAmt * price / equity
            position_size_pct = 0.0
            if abs(position_amount) > 1e-8 and total_margin_balance > 0 and current_price > 0:
                position_notional = abs(position_amount * current_price)
                position_size_pct = position_notional / total_margin_balance
            norm_position_size = position_size_pct / self.position_size_pct_equity
            
            # b. Dirección normalizada (-1 a 1)
            norm_position_direction = position_direction  # Ya está en [-1, 0, 1]
            
            # c. PnL no realizado normalizado por el tamaño de posición
            norm_unrealized_pnl = 0.0
            if abs(position_amount) > 1e-8 and current_price > 0:
                position_notional = abs(position_amount * current_price)
                if position_notional > 0:
                    norm_unrealized_pnl = unrealized_pnl / position_notional
            
            # d. Pasos normalizados en la posición actual
            norm_steps_in_position = self.steps_in_current_position / self.max_steps_in_position
            # Recortar si supera 1.0
            norm_steps_in_position = min(norm_steps_in_position, 1.0)
            
            # e. Precio de entrada normalizado vs precio actual
            norm_entry_vs_current = 0.0
            if entry_price > 0 and current_price > 0 and abs(position_amount) > 1e-8:
                if position_direction > 0:  # Long
                    # (current - entry) / entry: ganancia/pérdida relativa
                    norm_entry_vs_current = (current_price - entry_price) / entry_price
                elif position_direction < 0:  # Short
                    # (entry - current) / entry: ganancia/pérdida relativa
                    norm_entry_vs_current = (entry_price - current_price) / entry_price
            
            # f. Saldo disponible vs saldo total (para estimar margen usado)
            norm_available_balance = 0.0
            if total_wallet_balance > 0:
                norm_available_balance = available_balance / total_wallet_balance
            
            # g. Margen de liquidación
            # En Binance Futures, para longs: liq_price ≈ entry_price * (1 - 1 / leverage)
            # Para shorts: liq_price ≈ entry_price * (1 + 1 / leverage)
            # Normalizar como distancia entre precio actual y precio de liquidación
            norm_liquidation_margin = 0.0
            if abs(position_amount) > 1e-8 and entry_price > 0 and current_price > 0:
                if position_direction > 0:  # Long
                    liq_price_approx = entry_price * (1 - 0.8 / current_leverage)
                    # Que tan cerca estamos del precio de liquidación (0 = liquidado, 1 = lejos)
                    if current_price > liq_price_approx:
                        norm_liquidation_margin = (current_price - liq_price_approx) / entry_price
                elif position_direction < 0:  # Short
                    liq_price_approx = entry_price * (1 + 0.8 / current_leverage)
                    if current_price < liq_price_approx:
                        norm_liquidation_margin = (liq_price_approx - current_price) / entry_price
                
                # Normalizar margen para que 1.0 sea el margen "normal" y valores menores
                # indiquen proximidad a la liquidación
                norm_liquidation_margin = min(norm_liquidation_margin * current_leverage, 1.0)
            else:
                # Sin posición, no hay riesgo de liquidación
                norm_liquidation_margin = 1.0
            
            # h. Relación PnL total vs balance de billetera
            norm_total_pnl = 0.0
            if total_wallet_balance > 0:
                norm_total_pnl = total_pnl / total_wallet_balance
            
            # 5. Ensamblar el vector de características de cartera (8 características)
            portfolio_features = np.array([
                norm_position_direction,
                norm_position_size,
                norm_unrealized_pnl,
                norm_steps_in_position,
                norm_entry_vs_current,
                norm_available_balance,
                norm_liquidation_margin,
                norm_total_pnl
            ], dtype=np.float32)
            
            logger.debug(f"Portfolio features generadas: {portfolio_features}")
            
            # NOTA: Devolvemos directamente el array 1D (8,) y no lo replicamos
            # El CustomTransformerFeatureExtractor se encarga de expandir estas características internamente
            # mediante portfolio_features.unsqueeze(1).expand(-1, self.seq_length, -1)
            logger.debug(f"Portfolio features generadas (forma 1D): {portfolio_features.shape}")
            return portfolio_features
            
        except Exception as e:
            logger.error(f"Error construyendo características de cartera: {e}", exc_info=True)
            # Devolver un vector por defecto con valores en 0 (array 1D)
            default_features = np.zeros(8, dtype=np.float32)
            default_features[6] = 1.0  # Margin de liquidación en 1.0 (sin riesgo)
            return default_features
