"""
Módulo de cálculo de indicadores técnicos.
Añade indicadores técnicos al dataframe OHLCV usando la biblioteca pandas-ta.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, Dict
import logging

from src.configuration.constants import (
    COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME,
    COLUMNS_OHLCV
)


class Indicadores:
    """Clase para calcular y añadir indicadores técnicos al dataframe."""
    
    def __init__(self, dataframe: pd.DataFrame, config_dict: Dict = None):
        """
        Inicializa la clase de indicadores.
        
        Args:
            dataframe (pd.DataFrame): DataFrame con datos OHLCV con índice temporal
            config_dict (Dict): Diccionario con la configuración de indicadores
        """
        self.dataframe = dataframe.copy()
        self.initial_length = len(self.dataframe)
        
        # Almacenar configuración inyectada
        self.config = config_dict or {}
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validar que el dataframe tenga las columnas necesarias
        required_columns = COLUMNS_OHLCV
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas en el dataframe: {missing_columns}")
        
        self.logger.info(f"Inicializada clase Indicadores con {len(self.dataframe)} filas")
    
    def main(self) -> pd.DataFrame:
        """
        Método principal que orquesta todo el proceso de cálculo de indicadores.
        
        Returns:
            pd.DataFrame: DataFrame con indicadores técnicos añadidos y NaNs eliminados
        """
        self.logger.info("Iniciando cálculo de indicadores técnicos...")
        
        # Paso 1: Calcular todos los indicadores técnicos
        self._calculate_technical_indicators()
        
        # Paso 2: Manejar NaNs iniciales de los indicadores
        self._handle_initial_indicator_NaNs()
        
        final_length = len(self.dataframe)
        removed_rows = self.initial_length - final_length
        
        self.logger.info(f"Proceso completado. Filas removidas por NaNs: {removed_rows}")
        self.logger.info(f"DataFrame final: {final_length} filas, {len(self.dataframe.columns)} columnas")
        
        return self.dataframe
    
    def _calculate_technical_indicators(self):
        """Calcular y añadir todos los indicadores técnicos configurados al DataFrame."""
        self.logger.info("Calculando indicadores técnicos...")
        
        # Diccionario para almacenar todos los nuevos indicadores
        new_indicators = {}
        
        # Obtener configuraciones de indicadores desde la configuración inyectada
        trend_config = self.config.get('trend_indicators', {})
        momentum_config = self.config.get('momentum_indicators', {})
        volatility_config = self.config.get('volatility_indicators', {})
        volume_config = self.config.get('volume_indicators', {})
        
        # 1. Indicadores de Tendencia
        if trend_config.get('ema_20', {}).get('enabled', False):
            period = trend_config['ema_20']['period']
            new_indicators[f'EMA_{period}'] = ta.ema(self.dataframe[COLUMN_CLOSE], length=period)
            self.logger.info(f"EMA {period} calculado")
        
        if trend_config.get('ema_50', {}).get('enabled', False):
            period = trend_config['ema_50']['period']
            new_indicators[f'EMA_{period}'] = ta.ema(self.dataframe[COLUMN_CLOSE], length=period)
            self.logger.info(f"EMA {period} calculado")
        
        if trend_config.get('adx', {}).get('enabled', False):
            period = trend_config['adx']['period']
            adx_result = ta.adx(
                high=self.dataframe[COLUMN_HIGH],
                low=self.dataframe[COLUMN_LOW], 
                close=self.dataframe[COLUMN_CLOSE],
                length=period
            )
            if adx_result is not None:
                # Buscar la columna ADX correcta
                adx_column = f'ADX_{period}'
                if adx_column in adx_result.columns:
                    new_indicators[adx_column] = adx_result[adx_column]
                    self.logger.info(f"ADX {period} calculado")
                else:
                    # Buscar cualquier columna que contenga ADX
                    adx_cols = [col for col in adx_result.columns if 'ADX' in col]
                    if adx_cols:
                        new_indicators[f'ADX_{period}'] = adx_result[adx_cols[0]]
                        self.logger.info(f"ADX {period} calculado usando {adx_cols[0]}")
                    else:
                        self.logger.warning(f"No se pudo calcular ADX, columnas disponibles: {adx_result.columns.tolist()}")
            else:
                self.logger.warning(f"ADX {period} retornó None")
        
        # 2. Indicadores de Momento
        if momentum_config.get('rsi', {}).get('enabled', False):
            period = momentum_config['rsi']['period']
            new_indicators[f'RSI_{period}'] = ta.rsi(self.dataframe[COLUMN_CLOSE], length=period)
            self.logger.info(f"RSI {period} calculado")
        
        if momentum_config.get('stoch', {}).get('enabled', False):
            k_period = momentum_config['stoch']['k_period']
            d_period = momentum_config['stoch']['d_period']
            smooth_k = momentum_config['stoch']['smooth_k']
            
            stoch_result = ta.stoch(
                high=self.dataframe[COLUMN_HIGH],
                low=self.dataframe[COLUMN_LOW],
                close=self.dataframe[COLUMN_CLOSE],
                k=k_period,
                d=d_period,
                smooth_k=smooth_k
            )
            if stoch_result is not None:
                # El nombre de la columna en pandas-ta es diferente
                column_name = f'STOCHk_{k_period}_{d_period}_{smooth_k}'
                if column_name in stoch_result.columns:
                    new_indicators[f'STOCHK_{k_period}_{d_period}_{smooth_k}'] = stoch_result[column_name]
                    self.logger.info(f"STOCHK ({k_period},{d_period},{smooth_k}) calculado")
                else:
                    # Intentar con el primer nombre disponible
                    available_cols = [col for col in stoch_result.columns if 'STOCH' in col]
                    if available_cols:
                        new_indicators[f'STOCHK_{k_period}_{d_period}_{smooth_k}'] = stoch_result[available_cols[0]]
                        self.logger.info(f"STOCHK ({k_period},{d_period},{smooth_k}) calculado usando {available_cols[0]}")
                    else:
                        self.logger.warning(f"No se pudo calcular STOCHK, columnas disponibles: {stoch_result.columns.tolist()}")
        
        # 3. Indicadores de Volatilidad
        if volatility_config.get('atr', {}).get('enabled', False):
            period = volatility_config['atr']['period']
            new_indicators[f'ATR_{period}'] = ta.atr(
                high=self.dataframe[COLUMN_HIGH],
                low=self.dataframe[COLUMN_LOW],
                close=self.dataframe[COLUMN_CLOSE],
                length=period
            )
            self.logger.info(f"ATR {period} calculado")
        
        # 4. Indicadores de Volumen
        if volume_config.get('obv', {}).get('enabled', False):
            new_indicators['OBV'] = ta.obv(
                close=self.dataframe[COLUMN_CLOSE],
                volume=self.dataframe[COLUMN_VOLUME]
            )
            self.logger.info("OBV calculado")
        
        # Añadir todos los indicadores al DataFrame en una sola operación
        if new_indicators:
            indicators_df = pd.DataFrame(new_indicators, index=self.dataframe.index)
            self.dataframe = pd.concat([self.dataframe, indicators_df], axis=1)
            self.logger.info(f"Añadidos {len(new_indicators)} indicadores al DataFrame en una sola operación")
        
        self.logger.info("Todos los indicadores técnicos calculados exitosamente")
    
    def _handle_initial_indicator_NaNs(self):
        """Eliminar las filas al principio del dataframe que contienen valores NaNs debido al cálculo de indicadores."""
        self.logger.info("Eliminando filas con NaNs iniciales de indicadores...")
        
        initial_rows = len(self.dataframe)
        
        # Identificar columnas de indicadores (todas excepto OHLCV)
        ohlcv_columns = COLUMNS_OHLCV
        indicator_columns = [col for col in self.dataframe.columns if col not in ohlcv_columns]
        
        if not indicator_columns:
            self.logger.warning("No se encontraron columnas de indicadores")
            return
        
        # Eliminar filas donde cualquier indicador tenga NaN
        # Usamos dropna con subset de columnas de indicadores
        self.dataframe = self.dataframe.dropna(subset=indicator_columns)
        
        final_rows = len(self.dataframe)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            self.logger.info(f"Eliminadas {removed_rows} filas con NaNs iniciales")
            self.logger.info(f"DataFrame resultante: {final_rows} filas")
        else:
            self.logger.info("No se encontraron NaNs para eliminar")
        
        # Resetear el índice si es necesario (mantener el datetime index)
        if not self.dataframe.index.is_monotonic_increasing:
            self.dataframe = self.dataframe.sort_index()
            self.logger.info("Índice reordenado cronológicamente")