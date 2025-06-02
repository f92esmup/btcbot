"""
Módulo de adquisición de datos de la API de Binance.
Descarga y procesa datos OHLCV para futuros de Bitcoin usando python-binance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from typing import List, Optional
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from ..configuration.config import config


class Adquisicion:
    """Clase para adquirir y procesar datos OHLCV de Binance."""
    
    def __init__(self, symbol: str, interval: str, start_date: str):
        """
        Inicializa la clase de adquisición.
        
        Args:
            symbol (str): Símbolo del par de trading (ej: 'BTCUSDT')
            interval (str): Intervalo de tiempo (ej: '1h', '4h', '1d')
            start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.raw_data = []
        self.dataframe = None
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar cliente de Binance con API keys desde Google Cloud Secret Manager
        try:
            api_key = config.binance_api_key
            api_secret = config.binance_api_secret
            
            if config.is_testnet:
                # Configurar cliente para testnet
                self.client = Client(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=True
                )
                self.logger.info("Cliente de Binance inicializado en modo TESTNET")
            else:
                # Configurar cliente para producción
                self.client = Client(
                    api_key=api_key,
                    api_secret=api_secret
                )
                self.logger.info("Cliente de Binance inicializado en modo PRODUCCIÓN")
                
        except Exception as e:
            self.logger.warning(f"No se pudieron cargar las API keys: {e}")
            self.logger.info("Inicializando cliente sin API keys (solo datos públicos)")
            self.client = Client()
        
    def main(self) -> pd.DataFrame:
        """
        Orquesta todo el proceso de adquisición y procesamiento de datos.
        
        Returns:
            pd.DataFrame: DataFrame procesado con datos OHLCV limpios
        """
        self.logger.info(f"Iniciando adquisición de datos para {self.symbol}")
        
        # 1. Descargar datos de la API
        self._download_klines_from_api()
        
        # 2. Crear y estructurar DataFrame
        self._create_and_structure_dataframe()
        
        # 3. Eliminar duplicados
        self._remove_duplicates()
        
        # 4. Interpolar NaNs parciales
        self._interpolate_partial_nans()
        
        # 5. Reconstruir secuencia completa
        self._reconstruct_full_sequence()
        
        # 6. Interpolar velas faltantes
        self._reconstruct_missing_candles()
        
        # 7. Limpieza final de NaNs
        self._final_nan_cleanup()
        
        self.logger.info(f"Proceso completado. DataFrame final: {self.dataframe.shape}")
        return self.dataframe
    
    def _download_klines_from_api(self) -> None:
        """Descarga datos de velas de la API de Binance usando múltiples llamadas secuenciales."""
        self.logger.info("Descargando datos de la API de Binance...")
        
        # Convertir fecha de inicio a timestamp en milisegundos
        start_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        start_timestamp = int(start_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        current_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_klines = []
        current_start = start_timestamp
        call_count = 0
        
        self.logger.info(f"Descargando datos desde {self.start_date} hasta ahora...")
        self.logger.info(f"Límite por llamada: {config.api_call_limit} velas")
        
        while current_start < current_timestamp:
            call_count += 1
            retry_count = 0
            max_retries = config.max_api_retries
            klines = []  # Inicializar klines para cada iteración
            
            while retry_count <= max_retries:
                try:
                    self.logger.info(f"Llamada #{call_count} - Descargando desde timestamp: {current_start}")
                    
                    # Hacer llamada a la API con límite configurado
                    klines = self.client.futures_klines(
                        symbol=self.symbol,
                        interval=self.interval,
                        startTime=current_start,
                        limit=config.api_call_limit
                    )
                    
                    if not klines:
                        self.logger.info("No se recibieron más datos de la API")
                        break
                    
                    # Añadir las velas recibidas a la lista total
                    all_klines.extend(klines)
                    
                    # Actualizar el timestamp de inicio para la siguiente llamada
                    # Usar el timestamp de cierre de la última vela + 1ms
                    last_close_time = int(klines[-1][6])  # close_time está en índice 6
                    current_start = last_close_time + 1
                    
                    self.logger.info(f"Descargadas {len(klines)} velas. Total acumulado: {len(all_klines)}")
                    
                    # Delay entre llamadas para respetar rate limits
                    if config.retry_delay > 0:
                        time.sleep(config.retry_delay)
                    
                    break  # Salir del bucle de reintentos si fue exitoso
                    
                except BinanceAPIException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error de API de Binance después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error de API (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(config.retry_delay * retry_count)  # Delay incremental
                        
                except BinanceRequestException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error de petición a Binance después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error de petición (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(config.retry_delay * retry_count)  # Delay incremental
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error inesperado después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error inesperado (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(config.retry_delay * retry_count)  # Delay incremental
            
            # Si no obtuvimos datos en esta iteración, salir del bucle principal
            if not klines:
                break
                
            # Verificar si ya llegamos al presente para evitar llamadas innecesarias
            if len(klines) < config.api_call_limit:
                self.logger.info("Recibidas menos velas que el límite, probablemente llegamos al presente")
                break
        
        self.raw_data = all_klines
        self.logger.info(f"Descarga completada: {len(self.raw_data)} velas totales en {call_count} llamadas")
        
        if not self.raw_data:
            raise ValueError("No se pudieron descargar datos de la API de Binance")
    
    def _create_and_structure_dataframe(self) -> None:
        """Convierte los datos crudos en un DataFrame estructurado."""
        self.logger.info("Creando y estructurando DataFrame...")
        
        # Crear DataFrame con las columnas de Binance
        columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                  'close_time', 'quote_asset_volume', 'number_of_trades',
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(self.raw_data, columns=columns)
        
        # Seleccionar solo las columnas OHLCV necesarias
        df = df[['timestamp'] + config.ohlcv_columns]
        
        # Convertir timestamp a datetime con zona horaria Madrid
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert(config.target_timezone)
        
        # Establecer timestamp como índice
        df.set_index('timestamp', inplace=True)
        
        # Convertir columnas a tipos eficientes para RAM
        for col in config.ohlcv_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(config.data_dtypes[col])
        
        self.dataframe = df
        self.logger.info(f"DataFrame creado con {len(df)} filas y {len(df.columns)} columnas")
    
    def _remove_duplicates(self) -> None:
        """Elimina filas con timestamps duplicados, conservando la primera aparición."""
        self.logger.info("Eliminando duplicados...")
        
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe[~self.dataframe.index.duplicated(keep='first')]
        final_count = len(self.dataframe)
        
        duplicates_removed = initial_count - final_count
        if duplicates_removed > 0:
            self.logger.info(f"Eliminados {duplicates_removed} duplicados")
        else:
            self.logger.info("No se encontraron duplicados")
    
    def _interpolate_partial_nans(self) -> None:
        """Identifica y rellena valores NaN esporádicos dentro de velas existentes."""
        self.logger.info("Interpolando NaNs parciales...")
        
        # Contar NaNs antes de interpolar
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Interpolar usando el método configurado
        self.dataframe = self.dataframe.interpolate(
            method=config.interpolation_method,
            limit_direction=config.interpolation_limit_direction
        )
        
        nan_count_after = self.dataframe.isnull().sum().sum()
        interpolated = nan_count_before - nan_count_after
        
        if interpolated > 0:
            self.logger.info(f"Interpolados {interpolated} valores NaN parciales")
        else:
            self.logger.info("No se encontraron NaNs parciales para interpolar")
    
    def _reconstruct_full_sequence(self) -> None:
        """Asegura que la serie temporal tenga una entrada para cada intervalo esperado."""
        self.logger.info("Reconstruyendo secuencia completa...")
        
        # Determinar la frecuencia basada en el intervalo
        freq_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        if self.interval not in freq_map:
            raise ValueError(f"Intervalo no soportado: {self.interval}")
        
        freq = freq_map[self.interval]
        
        # Crear rango completo de timestamps
        start_time = self.dataframe.index.min()
        end_time = self.dataframe.index.max()
        complete_range = pd.date_range(start=start_time, end=end_time, freq=freq, tz=config.target_timezone)
        
        # Reindexar con el rango completo
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe.reindex(complete_range)
        final_count = len(self.dataframe)
        
        missing_candles = final_count - initial_count
        if missing_candles > 0:
            self.logger.info(f"Agregadas {missing_candles} filas para velas faltantes")
        else:
            self.logger.info("Secuencia temporal ya estaba completa")
    
    def _reconstruct_missing_candles(self) -> None:
        """Interpola datos para las velas que estaban completamente ausentes."""
        self.logger.info("Interpolando velas faltantes...")
        
        # Contar NaNs antes de interpolar
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Interpolar las filas completamente NaN
        self.dataframe = self.dataframe.interpolate(
            method=config.interpolation_method,
            limit_direction=config.interpolation_limit_direction
        )
        
        nan_count_after = self.dataframe.isnull().sum().sum()
        interpolated = nan_count_before - nan_count_after
        
        if interpolated > 0:
            self.logger.info(f"Interpolados {interpolated} valores en velas faltantes")
        else:
            self.logger.info("No se encontraron velas faltantes para interpolar")
    
    def _final_nan_cleanup(self) -> None:
        """Aplica limpieza final de NaNs usando forward fill y backward fill."""
        self.logger.info("Aplicando limpieza final de NaNs...")
        
        # Contar NaNs antes de la limpieza
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Aplicar forward fill seguido de backward fill
        self.dataframe = self.dataframe.ffill().bfill()
        
        nan_count_after = self.dataframe.isnull().sum().sum()
        cleaned = nan_count_before - nan_count_after
        
        if cleaned > 0:
            self.logger.info(f"Limpiados {cleaned} NaNs finales")
        
        if nan_count_after > 0:
            self.logger.warning(f"Aún quedan {nan_count_after} NaNs después de la limpieza")
        else:
            self.logger.info("DataFrame completamente limpio, sin NaNs")
    