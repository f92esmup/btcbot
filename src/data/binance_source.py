"""
Módulo de adquisición de datos de la API de Binance.
Descarga y procesa datos OHLCV para futuros de Bitcoin usando python-binance.
"""

import pandas as pd
from datetime import datetime, timezone
import time
from typing import Optional
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .abstractions import DataSource
from src.config import (
    AppConfig
)

class BinanceDataSource(DataSource):
    """Fuente de datos para adquirir y procesar datos OHLCV de Binance."""

    def __init__(self, symbol: str, interval: str, start_date: str, config: AppConfig, end_date: Optional[str] = None):
        """
        Inicializa la clase de adquisición
        
        Args:
            symbol (str): Símbolo del par de trading (ej: 'BTCUSDT')
            interval (str): Intervalo de tiempo (ej: '1h', '4h', '1d')
            start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
            end_date (str, optional): Fecha de fin en formato 'YYYY-MM-DD'
            config_dict (dict): Diccionario con la configuración necesaria
            api_key (str, optional): API key de Binance
            api_secret (str, optional): API secret de Binance
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = []
        self.dataframe = None
        self.time_offset_ms = 0
        
        # Almacenar configuración inyectada
        self.config = config
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Inicializar cliente de Binance con API keys pasadas como argumentos
        try:
            self.client = Client(
                api_key=self.config.credenciales.api_key.get_secret_value(),
                api_secret=self.config.credenciales.api_secret.get_secret_value(),
                testnet=self.config.credenciales.testnet
            )   
            self.logger.info("Cliente de Binance inicializado con API keys")
        except Exception as e:
            self.logger.warning(f"Error inicializando cliente de Binance: {e}")
            self.logger.info("Inicializando cliente sin API keys (solo datos públicos)")
            self.client = Client()
        
        # Sincronizar tiempo con el servidor de Binance
        self._sync_time()
        
    def _sync_time(self):
        """Sincroniza el tiempo local con el del servidor de Binance para calcular un offset."""
        try:
            server_time = self.client.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            self.time_offset_ms = server_time - local_time
            self.logger.info(f"Sincronización de tiempo con Binance completada. Offset: {self.time_offset_ms} ms")
        except (BinanceAPIException, BinanceRequestException) as e:
            self.logger.warning(f"No se pudo sincronizar el tiempo con Binance: {e}. Se usará el tiempo local (offset 0 ms).")
            self.time_offset_ms = 0
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Orquesta todo el proceso de adquisición y procesamiento de datos.
        
        Returns:
            pd.DataFrame: DataFrame procesado con datos OHLCV limpios
        """
        self.logger.info(f"Iniciando adquisición de datos para {self.symbol}")
        
        # 1. Descargar datos de la API (secuencial por defecto, paralelo opcional)
        self._download_klines_from_api()
        
        # Procesar datos descargados
        
        # 2. Crear y estructurar DataFrame
        self._create_and_structure_dataframe()
        
        # 3. Eliminar duplicados
        self._remove_duplicates()
        
        # 4. Interpolar velas faltantes
        self._interpolate_missing_values()
        
        # 5. Reconstruir secuencia completa
        self._reconstruct_full_sequence()
        
        # 6. Interpolar velas faltantes
        self._interpolate_missing_values()
        
        # 7. Limpieza final de NaNs (añade robustez, pero probablemente nunca participe)
        self._final_nan_cleanup()
        
        self.logger.info(f"Proceso completado. DataFrame final: {self.dataframe.shape}")
        return self.dataframe
        
    def _download_klines_from_api(self) -> None:
        """Descarga datos de velas de la API de Binance usando múltiples llamadas secuenciales."""
        self.logger.info("Descargando datos de la API de Binance...")
        
        # Convertir fecha de inicio a timestamp en milisegundos
        start_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        start_timestamp = int(start_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        # Calcular timestamp final
        if self.end_date is not None:
            # Si se especifica end_date, usarlo como límite
            end_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
            end_timestamp = int(end_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            # Si no se especifica end_date, usar el timestamp actual ajustado con el offset
            end_timestamp = int(time.time() * 1000) + self.time_offset_ms
        
        current_timestamp = end_timestamp
        
        all_klines = []
        current_start = start_timestamp
        call_count = 0
        
        end_date_display = self.end_date if self.end_date else "ahora"
        self.logger.info(f"Descargando datos desde {self.start_date} hasta {end_date_display}...")
        
        api_call_limit = self.config.dataset.binanceapi.call_limit
        self.logger.info(f"Límite por llamada: {api_call_limit} velas")
        
        while current_start < current_timestamp:
            call_count += 1
            retry_count = 0
            max_retries = self.config.dataset.binanceapi.max_retries
            klines = []  # Inicializar klines para cada iteración
            
            while retry_count <= max_retries:
                try:
                    self.logger.info(f"Llamada #{call_count} - Descargando desde timestamp: {current_start}")
                    
                    # Hacer llamada a la API con límite configurado
                    klines = self.client.futures_klines(
                        symbol=self.symbol,
                        interval=self.interval,
                        startTime=current_start,
                        limit=api_call_limit
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
                    retry_delay = self.config.dataset.binanceapi.retry_delay
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    
                    break  # Salir del bucle de reintentos si fue exitoso
                    
                except BinanceAPIException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error de API de Binance después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error de API (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(retry_delay * retry_count)  # Delay incremental
                        
                except BinanceRequestException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error de petición a Binance después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error de petición (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(retry_delay * retry_count)  # Delay incremental
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error inesperado después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error inesperado (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(retry_delay * retry_count)  # Delay incremental
            
            # Si no obtuvimos datos en esta iteración, salir del bucle principal
            if not klines:
                break
                
            # Verificar si ya llegamos al presente para evitar llamadas innecesarias
            if len(klines) < api_call_limit:
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
        columns = ["timestamp", "Open", "High", "Low", "Close", "Volume", 
                  'close_time', 'quote_asset_volume', 'number_of_trades',
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(self.raw_data, columns=columns)
        
        # Seleccionar solo las columnas OHLCV necesarias
        ohlcv_columns = self.config.dataset.columns
        df = df[["timestamp"] + ohlcv_columns]
        
        # Convertir timestamp a datetime con zona horaria
        target_timezone = self.config.dataset.timezone.target
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(target_timezone)
        
        # Establecer timestamp como índice
        df.set_index("timestamp", inplace=True)

        # Convertir columnas a tipos eficientes para RAM
        data_dtypes = self.config.dataset.dtypes.model_dump()
        for col in ohlcv_columns:
            if col in data_dtypes:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(data_dtypes[col])
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
    
    def _interpolate_missing_values(self) -> None:
        """Identifica y rellena valores NaN esporádicos dentro de velas existentes."""
        self.logger.info("Interpolando NaNs parciales...")
        
        # Contar NaNs antes de interpolar
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Interpolar usando el método configurado
        interpolation_method = self.config.dataset.interpolation.method
        interpolation_limit_direction = self.config.dataset.interpolation.limit_direction

        self.dataframe = self.dataframe.interpolate(
            method=interpolation_method,
            limit_direction=interpolation_limit_direction
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
        target_timezone = self.config.dataset.timezone.target
        complete_range = pd.date_range(start=start_time, end=end_time, freq=freq, tz=target_timezone)
        
        # Reindexar con el rango completo
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe.reindex(complete_range)
        final_count = len(self.dataframe)
        
        missing_candles = final_count - initial_count
        if missing_candles > 0:
            self.logger.info(f"Agregadas {missing_candles} filas para velas faltantes")
        else:
            self.logger.info("Secuencia temporal ya estaba completa")
    
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
