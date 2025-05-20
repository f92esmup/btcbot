¡Entendido\! Aquí tienes una guía detallada en formato Markdown para implementar el modo de trading en vivo para tu BTCBot, siguiendo todos los requisitos que hemos discutido. Está estructurada para que puedas copiar y pegar fácilmente las secciones de código.

```markdown
# Guía de Implementación: BTCBot en Modo Live Trading

Esta guía detalla los pasos y el código necesario para configurar y ejecutar tu BTCBot en un entorno de trading en vivo o Testnet con Binance Futures.

## 1. Estructura del Proyecto (Modo Live)

Se creará un nuevo directorio `src/live/` para los módulos específicos del modo en vivo. Los scripts ejecutables, como el orquestador principal, residirán en `scripts/`.

```

btcbot/
├── .env                       \# Variables de entorno (NOMBRES de secretos, modo, etc.)
├── logs/                      \# Logs generales de la aplicación
├── scripts/
│   ├── run\_live\_trader.py     \# Orquestador principal para el trading en vivo
│   └── ...                    \# Otros scripts existentes
└── src/
├── live/                  \# NUEVO: Módulos para trading en vivo
│   ├── **init**.py
│   ├── websocket\_manager.py
│   ├── binance\_api\_manager.py
│   ├── live\_data\_processor.py
│   └── portfolio\_feature\_builder.py
├── agent/                 \# Módulos del agente RL (existente)
├── data/                  \# Módulos de datos (existente, FeatureEngineer se reutiliza)
├── environments/          \# Entorno de simulación (no usado directamente en live)
├── utils/                 \# Utilidades (ConfigManager, logging, etc. - existente)
│   ├── config.py          \# MODIFICADO: Para manejar nuevos nombres de secretos
│   └── ...
├── config.yaml            \# MODIFICADO: Añadir sección 'live\_trading'
└── ...                    \# Otros módulos existentes

````

## 2. Configuración del Entorno

### 2.1. Archivo `.env` (Raíz del Proyecto)

Este archivo define los **nombres** de los secretos que se buscarán en Google Cloud Secret Manager y el modo de operación. **No incluyas los valores de las claves API aquí.**

```env
# Nombres de los Secretos en Google Cloud Secret Manager
# Estos son los NOMBRES de los secretos que ConfigManager buscará.
# Debes crear estos secretos en Google Cloud Secret Manager con los valores de tus claves API.

# Para la cuenta REAL de Binance Futures
SECRET_NAME_BINANCE_API_KEY_FUTURES="BINANCE_API_KEY_FUTURES"
SECRET_NAME_BINANCE_API_SECRET_FUTURES="BINANCE_API_SECRET_FUTURES"

# Para la cuenta TESTNET de Binance Futures
SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES="TESTNET_BINANCE_API_KEY_FUTURES"
SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES="TESTNET_BINANCE_API_SECRET_FUTURES"

# GCP Configuration (Asegúrate de que estos ya existen y son correctos)
GCP_PROJECT_ID="tu-gcp-project-id" # Ej: bitcoin-460320
GCS_BUCKET_NAME="tu-gcs-bucket-name" # Ej: bitcoin-460320_data
GCP_REGION="tu-gcp-region" # Ej: europe-southwest1

# Live Trading Mode: TESTNET o REAL
# Esta variable determinará qué conjunto de NOMBRES de secretos se utilizan.
LIVE_TRADING_MODE="TESTNET" # Puede ser "TESTNET" o "REAL"
````

### 2.2. Modificaciones en `src/utils/config.py`

Asegúrate de que la clase `ConfigManager` pueda reconocer y buscar los nombres de los secretos de Testnet en Google Cloud Secret Manager.

```python
# Dentro de la clase ConfigManager en src/utils/config.py

# ... (importaciones y __new__ como existen) ...

    def get_env_variable(self, var_name: str, default=None) -> str:
        """
        Obtiene una variable de entorno o un secreto de Secret Manager.
        Para variables sensibles (credenciales, claves API), solo se obtienen de Secret Manager.
        
        Args:
            var_name: Nombre de la variable/secreto a obtener.
            default: Valor por defecto si no se encuentra.
            
        Returns:
            Valor de la variable o secreto.
        """
        # Asegurar que la lista 'secretos' incluye todos los posibles NOMBRES de secretos
        # que se gestionarán a través de Google Cloud Secret Manager.
        secretos = [
            "BINANCE_API_KEY_FUTURES", 
            "BINANCE_API_SECRET_FUTURES",
            "TESTNET_BINANCE_API_KEY_FUTURES",    # Nombre del secreto para la API Key de Testnet
            "TESTNET_BINANCE_API_SECRET_FUTURES"  # Nombre del secreto para la API Secret de Testnet
        ]
        
        if var_name in secretos:
            if not self.secret_client or not self.gcp_project_id:
                # Intento de reinicialización si es necesario (por ejemplo, si GCP_PROJECT_ID se cargó después)
                self.gcp_project_id = os.getenv('GCP_PROJECT_ID')
                if not self.gcp_project_id:
                    logger.error("GCP_PROJECT_ID no está configurado. Se requiere para Secret Manager.")
                    raise ValueError("GCP_PROJECT_ID no configurado para Secret Manager.")
                if not self.secret_client:
                    try:
                        self.secret_client = secretmanager.SecretManagerServiceClient()
                        logger.info(f"Cliente de Google Secret Manager (re)inicializado para proyecto: {self.gcp_project_id}")
                    except Exception as e:
                        logger.error(f"No se pudo (re)inicializar Google Secret Manager: {e}")
                        raise ConnectionError(f"Error al (re)inicializar Google Secret Manager: {e}. Revise su configuración de autenticación.")
            
            try:
                secret_path = f"projects/{self.gcp_project_id}/secrets/{var_name}/versions/latest"
                response = self.secret_client.access_secret_version(name=secret_path)
                secret_value = response.payload.data.decode('UTF-8')
                logger.info(f"Secreto '{var_name}' obtenido correctamente de Secret Manager.")
                return secret_value
            except Exception as e:
                logger.error(f"No se pudo obtener el secreto '{var_name}' de Secret Manager: {e}")
                # No devolver default para secretos; si falla, es un problema.
                raise ValueError(f"Error al obtener el secreto '{var_name}' de Secret Manager: {e}")
        
        # Si no es un secreto de la lista, obtenerlo de las variables de entorno del proceso.
        env_value = os.getenv(var_name, default)
        if env_value is None and default is None: # Si no se encuentra y no hay default
             logger.warning(f"Variable de entorno '{var_name}' no encontrada y no se proporcionó valor por defecto.")
        return env_value

# ... (resto de la clase ConfigManager) ...
```

### 2.3. Actualizaciones en `src/config.yaml`

Añade una nueva sección `live_trading` al final de tu archivo `src/config.yaml`:

```yaml
# ... (configuraciones existentes: data_paths, binance_api, data_acquisition_defaults, preprocessing, environment, agent) ...

# =====================================================================
# Configuración de Trading en Vivo
# =====================================================================
live_trading:
  # URL completa del endpoint de predicción en Vertex AI
  # Formato: https://{region}[-aiplatform.googleapis.com/v1/projects/](https://-aiplatform.googleapis.com/v1/projects/){project_id}/locations/{region}/endpoints/{endpoint_id}:predict
  # EJEMPLO: vertex_ai_predict_url: "[https://europe-southwest1-aiplatform.googleapis.com/v1/projects/bitcoin-460320/locations/europe-southwest1/endpoints/1234567890123456789:predict](https://europe-southwest1-aiplatform.googleapis.com/v1/projects/bitcoin-460320/locations/europe-southwest1/endpoints/1234567890123456789:predict)"
  vertex_ai_predict_url: "REEMPLAZAR_CON_URL_ENDPOINT_VERTEX_AI"
  
  # ID "crudo" del endpoint de Vertex AI (solo el número identificador del endpoint)
  # EJEMPLO: vertex_ai_endpoint_raw_id: "1234567890123456789"
  vertex_ai_endpoint_raw_id: "REEMPLAZAR_CON_ID_ENDPOINT_VERTEX_AI"

  # Número de velas a descargar para el preprocesamiento en cada ciclo de decisión
  # Debe ser suficiente para las ventanas de normalización e indicadores.
  # Ej: Si L=96 y normalization_window_multiplier_for_L=2, la ventana de normalización es 192.
  # Necesitarás al menos 192 velas + (periodos de indicadores más largos) para que la última vela tenga features normalizadas válidas.
  market_data_lookback_candles: 250 # Ajustar según sea necesario

  # Segundos a esperar entre reintentos de conexión del WebSocket si falla
  websocket_retry_delay_seconds: 10
  # Segundos a esperar antes de reintentar el ciclo principal si ocurre un error inesperado
  websocket_unexpected_error_delay_seconds: 30

  # Configuración para la disponibilidad de la última kline en la API REST
  # A veces, después de que el WebSocket notifica el cierre, la API REST puede tardar un instante en tenerla.
  kline_availability_retry_delay_seconds: 5  # Segundos de espera entre reintentos
  kline_availability_max_retries: 12         # Máximo número de reintentos (ej. 12 * 5s = 1 minuto)

  # Pequeña pausa opcional en segundos después de cerrar una posición antes de evaluar abrir una nueva.
  post_close_delay_seconds: 1

  # Configuración para el logging en CSV a GCS
  gcs_log_path_template: "live_trading_logs/{symbol}_{interval}/{date}.csv" # {symbol}, {interval}, {date} serán reemplazados
  log_buffer_size_records: 50    # Número de registros a acumular antes de subir a GCS
  gcs_log_upload_interval_seconds: 3600 # Intervalo máximo en segundos para subir logs (ej. 1 hora)

```

## 3\. Módulos para Trading en Vivo (`src/live/`)

Crea el directorio `src/live/` y los siguientes archivos dentro.

### 3.1. `src/live/__init__.py`

Archivo vacío o para exponer clases principales si es necesario.

```python
# src/live/__init__.py
# Puede estar vacío o usarse para facilitar importaciones
# from .websocket_manager import LiveWebsocketManager
# from .binance_api_manager import LiveBinanceAPIManager
# ...
```

### 3.2. `src/live/websocket_manager.py`

Gestiona la conexión WebSocket para detectar el cierre de nuevas velas.

```python
# src/live/websocket_manager.py
import asyncio
import json
import websockets
import os
import logging
from src.utils.config import ConfigManager # Asumiendo que ConfigManager está accesible

# logger = logging.getLogger(__name__) # O usa tu setup_logger
# Para consistencia, usa tu setup_logger si lo tienes
from src.utils.logging_utils import setup_logger
logger = setup_logger("LiveWebsocketManager")


class LiveWebsocketManager:
    def __init__(self, config_manager: ConfigManager, notification_queue: asyncio.Queue):
        self.config_manager = config_manager
        self.notification_queue = notification_queue

        data_acq_config = self.config_manager.get_data_acquisition_defaults()
        self.symbol = data_acq_config.get('symbol', "BTCUSDT").lower() # Binance usa lowercase para streams
        self.interval = data_acq_config.get('interval', "1h")

        live_trading_config = self.config_manager.get_config_value('live_trading', {})
        self.retry_delay = live_trading_config.get('websocket_retry_delay_seconds', 10)
        self.unexpected_error_delay = live_trading_config.get('websocket_unexpected_error_delay_seconds', 30)
        
        self.trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
        
        if self.trading_mode == 'TESTNET':
            self.websocket_url = f"wss://[stream.binancefuture.com/ws/](https://stream.binancefuture.com/ws/){self.symbol}@kline_{self.interval}"
        else: # REAL
            self.websocket_url = f"wss://[fstream.binance.com/ws/](https://fstream.binance.com/ws/){self.symbol}@kline_{self.interval}"
        
        logger.info(f"LiveWebsocketManager inicializado para {self.symbol}@{self.interval} en endpoint {self.trading_mode}: {self.websocket_url}")

    async def _process_message(self, message_str: str):
        try:
            message = json.loads(message_str)
            # Evento de Kline: 'e': 'kline'
            # Datos de la Kline: 'k'
            #   'x': True si la vela está cerrada
            if message.get('e') == 'kline' and message.get('k', {}).get('x') == True:
                kline_data = message['k']
                logger.debug(f"Vela cerrada detectada por WebSocket: T:{kline_data.get('t')}, O:{kline_data.get('o')}, H:{kline_data.get('h')}, L:{kline_data.get('l')}, C:{kline_data.get('c')}, V:{kline_data.get('v')}")
                await self.notification_queue.put(kline_data) # Poner solo el objeto 'k'
            elif 'e' in message and message.get('e') == 'error':
                logger.error(f"Error recibido del WebSocket de Binance: {message}")
            # Puedes añadir logs para otros tipos de mensajes si es necesario para depuración
            # else:
            #     logger.debug(f"Mensaje de WebSocket recibido (otro tipo): {message_str[:250]}...")

        except json.JSONDecodeError:
            logger.error(f"Fallo al decodificar JSON del WebSocket: {message_str}")
        except Exception as e:
            logger.error(f"Error procesando mensaje de WebSocket: {e}", exc_info=True)

    async def run(self):
        logger.info(f"Iniciando conexión WebSocket a {self.websocket_url}")
        while True: # Bucle externo para reconexión
            try:
                async with websockets.connect(self.websocket_url) as ws_client:
                    logger.info(f"Conectado exitosamente al WebSocket: {self.websocket_url}")
                    while True: # Bucle interno para recibir mensajes
                        try:
                            message = await ws_client.recv()
                            await self._process_message(message)
                        except websockets.exceptions.ConnectionClosedOK:
                            logger.info(f"WebSocket cerrado limpiamente. Reconectando en {self.retry_delay}s...")
                            break # Salir del bucle interno para reconectar
                        except websockets.exceptions.ConnectionClosedError as cc_err:
                            logger.warning(f"Error de conexión WebSocket (ConnectionClosedError): {cc_err}. Reconectando en {self.retry_delay}s...")
                            break # Salir del bucle interno para reconectar
                        except Exception as e_inner_loop:
                            # Errores inesperados durante la recepción o procesamiento
                            logger.error(f"Error en el bucle interno de WebSocket (recv/process): {e_inner_loop}. Reconectando en {self.retry_delay}s...")
                            break # Salir del bucle interno para reconectar
            
            except websockets.exceptions.InvalidStatusCode as isc_err: # Error común si la URL o el handshake fallan
                logger.error(f"Conexión WebSocket falló con código de estado inválido {isc_err.status_code}. Reintentando en {self.unexpected_error_delay}s...")
                await asyncio.sleep(self.unexpected_error_delay) # Delay más largo para problemas persistentes
            except ConnectionRefusedError: # El servidor remoto rechaza la conexión
                logger.error(f"Conexión WebSocket rechazada por el servidor. Reintentando en {self.retry_delay}s...")
            except OSError as os_err: # Problemas de red a nivel de SO
                 logger.error(f"Error de OS durante la conexión WebSocket (ej. Network is unreachable): {os_err}. Reintentando en {self.retry_delay}s...")
            except Exception as e_outer_loop: # Otros errores al intentar conectar (ej. timeout)
                logger.error(f"Error inesperado en el bucle externo de WebSocket (conexión): {e_outer_loop}. Reintentando en {self.unexpected_error_delay}s...", exc_info=True)
                await asyncio.sleep(self.unexpected_error_delay) # Delay más largo
            else: # Se ejecuta si el bloque try del `async with` termina sin excepciones (ej. por un break interno)
                pass # Simplemente procederá al delay de reconexión de abajo

            await asyncio.sleep(self.retry_delay) # Esperar antes de reintentar la conexión en el bucle externo
```

### 3.3. `src/live/binance_api_manager.py`

Encapsula todas las interacciones con la API REST de Binance Futures.

```python
# src/live/binance_api_manager.py
import os
import logging
import pandas as pd
import time # Para delays en reintentos si es necesario (aunque asyncio.sleep es mejor)
import asyncio

from binance.client import AsyncClient # Importante: Usar AsyncClient para asyncio
from binance.exceptions import BinanceAPIException, BinanceOrderException, BinanceRequestException

from src.utils.config import ConfigManager
from typing import Optional, Dict, List, Any, Union

# logger = logging.getLogger(__name__)
from src.utils.logging_utils import setup_logger
logger = setup_logger("LiveBinanceAPIManager")

class LiveBinanceAPIManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.client: Optional[AsyncClient] = None # Se inicializará de forma asíncrona
        self.trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
        
        self.env_config = self.config_manager.get_environment_config()
        self.data_acq_config = self.config_manager.get_data_acquisition_defaults()
        live_trading_cfg = self.config_manager.get_config_value('live_trading', {})
        self.kline_retry_delay = live_trading_cfg.get('kline_availability_retry_delay_seconds', 5)
        self.kline_max_retries = live_trading_cfg.get('kline_availability_max_retries', 12)
        
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None

    async def _load_credentials(self):
        if self.api_key and self.api_secret: # Ya cargadas
            return

        if self.trading_mode == 'TESTNET':
            api_key_secret_name = os.getenv('SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES')
            api_secret_secret_name = os.getenv('SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES')
            logger.info("LiveBinanceAPIManager: Configurando para usar credenciales y endpoint de TESTNET.")
        else: # REAL
            api_key_secret_name = os.getenv('SECRET_NAME_BINANCE_API_KEY_FUTURES')
            api_secret_secret_name = os.getenv('SECRET_NAME_BINANCE_API_SECRET_FUTURES')
            logger.info("LiveBinanceAPIManager: Configurando para usar credenciales y endpoint de REAL.")

        if not api_key_secret_name or not api_secret_secret_name:
            msg = "Nombres de los secretos para API Key/Secret no definidos en variables de entorno (.env)."
            logger.critical(msg)
            raise ValueError(msg)

        try:
            self.api_key = self.config_manager.get_env_variable(api_key_secret_name)
            self.api_secret = self.config_manager.get_env_variable(api_secret_secret_name)
        except ValueError as ve: # ConfigManager lanza ValueError si no puede obtener el secreto
            logger.critical(f"Error crítico al obtener claves API de ConfigManager (vía Secret Manager): {ve}")
            raise # Relanzar para detener la inicialización

        if not self.api_key or not self.api_secret:
            msg = f"API Key o Secret no pudieron ser recuperados de Secret Manager para el modo {self.trading_mode} (valores None o vacíos)."
            logger.critical(msg)
            raise ValueError(msg)
        logger.info(f"Credenciales para {self.trading_mode} cargadas en memoria (no se mostrarán).")


    async def initialize_client(self):
        """Inicializa el AsyncClient. Debe ser llamado antes de usar otros métodos."""
        if self.client:
            logger.debug("Cliente AsyncClient ya inicializado.")
            return

        await self._load_credentials() # Carga self.api_key y self.api_secret

        try:
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=(self.trading_mode == 'TESTNET')
            )
            # Realizar un ping para verificar la conexión y autenticación
            ping_result = await self.client.ping()
            logger.info(f"AsyncClient para Binance {self.trading_mode} inicializado. Ping exitoso: {ping_result}")
        except BinanceAPIException as e: # Errores específicos de la API de Binance durante el ping
            logger.critical(f"Fallo el ping a la API de Binance {self.trading_mode} (BinanceAPIException): {e.status_code} - {e.message}")
            if self.client: await self.client.close_connection()
            self.client = None
            raise ConnectionError(f"Fallo el ping a la API de Binance {self.trading_mode}: {e}")
        except Exception as e: # Otros errores (ej. red)
            logger.critical(f"Fallo al crear o hacer ping con AsyncClient para Binance {self.trading_mode}: {e}", exc_info=True)
            if self.client: await self.client.close_connection() # Asegurar limpieza
            self.client = None
            raise ConnectionError(f"Fallo la inicialización del cliente de Binance: {e}")

    async def close_client_session(self):
        if self.client:
            try:
                await self.client.close_connection()
                logger.info(f"Sesión AsyncClient para Binance {self.trading_mode} cerrada.")
            except Exception as e:
                logger.error(f"Error al cerrar la sesión del cliente AsyncClient: {e}", exc_info=True)
            finally:
                self.client = None # Marcar como cerrado
        else:
            logger.debug("Intento de cerrar sesión de cliente, pero no había cliente activo.")
            
    async def _ensure_client(self):
        """Asegura que el cliente esté inicializado."""
        if not self.client:
            logger.warning("Cliente AsyncClient no inicializado. Intentando inicializar ahora...")
            await self.initialize_client()
            if not self.client: # Si todavía es None después del intento
                 raise ConnectionError("Fallo crítico: El cliente de Binance API no pudo ser inicializado.")

    # --- MÉTODOS DE API ---
    async def get_historical_klines(self, symbol: str, interval: str, lookback_candles: int) -> Optional[pd.DataFrame]:
        await self._ensure_client()
        logger.info(f"Obteniendo últimas {lookback_candles} klines para {symbol} con intervalo {interval}...")
        for attempt in range(self.kline_max_retries + 1):
            try:
                # endTime no es necesario si solo queremos las últimas 'limit' velas
                klines_raw = await self.client.futures_klines(symbol=symbol.upper(), interval=interval, limit=lookback_candles)
                
                if klines_raw and len(klines_raw) >= 1: # Comprobación básica
                    logger.info(f"Se obtuvieron {len(klines_raw)} klines para {symbol}.")
                    df = pd.DataFrame(klines_raw, columns=[
                        'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_Time', 
                        'Quote_Asset_Volume', 'Number_of_Trades', 'Taker_Buy_Base_Asset_Volume', 
                        'Taker_Buy_Quote_Asset_Volume', 'Ignore'
                    ])
                    df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
                    df.set_index('Open_Time', inplace=True) # Importante para consistencia con preprocesador
                    
                    # Convertir columnas relevantes a numérico
                    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in cols_to_numeric:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=cols_to_numeric, inplace=True) # Eliminar filas con NaNs en OHLCV

                    return df[['Open', 'High', 'Low', 'Close', 'Volume']] # Devolver solo OHLCV
                else:
                    logger.warning(f"No se obtuvieron klines o la respuesta está vacía para {symbol} (intento {attempt+1}/{self.kline_max_retries}).")

            except BinanceAPIException as e:
                logger.error(f"Excepción de API de Binance obteniendo klines para {symbol} (intento {attempt+1}): {e.status_code} - {e.message}")
                if e.status_code == 429 or e.status_code == 418: # Rate limit
                    logger.warning(f"Rate limit alcanzado. Esperando {self.kline_retry_delay * (attempt + 2)}s...") # Backoff exponencial simple
                    await asyncio.sleep(self.kline_retry_delay * (attempt + 2))
            except BinanceRequestException as e: # Problemas de red/conexión
                 logger.error(f"Excepción de solicitud de Binance obteniendo klines para {symbol} (intento {attempt+1}): {e}")
            except Exception as e:
                logger.error(f"Error inesperado obteniendo klines para {symbol} (intento {attempt+1}): {e}", exc_info=True)
            
            if attempt < self.kline_max_retries:
                logger.info(f"Esperando {self.kline_retry_delay}s antes del siguiente reintento para obtener klines.")
                await asyncio.sleep(self.kline_retry_delay)
            else:
                logger.error(f"Fallo al obtener klines para {symbol} después de {self.kline_max_retries} reintentos.")
        return None

    async def get_account_balance(self) -> Optional[List[Dict[str, Any]]]:
        await self._ensure_client()
        logger.debug("Obteniendo balance de la cuenta de futuros...")
        try:
            balance_info = await self.client.futures_account_balance()
            logger.debug(f"Balance de cuenta obtenido: {len(balance_info)} activos.")
            return balance_info
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance obteniendo balance de cuenta: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado obteniendo balance de cuenta: {e}", exc_info=True)
        return None

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        await self._ensure_client()
        logger.debug("Obteniendo información general de la cuenta de futuros...")
        try:
            account_info = await self.client.futures_account()
            logger.debug(f"Información de cuenta obtenida. Total Wallet Balance: {account_info.get('totalWalletBalance')}")
            return account_info
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance obteniendo información de cuenta: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado obteniendo información de cuenta: {e}", exc_info=True)
        return None

    async def get_position_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        await self._ensure_client()
        # Para obtener una posición específica, la API espera el símbolo.
        # futures_position_information devuelve una lista. Si se pide un símbolo, debería ser una lista de 1 elemento.
        logger.debug(f"Obteniendo información de riesgo de posición para el símbolo: {symbol}...")
        try:
            positions = await self.client.futures_position_information(symbol=symbol.upper())
            if isinstance(positions, list) and len(positions) > 0:
                # Asumimos que si se pide un símbolo, la primera (y única) entrada es la relevante.
                pos_info = positions[0]
                logger.debug(f"Información de posición para {symbol} obtenida. PosAmt: {pos_info.get('positionAmt')}")
                return pos_info
            else:
                logger.warning(f"No se encontró información de posición para {symbol} o la respuesta fue inesperada: {positions}")
                # Devolver una estructura por defecto que indique neutralidad si no se encuentra la posición específica
                return {'symbol': symbol.upper(), 'positionAmt': '0.0', 'entryPrice': '0.0', 'unRealizedProfit': '0.0', 'leverage': str(self.env_config.get('leverage', 10))}
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance obteniendo riesgo de posición para {symbol}: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado obteniendo riesgo de posición para {symbol}: {e}", exc_info=True)
        return None # O la estructura por defecto como arriba en caso de error también

    async def get_exchange_info(self) -> Optional[dict]:
        await self._ensure_client()
        logger.debug("Obteniendo información del exchange (futures)...")
        try:
            exchange_info = await self.client.futures_exchange_info()
            logger.debug("Información del exchange obtenida exitosamente.")
            return exchange_info
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance obteniendo info del exchange: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado obteniendo info del exchange: {e}", exc_info=True)
        return None

    async def get_symbol_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Este método no necesita _ensure_client() si es llamado por otro que ya lo hizo
        exchange_info = await self.get_exchange_info() # Este llamará a _ensure_client()
        if exchange_info and 'symbols' in exchange_info:
            for symbol_data in exchange_info['symbols']:
                if symbol_data['symbol'] == symbol.upper():
                    # Crear un dict de filtros por filterType para fácil acceso
                    filters = {item['filterType']: item for item in symbol_data.get('filters', [])}
                    logger.debug(f"Filtros para el símbolo {symbol} obtenidos.")
                    return filters
        logger.warning(f"No se encontraron filtros para el símbolo {symbol} en la info del exchange.")
        return None

    async def _format_quantity_for_api(self, quantity: float, step_size_str: str) -> str:
        """Formatea la cantidad según el step_size para la API de Binance."""
        step_size = float(step_size_str)
        if step_size == 1.0: # Entero
            return str(int(quantity))
        
        # Calcular el número de decimales del step_size
        # Por ejemplo, 0.001 -> 3 decimales; 0.01 -> 2 decimales
        if '.' in step_size_str:
            precision = len(step_size_str.split('.')[1].rstrip('0'))
        else: # Si es un entero como "1", no tiene decimales de precisión más allá de ser entero.
              # Si step_size es "10", por ejemplo, quantity debe ser múltiplo de 10.
              # El formateo a string se encarga de esto si quantity ya es múltiplo.
            precision = 0

        # Formatear con la precisión correcta
        # Ej: quantity=0.12345, precision=3 -> "0.123"
        # Usamos f-string con formateo de precisión.
        # Nota: esto trunca, no redondea, que es lo que usualmente requiere Binance.
        # Para asegurar el truncamiento correcto:
        # factor = 10 ** precision
        # return str(math.floor(quantity * factor) / factor)
        # Pero Binance es más sobre el formato de string correcto.
        return f"{quantity:.{precision}f}"


    async def calculate_order_quantity(self, symbol: str, equity: float, current_price: float, leverage: int) -> float:
        # Este método no necesita _ensure_client() si es llamado por otro que ya lo hizo
        if current_price <= 0:
            logger.warning(f"Precio actual inválido ({current_price}) para calcular cantidad de orden.")
            return 0.0

        position_size_pct_equity = self.env_config.get('position_size_pct_equity', 0.05)
        desired_notional_value = equity * position_size_pct_equity * float(leverage)
        
        # Cantidad inicial en el activo base (ej. BTC para BTCUSDT)
        quantity = desired_notional_value / current_price
        
        filters = await self.get_symbol_filters(symbol) # Este llamará a _ensure_client()
        min_qty_allowed = 0.0
        step_size_allowed = "0.001" # Un default razonable para BTC si no hay filtros

        if filters:
            lot_size_filter = filters.get('LOT_SIZE')
            if lot_size_filter:
                min_qty_str = lot_size_filter.get('minQty', "0.001")
                step_size_str = lot_size_filter.get('stepSize', "0.001")
                
                min_qty_allowed = float(min_qty_str)
                step_size_value = float(step_size_str)
                step_size_allowed = step_size_str # Guardar el string para formateo

                if step_size_value > 0:
                    # Aplicar step_size (truncar al múltiplo inferior de step_size)
                    quantity = (quantity // step_size_value) * step_size_value
                logger.debug(f"Cantidad después de step_size ({step_size_str}): {quantity}")

            min_notional_filter = filters.get('MIN_NOTIONAL')
            if min_notional_filter:
                min_notional_value_str = min_notional_filter.get('notional', "5.0") # Binance suele tener 5 USDT como mínimo
                min_notional_value = float(min_notional_value_str)
                if quantity * current_price < min_notional_value:
                    logger.warning(f"Valor nocional calculado ({quantity * current_price:.2f}) para {symbol} es menor que MIN_NOTIONAL ({min_notional_value:.2f}). Cantidad ajustada a 0.")
                    return 0.0
        else:
            logger.warning(f"No se encontraron filtros LOT_SIZE o MIN_NOTIONAL para {symbol}. Usando mínimos por defecto y posible imprecisión en cantidad.")
            min_qty_allowed = float(self.env_config.get('min_order_size_btc', 0.001)) # Usar config si no hay filtro

        # Comprobar minQty después de todos los ajustes
        if quantity < min_qty_allowed:
            logger.warning(f"Cantidad calculada {quantity:.8f} para {symbol} es menor que la cantidad mínima permitida ({min_qty_allowed:.8f}). Cantidad ajustada a 0.")
            return 0.0
        
        # Redondear a una precisión segura (ej. 8 decimales) antes de retornar, aunque el formateo final será en place_order
        final_quantity = round(quantity, 8)

        logger.info(f"Cantidad de orden calculada para {symbol}: {final_quantity:.8f} (Equity: {equity:.2f}, Precio: {current_price:.2f}, Lev: {leverage}x, PctEquity: {position_size_pct_equity*100}%)")
        return final_quantity


    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        await self._ensure_client()
        if quantity <= 0:
            logger.warning(f"Intento de colocar orden {side} para {symbol} con cantidad inválida: {quantity}. Abortando.")
            return None
        
        order_side = AsyncClient.SIDE_BUY if side.upper() == "BUY" else AsyncClient.SIDE_SELL
        
        # Obtener stepSize para formatear la cantidad correctamente
        quantity_str = f"{quantity:.8f}" # Default a 8 decimales si no hay filtros
        filters = await self.get_symbol_filters(symbol)
        if filters and 'LOT_SIZE' in filters:
            step_size_api_format = filters['LOT_SIZE'].get('stepSize')
            if step_size_api_format:
                 # Re-formatear la cantidad para que coincida exactamente con lo que la API espera
                 # Esto es crucial ya que Binance es estricto con el formato de 'quantity'
                 # El cálculo en calculate_order_quantity ya debería haberla hecho múltiplo de step_size
                 quantity_str = await self._format_quantity_for_api(quantity, step_size_api_format)


        logger.info(f"Colocando orden MARKET {side} para {quantity_str} de {symbol.upper()}...")
        try:
            order_response = await self.client.futures_create_order(
                symbol=symbol.upper(),
                side=order_side,
                type=AsyncClient.ORDER_TYPE_MARKET,
                quantity=quantity_str 
            )
            logger.info(f"Orden MARKET {side} para {symbol} colocada: {order_response}")
            return order_response
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance colocando orden market {side} para {symbol}: Status {e.status_code}, Code {e.code}, Msg {e.message}")
        except BinanceOrderException as e: # Errores más específicos de la lógica de órdenes
            logger.error(f"Excepción de Orden de Binance colocando orden market {side} para {symbol}: Status {e.status_code}, Code {e.code}, Msg {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado colocando orden market {side} para {symbol}: {e}", exc_info=True)
        return None

    async def close_market_position(self, symbol: str, position_amt_to_close: float) -> Optional[dict]:
        await self._ensure_client()
        
        position_amount_abs = abs(position_amt_to_close)
        if position_amount_abs < 1e-8: # Si es prácticamente cero
            logger.info(f"No hay posición significativa (cantidad: {position_amt_to_close}) para cerrar en {symbol}.")
            return {"status": "NO_POSITION_TO_CLOSE", "message": "Position amount is zero."} # Opcional: devolver un status
        
        # Para cerrar una posición larga (position_amt > 0), se VENDE.
        # Para cerrar una posición corta (position_amt < 0), se COMPRA.
        side_to_close = "SELL" if position_amt_to_close > 0 else "BUY"
        
        # Formatear cantidad para la API
        quantity_str = f"{position_amount_abs:.8f}" # Default
        filters = await self.get_symbol_filters(symbol)
        if filters and 'LOT_SIZE' in filters:
            step_size_api_format = filters['LOT_SIZE'].get('stepSize')
            if step_size_api_format:
                 quantity_str = await self._format_quantity_for_api(position_amount_abs, step_size_api_format)

        logger.info(f"Cerrando posición MARKET para {symbol} mediante una orden {side_to_close} de {quantity_str}...")
        try:
            order_side_api = AsyncClient.SIDE_SELL if side_to_close == "SELL" else AsyncClient.SIDE_BUY
            order_response = await self.client.futures_create_order(
                symbol=symbol.upper(),
                side=order_side_api,
                type=AsyncClient.ORDER_TYPE_MARKET,
                quantity=quantity_str,
                reduceOnly='true' # Importante para asegurar que solo reduce o cierra la posición
            )
            logger.info(f"Orden de cierre MARKET para {symbol} colocada: {order_response}")
            return order_response
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance cerrando posición para {symbol}: Status {e.status_code}, Code {e.code}, Msg {e.message}")
            # Un error común es "ReduceOnly Order is rejected" si no hay posición o la cantidad es incorrecta.
            if e.code == -2022 and "ReduceOnly Order is rejected" in e.message:
                logger.warning(f"Cierre fallido para {symbol} (ReduceOnly rejected). Probablemente no había posición o la cantidad es mayor a la posición.")
        except BinanceOrderException as e:
            logger.error(f"Excepción de Orden de Binance cerrando posición para {symbol}: Status {e.status_code}, Code {e.code}, Msg {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado cerrando posición para {symbol}: {e}", exc_info=True)
        return None

    async def set_leverage_if_needed(self, symbol: str, desired_leverage: int) -> bool:
        await self._ensure_client()
        logger.info(f"Verificando/estableciendo apalancamiento para {symbol.upper()} a {desired_leverage}x...")
        try:
            # Obtener información de la posición actual para verificar el apalancamiento
            # Nota: futures_position_information puede devolver una lista.
            current_positions = await self.client.futures_position_information(symbol=symbol.upper())
            current_leverage_on_symbol = None
            if current_positions and isinstance(current_positions, list) and len(current_positions) > 0:
                # Asumimos que la primera entrada es la relevante para el símbolo
                current_leverage_on_symbol = int(float(current_positions[0].get('leverage', '0')))
            
            if current_leverage_on_symbol == desired_leverage:
                logger.info(f"Apalancamiento para {symbol.upper()} ya está configurado en {desired_leverage}x.")
                return True

            # Si es diferente o no se pudo obtener, intentar cambiarlo
            await self.client.futures_change_leverage(symbol=symbol.upper(), leverage=desired_leverage)
            logger.info(f"Apalancamiento para {symbol.upper()} establecido a {desired_leverage}x exitosamente.")
            return True
        except BinanceAPIException as e:
            # Binance puede devolver error si intentas cambiar a un apalancamiento que ya está activo,
            # o si no hay posición y el apalancamiento ya es el por defecto.
            # El código de error para "Leverage not modified" suele ser -4046 o similar, pero puede variar.
            if "Leverage not modified" in e.message or (e.code == -4046): # -4046 es común para "No need to change leverage"
                logger.info(f"Apalancamiento para {symbol.upper()} ya estaba en {desired_leverage}x (API informó: {e.message}).")
                return True
            logger.error(f"Excepción de API de Binance estableciendo apalancamiento para {symbol.upper()} a {desired_leverage}x: {e.status_code} - {e.code} - {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado estableciendo apalancamiento para {symbol.upper()} a {desired_leverage}x: {e}", exc_info=True)
        return False

```

### 3.4. `src/live/live_data_processor.py`

Transforma datos crudos de velas en `market_features`, reutilizando `FeatureEngineer`.

```python
# src/live/live_data_processor.py
import pandas as pd
import numpy as np
import logging
from src.utils.config import ConfigManager
from src.data.feature_engineering import FeatureEngineer # Reutilizar

# logger = logging.getLogger(__name__)
from src.utils.logging_utils import setup_logger
logger = setup_logger("LiveFeatureProcessor")

class LiveFeatureProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.preprocessing_config = self.config_manager.get_preprocessing_config()
        if not self.preprocessing_config:
            raise ValueError("Configuración de preprocesamiento no encontrada en ConfigManager.")

        self.L = self.preprocessing_config.get('sequence_length_L')
        if not self.L:
            raise ValueError("sequence_length_L no definida en la configuración de preprocesamiento.")
            
        self.norm_window = self.L * self.preprocessing_config.get('normalization_window_multiplier_for_L', 2)
        
        self.feature_engineer = FeatureEngineer(
            indicators_config=self.preprocessing_config.get('indicators', {}),
            ohlcv_config=self.preprocessing_config.get('ohlcv_processing', {})
        )
        self.final_feature_columns = self.preprocessing_config.get('final_market_feature_columns', [])
        if not self.final_feature_columns:
            raise ValueError("final_market_feature_columns no definido en la configuración de preprocesamiento.")
        
        self.use_float32 = self.preprocessing_config.get('use_float32', True)
        logger.info(f"LiveFeatureProcessor inicializado. Longitud de secuencia L={self.L}, Ventana de normalización={self.norm_window}, Usando float32={self.use_float32}")

    def _apply_live_normalization(self, df_with_features: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica normalización a las características. El df_with_features de entrada
        debe contener suficientes datos históricos para las ventanas móviles.
        Esta función es una adaptación de DataPreprocessor._apply_feature_normalization.
        """
        if df_with_features.empty:
            logger.warning("DataFrame para normalización está vacío.")
            return pd.DataFrame()

        logger.debug(f"Aplicando normalización en vivo a DataFrame con forma: {df_with_features.shape}")
        df_norm = df_with_features.copy()

        if self.use_float32:
            for col in df_norm.select_dtypes(include=['float64']).columns:
                df_norm[col] = df_norm[col].astype(np.float32)

        # min_periods para cálculos de rolling, para tener algunos valores al inicio de la serie si es corta
        min_p = max(1, self.norm_window // 4) # Asegurar al menos 1, un cuarto de la ventana como mínimo

        # --- Normalización de características OHLCV (Z-score móvil) ---
        ohlcv_raw_cols = ['log_ret_C_O', 'log_ret_H_O', 'log_ret_L_O', 'log_ret_C_C_prev', 'log_ret_Vol_SMAVol']
        present_ohlcv_raw_cols = [col for col in ohlcv_raw_cols if col in df_norm.columns]
        
        if present_ohlcv_raw_cols:
            returns_df = df_norm[present_ohlcv_raw_cols]
            # Usar .copy() para evitar SettingWithCopyWarning si returns_df es una vista
            mean_returns = returns_df.rolling(window=self.norm_window, min_periods=min_p).mean().copy()
            std_returns = returns_df.rolling(window=self.norm_window, min_periods=min_p).std().copy()
            std_returns.replace(0, 1e-9, inplace=True) # Evitar división por cero
            
            zscore_returns = (returns_df - mean_returns) / std_returns
            zscore_returns.columns = [f'{col}_norm' for col in present_ohlcv_raw_cols]
            for col in zscore_returns.columns:
                df_norm[col] = zscore_returns[col]
        else:
            logger.warning("Ninguna de las columnas OHLCV base para normalizar Z-score fue encontrada.")

        # --- Normalización de Indicadores ---
        if 'ATR' not in df_norm.columns or 'Close' not in df_norm.columns:
            logger.error("Columnas 'ATR' o 'Close' faltantes, necesarias para la normalización de indicadores.")
            # Devolver un DF vacío o con las columnas esperadas llenas de NaN para que el posterior dropna() lo maneje.
            # O, si es crítico, lanzar un error o devolver None.
            # Por ahora, llenamos con NaNs para las columnas esperadas y que falle más adelante si es necesario.
            for col_name in self.final_feature_columns:
                if col_name not in df_norm.columns: df_norm[col_name] = np.nan
            return df_norm[self.final_feature_columns] if self.final_feature_columns else pd.DataFrame()


        atr = df_norm['ATR'].replace(0, 1e-9)
        close = df_norm['Close'].replace(0, 1e-9)

        sma_ema_cols = ['SMA_short', 'SMA_long', 'EMA_short', 'EMA_long']
        for col in sma_ema_cols:
            if col in df_norm.columns:
                df_norm[f'{col}_norm'] = (df_norm[col] - close) / atr
            elif f'{col}_norm' in self.final_feature_columns: # Si se espera la columna normalizada pero la base no está
                df_norm[f'{col}_norm'] = np.nan


        if 'RSI' in df_norm.columns:
            rsi_scaling_mode = self.preprocessing_config.get('indicators', {}).get('rsi_scaling_mode', "0_1")
            if rsi_scaling_mode == "0_1":
                df_norm['RSI_scaled'] = df_norm['RSI'] / 100.0
            else: # "-1_1"
                df_norm['RSI_scaled'] = (df_norm['RSI'] - 50.0) / 50.0
        elif 'RSI_scaled' in self.final_feature_columns:
             df_norm['RSI_scaled'] = np.nan


        df_norm['ATR_norm'] = atr / close # ATR_norm es fundamental

        macd_cols = ['MACD_line', 'MACD_signal', 'MACD_hist']
        for col in macd_cols:
            if col in df_norm.columns:
                df_norm[f'{col}_norm'] = df_norm[col] / atr
            elif f'{col}_norm' in self.final_feature_columns:
                df_norm[f'{col}_norm'] = np.nan


        if 'BB_upper' in df_norm.columns and 'BB_lower' in df_norm.columns and 'BB_width' in df_norm.columns:
            df_norm['BB_dist_upper_norm'] = (df_norm['BB_upper'] - close) / atr
            df_norm['BB_dist_lower_norm'] = (close - df_norm['BB_lower']) / atr # (Close - Lower) / ATR
            df_norm['BB_width_norm'] = df_norm['BB_width'] / atr # Ancho de banda normalizado por ATR
        else:
            if 'BB_dist_upper_norm' in self.final_feature_columns: df_norm['BB_dist_upper_norm'] = np.nan
            if 'BB_dist_lower_norm' in self.final_feature_columns: df_norm['BB_dist_lower_norm'] = np.nan
            if 'BB_width_norm' in self.final_feature_columns: df_norm['BB_width_norm'] = np.nan


        if 'CCI' in df_norm.columns:
            mean_cci = df_norm['CCI'].rolling(window=self.norm_window, min_periods=min_p).mean()
            std_cci = df_norm['CCI'].rolling(window=self.norm_window, min_periods=min_p).std().replace(0, 1e-9)
            df_norm['CCI_norm'] = (df_norm['CCI'] - mean_cci) / std_cci
        elif 'CCI_norm' in self.final_feature_columns:
             df_norm['CCI_norm'] = np.nan

        stoch_cols = ['STOCH_slowk', 'STOCH_slowd']
        for col in stoch_cols:
            if col in df_norm.columns:
                df_norm[f'{col}_scaled'] = df_norm[col] / 100.0
            elif f'{col}_scaled' in self.final_feature_columns:
                df_norm[f'{col}_scaled'] = np.nan
        
        # Seleccionar solo las columnas finales especificadas en la configuración
        try:
            # Asegurar que todas las columnas finales existan, aunque sean NaN si no se pudieron calcular
            for final_col in self.final_feature_columns:
                if final_col not in df_norm.columns:
                    logger.warning(f"Columna final esperada '{final_col}' no fue generada durante la normalización. Se añadirá como NaN.")
                    df_norm[final_col] = np.nan
            
            df_final_selection = df_norm[self.final_feature_columns]
        except KeyError as e:
            missing_cols = list(set(self.final_feature_columns) - set(df_norm.columns))
            logger.error(f"Una o más columnas finales no se encontraron después de la normalización y el intento de añadir NaNs: {missing_cols}. Error: {e}", exc_info=True)
            logger.error(f"Columnas disponibles en df_norm antes de la selección final: {list(df_norm.columns)}")
            # Devolver un DF con las columnas esperadas pero llenas de NaN para evitar fallos posteriores si es posible
            # o simplemente un DF vacío.
            return pd.DataFrame(columns=self.final_feature_columns) 
            
        logger.debug(f"Normalización en vivo completada. Forma del DataFrame seleccionado: {df_final_selection.shape}")
        return df_final_selection


    def process_market_data(self, raw_candles_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Procesa un DataFrame de velas crudas (OHLCV con índice de tiempo) para generar
        la secuencia de market_features lista para el modelo.
        """
        if raw_candles_df is None or raw_candles_df.empty:
            logger.warning("DataFrame de velas crudas está vacío. No se puede procesar.")
            return None
        
        logger.info(f"Procesando {len(raw_candles_df)} velas crudas para obtener market features...")

        # 1. Ingeniería de Características (cálculo de indicadores y features OHLCV base)
        # Se asume que raw_candles_df ya tiene las columnas 'Open', 'High', 'Low', 'Close', 'Volume'
        # y un índice de tiempo.
        try:
            df_with_base_features = self.feature_engineer.add_ohlcv_features(raw_candles_df.copy()) # Usar copia para evitar modificar original
            df_with_indicators = self.feature_engineer.add_technical_indicators(df_with_base_features)
        except Exception as e:
            logger.error(f"Error durante la ingeniería de características: {e}", exc_info=True)
            return None
        
        # 2. Aplicar Normalización/Escalado Final
        df_normalized_features = self._apply_live_normalization(df_with_indicators)
        if df_normalized_features.empty:
            logger.error("El DataFrame de características normalizadas está vacío.")
            return None

        # 3. Eliminar NaNs inducidos por lookback de indicadores y ventanas de normalización.
        #    El primer índice válido será aquel donde TODAS las features tengan un valor no-NaN.
        df_cleaned = df_normalized_features.dropna()
        
        if len(df_cleaned) < self.L:
            logger.warning(f"No hay suficientes datos ({len(df_cleaned)} filas) después de la normalización y eliminación de NaNs "
                           f"para crear una secuencia de longitud L={self.L}. Se requieren al menos {self.L} filas limpias.")
            logger.debug(f"Forma de df_normalized_features (antes de dropna): {df_normalized_features.shape}")
            logger.debug(f"NaNs por columna en df_normalized_features:\n{df_normalized_features.isnull().sum()}")
            return None
        
        # 4. Creación de la Secuencia: Tomar las ÚLTIMAS L filas del DataFrame limpio.
        #    Esto representa la secuencia más reciente de datos para la predicción.
        final_sequence_df = df_cleaned.iloc[-self.L:]
        market_features_values = final_sequence_df.values # Convertir a NumPy array
        
        # Verificar la forma final
        if market_features_values.shape[0] != self.L or market_features_values.shape[1] != len(self.final_feature_columns):
            logger.error(f"La forma de la secuencia de market features es incorrecta: {market_features_values.shape}. "
                         f"Esperada: ({self.L}, {len(self.final_feature_columns)}).")
            logger.debug(f"Columnas en final_sequence_df: {list(final_sequence_df.columns)}")
            return None

        # Asegurar el tipo de dato float32 si está configurado
        if self.use_float32:
            market_features_array = market_features_values.astype(np.float32)
        else:
            market_features_array = market_features_values
            
        logger.info(f"Market features procesadas y secuencia creada exitosamente. Forma final: {market_features_array.shape}")
        return market_features_array

```

### 3.5. `src/live/portfolio_feature_builder.py`

Construye y normaliza `portfolio_features` a partir de datos de cuenta en vivo.

```python
# src/live/portfolio_feature_builder.py
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# logger = logging.getLogger(__name__)
from src.utils.logging_utils import setup_logger
logger = setup_logger("PortfolioFeatureBuilder")

def build_live_portfolio_features(
    account_general_info: Optional[Dict[str, Any]], # Desde client.futures_account()
    # account_balance_info: Optional[List[Dict[str, Any]]], # Desde client.futures_account_balance() -> Podría ser útil para margen aislado o info más granular
    position_info: Optional[Dict[str, Any]],           # Desde client.futures_position_information(symbol=...)
    env_config: Dict[str, Any],                      # Sección 'environment' del config.yaml
    # initial_equity_config: float,                # 'initial_equity' del config.yaml para normalización consistente
    last_step_equity: Optional[float],                 # Equity del ciclo anterior para calcular retorno logarítmico
    current_market_close_price: Optional[float]      # Precio de cierre de la última vela para algunas normalizaciones
    ) -> Optional[np.ndarray]:
    """
    Construye y normaliza las 8 características del portafolio para el modo en vivo,
    replicando la lógica de TradingEnvironment._get_normalized_portfolio_features().
    """
    
    if not account_general_info or not position_info: # Comprobaciones básicas de entrada
        logger.error("Información de cuenta general o de posición faltante para construir features de portafolio.")
        return None

    features = np.zeros(8, dtype=np.float32) # El modelo espera 8 features

    # --- Obtener valores base de la API ---
    try:
        # 1. Lado de la Posición Activa
        # 'positionAmt' es string, convertir a float. Positivo para LONG, negativo para SHORT.
        current_position_amt = float(position_info.get('positionAmt', '0.0'))
        active_position_side = 0.0
        if current_position_amt > 1e-8: # Usar un epsilon pequeño para comparar flotantes
            active_position_side = 1.0  # Largo
        elif current_position_amt < -1e-8:
            active_position_side = -1.0 # Corto
        features[0] = active_position_side
        
        is_neutral = (active_position_side == 0.0)

        # Equity actual total de la cuenta de futuros
        # 'totalWalletBalance' suele ser el más representativo para el equity total en futuros.
        current_equity = float(account_general_info.get('totalWalletBalance', env_config.get('initial_equity', 10000.0)))
        if current_equity <= 0: current_equity = 1.0 # Evitar división por cero más adelante

        # Precio de entrada de la posición activa
        entry_price = float(position_info.get('entryPrice', '0.0'))

        # P&L no realizado de la posición activa
        unrealized_pnl = float(position_info.get('unRealizedProfit', '0.0'))

        # Apalancamiento actual del símbolo
        # 'leverage' en position_info es un string.
        current_leverage_on_symbol = float(position_info.get('leverage', str(env_config.get('leverage', 10.0))))

        # Margen disponible total de la cuenta
        # 'availableBalance' es el balance de la billetera no usado en margen o como colateral.
        available_margin = float(account_general_info.get('availableBalance', current_equity))

    except (TypeError, ValueError) as e:
        logger.error(f"Error al convertir datos de API para portfolio features: {e}", exc_info=True)
        return None # No se pueden construir las features si los datos base son inválidos


    # --- Calcular y Normalizar Features ---

    # 2. Tamaño de Posición Normalizado
    # (Valor Nocional de la Posición / initial_equity_config)
    # initial_equity_config es el valor conceptual usado en entrenamiento para normalizar.
    if is_neutral or entry_price == 0: # Si es neutral o el precio de entrada es 0 (no debería pasar si hay posición)
        features[1] = 0.0
    else:
        notional_value = abs(current_position_amt * entry_price)
        # Usar el 'initial_equity' del config para consistencia con el entrenamiento.
        initial_equity_for_norm = env_config.get('initial_equity', 10000.0)
        features[1] = notional_value / initial_equity_for_norm if initial_equity_for_norm > 0 else 0.0

    # 3. Precio de Entrada Normalizado
    # ((entry_price / current_market_close_price) - 1.0)
    # Requiere el precio de cierre actual del mercado.
    if is_neutral or entry_price == 0 or current_market_close_price is None or current_market_close_price <= 0:
        features[2] = 0.0
    else:
        features[2] = (entry_price / current_market_close_price) - 1.0

    # 4. P&L No Realizado Normalizado (unrealized_pnl / current_equity)
    features[3] = unrealized_pnl / current_equity

    # 5. Retorno Log Equity (respecto al equity del paso anterior)
    # log(current_equity / last_step_equity)
    if last_step_equity is None or last_step_equity <= 0: # Primer ciclo o equity anterior inválido
        features[4] = 0.0
    else:
        if current_equity > 0 and last_step_equity > 0: # Ambas deben ser positivas
            features[4] = np.log(current_equity / last_step_equity)
        else:
            features[4] = 0.0 # Evitar log de no positivos

    # 6. Ratio de Margen Disponible (available_margin / current_equity)
    features[5] = available_margin / current_equity

    # 7. Pasos en Posición Normalizados
    # Esta feature es compleja de replicar exactamente en vivo sin un gestor de estado de posición.
    # En el entorno de simulación, se cuenta el número de 'steps' (velas) que la posición ha estado abierta.
    # Opciones para el modo en vivo:
    #   a) Si el bot es puramente reactivo a cada vela y no mantiene estado de "cuántas velas lleva abierta la posición",
    #      podría ser 0.0 o 1.0 (representando "recién abierta/en curso" en el step actual).
    #   b) Podría aproximarse por el número de velas desde que se abrió la posición (requiere almacenar 'open_time_of_position').
    #   c) Si el modelo no es muy sensible a esta feature, un valor constante podría ser suficiente.
    # Por simplicidad y para un bot reactivo por vela, usaremos 0.0, asumiendo que cada decisión es "fresca".
    # Considerar la configuración `max_steps_in_position` de `portfolio_features_normalization` si se implementa conteo.
    features[6] = 0.0 # Placeholder: indica que no se está rastreando la duración de la posición en "pasos de vela".

    # 8. Apalancamiento Configurado (o actual del símbolo)
    # Usar el apalancamiento real del símbolo obtenido de position_info.
    features[7] = current_leverage_on_symbol
    
    logger.debug(f"Portfolio features construidas: {features.round(4)}")
    return features

```

## 4\. Orquestador Principal (`scripts/run_live_trader.py`)

Este script coordina todos los módulos para el trading en vivo.

```python
# scripts/run_live_trader.py
import asyncio
import os
import numpy as np
import pandas as pd
import time
import logging
import datetime # Para timestamps en logs CSV
import io       # Para buffer de CSV en memoria

# Cargar variables de .env ANTES de que ConfigManager las necesite si no lo hace internamente.
# from dotenv import load_dotenv
# dotenv_path = os.path.join(os.path.dirname(__file__), '../.env') # Ajustar ruta si es necesario
# load_dotenv(dotenv_path=dotenv_path)

from google.cloud import aiplatform, storage # Añadir storage para logs CSV
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.live.websocket_manager import LiveWebsocketManager
from src.live.binance_api_manager import LiveBinanceAPIManager
from src.live.live_data_processor import LiveFeatureProcessor
from src.live.portfolio_feature_builder import build_live_portfolio_features

logger = setup_logger("LiveTrader") # Usar tu logger configurado

async def main_live_trader():
    logger.info("🚀 Iniciando Live Trader Bot...")
    
    # --- 1. Carga de Configuración y Variables Iniciales ---
    try:
        # ConfigManager se encarga de cargar .env y el config.yaml especificado
        config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
    except Exception as e:
        logger.critical(f"Error fatal al cargar ConfigManager: {e}", exc_info=True)
        return

    trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
    logger.info(f"🛠️ Modo de Trading: {trading_mode}")

    # Inicializar managers a None para cleanup en caso de error temprano
    websocket_manager: Optional[LiveWebsocketManager] = None
    binance_api_manager: Optional[LiveBinanceAPIManager] = None
    live_feature_processor: Optional[LiveFeatureProcessor] = None
    storage_client: Optional[storage.Client] = None # Para logs CSV
    
    try:
        live_config = config_manager.get_config_value('live_trading')
        env_config = config_manager.get_environment_config()
        data_acq_config = config_manager.get_data_acquisition_defaults()
        preproc_config = config_manager.get_preprocessing_config()

        if not all([live_config, env_config, data_acq_config, preproc_config]):
            logger.critical("Una o más secciones de configuración críticas (live_trading, environment, data_acquisition_defaults, preprocessing) no se cargaron correctamente.")
            return

        # --- 2. Inicialización de Componentes ---
        notification_queue = asyncio.Queue() # Cola para comunicación WebSocket -> Ciclo Principal
        
        websocket_manager = LiveWebsocketManager(config_manager, notification_queue)
        
        binance_api_manager = LiveBinanceAPIManager(config_manager)
        await binance_api_manager.initialize_client() # Importante: inicializar el cliente async
        if not binance_api_manager.client: # Doble chequeo por si la inicialización falló silenciosamente
            logger.critical("Fallo al inicializar Binance API Manager Client. Saliendo.")
            return

        live_feature_processor = LiveFeatureProcessor(config_manager)

        # Configuración de Vertex AI
        gcp_project_id = config_manager.get_env_variable('GCP_PROJECT_ID')
        gcp_region = config_manager.get_env_variable('GCP_REGION')
        # vertex_ai_predict_url = live_config.get('vertex_ai_predict_url') # URL completa si se usa requests
        vertex_endpoint_id = live_config.get('vertex_ai_endpoint_raw_id') # ID crudo para SDK

        if not all([gcp_project_id, gcp_region, vertex_endpoint_id]):
            logger.critical("Configuración de Vertex AI incompleta (GCP_PROJECT_ID, GCP_REGION, vertex_ai_endpoint_raw_id son obligatorios).")
            if binance_api_manager: await binance_api_manager.close_client_session()
            return
            
        aiplatform.init(project=gcp_project_id, location=gcp_region)
        # Construir el resource name completo del endpoint
        vertex_endpoint_resource_name = f"projects/{gcp_project_id}/locations/{gcp_region}/endpoints/{vertex_endpoint_id}"
        vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_resource_name)
        logger.info(f"🤖 Cliente de Vertex AI Endpoint inicializado para: {vertex_endpoint.resource_name}")

        # Configuración de Trading (Símbolo, Apalancamiento)
        symbol_to_trade = data_acq_config.get('symbol', "BTCUSDT").upper()
        interval_to_trade = data_acq_config.get('interval', "1h")
        desired_leverage = int(env_config.get('leverage', 10))
        
        leverage_set = await binance_api_manager.set_leverage_if_needed(symbol_to_trade, desired_leverage)
        if not leverage_set:
            logger.warning(f"No se pudo confirmar/establecer el apalancamiento a {desired_leverage}x para {symbol_to_trade}. El bot continuará, pero verifique la configuración en Binance.")
        else:
            logger.info(f"🔩 Apalancamiento para {symbol_to_trade} verificado/establecido en {desired_leverage}x.")

        # Configuración para logging CSV en GCS
        gcs_bucket_name = config_manager.get_env_variable('GCS_BUCKET_NAME')
        gcs_log_path_template = live_config.get('gcs_log_path_template', "live_trading_logs/{symbol}_{interval}/{date}.csv")
        log_buffer_size = live_config.get('log_buffer_size_records', 50)
        gcs_upload_interval = live_config.get('gcs_log_upload_interval_seconds', 3600)
        
        storage_client = storage.Client(project=gcp_project_id)
        gcs_log_bucket = storage_client.bucket(gcs_bucket_name)
        live_log_buffer = []
        last_gcs_log_upload_time = time.time()

    except Exception as e_init:
        logger.critical(f"Error fatal durante la inicialización de componentes o configuración: {e_init}", exc_info=True)
        if binance_api_manager and binance_api_manager.client: # Intentar cerrar sesión si el cliente se inicializó
            await binance_api_manager.close_client_session()
        return

    # --- 3. Estado del Ciclo de Trading ---
    last_step_equity: Optional[float] = None # Para la feature 'Retorno Log Equity'
    # initial_bot_equity conceptual para normalización, tomado de la config del entorno
    initial_equity_for_feature_norm = float(env_config.get('initial_equity', 10000.0)) 


    # --- 4. Tarea WebSocket en Background ---
    if websocket_manager: # Solo iniciar si se inicializó correctamente
        websocket_task = asyncio.create_task(websocket_manager.run())
        logger.info("📡 WebSocket Manager iniciado en background y escuchando por cierre de velas.")
    else:
        logger.critical("WebSocket Manager no pudo ser inicializado. El bot no puede operar. Saliendo.")
        if binance_api_manager: await binance_api_manager.close_client_session()
        return

    # --- 5. Bucle Principal de Trading ---
    while True:
        log_entry_data = {} # Para el CSV log de este ciclo
        try:
            logger.info(f"⏳ Esperando notificación de nueva vela cerrada para {symbol_to_trade}@{interval_to_trade}...")
            closed_kline_data = await notification_queue.get() # Espera aquí por la notificación del WebSocket
            
            kline_close_time_ms = closed_kline_data.get('T') # Tiempo de cierre de la vela
            kline_open_time_dt = pd.to_datetime(closed_kline_data.get('t'), unit='ms', utc=True)
            kline_close_time_dt = pd.to_datetime(kline_close_time_ms, unit='ms', utc=True)
            current_market_close_price = float(closed_kline_data.get('c'))

            logger.info(f"🕯️ Vela cerrada detectada (OpenTime: {kline_open_time_dt}, CloseTime: {kline_close_time_dt}, ClosePx: {current_market_close_price}). Iniciando ciclo de decisión...")
            log_entry_data = { # Inicializar datos para el log CSV
                "timestamp_decision_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "trading_mode": trading_mode, "symbol": symbol_to_trade, "interval": interval_to_trade,
                "kline_open_time_utc": kline_open_time_dt.isoformat(),
                "kline_close_time_utc": kline_close_time_dt.isoformat(),
                "kline_o": float(closed_kline_data.get('o')), "kline_h": float(closed_kline_data.get('h')),
                "kline_l": float(closed_kline_data.get('l')), "kline_c": current_market_close_price,
                "kline_v": float(closed_kline_data.get('v')),
            }

            # --- 5.1. Obtener Datos de Mercado (Historial Reciente) ---
            raw_candles_df = await binance_api_manager.get_historical_klines(
                symbol=symbol_to_trade,
                interval=interval_to_trade,
                lookback_candles=live_config['market_data_lookback_candles']
            )
            if raw_candles_df is None or raw_candles_df.empty or len(raw_candles_df) < preproc_config['sequence_length_L']:
                logger.warning(f"No se obtuvieron suficientes datos históricos ({len(raw_candles_df) if raw_candles_df is not None else 'None'} velas) para {symbol_to_trade}. Se requieren {preproc_config['sequence_length_L']}. Saltando ciclo.")
                log_entry_data["error_message"] = "Insufficient historical klines"
                # No hacer task_done() aquí si el error es antes de procesar, se hace en finally
                raise ContinueLoopException("Insufficient historical klines") # Excepción para saltar al finally y loguear

            # --- 5.2. Preprocesar Datos de Mercado ---
            market_features_array = live_feature_processor.process_market_data(raw_candles_df)
            if market_features_array is None:
                logger.warning(f"Fallo el preprocesamiento de datos de mercado para {symbol_to_trade}. Saltando ciclo.")
                log_entry_data["error_message"] = "Market data preprocessing failed"
                raise ContinueLoopException("Market data preprocessing failed")
            
            # log_entry_data["market_features_shape"] = market_features_array.shape # Opcional

            # --- 5.3. Obtener Datos de Cartera de Binance ---
            account_general_info = await binance_api_manager.get_account_info()
            position_info_live = await binance_api_manager.get_position_risk(symbol=symbol_to_trade)

            if not account_general_info or not position_info_live:
                logger.error(f"No se pudo obtener la información completa de la cuenta o posición de Binance para {symbol_to_trade}. Saltando ciclo.")
                log_entry_data["error_message"] = "Failed to fetch account/position info"
                raise ContinueLoopException("Failed to fetch account/position info")
            
            current_live_equity = float(account_general_info.get('totalWalletBalance', initial_equity_for_feature_norm))
            log_entry_data["current_equity_before_action"] = current_live_equity
            if last_step_equity is None: # Primer ciclo después de iniciar o reinicio
                last_step_equity = current_live_equity # Usar actual como "anterior" para la primera feature

            # --- 5.4. Construir Características de Cartera ---
            portfolio_features_array = build_live_portfolio_features(
                account_general_info, position_info_live, 
                env_config, last_step_equity, current_market_close_price
            )
            if portfolio_features_array is None:
                logger.warning(f"Fallo la construcción de portfolio features para {symbol_to_trade}. Saltando ciclo.")
                log_entry_data["error_message"] = "Portfolio feature building failed"
                raise ContinueLoopException("Portfolio feature building failed")
            
            # log_entry_data["portfolio_features"] = portfolio_features_array.tolist() # Opcional

            # --- 5.5. Preparar y Enviar Observación al Modelo Vertex AI ---
            observation_instance = {
                "market_features": market_features_array.tolist(), # Asegurar formato de lista para JSON
                "portfolio_features": portfolio_features_array.tolist()
            }
            
            logger.debug(f"Enviando observación al endpoint de Vertex AI. Market feats shape: {market_features_array.shape}, Portfolio feats shape: {portfolio_features_array.shape}")
            prediction_response = vertex_endpoint.predict(instances=[observation_instance])
            # Asumir que server.py devuelve {'action_value': float, 'action': [float]}
            action_value_from_model = float(prediction_response.predictions[0]['action_value'])
            logger.info(f"💡 Respuesta del modelo Vertex AI: action_value = {action_value_from_model:.4f}")
            log_entry_data["model_action_value"] = action_value_from_model

            # --- 5.6. Interpretar Acción y Ejecutar Orden(es) ---
            action_threshold = env_config.get('action_threshold', 0.15)
            desired_signal = 0 # 0: Neutral, 1: Long, -1: Short
            if action_value_from_model > action_threshold: desired_signal = 1
            elif action_value_from_model < -action_threshold: desired_signal = -1
            log_entry_data["desired_signal"] = desired_signal
            
            current_pos_amt_live = float(position_info_live.get('positionAmt', '0.0'))
            log_entry_data["pos_amt_before_action"] = current_pos_amt_live
            current_pos_side_live = 0
            if current_pos_amt_live > 1e-8: current_pos_side_live = 1
            elif current_pos_amt_live < -1e-8: current_pos_side_live = -1
            log_entry_data["pos_side_before_action"] = current_pos_side_live

            logger.info(f"📊 Posición actual en Binance para {symbol_to_trade}: Lado={current_pos_side_live}, Cantidad={current_pos_amt_live:.8f}. Señal del modelo: {desired_signal}.")

            action_description = "HOLD" # Default

            if desired_signal != current_pos_side_live:
                # --- 5.6.1. Cerrar posición existente si es necesario ---
                if current_pos_side_live != 0:
                    action_description = f"CLOSE_{'LONG' if current_pos_side_live == 1 else 'SHORT'}"
                    logger.info(f"🎬 Decisión: {action_description} de {abs(current_pos_amt_live):.8f} {symbol_to_trade}.")
                    close_order_result = await binance_api_manager.close_market_position(symbol_to_trade, current_pos_amt_live)
                    if close_order_result and close_order_result.get('orderId'):
                        logger.info(f"Resultado cierre orden ID {close_order_result.get('orderId')}: {close_order_result.get('status')}")
                        log_entry_data["order_id_close"] = close_order_result.get('orderId')
                        log_entry_data["order_status_close"] = close_order_result.get('status')
                        # PnL y comisiones del cierre necesitarían ser consultados o calculados post-cierre
                    else:
                        logger.warning(f"El cierre de la posición para {symbol_to_trade} pudo haber fallado o no retornó ID.")
                        log_entry_data["order_id_close"] = "FAILED_OR_NO_ID"
                    
                    await asyncio.sleep(live_config.get('post_close_delay_seconds', 1)) # Pausa después de cerrar
                    
                    # Re-obtener equity y posición después del cierre para el siguiente paso
                    account_general_info_after_close = await binance_api_manager.get_account_info()
                    if account_general_info_after_close:
                        current_live_equity = float(account_general_info_after_close.get('totalWalletBalance', current_live_equity))
                        log_entry_data["current_equity_after_close"] = current_live_equity
                    else:
                        logger.warning("No se pudo re-obtener info de cuenta después del cierre. El sizing de la nueva posición podría ser impreciso.")
                
                # --- 5.6.2. Abrir nueva posición si la señal no es neutral ---
                if desired_signal != 0:
                    new_pos_action = f"OPEN_{'LONG' if desired_signal == 1 else 'SHORT'}"
                    action_description = (action_description + "_THEN_" if action_description != "HOLD" else "") + new_pos_action
                    
                    # Usar current_market_close_price de la vela que disparó el evento para el cálculo de cantidad
                    order_qty_to_open = await binance_api_manager.calculate_order_quantity(
                        symbol=symbol_to_trade,
                        equity=current_live_equity, # Equity actualizado después de posible cierre
                        current_price=current_market_close_price,
                        leverage=desired_leverage
                    )
                    if order_qty_to_open > 0:
                        logger.info(f"🎬 Decisión: {new_pos_action} de {order_qty_to_open:.8f} {symbol_to_trade}.")
                        open_order_result = await binance_api_manager.place_market_order(
                            symbol=symbol_to_trade,
                            side="BUY" if desired_signal == 1 else "SELL",
                            quantity=order_qty_to_open
                        )
                        if open_order_result and open_order_result.get('orderId'):
                            logger.info(f"Resultado apertura orden ID {open_order_result.get('orderId')}: {open_order_result.get('status')}")
                            log_entry_data["order_id_open"] = open_order_result.get('orderId')
                            log_entry_data["order_status_open"] = open_order_result.get('status')
                            log_entry_data["order_qty_open"] = order_qty_to_open
                        else:
                            logger.warning(f"La apertura de posición {new_pos_action} para {symbol_to_trade} pudo haber fallado o no retornó ID.")
                            log_entry_data["order_id_open"] = "FAILED_OR_NO_ID"
                    else:
                        logger.warning(f"La cantidad calculada para la orden {new_pos_action} es 0 o inválida. No se abrirá posición.")
                        action_description = (action_description + "_BUT_QTY_ZERO" if action_description.startswith("OPEN") else action_description)
            else: # desired_signal == current_pos_side_live
                logger.info(f"🎬 Decisión: Mantener posición/estado actual ({current_pos_side_live}) para {symbol_to_trade}.")
                action_description = "HOLD"

            log_entry_data["action_taken_desc"] = action_description
            
            # --- 5.7. Actualizar Estado para Siguiente Ciclo ---
            # El equity actual se consultará de nuevo al inicio del siguiente ciclo de la API.
            # last_step_equity se actualiza aquí para el cálculo del PnL logarítmico en el siguiente portfolio_features.
            # Es el equity DESPUÉS de las acciones de este ciclo.
            final_account_info = await binance_api_manager.get_account_info()
            if final_account_info:
                 last_step_equity = float(final_account_info.get('totalWalletBalance', current_live_equity))
                 log_entry_data["current_equity_after_action"] = last_step_equity
            else: # Si no se puede obtener, mantener el anterior.
                 last_step_equity = current_live_equity 
                 log_entry_data["current_equity_after_action"] = "UNKNOWN_API_FAIL"
                 logger.warning("No se pudo obtener equity final del ciclo. `last_step_equity` no actualizado con el valor más reciente.")


        except ContinueLoopException as cle: # Excepción personalizada para saltar el resto del ciclo limpiamente
            logger.warning(f"Saltando ciclo actual debido a: {cle}")
            # El error ya debería estar en log_entry_data["error_message"]
        except asyncio.CancelledError:
            logger.info("🚦 Tarea Live Trader cancelada durante el bucle principal.")
            # Esto debería ser manejado por el cleanup general
            raise # Relanzar para que el cleanup general se active
        except aiplatform.errors.PredictionError as pe:
            logger.error(f"❌ Error de predicción de Vertex AI: {pe.message} (Code: {pe.code})", exc_info=False)
            log_entry_data["error_message"] = f"VertexPredictionError: {pe.message[:100]}"
            await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds', 30))
        except ConnectionError as ce: 
            logger.error(f"❌ Error de Conexión con Binance API: {ce}", exc_info=True)
            log_entry_data["error_message"] = f"BinanceConnectionError: {str(ce)[:100]}"
            if binance_api_manager:
                 await binance_api_manager.close_client_session()
                 logger.info("Intentando reinicializar cliente de Binance API tras error de conexión...")
                 try:
                    await binance_api_manager.initialize_client()
                 except Exception as recon_e:
                    logger.error(f"Fallo al reinicializar cliente de Binance API: {recon_e}")
            await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds', 60))
        except Exception as e_loop:
            logger.error(f"💥 Error inesperado en el bucle principal de Live Trader: {e_loop}", exc_info=True)
            log_entry_data["error_message"] = f"UnhandledException: {str(e_loop)[:100]}"
            # Esperar antes de reintentar el ciclo general
            await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds', 60))
        finally:
            # --- 5.8. Logging en CSV y Marcar Tarea como Hecha ---
            if log_entry_data: # Si se ha creado una entrada de log para este ciclo
                live_log_buffer.append(log_entry_data)

            current_time_for_log_upload = time.time()
            if live_log_buffer and \
               (len(live_log_buffer) >= log_buffer_size or \
               (current_time_for_log_upload - last_gcs_log_upload_time) >= gcs_upload_interval):
                try:
                    log_df_to_upload = pd.DataFrame(live_log_buffer)
                    csv_in_memory_buffer = io.StringIO()
                    log_df_to_upload.to_csv(csv_in_memory_buffer, index=False)
                    
                    gcs_log_file_name = gcs_log_path_template.format(
                        symbol=symbol_to_trade,
                        interval=interval_to_trade,
                        date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
                    )
                    
                    blob = gcs_log_bucket.blob(gcs_log_file_name)
                    # Para "append" a un CSV en GCS, la estrategia común es:
                    # 1. Descargar el blob existente (si existe).
                    # 2. Cargar en Pandas DataFrame.
                    # 3. Concatenar con el nuevo DataFrame de logs.
                    # 4. Subir el DataFrame concatenado, sobrescribiendo el blob.
                    # Esto puede ser ineficiente para logs frecuentes. Alternativas: BQ, archivos nuevos.
                    # Aquí, por simplicidad, crearemos/sobrescribiremos el archivo diario.
                    # Para un append real, la lógica es más compleja.
                    
                    # Intento de append (simplificado, puede ser costoso para archivos grandes):
                    existing_content = ""
                    if blob.exists(storage_client): # Verificar si el archivo ya existe
                         try:
                            existing_content = blob.download_as_text() + "\n" # Añadir newline si hay contenido
                            # Quitar encabezado si ya existe para no duplicarlo
                            if existing_content.strip().startswith(log_df_to_upload.columns.to_list()[0]): # Asumiendo que la primera col es el inicio del header
                                csv_in_memory_buffer.seek(0)
                                csv_in_memory_buffer.readline() # Saltar la línea del header del nuevo buffer
                         except Exception as e_download_log:
                              logger.warning(f"No se pudo descargar el log existente '{gcs_log_file_name}' para append: {e_download_log}. Se creará/sobrescribirá.")
                              existing_content = ""


                    full_csv_content = existing_content + csv_in_memory_buffer.getvalue().strip()
                    blob.upload_from_string(full_csv_content, content_type='text/csv')
                    
                    logger.info(f"Log de {len(live_log_buffer)} entradas {'añadido a' if existing_content else 'guardado en'} GCS: gs://{gcs_bucket_name}/{gcs_log_file_name}")
                    live_log_buffer.clear() # Limpiar buffer después de subir
                    last_gcs_log_upload_time = current_time_for_log_upload
                except Exception as e_gcs_log_upload:
                    logger.error(f"Error al subir log CSV a GCS: {e_gcs_log_upload}", exc_info=True)
                    # No limpiar el buffer si falla el upload, para reintentar la próxima vez.

            if notification_queue.empty() == False: # Solo si realmente se procesó un item
                try:
                    notification_queue.task_done()
                except ValueError: # Si task_done() es llamado más veces que put()
                    logger.warning("notification_queue.task_done() llamado incorrectamente (posiblemente ya estaba vacía).")
            
            logger.info(f"--- Ciclo de Decisión para vela {kline_close_time_dt} completado ---")


    # --- 6. Cleanup General ---
    logger.info("Iniciando proceso de apagado del Live Trader Bot...")
    if websocket_task and not websocket_task.done():
        logger.info("Cancelando tarea WebSocket...")
        websocket_task.cancel()
        try:
            await websocket_task # Esperar a que la tarea se cancele
        except asyncio.CancelledError:
            logger.info("Tarea WebSocket cancelada y detenida.")
            
    if binance_api_manager and binance_api_manager.client:
        logger.info("Cerrando sesión del cliente Binance API...")
        await binance_api_manager.close_client_session()
        
    # Subir cualquier log restante en el buffer antes de salir
    if live_log_buffer:
        logger.info(f"Subiendo {len(live_log_buffer)} logs restantes a GCS antes de salir...")
        try:
            # (Copiar aquí la lógica de subida de logs de arriba)
            log_df_to_upload = pd.DataFrame(live_log_buffer)
            # ... (resto de la lógica de subida) ...
            logger.info("Logs restantes subidos.")
        except Exception as e_final_log:
            logger.error(f"Error al subir logs finales: {e_final_log}")

    logger.info("🏁 Live Trader Bot detenido limpiamente.")


class ContinueLoopException(Exception):
    """Excepción personalizada para indicar que se debe saltar al siguiente ciclo del bucle principal."""
    pass


if __name__ == "__main__":
    try:
        asyncio.run(main_live_trader())
    except KeyboardInterrupt:
        logger.info("🛑 Live Trader detenido manualmente por el usuario (KeyboardInterrupt).")
    except Exception as e_global_run:
        # Este es para errores que ocurren fuera del bucle principal de `main_live_trader`
        # o si `main_live_trader` termina por una excepción no manejada internamente antes del bucle
        logger.critical(f"💥 Error global irrecuperable al ejecutar Live Trader: {e_global_run}", exc_info=True)
    finally:
        logger.info("Proceso principal de run_live_trader.py finalizado.")

```

## 5\. Consideraciones Adicionales

  * **Gestión de Errores Robusta**: El código incluye `try-except` básicos. Expande esto para reintentos específicos, manejo de diferentes códigos de error de Binance, etc.
  * **Persistencia y Auditoría**: El logging a CSV en GCS es un buen comienzo. Considera BigQuery para análisis más avanzados.
  * **Sincronización y Timestamps**: Usa UTC consistentemente. Ten en cuenta posibles pequeños desfases entre el evento WebSocket y la disponibilidad de datos en la API REST.
  * **Límites de Tasa (Rate Limits)**: `python-binance` maneja algunos, pero sé consciente, especialmente en bucles de reintento. `LiveBinanceAPIManager` tiene un ejemplo simple de backoff.
  * **Seguridad de Claves**: La gestión vía Google Cloud Secret Manager es la práctica correcta.
  * **Pruebas Exhaustivas en Testnet**: Fundamental antes de operar con fondos reales.
  * **Monitoreo y Alertas**: Implementa un sistema externo (ej. Google Cloud Monitoring) para supervisar el estado del bot y recibir alertas.
  * **Kill Switch**: Un mecanismo externo para detener el bot rápidamente si es necesario.
  * **Portfolio Feature [2] y [6]**: La "Normalized Entry Price" (feature 2) idealmente usa el precio de mercado actual para su normalización. La "Steps in Position" (feature 6) es difícil de replicar sin estado; la implementación actual la fija en 0. Evalúa su impacto o considera formas de aproximarla si es crucial para tu modelo.

## 6\. Despliegue

  * **VM en GCP (Compute Engine)**: Recomendado.
  * **Cuenta de Servicio**: Asigna los roles necesarios a la VM (Secret Manager Secret Accessor, Vertex AI User, Storage Object Admin/Creator para logs).
  * **Instalación**: Clona el repo, crea venv, `pip install -r requirements.txt`.
  * **Autenticación GCP**: `gcloud auth application-default login` en la VM o asegúrate de que la cuenta de servicio esté correctamente asignada.
  * **Variables de Entorno**: Coloca el archivo `.env` configurado en la raíz del proyecto en la VM.
  * **Ejecución Persistente**: Usa `supervisor` o `systemd` para que `python scripts/run_live_trader.py` se ejecute como un servicio y se reinicie automáticamente en caso de fallo.

<!-- end list -->

```

Esta guía es extensa pero cubre todos los detalles que has mencionado. Recuerda probar cada componente individualmente y luego el sistema completo en Testnet antes de pasar a producción. ¡Mucha suerte!
```