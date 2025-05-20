# BTCBot: Guía de Implementación para Trading en Vivo con WebSockets (Gestión de Secretos Unificada)

## 1. Introducción

Esta guía detalla la arquitectura y los componentes necesarios para operar el BTCBot en un entorno de trading en vivo (o en Testnet) con Binance Futures. El sistema se basa en la detección de cierre de velas mediante WebSockets para una alta reactividad y utiliza llamadas a la API REST para la obtención de datos históricos necesarios para el preprocesamiento, la gestión de la cuenta y la ejecución de órdenes. Todas las credenciales de API (tanto para el entorno real como para Testnet) se gestionarán a través de Google Cloud Secret Manager.

**Objetivo Principal:** Tomar decisiones de trading basadas en un modelo de RL desplegado en Vertex AI, interactuando en tiempo real con la API de Binance Futures de forma segura.

**Flujo General por Ciclo de Decisión:**
1.  **Detección de Cierre de Vela:** Un gestor de WebSockets notifica el cierre de una nueva vela para el par y el intervalo configurados.
2.  **Obtención de Datos de Mercado:** Tras la notificación, se descarga la ventana histórica de velas necesaria (ej. ~250 velas) desde la API REST de Binance, finalizando con la vela recién cerrada.
3.  **Preprocesamiento de Datos de Mercado:** Los datos crudos se transforman en `market_features` (secuencia `L x N_features`) usando la misma lógica que en el entrenamiento.
4.  **Obtención de Datos de Cartera:** Se consulta a la API REST de Binance el estado actual de la cuenta (equity, margen, P&L) y la posición activa (si existe).
5.  **Construcción de Características de Cartera:** Los datos reales de la cartera se transforman y normalizan en `portfolio_features`.
6.  **Inferencia del Modelo:** La observación combinada (`market_features` + `portfolio_features`) se envía al endpoint del modelo en Vertex AI.
7.  **Interpretación y Ejecución:** La acción devuelta por el modelo se interpreta para decidir si se abre, cierra o mantiene una posición. Las órdenes se ejecutan a través de la API REST de Binance.
8.  **Registro y Espera:** Se registran todas las acciones y el sistema espera la notificación del cierre de la siguiente vela a través del WebSocket.

---

## 2. Configuración del Entorno

### 2.1. Variables de Entorno y Flags

La distinción entre Testnet y Real se manejará mediante una variable de entorno que determinará qué conjunto de secretos se utiliza.

* **Archivo `.env` (Raíz del Proyecto):**
    * Asegúrate de que este archivo contenga solo los nombres de las variables de entorno que el código buscará. Los *valores* de los secretos de Binance se obtendrán exclusivamente de Google Secret Manager.

        ```dotenv
        # Nombres de los Secretos en Google Cloud Secret Manager
        # Estos son los NOMBRES de los secretos que ConfigManager buscará.
        # Debes crear estos secretos en Google Cloud Secret Manager con los valores de tus claves API.

        # Para la cuenta REAL de Binance Futures
        SECRET_NAME_BINANCE_API_KEY_FUTURES="BINANCE_API_KEY_FUTURES"
        SECRET_NAME_BINANCE_API_SECRET_FUTURES="BINANCE_API_SECRET_FUTURES"

        # Para la cuenta TESTNET de Binance Futures
        SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES="TESTNET_BINANCE_API_KEY_FUTURES"
        SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES="TESTNET_BINANCE_API_SECRET_FUTURES"

        # GCP Configuration
        GCP_PROJECT_ID="tu-gcp-project-id" # Ej: bitcoin-460320
        GCS_BUCKET_NAME="tu-gcs-bucket-name" # Ej: bitcoin-460320_data
        GCP_REGION="tu-gcp-region" # Ej: europe-southwest1

        # Live Trading Mode: TESTNET o REAL
        # Esta variable determinará qué conjunto de NOMBRES de secretos se utilizan.
        LIVE_TRADING_MODE="TESTNET"
        ```

* **`src/utils/config.py` (Funcionamiento con Secretos):**
    * Tu `ConfigManager` ya está diseñado para obtener secretos si el nombre de la variable está en la lista `secretos = ["BINANCE_API_KEY_FUTURES", "BINANCE_API_SECRET_FUTURES"]`.
    * Para que esto funcione con los nombres de secretos definidos en `.env` y el modo Testnet/Real:
        * El módulo `BinanceAPIManager` (ver sección 3.2) leerá `LIVE_TRADING_MODE`.
        * Luego, construirá dinámicamente el *nombre del secreto* a solicitar a `ConfigManager`. Por ejemplo, si `LIVE_TRADING_MODE` es "TESTNET", solicitará los secretos cuyos nombres fueron cargados desde `SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES` y `SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES` del archivo `.env`.
        * `ConfigManager.get_env_variable()` buscará estos nombres. Si estos nombres coinciden con los que has predefinido como "secretos" en la lista `secretos` dentro de `ConfigManager`, entonces intentará obtenerlos de Secret Manager. **Deberás añadir los nombres de los secretos de Testnet (ej. "TESTNET_BINANCE_API_KEY_FUTURES") a la lista `secretos` dentro de `ConfigManager` para que los trate como tales.**

    * **Modificación sugerida en `ConfigManager` o en `BinanceAPIManager`:**
        Al inicializar `BinanceAPIManager`:
        ```python
        # Dentro de BinanceAPIManager.__init__
        # ...
        # trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
        #
        # if trading_mode == 'TESTNET':
        #     # Obtener los NOMBRES de los secretos de Testnet desde las variables de entorno
        #     # (que a su vez fueron leídas del .env por load_dotenv)
        #     api_key_secret_name = os.getenv('SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES')
        #     api_secret_secret_name = os.getenv('SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES')
        # else: # REAL
        #     api_key_secret_name = os.getenv('SECRET_NAME_BINANCE_API_KEY_FUTURES')
        #     api_secret_secret_name = os.getenv('SECRET_NAME_BINANCE_API_SECRET_FUTURES')
        #
        # # Ahora solicitar los valores de estos secretos a ConfigManager
        # # ConfigManager intentará obtenerlos de Secret Manager si los nombres están en su lista 'secretos'
        # self.api_key = config_manager.get_env_variable(api_key_secret_name)
        # self.api_secret = config_manager.get_env_variable(api_secret_secret_name)
        # ...
        # # Y en ConfigManager, asegurarse de que la lista 'secretos' incluya todos los posibles nombres:
        # # secretos = ["BINANCE_API_KEY_FUTURES", "BINANCE_API_SECRET_FUTURES",
        # #             "TESTNET_BINANCE_API_KEY_FUTURES", "TESTNET_BINANCE_API_SECRET_FUTURES"]
        ```

* **URLs Base de la API de Binance y WebSocket:**
    * El cliente `python-binance` y el conector WebSocket deben usar los endpoints correctos:
        * **API REST Real:** `https://fapi.binance.com`
        * **API REST Testnet:** `https://testnet.binancefuture.com`
        * **WebSocket Real:** `wss://fstream.binance.com/ws/`
        * **WebSocket Testnet:** `wss://stream.binancefuture.com/ws/`
    * Esto se configura al instanciar `Client(api_key, api_secret, testnet=(trading_mode == 'TESTNET'))` y al construir la URL del WebSocket en `WebsocketManager`.

### 2.2. Fichero `src/config.yaml`

Asegurar la disponibilidad de los siguientes parámetros, accesibles vía `ConfigManager`:
* `data_acquisition_defaults`: `symbol`, `interval`.
* `preprocessing`: `sequence_length_L`, `normalization_window_multiplier_for_L`, `ohlcv_processing`, `indicators`, `final_market_feature_columns`.
* `environment`: `leverage`, `position_size_pct_equity`, `action_threshold`, `min_order_size_btc` (aunque los mínimos reales se verificarán con `exchangeInfo`).
* Añadir/modificar la sección para configuración específica del trading en vivo:
    ```yaml
    # Dentro de src/config.yaml

    live_trading:
      # URL completa del endpoint de predicción en Vertex AI
      # Formato: https://{region}[-aiplatform.googleapis.com/v1/projects/](https://-aiplatform.googleapis.com/v1/projects/){project_id}/locations/{region}/endpoints/{endpoint_id}:predict
      vertex_ai_predict_url: "[https://europe-southwest1-aiplatform.googleapis.com/v1/projects/TU_GCP_PROJECT_ID/locations/europe-southwest1/endpoints/TU_ENDPOINT_ID:predict](https://europe-southwest1-aiplatform.googleapis.com/v1/projects/TU_GCP_PROJECT_ID/locations/europe-southwest1/endpoints/TU_ENDPOINT_ID:predict)" # REEMPLAZAR ESTOS VALORES
      
      # ID "crudo" del endpoint de Vertex AI (solo el número, para usar con la SDK de aiplatform)
      vertex_ai_endpoint_raw_id: "TU_ENDPOINT_ID" # REEMPLAZAR ESTE VALOR

      # Número de velas a descargar para el preprocesamiento en cada ciclo
      market_data_lookback_candles: 250 

      # Segundos a esperar entre reintentos de conexión del WebSocket
      websocket_retry_delay_seconds: 10
      websocket_unexpected_error_delay_seconds: 30

      # Segundos a esperar entre reintentos para obtener la última vela cerrada si no está disponible inmediatamente
      kline_availability_retry_delay_seconds: 5
      kline_availability_max_retries: 12 # ej. 12 * 5s = 1 minuto de reintentos

      # Pequeña pausa en segundos después de cerrar una posición antes de abrir una nueva (si aplica)
      post_close_delay_seconds: 1
    ```

---

## 3. Módulos Principales para Trading en Vivo

Se recomienda crear un nuevo directorio `src/live/` para estos módulos.

### 3.1. `src/live/websocket_manager.py`

* **Propósito:** Gestionar la conexión WebSocket para detectar el cierre de nuevas velas de forma reactiva.
* **Implementación:**
    * Usar `asyncio` y la librería `websockets`.
    * **Clase `LiveWebsocketManager`:**
        * **`__init__(self, config_manager: ConfigManager, notification_queue: asyncio.Queue)`:**
            * Obtiene `symbol` e `interval` de `config_manager.get_data_acquisition_defaults()`.
            * Obtiene `LIVE_TRADING_MODE` del `os.getenv()`.
            * Construye la `self.websocket_url` correcta (Testnet o Real).
            * Almacena la `notification_queue`.
            * Obtiene `websocket_retry_delay_seconds` y `websocket_unexpected_error_delay_seconds` de `config_manager.get_config_value('live_trading')`.
        * **`async def _process_message(self, message_str: str)`:** (Igual que en la guía anterior)
            * Parsea JSON. Si es k-line cerrada (`message['k']['x'] == True`), pone `message['k']` en `self.notification_queue`.
        * **`async def run(self)`:** (Igual que en la guía anterior, usando los delays de reintento de la config)
            * Bucle infinito `try/except` con `websockets.connect()`.
            * Bucle interno `await ws.recv()` y llama a `_process_message()`.
            * Manejo de excepciones con reintentos y `await asyncio.sleep(self.retry_delay)`.

### 3.2. `src/live/binance_api_manager.py`

* **Propósito:** Encapsular todas las interacciones con la API REST de Binance Futures, manejando la distinción entre Testnet y Real y obteniendo credenciales de Secret Manager.
* **Implementación:**
    * **Clase `LiveBinanceAPIManager`:**
        * **`__init__(self, config_manager: ConfigManager)`:**
            * `self.config_manager = config_manager`
            * `self.trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()`
            * Lógica para determinar `api_key_secret_name` y `api_secret_secret_name` basada en `self.trading_mode` y los nombres de secretos en `.env` (como se describió en la sección 2.1).
            * `api_key = self.config_manager.get_env_variable(api_key_secret_name)`
            * `api_secret = self.config_manager.get_env_variable(api_secret_secret_name)`
            * Si `api_key` o `api_secret` son `None`, lanzar un error crítico.
            * Instanciar `python-binance.AsyncClient(api_key, api_secret, testnet=(self.trading_mode == 'TESTNET'))`.
            * Almacenar `env_config = config_manager.get_environment_config()`.
        * **Métodos asíncronos (`async def`) para:** (Igual que en la guía anterior)
            * `get_historical_klines(self, symbol: str, interval: str, lookback_candles: int) -> pd.DataFrame`
            * `get_account_balance(self) -> dict | None`
            * `get_account_info(self) -> dict | None`
            * `get_position_risk(self, symbol: str) -> dict | None`
            * `get_exchange_info(self) -> dict | None`
            * `get_symbol_filters(self, symbol: str) -> dict | None`
            * `calculate_order_quantity(self, symbol: str, equity: float, current_price: float, leverage: float) -> float`: Usa `self.env_config['position_size_pct_equity']`. Obtiene filtros con `get_symbol_filters`.
            * `place_market_order(self, symbol: str, side: str, quantity: float) -> dict | None`
            * `close_market_position(self, symbol: str, position_amt_to_close: float) -> dict | None`
            * `set_leverage_if_needed(self, symbol: str, desired_leverage: int)`: Usa `self.env_config['leverage']`.
    * Todo método que interactúe con la API de Binance debe estar envuelto en un `try-except` robusto, logueando errores y devolviendo `None` o lanzando excepciones personalizadas si la operación falla después de reintentos.

### 3.3. `src/live/live_data_processor.py`

* **Propósito:** Transformar datos crudos de velas en `market_features`. Sin cambios respecto a la guía anterior, ya que su lógica es independiente del origen de las claves API.
* **Implementación:**
    * **Clase `LiveFeatureProcessor`:**
        * `__init__(self, config_manager: ConfigManager)`
        * `_apply_live_normalization(self, df_with_features: pd.DataFrame) -> pd.DataFrame`
        * `process_market_data(self, raw_candles_df: pd.DataFrame) -> np.ndarray | None`

### 3.4. `src/live/portfolio_feature_builder.py`

* **Propósito:** Construir y normalizar `portfolio_features`. Sin cambios directos por la gestión de secretos, pero su entrada (`account_info`, `position_info`) vendrá del `LiveBinanceAPIManager` que sí maneja los secretos.
* **Implementación:**
    * **Función `build_live_portfolio_features(account_info: dict, account_balance_info: dict, position_info: dict, env_config: dict, initial_equity_config: float, last_step_equity: float) -> np.ndarray`:** (Añadido `account_balance_info` y `last_step_equity` para mayor precisión)
        * `last_step_equity` es el equity del ciclo anterior, necesario para `log(current_equity / last_step_equity)`.
        * El resto de la lógica como se describió antes, replicando `TradingEnvironment._get_normalized_portfolio_features()`.

---

## 4. Orquestador Principal: `scripts/run_live_trader.py`

Este script coordina todos los módulos. La lógica principal no cambia mucho, pero la inicialización de `LiveBinanceAPIManager` ahora depende de los nombres de secretos y `LIVE_TRADING_MODE`.

* **Implementación (usando `asyncio`):**
    ```python
    # scripts/run_live_trader.py
    import asyncio
    import os
    import numpy as np
    import pandas as pd
    import time
    from google.cloud import aiplatform # Para llamar a Vertex AI

    from src.utils.config import ConfigManager
    from src.utils.logging_utils import setup_logger #
    from src.live.websocket_manager import LiveWebsocketManager 
    from src.live.binance_api_manager import LiveBinanceAPIManager
    from src.live.live_data_processor import LiveFeatureProcessor
    from src.live.portfolio_feature_builder import build_live_portfolio_features

    logger = setup_logger("LiveTrader")

    async def main_live_trader():
        logger.info("Iniciando Live Trader Bot...")
        try:
            # load_dotenv() es implícito si ConfigManager lo usa, o llamar explícitamente aquí.
            # from dotenv import load_dotenv
            # load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env')) # Ajustar ruta al .env
            
            config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env") # .env carga las variables con los NOMBRES de los secretos
        except Exception as e:
            logger.error(f"Error fatal al cargar ConfigManager: {e}", exc_info=True)
            return

        trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper() # Esta variable es leída por BinanceAPIManager
        logger.info(f"Modo de Trading: {trading_mode}")

        try:
            live_config = config_manager.get_config_value('live_trading')
            env_config = config_manager.get_environment_config()
            data_acq_config = config_manager.get_data_acquisition_defaults()
            preproc_config = config_manager.get_preprocessing_config()

            notification_queue = asyncio.Queue()
            # Los managers ahora deberían leer LIVE_TRADING_MODE internamente si es necesario para URLs
            # o el BinanceAPIManager para seleccionar las claves API correctas vía ConfigManager
            websocket_manager = LiveWebsocketManager(config_manager, notification_queue)
            binance_api_manager = LiveBinanceAPIManager(config_manager) # Este es el principal que usará LIVE_TRADING_MODE para las claves
            live_feature_processor = LiveFeatureProcessor(config_manager)

            project_id = os.getenv('GCP_PROJECT_ID') # Leído de .env
            region = os.getenv('GCP_REGION') # Leído de .env
            vertex_endpoint_id = live_config['vertex_ai_endpoint_raw_id']
            if not all([project_id, region, vertex_endpoint_id]):
                logger.error("Configuración de Vertex AI incompleta (GCP_PROJECT_ID, GCP_REGION, vertex_ai_endpoint_raw_id).")
                return
            aiplatform.init(project=project_id, location=region)
            vertex_endpoint = aiplatform.Endpoint(vertex_endpoint_id)
            logger.info(f"Cliente de Vertex AI Endpoint inicializado para: {vertex_endpoint.resource_name}")

            await binance_api_manager.set_leverage_if_needed(data_acq_config['symbol'], int(env_config['leverage']))
            logger.info(f"Apalancamiento para {data_acq_config['symbol']} verificado/establecido en {env_config['leverage']}x.")

        except Exception as e:
            logger.error(f"Error fatal durante la inicialización de managers o configuración de apalancamiento: {e}", exc_info=True)
            return

        last_step_equity = None # Para la feature 'Retorno Log Equity'

        asyncio.create_task(websocket_manager.run())
        logger.info("WebSocket Manager iniciado y escuchando.")

        while True:
            try:
                logger.info(f"Esperando notificación de nueva vela cerrada para {data_acq_config['symbol']}@{data_acq_config['interval']}...")
                closed_kline_message = await notification_queue.get() # Espera aquí
                # closed_kline_data es el objeto 'k' del mensaje del WebSocket
                closed_kline_data = closed_kline_message 
                
                logger.info(f"Nueva vela cerrada detectada (TS: {closed_kline_data.get('t')}, Px: {closed_kline_data.get('c')}). Procesando...")

                # 1. Obtener Historial de Velas
                raw_candles_df = await binance_api_manager.get_historical_klines(
                    symbol=data_acq_config['symbol'],
                    interval=data_acq_config['interval'],
                    lookback_candles=live_config['market_data_lookback_candles']
                )
                if raw_candles_df is None or raw_candles_df.empty or len(raw_candles_df) < preproc_config['sequence_length_L']:
                    logger.warning(f"No se obtuvieron suficientes datos históricos. ({len(raw_candles_df) if raw_candles_df is not None else 'None'}). Saltando ciclo.")
                    notification_queue.task_done()
                    continue
                
                # 2. Preprocesar Datos de Mercado
                market_features_array = live_feature_processor.process_market_data(raw_candles_df)
                if market_features_array is None:
                    logger.warning("Fallo el preprocesamiento de datos de mercado. Saltando ciclo.")
                    notification_queue.task_done()
                    continue
                
                # 3. Obtener Datos de Cartera de Binance
                account_balance_info = await binance_api_manager.get_account_balance()
                account_general_info = await binance_api_manager.get_account_info()
                position_info = await binance_api_manager.get_position_risk(symbol=data_acq_config['symbol'])

                if not account_general_info or not position_info or not account_balance_info:
                    logger.error("No se pudo obtener la información completa de la cuenta/posición/balance de Binance. Saltando ciclo.")
                    notification_queue.task_done()
                    continue
                
                current_equity = float(account_general_info['totalWalletBalance'])
                if last_step_equity is None: last_step_equity = current_equity 

                portfolio_features_array = build_live_portfolio_features(
                    account_general_info, account_balance_info, position_info, 
                    env_config, float(env_config['initial_equity']), last_step_equity
                )
                last_step_equity = current_equity

                # 4. Preparar y Enviar Observación al Modelo
                observation_instance = {
                    "market_features": market_features_array.tolist(),
                    "portfolio_features": portfolio_features_array.tolist()
                }
                
                logger.debug("Enviando observación al endpoint de Vertex AI...")
                prediction_response = vertex_endpoint.predict(instances=[observation_instance])
                action_value_from_model = float(prediction_response.predictions[0]['action_value']) # Ajustar si la salida de server.py es diferente
                logger.info(f"Respuesta del modelo: {action_value_from_model:.4f}")

                # 5. Interpretar Acción y Ejecutar
                action_threshold = env_config['action_threshold']
                desired_signal = 0 # 0 Neutral, 1 Long, -1 Short
                if action_value_from_model > action_threshold: desired_signal = 1
                elif action_value_from_model < -action_threshold: desired_signal = -1
                
                current_position_amt = float(position_info['positionAmt'])
                current_position_side = 0
                if current_position_amt > 0: current_position_side = 1
                elif current_position_amt < 0: current_position_side = -1

                logger.info(f"Posición actual en Binance: {current_position_side} (Amt: {current_position_amt}). Señal del modelo: {desired_signal}.")

                if desired_signal != current_position_side:
                    # Cerrar posición existente si es necesario
                    if current_position_side != 0:
                        logger.info(f"Decisión: Cerrar posición actual ({current_position_side}) de {abs(current_position_amt)} {data_acq_config['symbol']}.")
                        close_result = await binance_api_manager.close_market_position(data_acq_config['symbol'], current_position_amt)
                        if close_result: logger.info(f"Resultado cierre: {close_result.get('status')}")
                        await asyncio.sleep(live_config.get('post_close_delay_seconds', 1))
                    
                    # Abrir nueva posición si la señal no es neutral
                    if desired_signal != 0:
                        # Podrías querer re-consultar equity/precio aquí si el cierre fue grande
                        # Por simplicidad, usamos el equity antes del cierre para el tamaño de la nueva posición.
                        # El precio lo tomamos de la vela cerrada.
                        current_market_price_for_qty = float(closed_kline_data.get('c'))

                        order_qty = await binance_api_manager.calculate_order_quantity(
                            symbol=data_acq_config['symbol'],
                            equity=current_equity, 
                            current_price=current_market_price_for_qty,
                            leverage=float(env_config['leverage'])
                        )
                        if order_qty > 0:
                            logger.info(f"Decisión: Abrir nueva posición {desired_signal} de {order_qty} {data_acq_config['symbol']}.")
                            order_result = await binance_api_manager.place_market_order(
                                symbol=data_acq_config['symbol'],
                                side="BUY" if desired_signal == 1 else "SELL",
                                quantity=order_qty
                            )
                            if order_result: logger.info(f"Resultado apertura: {order_result.get('status')}")
                        else:
                            logger.warning(f"La cantidad calculada para la orden es 0 o inválida. No se abrirá posición {desired_signal}.")
                else:
                    logger.info("Decisión: Mantener posición/estado actual.")

                notification_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Live Trader cancelado.")
                break
            except aiplatform.errors.PredictionError as pe: # Captura específica para errores de Vertex
                logger.error(f"Error de predicción de Vertex AI: {pe.message} (Code: {pe.code})", exc_info=False) # No imprimir toda la traza si es error de API
                # Podrías añadir lógica para reintentar la predicción o manejar el error de forma específica
                await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds',30)) # Pausa antes del siguiente ciclo
            except Exception as e:
                logger.error(f"Error en el bucle principal del Live Trader: {e}", exc_info=True)
                await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds',60)) # Esperar antes de reintentar el ciclo general

    if __name__ == "__main__":
        try:
            asyncio.run(main_live_trader())
        except KeyboardInterrupt:
            logger.info("Live Trader detenido manualmente por el usuario.")
        except Exception as e_global:
            logger.critical(f"Error global irrecuperable en Live Trader: {e_global}", exc_info=True)

    ```

---

## 5. Consideraciones Adicionales y Mejores Prácticas

(Igual que en la guía anterior: Gestión de Errores Robusta, Persistencia y Auditoría, Sincronización y Timestamps, Límites de Tasa, Seguridad de Claves, Pruebas Exhaustivas en Testnet, Monitoreo y Alertas, Kill Switch).

---

## 6. Despliegue

(Igual que en la guía anterior: VM en GCP, Cuenta de Servicio, Instalación de software, Configuración de `.env`, Ejecución persistente con `supervisor` o `systemd`).

Este enfoque unificado para la gestión de secretos es más seguro y una mejor práctica. Asegúrate de que los nombres de los secretos en tu archivo `.env` coincidan exactamente con los nombres de los secretos que creas en Google Cloud Secret Manager, y que tu `ConfigManager` esté adaptado para usar estos nombres de manera dinámica según `LIVE_TRADING_MODE`.