#!/usr/bin/env python3
# scripts/run_live_trader.py
import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
import json
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple
import requests
import signal

# Asegurar que src esté en el PYTHONPATH para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.live.websocket_manager import LiveWebsocketManager
from src.live.binance_api_manager import LiveBinanceAPIManager
from src.live.live_data_processor import LiveFeatureProcessor
from src.live.portfolio_feature_builder import PortfolioFeatureBuilder

# Configurar logger principal
logger = setup_logger("LiveTrader")

class LiveTrader:
    """
    Orquestador principal para el trading en vivo.
    
    Responsabilidades:
    1. Iniciar y mantener una conexión WebSocket para detectar nuevas velas
    2. Procesar datos OHLCV históricos y en vivo usando el Feature Engineering existente
    3. Preparar datos de características de mercado y cartera para predicción
    4. Llamar al endpoint de predicción en Vertex AI usando los datos procesados
    5. Ejecutar decisiones de trading basadas en las predicciones
    6. Manejar la lógica de posiciones y la gestión de errores
    """
    def __init__(self):
        self.config_manager = ConfigManager()
        self.notification_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        
        # Cargar configuraciones
        self.env_config = self.config_manager.get_environment_config()
        self.data_acq_config = self.config_manager.get_data_acquisition_defaults()
        self.live_trading_config = self.config_manager.get_config_value('live_trading', {})
        
        # Datos básicos de trading
        self.symbol = self.data_acq_config.get('symbol', "BTCUSDT")
        self.interval = self.data_acq_config.get('interval', "1h")
        self.leverage = float(self.env_config.get('leverage', 10.0))
        self.action_threshold = float(self.env_config.get('action_threshold', 0.15))
        
        # Obtener lookback para datos de mercado
        self.lookback_candles = self.live_trading_config.get('market_data_lookback_candles', 250)
        
        # Configurar endpoint de Vertex AI
        self.vertex_ai_predict_url = self.live_trading_config.get('vertex_ai_predict_url', "")
        if not self.vertex_ai_predict_url or self.vertex_ai_predict_url == "REEMPLAZAR_CON_URL_ENDPOINT_VERTEX_AI":
            logger.error("URL de endpoint de Vertex AI no configurada correctamente en config.yaml")
            raise ValueError("URL de endpoint de Vertex AI no configurada. Actualice 'vertex_ai_predict_url' en config.yaml")
        
        # Inicialización de gestores
        self.websocket_manager = LiveWebsocketManager(self.config_manager, self.notification_queue)
        self.api_manager = LiveBinanceAPIManager(self.config_manager)
        self.feature_processor = LiveFeatureProcessor(self.config_manager)
        self.portfolio_builder = PortfolioFeatureBuilder(self.config_manager)
        
        # Variable para almacenar el registro de trading para el CSV
        self.trading_log = []
        self.last_log_upload_time = time.time()
        self.log_buffer_size = self.live_trading_config.get('log_buffer_size_records', 50)
        self.log_upload_interval_secs = self.live_trading_config.get('gcs_log_upload_interval_seconds', 3600)
        
        # Delay después de cerrar una posición
        self.post_close_delay = self.live_trading_config.get('post_close_delay_seconds', 1)
        
        logger.info(f"LiveTrader inicializado para {self.symbol}@{self.interval}, "
                   f"threshold={self.action_threshold}, leverage={self.leverage}x")

    async def start(self):
        """Inicia el orquestador de trading en vivo."""
        try:
            # Registrar manejador de señal para finalización ordenada
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(self.shutdown()))
            
            # Inicializar cliente de API
            await self.api_manager.initialize_client()
            
            # Establecer apalancamiento, solo debería hacer falta una vez
            leverage_set = await self.api_manager.set_leverage_if_needed(
                self.symbol, int(self.leverage))
            if not leverage_set:
                logger.warning(f"No se pudo establecer apalancamiento a {int(self.leverage)}x. "
                              f"Verifica permisos de la API o estado de la cuenta.")
            
            # Crear y ejecutar tareas
            websocket_task = asyncio.create_task(
                self.websocket_manager.run(), name="websocket_manager")
            trader_task = asyncio.create_task(
                self.trading_loop(), name="trading_loop")
            
            # Esperar a que alguna de las tareas termine, o hasta que se solicite shutdown
            await asyncio.wait(
                [websocket_task, trader_task, self.shutdown_event.wait()],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Si llegamos aquí, algo ha terminado, iniciar shutdown ordenado
            if not self.shutdown_event.is_set():
                await self.shutdown()
                
        except Exception as e:
            logger.error(f"Error inesperado en start: {e}", exc_info=True)
            await self.shutdown()
        finally:
            # Asegurar que cerramos todo correctamente
            await self.api_manager.close_client_session()

    async def shutdown(self):
        """Realiza una finalización ordenada del trader."""
        if self.shutdown_event.is_set():
            return  # Evitar múltiples shutdowns
        
        logger.info("Iniciando finalización ordenada del LiveTrader...")
        self.shutdown_event.set()
        
        # Forzar una última subida de logs pendientes
        await self.upload_trading_logs()
        
        logger.info("LiveTrader finalizado correctamente.")

    async def trading_loop(self):
        """Loop principal de trading que reacciona a notificaciones de nuevas velas."""
        logger.info("Iniciando bucle principal de trading...")
        
        # Esperar notificaciones del WebSocket sobre cierre de velas
        while not self.shutdown_event.is_set():
            try:
                # Esperar a la siguiente vela cerrada (con timeout)
                try:
                    kline_data = await asyncio.wait_for(
                        self.notification_queue.get(), 
                        timeout=60  # 60s timeout para hacer comprobaciones periódicas
                    )
                    logger.info(f"Nueva vela cerrada detectada: {kline_data.get('t')}")
                    
                    # Procesar la nueva vela y tomar decisiones de trading
                    await self.process_new_candle(kline_data)
                    
                    # Verificar si es momento de subir los logs de trading
                    elapsed_since_last_upload = time.time() - self.last_log_upload_time
                    log_buffer_full = len(self.trading_log) >= self.log_buffer_size
                    
                    if log_buffer_full or elapsed_since_last_upload >= self.log_upload_interval_secs:
                        await self.upload_trading_logs()
                    
                except asyncio.TimeoutError:
                    # No hay problema, solo estamos haciendo una comprobación periódica
                    continue
                
            except asyncio.CancelledError:
                logger.info("Trading loop cancelado durante espera de notificación.")
                break
                
            except Exception as e:
                logger.error(f"Error en bucle principal de trading: {e}", exc_info=True)
                # Esperar un poco antes de reintentar para evitar bucles rápidos en caso de error
                await asyncio.sleep(5)

    async def process_new_candle(self, kline_data: Dict[str, Any]):
        """
        Procesa una nueva vela, obtiene datos históricos, calcula características,
        obtiene predicción y ejecuta decisiones de trading.
        
        Args:
            kline_data: Datos de la vela cerrada desde el WebSocket
        """
        try:
            # 1. Obtener datos históricos de mercado
            klines_df = await self.api_manager.get_historical_klines(
                self.symbol, self.interval, self.lookback_candles
            )
            
            if klines_df is None or klines_df.empty:
                logger.error("No se pudieron obtener klines históricas. Saltando ciclo de decisión.")
                return
                
            # 2. Procesar datos para extraer características de mercado
            market_features_df = self.feature_processor.process_klines_data(klines_df)
            if market_features_df.empty:
                logger.error("Error procesando características de mercado. Saltando ciclo de decisión.")
                return
                
            # 3. Obtener datos de cuenta y posición
            account_info = await self.api_manager.get_account_info()
            position_info = await self.api_manager.get_position_risk(self.symbol)
            
            if not account_info or not position_info:
                logger.error("No se pudo obtener información de cuenta o posición. Saltando ciclo de decisión.")
                return
                
            # 4. Construir características de cartera
            current_price = float(klines_df.iloc[-1]['Close'])
            portfolio_features = self.portfolio_builder.build_portfolio_features(
                position_info, account_info, current_price
            )
            
            # 5. Extraer la secuencia de características de mercado
            market_features_sequence = self.feature_processor.get_latest_feature_sequence(market_features_df)
            if market_features_sequence is None:
                logger.error("No se pudo extraer secuencia de características de mercado. Saltando ciclo de decisión.")
                return
                
            # 6. Llamar al endpoint para obtener predicción
            action_value = await self.get_model_prediction(market_features_sequence, portfolio_features)
            if action_value is None:
                logger.error("No se pudo obtener predicción del modelo. Saltando ciclo de decisión.")
                return
                
            # 7. Ejecutar decisión de trading basada en la predicción
            await self.execute_trading_decision(
                action_value, position_info, account_info, current_price, klines_df
            )
            
        except Exception as e:
            logger.error(f"Error en process_new_candle: {e}", exc_info=True)

    async def get_model_prediction(self, 
                                market_features: np.ndarray, 
                                portfolio_features: np.ndarray) -> Optional[float]:
        """
        Obtiene predicción del modelo desde Vertex AI.
        
        Args:
            market_features: Características de mercado para el modelo
            portfolio_features: Características de cartera para el modelo
            
        Returns:
            Valor de acción predicho por el modelo o None si hay error
        """
        try:
            # Validar dimensiones de características
            if market_features.shape[0] != portfolio_features.shape[0]:
                raise ValueError(f"Dimensiones incompatibles: market {market_features.shape}, portfolio {portfolio_features.shape}")
            
            # Convertir a listas para el formato JSON
            market_features_list = market_features.tolist()
            portfolio_features_list = portfolio_features.tolist()
            
            # Preparar payload para el endpoint en formato JSON
            payload = {
                "instances": [
                    {
                        "market_features": market_features_list,
                        "portfolio_features": portfolio_features_list
                    }
                ]
            }
            
            # Comprobar tamaño del payload (logging)
            payload_size_kb = sys.getsizeof(json.dumps(payload)) / 1024
            logger.debug(f"Tamaño del payload: {payload_size_kb:.2f} KB")
            
            # Configurar encabezados para la API de Vertex AI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}"
            }
            
            # Realizar la solicitud HTTP
            logger.debug(f"Enviando solicitud a endpoint: {self.vertex_ai_predict_url}")
            start_time = time.time()
            
            # Usar requests en lugar de aiohttp para simplicidad con credenciales ADC
            # En un entorno de producción real, podría ser mejor usar aiohttp con autenticación adecuada
            response = requests.post(
                self.vertex_ai_predict_url,
                json=payload,
                headers=headers
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Respuesta recibida en {elapsed_time:.2f}s, status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Error llamando al endpoint de Vertex AI: HTTP {response.status_code}, {response.text}")
                return None
            
            # Parsear la respuesta y extraer el valor de acción
            response_data = response.json()
            if 'predictions' not in response_data or not response_data['predictions']:
                logger.error(f"Formato de respuesta inválido: {response_data}")
                return None
                
            # La estructura esperada es ['predictions'][0][0]
            # El primer [0] es para tomar la primera predicción del batch
            # El segundo [0] es porque el modelo devuelve un array con un solo valor
            action_value = float(response_data['predictions'][0][0])
            logger.info(f"Predicción recibida: {action_value:.4f}")
            
            return action_value
            
        except Exception as e:
            logger.error(f"Error obteniendo predicción: {e}", exc_info=True)
            return None

    async def execute_trading_decision(self, 
                                     action_value: float,
                                     position_info: Dict[str, Any],
                                     account_info: Dict[str, Any],
                                     current_price: float,
                                     klines_df: pd.DataFrame):
        """
        Ejecuta decisiones de trading basadas en la predicción del modelo.
        
        Args:
            action_value: Valor continuo entre -1 y 1 predicho por el modelo
            position_info: Información actual de la posición
            account_info: Información de la cuenta
            current_price: Precio actual
            klines_df: DataFrame con datos históricos de velas
        """
        try:
            # 1. Extraer datos actuales de posición y cuenta
            position_amount = float(position_info.get('positionAmt', '0'))
            unrealized_pnl = float(position_info.get('unRealizedProfit', '0'))
            entry_price = float(position_info.get('entryPrice', '0'))
            
            total_margin_balance = float(account_info.get('totalMarginBalance', '0'))
            
            # 2. Determinar la posición actual
            current_position = 0  # Neutral (sin posición)
            if abs(position_amount) > 1e-8:  # Si hay posición significativa
                current_position = 1 if position_amount > 0 else -1  # 1: Long, -1: Short
            
            # 3. Determinar la acción basada en el valor de acción y el umbral
            # Discretizar el valor continuo usando el umbral
            target_position = 0  # Por defecto neutral
            if action_value >= self.action_threshold:
                target_position = 1  # Long
            elif action_value <= -self.action_threshold:
                target_position = -1  # Short
                
            # 4. Registrar la situación actual para el log
            self.record_trading_info(
                action_value, current_position, target_position, 
                position_amount, current_price, entry_price, 
                unrealized_pnl, total_margin_balance, klines_df
            )
            
            # 5. Ejecutar órdenes según corresponda
            if current_position == target_position:
                # No hay cambio en la posición, mantener
                logger.info(f"Manteniendo posición actual: {current_position} (acción: {action_value:.4f})")
                return
            
            # Si llegamos aquí, hay un cambio de posición
            
            # Caso 1: Tenemos posición actual y debemos cerrarla
            if current_position != 0:
                logger.info(f"Cerrando posición actual {current_position} (cantidad: {position_amount})")
                close_order = await self.api_manager.close_market_position(
                    self.symbol, position_amount
                )
                
                if not close_order or (isinstance(close_order, dict) and close_order.get('status') != "NO_POSITION_TO_CLOSE"):
                    logger.info(f"Posición cerrada exitosamente: {close_order}")
                    
                    # Pequeño delay después de cerrar para que la cuenta se actualice
                    if self.post_close_delay > 0:
                        await asyncio.sleep(self.post_close_delay)
                else:
                    logger.warning(f"No se pudo cerrar la posición o no había posición para cerrar.")
            
            # Caso 2: Debemos abrir una nueva posición (si target no es neutral)
            if target_position != 0:
                # Refrescar datos de cuenta después de cerrar posición anterior
                account_info = await self.api_manager.get_account_info()
                if not account_info:
                    logger.error("No se pudo obtener info actualizada de cuenta. No se abrirá nueva posición.")
                    return
                    
                # Calcular cantidad para la orden según el equity disponible
                equity = float(account_info.get('totalMarginBalance', '0'))
                order_qty = await self.api_manager.calculate_order_quantity(
                    self.symbol, equity, current_price, int(self.leverage)
                )
                
                if order_qty <= 0:
                    logger.warning(f"Cantidad calculada para orden ({order_qty}) es inválida. No se abrirá posición.")
                    return
                
                # Determinar lado de la orden
                order_side = "BUY" if target_position > 0 else "SELL"
                
                # Abrir la nueva posición
                logger.info(f"Abriendo nueva posición {order_side} con cantidad {order_qty}")
                open_order = await self.api_manager.place_market_order(
                    self.symbol, order_side, order_qty
                )
                
                if open_order:
                    logger.info(f"Nueva posición {order_side} abierta exitosamente: {open_order}")
                else:
                    logger.error(f"Error al abrir nueva posición {order_side}.")
            
        except Exception as e:
            logger.error(f"Error ejecutando decisión de trading: {e}", exc_info=True)

    def record_trading_info(self, 
                          action_value: float,
                          current_position: int,
                          target_position: int,
                          position_amount: float,
                          current_price: float,
                          entry_price: float,
                          unrealized_pnl: float,
                          total_margin_balance: float,
                          klines_df: pd.DataFrame):
        """
        Registra información de trading para análisis y logging.
        
        Args:
            action_value: Valor predicho por el modelo
            current_position: Posición actual (1, 0, -1)
            target_position: Posición objetivo (1, 0, -1)
            position_amount: Cantidad de la posición actual
            current_price: Precio actual
            entry_price: Precio de entrada
            unrealized_pnl: PnL no realizado actual
            total_margin_balance: Balance total de margen
            klines_df: DataFrame con datos de velas
        """
        try:
            # Obtener datos de la última vela
            last_candle = klines_df.iloc[-1]
            timestamp = last_candle.name.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calcular información adicional
            position_change = target_position != current_position
            pnl_pct = 0.0
            if abs(position_amount) > 1e-8 and entry_price > 0:
                if current_position > 0:  # Long
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:  # Short
                    pnl_pct = (entry_price - current_price) / entry_price * 100
            
            # Crear registro para el log
            log_entry = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'interval': self.interval,
                'open': float(last_candle['Open']),
                'high': float(last_candle['High']),
                'low': float(last_candle['Low']),
                'close': float(last_candle['Close']),
                'volume': float(last_candle['Volume']),
                'action_value': float(action_value),
                'threshold': float(self.action_threshold),
                'current_position': int(current_position),
                'target_position': int(target_position),
                'position_change': bool(position_change),
                'position_amount': float(position_amount),
                'entry_price': float(entry_price) if entry_price > 0 else None,
                'current_price': float(current_price),
                'unrealized_pnl': float(unrealized_pnl),
                'pnl_percent': float(pnl_pct),
                'margin_balance': float(total_margin_balance),
                'leverage': float(self.leverage)
            }
            
            # Añadir al buffer de logs
            self.trading_log.append(log_entry)
            
            logger.info(f"Estado de trading registrado: {timestamp}, "
                       f"pos={current_position}->{target_position}, "
                       f"acción={action_value:.4f}, pnl={unrealized_pnl:.2f}USDT, "
                       f"equity={total_margin_balance:.2f}USDT")
            
        except Exception as e:
            logger.error(f"Error registrando información de trading: {e}", exc_info=True)

    async def upload_trading_logs(self):
        """Sube registros de trading acumulados a Google Cloud Storage."""
        if not self.trading_log:
            logger.debug("No hay logs de trading para subir.")
            return
        
        try:
            # Crear DataFrame con los logs acumulados
            logs_df = pd.DataFrame(self.trading_log)
            
            # Obtener fecha actual para nombre del archivo
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Construir ruta del archivo de log según el template
            log_path_template = self.live_trading_config.get(
                'gcs_log_path_template', 
                "live_trading_logs/{symbol}_{interval}/{date}.csv"
            )
            
            log_path = log_path_template.format(
                symbol=self.symbol.lower(),
                interval=self.interval,
                date=current_date
            )
            
            # Importar aquí para evitar dependencias circulares
            from src.utils.gcs_utils import upload_dataframe_to_gcs
            
            # Subir a GCS
            success = upload_dataframe_to_gcs(
                logs_df,
                self.config_manager.get_env_variable('GCS_BUCKET_NAME'),
                log_path,
                if_exists='append'  # Importante: append para acumular logs del día
            )
            
            if success:
                logger.info(f"Logs de trading ({len(self.trading_log)} registros) subidos a gs://{self.config_manager.get_env_variable('GCS_BUCKET_NAME')}/{log_path}")
                # Limpiar buffer de logs
                self.trading_log = []
                self.last_log_upload_time = time.time()
            else:
                logger.error(f"Error al subir logs de trading a GCS.")
                
        except Exception as e:
            logger.error(f"Error en upload_trading_logs: {e}", exc_info=True)

async def main():
    """Función principal para iniciar el LiveTrader."""
    logger.info("Iniciando BTCBot en modo trading en vivo...")
    trader = LiveTrader()
    await trader.start()

if __name__ == "__main__":
    try:
        # Ejecutar el bucle de eventos para correr el LiveTrader
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Programa interrumpido por usuario.")
    except Exception as e:
        logger.error(f"Error crítico en main: {e}", exc_info=True)
    finally:
        logger.info("BTCBot en modo live trading finalizado.")
