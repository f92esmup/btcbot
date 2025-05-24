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
import argparse
from google.cloud import bigquery  # Add BigQuery import

# Asegurar que src esté en el PYTHONPATH para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.live.websocket_manager import LiveWebsocketManager
from src.agent.rl_agent_manager import RLAgentManager
from src.live.binance_api_manager import LiveBinanceAPIManager
from src.live.live_data_processor import LiveFeatureProcessor
from src.live.portfolio_feature_builder import PortfolioFeatureBuilder

# Configurar logger principal
logger = setup_logger("LiveTrader")

def parse_arguments():
    parser = argparse.ArgumentParser(description="BTCBot Live Trader")
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/config.yaml",
        help="Ruta al archivo de configuración centralizada"
    )
    return parser.parse_args()

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
    def __init__(self, args=None):
        self.config_manager = ConfigManager(args.config_path if args else None)
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
        
        # Inicialización de gestores
        self.websocket_manager = LiveWebsocketManager(self.config_manager, self.notification_queue)
        self.api_manager = LiveBinanceAPIManager(self.config_manager)
        self.feature_processor = LiveFeatureProcessor(self.config_manager)
        self.portfolio_builder = PortfolioFeatureBuilder(self.config_manager)

        # Inicializar RLAgentManager y cargar el modelo
        try:
            self.agent_manager = RLAgentManager(config_path=args.config_path if args else None)
            # self.agent_manager.setup_environment() # No es necesario si load_model no lo requiere explícitamente
            agent_config = self.config_manager.get_agent_config()
            model_path = agent_config.get('live_model_path')
            if not model_path:
                logger.error("Ruta del modelo en vivo no configurada en agent_config.live_model_path")
                raise ValueError("Ruta del modelo en vivo no configurada.")
            self.agent_manager.load_model(model_path)
            logger.info(f"Modelo RL cargado exitosamente desde: {model_path}")
        except Exception as e_agent:
            logger.error(f"Error inicializando RLAgentManager o cargando modelo: {e_agent}", exc_info=True)
            raise  # Re-lanzar la excepción para detener la inicialización si falla la carga del modelo
        
        # NUEVO: Inicializar cliente de BigQuery y configuración
        try:
            self.gcp_project_id = self.config_manager.get_env_variable('GCP_PROJECT_ID')
            self.bigquery_client = bigquery.Client(project=self.gcp_project_id)
            # Define los IDs del dataset y tabla usando configuración o valores por defecto
            self.bigquery_dataset_id = self.live_trading_config.get('bigquery_dataset_id', 'btcbot_logs')
            self.bigquery_table_name = self.live_trading_config.get('bigquery_table_name', 'live_trading_events')
            self.bigquery_table_id = f"{self.gcp_project_id}.{self.bigquery_dataset_id}.{self.bigquery_table_name}"
            logger.info(f"Logging de trading configurado para BigQuery table: {self.bigquery_table_id}")
            
            # Determinar el modo de trading (TESTNET/REAL) para los logs
            self.trading_mode = "TESTNET" if self.config_manager.get_env_variable('USE_TESTNET', 'true').lower() == 'true' else "REAL"
            logger.info(f"Modo de trading configurado: {self.trading_mode}")
        except Exception as e_bq_init:
            logger.error(f"Error inicializando cliente de BigQuery o IDs: {e_bq_init}. El logging a BigQuery no funcionará.")
            self.bigquery_client = None
            self.bigquery_table_id = None
        
        # Variable para almacenar el registro de trading para el CSV (ya no se usará activamente)
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
                    sig, lambda s=sig: asyncio.create_task(
                        self.handle_signal(s)))
            
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
            # Almacenar la tarea en el gestor de websocket para poder cerrarla después
            self.websocket_manager.connection_task = websocket_task
            
            trader_task = asyncio.create_task(
                self.trading_loop(), name="trading_loop")
            shutdown_task = asyncio.create_task(
                self.shutdown_event.wait(), name="shutdown_waiter")
            
            # Esperar a que alguna de las tareas termine, o hasta que se solicite shutdown
            await asyncio.wait(
                [websocket_task, trader_task, shutdown_task],
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
        
        # Asegurarse de que el websocket se cierre correctamente
        if hasattr(self, 'websocket_manager'):
            await self.websocket_manager.close()
            
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
        # Diccionario para el log de BigQuery
        log_entry_data_for_bq = {}
        
        try:
            # 1. Obtener datos históricos de mercado
            klines_df = await self.api_manager.get_historical_klines(
                self.symbol, self.interval, self.lookback_candles
            )
            
            if klines_df is None or klines_df.empty:
                logger.error("No se pudieron obtener klines históricas. Saltando ciclo de decisión.")
                log_entry_data_for_bq["error_message_bq"] = "No se pudieron obtener klines históricas"
                return
                
            # 2. Procesar datos para extraer características de mercado
            market_features_df = self.feature_processor.process_klines_data(klines_df)
            if market_features_df.empty:
                logger.error("Error procesando características de mercado. Saltando ciclo de decisión.")
                log_entry_data_for_bq["error_message_bq"] = "Error procesando características de mercado"
                return
                
            # 3. Obtener datos de cuenta y posición (guardar para el log)
            account_info_start_cycle = await self.api_manager.get_account_info()
            position_info_live_start_cycle = await self.api_manager.get_position_risk(self.symbol)
            
            account_info = account_info_start_cycle
            position_info = position_info_live_start_cycle
            
            if not account_info or not position_info:
                logger.error("No se pudo obtener información de cuenta o posición. Saltando ciclo de decisión.")
                log_entry_data_for_bq["error_message_bq"] = "No se pudo obtener información de cuenta o posición"
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
                log_entry_data_for_bq["error_message_bq"] = "No se pudo extraer secuencia de características de mercado"
                return
                
            # 6. Llamar al endpoint para obtener predicción
            action_value = await self.get_model_prediction(market_features_sequence, portfolio_features)
            if action_value is None:
                logger.error("No se pudo obtener predicción del modelo. Saltando ciclo de decisión.")
                log_entry_data_for_bq["error_message_bq"] = "No se pudo obtener predicción del modelo"
                return
                
            # Determinar la posición actual
            current_position = 0  # Neutral (sin posición)
            position_amount = float(position_info.get('positionAmt', '0'))
            if abs(position_amount) > 1e-8:  # Si hay posición significativa
                current_position = 1 if position_amount > 0 else -1  # 1: Long, -1: Short
                
            # Determinar la acción basada en el valor de acción y el umbral
            target_position = 0  # Por defecto neutral
            if action_value >= self.action_threshold:
                target_position = 1  # Long
            elif action_value <= -self.action_threshold:
                target_position = -1  # Short
            
            # Registrar variables para el log de BigQuery
            cycle_vars = {
                'kline_data': kline_data,
                'action_value': action_value,
                'position_info': position_info,
                'account_info': account_info,
                'current_price': current_price,
                'current_position': current_position,
                'target_position': target_position,
                'position_info_live_start_cycle': position_info_live_start_cycle,
                'account_info_start_cycle': account_info_start_cycle
            }
                
            # 7. Ejecutar decisión de trading basada en la predicción
            await self.execute_trading_decision(
                action_value, position_info, account_info, current_price, klines_df
            )
            
            # Obtener el equity final para el log
            final_account_info = await self.api_manager.get_account_info()
            if final_account_info:
                last_equity = float(final_account_info.get('totalWalletBalance', 0.0))
                cycle_vars['current_equity_after_action_bq'] = last_equity
            
        except Exception as e:
            logger.error(f"Error en process_new_candle: {e}", exc_info=True)
            log_entry_data_for_bq["error_message_bq"] = f"Error en process_new_candle: {str(e)[:200]}"
        
        finally:
            # Siempre intentar enviar logs a BigQuery, incluso si hubo errores
            if self.bigquery_client and self.bigquery_table_id:
                try:
                    # Preparar el log completo para BigQuery
                    self._prepare_log_entry_for_bq(log_entry_data_for_bq, cycle_vars if 'cycle_vars' in locals() else {})
                    
                    # Importar aquí para evitar dependencias circulares
                    from src.utils.gcs_utils import stream_row_to_bigquery
                    
                    # Enviar a BigQuery usando asyncio.to_thread para no bloquear
                    await asyncio.to_thread(
                        stream_row_to_bigquery,
                        self.bigquery_client,
                        self.bigquery_table_id,
                        log_entry_data_for_bq
                    )
                    
                    logger.debug("Log enviado a BigQuery correctamente")
                except Exception as e_bq:
                    logger.error(f"Error enviando log a BigQuery: {e_bq}", exc_info=True)

    async def get_model_prediction(self, 
                                market_features: np.ndarray, 
                                portfolio_features: np.ndarray) -> Optional[float]:
        """
        Obtiene predicción del modelo desde Vertex AI o servidor local.
        
        Args:
            market_features: Características de mercado para el modelo
            portfolio_features: Características de cartera para el modelo
            
        Returns:
            Valor de acción predicho por el modelo o None si hay error
        """
        try:
            # Validar dimensiones de características
            # market_features debe ser 2D (L, num_market_features)
            # portfolio_features debe ser 1D (num_portfolio_features)
            # Estas validaciones se mantienen ya que son buenas prácticas.
            # Ajusta los números (15, 8) si las características cambian con el nuevo modelo.
            expected_market_shape_part = (self.feature_processor.sequence_length_L,) # Solo L, ya que num_market_features puede variar
            expected_portfolio_shape_part = () # No hay una forma fija para portfolio_features en este punto.

            logger.debug(f"Forma recibida de market_features: {market_features.shape}, esperada L={expected_market_shape_part[0]}")
            logger.debug(f"Forma recibida de portfolio_features: {portfolio_features.shape}")

            if market_features.ndim != 2 or market_features.shape[0] != self.feature_processor.sequence_length_L :
                raise ValueError(f"Market features debe ser 2D (L, num_market_features) pero tiene dimensiones: {market_features.shape}")
            if portfolio_features.ndim != 1:
                raise ValueError(f"Portfolio features debe ser 1D pero tiene dimensiones: {portfolio_features.shape}")

            # Asegurar que los datos son np.float32
            market_features_f32 = market_features.astype(np.float32)
            portfolio_features_f32 = portfolio_features.astype(np.float32)
            
            logger.debug(f"Valores de portfolio_features (float32): {portfolio_features_f32}")

            # Preparar observación para el agente
            observation = {
                "market_features": market_features_f32,
                "portfolio_features": portfolio_features_f32
            }
            
            logger.debug("Obteniendo predicción del RLAgentManager...")
            start_time = time.time()
            
            # Llamar al método predict_action del RLAgentManager
            # El método predict_action debe devolver un valor de acción (posiblemente en un array)
            action_output = self.agent_manager.predict_action(observation)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Predicción recibida del agente en {elapsed_time:.2f}s")

            # Extraer el valor de acción. Asumiendo que es un array y necesitamos el primer elemento.
            # Ajustar según lo que devuelva exactamente predict_action.
            if isinstance(action_output, np.ndarray):
                if action_output.size == 1:
                    action_value = float(action_output.item()) # .item() para extraer el escalar de un array de un solo elemento
                else:
                    # Si devuelve múltiples valores, tomar el primero o ajustar la lógica.
                    # Por ahora, se asume que el valor de acción deseado es el primero.
                    action_value = float(action_output[0])
                    logger.warning(f"predict_action devolvió un array con múltiples elementos: {action_output}. Usando el primero.")
            elif isinstance(action_output, (float, int)):
                action_value = float(action_output)
            else:
                logger.error(f"Tipo de salida de predict_action no esperado: {type(action_output)}, valor: {action_output}")
                return None
                
            logger.info(f"Predicción recibida (RLAgentManager): {action_value:.4f}")
            
            return action_value
            
        except Exception as e:
            logger.error(f"Error obteniendo predicción del RLAgentManager: {e}", exc_info=True)
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
        # Variables para el ciclo que se usarán en logs de BigQuery
        close_order = None
        open_order = None
        order_qty = 0.0
        
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
                
            # 4. Registrar la situación actual para el log (mantener para retrocompatibilidad)
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
            
            # Actualizar variables del ciclo en process_new_candle
            # (Buscamos cycle_vars en el stack frame de process_new_candle)
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'cycle_vars' in frame.f_locals:
                    # Encontramos cycle_vars, actualizar con la info de órdenes
                    frame.f_locals['cycle_vars'].update({
                        'close_order': close_order,
                        'open_order': open_order,
                        'order_qty': order_qty
                    })
                    break
                frame = frame.f_back
            
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
        """
        Método legacy para subir registros a Google Cloud Storage.
        Se mantiene por compatibilidad pero ahora los logs se envían directamente
        a BigQuery en cada ciclo.
        """
        if not self.trading_log:
            logger.debug("No hay logs legacy de trading para subir.")
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
                logger.info(f"Logs legacy de trading ({len(self.trading_log)} registros) subidos a gs://{self.config_manager.get_env_variable('GCS_BUCKET_NAME')}/{log_path}")
                # Limpiar buffer de logs
                self.trading_log = []
                self.last_log_upload_time = time.time()
            else:
                logger.error(f"Error al subir logs legacy de trading a GCS.")
                
        except Exception as e:
            logger.error(f"Error en upload_trading_logs: {e}", exc_info=True)

    async def handle_signal(self, sig):
        """Maneja señales del sistema como SIGINT y SIGTERM."""
        logger.info(f"Recibida señal {sig.name}. Iniciando apagado ordenado...")
        await self.shutdown()
        
    def _prepare_log_entry_for_bq(self, log_entry: Dict[str, Any], cycle_vars: Dict[str, Any]):
        """
        Ayudante para poblar el diccionario de log con variables del ciclo actual para BigQuery.
        
        Args:
            log_entry: Diccionario donde se almacenarán los datos para BigQuery
            cycle_vars: Diccionario con las variables del ciclo actual
        """
        # Datos básicos y timestamp
        log_entry["timestamp_decision_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_entry["trading_mode"] = self.trading_mode
        log_entry["symbol"] = self.symbol
        log_entry["interval"] = self.interval
        
        # Datos de la vela
        kline_data = cycle_vars.get('kline_data', {})
        if kline_data:
            log_entry["kline_open_time_utc"] = pd.to_datetime(kline_data.get('t'), unit='ms', utc=True).isoformat() if kline_data.get('t') else None
            log_entry["kline_close_time_utc"] = pd.to_datetime(kline_data.get('T'), unit='ms', utc=True).isoformat() if kline_data.get('T') else None
            log_entry["kline_o"] = float(kline_data.get('o', 0.0))
            log_entry["kline_h"] = float(kline_data.get('h', 0.0))
            log_entry["kline_l"] = float(kline_data.get('l', 0.0))
            log_entry["kline_c"] = float(kline_data.get('c', 0.0))
            log_entry["kline_v"] = float(kline_data.get('v', 0.0))
        
        # Datos del modelo y acción
        log_entry["model_action_value"] = float(cycle_vars.get('action_value', 0.0))
        log_entry["action_threshold"] = float(self.action_threshold)
        
        # Datos de posición y cuenta
        position_info = cycle_vars.get('position_info', {})
        account_info = cycle_vars.get('account_info', {})
        
        # Procesar posición actual
        pos_amt = float(position_info.get('positionAmt', '0.0'))
        current_pos_side = 0
        if pos_amt > 1e-8:
            current_pos_side = 1  # Long
        elif pos_amt < -1e-8:
            current_pos_side = -1  # Short
            
        log_entry["current_position_side_bq"] = current_pos_side
        log_entry["pos_amt_before_action_bq"] = pos_amt
        log_entry["entry_price_bq"] = float(position_info.get('entryPrice', '0.0'))
        log_entry["unrealized_pnl_bq"] = float(position_info.get('unRealizedProfit', '0.0'))
        
        # Equity y PnL
        current_price = cycle_vars.get('current_price', 0.0)
        initial_equity = float(self.env_config.get('initial_equity', 10000.0))
        current_equity = float(account_info.get('totalWalletBalance', initial_equity))
        log_entry["current_equity_before_action_bq"] = current_equity
        
        # Calcular PNL porcentual
        entry_price = float(position_info.get('entryPrice', '0.0'))
        pnl_pct = 0.0
        if abs(pos_amt) > 1e-8 and entry_price > 0 and current_price > 0:
            if current_pos_side > 0:  # Long
                pnl_pct = (current_price - entry_price) / entry_price * 100
            elif current_pos_side < 0:  # Short
                pnl_pct = (entry_price - current_price) / entry_price * 100
        log_entry["pnl_percent_bq"] = pnl_pct
        
        # Datos de la señal y posición deseada
        target_position = cycle_vars.get('target_position', 0)
        log_entry["desired_signal_bq"] = int(target_position)
        log_entry["position_change_bq"] = bool(current_pos_side != target_position)
        log_entry["current_price_bq"] = float(current_price)
        
        # Datos de configuración
        log_entry["leverage_bq"] = float(self.leverage)
        
        # Descripción de la acción tomada
        action_desc = "HOLD"
        if log_entry["position_change_bq"]:
            if current_pos_side == 0 and target_position > 0:
                action_desc = "OPEN_LONG"
            elif current_pos_side == 0 and target_position < 0:
                action_desc = "OPEN_SHORT"
            elif current_pos_side > 0 and target_position == 0:
                action_desc = "CLOSE_LONG"
            elif current_pos_side < 0 and target_position == 0:
                action_desc = "CLOSE_SHORT"
            elif current_pos_side > 0 and target_position < 0:
                action_desc = "CLOSE_LONG_OPEN_SHORT"
            elif current_pos_side < 0 and target_position > 0:
                action_desc = "CLOSE_SHORT_OPEN_LONG"
        log_entry["action_taken_desc_bq"] = action_desc
        
        # Datos de órdenes
        close_order = cycle_vars.get('close_order', {})
        open_order = cycle_vars.get('open_order', {})
        log_entry["order_id_close_bq"] = close_order.get('orderId') if close_order else None
        log_entry["order_status_close_bq"] = close_order.get('status') if close_order else None
        log_entry["order_id_open_bq"] = open_order.get('orderId') if open_order else None
        log_entry["order_status_open_bq"] = open_order.get('status') if open_order else None
        log_entry["order_qty_open_bq"] = float(cycle_vars.get('order_qty', 0.0))
        
        # Datos finales (serán actualizados después de ejecutar las órdenes)
        if "current_equity_after_action_bq" not in log_entry:
            log_entry["current_equity_after_action_bq"] = None
        if "error_message_bq" not in log_entry:
            log_entry["error_message_bq"] = None
            
        # Garantizar que todos los campos necesarios existan
        keys_to_ensure = [
            "kline_open_time_utc", "kline_close_time_utc", 
            "kline_o", "kline_h", "kline_l", "kline_c", "kline_v",
            "model_action_value", "action_threshold", 
            "current_position_side_bq", "desired_signal_bq",
            "position_change_bq", "pos_amt_before_action_bq", 
            "entry_price_bq", "current_price_bq",
            "unrealized_pnl_bq", "pnl_percent_bq", 
            "current_equity_before_action_bq", "leverage_bq",
            "action_taken_desc_bq", "order_id_close_bq", 
            "order_status_close_bq", "order_id_open_bq",
            "order_status_open_bq", "order_qty_open_bq", 
            "current_equity_after_action_bq", "error_message_bq"
        ]
        
        for key in keys_to_ensure:
            if key not in log_entry:
                log_entry[key] = None
                
        logger.info(f"Registro para BigQuery preparado: {log_entry['timestamp_decision_utc']}, acción={log_entry.get('model_action_value', 'N/A')}")
        
        return log_entry

async def main():
    """Función principal para iniciar el LiveTrader."""
    logger.info("Iniciando BTCBot en modo trading en vivo...")
    args = parse_arguments()
    trader = LiveTrader(args)
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
