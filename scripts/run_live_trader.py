import asyncio
import os
import numpy as np
import pandas as pd
import time
import sys
from dotenv import load_dotenv
from pathlib import Path
from google.cloud import aiplatform

# Agregar la ruta del proyecto para importaciones correctas
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.live.websocket_manager import LiveWebsocketManager 
from src.live.binance_api_manager import LiveBinanceAPIManager
from src.live.live_data_processor import LiveFeatureProcessor
from src.live.portfolio_feature_builder import build_live_portfolio_features

logger = setup_logger("LiveTrader")

async def main_live_trader():
    logger.info("Iniciando Live Trader Bot...")
    try:
        # Cargar variables de entorno
        load_dotenv(dotenv_path=os.path.join(project_root, '.env'))
        config_manager = ConfigManager(config_path=os.path.join(project_root, "src/config.yaml"), env_path=os.path.join(project_root, ".env"))
    except Exception as e:
        logger.error(f"Error fatal al cargar ConfigManager: {e}", exc_info=True)
        return

    trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
    logger.info(f"Modo de Trading: {trading_mode}")

    try:
        live_config = config_manager.get_config_value('live_trading')
        env_config = config_manager.get_environment_config()
        data_acq_config = config_manager.get_data_acquisition_defaults()
        preproc_config = config_manager.get_preprocessing_config()

        notification_queue = asyncio.Queue()
        
        # Inicializar gestores
        websocket_manager = LiveWebsocketManager(config_manager, notification_queue)
        binance_api_manager = LiveBinanceAPIManager(config_manager)
        live_feature_processor = LiveFeatureProcessor(config_manager)

        # Inicializar cliente de Vertex AI
        project_id = os.getenv('GCP_PROJECT_ID')
        region = os.getenv('GCP_REGION')
        vertex_endpoint_id = live_config['vertex_ai_endpoint_raw_id']
        
        if not all([project_id, region, vertex_endpoint_id]):
            logger.error("Configuración de Vertex AI incompleta (GCP_PROJECT_ID, GCP_REGION, vertex_ai_endpoint_raw_id).")
            return
        
        aiplatform.init(project=project_id, location=region)
        vertex_endpoint = aiplatform.Endpoint(vertex_endpoint_id)
        logger.info(f"Cliente de Vertex AI Endpoint inicializado para: {vertex_endpoint.resource_name}")

        # Inicializar el cliente de Binance y configurar apalancamiento
        await binance_api_manager.init_client()
        await binance_api_manager.set_leverage_if_needed(
            data_acq_config['symbol'], 
            int(env_config['leverage'])
        )
        logger.info(f"Apalancamiento para {data_acq_config['symbol']} verificado/establecido en {env_config['leverage']}x.")

    except Exception as e:
        logger.error(f"Error fatal durante la inicialización de managers o configuración de apalancamiento: {e}", exc_info=True)
        return

    # Inicializar variables para el ciclo de trading
    last_step_equity = None

    # Iniciar el WebSocket Manager en una tarea separada
    websocket_task = asyncio.create_task(websocket_manager.run())
    logger.info("WebSocket Manager iniciado y escuchando.")

    try:
        # Bucle principal de trading
        while True:
            try:
                logger.info(f"Esperando notificación de nueva vela cerrada para {data_acq_config['symbol']}@{data_acq_config['interval']}...")
                closed_kline_message = await notification_queue.get()
                
                logger.info(f"Nueva vela cerrada detectada (TS: {closed_kline_message.get('t')}, Px: {closed_kline_message.get('c')}). Procesando...")

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
                    logger.warning("Falló el preprocesamiento de datos de mercado. Saltando ciclo.")
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
                if last_step_equity is None:
                    last_step_equity = current_equity 

                # 4. Construir Características de Cartera
                portfolio_features_array = build_live_portfolio_features(
                    account_general_info, account_balance_info, position_info, 
                    env_config, float(env_config['initial_equity']), last_step_equity
                )
                last_step_equity = current_equity

                # 5. Preparar y Enviar Observación al Modelo
                observation_instance = {
                    "market_features": market_features_array.tolist(),
                    "portfolio_features": portfolio_features_array.tolist()
                }
                
                logger.debug("Enviando observación al endpoint de Vertex AI...")
                prediction_response = vertex_endpoint.predict(instances=[observation_instance])
                action_value_from_model = float(prediction_response.predictions[0]['action_value'])
                logger.info(f"Respuesta del modelo: {action_value_from_model:.4f}")

                # 6. Interpretar Acción y Ejecutar
                action_threshold = env_config['action_threshold']
                desired_signal = 0  # 0 Neutral, 1 Long, -1 Short
                
                if action_value_from_model > action_threshold:
                    desired_signal = 1
                elif action_value_from_model < -action_threshold:
                    desired_signal = -1
                
                current_position_amt = float(position_info['positionAmt'])
                current_position_side = 0
                
                if current_position_amt > 0:
                    current_position_side = 1
                elif current_position_amt < 0:
                    current_position_side = -1

                logger.info(f"Posición actual en Binance: {current_position_side} (Amt: {current_position_amt}). Señal del modelo: {desired_signal}.")

                if desired_signal != current_position_side:
                    # Cerrar posición existente si es necesario
                    if current_position_side != 0:
                        logger.info(f"Decisión: Cerrar posición actual ({current_position_side}) de {abs(current_position_amt)} {data_acq_config['symbol']}.")
                        close_result = await binance_api_manager.close_market_position(data_acq_config['symbol'], current_position_amt)
                        if close_result:
                            logger.info(f"Resultado cierre: {close_result.get('status')}")
                        await asyncio.sleep(live_config.get('post_close_delay_seconds', 1))
                    
                    # Abrir nueva posición si la señal no es neutral
                    if desired_signal != 0:
                        current_market_price_for_qty = float(closed_kline_message.get('c'))

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
                            if order_result:
                                logger.info(f"Resultado apertura: {order_result.get('status')}")
                        else:
                            logger.warning(f"La cantidad calculada para la orden es 0 o inválida. No se abrirá posición {desired_signal}.")
                else:
                    logger.info("Decisión: Mantener posición/estado actual.")

                notification_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Live Trader cancelado.")
                break
            except aiplatform.errors.PredictionError as pe:
                logger.error(f"Error de predicción de Vertex AI: {pe} (Code: {getattr(pe, 'code', 'unknown')})", exc_info=False)
                await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds', 30))
            except Exception as e:
                logger.error(f"Error en el bucle principal del Live Trader: {e}", exc_info=True)
                await asyncio.sleep(live_config.get('websocket_unexpected_error_delay_seconds', 60))
    
    except Exception as e:
        logger.error(f"Error global en Live Trader: {e}", exc_info=True)
    
    finally:
        # Limpiar recursos
        if websocket_task and not websocket_task.done():
            websocket_task.cancel()
        
        # Cerrar cliente de Binance
        try:
            await binance_api_manager.close_client()
        except Exception as e:
            logger.error(f"Error cerrando cliente de Binance: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main_live_trader())
    except KeyboardInterrupt:
        logger.info("Live Trader detenido manualmente por el usuario.")
    except Exception as e_global:
        logger.critical(f"Error global irrecuperable en Live Trader: {e_global}", exc_info=True)
