# src/live/websocket_manager.py
import asyncio
import json
import websockets
import os
import logging
from src.utils.config import ConfigManager # Asumiendo que ConfigManager está accesible

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
            self.websocket_url = f"wss://stream.binancefuture.com/ws/{self.symbol}@kline_{self.interval}"
        else: # REAL
            self.websocket_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.interval}"
        
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
                await asyncio.sleep(self.retry_delay)
            except OSError as os_err: # Problemas de red a nivel de SO
                logger.error(f"Error de OS durante la conexión WebSocket (ej. Network is unreachable): {os_err}. Reintentando en {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)
            except Exception as e_outer_loop: # Otros errores al intentar conectar (ej. timeout)
                logger.error(f"Error inesperado en el bucle externo de WebSocket (conexión): {e_outer_loop}. Reintentando en {self.unexpected_error_delay}s...", exc_info=True)
                await asyncio.sleep(self.unexpected_error_delay) # Delay más largo
            else: # Se ejecuta si el bloque try del `async with` termina sin excepciones (ej. por un break interno)
                pass # Simplemente procederá al delay de reconexión de abajo

            await asyncio.sleep(self.retry_delay) # Esperar antes de reintentar la conexión en el bucle externo
