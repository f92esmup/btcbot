import asyncio
import json
import logging
import os
import websockets
from websockets.exceptions import ConnectionClosed

from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger

logger = setup_logger('WebsocketManager')

class LiveWebsocketManager:
    """
    Gestiona la conexión WebSocket para detectar el cierre de nuevas velas en Binance Futures.
    Envía notificaciones a una cola asyncio cuando se cierra una vela.
    """
    def __init__(self, config_manager: ConfigManager, notification_queue: asyncio.Queue):
        """
        Inicializa el gestor de WebSockets.
        
        Args:
            config_manager: Instancia de ConfigManager para obtener la configuración
            notification_queue: Cola para enviar notificaciones de velas cerradas
        """
        self.config_manager = config_manager
        self.notification_queue = notification_queue
        
        # Obtener los defaults para symbol e interval
        data_acq_defaults = config_manager.get_data_acquisition_defaults()
        self.symbol = data_acq_defaults.get('symbol', 'BTCUSDT').lower()
        self.interval = data_acq_defaults.get('interval', '1h')
        
        # Obtener el modo de trading (TESTNET o REAL)
        self.trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
        
        # Configurar la URL del WebSocket según el modo
        if self.trading_mode == 'TESTNET':
            self.websocket_base_url = 'wss://stream.binancefuture.com/ws/'
        else:  # REAL
            self.websocket_base_url = 'wss://fstream.binance.com/ws/'
        
        # Construir la URL completa
        self.stream_name = f"{self.symbol}@kline_{self.interval}"
        self.websocket_url = f"{self.websocket_base_url}{self.stream_name}"
        
        # Obtener configuración de reintentos
        live_config = config_manager.get_config_value('live_trading')
        self.retry_delay = live_config.get('websocket_retry_delay_seconds', 10)
        self.unexpected_error_delay = live_config.get('websocket_unexpected_error_delay_seconds', 30)
        
        logger.info(f"WebSocket Manager inicializado para {self.stream_name} en modo {self.trading_mode}")
        logger.debug(f"URL del WebSocket: {self.websocket_url}")

    async def _process_message(self, message_str: str):
        """
        Procesa un mensaje recibido del WebSocket.
        Si es una vela cerrada, la envía a la cola de notificaciones.
        
        Args:
            message_str: Mensaje JSON del WebSocket como string
        """
        try:
            message = json.loads(message_str)
            
            # Verificar si es un mensaje de kline
            if 'e' in message and message['e'] == 'kline':
                kline_data = message['k']
                
                # Verificar si la vela está cerrada
                if kline_data['x'] is True:  # 'x' indica si la vela está cerrada
                    logger.info(f"Vela cerrada detectada para {self.symbol}@{self.interval}")
                    logger.debug(f"Datos de vela cerrada: Open={kline_data['o']}, Close={kline_data['c']}, High={kline_data['h']}, Low={kline_data['l']}, Volume={kline_data['v']}")
                    
                    # Poner el mensaje en la cola de notificaciones
                    await self.notification_queue.put(kline_data)
        except json.JSONDecodeError:
            logger.error(f"Error decodificando mensaje JSON del WebSocket: {message_str}")
        except Exception as e:
            logger.error(f"Error procesando mensaje del WebSocket: {e}", exc_info=True)

    async def run(self):
        """
        Ejecuta el bucle principal del WebSocket.
        Se reconecta automáticamente en caso de desconexión.
        """
        while True:
            try:
                logger.info(f"Conectando al WebSocket: {self.websocket_url}")
                async with websockets.connect(self.websocket_url) as ws:
                    logger.info(f"Conexión WebSocket establecida para {self.stream_name}")
                    
                    while True:
                        try:
                            message = await ws.recv()
                            await self._process_message(message)
                        except ConnectionClosed:
                            logger.warning("Conexión WebSocket cerrada. Intentando reconectar...")
                            break
            except Exception as e:
                logger.error(f"Error en conexión WebSocket: {e}", exc_info=True)
                logger.info(f"Reintentando conexión en {self.retry_delay} segundos...")
                
                # Esperar antes de reintentar
                await asyncio.sleep(self.retry_delay)
