import pandas as pd
import threading
import time
from src.data.binance_source import BinanceDataSource
from binance.client import Client
from binance import ThreadedWebsocketManager


class BinanceLiveDataReader:
    def __init__(self, symbol: str, interval: str, subscriber: object, warm_up_candles: int):
        self.symbol = symbol
        self.interval = interval
        self.subscriber = subscriber
        self.warm_up_candles = warm_up_candles
        self.historical_df = pd.DataFrame()
        self.api_client = Client()
    
    def start(self):
        self._warm_up()
        self._start_websocket_listener()
    
    def _warm_up(self):
        print("Realizando warm-up de datos históricos...")
        
        # Descargar las últimas velas históricas
        klines = self.api_client.futures_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.warm_up_candles
        )
        
        # Crear DataFrame con los datos descargados
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Seleccionar solo las columnas necesarias
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convertir columnas OHLCV a float
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Establecer timestamp como índice y asignar al DataFrame histórico
        df.set_index('timestamp', inplace=True)
        self.historical_df = df
    
    def _start_websocket_listener(self):
        print("Iniciando conexión WebSocket...")
        
        # Crear instancia del ThreadedWebsocketManager
        twm = ThreadedWebsocketManager()
        
        # Iniciar el manager
        twm.start()
        
        # Suscribirse al stream de klines
        twm.start_kline_socket(
            callback=self._on_new_websocket_message,
            symbol=self.symbol,
            interval=self.interval
        )
        
        # Mantener el programa en ejecución
        twm.join()
    
    def _on_new_websocket_message(self, msg):
        # Verificar si es un mensaje de error
        if msg.get('e') == 'error':
            print(f"Error en WebSocket: {msg}")
            return
        
        # Verificar si la vela está cerrada
        if msg.get('k', {}).get('x', False):
            kline_data = msg['k']
            
            # Formatear los datos de la vela
            candle_data = {
                'timestamp': pd.to_datetime(kline_data['t'], unit='ms'),
                'Open': float(kline_data['o']),
                'High': float(kline_data['h']),
                'Low': float(kline_data['l']),
                'Close': float(kline_data['c']),
                'Volume': float(kline_data['v'])
            }
            
            # Convertir la nueva vela en un DataFrame de una sola fila con timestamp como índice
            new_candle_df = pd.DataFrame([candle_data])
            new_candle_df.set_index('timestamp', inplace=True)
            
            # Concatenar la nueva vela al DataFrame histórico
            self.historical_df = pd.concat([self.historical_df, new_candle_df])
            
            # Implementar lógica de de-duplicación: eliminar filas con índices duplicados,
            # conservando la última aparición (el dato del WebSocket prevalece)
            self.historical_df = self.historical_df[~self.historical_df.index.duplicated(keep='last')]
            
            # Recortar el DataFrame para mantener solo las últimas warm_up_candles filas
            self.historical_df = self.historical_df.tail(self.warm_up_candles)
            
            # Notificar al suscriptor si existe, pasando una copia del DataFrame
            if self.subscriber is not None:
                self.subscriber.on_new_candle(self.historical_df.copy())
