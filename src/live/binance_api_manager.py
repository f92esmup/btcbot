import asyncio
import logging
import os
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from binance import AsyncClient, Client, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException

from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger

logger = setup_logger('BinanceAPIManager')

class LiveBinanceAPIManager:
    """
    Encapsula todas las interacciones con la API REST de Binance Futures.
    Maneja la distinción entre Testnet y Real, obteniendo credenciales de Secret Manager.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Inicializa el gestor de API de Binance.
        
        Args:
            config_manager: Instancia de ConfigManager para obtener la configuración
        """
        self.config_manager = config_manager
        
        # Obtener el modo de trading (TESTNET o REAL)
        self.trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
        
        # Determinar los nombres de los secretos según el modo de trading
        if self.trading_mode == 'TESTNET':
            api_key_secret_name = os.getenv('SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES')
            api_secret_secret_name = os.getenv('SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES')
            logger.info("Usando credenciales de Testnet para Binance Futures")
        else:  # REAL
            api_key_secret_name = os.getenv('SECRET_NAME_BINANCE_API_KEY_FUTURES')
            api_secret_secret_name = os.getenv('SECRET_NAME_BINANCE_API_SECRET_FUTURES')
            logger.info("Usando credenciales de producción para Binance Futures")
        
        # Obtener los valores de los secretos mediante ConfigManager
        self.api_key = config_manager.get_env_variable(api_key_secret_name)
        self.api_secret = config_manager.get_env_variable(api_secret_secret_name)
        
        if not self.api_key or not self.api_secret:
            error_msg = f"Credenciales de API no disponibles para el modo {self.trading_mode}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        # Configuración del entorno
        self.env_config = config_manager.get_environment_config()
        
        # Inicializar el cliente asincrónico para Binance Futures
        self.client = None  # Se inicializará en init_client
    
    async def init_client(self):
        """Inicializa el cliente asincrónico de Binance"""
        if self.client is None:
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=(self.trading_mode == 'TESTNET')
            )
            logger.info(f"Cliente de Binance inicializado para {self.trading_mode}")
    
    async def close_client(self):
        """Cierra el cliente asincrónico"""
        if self.client:
            await self.client.close_connection()
            self.client = None
            logger.info("Cliente de Binance cerrado")
    
    async def _retry_async_binance_call(self, func, *args, max_retries=3, **kwargs):
        """
        Función auxiliar para reintentar llamadas a la API de Binance con manejo de errores.
        
        Args:
            func: Función asincrónica a ejecutar
            *args: Argumentos posicionales para la función
            max_retries: Número máximo de reintentos
            **kwargs: Argumentos de palabras clave para la función
            
        Returns:
            Resultado de la función o None en caso de error
        """
        await self.init_client()  # Asegurarse de que el cliente está inicializado
        
        retries = 0
        while retries < max_retries:
            try:
                result = await func(*args, **kwargs)
                return result
            except BinanceAPIException as e:
                logger.error(f"Error de API de Binance: {e.message} (Código: {e.code})")
                if e.code == -1021:  # Timestamp error
                    logger.warning("Error de timestamp. Sincronizando...")
                elif e.code == -2010:  # Funds error
                    logger.error("Fondos insuficientes")
                    return None
                elif e.code == -1013:  # Invalid quantity or price
                    logger.error("Cantidad o precio inválido")
                    return None
            except BinanceRequestException as e:
                logger.error(f"Error de solicitud a Binance: {e}")
            except Exception as e:
                logger.error(f"Error general en llamada a Binance: {e}", exc_info=True)
            
            retries += 1
            if retries < max_retries:
                logger.info(f"Reintentando llamada ({retries}/{max_retries})...")
                await asyncio.sleep(2 ** retries)  # Espera exponencial
            else:
                logger.error(f"Máximo de reintentos alcanzado ({max_retries})")
                return None
    
    async def get_historical_klines(self, symbol: str, interval: str, lookback_candles: int) -> Optional[pd.DataFrame]:
        """
        Obtiene las velas históricas para un símbolo e intervalo.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            interval: Intervalo de tiempo (ej. 1h, 4h, 1d)
            lookback_candles: Número de velas a obtener
            
        Returns:
            DataFrame con las velas históricas o None en caso de error
        """
        try:
            # Convertir lookback_candles a límite de tiempo (ms)
            interval_ms = self._interval_to_milliseconds(interval)
            lookback_ms = interval_ms * lookback_candles
            end_time = int(time.time() * 1000)  # Tiempo actual en ms
            start_time = end_time - lookback_ms
            
            # Hacer la llamada a la API
            klines = await self._retry_async_binance_call(
                self.client.futures_klines,
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=lookback_candles
            )
            
            if not klines:
                logger.error(f"No se obtuvieron velas para {symbol}@{interval}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Usar timestamp como índice
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Obtenidas {len(df)} velas históricas para {symbol}@{interval}")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo velas históricas: {e}", exc_info=True)
            return None
    
    async def get_account_balance(self) -> Optional[Dict]:
        """
        Obtiene el balance de la cuenta.
        
        Returns:
            Diccionario con el balance de la cuenta o None en caso de error
        """
        try:
            balance = await self._retry_async_binance_call(
                self.client.futures_account_balance
            )
            if not balance:
                logger.error("No se pudo obtener el balance de la cuenta")
                return None
            
            # Organizar por asset para fácil acceso
            balance_dict = {item['asset']: item for item in balance}
            logger.info(f"Balance de cuenta obtenido: {balance_dict.get('USDT', {}).get('balance', 'N/A')} USDT")
            return balance_dict
        except Exception as e:
            logger.error(f"Error obteniendo balance de cuenta: {e}", exc_info=True)
            return None
    
    async def get_account_info(self) -> Optional[Dict]:
        """
        Obtiene información completa de la cuenta.
        
        Returns:
            Diccionario con la información de la cuenta o None en caso de error
        """
        try:
            account_info = await self._retry_async_binance_call(
                self.client.futures_account
            )
            if not account_info:
                logger.error("No se pudo obtener la información de la cuenta")
                return None
            
            logger.info(f"Información de cuenta obtenida. Equity: {account_info.get('totalWalletBalance', 'N/A')} USDT")
            return account_info
        except Exception as e:
            logger.error(f"Error obteniendo información de cuenta: {e}", exc_info=True)
            return None
    
    async def get_position_risk(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene información de riesgo de posición para un símbolo.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            
        Returns:
            Diccionario con la información de riesgo o None en caso de error
        """
        try:
            position_risk = await self._retry_async_binance_call(
                self.client.futures_position_risk,
                symbol=symbol
            )
            if not position_risk:
                logger.error(f"No se pudo obtener la información de riesgo para {symbol}")
                return None
            
            # Si es una lista, tomar el primer elemento que coincida con el símbolo
            if isinstance(position_risk, list):
                for pos in position_risk:
                    if pos['symbol'] == symbol:
                        logger.info(f"Posición para {symbol}: {pos.get('positionAmt', '0')} (PNL: {pos.get('unRealizedProfit', '0')})")
                        return pos
                
                logger.warning(f"No se encontró posición para {symbol} en la respuesta")
                return {'symbol': symbol, 'positionAmt': '0', 'unRealizedProfit': '0'}
            else:
                logger.info(f"Posición para {symbol}: {position_risk.get('positionAmt', '0')} (PNL: {position_risk.get('unRealizedProfit', '0')})")
                return position_risk
        except Exception as e:
            logger.error(f"Error obteniendo información de riesgo: {e}", exc_info=True)
            return None
    
    async def get_exchange_info(self) -> Optional[Dict]:
        """
        Obtiene información de exchange.
        
        Returns:
            Diccionario con la información de exchange o None en caso de error
        """
        try:
            exchange_info = await self._retry_async_binance_call(
                self.client.futures_exchange_info
            )
            if not exchange_info:
                logger.error("No se pudo obtener la información de exchange")
                return None
            
            logger.info(f"Información de exchange obtenida. {len(exchange_info.get('symbols', []))} símbolos disponibles.")
            return exchange_info
        except Exception as e:
            logger.error(f"Error obteniendo información de exchange: {e}", exc_info=True)
            return None
    
    async def get_symbol_filters(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene los filtros para un símbolo específico.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            
        Returns:
            Diccionario con los filtros del símbolo o None en caso de error
        """
        try:
            exchange_info = await self.get_exchange_info()
            if not exchange_info:
                return None
            
            symbol_info = None
            for s in exchange_info.get('symbols', []):
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                logger.error(f"No se encontró información para el símbolo {symbol}")
                return None
            
            # Organizar filtros por tipo para fácil acceso
            filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
            
            logger.info(f"Filtros obtenidos para {symbol}")
            return filters
        except Exception as e:
            logger.error(f"Error obteniendo filtros del símbolo: {e}", exc_info=True)
            return None
    
    async def calculate_order_quantity(self, symbol: str, equity: float, current_price: float, leverage: float) -> float:
        """
        Calcula la cantidad de la orden ajustada según los filtros del símbolo.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            equity: Equity actual (USDT)
            current_price: Precio actual del mercado
            leverage: Apalancamiento a utilizar
            
        Returns:
            Cantidad calculada para la orden (unidades del activo base)
        """
        try:
            # Obtener los filtros del símbolo
            filters = await self.get_symbol_filters(symbol)
            if not filters:
                return 0.0
            
            # Obtener configuración del tamaño de posición
            position_size_pct = float(self.env_config.get('position_size_pct_equity', 0.05))
            
            # Calcular el tamaño nominal de la orden
            notional_size = equity * position_size_pct * leverage
            
            # Convertir a cantidad de base asset (ej. BTC)
            base_quantity = notional_size / current_price
            
            # Aplicar filtro LOT_SIZE
            if 'LOT_SIZE' in filters:
                lot_filter = filters['LOT_SIZE']
                min_qty = float(lot_filter['minQty'])
                max_qty = float(lot_filter['maxQty'])
                step_size = float(lot_filter['stepSize'])
                
                # Redondear hacia abajo al múltiplo más cercano de stepSize
                precision = self._get_precision(step_size)
                base_quantity = self._floor_to_precision(base_quantity, precision)
                
                # Verificar mínimos y máximos
                base_quantity = max(min_qty, min(max_qty, base_quantity))
            
            # Aplicar filtro MIN_NOTIONAL si existe
            if 'MIN_NOTIONAL' in filters:
                min_notional = float(filters['MIN_NOTIONAL']['notional'])
                if base_quantity * current_price < min_notional:
                    logger.warning(f"Notional calculado ({base_quantity * current_price}) es menor que el mínimo requerido ({min_notional})")
                    if min_notional / current_price > float(lot_filter.get('minQty', 0)):
                        base_quantity = self._floor_to_precision(min_notional / current_price, precision)
                        logger.info(f"Ajustando cantidad a {base_quantity} para cumplir con MIN_NOTIONAL")
                    else:
                        logger.error(f"No se puede satisfacer MIN_NOTIONAL sin violar minQty")
                        return 0.0
            
            # Verificar el tamaño mínimo de orden de BTC configurado
            min_order_size_btc = float(self.env_config.get('min_order_size_btc', 0.001))
            if base_quantity < min_order_size_btc:
                logger.warning(f"Cantidad calculada ({base_quantity}) es menor que el mínimo configurado ({min_order_size_btc})")
                return 0.0
            
            logger.info(f"Cantidad calculada para orden: {base_quantity} {symbol.replace('USDT', '')}")
            return base_quantity
            
        except Exception as e:
            logger.error(f"Error calculando cantidad de orden: {e}", exc_info=True)
            return 0.0
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """
        Coloca una orden de mercado.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            side: Lado de la orden ('BUY' o 'SELL')
            quantity: Cantidad de la orden (unidades del activo base)
            
        Returns:
            Información de la orden o None en caso de error
        """
        if quantity <= 0:
            logger.error(f"Cantidad inválida para orden: {quantity}")
            return None
        
        try:
            # Redondear la cantidad a 3 decimales (típico para BTC)
            quantity = round(quantity, 3)
            
            logger.info(f"Colocando orden de mercado {side} para {quantity} {symbol}")
            
            # Configurar parámetros de la orden
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity
            }
            
            # Ejecutar la orden
            order_result = await self._retry_async_binance_call(
                self.client.futures_create_order,
                **order_params
            )
            
            if not order_result:
                logger.error(f"No se pudo colocar la orden {side} para {quantity} {symbol}")
                return None
            
            logger.info(f"Orden colocada exitosamente: {order_result.get('orderId')} ({order_result.get('status')})")
            return order_result
            
        except Exception as e:
            logger.error(f"Error colocando orden de mercado: {e}", exc_info=True)
            return None
    
    async def close_market_position(self, symbol: str, position_amt_to_close: float) -> Optional[Dict]:
        """
        Cierra una posición existente con una orden de mercado.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            position_amt_to_close: Cantidad de la posición a cerrar (+ para long, - para short)
            
        Returns:
            Información de la orden o None en caso de error
        """
        try:
            # Si no hay posición, no hacer nada
            if position_amt_to_close == 0:
                logger.info(f"No hay posición abierta para {symbol}")
                return None
            
            # Determinar side para cerrar (opuesto a la posición)
            side = "SELL" if position_amt_to_close > 0 else "BUY"
            quantity = abs(float(position_amt_to_close))
            
            logger.info(f"Cerrando posición {symbol}: {position_amt_to_close} con orden {side}")
            
            # Colocar orden de mercado para cerrar
            close_order = await self.place_market_order(symbol, side, quantity)
            return close_order
            
        except Exception as e:
            logger.error(f"Error cerrando posición: {e}", exc_info=True)
            return None
    
    async def set_leverage_if_needed(self, symbol: str, desired_leverage: int):
        """
        Establece el apalancamiento si es diferente al actual.
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            desired_leverage: Apalancamiento deseado
        """
        try:
            # Verificar apalancamiento actual
            position_info = await self.get_position_risk(symbol)
            if not position_info:
                logger.error(f"No se pudo verificar el apalancamiento actual para {symbol}")
                return
            
            current_leverage = int(float(position_info.get('leverage', 0)))
            
            # Si el apalancamiento ya está configurado, no hacer nada
            if current_leverage == desired_leverage:
                logger.info(f"Apalancamiento para {symbol} ya está configurado en {desired_leverage}x")
                return
            
            # Configurar nuevo apalancamiento
            logger.info(f"Cambiando apalancamiento de {current_leverage}x a {desired_leverage}x para {symbol}")
            
            result = await self._retry_async_binance_call(
                self.client.futures_change_leverage,
                symbol=symbol,
                leverage=desired_leverage
            )
            
            if result:
                logger.info(f"Apalancamiento configurado en {result.get('leverage')}x para {symbol}")
            else:
                logger.error(f"No se pudo configurar el apalancamiento para {symbol}")
                
        except Exception as e:
            logger.error(f"Error configurando apalancamiento: {e}", exc_info=True)
    
    def _interval_to_milliseconds(self, interval: str) -> int:
        """
        Convierte un intervalo de velas a milisegundos.
        
        Args:
            interval: Intervalo de tiempo (ej. 1h, 4h, 1d)
            
        Returns:
            Intervalo en milisegundos
        """
        # Definir multiplicadores
        seconds_per_unit = {
            'm': 60,
            'h': 60 * 60,
            'd': 24 * 60 * 60,
            'w': 7 * 24 * 60 * 60
        }
        
        # Extraer unidad y valor
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit in seconds_per_unit:
            return value * seconds_per_unit[unit] * 1000
        else:
            raise ValueError(f"Intervalo inválido: {interval}")
    
    def _get_precision(self, step_size: float) -> int:
        """
        Obtiene la precisión a partir del tamaño de paso.
        
        Args:
            step_size: Tamaño de paso
            
        Returns:
            Precisión decimal
        """
        step_str = "{:0.8f}".format(step_size)
        step_str = step_str.rstrip('0')
        
        if '.' in step_str:
            return len(step_str) - step_str.index('.') - 1
        else:
            return 0
    
    def _floor_to_precision(self, value: float, precision: int) -> float:
        """
        Redondea un valor hacia abajo según la precisión.
        
        Args:
            value: Valor a redondear
            precision: Precisión decimal
            
        Returns:
            Valor redondeado
        """
        factor = 10 ** precision
        return int(value * factor) / factor
