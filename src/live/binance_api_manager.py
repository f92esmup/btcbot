# src/live/binance_api_manager.py
import os
import logging
import pandas as pd
import time 
import asyncio

from binance.client import AsyncClient # Importante: Usar AsyncClient para asyncio
from binance.exceptions import BinanceAPIException, BinanceOrderException, BinanceRequestException

from src.utils.config import ConfigManager
from typing import Optional, Dict, List, Any, Union

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
                    logger.warning(f"Valor nocional calculado {quantity * current_price:.2f} USDT es menor que el mínimo permitido {min_notional_value} USDT.")
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
                reduceOnly=True  # Importante: para asegurar que solo cierre la posición existente
            )
            logger.info(f"Posición cerrada para {symbol} con orden MARKET {side_to_close}: {order_response}")
            return order_response
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance cerrando posición para {symbol}: Status {e.status_code}, Code {e.code}, Msg {e.message}")
        except BinanceOrderException as e:
            logger.error(f"Excepción de Orden de Binance cerrando posición para {symbol}: Status {e.status_code}, Code {e.code}, Msg {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado cerrando posición para {symbol}: {e}", exc_info=True)
        return None

    async def set_leverage_if_needed(self, symbol: str, desired_leverage: int) -> bool:
        await self._ensure_client()
        try:
            # Verificar la posición actual y su apalancamiento
            position_info = await self.get_position_risk(symbol)
            if not position_info:
                logger.warning(f"No se pudo obtener información de posición para {symbol}, no se actualizará el apalancamiento.")
                return False
                
            current_leverage = int(float(position_info.get('leverage', '1')))
            if current_leverage == desired_leverage:
                logger.debug(f"Apalancamiento para {symbol} ya está en {desired_leverage}x. No se necesita cambio.")
                return True
                
            # Si el apalancamiento actual es diferente del deseado, actualizar
            logger.info(f"Cambiando apalancamiento para {symbol} de {current_leverage}x a {desired_leverage}x...")
            response = await self.client.futures_change_leverage(
                symbol=symbol.upper(),
                leverage=desired_leverage
            )
            logger.info(f"Apalancamiento para {symbol} actualizado a {response.get('leverage')}x")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Excepción de API de Binance actualizando apalancamiento para {symbol}: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"Error inesperado actualizando apalancamiento para {symbol}: {e}", exc_info=True)
        return False
