#!/usr/bin/env python3
# scripts/test_binance_api.py
import asyncio
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Añadir la raíz del proyecto al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.live.binance_api_manager import LiveBinanceAPIManager
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger

# Configurar el logger
logger = setup_logger("BinanceAPITest")

class BinanceAPITester:
    def __init__(self):
        """Inicializa el tester de la API de Binance"""
        logger.info("Inicializando BinanceAPITester...")
        self.config_manager = ConfigManager()
        self.api_manager = LiveBinanceAPIManager(self.config_manager)
        
        # Usar configuración centralizada para testing
        testing_config = self.config_manager.get_testing_config()
        self.test_symbol = testing_config.get('default_test_symbol', "BTCUSDT")
        self.test_interval = testing_config.get('default_test_interval', "15m")
        self.test_lookback = testing_config.get('default_test_lookback_candles', 100)
        
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_skipped = 0
    
    async def initialize(self):
        """Inicializa el cliente de la API de Binance"""
        try:
            await self.api_manager.initialize_client()
            logger.info("✅ Cliente de Binance inicializado correctamente")
            self.tests_passed += 1
        except Exception as e:
            logger.error(f"❌ Error inicializando cliente de Binance: {e}")
            self.tests_failed += 1
            raise e  # Si falla la inicialización, no podemos continuar
    
    async def test_get_historical_klines(self):
        """Prueba la obtención de datos históricos de klines"""
        logger.info(f"🔍 Probando get_historical_klines para {self.test_symbol} con intervalo {self.test_interval}...")
        try:
            klines_df = await self.api_manager.get_historical_klines(
                symbol=self.test_symbol,
                interval=self.test_interval,
                lookback_candles=self.test_lookback
            )
            
            if klines_df is not None and not klines_df.empty:
                logger.info(f"✅ Se obtuvieron {len(klines_df)} klines para {self.test_symbol}")
                logger.info(f"📊 Primeras 5 velas:\n{klines_df.head()}")
                logger.info(f"📈 Última vela: {klines_df.index[-1]}")
                self.tests_passed += 1
                return True
            else:
                logger.error(f"❌ No se pudieron obtener klines para {self.test_symbol}")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_get_historical_klines: {e}")
            self.tests_failed += 1
            return False
    
    async def test_get_account_balance(self):
        """Prueba la obtención del balance de la cuenta"""
        logger.info("🔍 Probando get_account_balance...")
        try:
            balance = await self.api_manager.get_account_balance()
            if balance is not None:
                logger.info(f"✅ Balance obtenido correctamente - {len(balance)} activos")
                # Mostrar los principales activos (USDT, BTC, etc.)
                for asset in balance:
                    if asset.get('asset') in ['USDT', 'BTC', 'ETH'] or float(asset.get('balance', 0)) > 0:
                        logger.info(f"💰 {asset.get('asset')}: {asset.get('balance')} (disponible: {asset.get('withdrawAvailable')})")
                self.tests_passed += 1
                return True
            else:
                logger.error("❌ No se pudo obtener el balance de la cuenta")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_get_account_balance: {e}")
            self.tests_failed += 1
            return False
    
    async def test_get_account_info(self):
        """Prueba la obtención de la información de la cuenta"""
        logger.info("🔍 Probando get_account_info...")
        try:
            account_info = await self.api_manager.get_account_info()
            if account_info is not None:
                logger.info("✅ Información de cuenta obtenida correctamente")
                logger.info(f"💼 Total Wallet Balance: {account_info.get('totalWalletBalance')}")
                logger.info(f"💼 Total Unrealized Profit: {account_info.get('totalUnrealizedProfit')}")
                logger.info(f"💼 Available Balance: {account_info.get('availableBalance')}")
                self.tests_passed += 1
                return True
            else:
                logger.error("❌ No se pudo obtener la información de la cuenta")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_get_account_info: {e}")
            self.tests_failed += 1
            return False
    
    async def test_get_position_risk(self):
        """Prueba la obtención de la información de riesgo de posición"""
        logger.info(f"🔍 Probando get_position_risk para {self.test_symbol}...")
        try:
            position = await self.api_manager.get_position_risk(self.test_symbol)
            if position is not None:
                logger.info(f"✅ Información de posición obtenida correctamente para {self.test_symbol}")
                logger.info(f"📊 Cantidad: {position.get('positionAmt')}")
                logger.info(f"📊 Precio de entrada: {position.get('entryPrice')}")
                logger.info(f"📊 Beneficio no realizado: {position.get('unRealizedProfit')}")
                logger.info(f"📊 Apalancamiento: {position.get('leverage')}")
                self.tests_passed += 1
                return True
            else:
                logger.error(f"❌ No se pudo obtener la información de posición para {self.test_symbol}")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_get_position_risk: {e}")
            self.tests_failed += 1
            return False
    
    async def test_get_exchange_info(self):
        """Prueba la obtención de información del exchange"""
        logger.info("🔍 Probando get_exchange_info...")
        try:
            exchange_info = await self.api_manager.get_exchange_info()
            if exchange_info is not None:
                logger.info("✅ Información del exchange obtenida correctamente")
                logger.info(f"📊 Símbolos disponibles: {len(exchange_info.get('symbols', []))}")
                # Mostrar información del símbolo de prueba
                for symbol_data in exchange_info.get('symbols', []):
                    if symbol_data.get('symbol') == self.test_symbol:
                        logger.info(f"📊 Información para {self.test_symbol}:")
                        logger.info(f"📊 Status: {symbol_data.get('status')}")
                        logger.info(f"📊 Base Asset: {symbol_data.get('baseAsset')}")
                        logger.info(f"📊 Quote Asset: {symbol_data.get('quoteAsset')}")
                        break
                self.tests_passed += 1
                return True
            else:
                logger.error("❌ No se pudo obtener la información del exchange")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_get_exchange_info: {e}")
            self.tests_failed += 1
            return False
    
    async def test_get_symbol_filters(self):
        """Prueba la obtención de filtros para un símbolo"""
        logger.info(f"🔍 Probando get_symbol_filters para {self.test_symbol}...")
        try:
            filters = await self.api_manager.get_symbol_filters(self.test_symbol)
            if filters is not None:
                logger.info(f"✅ Filtros obtenidos correctamente para {self.test_symbol}")
                for filter_type, filter_data in filters.items():
                    logger.info(f"📊 Filtro {filter_type}: {json.dumps(filter_data, indent=2)}")
                self.tests_passed += 1
                return True
            else:
                logger.error(f"❌ No se pudieron obtener los filtros para {self.test_symbol}")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_get_symbol_filters: {e}")
            self.tests_failed += 1
            return False
    
    async def test_set_leverage(self):
        """Prueba la configuración del apalancamiento"""
        leverage = 3  # Apalancamiento a probar
        logger.info(f"🔍 Probando set_leverage_if_needed para {self.test_symbol} con apalancamiento {leverage}...")
        try:
            result = await self.api_manager.set_leverage_if_needed(self.test_symbol, leverage)
            if result:
                logger.info(f"✅ Apalancamiento configurado correctamente para {self.test_symbol} a {leverage}x")
                self.tests_passed += 1
                return True
            else:
                logger.error(f"❌ No se pudo configurar el apalancamiento para {self.test_symbol}")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_set_leverage: {e}")
            self.tests_failed += 1
            return False
    
    async def test_calculate_order_quantity(self):
        """Prueba el cálculo de cantidad de orden"""
        logger.info(f"🔍 Probando calculate_order_quantity para {self.test_symbol}...")
        try:
            # Obtener el precio actual (usando la última vela)
            klines_df = await self.api_manager.get_historical_klines(
                symbol=self.test_symbol,
                interval="1m",
                lookback_candles=1
            )
            
            if klines_df is not None and not klines_df.empty:
                current_price = float(klines_df['Close'].iloc[-1])
                equity = 1000.0  # Supongamos $1000 de equidad
                leverage = 3     # Apalancamiento de 3x
                
                quantity = await self.api_manager.calculate_order_quantity(
                    symbol=self.test_symbol,
                    equity=equity,
                    current_price=current_price,
                    leverage=leverage
                )
                
                logger.info(f"✅ Cantidad de orden calculada: {quantity} {self.test_symbol.replace('USDT', '')}")
                logger.info(f"📊 Precio actual: ${current_price}")
                logger.info(f"📊 Equidad: ${equity}")
                logger.info(f"📊 Apalancamiento: {leverage}x")
                logger.info(f"📊 Valor nocional: ${quantity * current_price:.2f}")
                self.tests_passed += 1
                return True
            else:
                logger.error("❌ No se pudo obtener el precio actual para calcular la cantidad de orden")
                self.tests_failed += 1
                return False
        except Exception as e:
            logger.error(f"❌ Error en test_calculate_order_quantity: {e}")
            self.tests_failed += 1
            return False
    
    async def run_all_tests(self):
        """Ejecuta todas las pruebas disponibles en el tester"""
        try:
            logger.info("🚀 Iniciando pruebas del API de Binance...")
            
            # Inicializar el cliente (requerido para todas las pruebas)
            await self.initialize()
            
            # Ejecutar pruebas
            await self.test_get_exchange_info()
            await self.test_get_symbol_filters()
            await self.test_get_historical_klines()
            await self.test_get_account_balance()
            await self.test_get_account_info()
            await self.test_get_position_risk()
            await self.test_set_leverage()
            await self.test_calculate_order_quantity()
            
            # No probamos place_market_order ni close_market_position para evitar operaciones reales
            
            # Cerrar el cliente al terminar
            await self.api_manager.close_client_session()
            
            # Mostrar resultados finales
            logger.info("\n" + "="*50)
            logger.info("📋 RESULTADOS DE LAS PRUEBAS:")
            logger.info(f"✅ Pruebas pasadas: {self.tests_passed}")
            logger.info(f"❌ Pruebas fallidas: {self.tests_failed}")
            logger.info(f"⚠️ Pruebas omitidas: {self.tests_skipped}")
            logger.info("="*50)
            
            return self.tests_failed == 0
        except Exception as e:
            logger.error(f"❌ Error ejecutando pruebas: {e}")
            # Intentar cerrar el cliente en caso de error
            try:
                await self.api_manager.close_client_session()
            except:
                pass
            return False

async def main():
    """Función principal para ejecutar el tester"""
    tester = BinanceAPITester()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    # Ejecutar la función principal con asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
