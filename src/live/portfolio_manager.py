from binance.client import Client
from binance.exceptions import BinanceAPIException
from enum import Enum
from src.configuration.config import config
import math


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class LivePortfolioManager:
    def __init__(self, api_key: str, api_secret: str, is_testnet: bool, symbol: str = "BTCUSDT"):
        self.client = Client(api_key, api_secret, testnet=is_testnet)
        self.symbol = symbol
        self.balance = 0.0
        self.current_position = None
        self.pnl = 0.0
        self.leverage = config.apalancamiento
        self.max_investment_pct = config.porcentaje_max_inversion_por_trade
        self.quantity_precision = self._get_quantity_precision()
    
    def _get_quantity_precision(self) -> int:
        """
        Obtiene la precisión requerida para la cantidad de una orden para el símbolo actual.
        """
        print(f"Obteniendo información de trading para {self.symbol}...")
        exchange_info = self.client.futures_exchange_info()
        for s_info in exchange_info['symbols']:
            if s_info['symbol'] == self.symbol:
                for filter in s_info['filters']:
                    if filter['filterType'] == 'LOT_SIZE':
                        step_size = float(filter['stepSize'])
                        # Calcula el número de decimales a partir del step_size
                        precision = int(round - math.log(step_size, 10), 0)
                        print(f"Precisión de cantidad para {self.symbol} establecida en: {precision} decimales.")
                        return precision
        raise ValueError(f"No se pudo obtener la precisión de cantidad para {self.symbol}")
    
    def initialize_state(self):
        print("Inicializando y sincronizando estado del portfolio...")
        account_balance = self.client.futures_account_balance()
        for asset in account_balance:
            if asset['asset'] == 'USDT':
                self.balance = float(asset['balance'])
                break
        
        positions = self.client.futures_position_information()
        for position in positions:
            if position['symbol'] == self.symbol:
                position_amt = float(position['positionAmt'])
                if position_amt != 0:
                    print(f"⚠️ Advertencia: Posición abierta detectada para {self.symbol}: {position_amt}")
                    print("Cerrando posición preexistente...")
                    order_side = 'SELL' if position_amt > 0 else 'BUY'
                    self.client.futures_create_order(
                        symbol=self.symbol,
                        side=order_side,
                        type='MARKET',
                        quantity=abs(position_amt)
                    )
                    print("Posición cerrada exitosamente.")
                break
        print(f"✅ Estado sincronizado. Balance inicial: {self.balance:.2f} USDT")
    
    def update_pnl(self):
        pass
    
    def execute_order(self, order_type: OrderType, price: float, magnitud: float):
        """
        Calcula la cantidad, la ajusta a la precisión correcta y envía la orden.
        """
        # 1. Cálculo de la cantidad
        margen_a_usar = self.balance * self.max_investment_pct * magnitud
        valor_nocional = margen_a_usar * self.leverage
        quantity = valor_nocional / price
        
        # 2. Ajuste de la cantidad a la precisión requerida (redondeo hacia abajo)
        factor = 10 ** self.quantity_precision
        adjusted_quantity = math.floor(quantity * factor) / factor
        
        print(f"Orden calculada: Qty={quantity}, Qty Ajustada={adjusted_quantity}")
        
        # 3. Envío de la orden con la cantidad ajustada
        order_response = self.client.futures_create_order(
            symbol=self.symbol,
            side=order_type.value,
            type='MARKET',
            quantity=adjusted_quantity
        )
        
        # 4. Actualización de estado (simplificada)
        self.current_position = {
            'type': order_type.value,
            'quantity': adjusted_quantity,
            'entry_price': price,
            'margen_usado': margen_a_usar,
            'pasos_en_posicion': 1
        }
        print(f"✅ Orden enviada: {order_type.value} {adjusted_quantity} {self.symbol} a precio ~{price}")
        return order_response
    
    def close_current_position(self, price: float):
        """
        Cierra la posición actual con una orden de mercado.
        Devuelve una tupla (order_response, pnl_neto).
        """
        # Verificar si hay una posición abierta
        if self.current_position is None:
            print("No hay posición actual para cerrar")
            return None, 0.0
        
        # Calcular el PnL antes de cerrar
        entry_price = self.current_position['entry_price']
        quantity = self.current_position['quantity']
        
        if self.current_position['type'] == 'BUY':
            # Posición larga: ganancia si el precio subió
            pnl_neto = (price - entry_price) * quantity
            order_side = 'SELL'
        else:
            # Posición corta: ganancia si el precio bajó
            pnl_neto = (entry_price - price) * quantity
            order_side = 'BUY'
        
        print(f"Cerrando posición {self.current_position['type']} de {quantity} {self.symbol}...")
        print(f"PnL calculado: {pnl_neto:.4f} USDT (Entrada: {entry_price:.4f}, Salida: {price:.4f})")
        
        # Enviar orden de cierre
        order_response = self.client.futures_create_order(
            symbol=self.symbol,
            side=order_side,
            type='MARKET',
            quantity=quantity
        )
        
        # Restablecer la posición current_position
        self.current_position = None
        
        print(f"✅ Posición cerrada exitosamente con orden {order_side} de {quantity} {self.symbol} a precio ~{price}")
        return order_response, pnl_neto
    
    def close_all_positions(self):
        """
        Cierra todas las posiciones abiertas. Método de emergencia (kill switch).
        """
        # Verificar si hay una posición abierta
        if self.current_position is None:
            print("No hay posiciones abiertas para el cierre de emergencia.")
            return
        
        print("Cierre de emergencia activado. Cerrando posición actual...")
        
        # Determinar la order_side opuesta
        if self.current_position['type'] == 'BUY':
            order_side = 'SELL'
        else:
            order_side = 'BUY'
        
        # Obtener la cantidad a cerrar
        quantity = self.current_position['quantity']
        
        # Enviar orden de cierre de emergencia
        order_response = self.client.futures_create_order(
            symbol=self.symbol,
            side=order_side,
            type='MARKET',
            quantity=quantity
        )
        
        print(f"✅ Cierre de emergencia completado: {order_side} {quantity} {self.symbol}")
        
        # Restablecer la posición current_position
        self.current_position = None
        
        return order_response
    
    def get_current_state(self):
        """
        Obtener el estado actual del portafolio para la construcción de la observación.
        Consulta la API de Binance para obtener PnL en tiempo real si hay una posición abierta.
        """
        # Si no hay ninguna posición abierta gestionada internamente, devuelve un estado neutral.
        if not self.current_position:
            return {
                'tipo_posicion': 'NEUTRAL',
                'pnl_no_realizado_roe': 0.0,
                'pasos_en_posicion': 0,
                'precio_entrada': 0.0
            }

        pnl_no_realizado_roe = 0.0
        
        try:
            # Consultar la información de la posición desde la API de Binance
            positions = self.client.futures_position_information(symbol=self.symbol)
            
            if positions:
                position_info = positions[0]
                unrealized_pnl = float(position_info.get('unrealizedProfit', 0.0))
                
                # Calcular ROE usando el margen guardado al abrir la posición
                margen_usado = self.current_position.get('margen_usado', 0.0)
                if margen_usado > 0:
                    pnl_no_realizado_roe = unrealized_pnl / margen_usado
                
        except BinanceAPIException as e:
            print(f"Error de API al obtener PnL para {self.symbol}: {e}. Usando ROE=0.0")
            pnl_no_realizado_roe = 0.0
        except Exception as e:
            print(f"Error inesperado al obtener PnL: {e}. Usando ROE=0.0")
            pnl_no_realizado_roe = 0.0

        # Construir el estado final con los datos actualizados
        return {
            'tipo_posicion': self.current_position['type'],
            'pnl_no_realizado_roe': pnl_no_realizado_roe,
            'pasos_en_posicion': self.current_position.get('pasos_en_posicion', 0),
            'precio_entrada': self.current_position['entry_price']
        }
    
    def get_current_equity(self):
        """
        Obtiene el equity actual del portafolio, incluyendo el PnL no realizado.
        """
        # Si no hay una posición abierta, devolver simplemente el balance
        if self.current_position is None:
            return self.balance
        
        try:
            # Obtener información de la posición desde la API de Binance
            positions = self.client.futures_position_information(symbol=self.symbol)
            
            if positions:
                position_info = positions[0]
                unrealized_pnl = float(position_info.get('unrealizedProfit', 0.0))
                
                # Devolver balance + PnL no realizado
                return self.balance + unrealized_pnl
            else:
                # Si no hay información de posición, devolver balance
                return self.balance
                
        except BinanceAPIException as e:
            print(f"⚠️ Error de API al obtener equity para {self.symbol}: {e}. Usando balance como valor seguro.")
            return self.balance
        except Exception as e:
            print(f"⚠️ Error inesperado al obtener equity: {e}. Usando balance como valor seguro.")
            return self.balance
