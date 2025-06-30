from binance.client import Client
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
                        precision = int(round(-math.log(step_size, 10), 0))
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
    
    def execute_order(self, order_type: OrderType, price: float):
        """
        Calcula la cantidad, la ajusta a la precisión correcta y envía la orden.
        """
        # 1. Cálculo de la cantidad
        margen_a_usar = self.balance * self.max_investment_pct
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
            'entry_price': price
        }
        print(f"✅ Orden enviada: {order_type.value} {adjusted_quantity} {self.symbol} a precio ~{price}")
        return order_response
    
    def close_current_position(self, price: float):
        """
        Cierra la posición actual con una orden de mercado.
        """
        # Verificar si hay una posición abierta
        if self.current_position is None:
            print("No hay posición actual para cerrar")
            return
        
        # Determinar la order_side opuesta
        if self.current_position['type'] == 'BUY':
            order_side = 'SELL'
        else:
            order_side = 'BUY'
        
        # Obtener la cantidad a cerrar
        quantity = self.current_position['quantity']
        
        print(f"Cerrando posición {self.current_position['type']} de {quantity} {self.symbol}...")
        
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
        return order_response
    
    def get_current_state(self):
        """
        Obtener el estado actual del portafolio para la construcción de la observación.
        
        Returns:
            dict: Diccionario con el estado actual del portafolio incluyendo tipo de posición,
                  PnL no realizado, pasos en posición y precio de entrada.
        """
        return {
            'tipo_posicion': self.current_position['type'] if self.current_position else 'NEUTRAL',
            'pnl_no_realizado_roe': 0.0,
            'pasos_en_posicion': 0,
            'precio_entrada': self.current_position['entry_price'] if self.current_position else 0.0
        }
