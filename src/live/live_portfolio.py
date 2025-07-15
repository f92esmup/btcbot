from binance.client import Client
from binance.exceptions import BinanceAPIException
import math
import time
from typing import Dict, Any, Tuple, List

from src.entorno.base_portfolio import BasePortfolio, TipoOperacion

class LivePortfolio(BasePortfolio):
    def __init__(self, api_key: str, api_secret: str, is_testnet: bool, symbol: str, 
                 portfolio_config: dict, telegram_notifier=None):
        self.client = Client(api_key, api_secret, testnet=is_testnet)
        self.symbol = symbol
        self.config = portfolio_config
        self._balance = 0.0
        self._current_position = None
        self._historial_trades = []
        self.leverage = self.config['apalancamiento']
        self.max_investment_pct = self.config['porcentaje_max_inversion_por_trade']
        self.quantity_precision = self._get_quantity_precision()
        self.telegram_notifier = telegram_notifier  # Para notificaciones de stop-loss
        self.reset()

    def reset(self):
        print("Inicializando y sincronizando estado del portfolio...")
        
        # --- INICIO DE LA NUEVA LÓGICA ---
        try:
            print(f"Estableciendo apalancamiento a {self.leverage}x para el símbolo {self.symbol}...")
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            print(f"✅ Apalancamiento establecido a {self.leverage}x exitosamente.")
        except BinanceAPIException as e:
            print(f"❌ ERROR CRÍTICO: No se pudo establecer el apalancamiento a {self.leverage}x.")
            print(f"   Respuesta de la API: {e}")
            print("   El bot no puede continuar con una configuración de riesgo incorrecta. Abortando.")
            raise RuntimeError(f"Fallo al establecer apalancamiento: {e}")
        # --- FIN DE LA NUEVA LÓGICA ---

        account_balance = self.client.futures_account_balance()
        for asset in account_balance:
            if asset['asset'] == 'USDT':
                self._balance = float(asset['balance'])
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
        self._current_position = {
            'tipo': TipoOperacion.NEUTRAL,
            'quantity': 0.0,
            'entry_price': 0.0,
            'margen_usado': 0.0,
            'pasos_en_posicion': 0
        }
        print(f"✅ Estado sincronizado. Balance inicial: {self._balance:.2f} USDT")

    def sync_balance(self):
        """
        Sincroniza el balance local con el balance real de la cuenta de Binance.
        """
        try:
            account_balance_info = self.client.futures_account_balance()
            for asset in account_balance_info:
                if asset['asset'] == 'USDT':
                    balance_remoto = float(asset['balance'])
                    if abs(self._balance - balance_remoto) > 0.01: # Comprobar si hay una diferencia significativa
                        print(f"🔄 Sincronizando balance. Local: {self._balance:.2f}, Remoto: {balance_remoto:.2f}. Diferencia: {balance_remoto - self._balance:.2f}")
                        self._balance = balance_remoto
                    return
        except BinanceAPIException as e:
            print(f"⚠️ Error de API al sincronizar balance: {e}. Se usará el balance local.")

    def execute_order(self, intencion: str, magnitud: float, precio: float) -> Tuple[bool, float]:
        posicion_actual_tipo = self._current_position['tipo']

        if intencion == "MANTENER":
            return False, 0.0

        es_operacion_opuesta = (
            (intencion == "VENDER" and posicion_actual_tipo == TipoOperacion.LARGO) or
            (intencion == "COMPRAR" and posicion_actual_tipo == TipoOperacion.CORTO)
        )

        if es_operacion_opuesta:
            _, pnl_realizado = self._close_current_position(precio)
            return True, pnl_realizado

        if posicion_actual_tipo == TipoOperacion.NEUTRAL and magnitud > 0:
            tipo_operacion = TipoOperacion.LARGO if intencion == "COMPRAR" else TipoOperacion.CORTO
            self._open_position(tipo_operacion, precio, magnitud)
            return True, 0.0

        return False, 0.0

    def _open_position(self, tipo_operacion: TipoOperacion, price: float, magnitud: float):
        margen_a_usar = self._balance * self.max_investment_pct * magnitud
        valor_nocional = margen_a_usar * self.leverage
        quantity = valor_nocional / price
        
        factor = 10 ** self.quantity_precision
        adjusted_quantity = math.floor(quantity * factor) / factor
        
        print(f"Orden calculada: Qty={quantity}, Qty Ajustada={adjusted_quantity}")
        
        # 1. Abrir la posición principal
        try:
            order_response = self.client.futures_create_order(
                symbol=self.symbol,
                side=tipo_operacion.name,  # BUY or SELL
                type='MARKET',
                quantity=adjusted_quantity
            )
            
            # 2. Obtener los detalles reales de la ejecución
            time.sleep(0.1)  # Esperar a que la API procese la orden
            trades = self.client.futures_get_user_trades(symbol=self.symbol, limit=1)
            if not trades:
                raise BinanceAPIException("No se pudo obtener información del trade ejecutado")
                
            last_trade = trades[0]
            real_quantity = float(last_trade['qty'])
            real_price = float(last_trade['price'])
            
            # 3. Calcular y colocar el stop-loss
            try:
                stop_loss_pct = self.config['stop_loss_pct']
                if tipo_operacion == TipoOperacion.LARGO:
                    stop_price = real_price * (1 - stop_loss_pct)
                    stop_side = 'SELL'
                else:  # CORTO
                    stop_price = real_price * (1 + stop_loss_pct)
                    stop_side = 'BUY'
                
                stop_order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=stop_side,
                    type='STOP_MARKET',
                    quantity=real_quantity,
                    stopPrice=stop_price,
                    reduceOnly=True  # Asegurar que solo cierra la posición
                )
                
                print(f"✅ Stop-loss colocado: {stop_side} {real_quantity} {self.symbol} @ {stop_price:.2f}")
                
                # 4. Actualizar el estado de la posición con los datos reales
                self._current_position = {
                    'tipo': tipo_operacion,
                    'quantity': real_quantity,
                    'entry_price': real_price,
                    'margen_usado': margen_a_usar,
                    'pasos_en_posicion': 1,
                    'stop_order_id': stop_order['orderId']
                }
                
            except BinanceAPIException as e:
                error_msg = f"❌ ERROR CRÍTICO: No se pudo colocar el stop-loss: {e}"
                print(error_msg)
                print("⚠️ Ejecutando cierre de emergencia de la posición...")
                
                # Notificar el error por Telegram si está configurado
                if self.telegram_notifier:
                    self.telegram_notifier.notify_emergency_closure(
                        symbol=self.symbol,
                        position_type=tipo_operacion.name,
                        error_details=str(e)
                    )
                
                # Cerrar la posición como medida de emergencia
                emergency_close = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=stop_side,  # Usar el mismo lado que habríamos usado para el stop
                    type='MARKET',
                    quantity=real_quantity
                )
                
                print("✅ Posición cerrada por seguridad al fallar el stop-loss")
                
                # Actualizar el historial con el cierre de emergencia
                if self._historial_trades:
                    self._historial_trades[-1]['cierre_por'] = 'EMERGENCY_HALT'
                    
                raise RuntimeError("Fallo al colocar stop-loss, posición cerrada por seguridad")
                
        except BinanceAPIException as e:
            print(f"❌ ERROR CRÍTICO: No se pudo abrir la posición principal: {e}")
            raise  # Re-lanzar la excepción para que sea manejada en niveles superiores
        
        print(f"✅ Posición abierta: {tipo_operacion.name} {real_quantity} {self.symbol} @ {real_price}")
        return order_response

    def _close_current_position(self, price: float):
        if self._current_position is None or self._current_position['tipo'] == TipoOperacion.NEUTRAL:
            print("No hay posición actual para cerrar")
            return None, 0.0
        
        # 1. Cancelar la orden de stop-loss si existe
        stop_order_id = self._current_position.get('stop_order_id')
        if stop_order_id:
            try:
                self.client.futures_cancel_order(
                    symbol=self.symbol,
                    orderId=stop_order_id
                )
                print(f"✅ Orden de stop-loss {stop_order_id} cancelada exitosamente")
            except BinanceAPIException as e:
                # Si la orden ya fue ejecutada o no existe, continuamos
                print(f"⚠️ No se pudo cancelar stop-loss {stop_order_id}: {e}")
        
        quantity = self._current_position['quantity']
        order_side = 'SELL' if self._current_position['tipo'] == TipoOperacion.LARGO else 'BUY'
        
        print(f"Cerrando posición {self._current_position['tipo'].name} de {quantity} {self.symbol} a precio de mercado...")
        
        # 1. Ejecutar la orden de cierre
        order_response = self.client.futures_create_order(
            symbol=self.symbol,
            side=order_side,
            type='MARKET',
            quantity=quantity
        )
        
        # 2. Esperar un momento para que la API procese el trade
        time.sleep(0.1)
        
        # 3. Consultar el último trade para obtener el PnL real
        try:
            last_trade = self.client.futures_get_user_trades(symbol=self.symbol, limit=1)[0]
            pnl_real = float(last_trade.get('realizedPnl', 0.0))
            
            if pnl_real == 0.0:
                print("Advertencia: El PnL real reportado por la API es 0.0. Verifique el historial de trades.")

        except (BinanceAPIException, IndexError) as e:
            print(f"Error crítico: No se pudo obtener el PnL real del último trade: {e}")
            print("El balance puede estar desincronizado. Se recomienda intervención manual.")
            pnl_real = 0.0 # Asumir 0 para evitar fallos, pero loggear el error es crucial

        print(f"PnL Real (API): {pnl_real:.4f} USDT")

        # 4. Actualizar el balance local con el PnL y registrarlo
        balance_anterior = self._balance
        self._balance += pnl_real
        print(f"Balance actualizado: {balance_anterior:.2f} -> {self._balance:.2f} USDT")

        # 5. Actualizar el historial y el estado del portfolio
        self._historial_trades.append({
            'tipo': self._current_position['tipo'].name,
            'pnl_abs': pnl_real 
        })

        self._current_position = {
            'tipo': TipoOperacion.NEUTRAL,
            'quantity': 0.0,
            'entry_price': 0.0,
            'margen_usado': 0.0,
            'pasos_en_posicion': 0
        }
        
        print(f"✅ Posición cerrada exitosamente. Orden: {order_side} {quantity} {self.symbol}")
        return order_response, pnl_real

    def update_state(self, precio_actual: float):
        # En modo live, el estado (como PnL) se obtiene directamente de la API
        # en `get_current_state`, por lo que este método puede no ser necesario.
        pass

    def get_current_state(self) -> Dict[str, Any]:
        if not self._current_position or self._current_position['tipo'] == TipoOperacion.NEUTRAL:
            return {
                'tipo': TipoOperacion.NEUTRAL,
                'pnl_no_realizado_roe': 0.0,
                'pasos_en_posicion': 0,
                'precio_entrada': 0.0
            }

        pnl_no_realizado_roe = 0.0
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            if positions:
                position_info = positions[0]
                unrealized_pnl = float(position_info.get('unrealizedProfit', 0.0))
                margen_usado = self._current_position.get('margen_usado', 0.0)
                if margen_usado > 0:
                    pnl_no_realizado_roe = unrealized_pnl / margen_usado
        except BinanceAPIException as e:
            print(f"Error de API al obtener PnL para {self.symbol}: {e}. Usando ROE=0.0")
            pnl_no_realizado_roe = 0.0

        return {
            'tipo': self._current_position['tipo'],
            'pnl_no_realizado_roe': pnl_no_realizado_roe,
            'pasos_en_posicion': self._current_position.get('pasos_en_posicion', 0),
            'precio_entrada': self._current_position['entry_price']
        }

    def advance_step(self):
        if self._current_position and self._current_position['tipo'] != TipoOperacion.NEUTRAL:
            self._current_position['pasos_en_posicion'] += 1

    @property
    def equity(self) -> float:
        if not self._current_position or self._current_position['tipo'] == TipoOperacion.NEUTRAL:
            return self._balance
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            if positions:
                position_info = positions[0]
                unrealized_pnl = float(position_info.get('unrealizedProfit', 0.0))
                return self._balance + unrealized_pnl
            return self._balance
        except BinanceAPIException as e:
            print(f"⚠️ Error de API al obtener equity para {self.symbol}: {e}. Usando balance como valor seguro.")
            return self._balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def posicion_actual(self) -> Dict[str, Any]:
        return self._current_position

    @property
    def historial_trades(self) -> List[Dict[str, Any]]:
        return self._historial_trades

    def _get_quantity_precision(self) -> int:
        print(f"Obteniendo información de trading para {self.symbol}...")
        exchange_info = self.client.futures_exchange_info()
        for s_info in exchange_info['symbols']:
            if s_info['symbol'] == self.symbol:
                for f in s_info['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        precision = int(round(-math.log(step_size, 10), 0))
                        print(f"Precisión de cantidad para {self.symbol} establecida en: {precision} decimales.")
                        return precision
        raise ValueError(f"No se pudo obtener la precisión de cantidad para {self.symbol}")
