import pandas as pd
import torch
import time
from src.live.data_reader import BinanceLiveDataReader
from src.live.observation_builder import LiveObservationBuilder
from src.live.decision_maker import DecisionMaker
from src.live.portfolio_manager import LivePortfolioManager, OrderType
from src.live.risk_manager import RiskManager
from src.live.telegram_notifier import TelegramNotifier
from src.training.run_manager import RunManager
from src.configuration.config import config


class LiveTradingManager:
    def __init__(self, run_id: str, symbol: str):
        self.run_id = run_id
        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trading_halted = False

        print("--- Iniciando Live Trading Manager ---")

        # --- Inicialización de Componentes ---
        # (Aquí irían las claves de Telegram, en una versión final desde un gestor de secretos)
        # bot_token = "TU_BOT_TOKEN"
        # chat_id = "TU_CHAT_ID"
        # self.notifier = TelegramNotifier(bot_token, chat_id)
        self.notifier = None # Desactivado por ahora para no requerir claves

        # ... (resto de la inicialización de componentes como estaba) ...
        run_manager = RunManager()
        run_manager.set_run_context(run_id=self.run_id)
        self.observation_builder = LiveObservationBuilder(run_manager)
        self.decision_maker = DecisionMaker(self.run_id, self.device)
        # ... etc ...
        api_key = config.binance_api_key
        api_secret = config.binance_api_secret
        is_testnet = config.is_testnet
        self.portfolio_manager = LivePortfolioManager(api_key, api_secret, is_testnet, self.symbol)
        max_drawdown = config.max_drawdown_configurado_cuenta
        self.risk_manager = RiskManager(self.portfolio_manager, max_drawdown)
        warm_up_candles = 200
        self.data_reader = BinanceLiveDataReader(self.symbol, '1h', self, warm_up_candles)
        
        if self.notifier:
            self.notifier.send_message(f"✅ Bot de Trading INICIADO\nRun ID: {self.run_id}\nSímbolo: {self.symbol}")
        
        print("--- Todos los componentes inicializados ---")

    def run(self):
        print("Iniciando el estado del portfolio...")
        self.portfolio_manager.initialize_state()

        # Inicializar el RiskManager con el balance inicial
        self.risk_manager.update_state(self.portfolio_manager.balance)

        print("Iniciando el flujo de datos en vivo...")
        self.data_reader.start()

    def on_new_candle(self, live_dataframe: pd.DataFrame):
        print(f"\n--- Nueva Vela Recibida: {live_dataframe.index[-1]} ---")
        
        # Incrementar pasos si hay una posición abierta
        if self.portfolio_manager.current_position:
            self.portfolio_manager.current_position['pasos_en_posicion'] += 1

        # 1. Actualizar estado de riesgo y PNL no realizado (lógica de PNL pendiente)
        current_equity = self.portfolio_manager.get_current_equity()
        self.risk_manager.update_state(current_equity)

        # 2. BARRERA DE SEGURIDAD: Comprobar si el riesgo ha superado el umbral
        if self.is_trading_halted or self.risk_manager.is_risk_threshold_exceeded():
            if not self.is_trading_halted:
                print("🚨 KILL SWITCH ACTIVADO POR RIESGO 🚨 - Se cerrarán todas las posiciones y se detendrá la operativa.")
                # En una versión final, aquí se llamaría a portfolio_manager.close_all_positions()
                self.is_trading_halted = True
            else:
                print("La operativa está detenida por riesgo. No se tomarán más acciones.")
            return

        # 3. Obtener el estado del portafolio
        live_portfolio_state = self.portfolio_manager.get_current_state()

        # 4. Construir la observación
        observation_vector = self.observation_builder.build(live_market_dataframe=live_dataframe, live_portfolio_state=live_portfolio_state)
        
        # 5. Tomar una decisión
        action = self.decision_maker.get_action(observation_vector)
        print(f"Decisión del agente: {action:.4f}")
        
        # 6. Interpretar y ejecutar la acción
        # Lógica de ejecución de órdenes
        zona_muerta = config.zona_muerta_mantener
        price = live_dataframe['Close'].iloc[-1]
        
        # Lógica de Intención
        if action > zona_muerta:
            intencion = OrderType.BUY
            magnitud_efectiva = (action - zona_muerta) / (1.0 - zona_muerta)
        elif action < -zona_muerta:
            intencion = OrderType.SELL
            magnitud_efectiva = (abs(action) - zona_muerta) / (1.0 - zona_muerta)
        else:
            magnitud_efectiva = 0.0
            print("Acción en zona muerta, manteniendo posición.")
            return
        
        print(f"Intención interpretada: {intencion.value} (precio actual: {price:.2f})")
        
        # Lógica de Ejecución
        posicion_actual = self.portfolio_manager.current_position
        
        if posicion_actual is None:
            # No hay posición, abrimos una nueva si hay intención
            print(f"Ejecutando nueva orden de {intencion.value}...")
            order = self.portfolio_manager.execute_order(intencion, price, magnitud_efectiva)
            if self.notifier:
                self.notifier.send_message(f"📈 NUEVA POSICIÓN: {intencion.value}\nCantidad: {order['origQty'] if order and 'origQty' in order else 'N/A'}\nPrecio: ~{price:.2f}")
        else:
            # Ya hay una posición abierta
            tipo_posicion_actual = OrderType(posicion_actual['type'])
            if intencion != tipo_posicion_actual:
                # La intención es opuesta a la posición actual -> cerrar
                print(f"Intención opuesta detectada. Cerrando posición de {tipo_posicion_actual.value}...")
                order = self.portfolio_manager.close_current_position(price)
                if self.notifier:
                    self.notifier.send_message(f"📉 POSICIÓN CERRADA: {tipo_posicion_actual.value}\nCantidad: {order['origQty'] if order and 'origQty' in order else 'N/A'}\nPrecio: ~{price:.2f}")
            else:
                # La intención es la misma que la posición actual -> mantener
                print(f"Intención coincide con la posición actual de {tipo_posicion_actual.value}. Manteniendo.")
