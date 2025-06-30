import pandas as pd
import torch
import time
from src.live.data_reader import BinanceLiveDataReader
from src.live.observation_builder import LiveObservationBuilder
from src.live.decision_maker import DecisionMaker
from src.live.portfolio_manager import LivePortfolioManager, OrderType
from src.live.risk_manager import RiskManager
from src.live.telegram_notifier import TelegramNotifier
from src.live.bigquery_logger import BigQueryLogger
from src.training.run_manager import RunManager
from src.configuration.config import config


class LiveTradingManager:
    def __init__(self, run_id: str, symbol: str, mode: str):
        self.run_id = run_id
        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trading_halted = False

        print("--- Iniciando Live Trading Manager ---")
        print(f"🔧 Modo de operación: {mode.upper()}")
        print(f"🎯 Conectando a: {'Testnet' if mode == 'testnet' else 'Producción'}")

        # --- Carga de Configuración Centralizada ---
        print(f"🔩 Cargando configuración para el run_id: {self.run_id}")
        run_manager = RunManager()
        run_manager.set_run_context(run_id=self.run_id)
        run_config = run_manager.download_and_load_yaml_config(self.run_id)
        if run_config is None:
            raise ValueError(f"No se pudo cargar la configuración para el run_id: '{self.run_id}'")
        
        # Extraer las configuraciones específicas del snapshot del entrenamiento
        env_config = run_config['config_snapshot']['environment']
        
        print("✅ Configuración del run cargada exitosamente.")

        # --- Inicialización de Componentes con Configuración del Run ---
        
        # Notificador de Telegram (usa credenciales globales)
        try:
            bot_token = config.telegram_bot_token
            chat_id = config.telegram_chat_id
            self.notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
            print("✅ TelegramNotifier inicializado correctamente.")
        except Exception as e:
            print(f"⚠️  Advertencia: No se pudo inicializar TelegramNotifier. {e}")
            self.notifier = None

        # Observation Builder (ahora pasamos el run_manager que ya tiene el contexto)
        self.observation_builder = LiveObservationBuilder(run_manager)

        # Decision Maker
        self.decision_maker = DecisionMaker(self.run_id, self.device)
        
        # Portfolio Manager (usa credenciales globales y config del run)
        api_key = config.binance_api_key
        api_secret = config.binance_api_secret
        is_testnet = (mode == 'testnet')
        self.portfolio_manager = LivePortfolioManager(
            api_key=api_key, 
            api_secret=api_secret, 
            is_testnet=is_testnet, 
            symbol=self.symbol,
            # Pasamos los parámetros de trading desde la config del run
            leverage=env_config['apalancamiento'],
            max_investment_pct=env_config['porcentaje_max_inversion_por_trade']
        )

        # Risk Manager (usa config del run)
        self.risk_manager = RiskManager(
            portfolio_manager=self.portfolio_manager, 
            # Pasamos los parámetros de riesgo desde la config del run
            max_drawdown_pct=env_config['max_drawdown_configurado_cuenta'], 
            max_consecutive_losses=env_config.get('max_consecutive_losses', 5) # Usamos .get con default para compatibilidad con runs antiguos
        )

        # Data Reader
        warm_up_candles = 200 # Este valor puede ser más estático
        self.data_reader = BinanceLiveDataReader(self.symbol, '1h', self, warm_up_candles)
        
        # BigQuery Logger (usa credenciales globales)
        try:
            project_id = config.project_id
            dataset_id = 'trading_logs' # Podríamos añadir esto a config.yaml global
            self.bq_logger = BigQueryLogger(project_id=project_id, dataset_id=dataset_id)
            print("✅ BigQueryLogger inicializado correctamente")
        except Exception as e:
            print(f"⚠️ Error al inicializar BigQueryLogger: {e}")
            self.bq_logger = None
        
        # Notificación de inicio
        if self.notifier:
            self.notifier.send_message(f"✅ Bot de Trading INICIADO\nRun ID: {self.run_id}\nSímbolo: {self.symbol}\nModo: {mode.upper()}")
        
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
        
        log_data = {
            'run_id': self.run_id,
            'symbol': self.symbol,
            'candle_timestamp': live_dataframe.index[-1].to_pydatetime()
        }
        
        if self.portfolio_manager.current_position:
            self.portfolio_manager.current_position['pasos_en_posicion'] += 1

        current_equity = self.portfolio_manager.get_current_equity()
        self.risk_manager.update_state(current_equity)

        # Calcular y registrar datos de riesgo
        drawdown_pct = 0.0
        if self.risk_manager.max_equity_so_far > 0:
            drawdown_pct = (self.risk_manager.max_equity_so_far - current_equity) / self.risk_manager.max_equity_so_far
        
        log_data['account_balance'] = self.portfolio_manager.balance
        log_data['account_equity'] = current_equity
        log_data['max_equity'] = self.risk_manager.max_equity_so_far
        log_data['drawdown_pct'] = drawdown_pct
        log_data['consecutive_losses'] = self.risk_manager.consecutive_losses_counter

        if self.is_trading_halted or self.risk_manager.is_risk_threshold_exceeded():
            if not self.is_trading_halted:
                print("🚨 KILL SWITCH ACTIVADO POR RIESGO 🚨 - Se cerrarán todas las posiciones y se detendrá la operativa.")
                self.portfolio_manager.close_all_positions()
                self.is_trading_halted = True
            else:
                print("La operativa está detenida por riesgo. No se tomarán más acciones.")
            
            log_data.update({
                'agent_action': 0.0,
                'interpreted_intent': "HALTED_BY_RISK",
                'trade_executed': False,
                'position_status': 'HALTED'
            })
            if self.bq_logger:
                self.bq_logger.log_step_data(log_data)
            return

        live_portfolio_state = self.portfolio_manager.get_current_state()
        log_data.update({
            'position_status': live_portfolio_state['tipo_posicion'],
            'position_pnl_roe': live_portfolio_state['pnl_no_realizado_roe'],
            'position_duration': live_portfolio_state['pasos_en_posicion'],
            'position_entry_price': live_portfolio_state['precio_entrada']
        })

        observation_vector = self.observation_builder.build(live_market_dataframe=live_dataframe, live_portfolio_state=live_portfolio_state)
        
        action = self.decision_maker.get_action(observation_vector)
        print(f"Decisión del agente: {action:.4f}")
        
        price = live_dataframe['Close'].iloc[-1]
        log_data['market_price'] = price
        log_data['agent_action'] = action
        
        zona_muerta = config.zona_muerta_mantener
        trade_executed = False

        if action > zona_muerta:
            intencion = OrderType.BUY
            magnitud_efectiva = (action - zona_muerta) / (1.0 - zona_muerta)
            log_data['interpreted_intent'] = "BUY"
        elif action < -zona_muerta:
            intencion = OrderType.SELL
            magnitud_efectiva = (abs(action) - zona_muerta) / (1.0 - zona_muerta)
            log_data['interpreted_intent'] = "SELL"
        else:
            log_data['interpreted_intent'] = "HOLD"
            log_data['trade_executed'] = False
            print("Acción en zona muerta, manteniendo posición.")
            if self.bq_logger:
                self.bq_logger.log_step_data(log_data)
            return
            
        print(f"Intención interpretada: {intencion.value} (precio actual: {price:.2f})")
        
        posicion_actual = self.portfolio_manager.current_position
        
        if posicion_actual is None:
            print(f"Ejecutando nueva orden de {intencion.value}...")
            order = self.portfolio_manager.execute_order(intencion, price, magnitud_efectiva)
            trade_executed = order is not None
            if self.notifier:
                self.notifier.send_message(f"📈 NUEVA POSICIÓN: {intencion.value}\nCantidad: {order['origQty'] if order and 'origQty' in order else 'N/A'}\nPrecio: ~{price:.2f}")
        else:
            tipo_posicion_actual = OrderType(posicion_actual['type'])
            if intencion != tipo_posicion_actual:
                print(f"Intención opuesta detectada. Cerrando posición de {tipo_posicion_actual.value}...")
                order, pnl_realizado = self.portfolio_manager.close_current_position(price)
                trade_executed = order is not None
                self.risk_manager.register_trade(pnl_realizado)
                if self.notifier:
                    self.notifier.send_message(f"📉 POSICIÓN CERRADA: {tipo_posicion_actual.value}\nCantidad: {order['origQty'] if order and 'origQty' in order else 'N/A'}\nPrecio: ~{price:.2f}")
            else:
                print(f"Intención coincide con la posición actual de {tipo_posicion_actual.value}. Manteniendo.")
                trade_executed = False
                
        log_data['trade_executed'] = trade_executed
        
        if self.bq_logger:
            self.bq_logger.log_step_data(log_data)
