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


class LiveTradingManager:
    def __init__(self, run_id: str, symbol: str, interval: str, mode: str, run_config: dict, 
                 api_key: str, api_secret: str, telegram_bot_token: str = None, telegram_chat_id: str = None):
        """
        Initialize LiveTradingManager with explicit credentials.
        
        Args:
            run_id: ID of the training run to use
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Trading interval (e.g., '1h')
            mode: Trading mode ('testnet' or 'production')
            run_config: Configuration dictionary from the training run
            api_key: Binance API key
            api_secret: Binance API secret
            telegram_bot_token: Telegram bot token (optional)
            telegram_chat_id: Telegram chat ID (optional)
        """
        self.run_id = run_id
        self.symbol = symbol
        self.interval = interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trading_halted = False
        self.run_config = run_config
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

        print("--- Iniciando Live Trading Manager ---")
        print(f"🔧 Modo de operación: {mode.upper()}")
        print(f"🎯 Conectando a: {'Testnet' if mode == 'testnet' else 'Producción'}")

        # Extraer la configuración principal del run
        main_config = self.run_config.get('config', {})
        if not main_config:
            raise ValueError(f"No se encontró la configuración principal en 'config' para el run_id: {run_id}")
        
        # Extraer la configuración específica del entorno y hacerla disponible como atributo de instancia
        self.env_config = main_config['environment']
        
        print("✅ Configuración del run cargada exitosamente.")

        # --- Inicialización de Componentes con Configuración del Run ---
        main_config = self.run_config.get('config', {})
        storage_mode = main_config.get('normalization', {}).get('storage_mode', 'local')
        gcp_config = main_config.get('gcp') if storage_mode == 'gcp' else None

        # 1. Crear una única instancia de RunManager
        self.run_manager = RunManager(
            run_id=self.run_id,
            storage_mode=storage_mode,
            gcp_config=gcp_config
        )
        print(f"RunManager inicializado en modo '{storage_mode}'.")

        # 2. Inyectar RunManager en los componentes que lo necesitan
        self.observation_builder = LiveObservationBuilder(self.run_manager, self.run_config)
        self.decision_maker = DecisionMaker(self.run_manager, self.run_config, self.device)

        # 3. Notificador de Telegram (usa credenciales inyectadas)
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                self.notifier = TelegramNotifier(bot_token=self.telegram_bot_token, chat_id=self.telegram_chat_id)
                print("✅ TelegramNotifier inicializado correctamente.")
            except Exception as e:
                print(f"⚠️  Advertencia: No se pudo inicializar TelegramNotifier. {e}")
                self.notifier = None
        else:
            print("⚠️  Advertencia: Credenciales de Telegram no proporcionadas. TelegramNotifier deshabilitado.")
            self.notifier = None
        
        # Portfolio Manager (usa credenciales inyectadas)
        is_testnet = (mode == 'testnet')
        self.portfolio_manager = LivePortfolioManager(
            api_key=self.api_key, 
            api_secret=self.api_secret, 
            is_testnet=is_testnet, 
            symbol=self.symbol,
            portfolio_config=self.env_config # Inyección de configuración
        )

        # Risk Manager (usa config del run)
        self.risk_manager = RiskManager(
            portfolio_manager=self.portfolio_manager, 
            risk_config=self.env_config # Inyección de configuración
        )

        # Data Reader
        live_config = main_config.get('live_trading', {})
        warm_up_candles = live_config.get('warm_up_candles', 200)
        self.data_reader = BinanceLiveDataReader(self.symbol, self.interval, self, warm_up_candles)
        
        # BigQuery Logger (usa configuración del run)
        try:
            project_id = main_config.get('gcp', {}).get('project_id')
            if project_id:
                dataset_id = live_config.get('bigquery_dataset_id', 'trading_logs')
                table_id = live_config.get('bigquery_table_id', 'live_trading_log')
                self.bq_logger = BigQueryLogger(project_id=project_id, dataset_id=dataset_id, table_id=table_id)
                print("✅ BigQueryLogger inicializado correctamente")
            else:
                print("⚠️ project_id no encontrado en la configuración del run. BigQueryLogger deshabilitado.")
                self.bq_logger = None
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
        
        zona_muerta = self.env_config.get('zona_muerta_mantener', 0.05)  # Default value if not found
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
