import pandas as pd
import torch
import time
from src.live.data_reader import BinanceLiveDataReader
from src.live.live_data_processor import LiveDataProcessor
from src.live.observation_builder import LiveObservationBuilder
from src.live.decision_maker import DecisionMaker
from src.live.live_portfolio import LivePortfolio
from src.live.risk_manager import RiskManager
from src.live.telegram_notifier import TelegramNotifier
from src.live.bigquery_logger import BigQueryLogger
from src.training.checkpoint_manager import CheckpointManager
from src.data.artifact_manager import ArtifactManager
from src.entorno.base_portfolio import TipoOperacion


class LiveTradingManager:
    def __init__(self, run_id: str, mode: str, run_config: dict, 
                 data_run_id: str, api_key: str, api_secret: str, telegram_bot_token: str = None, 
                 telegram_chat_id: str = None):
        self.run_id = run_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trading_halted = False
        self.run_config = run_config
        self.data_run_id = data_run_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

        print("--- Iniciando Live Trading Manager ---")
        print(f"🔧 Modo de operación: {mode.upper()}")
        print(f"🎯 Conectando a: {'Testnet' if mode == 'testnet' else 'Producción'}")

        main_config = self.run_config.get('config', {})
        if not main_config:
            raise ValueError(f"No se encontró la configuración principal en 'config' para el run_id: {run_id}")
        
        self.env_config = main_config['environment']
        print("✅ Configuración del run cargada exitosamente.")

        storage_mode = main_config.get('normalization', {}).get('storage_mode', 'local')
        gcp_config = main_config.get('gcp') if storage_mode == 'gcp' else None

        self.checkpoint_manager = CheckpointManager(
            storage_mode=storage_mode,
            gcp_config=gcp_config
        )
        self.artifact_manager = ArtifactManager(
            storage_mode=storage_mode,
            gcp_config=gcp_config
        )
        print(f"Managers especializados inicializados en modo '{storage_mode}'.")

        # Load data artifacts (scaler and price_scaler) from the data_run_id
        print(f"Cargando artifacts de datos desde data_run_id: {self.data_run_id}...")
        try:
            _, self.scaler, self.price_scaler = self.artifact_manager.load_data_artifacts(self.data_run_id)
            print("✅ Scalers cargados exitosamente desde data artifacts.")
        except Exception as e:
            raise RuntimeError(f"Error cargando artifacts de datos: {e}")

        # Load data_run metadata to get symbol and interval (single source of truth)
        print(f"Cargando metadatos del data_run para obtener parámetros de datos...")
        try:
            data_run_metadata = self.artifact_manager.load_data_run_metadata(self.data_run_id)
            experiment_params = data_run_metadata['experiment_parameters']
            self.symbol = experiment_params['symbol']
            self.interval = experiment_params['interval']
            print(f"✅ Parámetros de datos cargados desde data_run:")
            print(f"  - Símbolo: {self.symbol}")
            print(f"  - Intervalo: {self.interval}")
        except Exception as e:
            raise RuntimeError(f"Error cargando metadatos del data_run: {e}")

        # Initialize components with injected scalers
        self.observation_builder = LiveObservationBuilder(self.scaler, self.price_scaler, self.run_config)
        self.data_processor = LiveDataProcessor(self.run_config)
        self.decision_maker = DecisionMaker(self.scaler, self.checkpoint_manager, self.run_config, self.device, self.run_id)

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
        
        is_testnet = (mode == 'testnet')
        self.portfolio = LivePortfolio(
            api_key=self.api_key, 
            api_secret=self.api_secret, 
            is_testnet=is_testnet, 
            symbol=self.symbol,
            portfolio_config=self.env_config
        )

        self.risk_manager = RiskManager(
            portfolio_manager=self.portfolio, 
            risk_config=self.env_config
        )

        live_config = main_config.get('live_trading', {})
        warm_up_candles = live_config.get('warm_up_candles', 200)
        self.data_reader = BinanceLiveDataReader(self.symbol, self.interval, self, warm_up_candles)
        
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
        
        if self.notifier:
            self.notifier.send_message(f"✅ Bot de Trading INICIADO\nRun ID: {self.run_id}\nSímbolo: {self.symbol}\nModo: {mode.upper()}")
        
        print("--- Todos los componentes inicializados ---")

    def run(self):
        print("Iniciando el estado del portfolio...")
        self.portfolio.reset()

        self.risk_manager.update_state(self.portfolio.balance)

        print("Iniciando el flujo de datos en vivo...")
        self.data_reader.start()

    def on_new_candle(self, live_dataframe: pd.DataFrame):
        print(f"\n--- Nueva Vela Recibida: {live_dataframe.index[-1]} ---")
        
        # Sincronizar el balance al inicio de cada ciclo
        self.portfolio.sync_balance()

        log_data = {
            'run_id': self.run_id,
            'symbol': self.symbol,
            'candle_timestamp': live_dataframe.index[-1].to_pydatetime().isoformat()
        }
        
        self.portfolio.advance_step()

        current_equity = self.portfolio.equity
        self.risk_manager.update_state(current_equity)

        drawdown_pct = 0.0
        if self.risk_manager.max_equity_so_far > 0:
            drawdown_pct = (self.risk_manager.max_equity_so_far - current_equity) / self.risk_manager.max_equity_so_far
        
        log_data['account_balance'] = self.portfolio.balance
        log_data['account_equity'] = current_equity
        log_data['max_equity'] = self.risk_manager.max_equity_so_far
        log_data['drawdown_pct'] = drawdown_pct
        log_data['consecutive_losses'] = self.risk_manager.consecutive_losses_counter

        if self.is_trading_halted or self.risk_manager.is_risk_threshold_exceeded():
            if not self.is_trading_halted:
                print("🚨 KILL SWITCH ACTIVADO POR RIESGO 🚨 - Se cerrarán todas las posiciones y se detendrá la operativa.")
                # self.portfolio.close_all_positions() # Este método necesita ser re-implementado en LivePortfolio
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

        live_portfolio_state = self.portfolio.get_current_state()
        log_data.update({
            'position_status': live_portfolio_state['tipo'].name,
            'position_pnl_roe': live_portfolio_state['pnl_no_realizado_roe'],
            'position_duration': live_portfolio_state['pasos_en_posicion'],
            'position_entry_price': live_portfolio_state['precio_entrada']
        })

        # Procesar los datos crudos antes de construir la observación
        try:
            processed_dataframe = self.data_processor.process(live_dataframe)
        except ValueError as e:
            print(f"❌ Error crítico en el procesamiento de datos: {e}")
            print("🛑 Deteniendo operativa para prevenir decisiones con datos corruptos.")
            self.is_trading_halted = True
            
            log_data.update({
                'agent_action': 0.0,
                'interpreted_intent': "HALTED_BY_DATA_ERROR",
                'trade_executed': False,
                'position_status': 'HALTED'
            })
            if self.bq_logger:
                self.bq_logger.log_step_data(log_data)
            return

        observation_vector = self.observation_builder.build(live_market_dataframe=processed_dataframe, live_portfolio_state=live_portfolio_state)
        
        action = self.decision_maker.get_action(observation_vector)
        print(f"Decisión del agente: {action:.4f}")
        
        price = live_dataframe['Close'].iloc[-1]
        log_data['market_price'] = price
        log_data['agent_action'] = action
        
        zona_muerta = self.env_config.get('zona_muerta_mantener', 0.05)

        if action > zona_muerta:
            intencion = "COMPRAR"
            magnitud = (action - zona_muerta) / (1.0 - zona_muerta)
        elif action < -zona_muerta:
            intencion = "VENDER"
            magnitud = (abs(action) - zona_muerta) / (1.0 - zona_muerta)
        else:
            intencion = "MANTENER"
            magnitud = 0

        log_data['interpreted_intent'] = intencion
        print(f"Intención interpretada: {intencion} (precio actual: {price:.2f}) con magnitud {magnitud:.2f}")

        trade_executed, pnl_realizado = self.portfolio.execute_order(intencion, magnitud, price)
        log_data['trade_executed'] = trade_executed

        if trade_executed:
            if pnl_realizado != 0:
                print(f"Posición cerrada con PNL: {pnl_realizado:.2f}")
                self.risk_manager.register_trade(pnl_realizado)
                if self.notifier:
                    self.notifier.send_message(f"📉 POSICIÓN CERRADA: {self.portfolio.posicion_actual['tipo'].name}\nPNL: {pnl_realizado:.2f} USDT")
            else:
                print(f"Nueva posición abierta: {self.portfolio.posicion_actual['tipo'].name}")
                if self.notifier:
                    self.notifier.send_message(f"📈 NUEVA POSICIÓN: {self.portfolio.posicion_actual['tipo'].name}\nPrecio: ~{price:.2f}")
        else:
            print("Acción no resultó en un trade (misma dirección o zona muerta).")

        if self.bq_logger:
            self.bq_logger.log_step_data(log_data)
