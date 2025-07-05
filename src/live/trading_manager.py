import pandas as pd
import torch
import time
import logging
from src.live.data_reader import BinanceLiveDataReader
from src.live.live_data_processor import LiveDataProcessor
from src.configuration.constants import (
    KEY_CONFIG, KEY_ENVIRONMENT, KEY_NORMALIZATION, KEY_STORAGE_MODE, KEY_GCP, 
    STORAGE_MODE_GCP, CONFIG_PATH_DEFAULT, KEY_LINEAGE, KEY_DATA_RUN_ID, 
    FILE_CONFIG_RUN_YAML
)
from src.utils.observation_builder import ObservationBuilder
from src.live.decision_maker import DecisionMaker
from src.live.live_portfolio import LivePortfolio
from src.live.risk_manager import RiskManager
from src.live.telegram_notifier import TelegramNotifier
from src.live.bigquery_logger import BigQueryLogger
from src.training.checkpoint_manager import CheckpointManager
from src.data.artifact_manager import ArtifactManager
from src.entorno.base_portfolio import TipoOperacion
from src.configuration.config_manager import ConfigManager
from src.configuration.secret_utils import SecretManagerUtils
from src.configuration import AppConfig


class LiveTradingManager:
    def __init__(self, run_id: str, mode: str):
        """
        Inicializa el LiveTradingManager con manejo centralizado de configuración y artefactos.
        
        Args:
            run_id: ID del run de entrenamiento a utilizar
            mode: Modo de operación ('testnet' o 'live')
        """
        self.run_id = run_id
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trading_halted = False
        self.logger = logging.getLogger(__name__)

        self.logger.info("--- Iniciando Live Trading Manager ---")
        self.logger.info(f"🔧 Modo de operación: {mode.upper()}")
        self.logger.info(f"🎯 Conectando a: {'Testnet' if mode == 'testnet' else 'Producción'}")

        # --- 1. Cargar configuración del run usando ConfigManager ---
        self.logger.info(f"📦 Cargando configuración para el run_id: {run_id}...")
        
        # Cargar la configuración local solo para obtener los detalles de GCP
        try:
            local_config_obj = AppConfig.from_yaml_file(CONFIG_PATH_DEFAULT)
            gcp_config_local = local_config_obj.gcp.model_dump() if hasattr(local_config_obj, 'gcp') else None
        except FileNotFoundError:
            self.logger.warning("No se encontró config.yaml local. Se asumirá que no se necesita gcp_config.")
            gcp_config_local = None

        self.run_config = ConfigManager.load_training_run_config(run_id, gcp_config=gcp_config_local)
        if not self.run_config:
            raise RuntimeError(f"No se pudo cargar la configuración para el run_id: {run_id}")
        self.logger.info("✅ Configuración del run cargada exitosamente.")

        # Convertir la configuración cargada a objeto Pydantic para el uso en el sistema
        self.app_config = AppConfig(**self.run_config.get(KEY_CONFIG, {}))
        
        # Extraer data_run_id desde la configuración del run
        try:
            self.data_run_id = self.run_config[KEY_LINEAGE][KEY_DATA_RUN_ID]
            self.logger.info(f"  - Data Run ID: {self.data_run_id}")
        except KeyError:
            raise RuntimeError(f"El {FILE_CONFIG_RUN_YAML} para '{run_id}' no contiene '{KEY_LINEAGE}.{KEY_DATA_RUN_ID}'.")

        # --- 2. Cargar credenciales desde Google Secret Manager ---
        self.logger.info("🔐 Cargando credenciales y secretos desde Google Secret Manager...")
        is_testnet = (mode == 'testnet')
        
        project_id = self.app_config.gcp.project_id
        if not project_id:
            raise RuntimeError("No se encontró 'project_id' en la configuración de GCP.")
            
        secret_manager = SecretManagerUtils(project_id=project_id)
        secrets_config = self.app_config.gcp.secrets

        try:
            if is_testnet:
                api_key_secret_id = secrets_config.testnet_binance_api_key_futures
                api_secret_secret_id = secrets_config.testnet_binance_api_secret_futures
            else:
                api_key_secret_id = secrets_config.binance_api_key_futures
                api_secret_secret_id = secrets_config.binance_api_secret_futures

            self.api_key = secret_manager.get_secret(api_key_secret_id)
            self.api_secret = secret_manager.get_secret(api_secret_secret_id)
            self.logger.info(f"✅ Credenciales de Binance {'testnet' if is_testnet else 'producción'} cargadas.")

            self.telegram_bot_token = secret_manager.get_secret(secrets_config.telegram_bot_token)
            self.telegram_chat_id = secret_manager.get_secret(secrets_config.telegram_chat_id)
            self.logger.info("✅ Credenciales de Telegram cargadas.")

        except (KeyError, RuntimeError) as e:
            raise RuntimeError(f"Error al cargar secretos: {e}")

        # --- 3. Inicializar managers centralizados ---
        main_config = self.run_config.get(KEY_CONFIG, {})
        if not main_config:
            raise ValueError(f"No se encontró la configuración principal en 'config' para el run_id: {run_id}")
        
        self.env_config = main_config[KEY_ENVIRONMENT]

        storage_mode = main_config.get(KEY_NORMALIZATION, {}).get(KEY_STORAGE_MODE, 'local')
        gcp_config = main_config.get(KEY_GCP) if storage_mode == STORAGE_MODE_GCP else None

        self.checkpoint_manager = CheckpointManager(
            storage_mode=storage_mode,
            gcp_config=gcp_config
        )
        self.artifact_manager = ArtifactManager(
            storage_mode=storage_mode,
            gcp_config=gcp_config
        )
        self.logger.info(f"✅ Managers especializados inicializados en modo '{storage_mode}'.")

        # --- 4. Cargar artefactos de datos usando ArtifactManager ---
        self.logger.info(f"📦 Cargando artifacts de datos desde data_run_id: {self.data_run_id}...")
        try:
            _, self.scaler, self.price_scaler = self.artifact_manager.load_data_artifacts(self.data_run_id)
            self.logger.info("✅ Scalers cargados exitosamente desde data artifacts.")
        except Exception as e:
            raise RuntimeError(f"Error cargando artifacts de datos: {e}")

        # --- 5. Cargar metadatos del data_run para obtener parámetros ---
        self.logger.info(f"📋 Cargando metadatos del data_run para obtener parámetros de datos...")
        try:
            data_run_metadata = self.artifact_manager.load_data_run_metadata(self.data_run_id)
            experiment_params = data_run_metadata['experiment_parameters']
            self.symbol = experiment_params['symbol']
            self.interval = experiment_params['interval']
            self.logger.info(f"✅ Parámetros de datos cargados desde data_run:")
            self.logger.info(f"  - Símbolo: {self.symbol}")
            self.logger.info(f"  - Intervalo: {self.interval}")
        except Exception as e:
            raise RuntimeError(f"Error cargando metadatos del data_run: {e}")

        # --- 6. Inicializar componentes con scalers y managers inyectados ---
        self._initialize_components()

    def _initialize_components(self):
        """Inicializa todos los componentes del trading manager."""
        main_config = self.run_config.get(KEY_CONFIG, {})
        
        # Initialize components with injected scalers and managers
        self.observation_builder = ObservationBuilder(self.scaler, self.price_scaler, self.run_config)
        self.data_processor = LiveDataProcessor(self.run_config)
        self.decision_maker = DecisionMaker(
            scaler=self.scaler, 
            checkpoint_manager=self.checkpoint_manager, 
            run_config=self.run_config, 
            device=self.device, 
            run_id=self.run_id
        )

        # Initialize Telegram notifier
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                self.notifier = TelegramNotifier(bot_token=self.telegram_bot_token, chat_id=self.telegram_chat_id)
                self.logger.info("✅ TelegramNotifier inicializado correctamente.")
            except Exception as e:
                self.logger.warning(f"⚠️  Advertencia: No se pudo inicializar TelegramNotifier. {e}")
                self.notifier = None
        else:
            self.logger.warning("⚠️  Advertencia: Credenciales de Telegram no proporcionadas. TelegramNotifier deshabilitado.")
            self.notifier = None
        
        # Initialize portfolio
        is_testnet = (self.mode == 'testnet')
        self.portfolio = LivePortfolio(
            api_key=self.api_key, 
            api_secret=self.api_secret, 
            is_testnet=is_testnet, 
            symbol=self.symbol,
            portfolio_config=self.env_config
        )

        # Initialize risk manager
        self.risk_manager = RiskManager(
            portfolio_manager=self.portfolio, 
            risk_config=self.env_config
        )

        # Initialize data reader
        live_config = main_config.get('live_trading', {})
        warm_up_candles = live_config.get('warm_up_candles', 200)
        self.data_reader = BinanceLiveDataReader(self.symbol, self.interval, self, warm_up_candles)
        
        # Initialize BigQuery logger
        try:
            project_id = main_config.get('gcp', {}).get('project_id')
            if project_id:
                dataset_id = live_config.get('bigquery_dataset_id', 'trading_logs')
                table_id = live_config.get('bigquery_table_id', 'live_trading_log')
                self.bq_logger = BigQueryLogger(project_id=project_id, dataset_id=dataset_id, table_id=table_id)
                self.logger.info("✅ BigQueryLogger inicializado correctamente")
            else:
                self.logger.warning("⚠️ project_id no encontrado en la configuración del run. BigQueryLogger deshabilitado.")
                self.bq_logger = None
        except Exception as e:
            self.logger.warning(f"⚠️ Error al inicializar BigQueryLogger: {e}")
            self.bq_logger = None
        
        # Send startup notification
        if self.notifier:
            self.notifier.send_message(f"✅ Bot de Trading INICIADO\nRun ID: {self.run_id}\nSímbolo: {self.symbol}\nModo: {self.mode.upper()}")
        
        self.logger.info("--- Todos los componentes inicializados ---")

    def run(self):
        """Inicia el sistema de trading en vivo."""
        self.logger.info("🚀 Iniciando el estado del portfolio...")
        self.portfolio.reset()

        self.risk_manager.update_state(self.portfolio.balance)

        self.logger.info("📡 Iniciando el flujo de datos en vivo...")
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
