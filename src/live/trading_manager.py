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
        max_consecutive_losses = config.max_consecutive_losses
        self.risk_manager = RiskManager(self.portfolio_manager, max_drawdown, max_consecutive_losses)
        warm_up_candles = 200
        self.data_reader = BinanceLiveDataReader(self.symbol, '1h', self, warm_up_candles)
        
        # --- BigQuery Logger Initialization ---
        try:
            project_id = getattr(config, 'gcp_project_id', 'your-project-id')
            dataset_id = getattr(config, 'bigquery_dataset_id', 'btcbot_dataset')
            self.bq_logger = BigQueryLogger(project_id=project_id, dataset_id=dataset_id)
            print("✅ BigQueryLogger inicializado correctamente")
        except Exception as e:
            print(f"⚠️ Error al inicializar BigQueryLogger: {e}")
            print("El bot continuará funcionando sin logging a BigQuery")
            self.bq_logger = None
        
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
