import pandas as pd
import torch
from src.live.data_reader import BinanceLiveDataReader
from src.live.observation_builder import LiveObservationBuilder
from src.live.decision_maker import DecisionMaker
from src.live.portfolio_manager import LivePortfolioManager, OrderType
from src.live.risk_manager import RiskManager
from src.configuration.config import config


class LiveTradingManager:
    def __init__(self, run_id: str, symbol: str):
        self.run_id = run_id
        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("--- Iniciando Live Trading Manager ---")
        
        # Inicializar componentes
        self.observation_builder = LiveObservationBuilder(self.run_id)
        self.decision_maker = DecisionMaker(self.run_id, self.device)
        
        # La inicialización del PortfolioManager necesita API keys
        # (En una versión final, estas vendrían de un gestor de secretos)
        api_key = config.binance_api_key
        api_secret = config.binance_api_secret
        is_testnet = config.is_testnet
        self.portfolio_manager = LivePortfolioManager(api_key, api_secret, is_testnet, self.symbol)
        
        # El DataReader necesita saber quién es su "suscriptor" para notificarle.
        # Pasamos 'self' (la propia instancia del Manager).
        warm_up_candles = 200  # Un valor razonable para la mayoría de indicadores
        self.data_reader = BinanceLiveDataReader(self.symbol, '1h', self, warm_up_candles)
        
        print("--- Todos los componentes inicializados ---")
    
    def run(self):
        """
        Inicia el proceso de trading en vivo.
        """
        print("Iniciando el estado del portfolio...")
        self.portfolio_manager.initialize_state()
        
        print("Iniciando el flujo de datos en vivo... (Esto bloqueará el hilo principal)")
        self.data_reader.start()
    
    def on_new_candle(self, live_dataframe: pd.DataFrame):
        """
        Este es el corazón del bot. Se ejecuta cada vez que el DataReader notifica una nueva vela.
        """
        print(f"\n--- Nueva Vela Recibida: {live_dataframe.index[-1]} ---")
        
        # 1. Construir la observación
        observation_vector = self.observation_builder.build(live_dataframe)
        
        # 2. Tomar una decisión
        action = self.decision_maker.get_action(observation_vector)
        print(f"Observación construida, decisión del agente: {action:.4f}")
        
        # 3. Interpretar y ejecutar la acción
        zona_muerta = config.zona_muerta_mantener
        price = live_dataframe['Close'].iloc[-1]
        
        # Lógica de Intención
        if action > zona_muerta:
            intencion = OrderType.BUY
        elif action < -zona_muerta:
            intencion = OrderType.SELL
        else:
            print("Acción en zona muerta, manteniendo posición.")
            return
        
        print(f"Intención interpretada: {intencion.value} (precio actual: {price:.2f})")
        
        # Lógica de Ejecución
        posicion_actual = self.portfolio_manager.current_position
        
        if posicion_actual is None:
            # No hay posición, abrimos una nueva si hay intención
            print(f"Ejecutando nueva orden de {intencion.value}...")
            self.portfolio_manager.execute_order(intencion, price)
        else:
            # Ya hay una posición abierta
            tipo_posicion_actual = OrderType(posicion_actual['type'])
            if intencion != tipo_posicion_actual:
                # La intención es opuesta a la posición actual -> cerrar
                print(f"Intención opuesta detectada. Cerrando posición de {tipo_posicion_actual.value}...")
                self.portfolio_manager.close_current_position(price)
            else:
                # La intención es la misma que la posición actual -> mantener
                print(f"Intención coincide con la posición actual de {tipo_posicion_actual.value}. Manteniendo.")
