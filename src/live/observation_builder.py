import pandas as pd
import numpy as np
from src.training.run_manager import RunManager
from src.data.indicadores import Indicadores
from src.data.normalization import Normalization
from src.entorno.environment import TipoOperacion


class LiveObservationBuilder:
    """
    Construye el vector de observación normalizado para el modo en vivo.
    
    Esta clase carga los scalers de un 'run_id' de entrenamiento específico
    y los utiliza para procesar los datos de mercado en tiempo real,
    asegurando que la entrada al agente sea consistente con el entrenamiento.
    """
    def __init__(self, run_manager: RunManager, run_config: dict):
        """
        Inicializa el constructor de observaciones.
        
        Args:
            run_manager (RunManager): El gestor del run de entrenamiento cuyos artefactos (scalers) se deben cargar.
            run_config (dict): Configuración del run cargada previamente.
        """
        self.run_manager = run_manager
        self.run_config = run_config
        print(f"LiveObservationBuilder: Inicializando para run_id '{self.run_manager.run_id}'...")
        
        # Validar que la configuración sea válida
        if not run_config or 'config' not in run_config:
            raise ValueError(f"Configuración del run inválida para run_id '{self.run_manager.run_id}' - debe contener la clave 'config'")
        
        # Cargar scalers
        self.scaler = self.run_manager.load_scaler()
        self.price_scaler = self.run_manager.load_price_scaler()
        
        # Acceder a la configuración del entorno desde la clave 'config'
        main_config = self.run_config.get('config', {})
        self.env_config = main_config.get('environment', {})
        
        print("Scalers y configuración cargados exitosamente.")
    
    def build(self, live_market_dataframe: pd.DataFrame, live_portfolio_state: dict) -> np.ndarray:
        """
        Construye el vector de observación a partir de los datos de mercado y el estado del portafolio.
        
        Args:
            live_market_dataframe (pd.DataFrame): DataFrame con datos de mercado en vivo
            live_portfolio_state (dict): Estado actual del portafolio
            
        Returns:
            np.ndarray: Vector de observación normalizado
        """
        # 1. Procesar la parte del MERCADO
        market_obs_vector = self._build_market_observation(live_market_dataframe)

        # 2. Procesar la parte del PORTAFOLIO
        portfolio_obs_vector = self._build_portfolio_observation(live_portfolio_state)

        # 3. Concatenar ambos vectores para crear la observación final
        final_observation = np.concatenate([market_obs_vector, portfolio_obs_vector])
        
        return final_observation.astype(np.float32)

    def _build_market_observation(self, live_market_dataframe: pd.DataFrame) -> np.ndarray:
        """
        Construye el vector de observación del mercado a partir de un DataFrame de datos de mercado en vivo.
        
        Args:
            live_market_dataframe (pd.DataFrame): DataFrame con datos de mercado en vivo
            
        Returns:
            np.ndarray: Vector de características de mercado normalizado
        """
        # 1. Calcular los indicadores técnicos sobre los datos en vivo, usando la configuración del run.
        main_config = self.run_config.get('config', {})
        indicadores = Indicadores(live_market_dataframe, config_dict=main_config.get('indicators', {}))
        df_with_indicators = indicadores.main()
        
        # 2. Asegurar que las columnas estén en el mismo orden que en el entrenamiento.
        # El objeto scaler de scikit-learn guarda esta información.
        if hasattr(self.scaler, 'feature_names_in_'):
            feature_columns_in_order = self.scaler.feature_names_in_
            data_to_transform = df_with_indicators[feature_columns_in_order]
        else:
            # Fallback por si el scaler no tiene los nombres (versiones antiguas)
            data_to_transform = df_with_indicators
        
        # 3. Usar el scaler (cargado en __init__) para transformar los datos directamente.
        transformed_data = self.scaler.transform(data_to_transform)
        
        # 4. Aplanar el array para crear el vector de estado final.
        state_vector = transformed_data.ravel()
        
        # 5. Devolver el vector de estado.
        return state_vector

    def _build_portfolio_observation(self, portfolio_state: dict) -> np.ndarray:
        """
        Construye el vector de observación del portafolio normalizando el estado del portafolio.
        
        Args:
            portfolio_state (dict): Estado actual del portafolio
            
        Returns:
            np.ndarray: Vector de características del portafolio normalizado
        """
        # 1. Tipo de posición normalizado
        tipo_posicion = portfolio_state['tipo_posicion']
        if tipo_posicion == 'BUY':
            tipo_posicion_norm = 1.0
        elif tipo_posicion == 'NEUTRAL':
            tipo_posicion_norm = 0.5
        else:  # 'SELL'
            tipo_posicion_norm = 0.0
        
        # 2. PNL ROE normalizado y clipeado
        pnl_roe = portfolio_state['pnl_no_realizado_roe']
        pnl_roe_clipped = np.clip(
            pnl_roe,
            self.env_config['min_clip_pnl_roe'],
            self.env_config['max_clip_pnl_roe']
        )
        
        # Normalizar a [0, 1]
        min_roe = self.env_config['min_clip_pnl_roe']
        max_roe = self.env_config['max_clip_pnl_roe']
        if max_roe != min_roe:
            pnl_roe_norm = (pnl_roe_clipped - min_roe) / (max_roe - min_roe)
        else:
            pnl_roe_norm = 0.5
        
        # 3. Pasos en posición normalizado
        pasos_norm = min(1.0, portfolio_state['pasos_en_posicion'] / self.env_config['max_pasos_en_posicion'])
        
        # 4. Precio de entrada normalizado
        if portfolio_state['tipo_posicion'] != 'NEUTRAL' and portfolio_state['precio_entrada'] > 0:
            # Usar el price_scaler para normalizar el precio de entrada
            precio_entrada_scaled = self.price_scaler.transform([[portfolio_state['precio_entrada']]])[0][0]
            precio_entrada_norm = np.clip(precio_entrada_scaled, 0.0, 1.0)
        else:
            precio_entrada_norm = 0.5  # Valor neutral
        
        return np.array([tipo_posicion_norm, pnl_roe_norm, pasos_norm, precio_entrada_norm], dtype=np.float32)
