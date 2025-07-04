import pandas as pd
import numpy as np
from src.data.normalization import Normalization
from src.entorno.base_portfolio import TipoOperacion
from src.entorno.portfolio_state import get_normalized_portfolio_features


class ObservationBuilder:
    """
    Construye el vector de observación normalizado para el entrenamiento y modo en vivo.
    
    Esta clase centraliza la lógica de construcción de observaciones, aplicando el principio DRY
    y garantizando consistencia entre entrenamiento y producción.
    """
    def __init__(self, scaler, price_scaler, run_config: dict):
        """
        Inicializa el constructor de observaciones.
        
        Args:
            scaler: El scaler ya cargado para normalizar las características de mercado.
            price_scaler: El price_scaler ya cargado para normalizar precios.
            run_config (dict): Configuración del run cargada previamente.
        """
        self.scaler = scaler
        self.price_scaler = price_scaler
        self.run_config = run_config
        print(f"ObservationBuilder: Inicializando con scalers inyectados...")
        
        # Validar que la configuración sea válida
        if not run_config or 'config' not in run_config:
            raise ValueError(f"Configuración del run inválida - debe contener la clave 'config'")
        
        # Validar que los scalers estén correctamente cargados
        if self.scaler is None or self.price_scaler is None:
            raise ValueError("Los scalers inyectados no pueden ser None")
        
        # Acceder a la configuración del entorno desde la clave 'config'
        main_config = self.run_config.get('config', {})
        self.env_config = main_config.get('environment', {})
        
        print("✅ Scalers inyectados y configuración cargados exitosamente.")
    
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
        
        Los datos de entrada ya vienen procesados con indicadores técnicos desde el LiveDataProcessor.
        Esta función se encarga únicamente de la normalización de la ventana de observación relevante.
        
        Args:
            live_market_dataframe (pd.DataFrame): DataFrame con datos de mercado en vivo ya procesados
            
        Returns:
            np.ndarray: Vector de características de mercado normalizado
        """
        # 1. Extraer el tamaño de la ventana de observación desde la configuración del entorno
        ventana_size = self.env_config['ventana_observacion_size']
        
        # 2. Seleccionar únicamente las últimas ventana_size filas del DataFrame
        observation_window = live_market_dataframe.tail(ventana_size)
        
        # 3. Asegurar que las columnas estén en el mismo orden que en el entrenamiento.
        # El objeto scaler de scikit-learn guarda esta información.
        if hasattr(self.scaler, 'feature_names_in_'):
            feature_columns_in_order = self.scaler.feature_names_in_
            data_to_transform = observation_window[feature_columns_in_order]
        else:
            # Fallback por si el scaler no tiene los nombres (versiones antiguas)
            data_to_transform = observation_window
        
        # 4. Usar el scaler (cargado en __init__) para transformar los datos directamente.
        transformed_data = self.scaler.transform(data_to_transform)
        
        # 5. Aplanar el array para crear el vector de estado final.
        state_vector = transformed_data.ravel()
        
        # 6. Devolver el vector de estado.
        return state_vector

    def _build_portfolio_observation(self, portfolio_state: dict) -> np.ndarray:
        """
        Construye el vector de observación del portafolio normalizando el estado del portafolio.
        
        Args:
            portfolio_state (dict): Estado actual del portafolio
            
        Returns:
            np.ndarray: Vector de características del portafolio normalizado
        """
        return get_normalized_portfolio_features(
            portfolio_state=portfolio_state,
            env_config=self.env_config,
            price_scaler=self.price_scaler
        )
