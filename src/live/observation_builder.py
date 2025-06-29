import pandas as pd
import numpy as np
from src.training.run_manager import RunManager
from src.data.indicadores import Indicadores
from src.data.normalization import Normalization


class LiveObservationBuilder:
    """
    Construye el vector de observación normalizado para el modo en vivo.
    
    Esta clase carga los scalers de un 'run_id' de entrenamiento específico
    y los utiliza para procesar los datos de mercado en tiempo real,
    asegurando que la entrada al agente sea consistente con el entrenamiento.
    """
    def __init__(self, run_id: str):
        """
        Inicializa el constructor de observaciones.
        
        Args:
            run_id (str): El ID del entrenamiento cuyos artefactos (scalers) se deben cargar.
        """
        self.run_id = run_id
        print(f"LiveObservationBuilder: Inicializando para run_id '{self.run_id}'...")
        run_manager = RunManager()
        run_manager.set_run_context(run_id=self.run_id)
        self.scaler = run_manager.load_scaler()
        self.price_scaler = run_manager.load_price_scaler()
        print("Scalers cargados exitosamente.")
    
    def build(self, live_dataframe: pd.DataFrame) -> np.ndarray:
        """
        Construye el vector de observación a partir de un DataFrame de datos de mercado en vivo.
        """
        # 1. Calcular los indicadores técnicos sobre los datos en vivo.
        indicadores = Indicadores(live_dataframe)
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
