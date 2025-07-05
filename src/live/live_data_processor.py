import pandas as pd
from datetime import datetime
from src.data.indicadores import Indicadores


class LiveDataProcessor:
    """
    Clase responsable de procesar la ventana de datos de mercado en tiempo real,
    garantizando su integridad y enriqueciéndola con indicadores técnicos.
    
    Implementa validaciones de continuidad temporal y ausencia de NaNs para
    asegurar que los datos que llegan al agente son de alta calidad.
    """
    
    def __init__(self, run_config: dict):
        """
        Inicializa el procesador de datos en vivo.
        
        Args:
            run_config (dict): Diccionario de configuración del run que contiene
                             los parámetros necesarios de 'environment' y 'data'.
        """
        self.run_config = run_config
        
        # La configuración principal del run está anidada bajo una clave 'config'
        self.main_config = run_config.get('config', {})
        
        # Corregir la obtención del interval desde experiment_definition (legacy) o metadata
        # Primero intentar desde experiment_definition (para compatibilidad con training_runs antiguos)
        interval_from_exp_def = self.main_config.get('experiment_definition', {}).get('interval')
        # Luego intentar desde metadata.experiment_parameters (para nuevos training_runs)
        interval_from_metadata = self.main_config.get('metadata', {}).get('experiment_parameters', {}).get('interval')
        # Usar el valor disponible, con fallback a '1h'
        self.interval = interval_from_exp_def or interval_from_metadata or '1h'
        
        # Corregir la obtención del ventana_observacion_size desde environment
        self.ventana_observacion_size = self.main_config.get('environment', {}).get('ventana_observacion_size', 20)
        
        # Mapeo de intervalos a frecuencias de pandas
        self.interval_to_freq = {
            '1m': 'T',    # minutely
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': 'H',    # hourly
            '4h': '4H',
            '1d': 'D',    # daily
        }
    
    def process(self, raw_market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Método principal que orquesta todo el procesamiento de datos.
        
        Args:
            raw_market_data (pd.DataFrame): DataFrame con datos de mercado crudos
                                          con timestamp como índice.
        
        Returns:
            pd.DataFrame: DataFrame procesado y validado con indicadores técnicos.
        
        Raises:
            ValueError: Si se encuentran NaNs en la ventana de observación final.
        """
        # Crear una copia para no modificar el original
        df_processed = raw_market_data.copy()
        
        # Validar continuidad temporal antes de añadir indicadores
        self._check_continuity(df_processed, "pre-indicadores")
        
        # Instanciar y usar la clase Indicadores para enriquecer los datos
        # La clase Indicadores requiere el dataframe en el constructor y la configuración completa
        indicadores = Indicadores(df_processed, self.main_config)
        df_processed = indicadores.main()
        
        # Validar continuidad temporal después de añadir indicadores
        self._check_continuity(df_processed, "post-indicadores")
        
        # Verificar que la ventana de observación final no contenga NaNs
        self._check_final_window_for_nans(df_processed)
        
        return df_processed
    
    def _check_continuity(self, df: pd.DataFrame, step_name: str):
        """
        Valida la continuidad temporal del DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a validar.
            step_name (str): Nombre del paso donde se realiza la validación
                           (ej. 'pre-indicadores', 'post-indicadores').
        """
        if df.empty:
            print(f"⚠️  ADVERTENCIA [{step_name}]: DataFrame vacío")
            return
        
        # Inferir la frecuencia del índice
        inferred_freq = pd.infer_freq(df.index)
        expected_freq = self.interval_to_freq.get(self.interval)
        
        if inferred_freq is None:
            print(f"⚠️  ADVERTENCIA [{step_name}]: No se pudo inferir la frecuencia del DataFrame. "
                  f"Puede haber saltos temporales o datos irregulares.")
            return
        
        # Normalizar frecuencias para comparación (manejar variaciones como 'H' vs '60T')
        if expected_freq and not self._frequencies_match(inferred_freq, expected_freq):
            print(f"⚠️  ADVERTENCIA [{step_name}]: Frecuencia inferida '{inferred_freq}' "
                  f"no coincide con la esperada '{expected_freq}' para el intervalo '{self.interval}'. "
                  f"Puede haber discontinuidades en los datos.")
        else:
            print(f"✅ CONTINUIDAD [{step_name}]: Datos temporalmente consistentes "
                  f"(frecuencia: {inferred_freq})")
    
    def _frequencies_match(self, freq1: str, freq2: str) -> bool:
        """
        Compara si dos frecuencias son equivalentes.
        
        Args:
            freq1 (str): Primera frecuencia.
            freq2 (str): Segunda frecuencia.
        
        Returns:
            bool: True si las frecuencias son equivalentes.
        """
        # Normalizar frecuencias comunes
        freq_equivalents = {
            'H': '60T',
            '4H': '240T',
            '5T': '5min',
            '15T': '15min',
            '30T': '30min',
            'T': 'min'
        }
        
        normalized_freq1 = freq_equivalents.get(freq1, freq1)
        normalized_freq2 = freq_equivalents.get(freq2, freq2)
        
        return normalized_freq1 == normalized_freq2 or freq1 == freq2
    
    def _check_final_window_for_nans(self, df: pd.DataFrame):
        """
        Verifica que la ventana de observación final no contenga valores NaN.
        
        Args:
            df (pd.DataFrame): DataFrame a verificar.
        
        Raises:
            ValueError: Si se encuentran NaNs en la ventana de observación final.
        """
        if len(df) < self.ventana_observacion_size:
            raise ValueError(
                f"❌ ERROR CRÍTICO: DataFrame tiene {len(df)} filas, "
                f"pero se necesitan al menos {self.ventana_observacion_size} "
                f"para la ventana de observación."
            )
        
        # Extraer la ventana de observación final
        observation_window = df.tail(self.ventana_observacion_size)
        
        # Comprobar si hay algún NaN
        if observation_window.isnull().any().any():
            nan_columns = observation_window.columns[observation_window.isnull().any()].tolist()
            nan_count = observation_window.isnull().sum().sum()
            
            raise ValueError(
                f"❌ ERROR CRÍTICO: Se encontraron {nan_count} valores NaN en la ventana "
                f"de observación final (últimas {self.ventana_observacion_size} filas). "
                f"Columnas afectadas: {nan_columns}. Esto impediría que el agente "
                f"pueda tomar decisiones."
            )
        
        print(f"✅ INTEGRIDAD: Ventana de observación final ({self.ventana_observacion_size} filas) "
              f"libre de NaNs")
