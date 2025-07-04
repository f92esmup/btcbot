"""
Abstracciones para fuentes de datos.
Aplica el Principio de Inversión de Dependencias (DIP) para desacoplar
el DataPipeline de implementaciones específicas de fuentes de datos.
"""

import pandas as pd
from abc import ABC, abstractmethod


class DataSource(ABC):
    """
    Interfaz abstracta para fuentes de datos.
    Define el contrato que debe cumplir cualquier implementación de fuente de datos.
    """
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Obtiene los datos de la fuente específica.
        
        Returns:
            pd.DataFrame: DataFrame con datos OHLCV procesados
        """
        pass
