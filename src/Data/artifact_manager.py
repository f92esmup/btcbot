"""
ArtifactManager: Specialized manager for data artifact operations.
Ya veremos que hace este ahora jejeje
"""

import logging
import pickle
import yaml
from pathlib import Path
from typing import Any
import joblib

from src.config import (
    AppConfig
)


class ArtifactManager:
    """
    Specialized manager for data artifact operations.
    
    Handles loading, saving, and management of data artifacts including DataFrames,
    scalers, and metadata across local.
    
    Supported operations:
    - Save and load artifacts
    - Check artifact existence
    - Get artifact information
    
    Supported storage modes:
    - 'local': Local filesystem storage
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the ArtifactManager.
        
        Args:
            config: Configuration dictionary (can include storage_mode and gcp_config)
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)

        self.config = config

    def save_all_artifacts(self, data_run_id: str, normalized_dataframe: Any, scaler: Any, price_scaler: Any, dataset_metadata: Any) -> bool:
        """
        Save all artifacts for a given data_run.
        
        Args:
            data_run_id: ID of the data run.
            normalized_dataframe: Normalized DataFrame to save.
            scaler: Scaler object to save.
            price_scaler: Price scaler object to save.
            dataset_metadata: Metadata dictionary to save.
        
        Returns:
            bool: True if all artifacts were saved successfully.
        """        
        # Save each artifact individually
        success = (
            self.save_artifact(data_run_id, 'dataframe', normalized_dataframe) and
            self.save_artifact(data_run_id, 'scaler', scaler) and
            self.save_artifact(data_run_id, 'price_scaler', price_scaler) and
            self.save_artifact(data_run_id, 'metadata', dataset_metadata)
        )
        
        return success

    def save_artifact(self, data_run_id: str, artifact_type: str, artifact_obj: Any) -> bool:
        """
        Guarda un artefacto individual de un data_run en modo local.

        Args:
            data_run_id: ID del data_run.
            artifact_type: 'dataframe', 'scaler', 'price_scaler', 'metadata'.
            artifact_obj: Objeto a guardar.

        Returns:
            bool: True si el guardado fue exitoso.
        """
        self.logger.info(f"Guardando artefacto '{artifact_type}' para data_run: {data_run_id}")

        # Mapeo de tipo de artefacto a archivo y función de guardado
        artifact_map = {
            'dataframe':  {'filename': self.config.base.artifacts.normalized_dataframe, 'saver': lambda obj, p: pickle.dump(obj, open(p, 'wb'))},
            'scaler':     {'filename': self.config.base.artifacts.scaler,    'saver': joblib.dump},
            'price_scaler': {'filename': self.config.base.artifacts.price_scaler, 'saver': joblib.dump},
            'metadata':   {'filename': self.config.base.artifacts.dataset_metadata, 'saver': lambda obj, p: yaml.dump(obj, open(p, 'w', encoding='utf-8'), default_flow_style=False, allow_unicode=True)}
        }

        if artifact_type not in artifact_map:
            raise ValueError(f"Tipo de artefacto desconocido: {artifact_type}")

        data_run_path = Path(self.config.base.dir.data_runs) / data_run_id
        data_run_path.mkdir(parents=True, exist_ok=True)
        artifact_path = data_run_path / artifact_map[artifact_type]['filename']

        try:
            saver = artifact_map[artifact_type]['saver']
            # Para joblib.dump, la firma es (obj, filename)
            if saver is joblib.dump:
                saver(artifact_obj, artifact_path)
            else:
                saver(artifact_obj, artifact_path)
            self.logger.info(f"Artefacto '{artifact_type}' guardado en: {artifact_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error guardando artefacto '{artifact_type}' en {artifact_path}: {e}")
            return False

    def load_artifact(self, data_run_id: str, artifact_type: str) -> Any:
        """
        Carga un artefacto individual de un data_run en modo local.

        Args:
            data_run_id: ID del data_run.
            artifact_type: 'dataframe', 'scaler', 'price_scaler', 'metadata'.

        Returns:
            El objeto artefacto cargado.
        """
        self.logger.info(f"Cargando artefacto '{artifact_type}' de data_run: {data_run_id}")

        # Mapeo de tipo de artefacto a archivo y función de carga
        artifact_map = {
            'dataframe':  {'filename': self.config.base.artifacts.normalized_dataframe, 'loader': lambda p: pickle.load(open(p, 'rb'))},
            'scaler':     {'filename': self.config.base.artifacts.scaler,    'loader': joblib.load},
            'price_scaler': {'filename': self.config.base.artifacts.price_scaler, 'loader': joblib.load},
            'metadata':   {'filename': self.config.base.artifacts.dataset_metadata, 'loader': lambda p: yaml.safe_load(open(p, 'r', encoding='utf-8'))}
        }

        if artifact_type not in artifact_map:
            raise ValueError(f"Tipo de artefacto desconocido: {artifact_type}")

        data_run_path = Path(self.config.base.dir.data_runs) / data_run_id
        if not data_run_path.exists():
            raise FileNotFoundError(f"Directorio de data_run no encontrado: {data_run_path}")

        artifact_path = data_run_path / artifact_map[artifact_type]['filename']
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artefacto '{artifact_type}' no encontrado: {artifact_path}")

        try:
            artifact = artifact_map[artifact_type]['loader'](artifact_path)
            self.logger.info(f"Artefacto '{artifact_type}' cargado desde: {artifact_path}")
            return artifact
        except Exception as e:
            self.logger.error(f"Error cargando artefacto '{artifact_type}' de {artifact_path}: {e}")
            raise