"""
ArtifactManager: Specialized manager for data artifact operations.
Ya veremos que hace este ahora jejeje
"""

import os
import tempfile
import logging
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
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
        
    def artifact_exists(self, data_run_id: str, artifact_type: str) -> bool:
        """
        Check if a specific artifact exists for a data_run.
        
        Args:
            data_run_id: The ID of the data_run to check
            artifact_type: Type of artifact ('dataframe', 'scaler', 'price_scaler', 'metadata')
            
        Returns:
            bool: True if the artifact exists
        """
        prefix = self._get_data_run_prefix(data_run_id)
        
        # Map artifact types to filenames
        artifact_files = {
            'dataframe': FILE_DATAFRAME_PKL,
            'scaler': FILE_SCALER_PKL,
            'price_scaler': FILE_PRICE_SCALER_PKL,
            'metadata': FILE_DATA_RUN_METADATA_YAML
        }
        
        if artifact_type not in artifact_files:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        filename = artifact_files[artifact_type]
        
        try:
            if self.storage_mode == "gcp":
                blob_name = f"{prefix}/{filename}"
                bucket = self.gcs_utils.client.bucket(self.gcs_bucket_name)
                blob = bucket.blob(blob_name)
                return blob.exists()
            else:
                artifact_path = Path(prefix) / filename
                return artifact_path.exists()
                
        except Exception as e:
            self.logger.error(f"Error checking if artifact exists for {data_run_id}: {str(e)}")
            return False

    def get_artifact_info(self, data_run_id: str) -> Dict[str, Any]:
        """
        Get information about all artifacts for a data_run.
        
        Args:
            data_run_id: The ID of the data_run to get info for
            
        Returns:
            Dictionary containing artifact information
        """
        prefix = self._get_data_run_prefix(data_run_id)
        
        info = {
            'data_run_id': data_run_id,
            'storage_mode': self.storage_mode,
            'prefix': prefix,
            'artifacts': {}
        }
        
        # Check each artifact type
        artifact_types = ['dataframe', 'scaler', 'price_scaler', 'metadata']
        for artifact_type in artifact_types:
            info['artifacts'][artifact_type] = self.artifact_exists(data_run_id, artifact_type)
        
        return info
