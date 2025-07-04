"""
ArtifactManager: Specialized manager for data artifact operations.

This module provides centralized management for data artifact operations including:
- Loading normalized DataFrames from data runs
- Loading scalers and price scalers
- Data run metadata handling
- Both local and GCS storage support
"""

import os
import tempfile
import logging
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

from src.configuration.constants import (
    FILE_DATAFRAME_PKL, FILE_SCALER_PKL, FILE_PRICE_SCALER_PKL,
    FILE_DATA_RUN_METADATA_YAML
)


class ArtifactManager:
    """
    Specialized manager for data artifact operations.
    
    Handles loading and management of data artifacts including DataFrames,
    scalers, and metadata across both local and GCS storage modes.
    """
    
    def __init__(self, storage_mode: str = "local", gcp_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ArtifactManager.
        
        Args:
            storage_mode: Storage mode ('local' or 'gcp', defaults to 'local')
            gcp_config: GCP configuration dictionary (required when storage_mode is 'gcp')
        """
        self.storage_mode = storage_mode
        
        # Extract GCS bucket name from gcp_config if provided
        if gcp_config:
            self.gcs_bucket_name = gcp_config.get('storage', {}).get('bucket_name')
        else:
            self.gcs_bucket_name = None
        
        # Validate GCS configuration
        if self.storage_mode == "gcp" and not self.gcs_bucket_name:
            raise ValueError("gcs_bucket_name is required when storage_mode is 'gcp'. Provide it via gcp_config parameter.")
        
        # Store gcp_config for worker processes
        self.gcp_config = gcp_config
        
        # Handle GCS utils
        if self.storage_mode == "gcp":
            if gcp_config is None:
                raise ValueError("gcp_config is required when storage_mode is 'gcp'")
            
            # Create new GCSUtils instance with the provided configuration
            from src.configuration.gcs_utils import GCSUtils
            self.gcs_utils = GCSUtils(gcp_config)
        else:
            self.gcs_utils = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _get_data_run_prefix(self, data_run_id: str) -> str:
        """
        Get the path prefix for a specific data run.
        
        Args:
            data_run_id: The unique identifier for the data run
            
        Returns:
            The path prefix string for the data run (e.g., "data_runs/ID_123")
        """
        return f"data_runs/{data_run_id}"
    
    def load_data_artifacts(self, data_run_id: str) -> Tuple[Any, Any, Any]:
        """
        Load all data artifacts from a specific data_run.
        
        Args:
            data_run_id: The ID of the data_run to load artifacts from
            
        Returns:
            Tuple of (normalized_dataframe, scaler, price_scaler)
        """
        self.logger.info(f"Loading data artifacts from data_run: {data_run_id}")
        
        # Get data run prefix using helper
        prefix = self._get_data_run_prefix(data_run_id)
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: load from Google Cloud Storage
                self.logger.info("Loading data artifacts from Google Cloud Storage...")
                
                # Use temporary files for downloading
                import tempfile
                
                # Load normalized_dataframe
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                    temp_dataframe_path = temp_file.name
                
                try:
                    dataframe_blob_name = f"{prefix}/{FILE_DATAFRAME_PKL}"
                    if self.gcs_utils.download_file_from_gcs(dataframe_blob_name, temp_dataframe_path):
                        with open(temp_dataframe_path, 'rb') as f:
                            normalized_dataframe = pickle.load(f)
                        self.logger.info(f"DataFrame loaded from GCS - Shape: {normalized_dataframe.shape}")
                    else:
                        raise FileNotFoundError(f"Failed to download dataframe from GCS: {dataframe_blob_name}")
                finally:
                    os.unlink(temp_dataframe_path)
                
                # Load scaler
                scaler_blob_name = f"{prefix}/{FILE_SCALER_PKL}"
                scaler = self.gcs_utils.load_scaler_from_gcs(scaler_blob_name)
                self.logger.info("Scaler loaded from GCS")
                
                # Load price_scaler
                price_scaler_blob_name = f"{prefix}/{FILE_PRICE_SCALER_PKL}"
                price_scaler = self.gcs_utils.load_price_scaler_from_gcs(price_scaler_blob_name)
                self.logger.info("Price scaler loaded from GCS")
                
                return normalized_dataframe, scaler, price_scaler
                
            else:
                # Local mode using prefix
                data_run_path = Path(prefix)
                
                if not data_run_path.exists():
                    raise FileNotFoundError(f"Data run directory not found: {data_run_path}")
                
                # Load normalized_dataframe
                dataframe_path = data_run_path / FILE_DATAFRAME_PKL
                if not dataframe_path.exists():
                    raise FileNotFoundError(f"Normalized dataframe not found: {dataframe_path}")
                
                with open(dataframe_path, 'rb') as f:
                    normalized_dataframe = pickle.load(f)
                self.logger.info(f"DataFrame loaded from: {dataframe_path} - Shape: {normalized_dataframe.shape}")
                
                # Load scaler
                scaler_path = data_run_path / FILE_SCALER_PKL
                if not scaler_path.exists():
                    raise FileNotFoundError(f"Scaler not found: {scaler_path}")
                
                scaler = joblib.load(scaler_path)
                self.logger.info(f"Scaler loaded from: {scaler_path}")
                
                # Load price_scaler
                price_scaler_path = data_run_path / FILE_PRICE_SCALER_PKL
                if not price_scaler_path.exists():
                    raise FileNotFoundError(f"Price scaler not found: {price_scaler_path}")
                
                price_scaler = joblib.load(price_scaler_path)
                self.logger.info(f"Price scaler loaded from: {price_scaler_path}")
                
                return normalized_dataframe, scaler, price_scaler
                
        except Exception as e:
            self.logger.error(f"Error loading data artifacts from data_run {data_run_id}: {str(e)}")
            raise

    def load_data_run_metadata(self, data_run_id: str) -> Dict[str, Any]:
        """
        Load metadata from a specific data_run.
        
        Args:
            data_run_id: The ID of the data_run to load metadata from
            
        Returns:
            Dictionary containing the data_run metadata
        """
        self.logger.info(f"Loading metadata from data_run: {data_run_id}")
        
        # Get data run prefix using helper
        prefix = self._get_data_run_prefix(data_run_id)
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: load from Google Cloud Storage
                blob_name = f"{prefix}/{FILE_DATA_RUN_METADATA_YAML}"
                
                # Download blob content directly to memory
                bucket = self.gcs_utils.client.bucket(self.gcs_bucket_name)
                blob = bucket.blob(blob_name)
                
                if not blob.exists():
                    raise FileNotFoundError(f"Data run metadata not found in GCS: {blob_name}")
                
                yaml_content = blob.download_as_string().decode('utf-8')
                metadata = yaml.safe_load(yaml_content)
                self.logger.info(f"Data run metadata loaded from GCS: {blob_name}")
                
                return metadata
                
            else:
                # Local mode using prefix
                metadata_path = Path(f"{prefix}/{FILE_DATA_RUN_METADATA_YAML}")
                
                if not metadata_path.exists():
                    raise FileNotFoundError(f"Data run metadata not found: {metadata_path}")
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f)
                
                self.logger.info(f"Data run metadata loaded from: {metadata_path}")
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error loading data run metadata for {data_run_id}: {str(e)}")
            raise
