"""
ArtifactManager: Specialized manager for data artifact operations.

This module provides centralized management for data artifact operations including:
- Loading and saving normalized DataFrames from/to data runs
- Loading and saving scalers and price scalers
- Data run metadata handling
- Both local and GCS storage support

This class follows the Single Responsibility Principle (SRP) by centralizing
all artifact management operations that were previously scattered across
multiple classes like Normalization and create_dataset.py.
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
    
    Handles loading, saving, and management of data artifacts including DataFrames,
    scalers, and metadata across both local and GCS storage modes.
    
    This class centralizes artifact management operations that were previously
    scattered across multiple modules, following the Single Responsibility Principle.
    
    Supported operations:
    - Save and load normalized DataFrames
    - Save and load MinMaxScaler objects  
    - Save and load price scaler objects
    - Save and load data run metadata
    - Check artifact existence
    - Get artifact information
    
    Supported storage modes:
    - 'local': Local filesystem storage
    - 'gcp': Google Cloud Storage
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

    def save_data_artifacts(self, data_run_id: str, normalized_dataframe, scaler, price_scaler) -> bool:
        """
        Save all data artifacts for a specific data_run.
        
        Args:
            data_run_id: The ID of the data_run to save artifacts for
            normalized_dataframe: The normalized DataFrame to save
            scaler: The fitted MinMaxScaler to save
            price_scaler: The fitted price scaler to save
            
        Returns:
            bool: True if all artifacts were saved successfully
        """
        self.logger.info(f"Saving data artifacts for data_run: {data_run_id}")
        
        # Get data run prefix using helper
        prefix = self._get_data_run_prefix(data_run_id)
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: save to Google Cloud Storage
                self.logger.info("Saving data artifacts to Google Cloud Storage...")
                
                # Save normalized_dataframe
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                    temp_dataframe_path = temp_file.name
                    
                try:
                    with open(temp_dataframe_path, 'wb') as f:
                        pickle.dump(normalized_dataframe, f)
                    
                    dataframe_blob_name = f"{prefix}/{FILE_DATAFRAME_PKL}"
                    if not self.gcs_utils.upload_file_to_gcs(temp_dataframe_path, dataframe_blob_name):
                        raise RuntimeError(f"Failed to upload dataframe to GCS: {dataframe_blob_name}")
                    self.logger.info(f"DataFrame saved to GCS: {dataframe_blob_name}")
                finally:
                    os.unlink(temp_dataframe_path)
                
                # Save scaler
                scaler_blob_name = f"{prefix}/{FILE_SCALER_PKL}"
                if not self.gcs_utils.save_scaler_to_gcs(scaler, scaler_blob_name):
                    raise RuntimeError(f"Failed to save scaler to GCS: {scaler_blob_name}")
                self.logger.info(f"Scaler saved to GCS: {scaler_blob_name}")
                
                # Save price_scaler
                price_scaler_blob_name = f"{prefix}/{FILE_PRICE_SCALER_PKL}"
                if not self.gcs_utils.save_price_scaler_to_gcs(price_scaler, price_scaler_blob_name):
                    raise RuntimeError(f"Failed to save price_scaler to GCS: {price_scaler_blob_name}")
                self.logger.info(f"Price scaler saved to GCS: {price_scaler_blob_name}")
                
                return True
                
            else:
                # Local mode
                data_run_path = Path(prefix)
                data_run_path.mkdir(parents=True, exist_ok=True)
                
                # Save normalized_dataframe
                dataframe_path = data_run_path / FILE_DATAFRAME_PKL
                with open(dataframe_path, 'wb') as f:
                    pickle.dump(normalized_dataframe, f)
                self.logger.info(f"DataFrame saved to: {dataframe_path}")
                
                # Save scaler
                scaler_path = data_run_path / FILE_SCALER_PKL
                joblib.dump(scaler, scaler_path)
                self.logger.info(f"Scaler saved to: {scaler_path}")
                
                # Save price_scaler
                price_scaler_path = data_run_path / FILE_PRICE_SCALER_PKL
                joblib.dump(price_scaler, price_scaler_path)
                self.logger.info(f"Price scaler saved to: {price_scaler_path}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving data artifacts for data_run {data_run_id}: {str(e)}")
            return False

    def save_data_run_metadata(self, data_run_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Save metadata for a specific data_run.
        
        Args:
            data_run_id: The ID of the data_run to save metadata for
            metadata: Dictionary containing the metadata to save
            
        Returns:
            bool: True if metadata was saved successfully
        """
        self.logger.info(f"Saving metadata for data_run: {data_run_id}")
        
        # Get data run prefix using helper
        prefix = self._get_data_run_prefix(data_run_id)
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: save to Google Cloud Storage
                blob_name = f"{prefix}/{FILE_DATA_RUN_METADATA_YAML}"
                
                # Convert metadata to YAML string
                yaml_content = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
                
                # Upload to GCS
                bucket = self.gcs_utils.client.bucket(self.gcs_bucket_name)
                blob = bucket.blob(blob_name)
                blob.upload_from_string(yaml_content, content_type='text/yaml')
                
                self.logger.info(f"Data run metadata saved to GCS: {blob_name}")
                return True
                
            else:
                # Local mode
                data_run_path = Path(prefix)
                data_run_path.mkdir(parents=True, exist_ok=True)
                
                metadata_path = data_run_path / FILE_DATA_RUN_METADATA_YAML
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
                
                self.logger.info(f"Data run metadata saved to: {metadata_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving data run metadata for {data_run_id}: {str(e)}")
            return False

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
