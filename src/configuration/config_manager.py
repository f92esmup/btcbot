"""
ConfigManager: Specialized manager for configuration file operations.

This module provides centralized management for configuration operations including:
- Saving and loading training run configurations
- YAML file handling
- Evaluation summary persistence
- Both local and GCS storage support
"""

import os
import tempfile
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from src.configuration.constants import (
    FILE_CONFIG_RUN_YAML, FILE_EVALUATION_SUMMARY_JSON,
    DIR_TRAINING_RUNS, STORAGE_MODE_GCP, KEY_STORAGE, KEY_BUCKET_NAME
)


class ConfigManager:
    """
    Specialized manager for configuration file operations.
    
    Handles all configuration-related operations including YAML configs,
    evaluation summaries, and metadata across both local and GCS storage modes.
    """
    
    def __init__(self, storage_mode: str = "local", gcp_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            storage_mode: Storage mode ('local' or 'gcp', defaults to 'local')
            gcp_config: GCP configuration dictionary (required when storage_mode is 'gcp')
        """
        self.storage_mode = storage_mode
        
        # Extract GCS bucket name from gcp_config if provided
        if gcp_config:
            self.gcs_bucket_name = gcp_config.get(KEY_STORAGE, {}).get(KEY_BUCKET_NAME)
        else:
            self.gcs_bucket_name = None
        
        # Validate GCS configuration
        if self.storage_mode == STORAGE_MODE_GCP and not self.gcs_bucket_name:
            raise ValueError("gcs_bucket_name is required when storage_mode is 'gcp'. Provide it via gcp_config parameter.")
        
        # Store gcp_config for worker processes
        self.gcp_config = gcp_config
        
        # Handle GCS utils
        if self.storage_mode == STORAGE_MODE_GCP:
            if gcp_config is None:
                raise ValueError("gcp_config is required when storage_mode is 'gcp'")
            
            # Create new GCSUtils instance with the provided configuration
            from src.configuration.gcs_utils import GCSUtils
            self.gcs_utils = GCSUtils(gcp_config)
        else:
            self.gcs_utils = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _get_training_run_prefix(self, training_run_id: str) -> str:
        """
        Get the path prefix for a specific training run.
        
        Args:
            training_run_id: The unique identifier for the training run
            
        Returns:
            The path prefix string for the training run (e.g., "training_runs/ID_ABC")
        """
        return f"{DIR_TRAINING_RUNS}/{training_run_id}"
    
    def save_run_config(self, training_run_id: str, full_run_config: Dict[str, Any]) -> None:
        """
        Saves the complete, pre-assembled run configuration to config_run.yaml.

        Args:
            training_run_id: The ID of the training run to save configuration for
            full_run_config (Dict[str, Any]): The complete configuration dictionary to save.
                                          This dictionary should already contain all necessary
                                          sections (run_info, command_line_args, config).
        """
        self.logger.info(f"Saving run configuration for training run: {training_run_id}")
        
        # Get training run prefix using helper
        prefix = self._get_training_run_prefix(training_run_id)

        if self.storage_mode == STORAGE_MODE_GCP:
            # For GCP, save temporarily and upload
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                yaml.dump(full_run_config, temp_file, default_flow_style=False, allow_unicode=True)
                temp_path = temp_file.name
            
            try:
                # Upload to GCS using the prefix
                gcs_blob_name = f"{prefix}/{FILE_CONFIG_RUN_YAML}"
                if self.gcs_utils.upload_file_to_gcs(temp_path, gcs_blob_name):
                    self.logger.info(f"Run configuration saved to GCS: gs://{self.gcs_bucket_name}/{gcs_blob_name}")
                else:
                    self.logger.error("Error saving run configuration to GCS")
            finally:
                os.unlink(temp_path)
        else:
            # For local, save using the prefix
            config_path = Path(prefix) / FILE_CONFIG_RUN_YAML
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_run_config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Run configuration saved to: {config_path}")
    
    def save_evaluation_summary(self, training_run_id: str, metrics: dict) -> str:
        """
        Save evaluation metrics summary to evaluation_summary.json file.
        
        Args:
            training_run_id: The ID of the training run to save evaluation summary for
            metrics: Dictionary containing evaluation metrics to save
            
        Returns:
            Path to the saved evaluation summary file (local path or GCS URI)
        """
        self.logger.info(f"Saving evaluation summary for training run: {training_run_id}")
        
        # Get training run prefix using helper
        prefix = self._get_training_run_prefix(training_run_id)
        
        try:
            if self.storage_mode == "gcp":
                # For GCP, save temporarily and upload
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(metrics, temp_file, indent=2, ensure_ascii=False)
                    temp_path = temp_file.name
                
                try:
                    # Upload to GCS using the prefix with evaluation subdirectory
                    gcs_blob_name = f"{prefix}/evaluation/{FILE_EVALUATION_SUMMARY_JSON}"
                    if self.gcs_utils.upload_file_to_gcs(temp_path, gcs_blob_name):
                        final_path = f"gs://{self.gcs_bucket_name}/{gcs_blob_name}"
                        self.logger.info(f"Evaluation summary saved to GCS: {final_path}")
                        return final_path
                    else:
                        self.logger.error("Error saving evaluation summary to GCS")
                        raise RuntimeError("Failed to upload evaluation summary to GCS")
                finally:
                    os.unlink(temp_path)
            else:
                # For local storage, create evaluation subdirectory and save
                evaluation_dir = Path(prefix) / "evaluation"
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                
                summary_path = evaluation_dir / FILE_EVALUATION_SUMMARY_JSON
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Evaluation summary saved to: {summary_path}")
                return str(summary_path)
                
        except Exception as e:
            self.logger.error(f"Error saving evaluation summary for training run {training_run_id}: {str(e)}")
            raise
    
    @staticmethod
    def load_training_run_config(training_run_id: str, storage_mode: str = None, gcp_config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Load training run configuration without requiring a full ConfigManager instance.
        This method is useful when you need to load configuration to determine how to create a ConfigManager.
        
        Args:
            training_run_id: The training run ID for which to load the configuration
            storage_mode: Storage mode ('local' or 'gcp'). If None, tries to determine automatically
            gcp_config: GCP configuration dictionary (required if storage_mode is 'gcp')
            
        Returns:
            Dict containing the configuration, or None if file doesn't exist or error occurs
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading configuration for training run: {training_run_id}")
        
        # Get training run prefix using the same logic as the helper
        prefix = f"training_runs/{training_run_id}"
        
        # If storage_mode not provided, try to determine it
        if storage_mode is None:
            # Try local first
            local_config_path = Path(f"{prefix}/{FILE_CONFIG_RUN_YAML}")
            if local_config_path.exists():
                storage_mode = "local"
            else:
                # If not found locally, assume GCP as a fallback strategy.
                # This avoids a hard dependency on a global config file.
                storage_mode = "gcp"
                logger.info(f"Configuracion no encontrada localmente para el training_run_id {training_run_id}, asumiendo storage_mode='gcp'.")
                # Note: gcp_config must be provided by the caller in this case.
        
        try:
            if storage_mode == "gcp":
                # Validate GCP requirements
                if gcp_config is None:
                    raise ValueError("gcp_config is required for GCP storage mode")
                
                # Create GCSUtils instance
                from src.configuration.gcs_utils import GCSUtils
                gcs_utils = GCSUtils(gcp_config)
                
                # Extract bucket name from config
                gcs_bucket_name = gcp_config.get('storage', {}).get('bucket_name')
                if not gcs_bucket_name:
                    raise ValueError("bucket_name is required in gcp_config")
                
                # Construct GCS blob name using prefix
                blob_name = f"{prefix}/{FILE_CONFIG_RUN_YAML}"
                logger.info(f"Downloading config from GCS: {blob_name}")
                
                # Download blob content directly to memory
                bucket = gcs_utils.client.bucket(gcs_bucket_name)
                blob = bucket.blob(blob_name)
                
                if not blob.exists():
                    logger.warning(f"Training run configuration file not found in GCS: {blob_name}")
                    return None
                
                # Download as string and parse YAML
                yaml_content = blob.download_as_string().decode('utf-8')
                logger.info("Training run configuration downloaded successfully from GCS")
                
                # Load YAML content
                config_data = yaml.safe_load(yaml_content)
                logger.info("Training run configuration loaded successfully from GCS")
                return config_data
                            
            else:
                # Local storage mode using prefix
                config_path = Path(f"{prefix}/{FILE_CONFIG_RUN_YAML}")
                logger.info(f"Loading config from local path: {config_path}")
                
                if not config_path.exists():
                    logger.warning(f"Training run configuration file not found locally: {config_path}")
                    return None
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                logger.info("Training run configuration loaded successfully from local storage")
                return config_data
                
        except Exception as e:
            logger.error(f"Error loading configuration for training run {training_run_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
