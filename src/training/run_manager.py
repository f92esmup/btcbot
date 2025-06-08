"""
RunManager: Centralizes all file management operations for training runs.

This module provides a centralized manager for handling all file operations including:
- Run configuration saving
- Checkpoint management
- Agent model persistence
- Scaler loading
- Path management for both local and GCS storage
"""

import os
import re
import tempfile
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import joblib
import torch

from src.configuration.config import config
from src.agente.agent import TransformerSACAgent


class RunManager:
    """
    Centralized manager for all file operations in training runs.
    
    Handles persistence operations for models, configurations, and scalers
    across both local and GCS storage modes.
    """
    
    def __init__(self, base_path: str = None, run_id: str = None, gcs_utils=None):
        """
        Initialize the RunManager.
        
        Args:
            base_path: Base path for storing artifacts (optional, will use default if not provided)
            run_id: Unique identifier for this training run (optional, will generate if not provided)
            gcs_utils: GCS utilities instance (optional, will create if needed for GCP mode)
        """
        self.storage_mode = config.storage_mode
        
        # Handle base_path
        if base_path is None:
            if self.storage_mode == "gcp":
                # For GCP, we'll set base_path later when run_id is known
                self.base_path = None
            else:
                self.base_path = "Entrenamientos"
        else:
            self.base_path = base_path
        
        # Handle run_id
        if run_id is None:
            # Generate a temporary run_id that can be overridden later
            from datetime import datetime
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.run_id = f"temp_{current_time}"
            self._temp_run_id = True
        else:
            self.run_id = run_id
            self._temp_run_id = False
        
        # Update base_path with run_id if needed
        if self.base_path is None and self.storage_mode == "gcp":
            self.base_path = f"gs://{config.gcs_bucket_name}/{self.run_id}"
        
        # Handle GCS utils
        if self.storage_mode == "gcp":
            if gcs_utils is None:
                try:
                    from src.configuration.gcs_utils import gcs_utils as global_gcs_utils
                    self.gcs_utils = global_gcs_utils
                except ImportError:
                    raise ValueError("gcs_utils is required for GCP storage mode, but could not import global instance")
            else:
                self.gcs_utils = gcs_utils
        else:
            self.gcs_utils = gcs_utils
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def save_run_config(self, hparams: Dict, args) -> None:
        """
        Save the complete run configuration to config_run.yaml.
        
        Args:
            hparams: Dictionary of hyperparameters
            args: Command line arguments
        """
        self.logger.info("Saving run configuration...")
        
        # Create the complete run configuration
        run_config = {
            'run_info': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'storage_mode': self.storage_mode,
                'base_path': self.base_path
            },
            'command_line_args': {
                'symbol': args.symbol,
                'interval': args.interval,
                'start_date': args.start_date,
                'episodes': args.episodes,
                'eval_frequency': args.eval_frequency,
                'save_frequency': args.save_frequency,
                'no_cuda': args.no_cuda,
                'eval_episodes': args.eval_episodes
            },
            'hyperparameters': {k: v for k, v in hparams.items() if k not in ['run_id', 'storage_mode', 'base_path']},
            'config_snapshot': {
                'normalization': config.normalization_config,
                'environment': {
                    'capital_inicial': config.capital_inicial,
                    'apalancamiento': config.apalancamiento,
                    'porcentaje_max_inversion_por_trade': config.porcentaje_max_inversion_por_trade,
                    'max_drawdown_configurado_cuenta': config.max_drawdown_configurado_cuenta,
                    'comision_taker_porcentaje': config.comision_taker_porcentaje,
                    'slippage_porcentaje': config.slippage_porcentaje,
                    'ventana_observacion_size': config.ventana_observacion_size,
                    'max_pasos_en_posicion': config.max_pasos_en_posicion
                },
                'agent': {
                    'gamma': config.gamma,
                    'tau': config.tau,
                    'batch_size': config.batch_size,
                    'replay_buffer_size': config.replay_buffer_size,
                    'actor_learning_rate': config.actor_learning_rate,
                    'critic_learning_rate': config.critic_learning_rate,
                    'alpha_learning_rate': config.alpha_learning_rate,
                    'd_model': config.d_model,
                    'n_head': config.n_head,
                    'num_encoder_layers': config.num_encoder_layers
                }
            }
        }
        
        if self.storage_mode == "gcp":
            # For GCP, save temporarily and upload
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                yaml.dump(run_config, temp_file, default_flow_style=False, allow_unicode=True)
                temp_path = temp_file.name
            
            try:
                # Upload to GCS
                gcs_blob_name = f"{self.run_id}/config_run.yaml"
                if self.gcs_utils.upload_file_to_gcs(temp_path, gcs_blob_name):
                    self.logger.info(f"Run configuration saved to GCS: gs://{config.gcs_bucket_name}/{gcs_blob_name}")
                else:
                    self.logger.error("Error saving run configuration to GCS")
            finally:
                os.unlink(temp_path)
        else:
            # For local, save directly
            config_path = Path(self.base_path) / "config_run.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(run_config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Run configuration saved to: {config_path}")
    
    def find_latest_checkpoint(self, run_id_to_check: str) -> Optional[Tuple[str, int]]:
        """
        Find the latest checkpoint in a specific run.
        
        Args:
            run_id_to_check: The run ID to search for checkpoints
            
        Returns:
            Tuple of (checkpoint_path, episode_number) if found, None otherwise
        """
        self.logger.info(f"Searching for checkpoints in run: {run_id_to_check}")
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: search in GCS
                checkpoint_prefix = f"{run_id_to_check}/checkpoints/"
                
                # Search for checkpoint metadata files in the specific run
                bucket = self.gcs_utils._get_bucket()
                blobs = bucket.list_blobs(prefix=checkpoint_prefix)
                metadata_pattern = re.compile(r"checkpoint_episode_(\d+)_metadata\.pkl$")
                
                latest_episode = -1
                latest_checkpoint = None
                
                for blob in blobs:
                    filename = blob.name.split('/')[-1]
                    match = metadata_pattern.search(filename)
                    if match:
                        episode_num = int(match.group(1))
                        if episode_num > latest_episode:
                            latest_episode = episode_num
                            latest_checkpoint = f"{run_id_to_check}/checkpoints/checkpoint_episode_{episode_num}"
                
                if latest_checkpoint:
                    self.logger.info(f"Checkpoint found in GCS: {latest_checkpoint} (episode {latest_episode})")
                    return (latest_checkpoint, latest_episode)
                else:
                    self.logger.info(f"No checkpoints found in run {run_id_to_check} in GCS")
                    return None
                    
            else:
                # Local mode: search in local directory
                checkpoint_dir = Path(f"Entrenamientos/{run_id_to_check}/checkpoints/")
                self.logger.info(f"Searching locally: {checkpoint_dir}")
                
                if not checkpoint_dir.exists():
                    self.logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
                    return None
                
                # Search for checkpoint metadata files
                metadata_files = list(checkpoint_dir.glob("checkpoint_episode_*_metadata.pkl"))
                
                if not metadata_files:
                    self.logger.info(f"No metadata files found in: {checkpoint_dir}")
                    return None
                
                # Regex pattern to extract episode number
                metadata_pattern = re.compile(r"checkpoint_episode_(\d+)_metadata\.pkl$")
                
                latest_episode_number = -1
                latest_checkpoint_path = None
                
                # Find the most recent checkpoint
                for metadata_file in metadata_files:
                    filename = metadata_file.name
                    match = metadata_pattern.search(filename)
                    
                    if match:
                        episode_number = int(match.group(1))
                        
                        if episode_number > latest_episode_number:
                            latest_episode_number = episode_number
                            latest_checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode_number}"
                
                if latest_checkpoint_path:
                    self.logger.info(f"Checkpoint found locally: {latest_checkpoint_path} (episode {latest_episode_number})")
                    return (str(latest_checkpoint_path), latest_episode_number)
                else:
                    self.logger.info(f"No valid checkpoints found in run {run_id_to_check}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error searching for checkpoints in run {run_id_to_check}: {e}")
            return None
    
    def load_scaler(self, scaler_path: Optional[str] = None, blob_name: Optional[str] = None):
        """
        Load a previously saved scaler.
        
        Args:
            scaler_path: Specific scaler path (optional, for checkpoint loading)
            blob_name: Specific blob name in GCS (optional, for checkpoint loading)
            
        Returns:
            Loaded scaler object
        """
        try:
            if self.storage_mode == "gcp":
                # GCP mode: load from Google Cloud Storage
                self.logger.info("Loading scaler from Google Cloud Storage...")
                return self.gcs_utils.load_scaler_from_gcs(blob_name)
            else:
                # Local mode
                if scaler_path is None:
                    scaler_path = str(Path(self.base_path) / "scaler.pkl")
                
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
                
                scaler = joblib.load(scaler_path)
                self.logger.info(f"Scaler loaded successfully from: {scaler_path}")
                return scaler
                
        except Exception as e:
            self.logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def load_price_scaler(self, price_scaler_path: Optional[str] = None, blob_name: Optional[str] = None):
        """
        Load a previously saved price scaler.
        
        Args:
            price_scaler_path: Specific price scaler path (optional, for checkpoint loading)
            blob_name: Specific blob name in GCS (optional, for checkpoint loading)
            
        Returns:
            Loaded price scaler object
        """
        try:
            if self.storage_mode == "gcp":
                # GCP mode: load from Google Cloud Storage
                self.logger.info("Loading price_scaler from Google Cloud Storage...")
                return self.gcs_utils.load_price_scaler_from_gcs(blob_name)
            else:
                # Local mode
                if price_scaler_path is None:
                    price_scaler_path = str(Path(self.base_path) / "price_scaler.pkl")
                
                if not os.path.exists(price_scaler_path):
                    raise FileNotFoundError(f"Price scaler file not found: {price_scaler_path}")
                
                price_scaler = joblib.load(price_scaler_path)
                self.logger.info(f"Price scaler loaded successfully from: {price_scaler_path}")
                return price_scaler
                
        except Exception as e:
            self.logger.error(f"Error loading price scaler: {str(e)}")
            raise
    
    def save_agent_checkpoint(self, agent: TransformerSACAgent, episode: int) -> str:
        """
        Save agent checkpoint for a specific episode.
        
        Args:
            agent: The agent to save
            episode: Episode number for checkpoint naming
            
        Returns:
            Path where checkpoint was saved
        """
        if self.storage_mode == "gcp":
            # For GCS: use tempfile and upload each file individually
            gcs_checkpoint_directory_prefix = f"{self.run_id}/checkpoints/checkpoint_episode_{episode + 1}"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prefix for temporary local files
                local_temp_ckpt_prefix = os.path.join(temp_dir, f"ckpt_ep_{episode + 1}")
                
                # Save models to temporary directory
                agent.save_models(local_temp_ckpt_prefix)
                
                # Upload each file to GCS
                success_count = 0
                total_files = 0
                
                for local_file_path in Path(temp_dir).glob(f"ckpt_ep_{episode + 1}_*"):
                    total_files += 1
                    local_file_name_only = local_file_path.name
                    gcs_blob_name = f"{gcs_checkpoint_directory_prefix}/{local_file_name_only}"
                    
                    if self.gcs_utils.upload_file_to_gcs(str(local_file_path), gcs_blob_name):
                        success_count += 1
                    else:
                        self.logger.error(f"Error uploading checkpoint {local_file_path} to GCS")
                
                if success_count == total_files and total_files > 0:
                    self.logger.info(f"Checkpoint saved successfully to GCS: {gcs_checkpoint_directory_prefix}/")
                else:
                    self.logger.error(f"Error saving checkpoint to GCS. Only {success_count} of {total_files} files uploaded.")
                    
            checkpoint_path = gcs_checkpoint_directory_prefix
        else:
            # Local mode
            checkpoint_dir = Path(self.base_path) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode + 1}"
            agent.save_models(str(checkpoint_path))
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_agent_from_checkpoint(self, agent: TransformerSACAgent, checkpoint_prefix: str) -> None:
        """
        Load agent from a checkpoint.
        
        Args:
            agent: The agent to load into
            checkpoint_prefix: Prefix/path of the checkpoint to load
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_prefix}")
        agent.load_models(checkpoint_prefix)
        self.logger.info("✅ Checkpoint loaded successfully")
    
    def save_best_model(self, agent: TransformerSACAgent) -> str:
        """
        Save the best performing model.
        
        Args:
            agent: The agent to save
            
        Returns:
            Path where model was saved
        """
        if self.storage_mode == "gcp":
            best_model_path = f"{self.run_id}/best_model/model.pth"
        else:
            best_model_dir = Path(self.base_path) / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = best_model_dir / "model.pth"
        
        agent.save(best_model_path)
        self.logger.info(f"Best model saved: {best_model_path}")
        return str(best_model_path)
    
    def save_final_model(self, agent: TransformerSACAgent) -> str:
        """
        Save the final model at the end of training.
        
        Args:
            agent: The agent to save
            
        Returns:
            Path where model was saved
        """
        if self.storage_mode == "gcp":
            final_model_path = f"{self.run_id}/final_model/model.pth"
        else:
            final_model_dir = Path(self.base_path) / "final_model"
            final_model_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = final_model_dir / "model.pth"
        
        agent.save(final_model_path)
        self.logger.info(f"Final model saved: {final_model_path}")
        return str(final_model_path)
    
    def set_run_context(self, run_id: str, base_path: str = None):
        """
        Update the run context with specific run_id and base_path.
        
        This is useful when RunManager is created without parameters
        and the context becomes available later.
        
        Args:
            run_id: The actual run_id to use
            base_path: The actual base_path to use (optional)
        """
        self.run_id = run_id
        self._temp_run_id = False
        
        if base_path is not None:
            self.base_path = base_path
        elif self.storage_mode == "gcp":
            self.base_path = f"gs://{config.gcs_bucket_name}/{self.run_id}"
        else:
            self.base_path = f"Entrenamientos/{self.run_id}"
        
        self.logger.info(f"RunManager context updated - run_id: {self.run_id}, base_path: {self.base_path}")
