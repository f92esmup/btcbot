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
from multiprocessing import Process
import yaml
import joblib
import torch

from src.agente.agent import TransformerSACAgent


def _save_worker_local(agent_state_dicts: Dict[str, Any], path_prefix: str) -> None:
    """
    Worker function for saving agent state dictionaries to local storage.
    
    Args:
        agent_state_dicts: Dictionary containing all agent state dictionaries
        path_prefix: Path prefix for saving files
    """
    try:
        # Save networks
        torch.save(agent_state_dicts['actor'], f"{path_prefix}_actor.pth")
        torch.save(agent_state_dicts['critic_1'], f"{path_prefix}_critic_1.pth")
        torch.save(agent_state_dicts['critic_2'], f"{path_prefix}_critic_2.pth")
        torch.save(agent_state_dicts['critic_target_1'], f"{path_prefix}_critic_target_1.pth")
        torch.save(agent_state_dicts['critic_target_2'], f"{path_prefix}_critic_target_2.pth")
        
        # Save optimizers
        torch.save(agent_state_dicts['actor_optimizer'], f"{path_prefix}_actor_optimizer.pth")
        torch.save(agent_state_dicts['critic_1_optimizer'], f"{path_prefix}_critic_1_optimizer.pth")
        torch.save(agent_state_dicts['critic_2_optimizer'], f"{path_prefix}_critic_2_optimizer.pth")
        
        # Save alpha and its optimizer if present
        torch.save(agent_state_dicts['log_alpha'], f"{path_prefix}_log_alpha.pth")
        if 'alpha_optimizer' in agent_state_dicts:
            torch.save(agent_state_dicts['alpha_optimizer'], f"{path_prefix}_alpha_optimizer.pth")
        
        # Save metadata
        torch.save(agent_state_dicts['metadata'], f"{path_prefix}_metadata.pth")
        
        print(f"✅ Guardado local completado: {path_prefix}")
        
    except Exception as e:
        print(f"❌ Error en guardado local: {str(e)}")


def _save_worker_gcs(agent_state_dicts: Dict[str, Any], gcs_prefix: str, gcp_config: Dict[str, Any]) -> None:
    """
    Worker function for saving agent state dictionaries to GCS.
    
    Args:
        agent_state_dicts: Dictionary containing all agent state dictionaries
        gcs_prefix: GCS prefix for saving files
        gcp_config: GCP configuration dictionary for creating GCSUtils instance
    """
    try:
        # Import and create GCS utils in the worker process
        from src.configuration.gcs_utils import GCSUtils
        
        # Create GCS client in this process
        gcs_utils = GCSUtils(gcp_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all state dictionaries to temporary directory
            prefix = os.path.join(temp_dir, gcs_prefix.split('/')[-1])
            
            # Save networks
            torch.save(agent_state_dicts['actor'], f"{prefix}_actor.pth")
            torch.save(agent_state_dicts['critic_1'], f"{prefix}_critic_1.pth")
            torch.save(agent_state_dicts['critic_2'], f"{prefix}_critic_2.pth")
            torch.save(agent_state_dicts['critic_target_1'], f"{prefix}_critic_target_1.pth")
            torch.save(agent_state_dicts['critic_target_2'], f"{prefix}_critic_target_2.pth")
            
            # Save optimizers
            torch.save(agent_state_dicts['actor_optimizer'], f"{prefix}_actor_optimizer.pth")
            torch.save(agent_state_dicts['critic_1_optimizer'], f"{prefix}_critic_1_optimizer.pth")
            torch.save(agent_state_dicts['critic_2_optimizer'], f"{prefix}_critic_2_optimizer.pth")
            
            # Save alpha and its optimizer if present
            torch.save(agent_state_dicts['log_alpha'], f"{prefix}_log_alpha.pth")
            if 'alpha_optimizer' in agent_state_dicts:
                torch.save(agent_state_dicts['alpha_optimizer'], f"{prefix}_alpha_optimizer.pth")
            
            # Save metadata
            torch.save(agent_state_dicts['metadata'], f"{prefix}_metadata.pth")
            
            # Upload each file to GCS
            success_count = 0
            total_files = 0
            
            for local_file_path in Path(temp_dir).glob(f"{gcs_prefix.split('/')[-1]}_*"):
                total_files += 1
                local_file_name_only = local_file_path.name
                gcs_blob_name = f"{gcs_prefix}/{local_file_name_only}"
                
                if gcs_utils.upload_file_to_gcs(str(local_file_path), gcs_blob_name):
                    success_count += 1
                else:
                    print(f"❌ Error subiendo archivo {local_file_path} a GCS")
            
            if success_count == total_files and total_files > 0:
                print(f"✅ Guardado GCS completado: {gcs_prefix}/")
            else:
                print(f"❌ Error en guardado GCS. Solo {success_count} de {total_files} archivos subidos.")
                
    except Exception as e:
        print(f"❌ Error en guardado GCS: {str(e)}")


class RunManager:
    """
    Centralized manager for all file operations in training runs.
    
    Handles persistence operations for models, configurations, and scalers
    across both local and GCS storage modes.
    """
    
    @staticmethod
    def _to_cpu_state_dict(state_dict):
        """
        Move all tensors in a state_dict to CPU to avoid CUDA multiprocessing issues.
        
        Args:
            state_dict: Dictionary containing tensors
            
        Returns:
            Dictionary with all tensors moved to CPU
        """
        return {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in state_dict.items()}
    
    def __init__(self, base_path: str = None, run_id: str = None, gcs_utils=None, 
                 storage_mode: str = "local", gcp_config: Dict[str, Any] = None):
        """
        Initialize the RunManager.
        
        Args:
            base_path: Base path for storing artifacts (optional, will use default if not provided)
            run_id: Unique identifier for this training run (optional, will generate if not provided)
            gcs_utils: GCS utilities instance (optional, will create if needed for GCP mode)
            storage_mode: Storage mode ('local' or 'gcp', defaults to 'local')
            gcp_config: GCP configuration dictionary (required when storage_mode is 'gcp' and gcs_utils is None)
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
        
        # Handle base_path
        if base_path is None:
            if self.storage_mode == "gcp":
                # For GCP, we'll set base_path later when run_id is known
                self.base_path = None
            else:
                self.base_path = "training_runs"
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
            self.base_path = f"gs://{self.gcs_bucket_name}/training_runs/{self.run_id}"
        
        # Store gcp_config for worker processes
        self.gcp_config = gcp_config
        
        # Handle GCS utils
        if self.storage_mode == "gcp":
            if gcs_utils is None:
                if gcp_config is None:
                    raise ValueError("gcp_config is required when storage_mode is 'gcp' and gcs_utils is not provided")
                
                # Create new GCSUtils instance with the provided configuration
                from src.configuration.gcs_utils import GCSUtils
                self.gcs_utils = GCSUtils(gcp_config)
            else:
                self.gcs_utils = gcs_utils
        else:
            self.gcs_utils = gcs_utils
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def save_run_config(self, full_run_config: Dict[str, Any]) -> None:
        """
        Saves the complete, pre-assembled run configuration to config_run.yaml.

        Args:
            full_run_config (Dict[str, Any]): The complete configuration dictionary to save.
                                          This dictionary should already contain all necessary
                                          sections (run_info, command_line_args, config).
        """
        self.logger.info("Saving run configuration...")

        if self.storage_mode == "gcp":
            # For GCP, save temporarily and upload
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                yaml.dump(full_run_config, temp_file, default_flow_style=False, allow_unicode=True)
                temp_path = temp_file.name
            
            try:
                # Upload to GCS in training_runs directory
                gcs_blob_name = f"training_runs/{self.run_id}/config_run.yaml"
                if self.gcs_utils.upload_file_to_gcs(temp_path, gcs_blob_name):
                    self.logger.info(f"Run configuration saved to GCS: gs://{self.gcs_bucket_name}/{gcs_blob_name}")
                else:
                    self.logger.error("Error saving run configuration to GCS")
            finally:
                os.unlink(temp_path)
        else:
            # For local, save directly to training_runs directory
            config_path = Path(f"training_runs/{self.run_id}") / "config_run.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_run_config, f, default_flow_style=False, allow_unicode=True)
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
                checkpoint_prefix = f"training_runs/{run_id_to_check}/checkpoints/"
                
                # Search for checkpoint metadata files in the specific run
                bucket = self.gcs_utils._get_bucket()
                blobs = bucket.list_blobs(prefix=checkpoint_prefix)
                
                # Regex para encontrar el número de episodio en la ruta
                # Ej: .../checkpoints/checkpoint_episode_123/checkpoint_episode_123_metadata.pth
                metadata_pattern = re.compile(r"checkpoint_episode_(\d+)/checkpoint_episode_\1_metadata\.pth$")
                
                latest_episode = -1
                
                for blob in blobs:
                    match = metadata_pattern.search(blob.name)
                    if match:
                        episode_num = int(match.group(1))
                        if episode_num > latest_episode:
                            latest_episode = episode_num
                
                if latest_episode != -1:
                    latest_checkpoint_prefix = f"training_runs/{run_id_to_check}/checkpoints/checkpoint_episode_{latest_episode}"
                    self.logger.info(f"Último checkpoint encontrado en GCS: {latest_checkpoint_prefix} (episodio {latest_episode})")
                    return (latest_checkpoint_prefix, latest_episode)
                else:
                    self.logger.info(f"No se encontraron checkpoints válidos en el run {run_id_to_check} en GCS")
                    return None
                    
            else:
                # Local mode: search in local directory
                checkpoint_dir = Path(f"training_runs/{run_id_to_check}/checkpoints/")
                self.logger.info(f"Searching locally: {checkpoint_dir}")
                
                if not checkpoint_dir.exists():
                    self.logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
                    return None
                
                # Search for checkpoint metadata files
                metadata_files = list(checkpoint_dir.glob("checkpoint_episode_*_metadata.pth"))
                
                if not metadata_files:
                    self.logger.info(f"No metadata files found in: {checkpoint_dir}")
                    return None
                
                # Regex pattern to extract episode number
                metadata_pattern = re.compile(r"checkpoint_episode_(\d+)_metadata\.pth$")
                
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
    
    def load_data_artifacts(self, data_run_id: str) -> Tuple[Any, Any, Any]:
        """
        Load all data artifacts from a specific data_run.
        
        Args:
            data_run_id: The ID of the data_run to load artifacts from
            
        Returns:
            Tuple of (normalized_dataframe, scaler, price_scaler)
        """
        self.logger.info(f"Loading data artifacts from data_run: {data_run_id}")
        
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
                    dataframe_blob_name = f"{data_run_id}/normalized_dataframe.pkl"
                    if self.gcs_utils.download_file_from_gcs(dataframe_blob_name, temp_dataframe_path):
                        with open(temp_dataframe_path, 'rb') as f:
                            normalized_dataframe = pickle.load(f)
                        self.logger.info(f"DataFrame loaded from GCS - Shape: {normalized_dataframe.shape}")
                    else:
                        raise FileNotFoundError(f"Failed to download dataframe from GCS: {dataframe_blob_name}")
                finally:
                    os.unlink(temp_dataframe_path)
                
                # Load scaler
                scaler_blob_name = f"{data_run_id}/scaler.pkl"
                scaler = self.gcs_utils.load_scaler_from_gcs(scaler_blob_name)
                self.logger.info("Scaler loaded from GCS")
                
                # Load price_scaler
                price_scaler_blob_name = f"{data_run_id}/price_scaler.pkl"
                price_scaler = self.gcs_utils.load_price_scaler_from_gcs(price_scaler_blob_name)
                self.logger.info("Price scaler loaded from GCS")
                
                return normalized_dataframe, scaler, price_scaler
                
            else:
                # Local mode
                data_run_path = Path(f"{data_run_id}")
                
                if not data_run_path.exists():
                    raise FileNotFoundError(f"Data run directory not found: {data_run_path}")
                
                # Load normalized_dataframe
                dataframe_path = data_run_path / "normalized_dataframe.pkl"
                if not dataframe_path.exists():
                    raise FileNotFoundError(f"Normalized dataframe not found: {dataframe_path}")
                
                with open(dataframe_path, 'rb') as f:
                    normalized_dataframe = pickle.load(f)
                self.logger.info(f"DataFrame loaded from: {dataframe_path} - Shape: {normalized_dataframe.shape}")
                
                # Load scaler
                scaler_path = data_run_path / "scaler.pkl"
                if not scaler_path.exists():
                    raise FileNotFoundError(f"Scaler not found: {scaler_path}")
                
                scaler = joblib.load(scaler_path)
                self.logger.info(f"Scaler loaded from: {scaler_path}")
                
                # Load price_scaler
                price_scaler_path = data_run_path / "price_scaler.pkl"
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
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: load from Google Cloud Storage
                blob_name = f"{data_run_id}/data_run_metadata.yaml"
                
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
                # Local mode
                metadata_path = Path(f"{data_run_id}/data_run_metadata.yaml")
                
                if not metadata_path.exists():
                    raise FileNotFoundError(f"Data run metadata not found: {metadata_path}")
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f)
                
                self.logger.info(f"Data run metadata loaded from: {metadata_path}")
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error loading data run metadata for {data_run_id}: {str(e)}")
            raise
    
    def save_agent_checkpoint(self, agent: TransformerSACAgent, episode: int) -> str:
        """
        Save agent checkpoint for a specific episode asynchronously.
        
        Args:
            agent: The agent to save
            episode: Episode number for checkpoint naming
            
        Returns:
            Path where checkpoint will be saved
        """
        # Extract state dictionaries and move to CPU for multiprocessing safety
        # Using "Clean Save" principle: get underlying models without DDP wrappers
        agent_state = {
            'actor': self._to_cpu_state_dict(agent._get_actor_model().state_dict()),
            'critic_1': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_1).state_dict()),
            'critic_2': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_2).state_dict()),
            'critic_target_1': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_target_1).state_dict()),
            'critic_target_2': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_target_2).state_dict()),
            'actor_optimizer': self._to_cpu_state_dict(agent.actor_optimizer.state_dict()),
            'critic_1_optimizer': self._to_cpu_state_dict(agent.critic_1_optimizer.state_dict()),
            'critic_2_optimizer': self._to_cpu_state_dict(agent.critic_2_optimizer.state_dict()),
            'log_alpha': agent.log_alpha.detach().cpu(),
            'metadata': {
                'episode': episode + 1,
                'total_steps': agent.total_steps,
                'learning_steps': agent.learning_steps,
                'learn_alpha': agent.learn_alpha,
                'target_entropy': agent.target_entropy,
                'device': str(agent.device)
            }
        }
        
        # Add alpha optimizer if learnable
        if agent.learn_alpha:
            agent_state['alpha_optimizer'] = self._to_cpu_state_dict(agent.alpha_optimizer.state_dict())
        
        # Determine path and worker function based on storage mode
        if self.storage_mode == "gcp":
            path_prefix = f"training_runs/{self.run_id}/checkpoints/checkpoint_episode_{episode + 1}"
            args = (agent_state, path_prefix, self.gcp_config)
            target_worker = _save_worker_gcs
            checkpoint_path = f"gs://{self.gcs_utils.bucket_name}/{path_prefix}"
        else:
            # Ensure checkpoint directory exists for training runs
            checkpoint_dir = Path(f"training_runs/{self.run_id}/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path_prefix = str(checkpoint_dir / f"checkpoint_episode_{episode + 1}")
            args = (agent_state, path_prefix)
            target_worker = _save_worker_local
            checkpoint_path = path_prefix
        
        # Launch asynchronous save process
        save_process = Process(target=target_worker, args=args)
        save_process.start()
        
        self.logger.info(f"🚀 Guardado asíncrono del checkpoint del episodio {episode + 1} iniciado en segundo plano.")
        
        return checkpoint_path
    
    def load_agent_from_checkpoint(self, agent: TransformerSACAgent, checkpoint_prefix: str, reset_optimizers: bool = False) -> None:
        """
        Load agent from a checkpoint.
        
        Args:
            agent: The agent to load into
            checkpoint_prefix: Prefix/path of the checkpoint to load
            reset_optimizers: If True, skip loading optimizer states for fine-tuning
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_prefix}")
        
        if reset_optimizers:
            self.logger.warning("⚠️  MODO FINE-TUNING ACTIVADO: Los optimizadores serán reiniciados con los hiperparámetros actuales")
        
        try:
            if self.storage_mode == "gcp":
                # GCS mode: download files to temporary directory and load
                with tempfile.TemporaryDirectory() as temp_dir:
                    # List of component files to download
                    component_files = [
                        "actor.pth", "critic_1.pth", "critic_2.pth",
                        "critic_target_1.pth", "critic_target_2.pth",
                        "actor_optimizer.pth", "critic_1_optimizer.pth", "critic_2_optimizer.pth",
                        "log_alpha.pth", "metadata.pth"
                    ]
                    
                    # Add alpha optimizer if learnable
                    if agent.learn_alpha:
                        component_files.append("alpha_optimizer.pth")
                    
                    # Download each component file
                    for component_file in component_files:
                        gcs_blob_name = f"{checkpoint_prefix}/{checkpoint_prefix.split('/')[-1]}_{component_file}"
                        local_file_path = os.path.join(temp_dir, component_file)
                        
                        if not self.gcs_utils.download_file_from_gcs(gcs_blob_name, local_file_path):
                            if component_file == "alpha_optimizer.pth" and not agent.learn_alpha:
                                # Skip alpha optimizer if not learnable
                                continue
                            else:
                                raise FileNotFoundError(f"Failed to download checkpoint component: {gcs_blob_name}")
                    
                    # Load metadata first
                    metadata_path = os.path.join(temp_dir, "metadata.pth")
                    if os.path.exists(metadata_path):
                        metadata = torch.load(metadata_path, map_location=agent.device)
                        agent.total_steps = metadata.get('total_steps', 0)
                        agent.learning_steps = metadata.get('learning_steps', 0)
                        self.logger.info(f"Loaded metadata: episode={metadata.get('episode', 'unknown')}, steps={agent.total_steps}")
                    
                    # Load network state dictionaries
                    agent.actor.load_state_dict(torch.load(os.path.join(temp_dir, "actor.pth"), map_location=agent.device))
                    agent.critic_1.load_state_dict(torch.load(os.path.join(temp_dir, "critic_1.pth"), map_location=agent.device))
                    agent.critic_2.load_state_dict(torch.load(os.path.join(temp_dir, "critic_2.pth"), map_location=agent.device))
                    agent.critic_target_1.load_state_dict(torch.load(os.path.join(temp_dir, "critic_target_1.pth"), map_location=agent.device))
                    agent.critic_target_2.load_state_dict(torch.load(os.path.join(temp_dir, "critic_target_2.pth"), map_location=agent.device))
                    
                    # Load optimizer state dictionaries and log_alpha only if not in fine-tuning mode
                    if not reset_optimizers:
                        agent.actor_optimizer.load_state_dict(torch.load(os.path.join(temp_dir, "actor_optimizer.pth"), map_location=agent.device))
                        agent.critic_1_optimizer.load_state_dict(torch.load(os.path.join(temp_dir, "critic_1_optimizer.pth"), map_location=agent.device))
                        agent.critic_2_optimizer.load_state_dict(torch.load(os.path.join(temp_dir, "critic_2_optimizer.pth"), map_location=agent.device))
                        
                        # Load alpha and its optimizer
                        agent.log_alpha = torch.load(os.path.join(temp_dir, "log_alpha.pth"), map_location=agent.device)
                        if agent.learn_alpha and os.path.exists(os.path.join(temp_dir, "alpha_optimizer.pth")):
                            agent.alpha_optimizer.load_state_dict(torch.load(os.path.join(temp_dir, "alpha_optimizer.pth"), map_location=agent.device))
                        
            else:
                # Local mode: load directly from files
                prefix = Path(checkpoint_prefix)
                
                # Load metadata first
                metadata_path = f"{prefix}_metadata.pth"
                if os.path.exists(metadata_path):
                    metadata = torch.load(metadata_path, map_location=agent.device)
                    agent.total_steps = metadata.get('total_steps', 0)
                    agent.learning_steps = metadata.get('learning_steps', 0)
                    self.logger.info(f"Loaded metadata: episode={metadata.get('episode', 'unknown')}, steps={agent.total_steps}")
                
                # Load network state dictionaries
                agent.actor.load_state_dict(torch.load(f"{prefix}_actor.pth", map_location=agent.device))
                agent.critic_1.load_state_dict(torch.load(f"{prefix}_critic_1.pth", map_location=agent.device))
                agent.critic_2.load_state_dict(torch.load(f"{prefix}_critic_2.pth", map_location=agent.device))
                agent.critic_target_1.load_state_dict(torch.load(f"{prefix}_critic_target_1.pth", map_location=agent.device))
                agent.critic_target_2.load_state_dict(torch.load(f"{prefix}_critic_target_2.pth", map_location=agent.device))
                
                # Load optimizer state dictionaries and log_alpha only if not in fine-tuning mode
                if not reset_optimizers:
                    agent.actor_optimizer.load_state_dict(torch.load(f"{prefix}_actor_optimizer.pth", map_location=agent.device))
                    agent.critic_1_optimizer.load_state_dict(torch.load(f"{prefix}_critic_1_optimizer.pth", map_location=agent.device))
                    agent.critic_2_optimizer.load_state_dict(torch.load(f"{prefix}_critic_2_optimizer.pth", map_location=agent.device))
                    
                    # Load alpha and its optimizer
                    agent.log_alpha = torch.load(f"{prefix}_log_alpha.pth", map_location=agent.device)
                    if agent.learn_alpha and os.path.exists(f"{prefix}_alpha_optimizer.pth"):
                        agent.alpha_optimizer.load_state_dict(torch.load(f"{prefix}_alpha_optimizer.pth", map_location=agent.device))
            
            self.logger.info("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def save_best_model(self, agent: TransformerSACAgent) -> str:
        """
        Save the best performing model asynchronously.
        
        Args:
            agent: The agent to save
            
        Returns:
            Path where model will be saved
        """
        # Extract state dictionaries and move to CPU for multiprocessing safety
        # Using "Clean Save" principle: get underlying models without DDP wrappers
        agent_state = {
            'actor': self._to_cpu_state_dict(agent._get_actor_model().state_dict()),
            'critic_1': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_1).state_dict()),
            'critic_2': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_2).state_dict()),
            'critic_target_1': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_target_1).state_dict()),
            'critic_target_2': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_target_2).state_dict()),
            'actor_optimizer': self._to_cpu_state_dict(agent.actor_optimizer.state_dict()),
            'critic_1_optimizer': self._to_cpu_state_dict(agent.critic_1_optimizer.state_dict()),
            'critic_2_optimizer': self._to_cpu_state_dict(agent.critic_2_optimizer.state_dict()),
            'log_alpha': agent.log_alpha.detach().cpu(),
            'metadata': {
                'total_steps': agent.total_steps,
                'learning_steps': agent.learning_steps,
                'learn_alpha': agent.learn_alpha,
                'target_entropy': agent.target_entropy,
                'device': str(agent.device)
            }
        }
        
        # Add alpha optimizer if learnable
        if agent.learn_alpha:
            agent_state['alpha_optimizer'] = self._to_cpu_state_dict(agent.alpha_optimizer.state_dict())
        
        # Determine path and worker function based on storage mode
        if self.storage_mode == "gcp":
            path_prefix = f"training_runs/{self.run_id}/best_model"
            args = (agent_state, path_prefix, self.gcp_config)
            target_worker = _save_worker_gcs
            best_model_path = f"gs://{self.gcs_utils.bucket_name}/training_runs/{self.run_id}/best_model"
        else:
            # Ensure best model directory exists for training runs
            best_model_dir = Path(f"training_runs/{self.run_id}/best_model")
            best_model_dir.mkdir(parents=True, exist_ok=True)
            path_prefix = str(best_model_dir / "best_model")
            args = (agent_state, path_prefix)
            target_worker = _save_worker_local
            best_model_path = path_prefix
        
        # Launch asynchronous save process
        save_process = Process(target=target_worker, args=args)
        save_process.start()
        
        self.logger.info(f"🚀 Guardado asíncrono del mejor modelo iniciado en segundo plano.")
        
        return str(best_model_path)
    
    def save_final_model(self, agent: TransformerSACAgent) -> str:
        """
        Save the final model at the end of training asynchronously.
        
        Args:
            agent: The agent to save
            
        Returns:
            Path where model will be saved
        """
        # Extract state dictionaries and move to CPU for multiprocessing safety
        # Using "Clean Save" principle: get underlying models without DDP wrappers
        agent_state = {
            'actor': self._to_cpu_state_dict(agent._get_actor_model().state_dict()),
            'critic_1': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_1).state_dict()),
            'critic_2': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_2).state_dict()),
            'critic_target_1': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_target_1).state_dict()),
            'critic_target_2': self._to_cpu_state_dict(agent._get_critic_model(agent.critic_target_2).state_dict()),
            'actor_optimizer': self._to_cpu_state_dict(agent.actor_optimizer.state_dict()),
            'critic_1_optimizer': self._to_cpu_state_dict(agent.critic_1_optimizer.state_dict()),
            'critic_2_optimizer': self._to_cpu_state_dict(agent.critic_2_optimizer.state_dict()),
            'log_alpha': agent.log_alpha.detach().cpu(),
            'metadata': {
                'total_steps': agent.total_steps,
                'learning_steps': agent.learning_steps,
                'learn_alpha': agent.learn_alpha,
                'target_entropy': agent.target_entropy,
                'device': str(agent.device)
            }
        }
        
        # Add alpha optimizer if learnable
        if agent.learn_alpha:
            agent_state['alpha_optimizer'] = self._to_cpu_state_dict(agent.alpha_optimizer.state_dict())
        
        # Determine path and worker function based on storage mode
        if self.storage_mode == "gcp":
            path_prefix = f"training_runs/{self.run_id}/final_model"
            args = (agent_state, path_prefix, self.gcp_config)
            target_worker = _save_worker_gcs
            final_model_path = f"gs://{self.gcs_utils.bucket_name}/training_runs/{self.run_id}/final_model"
        else:
            # Ensure final model directory exists for training runs
            final_model_dir = Path(f"training_runs/{self.run_id}/final_model")
            final_model_dir.mkdir(parents=True, exist_ok=True)
            path_prefix = str(final_model_dir / "final_model")
            args = (agent_state, path_prefix)
            target_worker = _save_worker_local
            final_model_path = path_prefix
        
        # Launch asynchronous save process
        save_process = Process(target=target_worker, args=args)
        save_process.start()
        
        self.logger.info(f"🚀 Guardado asíncrono del modelo final iniciado en segundo plano.")
        
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
            self.base_path = f"gs://{self.gcs_bucket_name}/training_runs/{self.run_id}"
        else:
            self.base_path = f"training_runs/{self.run_id}"
        
        self.logger.info(f"RunManager context updated - run_id: {self.run_id}, base_path: {self.base_path}")

    
    
    @staticmethod
    def load_training_run_config(training_run_id: str, storage_mode: str = None, gcp_config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Load training run configuration without requiring a full RunManager instance.
        This method is useful when you need to load configuration to determine how to create a RunManager.
        
        Args:
            training_run_id: The training run ID for which to load the configuration
            storage_mode: Storage mode ('local' or 'gcp'). If None, tries to determine automatically
            gcp_config: GCP configuration dictionary (required if storage_mode is 'gcp')
            
        Returns:
            Dict containing the configuration, or None if file doesn't exist or error occurs
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading configuration for training run: {training_run_id}")
        
        # If storage_mode not provided, try to determine it
        if storage_mode is None:
            # Try local first
            local_config_path = Path(f"training_runs/{training_run_id}/config_run.yaml")
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
                
                # Construct GCS blob name for training runs
                blob_name = f"training_runs/{training_run_id}/config_run.yaml"
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
                # Local storage mode
                config_path = Path(f"training_runs/{training_run_id}/config_run.yaml")
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
