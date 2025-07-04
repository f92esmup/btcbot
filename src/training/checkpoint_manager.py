"""
CheckpointManager: Specialized manager for agent checkpoint operations.

This module provides centralized management for agent checkpoint operations including:
- Agent checkpoint saving and loading
- Best model persistence
- Final model saving
- Checkpoint discovery and metadata handling
"""

import os
import re
import tempfile
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from multiprocessing import Process

from src.agente.agent import TransformerSACAgent
from src.configuration.constants import (
    FILE_ACTOR_PTH, FILE_CRITIC_1_PTH, FILE_CRITIC_2_PTH,
    FILE_CRITIC_TARGET_1_PTH, FILE_CRITIC_TARGET_2_PTH,
    FILE_ACTOR_OPTIMIZER_PTH, FILE_CRITIC_1_OPTIMIZER_PTH, FILE_CRITIC_2_OPTIMIZER_PTH,
    FILE_ALPHA_OPTIMIZER_PTH, FILE_LOG_ALPHA_PTH, FILE_METADATA_PTH,
    EXT_PTH
)


def _save_worker_local(agent_state_dicts: Dict[str, Any], path_prefix: str) -> None:
    """
    Worker function for saving agent state dictionaries to local storage.
    
    Args:
        agent_state_dicts: Dictionary containing all agent state dictionaries
        path_prefix: Path prefix for saving files
    """
    try:
        # Save networks
        torch.save(agent_state_dicts['actor'], f"{path_prefix}/{FILE_ACTOR_PTH}")
        torch.save(agent_state_dicts['critic_1'], f"{path_prefix}/{FILE_CRITIC_1_PTH}")
        torch.save(agent_state_dicts['critic_2'], f"{path_prefix}/{FILE_CRITIC_2_PTH}")
        torch.save(agent_state_dicts['critic_target_1'], f"{path_prefix}/{FILE_CRITIC_TARGET_1_PTH}")
        torch.save(agent_state_dicts['critic_target_2'], f"{path_prefix}/{FILE_CRITIC_TARGET_2_PTH}")
        
        # Save optimizers
        torch.save(agent_state_dicts['actor_optimizer'], f"{path_prefix}/{FILE_ACTOR_OPTIMIZER_PTH}")
        torch.save(agent_state_dicts['critic_1_optimizer'], f"{path_prefix}/{FILE_CRITIC_1_OPTIMIZER_PTH}")
        torch.save(agent_state_dicts['critic_2_optimizer'], f"{path_prefix}/{FILE_CRITIC_2_OPTIMIZER_PTH}")
        
        # Save alpha and its optimizer if present
        torch.save(agent_state_dicts['log_alpha'], f"{path_prefix}/{FILE_LOG_ALPHA_PTH}")
        if 'alpha_optimizer' in agent_state_dicts:
            torch.save(agent_state_dicts['alpha_optimizer'], f"{path_prefix}/{FILE_ALPHA_OPTIMIZER_PTH}")
        
        # Save metadata
        torch.save(agent_state_dicts['metadata'], f"{path_prefix}/{FILE_METADATA_PTH}")
        
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
            path_prefix = Path(temp_dir)
            torch.save(agent_state_dicts['actor'], path_prefix / FILE_ACTOR_PTH)
            torch.save(agent_state_dicts['critic_1'], path_prefix / FILE_CRITIC_1_PTH)
            torch.save(agent_state_dicts['critic_2'], path_prefix / FILE_CRITIC_2_PTH)
            torch.save(agent_state_dicts['critic_target_1'], path_prefix / FILE_CRITIC_TARGET_1_PTH)
            torch.save(agent_state_dicts['critic_target_2'], path_prefix / FILE_CRITIC_TARGET_2_PTH)
            torch.save(agent_state_dicts['actor_optimizer'], path_prefix / FILE_ACTOR_OPTIMIZER_PTH)
            torch.save(agent_state_dicts['critic_1_optimizer'], path_prefix / FILE_CRITIC_1_OPTIMIZER_PTH)
            torch.save(agent_state_dicts['critic_2_optimizer'], path_prefix / FILE_CRITIC_2_OPTIMIZER_PTH)
            torch.save(agent_state_dicts['log_alpha'], path_prefix / FILE_LOG_ALPHA_PTH)
            if 'alpha_optimizer' in agent_state_dicts:
                torch.save(agent_state_dicts['alpha_optimizer'], path_prefix / FILE_ALPHA_OPTIMIZER_PTH)
            torch.save(agent_state_dicts['metadata'], path_prefix / FILE_METADATA_PTH)

            # Upload each file from the temporary directory to the GCS prefix
            success_count = 0
            total_files = 0
            for local_file_path in path_prefix.glob(f'*{EXT_PTH}'):
                total_files += 1
                gcs_blob_name = f"{gcs_prefix}/{local_file_path.name}"
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


class CheckpointManager:
    """
    Specialized manager for agent checkpoint operations.
    
    Handles all checkpoint-related operations including saving, loading,
    and discovery across both local and GCS storage modes.
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
    
    def __init__(self, storage_mode: str = "local", gcp_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CheckpointManager.
        
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
    
    def _get_training_run_prefix(self, training_run_id: str) -> str:
        """
        Get the path prefix for a specific training run.
        
        Args:
            training_run_id: The unique identifier for the training run
            
        Returns:
            The path prefix string for the training run (e.g., "training_runs/ID_ABC")
        """
        return f"training_runs/{training_run_id}"
    
    def find_latest_checkpoint(self, training_run_id: str) -> Optional[Tuple[str, int]]:
        """
        Find the latest checkpoint in a specific training run.
        
        Args:
            training_run_id: The training run ID to search for checkpoints
            
        Returns:
            Tuple of (checkpoint_path, episode_number) if found, None otherwise
        """
        self.logger.info(f"Searching for checkpoints in training run: {training_run_id}")
        
        # Get training run prefix using helper
        prefix = self._get_training_run_prefix(training_run_id)
        
        try:
            if self.storage_mode == "gcp":
                # GCP mode: search in GCS
                checkpoint_prefix = f"{prefix}/checkpoints/"
                self.logger.info(f"Searching for checkpoint metadata in GCS prefix: {checkpoint_prefix}")

                # Search for checkpoint metadata files in the specific run
                bucket = self.gcs_utils._get_bucket()
                blobs = bucket.list_blobs(prefix=checkpoint_prefix)
                
                # Regex para encontrar el número de episodio en la ruta
                # Ej: .../checkpoints/checkpoint_episode_123/metadata.pth
                metadata_pattern = re.compile(r"/checkpoint_episode_(\d+)/metadata\.pth$")
                self.logger.info(f"Using regex pattern: {metadata_pattern.pattern}")

                latest_episode = -1
                
                # Log all found blobs for debugging
                found_blobs = [blob.name for blob in blobs]
                if not found_blobs:
                    self.logger.warning(f"No blobs found in GCS prefix: {checkpoint_prefix}")
                else:
                    self.logger.info(f"Found {len(found_blobs)} blobs in prefix. Checking for matches...")
                    for blob_name in found_blobs:
                        self.logger.info(f"  - Checking blob: {blob_name}")

                # Re-initialize blobs iterator
                blobs = bucket.list_blobs(prefix=checkpoint_prefix)

                for blob in blobs:
                    match = metadata_pattern.search(blob.name)
                    if match:
                        episode_num = int(match.group(1))
                        self.logger.info(f"    -> Match found! Episode: {episode_num}")
                        if episode_num > latest_episode:
                            latest_episode = episode_num
                
                if latest_episode != -1:
                    latest_checkpoint_prefix = f"{prefix}/checkpoints/checkpoint_episode_{latest_episode}"
                    self.logger.info(f"Último checkpoint encontrado en GCS: {latest_checkpoint_prefix} (episodio {latest_episode})")
                    return (latest_checkpoint_prefix, latest_episode)
                else:
                    self.logger.info(f"No se encontraron checkpoints válidos en el training run {training_run_id} en GCS")
                    return None
                    
            else:
                # Local mode: search in local directory
                checkpoint_dir = Path(f"{prefix}/checkpoints/")
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
                    self.logger.info(f"No valid checkpoints found in training run {training_run_id}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error searching for checkpoints in training run {training_run_id}: {e}")
            return None
    
    def save_agent_checkpoint(self, training_run_id: str, agent: TransformerSACAgent, episode: int) -> str:
        """
        Save agent checkpoint for a specific episode asynchronously.
        
        Args:
            training_run_id: The ID of the training run to save checkpoint for
            agent: The agent to save
            episode: Episode number for checkpoint naming
            
        Returns:
            Path where checkpoint will be saved
        """
        # Get training run prefix using helper
        prefix = self._get_training_run_prefix(training_run_id)
        
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
            path_prefix = f"{prefix}/checkpoints/checkpoint_episode_{episode + 1}"
            args = (agent_state, path_prefix, self.gcp_config)
            target_worker = _save_worker_gcs
            checkpoint_path = f"gs://{self.gcs_utils.bucket_name}/{path_prefix}"
        else:
            # Ensure checkpoint directory exists using prefix
            path_prefix = Path(f"{prefix}/checkpoints/checkpoint_episode_{episode + 1}")
            path_prefix.mkdir(parents=True, exist_ok=True)
            args = (agent_state, str(path_prefix))
            target_worker = _save_worker_local
            checkpoint_path = str(path_prefix)
        
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
                        gcs_blob_name = f"{checkpoint_prefix}/{component_file}"
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
                path_prefix = Path(checkpoint_prefix)
                
                # Load metadata first
                metadata_path = path_prefix / "metadata.pth"
                if metadata_path.exists():
                    metadata = torch.load(metadata_path, map_location=agent.device)
                    agent.total_steps = metadata.get('total_steps', 0)
                    agent.learning_steps = metadata.get('learning_steps', 0)
                    self.logger.info(f"Loaded metadata: episode={metadata.get('episode', 'unknown')}, steps={agent.total_steps}")
                
                # Load network state dictionaries
                agent.actor.load_state_dict(torch.load(path_prefix / "actor.pth", map_location=agent.device))
                agent.critic_1.load_state_dict(torch.load(path_prefix / "critic_1.pth", map_location=agent.device))
                agent.critic_2.load_state_dict(torch.load(path_prefix / "critic_2.pth", map_location=agent.device))
                agent.critic_target_1.load_state_dict(torch.load(path_prefix / "critic_target_1.pth", map_location=agent.device))
                agent.critic_target_2.load_state_dict(torch.load(path_prefix / "critic_target_2.pth", map_location=agent.device))
                
                # Load optimizer state dictionaries and log_alpha only if not in fine-tuning mode
                if not reset_optimizers:
                    agent.actor_optimizer.load_state_dict(torch.load(path_prefix / "actor_optimizer.pth", map_location=agent.device))
                    agent.critic_1_optimizer.load_state_dict(torch.load(path_prefix / "critic_1_optimizer.pth", map_location=agent.device))
                    agent.critic_2_optimizer.load_state_dict(torch.load(path_prefix / "critic_2_optimizer.pth", map_location=agent.device))
                    
                    # Load alpha and its optimizer
                    agent.log_alpha = torch.load(path_prefix / "log_alpha.pth", map_location=agent.device)
                    if agent.learn_alpha and (path_prefix / "alpha_optimizer.pth").exists():
                        agent.alpha_optimizer.load_state_dict(torch.load(path_prefix / "alpha_optimizer.pth", map_location=agent.device))
            
            self.logger.info("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def save_best_model(self, training_run_id: str, agent: TransformerSACAgent) -> str:
        """
        Save the best performing model asynchronously.
        
        Args:
            training_run_id: The ID of the training run to save best model for
            agent: The agent to save
            
        Returns:
            Path where model will be saved
        """
        # Get training run prefix using helper
        prefix = self._get_training_run_prefix(training_run_id)
        
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
            path_prefix = f"{prefix}/best_model"
            args = (agent_state, path_prefix, self.gcp_config)
            target_worker = _save_worker_gcs
            best_model_path = f"gs://{self.gcs_utils.bucket_name}/{path_prefix}"
        else:
            # Ensure best model directory exists using prefix
            path_prefix = Path(f"{prefix}/best_model")
            path_prefix.mkdir(parents=True, exist_ok=True)
            args = (agent_state, str(path_prefix))
            target_worker = _save_worker_local
            best_model_path = str(path_prefix)
        
        # Launch asynchronous save process
        save_process = Process(target=target_worker, args=args)
        save_process.start()
        
        self.logger.info(f"🚀 Guardado asíncrono del mejor modelo iniciado en segundo plano.")
        
        return str(best_model_path)
    
    def save_final_model(self, training_run_id: str, agent: TransformerSACAgent) -> str:
        """
        Save the final model at the end of training asynchronously.
        
        Args:
            training_run_id: The ID of the training run to save final model for
            agent: The agent to save
            
        Returns:
            Path where model will be saved
        """
        # Get training run prefix using helper
        prefix = self._get_training_run_prefix(training_run_id)
        
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
            path_prefix = f"{prefix}/final_model"
            args = (agent_state, path_prefix, self.gcp_config)
            target_worker = _save_worker_gcs
            final_model_path = f"gs://{self.gcs_utils.bucket_name}/{path_prefix}"
        else:
            # Ensure final model directory exists using prefix
            path_prefix = Path(f"{prefix}/final_model")
            path_prefix.mkdir(parents=True, exist_ok=True)
            args = (agent_state, str(path_prefix))
            target_worker = _save_worker_local
            final_model_path = str(path_prefix)
        
        # Launch asynchronous save process
        save_process = Process(target=target_worker, args=args)
        save_process.start()
        
        self.logger.info(f"🚀 Guardado asíncrono del modelo final iniciado en segundo plano.")
        
        return str(final_model_path)
    
    def upload_tensorboard_logs(self, local_log_dir: str, training_run_id: str) -> None:
        """
        Upload TensorBoard logs directory to Google Cloud Storage.
        
        Args:
            local_log_dir: Local path to the TensorBoard logs directory
            training_run_id: The training run ID to organize logs in GCS
        """
        self.logger.info(f"Processing TensorBoard logs upload for training run: {training_run_id}")
        
        # Check if storage mode is GCP
        if self.storage_mode != "gcp":
            self.logger.info("Storage mode is not 'gcp'. Skipping TensorBoard logs upload.")
            return
        
        # Check if local log directory exists
        local_path = Path(local_log_dir)
        if not local_path.exists():
            self.logger.warning(f"Local TensorBoard log directory does not exist: {local_log_dir}")
            return
        
        if not local_path.is_dir():
            self.logger.warning(f"Local TensorBoard log path is not a directory: {local_log_dir}")
            return
        
        # Check if directory has any content
        log_files = list(local_path.glob('**/*'))
        if not log_files:
            self.logger.warning(f"Local TensorBoard log directory is empty: {local_log_dir}")
            return
        
        self.logger.info(f"Found {len(log_files)} files to upload from: {local_log_dir}")
        
        try:
            # Get training run prefix and construct TensorBoard destination
            prefix = self._get_training_run_prefix(training_run_id)
            gcs_destination_prefix = f"{prefix}/tensorboard"
            
            self.logger.info(f"Uploading TensorBoard logs to GCS prefix: {gcs_destination_prefix}")
            
            # Upload directory using existing GCS utils method
            success = self.gcs_utils.upload_directory_to_gcs(
                local_directory_path=str(local_path),
                gcs_prefix=gcs_destination_prefix
            )
            
            if success:
                self.logger.info(f"✅ TensorBoard logs successfully uploaded to: gs://{self.gcs_bucket_name}/{gcs_destination_prefix}")
            else:
                self.logger.error(f"❌ Failed to upload TensorBoard logs to GCS")
                
        except Exception as e:
            self.logger.error(f"Error uploading TensorBoard logs for training run {training_run_id}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
