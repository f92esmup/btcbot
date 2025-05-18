"""
Training component for the BTC Trading Bot pipeline.

This script serves as the entry point for the Vertex AI Pipeline component
responsible for training the reinforcement learning agent using the
Soft Actor-Critic (SAC) algorithm with a Transformer feature extractor.
"""

import os
import argparse
import logging
import json
import yaml
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from google.cloud import storage
import kfp
from kfp.v2.dsl import Output, Model, Metrics, Artifact

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.sac.policies import SACPolicy

from src.environments.trading_env import TradingEnvironment
from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define custom callback for TensorBoard logging to GCS
class GCSTensorBoardCallback(CheckpointCallback):
    """
    Custom callback for saving TensorBoard logs to Google Cloud Storage.
    
    This callback extends the CheckpointCallback to save TensorBoard logs
    to GCS periodically during training.
    
    Attributes:
        tensorboard_log_dir (str): Local directory for TensorBoard logs.
        gcs_bucket (str): GCS bucket for storing the logs.
        gcs_path (str): Path within the GCS bucket.
        project_id (str): Google Cloud project ID.
        save_freq (int): Frequency (in steps) of saving logs to GCS.
    """
    
    def __init__(
        self,
        tensorboard_log_dir: str,
        gcs_bucket: str,
        gcs_path: str,
        project_id: str,
        save_freq: int = 1000,
        name_prefix: str = "rl_model"
    ):
        """
        Initialize the GCSTensorBoardCallback.
        
        Args:
            tensorboard_log_dir (str): Local directory for TensorBoard logs.
            gcs_bucket (str): GCS bucket for storing the logs.
            gcs_path (str): Path within the GCS bucket.
            project_id (str): Google Cloud project ID.
            save_freq (int, optional): Frequency (in steps) of saving logs to GCS.
                Defaults to 1000.
            name_prefix (str, optional): Prefix for saved model files.
                Defaults to "rl_model".
        """
        super().__init__(
            save_freq=save_freq,
            save_path=tensorboard_log_dir,
            name_prefix=name_prefix,
            save_replay_buffer=False,
            save_vecnormalize=None
        )
        self.tensorboard_log_dir = tensorboard_log_dir
        self.gcs_bucket = gcs_bucket
        self.gcs_path = gcs_path
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(gcs_bucket)
    
    def _on_step(self) -> bool:
        """
        Run at each step of training.
        
        If it's time to save (according to save_freq), save the model
        and upload TensorBoard logs to GCS.
        
        Returns:
            bool: Whether the callback returns True (continue training).
        """
        # First, let the parent class save the model checkpoint
        super_result = super()._on_step()
        
        # If it's time to save, also upload the TensorBoard logs to GCS
        if self.n_calls % self.save_freq == 0:
            logger.info(f"Uploading TensorBoard logs to GCS at step {self.n_calls}")
            self._upload_logs_to_gcs()
        
        return super_result
    
    def _upload_logs_to_gcs(self):
        """
        Upload TensorBoard logs to Google Cloud Storage.
        """
        try:
            # Walk through all files in the TensorBoard log directory
            for root, _, files in os.walk(self.tensorboard_log_dir):
                for file in files:
                    # Skip checkpoint files (they are handled by the parent class)
                    if ".zip" in file:
                        continue
                    
                    # Get local file path
                    local_path = os.path.join(root, file)
                    
                    # Get GCS path (preserve subdirectory structure)
                    relative_path = os.path.relpath(local_path, self.tensorboard_log_dir)
                    gcs_file_path = os.path.join(self.gcs_path, relative_path)
                    
                    # Upload file to GCS
                    blob = self.bucket.blob(gcs_file_path)
                    blob.upload_from_filename(local_path)
                    
            logger.info(f"Successfully uploaded TensorBoard logs to gs://{self.gcs_bucket}/{self.gcs_path}")
            
        except Exception as e:
            logger.error(f"Error uploading TensorBoard logs to GCS: {str(e)}")
    
    def on_training_end(self):
        """
        Run when training ends.
        
        Upload all remaining TensorBoard logs to GCS.
        """
        super().on_training_end()
        logger.info("Training ended. Uploading final TensorBoard logs to GCS.")
        self._upload_logs_to_gcs()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training component.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train RL agent for BTC trading')
    
    # Required arguments
    parser.add_argument('--project_id', type=str, required=True,
                        help='Google Cloud project ID')
    parser.add_argument('--input_data_uri', type=str, required=True,
                        help='GCS URI of the preprocessed data sequences')
    parser.add_argument('--output_model_dir', type=str, required=True,
                        help='Output directory for the trained model')
    parser.add_argument('--output_tensorboard_dir', type=str, required=True,
                        help='Output directory for TensorBoard logs')
    
    # Optional arguments - GCS and Cloud Storage
    parser.add_argument('--gcs_bucket', type=str, required=True,
                        help='GCS bucket for storing model artifacts')
    parser.add_argument('--gcs_model_prefix', type=str, default='models',
                        help='Prefix for model artifacts in GCS bucket')
    parser.add_argument('--gcs_tensorboard_prefix', type=str, default='tensorboard',
                        help='Prefix for TensorBoard logs in GCS bucket')
    parser.add_argument('--checkpoint_freq', type=int, default=10000,
                        help='Frequency (in steps) of saving model checkpoints')
    parser.add_argument('--input_checkpoint_gcs_uri', type=str, default=None,
                        help='GCS URI of model checkpoint to resume training from')
    
    # Optional arguments - Training parameters
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='Size of the replay buffer')
    parser.add_argument('--learning_starts', type=int, default=10000,
                        help='Number of steps before learning starts')
    parser.add_argument('--train_freq', type=int, default=1,
                        help='Update the model every `train_freq` steps')
    parser.add_argument('--gradient_steps', type=int, default=1,
                        help='Gradient steps to do at each update')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient for target networks')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--ent_coef', type=str, default='auto',
                        help='Entropy coefficient (can be "auto" or a float value)')
    parser.add_argument('--target_entropy', type=str, default='auto',
                        help='Target entropy for adaptive entropy coefficient (can be "auto" or a float value)')
    
    # Optional arguments - Environment parameters
    parser.add_argument('--initial_balance_usd', type=float, default=10000.0,
                        help='Initial balance in USD')
    parser.add_argument('--max_position_btc', type=float, default=1.0,
                        help='Maximum position size in BTC')
    parser.add_argument('--action_threshold', type=float, default=0.5,
                        help='Threshold for converting continuous actions to discrete decisions')
    parser.add_argument('--random_episode_start', type=bool, default=True,
                        help='Whether to start episodes at random points in the data')
    parser.add_argument('--episode_steps', type=int, default=1000,
                        help='Maximum number of steps per episode')
    parser.add_argument('--commission_rate', type=float, default=0.0004,
                        help='Trading commission rate')
    
    # Optional arguments - Transformer and Neural Network parameters
    parser.add_argument('--features_dim', type=int, default=256,
                        help='Output dimension of the feature extractor')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimension of the Transformer model')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_encoder_layers', type=int, default=2,
                        help='Number of Transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Dimension of feedforward network in Transformer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    
    # Optional arguments - Policy network parameters
    parser.add_argument('--net_arch', type=str, default='[256, 256]',
                        help='Architecture of the policy network as a JSON list')
    
    # Configuration file
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to YAML configuration file (can be a GCS path)')
    
    # KFP v2 Output artifacts
    parser.add_argument('--trained_model_output_path', type=str, required=True,
                        help='Output path for the trained model artifact')
    parser.add_argument('--tensorboard_log_output_path', type=str, required=True,
                        help='Output path for the TensorBoard logs artifact')
    parser.add_argument('--training_metrics_output_path', type=str, required=True,
                        help='Output path for the training metrics artifact')
    
    return parser.parse_args()


def load_config_from_gcs(gcs_path: str, project_id: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file stored in GCS.
    
    Args:
        gcs_path (str): GCS path to the YAML configuration file.
        project_id (str): Google Cloud project ID.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
        
    Raises:
        Exception: If there's an error loading the configuration.
    """
    try:
        # Parse bucket and blob path from GCS URI
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path[5:]  # Remove 'gs://' prefix
        
        bucket_name, blob_path = gcs_path.split('/', 1)
        
        # Initialize GCS client and download the file
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download as string and parse YAML
        yaml_content = blob.download_as_text()
        config = yaml.safe_load(yaml_content)
        
        logger.info(f"Successfully loaded configuration from {gcs_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from GCS: {str(e)}")
        raise


def download_file_from_gcs(gcs_uri: str, local_path: str, project_id: str) -> str:
    """
    Download a file from Google Cloud Storage to a local path.
    
    Args:
        gcs_uri (str): GCS URI of the file to download.
        local_path (str): Local path to save the file.
        project_id (str): Google Cloud project ID.
        
    Returns:
        str: Local path where the file was saved.
    """
    try:
        logger.info(f"Downloading file from {gcs_uri} to {local_path}")
        
        # Parse bucket and blob path from GCS URI
        if gcs_uri.startswith('gs://'):
            gcs_uri = gcs_uri[5:]  # Remove 'gs://' prefix
        
        bucket_name, blob_path = gcs_uri.split('/', 1)
        
        # Initialize GCS client and download the file
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file
        blob.download_to_filename(local_path)
        
        logger.info(f"Successfully downloaded file to {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Error downloading file from GCS: {str(e)}")
        raise


def upload_file_to_gcs(local_path: str, gcs_uri: str, project_id: str) -> str:
    """
    Upload a file from local path to Google Cloud Storage.
    
    Args:
        local_path (str): Local path of the file to upload.
        gcs_uri (str): GCS URI where to upload the file.
        project_id (str): Google Cloud project ID.
        
    Returns:
        str: GCS URI where the file was uploaded.
    """
    try:
        logger.info(f"Uploading file from {local_path} to {gcs_uri}")
        
        # Parse bucket and blob path from GCS URI
        if gcs_uri.startswith('gs://'):
            gcs_uri = gcs_uri[5:]  # Remove 'gs://' prefix
        
        bucket_name, blob_path = gcs_uri.split('/', 1)
        
        # Initialize GCS client and upload the file
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Upload the file
        blob.upload_from_filename(local_path)
        
        logger.info(f"Successfully uploaded file to gs://{bucket_name}/{blob_path}")
        return f"gs://{bucket_name}/{blob_path}"
        
    except Exception as e:
        logger.error(f"Error uploading file to GCS: {str(e)}")
        raise


def setup_training_directories(output_model_dir: str, output_tensorboard_dir: str) -> Tuple[str, str]:
    """
    Set up local directories for model training and TensorBoard logs.
    
    Args:
        output_model_dir (str): Directory for saving model checkpoints.
        output_tensorboard_dir (str): Directory for saving TensorBoard logs.
        
    Returns:
        Tuple[str, str]: Paths to the created directories.
    """
    try:
        # Create output directories
        os.makedirs(output_model_dir, exist_ok=True)
        os.makedirs(output_tensorboard_dir, exist_ok=True)
        
        logger.info(f"Created output directories: {output_model_dir}, {output_tensorboard_dir}")
        return output_model_dir, output_tensorboard_dir
        
    except Exception as e:
        logger.error(f"Error setting up training directories: {str(e)}")
        raise


def create_training_callbacks(
    checkpoint_freq: int,
    model_dir: str,
    tensorboard_dir: str,
    gcs_bucket: str,
    gcs_model_prefix: str,
    gcs_tensorboard_prefix: str,
    project_id: str
) -> List:
    """
    Create callbacks for model training.
    
    Args:
        checkpoint_freq (int): Frequency (in steps) of saving model checkpoints.
        model_dir (str): Directory for saving model checkpoints.
        tensorboard_dir (str): Directory for saving TensorBoard logs.
        gcs_bucket (str): GCS bucket for storing model artifacts.
        gcs_model_prefix (str): Prefix for model artifacts in GCS bucket.
        gcs_tensorboard_prefix (str): Prefix for TensorBoard logs in GCS bucket.
        project_id (str): Google Cloud project ID.
        
    Returns:
        List: List of callbacks for model training.
    """
    try:
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_dir,
            name_prefix="sac_model",
            save_replay_buffer=True,
            save_vecnormalize=None
        )
        
        # Create GCS TensorBoard callback
        gcs_tensorboard_callback = GCSTensorBoardCallback(
            tensorboard_log_dir=tensorboard_dir,
            gcs_bucket=gcs_bucket,
            gcs_path=gcs_tensorboard_prefix,
            project_id=project_id,
            save_freq=checkpoint_freq,
            name_prefix="sac_model"
        )
        
        logger.info(f"Created training callbacks with checkpoint frequency {checkpoint_freq}")
        return [checkpoint_callback, gcs_tensorboard_callback]
        
    except Exception as e:
        logger.error(f"Error creating training callbacks: {str(e)}")
        raise


def create_sac_agent(
    env,
    learning_rate: float,
    batch_size: int,
    buffer_size: int,
    learning_starts: int,
    train_freq: int,
    gradient_steps: int,
    tau: float,
    gamma: float,
    ent_coef: str,
    target_entropy: str,
    tensorboard_log: str,
    policy_kwargs: Dict[str, Any]
) -> SAC:
    """
    Create a Soft Actor-Critic (SAC) agent.
    
    Args:
        env: Training environment.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size for training.
        buffer_size (int): Size of the replay buffer.
        learning_starts (int): Number of steps before learning starts.
        train_freq (int): Update the model every `train_freq` steps.
        gradient_steps (int): Gradient steps to do at each update.
        tau (float): Soft update coefficient for target networks.
        gamma (float): Discount factor.
        ent_coef (str): Entropy coefficient (can be "auto" or a float value).
        target_entropy (str): Target entropy for adaptive entropy coefficient
            (can be "auto" or a float value).
        tensorboard_log (str): Directory for saving TensorBoard logs.
        policy_kwargs (Dict[str, Any]): Additional arguments for the policy network.
        
    Returns:
        SAC: Initialized SAC agent.
    """
    try:
        # Parse ent_coef
        if ent_coef.lower() == 'auto':
            ent_coef_value = 'auto'
        else:
            ent_coef_value = float(ent_coef)
        
        # Parse target_entropy
        if target_entropy.lower() == 'auto':
            target_entropy_value = 'auto'
        else:
            target_entropy_value = float(target_entropy)
        
        # Create SAC agent
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef_value,
            target_entropy=target_entropy_value,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        logger.info(f"Created SAC agent with learning rate {learning_rate}, batch size {batch_size}")
        return model
        
    except Exception as e:
        logger.error(f"Error creating SAC agent: {str(e)}")
        raise


def train_agent(
    model: SAC,
    total_timesteps: int,
    callbacks: List,
    reset_num_timesteps: bool = True
) -> SAC:
    """
    Train the SAC agent.
    
    Args:
        model (SAC): SAC agent to train.
        total_timesteps (int): Total timesteps for training.
        callbacks (List): List of callbacks for training.
        reset_num_timesteps (bool, optional): Whether to reset the number of timesteps.
            Defaults to True.
        
    Returns:
        SAC: Trained SAC agent.
    """
    try:
        logger.info(f"Starting training for {total_timesteps} timesteps")
        start_time = time.time()
        
        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training agent: {str(e)}")
        raise


def save_trained_model(
    model: SAC,
    model_dir: str,
    model_output: Output[Model],
    training_metrics: Output[Metrics]
) -> str:
    """
    Save the trained model and record training metrics.
    
    Args:
        model (SAC): Trained SAC agent.
        model_dir (str): Directory for saving the model.
        model_output (Output[Model]): KFP output parameter for the model artifact.
        training_metrics (Output[Metrics]): KFP output parameter for training metrics.
        
    Returns:
        str: Path to the saved model.
    """
    try:
        # Save the model
        model_path = os.path.join(model_dir, "final_model.zip")
        model.save(model_path)
        
        # Also save the replay buffer
        buffer_path = os.path.join(model_dir, "replay_buffer.pkl")
        model.save_replay_buffer(buffer_path)
        
        logger.info(f"Saved trained model to {model_path}")
        
        # Save model to the model_output artifact
        model_uri = model_path
        with open(model_output.path, 'w') as f:
            f.write(model_uri)
        
        # Record training metrics
        metrics = {
            'total_timesteps': model.num_timesteps,
            'learning_rate': float(model.learning_rate),
            'entropy_coefficient': float(model.ent_coef) if not isinstance(model.ent_coef, str) else 'auto',
            'batch_size': model.batch_size,
            'training_completion_time': datetime.now().isoformat()
        }
        
        with open(training_metrics.path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Set metadata fields for discoverability
        model_output.metadata['framework'] = 'stable-baselines3'
        model_output.metadata['algorithm'] = 'SAC'
        model_output.metadata['total_timesteps'] = model.num_timesteps
        
        training_metrics.metadata['total_timesteps'] = model.num_timesteps
        training_metrics.metadata['learning_rate'] = float(model.learning_rate)
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error saving trained model: {str(e)}")
        raise


def run_training(
    project_id: str,
    input_data_uri: str,
    output_model_dir: str,
    output_tensorboard_dir: str,
    gcs_bucket: str,
    gcs_model_prefix: str,
    gcs_tensorboard_prefix: str,
    input_checkpoint_gcs_uri: Optional[str] = None,
    total_timesteps: int = 1000000,
    checkpoint_freq: int = 10000,
    learning_rate: float = 0.0003,
    batch_size: int = 256,
    buffer_size: int = 1000000,
    learning_starts: int = 10000,
    train_freq: int = 1,
    gradient_steps: int = 1,
    tau: float = 0.005,
    gamma: float = 0.99,
    ent_coef: str = 'auto',
    target_entropy: str = 'auto',
    initial_balance_usd: float = 10000.0,
    max_position_btc: float = 1.0,
    action_threshold: float = 0.5,
    random_episode_start: bool = True,
    episode_steps: int = 1000,
    commission_rate: float = 0.0004,
    features_dim: int = 256,
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    activation: str = 'relu',
    net_arch: List[int] = [256, 256],
    model_output: Output[Model] = None,
    tensorboard_log_output: Output[Artifact] = None,
    training_metrics: Output[Metrics] = None
) -> str:
    """
    Run the complete training pipeline.
    
    Args:
        project_id (str): Google Cloud project ID.
        input_data_uri (str): GCS URI of the preprocessed data sequences.
        output_model_dir (str): Directory for saving the model.
        output_tensorboard_dir (str): Directory for saving TensorBoard logs.
        gcs_bucket (str): GCS bucket for storing artifacts.
        gcs_model_prefix (str): Prefix for model artifacts in GCS.
        gcs_tensorboard_prefix (str): Prefix for TensorBoard logs in GCS.
        input_checkpoint_gcs_uri (str, optional): GCS URI of a checkpoint to resume training from.
        total_timesteps (int, optional): Total timesteps for training.
        checkpoint_freq (int, optional): Frequency of saving checkpoints.
        learning_rate (float, optional): Learning rate.
        batch_size (int, optional): Batch size for training.
        buffer_size (int, optional): Size of the replay buffer.
        learning_starts (int, optional): Steps before learning starts.
        train_freq (int, optional): Update frequency.
        gradient_steps (int, optional): Gradient steps per update.
        tau (float, optional): Soft update coefficient.
        gamma (float, optional): Discount factor.
        ent_coef (str, optional): Entropy coefficient.
        target_entropy (str, optional): Target entropy.
        initial_balance_usd (float, optional): Initial balance.
        max_position_btc (float, optional): Maximum position size.
        action_threshold (float, optional): Action threshold.
        random_episode_start (bool, optional): Random episode start.
        episode_steps (int, optional): Steps per episode.
        commission_rate (float, optional): Commission rate.
        features_dim (int, optional): Feature dimension.
        d_model (int, optional): Transformer dimension.
        n_heads (int, optional): Number of attention heads.
        n_encoder_layers (int, optional): Number of encoder layers.
        dim_feedforward (int, optional): Feedforward dimension.
        dropout (float, optional): Dropout rate.
        activation (str, optional): Activation function.
        net_arch (List[int], optional): Network architecture.
        model_output (Output[Model], optional): KFP model output.
        tensorboard_log_output (Output[Artifact], optional): KFP tensorboard log output.
        training_metrics (Output[Metrics], optional): KFP metrics output.
        
    Returns:
        str: Path to the saved model.
    """
    try:
        logger.info("Starting training pipeline")
        
        # Setup directories
        setup_training_directories(output_model_dir, output_tensorboard_dir)
        
        # Create the trading environment
        env = TradingEnvironment(
            project_id=project_id,
            gcs_processed_data_uri=input_data_uri,
            initial_balance_usd=initial_balance_usd,
            max_position_btc=max_position_btc,
            action_threshold=action_threshold,
            random_episode_start=random_episode_start,
            episode_steps=episode_steps,
            commission_rate=commission_rate
        )
        
        # Wrap the environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Create policy_kwargs for the Transformer feature extractor
        policy_kwargs = {
            'features_extractor_class': CustomTransformerFeatureExtractor,
            'features_extractor_kwargs': {
                'features_dim': features_dim,
                'd_model': d_model,
                'n_heads': n_heads,
                'n_encoder_layers': n_encoder_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'activation': activation
            },
            'net_arch': net_arch
        }
        
        # Create callbacks
        callbacks = create_training_callbacks(
            checkpoint_freq=checkpoint_freq,
            model_dir=output_model_dir,
            tensorboard_dir=output_tensorboard_dir,
            gcs_bucket=gcs_bucket,
            gcs_model_prefix=gcs_model_prefix,
            gcs_tensorboard_prefix=gcs_tensorboard_prefix,
            project_id=project_id
        )
        
        # Check if we need to resume from a checkpoint
        local_checkpoint_path = None
        if input_checkpoint_gcs_uri:
            local_checkpoint_path = os.path.join(output_model_dir, "checkpoint.zip")
            download_file_from_gcs(input_checkpoint_gcs_uri, local_checkpoint_path, project_id)
            
            # Also look for a replay buffer file
            replay_buffer_gcs_uri = input_checkpoint_gcs_uri.replace(".zip", "_replay_buffer.pkl")
            try:
                local_buffer_path = os.path.join(output_model_dir, "checkpoint_replay_buffer.pkl")
                download_file_from_gcs(replay_buffer_gcs_uri, local_buffer_path, project_id)
                has_replay_buffer = True
            except:
                has_replay_buffer = False
                logger.warning(f"No replay buffer found at {replay_buffer_gcs_uri}")
        
        # Create or load the agent
        if local_checkpoint_path and os.path.exists(local_checkpoint_path):
            logger.info(f"Loading agent from checkpoint: {local_checkpoint_path}")
            model = SAC.load(
                local_checkpoint_path,
                env=env,
                tensorboard_log=output_tensorboard_dir,
                custom_objects={
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'buffer_size': buffer_size
                }
            )
            
            # Load replay buffer if available
            if has_replay_buffer and os.path.exists(local_buffer_path):
                logger.info(f"Loading replay buffer from: {local_buffer_path}")
                model.load_replay_buffer(local_buffer_path)
            
            reset_num_timesteps = False
        else:
            logger.info("Creating new SAC agent")
            model = create_sac_agent(
                env=env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                tau=tau,
                gamma=gamma,
                ent_coef=ent_coef,
                target_entropy=target_entropy,
                tensorboard_log=output_tensorboard_dir,
                policy_kwargs=policy_kwargs
            )
            reset_num_timesteps = True
        
        # Train the agent
        trained_model = train_agent(
            model=model,
            total_timesteps=total_timesteps,
            callbacks=callbacks,
            reset_num_timesteps=reset_num_timesteps
        )
        
        # Save the final model
        model_path = save_trained_model(
            model=trained_model,
            model_dir=output_model_dir,
            model_output=model_output,
            training_metrics=training_metrics
        )
        
        # Save the TensorBoard logs URI for the output artifact
        if tensorboard_log_output:
            tensorboard_gcs_uri = f"gs://{gcs_bucket}/{gcs_tensorboard_prefix}"
            with open(tensorboard_log_output.path, 'w') as f:
                f.write(tensorboard_gcs_uri)
            
            tensorboard_log_output.metadata['gcs_uri'] = tensorboard_gcs_uri
        
        logger.info("Training pipeline completed successfully")
        return model_path
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        
        # Try to capture error information in metrics output
        if training_metrics:
            error_info = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(training_metrics.path, 'w') as f:
                json.dump(error_info, f, indent=2)
        
        raise


def main():
    """
    Main entry point for the training component.
    
    Parse command-line arguments and run the training pipeline.
    """
    args = parse_arguments()
    
    try:
        # Load configuration from file if specified
        config = {}
        if args.config_file:
            if args.config_file.startswith('gs://'):
                config = load_config_from_gcs(args.config_file, args.project_id)
            else:
                with open(args.config_file, 'r') as f:
                    config = yaml.safe_load(f)
        
        # Parse net_arch
        if isinstance(args.net_arch, str):
            try:
                net_arch = json.loads(args.net_arch)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse net_arch: {args.net_arch}. Using default [256, 256].")
                net_arch = [256, 256]
        else:
            net_arch = [256, 256]
        
        # Parse ent_coef
        ent_coef = args.ent_coef
        if ent_coef.lower() != 'auto':
            try:
                float(ent_coef)
            except ValueError:
                logger.warning(f"Could not parse ent_coef: {ent_coef}. Using 'auto'.")
                ent_coef = 'auto'
        
        # Create the output parameters
        model_output = Output(type=Model, path=args.trained_model_output_path)
        tensorboard_log_output = Output(type=Artifact, path=args.tensorboard_log_output_path)
        training_metrics = Output(type=Metrics, path=args.training_metrics_output_path)
        
        # Command line arguments take precedence over config file
        # Use a dictionary with default values from args, then update with config
        run_params = vars(args)
        
        # Update with config values only where args are None or not specified
        for key in config:
            if key in run_params and run_params[key] is None:
                run_params[key] = config[key]
        
        # Run training
        model_path = run_training(
            project_id=run_params['project_id'],
            input_data_uri=run_params['input_data_uri'],
            output_model_dir=run_params['output_model_dir'],
            output_tensorboard_dir=run_params['output_tensorboard_dir'],
            gcs_bucket=run_params['gcs_bucket'],
            gcs_model_prefix=run_params['gcs_model_prefix'],
            gcs_tensorboard_prefix=run_params['gcs_tensorboard_prefix'],
            input_checkpoint_gcs_uri=run_params['input_checkpoint_gcs_uri'],
            total_timesteps=run_params['total_timesteps'],
            checkpoint_freq=run_params['checkpoint_freq'],
            learning_rate=run_params['learning_rate'],
            batch_size=run_params['batch_size'],
            buffer_size=run_params['buffer_size'],
            learning_starts=run_params['learning_starts'],
            train_freq=run_params['train_freq'],
            gradient_steps=run_params['gradient_steps'],
            tau=run_params['tau'],
            gamma=run_params['gamma'],
            ent_coef=ent_coef,
            target_entropy=run_params['target_entropy'],
            initial_balance_usd=run_params['initial_balance_usd'],
            max_position_btc=run_params['max_position_btc'],
            action_threshold=run_params['action_threshold'],
            random_episode_start=run_params['random_episode_start'],
            episode_steps=run_params['episode_steps'],
            commission_rate=run_params['commission_rate'],
            features_dim=run_params['features_dim'],
            d_model=run_params['d_model'],
            n_heads=run_params['n_heads'],
            n_encoder_layers=run_params['n_encoder_layers'],
            dim_feedforward=run_params['dim_feedforward'],
            dropout=run_params['dropout'],
            activation=run_params['activation'],
            net_arch=net_arch,
            model_output=model_output,
            tensorboard_log_output=tensorboard_log_output,
            training_metrics=training_metrics
        )
        
        logger.info(f"Training component completed successfully. Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Error in training component: {str(e)}")
        raise

if __name__ == '__main__':
    main()
