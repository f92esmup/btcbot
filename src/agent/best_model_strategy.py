"""
Best Model Only Strategy 

Implements a simplified model management approach where only the best performing model
is saved and updated during training, rather than keeping multiple checkpoints.
"""

import os
import logging
import tempfile
from typing import Optional
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from src.utils.gcs_utils import upload_model_to_gcs, download_model_from_gcs

logger = logging.getLogger('BestModelStrategy')

class BestModelOnlyCallback(BaseCallback):
    """
    Callback for evaluating and maintaining a single "best model" during training.
    
    Instead of saving multiple checkpoints, this approach only keeps track of the 
    best model found during training, based on evaluation metrics.
    """
    
    def __init__(
            self, 
            eval_env: gym.Env,
            bucket_name: str, 
            best_model_path: str,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            deterministic: bool = True,
            verbose: int = 0
    ):
        """
        Initialize the callback.
        
        Args:
            eval_env: Environment used for evaluation
            bucket_name: GCS bucket name
            best_model_path: Path in GCS for the best model (without bucket prefix)
            eval_freq: Evaluate the agent every n steps
            n_eval_episodes: Number of episodes to use for evaluation
            deterministic: Whether to use deterministic actions during evaluation
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.bucket_name = bucket_name
        self.best_model_path = best_model_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        
        # Metrics tracking
        self.best_mean_reward = -float('inf')
        self.current_best_model_local_path = None
        
        # Create temp directory for model storage
        self.temp_dir = tempfile.mkdtemp(prefix="best_model_")
        
        # Try to download existing best model if available
        self.existing_model_found = self._try_load_existing_best_model()
        
    def _try_load_existing_best_model(self) -> bool:
        """
        Attempt to download the existing best model from GCS.
        
        Returns:
            True if a model was found and downloaded, False otherwise
        """
        try:
            full_gcs_path = f"{self.bucket_name}/{self.best_model_path}"
            logger.info(f"Checking for existing best model at gs://{full_gcs_path}")
            
            # Try to download the model
            local_path = download_model_from_gcs(full_gcs_path)
            
            if local_path:
                logger.info(f"Found existing best model, downloaded to {local_path}")
                
                # Evaluate the existing model to get baseline performance
                model = SAC.load(local_path, env=self.eval_env)
                
                # Run evaluation episodes
                total_rewards = []
                for _ in range(self.n_eval_episodes):
                    obs, _ = self.eval_env.reset()
                    done = False
                    truncated = False
                    episode_reward = 0
                    
                    while not done and not truncated:
                        action, _ = model.predict(obs, deterministic=self.deterministic)
                        obs, reward, done, truncated, _ = self.eval_env.step(action)
                        episode_reward += reward
                    
                    total_rewards.append(episode_reward)
                
                # Set as current best
                self.best_mean_reward = sum(total_rewards) / len(total_rewards)
                self.current_best_model_local_path = local_path
                
                logger.info(f"Existing model baseline performance: {self.best_mean_reward}")
                return True
            
        except Exception as e:
            logger.info(f"No existing best model found or error loading it: {e}")
        
        return False
    
    def _on_step(self) -> bool:
        """
        Run evaluation and update best model if needed.
        
        Returns:
            True to continue training, False to stop
        """
        if self.n_calls % self.eval_freq != 0:
            return True
        
        # Run evaluation episodes
        total_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        # Calculate mean reward
        mean_reward = sum(total_rewards) / len(total_rewards)
        
        logger.info(f"Evaluation at step {self.num_timesteps}: Mean reward = {mean_reward:.2f} vs. best = {self.best_mean_reward:.2f}")
        
        # Save best model if better than previous best
        if mean_reward > self.best_mean_reward:
            logger.info(f"New best model found at step {self.num_timesteps}! Mean reward increased from {self.best_mean_reward:.2f} to {mean_reward:.2f}")
            self.best_mean_reward = mean_reward
            
            # Save model locally first
            local_path = os.path.join(self.temp_dir, "best_model.zip")
            self.model.save(local_path)
            
            # Clean up previous best model if it exists
            if self.current_best_model_local_path and os.path.exists(self.current_best_model_local_path):
                try:
                    os.remove(self.current_best_model_local_path)
                except Exception as e:
                    logger.warning(f"Error removing previous best model: {e}")
            
            self.current_best_model_local_path = local_path
            
            # Upload to GCS (overwriting previous best)
            try:
                full_gcs_path = f"{self.bucket_name}/{self.best_model_path}"
                gcs_url = upload_model_to_gcs(local_path, full_gcs_path)
                logger.info(f"Best model saved to GCS: {gcs_url}")
            except Exception as e:
                logger.error(f"Error uploading best model to GCS: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Clean up resources when training ends."""
        # Clean up the temporary model file if still exists
        if self.current_best_model_local_path and os.path.exists(self.current_best_model_local_path):
            try:
                os.remove(self.current_best_model_local_path)
            except Exception as e:
                logger.warning(f"Error removing best model during cleanup: {e}")


def load_best_model(env: gym.Env, bucket_name: str, best_model_path: str, device: str) -> Optional[SAC]:
    """
    Load the best model from Google Cloud Storage.
    
    Args:
        env: The environment for the model
        bucket_name: GCS bucket name
        best_model_path: Path to the best model in GCS (without bucket prefix)
        device: Device to load the model on ('cpu', 'cuda', 'mps')
        
    Returns:
        Loaded SAC model, or None if no model was found
    """
    try:
        full_gcs_path = f"{bucket_name}/{best_model_path}"
        logger.info(f"Attempting to load best model from gs://{full_gcs_path}")
        
        # Download the model
        local_path = download_model_from_gcs(full_gcs_path)
        
        # Load the model
        model = SAC.load(local_path, env=env, device=device)
        logger.info(f"Best model loaded successfully on {device}")
        
        # Clean up the temporary file
        os.remove(local_path)
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
        return None
