#!/usr/bin/env python3
# filepath: /workspaces/btcbot/scripts/train_with_best_model.py
"""
Train RL Agent with Smart Checkpointing

This script implements the "Best Model Only" strategy where:
1. Training starts from existing "best model" if available
2. Only a single "best model" file is maintained in GCS
3. The best model is automatically updated during training when a better model is found

Usage:
  python train_with_best_model.py [--config CONFIG_PATH] [--steps STEPS]
"""

import os
import sys
import argparse
import logging
import numpy as np
import gymnasium as gym
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.rl_agent_manager import RLAgentManager
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.callbacks.bigquery_callbacks import BigQueryLoggingCallback

# Set up logging
logger = setup_logger("TrainBestModelOnly")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent using Best Model Only strategy")
    parser.add_argument("--config", type=str, help="Path to config file", default="src/config.yaml")
    parser.add_argument("--steps", type=int, help="Number of training steps (default: from config)", default=None)
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_arguments()
    
    logger.info("Starting RL training with Best Model Only strategy")
    
    # Load configuration
    config_manager = ConfigManager(config_path=args.config)
    agent_config = config_manager.get_agent_config()
    
    # Check if Best Model Only strategy is enabled in config
    enable_best_model_only = agent_config.get("enable_best_model_only", False)
    if not enable_best_model_only:
        logger.warning(
            "Best Model Only strategy is not enabled in config. "
            "Set enable_best_model_only: true in config.yaml"
        )
    
    # Get GCS bucket name
    gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        logger.error("GCS_BUCKET_NAME environment variable is not set!")
        sys.exit(1)
    
    # Initialize RLAgentManager
    try:
        # Set up the agent manager
        agent_manager = RLAgentManager(config_path=args.config)
        
        # Set up BigQuery callback for metrics logging
        project_id = os.environ.get("GCP_PROJECT_ID")
        dataset_id = os.environ.get("BIGQUERY_LOG_DATASET_ID", "btcbot_logs")
        
        bigquery_callback = BigQueryLoggingCallback(
            project_id=project_id,
            dataset_id=dataset_id,
            config_manager=config_manager
        )
        
        # Get best model path
        best_model_path = agent_config.get("best_model_path", "models/best_sac_model.zip")
        
        # Try to set up agent with existing best model
        try:
            logger.info(f"Attempting to load existing best model from {best_model_path}")
            agent_manager.setup_agent(load_model=True, model_path=best_model_path)
            logger.info("Loaded existing best model - will continue training from this checkpoint")
        except Exception as e:
            logger.warning(f"Could not load existing best model: {e}")
            logger.info("Setting up new agent from scratch")
            agent_manager.setup_agent()
        
        # Get training timesteps
        total_timesteps = args.steps if args.steps is not None else agent_config.get("total_training_timesteps", 1000000)
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Train the agent with BigQuery callback
        agent_manager.train_agent(
            total_timesteps=total_timesteps,
            user_callbacks=[bigquery_callback]
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
