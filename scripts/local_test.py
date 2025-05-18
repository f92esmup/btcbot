#!/usr/bin/env python3
"""
This script runs a local test of the Bitcoin trading bot components without using KFP pipelines.
It's designed for local development and testing of the individual components.
"""

import os
import sys
import json
import argparse
import tempfile
from datetime import datetime
from pathlib import Path

# Ensure the src package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Run a local test of the Bitcoin trading bot components."""
    parser = argparse.ArgumentParser(description='Run a local test of the BTC trading bot')
    parser.add_argument('--project-id', type=str, default='local-project',
                        help='Project ID for GCP resources (not used in local mode)')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                        help='Start date for data download (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2022-01-31',
                        help='End date for data download (YYYY-MM-DD)')
    parser.add_argument('--training-steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='Temporary directory for test data (if not specified, will create one)')
    args = parser.parse_args()
    
    # Create a temporary directory for the test if not specified
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
    
    # Create subdirectories
    raw_data_dir = temp_dir / 'data' / 'raw'
    train_data_dir = temp_dir / 'data' / 'processed' / 'train'
    test_data_dir = temp_dir / 'data' / 'processed' / 'test'
    model_dir = temp_dir / 'models'
    metrics_dir = temp_dir / 'metrics'
    plots_dir = temp_dir / 'plots'
    
    for directory in [raw_data_dir, train_data_dir, test_data_dir, model_dir, metrics_dir, plots_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"🏠 Test data directory: {temp_dir}")
    
    # Run data acquisition component
    print("\n📊 Running data acquisition component...")
    try:
        # Import the module once
        import src.components.run_data_acquisition as data_acq
        
        # Call the main function with a mock Output object
        class MockOutput:
            def __init__(self, path):
                self.path = path
                
        data_acq.main([
            "--project-id", args.project_id,
            "--api-key", "test-api-key",
            "--api-secret", "test-api-secret",
            "--symbol", "BTCUSDT",
            "--timeframe", "1h",
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--output-uri", str(raw_data_dir),
            "--dataset-output-path", str(raw_data_dir / "metadata.json")
        ])
    except Exception as e:
        print(f"Error in data acquisition: {e}")
        
    # Run preprocessing component
    print("\n🔄 Running preprocessing component...")
    try:
        import src.components.run_preprocessing as preproc
        
        preproc.main([
            "--project-id", args.project_id,
            "--input-uri", str(raw_data_dir),
            "--train-test-split-date", "2022-01-15",
            "--sequence-length", "100",
            "--output-train-uri", str(train_data_dir),
            "--output-test-uri", str(test_data_dir),
            "--train-dataset-output-path", str(train_data_dir / "metadata.json"),
            "--test-dataset-output-path", str(test_data_dir / "metadata.json")
        ])
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        
    # Define environment parameters
    env_params = {
        "project_id": args.project_id,
        "gcs_processed_data_uri": str(train_data_dir),
        "initial_balance_usd": 10000.0,
        "max_position_btc": 1.0,
        "commission_rate": 0.0004,
        "max_leverage": 20,
        "random_episode_start": True,
        "episode_steps": 1000,
        "slippage_model": "atr_based",
        "slippage_factor": 0.05
    }
    
    # Define transformer parameters
    transformer_params = {
        "n_heads": 4,
        "n_layers": 2,
        "d_model": 64
    }
    
    # Run training component
    print("\n🧠 Running training component...")
    try:
        import src.components.run_train_agent as trainer
        
        trainer.main([
            "--project-id", args.project_id,
            "--train-data-uri", str(train_data_dir),
            "--env-params", json.dumps(env_params),
            "--transformer-params", json.dumps(transformer_params),
            "--training-steps", str(args.training_steps),
            "--output-model-uri", str(model_dir),
            "--model-output-path", str(model_dir / "metadata.json"),
            "--metrics-output-path", str(metrics_dir / "training_metrics.json")
        ])
    except Exception as e:
        print(f"Error in training: {e}")
    
    # Run backtest component
    print("\n🧪 Running backtest component...")
    try:
        import src.components.run_backtest_agent as backtester
        
        backtester.main([
            "--project-id", args.project_id,
            "--model-uri", str(model_dir),
            "--test-data-uri", str(test_data_dir),
            "--env-params", json.dumps(env_params),
            "--n-episodes", "1",
            "--metrics-output-path", str(metrics_dir / "backtest_metrics.json"),
            "--plots-output-path", str(plots_dir)
        ])
    except Exception as e:
        print(f"Error in backtesting: {e}")
    
    print("\n✅ Local test completed!")
    print(f"Results available in: {temp_dir}")
    
    # Cleanup temp directory if we created it
    if not args.temp_dir:
        temp_dir_obj.cleanup()

if __name__ == "__main__":
    main()
