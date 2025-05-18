"""
Pipeline definition for the BTC Trading Bot using Vertex AI Pipelines.

This script defines a Kubeflow Pipeline (KFP) that orchestrates the entire workflow
for the Bitcoin trading bot. It connects data acquisition, preprocessing, training,
and backtesting components into an end-to-end machine learning pipeline.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics,
    ClassificationMetrics,
    pipeline,
    ParallelFor
)

# Define the pipeline
@pipeline(
    name="btc-trading-bot-pipeline",
    description="End-to-end ML pipeline for Bitcoin trading bot"
)
def btc_trading_pipeline(
    project_id: str,
    region: str,
    gcs_bucket: str,
    binance_api_key: str = "",
    binance_api_secret: str = "",
    download_start_date: str = "2020-01-01",
    download_end_date: str = "2022-12-31",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    train_test_split_date: str = "2022-07-01",
    sequence_length: int = 100,
    initial_balance_usd: float = 10000.0,
    max_position_btc: float = 1.0,
    commission_rate: float = 0.0004,
    max_leverage: int = 20,
    training_steps: int = 1000000,
    n_backtest_episodes: int = 5,
    transformer_params_json: str = '{"n_heads": 4, "n_layers": 2, "d_model": 64}',
    image_uri: str = None
):
    """
    Define the Bitcoin trading bot pipeline.
    
    Args:
        project_id (str): Google Cloud project ID.
        region (str): GCP region for pipeline execution.
        gcs_bucket (str): GCS bucket for storing data and artifacts.
        binance_api_key (str, optional): Binance API key for data download.
        binance_api_secret (str, optional): Binance API secret for data download.
        download_start_date (str, optional): Start date for data download (YYYY-MM-DD).
        download_end_date (str, optional): End date for data download (YYYY-MM-DD).
        symbol (str, optional): Trading symbol. Defaults to "BTCUSDT".
        timeframe (str, optional): Trading timeframe. Defaults to "1h".
        train_test_split_date (str, optional): Date to split train/test data (YYYY-MM-DD).
        sequence_length (int, optional): Length of input sequences. Defaults to 100.
        initial_balance_usd (float, optional): Initial account balance in USD.
        max_position_btc (float, optional): Maximum position size in BTC.
        commission_rate (float, optional): Trading commission rate.
        max_leverage (int, optional): Maximum leverage for trading.
        training_steps (int, optional): Number of training steps.
        n_backtest_episodes (int, optional): Number of backtest episodes.
        transformer_params_json (str, optional): JSON string of transformer parameters.
        image_uri (str, optional): URI of the Docker image to use for components.
    """
    # Define component URIs
    gcs_path_base = f"gs://{gcs_bucket}/btc-trading-bot"
    
    # Container image URI for all components
    # If a custom image_uri is provided, use it; otherwise, use the default
    container_image_uri = image_uri if image_uri else f"{region}-docker.pkg.dev/{project_id}/btc-trading-bot/btc-trading-bot:latest"
    
    # Define the data acquisition component
    @component(
        base_image=container_image_uri,
        packages_to_install=["google-cloud-storage>=2.0.0"]
    )
    def data_acquisition_op(
        project_id: str,
        api_key: str,
        api_secret: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        output_data_uri: str,
        output_dataset: Output[Dataset]
    ):
        import subprocess
        
        cmd = [
            "python", "-m", "src.components.run_data_acquisition",
            "--project-id", project_id,
            "--api-key", api_key,
            "--api-secret", api_secret,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--start-date", start_date,
            "--end-date", end_date,
            "--output-uri", output_data_uri,
            "--dataset-output-path", output_dataset.path
        ]
        
        subprocess.run(cmd, check=True)
    
    # Define the preprocessing component
    @component(
        base_image=container_image_uri,
        packages_to_install=["google-cloud-storage>=2.0.0", "pandas>=1.3.0", "pandas-ta>=0.3.0"]
    )
    def preprocessing_op(
        project_id: str,
        input_data_uri: str,
        train_test_split_date: str,
        sequence_length: int,
        output_train_data_uri: str,
        output_test_data_uri: str,
        output_train_dataset: Output[Dataset],
        output_test_dataset: Output[Dataset]
    ):
        import subprocess
        
        cmd = [
            "python", "-m", "src.components.run_preprocessing",
            "--project-id", project_id,
            "--input-uri", input_data_uri,
            "--train-test-split-date", train_test_split_date,
            "--sequence-length", str(sequence_length),
            "--output-train-uri", output_train_data_uri,
            "--output-test-uri", output_test_data_uri,
            "--train-dataset-output-path", output_train_dataset.path,
            "--test-dataset-output-path", output_test_dataset.path
        ]
        
        subprocess.run(cmd, check=True)
    
    # Define the training component
    @component(
        base_image=container_image_uri,
        packages_to_install=[
            "google-cloud-storage>=2.0.0", 
            "torch>=1.10.0", 
            "stable-baselines3>=1.6.0", 
            "gymnasium>=0.26.0"
        ]
    )
    def train_agent_op(
        project_id: str,
        train_data_uri: str,
        env_params_json: str,
        transformer_params_json: str,
        training_steps: int,
        output_model_uri: str,
        model_output: Output[Model],
        metrics_output: Output[Metrics]
    ):
        import subprocess
        
        cmd = [
            "python", "-m", "src.components.run_train_agent",
            "--project-id", project_id,
            "--train-data-uri", train_data_uri,
            "--env-params", env_params_json,
            "--transformer-params", transformer_params_json,
            "--training-steps", str(training_steps),
            "--output-model-uri", output_model_uri,
            "--model-output-path", model_output.path,
            "--metrics-output-path", metrics_output.path
        ]
        
        subprocess.run(cmd, check=True)
    
    # Define the backtest component
    @component(
        base_image=container_image_uri,
        packages_to_install=[
            "google-cloud-storage>=2.0.0", 
            "torch>=1.10.0", 
            "stable-baselines3>=1.6.0", 
            "gymnasium>=0.26.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0"
        ]
    )
    def backtest_agent_op(
        project_id: str,
        model_uri: str,
        test_data_uri: str,
        env_params_json: str,
        n_episodes: int,
        metrics_output: Output[Metrics],
        plots_output: Output[Artifact]
    ):
        import subprocess
        
        cmd = [
            "python", "-m", "src.components.run_backtest_agent",
            "--project-id", project_id,
            "--model-uri", model_uri,
            "--test-data-uri", test_data_uri,
            "--env-params", env_params_json,
            "--n-episodes", str(n_episodes),
            "--metrics-output-path", metrics_output.path,
            "--plots-output-path", plots_output.path
        ]
        
        subprocess.run(cmd, check=True)
    
    # Configure pipeline execution parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output URIs
    raw_data_uri = f"{gcs_path_base}/data/raw/{timestamp}"
    train_data_uri = f"{gcs_path_base}/data/processed/train/{timestamp}"
    test_data_uri = f"{gcs_path_base}/data/processed/test/{timestamp}"
    model_uri = f"{gcs_path_base}/models/{timestamp}"
    metrics_uri = f"{gcs_path_base}/metrics/{timestamp}"
    plots_uri = f"{gcs_path_base}/plots/{timestamp}"
    
    # Environment parameters for trading environment
    env_params = {
        "project_id": project_id,
        "gcs_processed_data_uri": train_data_uri,
        "initial_balance_usd": initial_balance_usd,
        "max_position_btc": max_position_btc,
        "commission_rate": commission_rate,
        "max_leverage": max_leverage,
        "random_episode_start": True,
        "episode_steps": 1000,
        "slippage_model": "atr_based",
        "slippage_factor": 0.05
    }
    
    env_params_json = json.dumps(env_params)
    
    # Run the components
    data_task = data_acquisition_op(
        project_id=project_id,
        api_key=binance_api_key,
        api_secret=binance_api_secret,
        symbol=symbol,
        timeframe=timeframe,
        start_date=download_start_date,
        end_date=download_end_date,
        output_data_uri=raw_data_uri
    )
    
    preprocess_task = preprocessing_op(
        project_id=project_id,
        input_data_uri=raw_data_uri,
        train_test_split_date=train_test_split_date,
        sequence_length=sequence_length,
        output_train_data_uri=train_data_uri,
        output_test_data_uri=test_data_uri
    ).after(data_task)
    
    train_task = train_agent_op(
        project_id=project_id,
        train_data_uri=train_data_uri,
        env_params_json=env_params_json,
        transformer_params_json=transformer_params_json,
        training_steps=training_steps,
        output_model_uri=model_uri
    ).after(preprocess_task)
    
    backtest_task = backtest_agent_op(
        project_id=project_id,
        model_uri=model_uri,
        test_data_uri=test_data_uri,
        env_params_json=env_params_json,
        n_episodes=n_backtest_episodes
    ).after(train_task)

def compile_pipeline(output_file: str = "btc_trading_pipeline.json", image_uri: str = None):
    """
    Compile the pipeline to a JSON file.
    
    Args:
        output_file (str, optional): Path to the output JSON file.
            Defaults to "btc_trading_pipeline.json".
        image_uri (str, optional): URI of the Docker image to use for components.
            If None, the default image URI will be used.
    """
    compiler.Compiler().compile(
        pipeline_func=lambda **kwargs: btc_trading_pipeline(image_uri=image_uri, **kwargs) if image_uri else btc_trading_pipeline(**kwargs),
        package_path=output_file
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compile the BTC trading bot pipeline')
    parser.add_argument('--output-file', type=str, default="btc_trading_pipeline.json",
                        help='Path to the output JSON file')
    parser.add_argument('--image-uri', type=str, default=None,
                        help='URI of the Docker image to use for components')
    
    return parser.parse_args()

def main():
    """Main entry point for pipeline compilation."""
    args = parse_args()
    compile_pipeline(output_file=args.output_file, image_uri=args.image_uri)
    print(f"Pipeline compiled successfully to {args.output_file}")
    if args.image_uri:
        print(f"Using custom image URI: {args.image_uri}")
    else:
        print("Using default image URI from pipeline definition")

if __name__ == '__main__':
    main()