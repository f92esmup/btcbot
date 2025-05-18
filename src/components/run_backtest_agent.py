"""
Backtest component for the BTC Trading Bot pipeline.

This script serves as the entry point for the Vertex AI Pipeline component
responsible for backtesting a trained RL agent against historical market data.
It evaluates the agent's performance, calculates metrics, and generates visualizations.
"""

import os
import argparse
import logging
import json
import yaml
from typing import Dict, Any

import kfp
from kfp.v2.dsl import Output, Metrics, Artifact

from src.backtesting.backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backtest(
    project_id: str,
    model_uri: str,
    test_data_uri: str,
    env_params_json: str,
    n_episodes: int,
    output_metrics_path: str,
    output_plots_path: str
) -> Dict[str, Any]:
    """
    Run backtest of a trained agent on historical data.
    
    Args:
        project_id (str): Google Cloud project ID.
        model_uri (str): GCS URI of the trained model.
        test_data_uri (str): GCS URI of the test data sequences.
        env_params_json (str): JSON string of environment parameters.
        n_episodes (int): Number of backtest episodes to run.
        output_metrics_path (str): GCS URI to save metrics to.
        output_plots_path (str): GCS URI to save plots to.
    
    Returns:
        Dict[str, Any]: Dictionary of backtest results.
    """
    # Parse environment parameters
    env_params = json.loads(env_params_json)
    
    # Create backtester
    backtester = Backtester(
        project_id=project_id,
        model_uri=model_uri,
        test_data_uri=test_data_uri,
        env_params=env_params
    )
    
    # Run backtest
    results = backtester.run_backtest(n_episodes=n_episodes)
    
    # Calculate additional metrics
    backtester.calculate_metrics()
    
    # Save results
    output_base_uri = f"gs://{os.path.dirname(output_metrics_path)}"
    backtester.save_results(output_base_uri)
    
    # Return summary metrics
    return {
        'final_balance_usd': results['final_balance_usd']['mean'],
        'sharpe_ratio': results['sharpe_ratio']['mean'],
        'max_drawdown_pct': results['max_drawdown_pct']['mean'],
        'win_rate': results['win_rate']['mean'],
        'total_trades': results['total_trades']['mean']
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run backtest of a trained RL agent')
    parser.add_argument('--project-id', type=str, required=True,
                        help='Google Cloud project ID')
    parser.add_argument('--model-uri', type=str, required=True,
                        help='GCS URI of the trained model')
    parser.add_argument('--test-data-uri', type=str, required=True,
                        help='GCS URI of the test data sequences')
    parser.add_argument('--env-params', type=str, required=True,
                        help='JSON string of environment parameters')
    parser.add_argument('--n-episodes', type=int, default=5,
                        help='Number of backtest episodes to run')
    parser.add_argument('--metrics-output-path', type=str, required=True,
                        help='Path to output metrics artifact')
    parser.add_argument('--plots-output-path', type=str, required=True,
                        help='Path to output plots artifact')
    
    return parser.parse_args()

def main():
    """Main entry point for the component."""
    args = parse_args()
    
    # Create output artifact instances
    metrics = Output(type=Metrics, path=args.metrics_output_path)
    plots = Output(type=Artifact, path=args.plots_output_path)
    
    # Run backtest
    results = run_backtest(
        project_id=args.project_id,
        model_uri=args.model_uri,
        test_data_uri=args.test_data_uri,
        env_params_json=args.env_params,
        n_episodes=args.n_episodes,
        output_metrics_path=args.metrics_output_path,
        output_plots_path=args.plots_output_path
    )
    
    # Log metrics to the metrics artifact
    for name, value in results.items():
        metrics.log_metric(name, value)
    
    logger.info("Backtest completed successfully")
    logger.info(f"Final Balance: ${results['final_balance_usd']:.2f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate'] * 100:.2f}%")
    logger.info(f"Total Trades: {results['total_trades']:.0f}")

if __name__ == '__main__':
    main()
