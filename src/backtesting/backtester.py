"""
Backtesting module for Bitcoin trading bot.

This module contains the Backtester class, which is responsible for running
backtests of trained RL agents against historical market data. It calculates
various performance metrics and generates visualizations of the trading strategy's
performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile

from google.cloud import storage
import joblib
import torch
from stable_baselines3 import SAC

from src.environments.trading_env import TradingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Backtester:
    """
    A class for backtesting trained RL agents on historical market data.
    
    This class loads a trained RL agent and evaluates its performance on
    historical market data. It calculates various performance metrics and
    generates visualizations of the trading strategy's performance.
    
    Attributes:
        logger (logging.Logger): Class logger.
        project_id (str): Google Cloud project ID.
        model_uri (str): GCS URI of the trained model.
        test_data_uri (str): GCS URI of the test data.
        env_params (Dict[str, Any]): Parameters for the trading environment.
        storage_client (storage.Client): Google Cloud Storage client.
        model (SAC): Loaded RL agent model.
        env (TradingEnvironment): Trading environment for backtesting.
        results (Dict[str, Any]): Backtest results and metrics.
    """
    
    def __init__(
        self,
        project_id: str,
        model_uri: str,
        test_data_uri: str,
        env_params: Dict[str, Any]
    ):
        """
        Initialize the Backtester.
        
        Args:
            project_id (str): Google Cloud project ID.
            model_uri (str): GCS URI of the trained model.
            test_data_uri (str): GCS URI of the test data.
            env_params (Dict[str, Any]): Parameters for the trading environment.
        """
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.model_uri = model_uri
        self.test_data_uri = test_data_uri
        self.env_params = env_params
        self.storage_client = storage.Client(project=project_id)
        
        # Will be set later
        self.model = None
        self.env = None
        self.results = {}
    
    def download_from_gcs(self, gcs_uri: str, local_path: str) -> None:
        """
        Download a file from Google Cloud Storage.
        
        Args:
            gcs_uri (str): GCS URI of the file to download.
            local_path (str): Local path to save the file.
        """
        try:
            bucket_name = gcs_uri.split('/')[2]
            blob_path = '/'.join(gcs_uri.split('/')[3:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            blob.download_to_filename(local_path)
            self.logger.info(f"Downloaded {gcs_uri} to {local_path}")
        except Exception as e:
            self.logger.error(f"Error downloading from GCS: {e}")
            raise
    
    def upload_to_gcs(self, local_path: str, gcs_uri: str) -> None:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            local_path (str): Local path of the file to upload.
            gcs_uri (str): GCS URI to upload the file to.
        """
        try:
            bucket_name = gcs_uri.split('/')[2]
            blob_path = '/'.join(gcs_uri.split('/')[3:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            blob.upload_from_filename(local_path)
            self.logger.info(f"Uploaded {local_path} to {gcs_uri}")
        except Exception as e:
            self.logger.error(f"Error uploading to GCS: {e}")
            raise
    
    def load_model(self) -> None:
        """
        Load the trained model from GCS.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_local_path = os.path.join(temp_dir, "model.zip")
            self.download_from_gcs(self.model_uri, model_local_path)
            
            # Load the model
            self.model = SAC.load(model_local_path)
            self.logger.info("Model loaded successfully")
    
    def setup_environment(self) -> None:
        """
        Set up the trading environment for backtesting.
        """
        # Update the env_params with the test data URI
        self.env_params['gcs_processed_data_uri'] = self.test_data_uri
        
        # Create the environment
        self.env = TradingEnvironment(**self.env_params)
        self.logger.info("Trading environment set up for backtesting")
    
    def run_backtest(self, n_episodes: int = 1) -> Dict[str, Any]:
        """
        Run the backtest for the specified number of episodes.
        
        Args:
            n_episodes (int, optional): Number of episodes to run.
                Defaults to 1.
        
        Returns:
            Dict[str, Any]: Backtest results and metrics.
        """
        if self.model is None:
            self.load_model()
        
        if self.env is None:
            self.setup_environment()
        
        # Lists to store results from all episodes
        all_portfolio_history = []
        all_action_history = []
        all_final_balances = []
        all_sharpe_ratios = []
        all_max_drawdowns = []
        all_total_trades = []
        all_win_rates = []
        
        for episode in range(n_episodes):
            self.logger.info(f"Running backtest episode {episode+1}/{n_episodes}")
            
            obs, _ = self.env.reset()
            done = False
            info = {}
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            
            # Append results from this episode
            all_portfolio_history.append(self.env.portfolio_history)
            all_action_history.append(self.env.action_history)
            all_final_balances.append(info['final_balance_usd'])
            all_sharpe_ratios.append(info['sharpe_ratio'])
            all_max_drawdowns.append(info['max_drawdown_pct'])
            all_total_trades.append(info['total_trades'])
            
            # Calculate win rate
            win_rate = 0
            if info['total_trades'] > 0:
                win_rate = info['winning_trades'] / info['total_trades']
            all_win_rates.append(win_rate)
        
        # Aggregate results from all episodes
        self.results = {
            'final_balance_usd': {
                'mean': np.mean(all_final_balances),
                'std': np.std(all_final_balances),
                'min': np.min(all_final_balances),
                'max': np.max(all_final_balances),
                'values': all_final_balances
            },
            'sharpe_ratio': {
                'mean': np.mean(all_sharpe_ratios),
                'std': np.std(all_sharpe_ratios),
                'min': np.min(all_sharpe_ratios),
                'max': np.max(all_sharpe_ratios),
                'values': all_sharpe_ratios
            },
            'max_drawdown_pct': {
                'mean': np.mean(all_max_drawdowns),
                'std': np.std(all_max_drawdowns),
                'min': np.min(all_max_drawdowns),
                'max': np.max(all_max_drawdowns),
                'values': all_max_drawdowns
            },
            'total_trades': {
                'mean': np.mean(all_total_trades),
                'std': np.std(all_total_trades),
                'min': np.min(all_total_trades),
                'max': np.max(all_total_trades),
                'values': all_total_trades
            },
            'win_rate': {
                'mean': np.mean(all_win_rates),
                'std': np.std(all_win_rates),
                'min': np.min(all_win_rates),
                'max': np.max(all_win_rates),
                'values': all_win_rates
            },
            'portfolio_history': all_portfolio_history,
            'action_history': all_action_history
        }
        
        self.logger.info(f"Backtest completed with average final balance: ${self.results['final_balance_usd']['mean']:.2f}")
        return self.results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate additional performance metrics from the backtest results.
        
        Returns:
            Dict[str, Any]: Dictionary of calculated metrics.
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Get the portfolio history from the best episode (highest final balance)
        best_episode_idx = np.argmax(self.results['final_balance_usd']['values'])
        portfolio_history = self.results['portfolio_history'][best_episode_idx]
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(portfolio_history)
        
        # Calculate daily returns (assuming the data has timestamps)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['daily_return'] = df['balance_usd'].pct_change()
        
        # Calculate metrics
        metrics = {
            'annualized_return': df['daily_return'].mean() * 252 * 100,  # Annualized percentage
            'annualized_volatility': df['daily_return'].std() * np.sqrt(252) * 100,  # Annualized percentage
            'sortino_ratio': self._calculate_sortino_ratio(df['daily_return']),
            'calmar_ratio': self._calculate_calmar_ratio(df['daily_return']),
            'profit_factor': self._calculate_profit_factor(df['daily_return']),
            'avg_trade_duration': self._calculate_avg_trade_duration(self.results['action_history'][best_episode_idx]),
            'largest_winning_trade': self._find_largest_winning_trade(self.results['action_history'][best_episode_idx]),
            'largest_losing_trade': self._find_largest_losing_trade(self.results['action_history'][best_episode_idx])
        }
        
        # Add these metrics to the results
        self.results.update(metrics)
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the Sortino ratio, which measures return per unit of downside risk.
        
        Args:
            returns (pd.Series): Series of daily returns.
            risk_free_rate (float, optional): Annual risk-free rate.
                Defaults to 0.0.
        
        Returns:
            float: Sortino ratio.
        """
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate downside returns (negative returns only)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')  # No downside or no downside volatility
        
        # Calculate Sortino ratio
        excess_return = returns.mean() - daily_risk_free
        downside_deviation = downside_returns.std()
        
        sortino = excess_return / downside_deviation * np.sqrt(252)  # Annualized
        return sortino
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate the Calmar ratio, which is the annual return divided by the maximum drawdown.
        
        Args:
            returns (pd.Series): Series of daily returns.
        
        Returns:
            float: Calmar ratio.
        """
        # Calculate annualized return
        annual_return = returns.mean() * 252
        
        # Maximum drawdown already calculated in percentage
        max_drawdown = self.results['max_drawdown_pct']['mean'] / 100  # Convert from percentage
        
        if max_drawdown == 0:
            return float('inf')  # No drawdown
        
        calmar = annual_return / max_drawdown
        return calmar
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate the profit factor, which is the gross profit divided by the gross loss.
        
        Args:
            returns (pd.Series): Series of daily returns.
        
        Returns:
            float: Profit factor.
        """
        # Separate positive and negative returns
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf')  # No losses
        
        profit_factor = profits / losses
        return profit_factor
    
    def _calculate_avg_trade_duration(self, action_history: List[Dict]) -> float:
        """
        Calculate the average duration of trades in minutes.
        
        Args:
            action_history (List[Dict]): History of actions taken during the backtest.
        
        Returns:
            float: Average trade duration in minutes.
        """
        durations = []
        entry_time = None
        position = 0
        
        for action in action_history:
            timestamp = datetime.fromisoformat(action['timestamp'])
            new_position = action['position_btc']
            
            # Position changed from 0 to non-zero (entry)
            if position == 0 and new_position != 0:
                entry_time = timestamp
            
            # Position changed from non-zero to 0 (exit)
            elif position != 0 and new_position == 0 and entry_time is not None:
                duration = (timestamp - entry_time).total_seconds() / 60  # in minutes
                durations.append(duration)
                entry_time = None
            
            position = new_position
        
        if not durations:
            return 0
        
        return np.mean(durations)
    
    def _find_largest_winning_trade(self, action_history: List[Dict]) -> float:
        """
        Find the largest winning trade in USD.
        
        Args:
            action_history (List[Dict]): History of actions taken during the backtest.
        
        Returns:
            float: Largest winning trade in USD.
        """
        trades = self._extract_trades(action_history)
        winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
        
        if not winning_trades:
            return 0
        
        return max(winning_trades)
    
    def _find_largest_losing_trade(self, action_history: List[Dict]) -> float:
        """
        Find the largest losing trade in USD.
        
        Args:
            action_history (List[Dict]): History of actions taken during the backtest.
        
        Returns:
            float: Largest losing trade in USD (as a negative number).
        """
        trades = self._extract_trades(action_history)
        losing_trades = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        if not losing_trades:
            return 0
        
        return min(losing_trades)
    
    def _extract_trades(self, action_history: List[Dict]) -> List[Dict]:
        """
        Extract trades from action history.
        
        Args:
            action_history (List[Dict]): History of actions taken during the backtest.
        
        Returns:
            List[Dict]: List of trades with PnL.
        """
        trades = []
        current_trade = None
        
        for action in action_history:
            timestamp = datetime.fromisoformat(action['timestamp'])
            position = action['position_btc']
            market_price = action['market_price']
            
            # New position opened
            if current_trade is None and position != 0:
                current_trade = {
                    'entry_time': timestamp,
                    'entry_price': action['entry_price'],
                    'position_size': position,
                    'side': 'long' if position > 0 else 'short'
                }
            
            # Position closed
            elif current_trade is not None and position == 0:
                # Calculate PnL
                exit_price = market_price
                entry_price = current_trade['entry_price']
                size = abs(current_trade['position_size'])
                side = current_trade['side']
                
                if side == 'long':
                    pnl = (exit_price - entry_price) * size
                else:  # short
                    pnl = (entry_price - exit_price) * size
                
                # Account for commissions (simplified)
                commission = (entry_price + exit_price) * size * 0.0004  # 0.04% commission
                pnl -= commission
                
                trades.append({
                    'entry_time': current_trade['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': size,
                    'side': side,
                    'pnl': pnl,
                    'duration_minutes': (timestamp - current_trade['entry_time']).total_seconds() / 60
                })
                
                current_trade = None
        
        return trades
    
    def generate_plots(self, output_dir: str) -> List[str]:
        """
        Generate performance plots from the backtest results.
        
        Args:
            output_dir (str): Directory to save the plots.
        
        Returns:
            List[str]: List of paths to the generated plot files.
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = []
        
        # Get the portfolio history from the best episode (highest final balance)
        best_episode_idx = np.argmax(self.results['final_balance_usd']['values'])
        portfolio_history = self.results['portfolio_history'][best_episode_idx]
        action_history = self.results['action_history'][best_episode_idx]
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # 1. Equity Curve Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['balance_usd'], label='Account Balance (USD)')
        
        # Add position entry/exit markers
        longs_entry = []
        longs_exit = []
        shorts_entry = []
        shorts_exit = []
        
        position = 0
        for action in action_history:
            timestamp = datetime.fromisoformat(action['timestamp'])
            new_position = action['position_btc']
            price = action['market_price']
            
            # Long entry
            if position == 0 and new_position > 0:
                longs_entry.append((timestamp, price))
            # Short entry
            elif position == 0 and new_position < 0:
                shorts_entry.append((timestamp, price))
            # Long exit
            elif position > 0 and new_position == 0:
                longs_exit.append((timestamp, price))
            # Short exit
            elif position < 0 and new_position == 0:
                shorts_exit.append((timestamp, price))
            
            position = new_position
        
        # Add trade markers to equity curve
        for entry in longs_entry:
            plt.axvline(x=entry[0], color='green', linestyle='--', alpha=0.3)
        for exit in longs_exit:
            plt.axvline(x=exit[0], color='red', linestyle='--', alpha=0.3)
        for entry in shorts_entry:
            plt.axvline(x=entry[0], color='red', linestyle='--', alpha=0.3)
        for exit in shorts_exit:
            plt.axvline(x=exit[0], color='green', linestyle='--', alpha=0.3)
        
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('USD')
        ax.legend()
        ax.grid(True)
        
        equity_curve_path = os.path.join(output_dir, 'equity_curve.png')
        plt.savefig(equity_curve_path)
        plt.close()
        plot_files.append(equity_curve_path)
        
        # 2. Drawdown Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df['equity_peak'] = df['balance_usd'].cummax()
        df['drawdown'] = (df['balance_usd'] - df['equity_peak']) / df['equity_peak'] * 100
        
        ax.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        drawdown_path = os.path.join(output_dir, 'drawdown.png')
        plt.savefig(drawdown_path)
        plt.close()
        plot_files.append(drawdown_path)
        
        # 3. Returns Distribution Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df['daily_return'] = df['balance_usd'].pct_change()
        sns.histplot(df['daily_return'].dropna() * 100, kde=True, ax=ax)
        ax.set_title('Daily Returns Distribution')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        
        # Add vertical line at mean return
        mean_return = df['daily_return'].mean() * 100
        ax.axvline(mean_return, color='r', linestyle='--')
        ax.text(mean_return*1.1, ax.get_ylim()[1]*0.9, f'Mean: {mean_return:.2f}%', color='r')
        
        returns_dist_path = os.path.join(output_dir, 'returns_distribution.png')
        plt.savefig(returns_dist_path)
        plt.close()
        plot_files.append(returns_dist_path)
        
        # 4. Position Size Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['position_btc'], label='Position Size (BTC)')
        ax.set_title('Position Size Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('BTC')
        ax.legend()
        ax.grid(True)
        
        position_path = os.path.join(output_dir, 'position_size.png')
        plt.savefig(position_path)
        plt.close()
        plot_files.append(position_path)
        
        # 5. Trade PnL Distribution
        trades = self._extract_trades(action_history)
        
        if trades:
            trade_pnls = [t['pnl'] for t in trades]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(trade_pnls, kde=True, ax=ax)
            ax.set_title('Trade PnL Distribution')
            ax.set_xlabel('PnL (USD)')
            ax.set_ylabel('Frequency')
            
            # Add vertical line at mean PnL
            mean_pnl = np.mean(trade_pnls)
            ax.axvline(mean_pnl, color='r', linestyle='--')
            ax.text(mean_pnl*1.1, ax.get_ylim()[1]*0.9, f'Mean: ${mean_pnl:.2f}', color='r')
            
            trade_pnl_path = os.path.join(output_dir, 'trade_pnl_distribution.png')
            plt.savefig(trade_pnl_path)
            plt.close()
            plot_files.append(trade_pnl_path)
        
        # 6. Performance Metrics Summary
        fig, ax = plt.subplots(figsize=(10, 8))
        metrics = [
            f"Final Balance: ${self.results['final_balance_usd']['mean']:.2f}",
            f"Return: {((self.results['final_balance_usd']['mean'] / self.env_params['initial_balance_usd']) - 1) * 100:.2f}%",
            f"Sharpe Ratio: {self.results['sharpe_ratio']['mean']:.2f}",
            f"Max Drawdown: {self.results['max_drawdown_pct']['mean']:.2f}%",
            f"Win Rate: {self.results['win_rate']['mean'] * 100:.2f}%",
            f"Total Trades: {self.results['total_trades']['mean']:.0f}",
            f"Profit Factor: {self.results.get('profit_factor', 0):.2f}",
            f"Sortino Ratio: {self.results.get('sortino_ratio', 0):.2f}",
            f"Calmar Ratio: {self.results.get('calmar_ratio', 0):.2f}",
            f"Avg Trade Duration: {self.results.get('avg_trade_duration', 0):.1f} min",
            f"Largest Win: ${self.results.get('largest_winning_trade', 0):.2f}",
            f"Largest Loss: ${self.results.get('largest_losing_trade', 0):.2f}"
        ]
        
        # No data to plot, just text
        ax.axis('off')
        y_pos = 0.95
        for metric in metrics:
            ax.text(0.5, y_pos, metric, ha='center', va='center', fontsize=12)
            y_pos -= 0.07
        
        ax.set_title('Performance Metrics Summary', fontsize=16)
        
        metrics_path = os.path.join(output_dir, 'metrics_summary.png')
        plt.savefig(metrics_path)
        plt.close()
        plot_files.append(metrics_path)
        
        return plot_files
    
    def save_results(self, output_gcs_uri: str) -> None:
        """
        Save the backtest results to GCS.
        
        Args:
            output_gcs_uri (str): GCS URI to save the results to.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save metrics as JSON
            metrics_path = os.path.join(temp_dir, 'metrics.json')
            
            # Create a copy of results without the large history arrays
            results_copy = {k: v for k, v in self.results.items() 
                          if k not in ['portfolio_history', 'action_history']}
            
            with open(metrics_path, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            # Generate plots
            plots_dir = os.path.join(temp_dir, 'plots')
            plot_files = self.generate_plots(plots_dir)
            
            # Upload results to GCS
            metrics_gcs_uri = f"{output_gcs_uri}/metrics.json"
            self.upload_to_gcs(metrics_path, metrics_gcs_uri)
            
            for plot_file in plot_files:
                plot_name = os.path.basename(plot_file)
                plot_gcs_uri = f"{output_gcs_uri}/plots/{plot_name}"
                self.upload_to_gcs(plot_file, plot_gcs_uri)
            
            self.logger.info(f"Backtest results saved to {output_gcs_uri}")
