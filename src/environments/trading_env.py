"""
Trading environment module for BTC trading bot.

This module contains the TradingEnvironment class, which implements a custom Gymnasium
environment for the reinforcement learning agent. It simulates trading in the
Bitcoin futures market using the SimulatedBroker class.
"""

import os
import numpy as np
import pandas as pd
import logging
import gymnasium as gym
from gymnasium import spaces
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, Union, Any, List

from google.cloud import storage

from src.environments.simulated_broker import SimulatedBroker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TradingEnvironment(gym.Env):
    """
    A custom Gymnasium environment for Bitcoin futures trading.
    
    This environment simulates trading BTC/USDT perpetual futures contracts
    based on historical market data. It implements the standard Gymnasium
    interface and uses the SimulatedBroker class to simulate order execution.
    
    Attributes:
        logger (logging.Logger): Class logger.
        market_data (np.ndarray): Historical market data sequences of shape (N_samples, L, N_features).
        timestamps (np.ndarray): Timestamps corresponding to each sequence.
        feature_names (List[str]): Names of the features in the market data.
        sequence_length (int): Length of each market data sequence.
        n_features (int): Number of features in the market data.
        broker (SimulatedBroker): Instance of SimulatedBroker for simulating order execution.
        initial_balance_usd (float): Initial account balance in USD.
        max_position_btc (float): Maximum allowed position size in BTC.
        action_threshold (float): Threshold for converting continuous actions to discrete decisions.
        random_episode_start (bool): Whether to start episodes at random points in the data.
        episode_steps (int): Maximum number of steps per episode.
        current_step (int): Current step in the episode.
        current_balance_usd (float): Current account balance in USD.
        current_position_btc (float): Current position size in BTC.
        current_position_entry_price (float): Entry price of the current position.
        current_market_price (float): Current market price.
        current_atr (float): Current Average True Range value.
        portfolio_history (List[Dict]): History of portfolio state at each step.
        action_history (List[Dict]): History of actions taken at each step.
        episode_count (int): Count of episodes run.
        observation_space (spaces.Dict): The observation space.
        action_space (spaces.Box): The action space.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        project_id: str,
        gcs_processed_data_uri: str,
        initial_balance_usd: float = 10000.0,
        max_position_btc: float = 1.0,
        action_threshold: float = 0.5,
        random_episode_start: bool = True,
        episode_steps: int = 1000,
        commission_rate: float = 0.0004,
        max_leverage: int = 20,
        min_order_size_usd: float = 5.0,
        slippage_model: str = 'atr_based',
        slippage_factor: float = 0.05,
        normalize_portfolio_features: bool = True,
        # Normalization factors for portfolio features
        max_position_norm_divisor: float = 1.0,
        entry_price_log_scale: bool = True, 
        unrealized_pnl_norm_divisor: float = 1000.0,
        balance_log_scale: bool = True,
        steps_in_position_norm_divisor: float = 100.0,
        max_drawdown_pct_norm_divisor: float = 100.0,
        atr_pct_norm_divisor: float = 10.0,
        maintenance_margin_pct_norm_divisor: float = 100.0
    ):
        """
        Initialize the trading environment.
        
        Args:
            project_id (str): Google Cloud project ID.
            gcs_processed_data_uri (str): GCS URI of the processed market data sequences.
            initial_balance_usd (float, optional): Initial account balance in USD.
                Defaults to 10000.0.
            max_position_btc (float, optional): Maximum allowed position size in BTC.
                Defaults to 1.0.
            action_threshold (float, optional): Threshold for converting continuous actions
                to discrete decisions. Defaults to 0.5.
            random_episode_start (bool, optional): Whether to start episodes at random
                points in the data. Defaults to True.
            episode_steps (int, optional): Maximum number of steps per episode.
                Defaults to 1000.
            commission_rate (float, optional): Trading commission rate. Defaults to 0.0004 (0.04%).
            max_leverage (int, optional): Maximum allowed leverage. Defaults to 20.
            min_order_size_usd (float, optional): Minimum order size in USD. Defaults to 5.0.
            slippage_model (str, optional): Model for simulating slippage. Defaults to 'atr_based'.
            slippage_factor (float, optional): Factor for slippage calculation. Defaults to 0.05.
            normalize_portfolio_features (bool, optional): Whether to normalize portfolio features.
                Defaults to True.
            max_position_norm_divisor (float, optional): Divisor for normalizing position size.
                Defaults to 1.0.
            entry_price_log_scale (bool, optional): Whether to use log scale for entry price.
                Defaults to True.
            unrealized_pnl_norm_divisor (float, optional): Divisor for normalizing unrealized P&L.
                Defaults to 1000.0.
            balance_log_scale (bool, optional): Whether to use log scale for account balance.
                Defaults to True.
            steps_in_position_norm_divisor (float, optional): Divisor for normalizing steps in position.
                Defaults to 100.0.
            max_drawdown_pct_norm_divisor (float, optional): Divisor for normalizing max drawdown.
                Defaults to 100.0.
            atr_pct_norm_divisor (float, optional): Divisor for normalizing ATR percentage.
                Defaults to 10.0.
            maintenance_margin_pct_norm_divisor (float, optional): Divisor for normalizing
                maintenance margin percentage. Defaults to 100.0.
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.gcs_processed_data_uri = gcs_processed_data_uri
        self.initial_balance_usd = initial_balance_usd
        self.max_position_btc = max_position_btc
        self.action_threshold = action_threshold
        self.random_episode_start = random_episode_start
        self.episode_steps = episode_steps
        self.normalize_portfolio_features = normalize_portfolio_features
        
        # Normalization factors for portfolio features
        self.max_position_norm_divisor = max_position_norm_divisor
        self.entry_price_log_scale = entry_price_log_scale
        self.unrealized_pnl_norm_divisor = unrealized_pnl_norm_divisor
        self.balance_log_scale = balance_log_scale
        self.steps_in_position_norm_divisor = steps_in_position_norm_divisor
        self.max_drawdown_pct_norm_divisor = max_drawdown_pct_norm_divisor
        self.atr_pct_norm_divisor = atr_pct_norm_divisor
        self.maintenance_margin_pct_norm_divisor = maintenance_margin_pct_norm_divisor
        
        # Load market data
        self._load_market_data()
        
        # Initialize the broker
        self.broker = SimulatedBroker(
            commission_rate=commission_rate,
            max_leverage=max_leverage,
            min_order_size_usd=min_order_size_usd,
            slippage_model=slippage_model,
            slippage_factor=slippage_factor
        )
        
        # Define the observation space
        self._define_observation_space()
        
        # Define the action space: continuous value from -1 to 1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize counters and history
        self.current_step = 0
        self.episode_count = 0
        self.portfolio_history = []
        self.action_history = []
        
        # Initialize other state variables
        self.current_balance_usd = self.initial_balance_usd
        self.current_position_btc = 0.0
        self.current_position_entry_price = 0.0
        self.current_market_price = 0.0
        self.current_atr = 0.0
        self.steps_in_position = 0
        self.max_position_value = 0.0
        self.max_drawdown_pct = 0.0
        self.position_direction = 0  # 0: no position, 1: long, -1: short
        self.last_equity = self.initial_balance_usd
        self.equity_peak = self.initial_balance_usd
        
        self.logger.info("TradingEnvironment initialized successfully.")
        self.logger.info(f"Market data shape: {self.market_data.shape}")
    
    def _load_market_data(self):
        """
        Load market data sequences from Google Cloud Storage.
        
        Raises:
            Exception: If there's an error loading the data.
        """
        try:
            self.logger.info(f"Loading market data from GCS: {self.gcs_processed_data_uri}")
            
            # Parse bucket and blob path from GCS URI
            if self.gcs_processed_data_uri.startswith('gs://'):
                gcs_path = self.gcs_processed_data_uri[5:]  # Remove 'gs://' prefix
            else:
                gcs_path = self.gcs_processed_data_uri
            
            bucket_name, blob_path = gcs_path.split('/', 1)
            
            # Initialize GCS client
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download to a temporary file
            temp_file = '/tmp/market_data.npz'
            blob.download_to_filename(temp_file)
            
            # Load the npz file
            data = np.load(temp_file)
            self.market_data = data['X_market']
            self.timestamps = data['timestamps']
            self.feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
            
            # Get dimensions
            self.n_samples, self.sequence_length, self.n_features = self.market_data.shape
            
            # Clean up
            os.remove(temp_file)
            
            self.logger.info(f"Successfully loaded market data with shape: {self.market_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            raise
    
    def _define_observation_space(self):
        """
        Define the observation space for the environment.
        
        The observation space is a Dict with two components:
        1. 'market_features': Box of shape (sequence_length, n_features)
        2. 'portfolio_features': Box of shape (8,) containing:
           - normalized_position
           - normalized_entry_price
           - unrealized_pnl_normalized
           - normalized_balance
           - steps_in_position_normalized
           - max_drawdown_pct_normalized
           - atr_pct_normalized
           - maintenance_margin_pct_normalized
        """
        # Market features (using actual dimensions from loaded data)
        market_features_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sequence_length, self.n_features),
            dtype=np.float32
        )
        
        # Portfolio features
        portfolio_features_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        
        # Define the full observation space
        self.observation_space = spaces.Dict({
            'market_features': market_features_space,
            'portfolio_features': portfolio_features_space
        })
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options. Defaults to None.
            
        Returns:
            tuple: A tuple containing:
                - observation (Dict): Initial observation.
                - info (Dict): Additional information.
        """
        super().reset(seed=seed)
        
        # Select a starting point in the data
        if self.random_episode_start:
            # Choose a random starting point, leaving room for a full episode
            max_start = max(0, self.n_samples - self.episode_steps - 1)
            self.current_idx = self.np_random.integers(max_start) if max_start > 0 else 0
        else:
            # Start from the beginning
            self.current_idx = 0
        
        # Reset state variables
        self.current_step = 0
        self.current_balance_usd = self.initial_balance_usd
        self.current_position_btc = 0.0
        self.current_position_entry_price = 0.0
        self.steps_in_position = 0
        self.max_position_value = 0.0
        self.max_drawdown_pct = 0.0
        self.position_direction = 0  # 0: no position, 1: long, -1: short
        self.last_equity = self.initial_balance_usd
        self.equity_peak = self.initial_balance_usd
        
        # Reset history
        self.portfolio_history = []
        self.action_history = []
        
        # Get the current market data and update current_market_price
        self._update_market_info()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Increment episode counter
        self.episode_count += 1
        
        self.logger.info(f"Starting episode {self.episode_count} at index {self.current_idx}")
        
        # Return initial observation and info
        info = {
            'timestamp': str(self.timestamps[self.current_idx]),
            'balance_usd': self.current_balance_usd,
            'position_btc': self.current_position_btc,
            'market_price': self.current_market_price,
            'episode': self.episode_count,
            'step': self.current_step
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        Args:
            action (np.ndarray): Action to take, a value in the range [-1, 1].
            
        Returns:
            tuple: A tuple containing:
                - observation (Dict): New observation.
                - reward (float): Reward for the action.
                - terminated (bool): Whether the episode has ended due to termination conditions.
                - truncated (bool): Whether the episode has ended due to truncation (e.g., max steps).
                - info (Dict): Additional information.
        """
        # Extract the action value
        action_value = float(action[0])
        
        # Convert continuous action to trading decision
        decision, target_position_btc = self._interpret_action(action_value)
        
        # Store the original portfolio state before action
        old_balance = self.current_balance_usd
        old_position = self.current_position_btc
        old_equity = self._calculate_equity()
        
        # Execute trading decision
        self._execute_trade(decision, target_position_btc)
        
        # Move to the next time step
        self.current_step += 1
        self.current_idx += 1
        
        # Check if the episode is done
        terminated = False
        truncated = False
        
        # Update market info for the new time step
        self._update_market_info()
        
        # Check for liquidation
        if self._check_liquidation():
            self.logger.info(f"Position liquidated at step {self.current_step}")
            terminated = True
        
        # Check if we've reached the maximum number of steps or the end of data
        if self.current_step >= self.episode_steps or self.current_idx >= self.n_samples - 1:
            truncated = True
        
        # Calculate reward (using log returns of equity)
        new_equity = self._calculate_equity()
        reward = self._calculate_reward(old_equity, new_equity)
        
        # Update equity peak and max drawdown
        if new_equity > self.equity_peak:
            self.equity_peak = new_equity
        
        current_drawdown_pct = (self.equity_peak - new_equity) / self.equity_peak * 100
        self.max_drawdown_pct = max(self.max_drawdown_pct, current_drawdown_pct)
        
        # Update last equity
        self.last_equity = new_equity
        
        # Get new observation
        observation = self._get_observation()
        
        # Record portfolio state
        self._record_portfolio_state()
        
        # Record action
        self._record_action(action_value, decision, target_position_btc)
        
        # Prepare info dictionary
        info = {
            'timestamp': str(self.timestamps[self.current_idx - 1]),  # Timestamp of the completed step
            'balance_usd': self.current_balance_usd,
            'position_btc': self.current_position_btc,
            'entry_price': self.current_position_entry_price,
            'market_price': self.current_market_price,
            'equity': new_equity,
            'action': action_value,
            'decision': decision,
            'target_position': target_position_btc,
            'reward': reward,
            'steps_in_position': self.steps_in_position,
            'max_drawdown_pct': self.max_drawdown_pct,
            'unrealized_pnl': self._calculate_unrealized_pnl()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _interpret_action(self, action_value):
        """
        Interpret the continuous action value as a trading decision.
        
        Args:
            action_value (float): Action value in the range [-1, 1].
            
        Returns:
            tuple: A tuple containing:
                - decision (str): Trading decision ('buy', 'sell', 'hold', 'close').
                - target_position_btc (float): Target position size in BTC.
        """
        # Determine decision based on action value and current position
        if self.current_position_btc == 0:
            # No position currently
            if action_value > self.action_threshold:
                # Open long position
                decision = 'buy'
                position_size_pct = (action_value - self.action_threshold) / (1 - self.action_threshold)
                target_position_btc = self.max_position_btc * position_size_pct
            elif action_value < -self.action_threshold:
                # Open short position
                decision = 'sell'
                position_size_pct = (-action_value - self.action_threshold) / (1 - self.action_threshold)
                target_position_btc = -self.max_position_btc * position_size_pct
            else:
                # Continue holding no position
                decision = 'hold'
                target_position_btc = 0.0
        
        elif self.current_position_btc > 0:
            # Currently long
            if action_value < -self.action_threshold:
                # Close long position or go short
                decision = 'sell'
                if action_value < -(self.action_threshold + 0.2):  # Requires stronger signal to flip position
                    # Go short
                    position_size_pct = (-action_value - (self.action_threshold + 0.2)) / (1 - (self.action_threshold + 0.2))
                    target_position_btc = -self.max_position_btc * position_size_pct
                else:
                    # Just close the position
                    decision = 'close'
                    target_position_btc = 0.0
            elif action_value > self.action_threshold:
                # Adjust long position size
                decision = 'buy'
                position_size_pct = (action_value - self.action_threshold) / (1 - self.action_threshold)
                target_position_btc = self.max_position_btc * position_size_pct
            else:
                # Hold current position
                decision = 'hold'
                target_position_btc = self.current_position_btc
        
        elif self.current_position_btc < 0:
            # Currently short
            if action_value > self.action_threshold:
                # Close short position or go long
                decision = 'buy'
                if action_value > (self.action_threshold + 0.2):  # Requires stronger signal to flip position
                    # Go long
                    position_size_pct = (action_value - (self.action_threshold + 0.2)) / (1 - (self.action_threshold + 0.2))
                    target_position_btc = self.max_position_btc * position_size_pct
                else:
                    # Just close the position
                    decision = 'close'
                    target_position_btc = 0.0
            elif action_value < -self.action_threshold:
                # Adjust short position size
                decision = 'sell'
                position_size_pct = (-action_value - self.action_threshold) / (1 - self.action_threshold)
                target_position_btc = -self.max_position_btc * position_size_pct
            else:
                # Hold current position
                decision = 'hold'
                target_position_btc = self.current_position_btc
        
        return decision, target_position_btc
    
    def _execute_trade(self, decision, target_position_btc):
        """
        Execute a trading decision.
        
        Args:
            decision (str): Trading decision ('buy', 'sell', 'hold', 'close').
            target_position_btc (float): Target position size in BTC.
        """
        if decision == 'hold':
            # No trade to execute
            if self.current_position_btc != 0:
                self.steps_in_position += 1
            return
        
        # Calculate the required change in position
        position_delta_btc = target_position_btc - self.current_position_btc
        
        if abs(position_delta_btc) < 1e-8:
            # Position change is too small to execute
            if self.current_position_btc != 0:
                self.steps_in_position += 1
            return
        
        # Determine order type
        if position_delta_btc > 0:
            order_type = 'buy'
        elif position_delta_btc < 0:
            order_type = 'sell'
        else:
            # No change in position
            if self.current_position_btc != 0:
                self.steps_in_position += 1
            return
        
        # Calculate order size in USD
        order_size_usd = abs(position_delta_btc) * self.current_market_price
        
        # Execute the order using the broker
        execution_details = self.broker.calculate_execution_details(
            order_type=order_type,
            market_price=self.current_market_price,
            order_size_usd=order_size_usd,
            atr=self.current_atr,
            current_position=self.current_position_btc,
            current_position_entry_price=self.current_position_entry_price
        )
        
        if not execution_details['executed']:
            self.logger.debug(f"Order not executed: {execution_details['reason']}")
            if self.current_position_btc != 0:
                self.steps_in_position += 1
            return
        
        # Update account balance with commission and slippage costs
        self.current_balance_usd -= execution_details['commission_usd']
        self.current_balance_usd -= execution_details['slippage_usd']
        
        # Calculate the new position details
        old_position_value = self.current_position_btc * self.current_position_entry_price
        new_position_delta = execution_details['size_btc']
        new_position_value = abs(new_position_delta) * execution_details['execution_price']
        
        # Update position size
        self.current_position_btc += new_position_delta
        
        # Update position direction
        if self.current_position_btc > 0:
            self.position_direction = 1
        elif self.current_position_btc < 0:
            self.position_direction = -1
        else:
            self.position_direction = 0
        
        # Update entry price (weighted average)
        if abs(self.current_position_btc) > 1e-8:
            # If adding to position
            if (self.current_position_btc - new_position_delta) * new_position_delta > 0:
                self.current_position_entry_price = (old_position_value + new_position_value) / abs(self.current_position_btc)
            # If reducing position
            elif abs(new_position_delta) < abs(self.current_position_btc - new_position_delta):
                # Entry price remains the same
                pass
            # If opening a new position or flipping direction
            else:
                self.current_position_entry_price = execution_details['execution_price']
        else:
            # Position closed
            self.current_position_entry_price = 0.0
            self.steps_in_position = 0
            self.position_direction = 0
        
        # If position exists, increment steps in position counter
        if self.current_position_btc != 0:
            if (self.current_position_btc - new_position_delta) * new_position_delta <= 0 and self.current_position_btc * (self.current_position_btc - new_position_delta) <= 0:
                # New position or position flipped, reset counter
                self.steps_in_position = 0
            else:
                # Existing position modified, increment counter
                self.steps_in_position += 1
        
        # Update max position value
        current_position_value = abs(self.current_position_btc) * self.current_market_price
        self.max_position_value = max(self.max_position_value, current_position_value)
        
        self.logger.debug(f"Executed {order_type} order: {abs(new_position_delta):.8f} BTC at {execution_details['execution_price']:.2f} USD")
    
    def _calculate_unrealized_pnl(self):
        """
        Calculate the unrealized profit and loss of the current position.
        
        Returns:
            float: Unrealized P&L in USD.
        """
        if self.current_position_btc == 0:
            return 0.0
        
        # For long positions: (current_price - entry_price) * position_size
        # For short positions: (entry_price - current_price) * |position_size|
        if self.current_position_btc > 0:
            return (self.current_market_price - self.current_position_entry_price) * self.current_position_btc
        else:
            return (self.current_position_entry_price - self.current_market_price) * abs(self.current_position_btc)
    
    def _calculate_equity(self):
        """
        Calculate the total equity (balance + unrealized P&L).
        
        Returns:
            float: Total equity in USD.
        """
        return self.current_balance_usd + self._calculate_unrealized_pnl()
    
    def _update_market_info(self):
        """
        Update market information for the current step.
        """
        if self.current_idx < self.n_samples:
            # Get the latest market data sequence
            current_sequence = self.market_data[self.current_idx]
            
            # Get the close price (assuming it's the 4th column, index 3)
            close_index = 3 if self.feature_names is None else self.feature_names.index('close')
            self.current_market_price = float(current_sequence[-1, close_index])
            
            # Get the ATR (assuming it's available in the features)
            atr_index = 8 if self.feature_names is None else self.feature_names.index('atr')
            self.current_atr = float(current_sequence[-1, atr_index])
    
    def _check_liquidation(self):
        """
        Check if the current position should be liquidated due to insufficient margin.
        
        Returns:
            bool: True if the position is liquidated, False otherwise.
        """
        if self.current_position_btc == 0:
            return False
        
        # Calculate liquidation price
        liquidation_price = self.broker.calculate_liquidation_price(
            position_size_btc=self.current_position_btc,
            entry_price=self.current_position_entry_price,
            account_balance_usd=self.current_balance_usd
        )
        
        if liquidation_price is None:
            return False
        
        # Check if liquidation price has been breached
        if self.current_position_btc > 0:  # Long position
            liquidated = self.current_market_price <= liquidation_price
        else:  # Short position
            liquidated = self.current_market_price >= liquidation_price
        
        if liquidated:
            # Simulate liquidation (close position at current price with extra fees)
            liquidation_fee = abs(self.current_position_btc) * self.current_market_price * 0.01  # 1% liquidation fee
            self.current_balance_usd -= liquidation_fee
            
            # Realize loss
            unrealized_pnl = self._calculate_unrealized_pnl()
            self.current_balance_usd += unrealized_pnl
            
            # Reset position
            self.current_position_btc = 0.0
            self.current_position_entry_price = 0.0
            self.steps_in_position = 0
            self.position_direction = 0
            
            self.logger.info(f"Position liquidated. Loss: {unrealized_pnl:.2f} USD, Fee: {liquidation_fee:.2f} USD")
        
        return liquidated
    
    def _get_portfolio_features(self):
        """
        Calculate and normalize portfolio features.
        
        Returns:
            np.ndarray: Array of normalized portfolio features.
        """
        # 1. Position size normalized
        normalized_position = self.current_position_btc / self.max_position_norm_divisor
        
        # 2. Entry price normalized
        if self.current_position_entry_price > 0 and self.entry_price_log_scale:
            normalized_entry_price = np.log(self.current_position_entry_price / self.current_market_price)
        elif self.current_position_entry_price > 0:
            normalized_entry_price = self.current_position_entry_price / self.current_market_price - 1.0
        else:
            normalized_entry_price = 0.0
        
        # 3. Unrealized P&L normalized
        unrealized_pnl = self._calculate_unrealized_pnl()
        unrealized_pnl_normalized = unrealized_pnl / self.unrealized_pnl_norm_divisor
        
        # 4. Account balance normalized
        if self.balance_log_scale:
            normalized_balance = np.log(self.current_balance_usd / self.initial_balance_usd)
        else:
            normalized_balance = self.current_balance_usd / self.initial_balance_usd - 1.0
        
        # 5. Steps in position normalized
        steps_in_position_normalized = self.steps_in_position / self.steps_in_position_norm_divisor
        
        # 6. Max drawdown percentage normalized
        max_drawdown_pct_normalized = self.max_drawdown_pct / self.max_drawdown_pct_norm_divisor
        
        # 7. ATR percentage normalized
        atr_pct = (self.current_atr / self.current_market_price) * 100
        atr_pct_normalized = atr_pct / self.atr_pct_norm_divisor
        
        # 8. Maintenance margin percentage normalized
        if abs(self.current_position_btc) > 0:
            position_value = abs(self.current_position_btc) * self.current_market_price
            margin_reqs = self.broker.calculate_required_margin(
                position_size_btc=self.current_position_btc,
                entry_price=self.current_position_entry_price
            )
            maintenance_margin_pct = (margin_reqs['maintenance_margin_usd'] / position_value) * 100
            maintenance_margin_pct_normalized = maintenance_margin_pct / self.maintenance_margin_pct_norm_divisor
        else:
            maintenance_margin_pct_normalized = 0.0
        
        # Combine all features
        portfolio_features = np.array([
            normalized_position,
            normalized_entry_price,
            unrealized_pnl_normalized,
            normalized_balance,
            steps_in_position_normalized,
            max_drawdown_pct_normalized,
            atr_pct_normalized,
            maintenance_margin_pct_normalized
        ], dtype=np.float32)
        
        return portfolio_features
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Dict: Observation dictionary with market and portfolio features.
        """
        # Market features
        market_features = self.market_data[self.current_idx].astype(np.float32)
        
        # Portfolio features
        portfolio_features = self._get_portfolio_features()
        
        return {
            'market_features': market_features,
            'portfolio_features': portfolio_features
        }
    
    def _calculate_reward(self, old_equity, new_equity):
        """
        Calculate the reward for the current step.
        
        Using log returns of equity as rewards.
        
        Args:
            old_equity (float): Equity before action.
            new_equity (float): Equity after action.
            
        Returns:
            float: Reward value.
        """
        # Prevent division by zero or negative equity
        if old_equity <= 0 or new_equity <= 0:
            return -1.0  # Penalize negative equity
        
        # Calculate log return
        log_return = np.log(new_equity / old_equity)
        
        return log_return
    
    def _record_portfolio_state(self):
        """
        Record the current portfolio state in the history.
        """
        state = {
            'timestamp': str(self.timestamps[self.current_idx - 1]),
            'step': self.current_step,
            'balance_usd': self.current_balance_usd,
            'position_btc': self.current_position_btc,
            'position_entry_price': self.current_position_entry_price,
            'market_price': self.current_market_price,
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'equity': self._calculate_equity(),
            'steps_in_position': self.steps_in_position,
            'max_drawdown_pct': self.max_drawdown_pct,
            'atr': self.current_atr
        }
        
        self.portfolio_history.append(state)
    
    def _record_action(self, action_value, decision, target_position_btc):
        """
        Record the action taken in the history.
        
        Args:
            action_value (float): Raw action value.
            decision (str): Trading decision.
            target_position_btc (float): Target position size.
        """
        action_record = {
            'timestamp': str(self.timestamps[self.current_idx - 1]),
            'step': self.current_step,
            'action_value': float(action_value),
            'decision': decision,
            'target_position_btc': float(target_position_btc),
            'actual_position_btc': float(self.current_position_btc),
            'position_changed': abs(self.current_position_btc - (self.portfolio_history[-2]['position_btc'] if len(self.portfolio_history) >= 2 else 0)) > 1e-8
        }
        
        self.action_history.append(action_record)
    
    def render(self, mode="human"):
        """
        Render the environment.
        
        Not implemented for this environment as it's intended for use in a headless environment.
        """
        pass
    
    def get_portfolio_history_df(self):
        """
        Get the portfolio history as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with portfolio history.
        """
        return pd.DataFrame(self.portfolio_history)
    
    def get_action_history_df(self):
        """
        Get the action history as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with action history.
        """
        return pd.DataFrame(self.action_history)
    
    def save_history_to_gcs(self, gcs_bucket, gcs_prefix=None):
        """
        Save the portfolio and action history to Google Cloud Storage.
        
        Args:
            gcs_bucket (str): GCS bucket name.
            gcs_prefix (str, optional): Prefix for GCS path. Defaults to None.
            
        Returns:
            tuple: A tuple containing:
                - portfolio_gcs_uri (str): GCS URI of the portfolio history file.
                - actions_gcs_uri (str): GCS URI of the action history file.
        """
        try:
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(gcs_bucket)
            
            # Generate GCS paths
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            base_path = f"{gcs_prefix}/" if gcs_prefix else ""
            
            portfolio_path = f"{base_path}portfolio_history_{timestamp}.json"
            actions_path = f"{base_path}action_history_{timestamp}.json"
            
            # Convert history to JSON
            portfolio_json = json.dumps(self.portfolio_history, indent=2)
            actions_json = json.dumps(self.action_history, indent=2)
            
            # Upload to GCS
            portfolio_blob = bucket.blob(portfolio_path)
            portfolio_blob.upload_from_string(portfolio_json, content_type='application/json')
            
            actions_blob = bucket.blob(actions_path)
            actions_blob.upload_from_string(actions_json, content_type='application/json')
            
            # Return URIs
            portfolio_uri = f"gs://{gcs_bucket}/{portfolio_path}"
            actions_uri = f"gs://{gcs_bucket}/{actions_path}"
            
            self.logger.info(f"Saved portfolio history to {portfolio_uri}")
            self.logger.info(f"Saved action history to {actions_uri}")
            
            return portfolio_uri, actions_uri
            
        except Exception as e:
            self.logger.error(f"Error saving history to GCS: {str(e)}")
            raise
