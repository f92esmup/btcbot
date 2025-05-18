"""
Unit tests for the trading environment module.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

from src.environments.trading_env import TradingEnvironment
from src.environments.simulated_broker import SimulatedBroker

@pytest.fixture
def mock_storage_client():
    """Fixture providing a mock for Google Cloud Storage client."""
    with patch('google.cloud.storage.Client') as mock:
        # Mock bucket and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Configure the mock storage client
        mock_client = mock.return_value
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Configure blob.download_as_bytes to return sample data
        mock_blob.download_as_bytes.return_value = np.array([
            # Sample market data sequences with shape (10, 100, 25)
            # 10 samples, 100 timesteps each, 25 features
            np.random.random((10, 100, 25)).astype(np.float32),
            # Timestamps for each sequence
            np.array([pd.Timestamp('2022-01-01 00:00:00')] * 10),
            # Feature names
            np.array(['open', 'high', 'low', 'close', 'volume'] + ['feature_' + str(i) for i in range(20)])
        ], dtype=object).tobytes()
        
        yield mock_client

@pytest.fixture
def mock_broker():
    """Fixture providing a mock for SimulatedBroker."""
    mock_broker = MagicMock(spec=SimulatedBroker)
    
    # Configure mock broker behavior
    mock_broker.calculate_liquidation_price.return_value = 45000.0
    mock_broker.execute_order.return_value = {
        'executed_price': 50000.0,
        'commission_paid': 20.0,
        'slippage_paid': 10.0
    }
    
    yield mock_broker

@pytest.fixture
def sample_env_params():
    """Fixture providing sample parameters for the trading environment."""
    return {
        'project_id': 'test-project',
        'gcs_processed_data_uri': 'gs://bucket/path/to/data',
        'initial_balance_usd': 10000.0,
        'max_position_btc': 1.0,
        'action_threshold': 0.5,
        'random_episode_start': True,
        'episode_steps': 100,
        'commission_rate': 0.0004,
        'max_leverage': 20,
        'min_order_size_usd': 5.0,
        'slippage_model': 'atr_based',
        'slippage_factor': 0.05,
        'normalize_portfolio_features': True
    }

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_initialization(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test initialization of TradingEnvironment class."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Create environment
    env = TradingEnvironment(**sample_env_params)
    
    # Check initialization
    assert env is not None
    assert env.initial_balance_usd == 10000.0
    assert env.max_position_btc == 1.0
    assert env.action_threshold == 0.5
    assert env.random_episode_start == True
    assert env.episode_steps == 100
    
    # Check broker initialization
    mock_broker_class.assert_called_once_with(
        commission_rate=0.0004,
        max_leverage=20,
        min_order_size_usd=5.0,
        slippage_model='atr_based',
        slippage_factor=0.05
    )
    
    # Check observation space
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')
    
    # Check action space is in the correct range (-1, 1)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_reset(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test resetting the trading environment."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Create environment
    env = TradingEnvironment(**sample_env_params)
    
    # Test reset with seed
    obs, info = env.reset(seed=42)
    
    # Check that the observation is a dictionary with the expected keys
    assert isinstance(obs, dict)
    assert 'market_features' in obs
    assert 'portfolio_features' in obs
    
    # Check that market_features has the right shape
    assert isinstance(obs['market_features'], np.ndarray)
    
    # Check that portfolio_features has the right shape
    assert isinstance(obs['portfolio_features'], np.ndarray)
    
    # Check that the environment state is reset
    assert env.current_step == 0
    assert env.current_balance_usd == env.initial_balance_usd
    assert env.current_position_btc == 0.0
    assert env.current_position_entry_price == 0.0
    assert len(env.portfolio_history) > 0
    assert len(env.action_history) == 0

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_step(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test stepping the trading environment."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Configure broker's execute_order to return a realistic result
    mock_broker_instance.execute_order.return_value = {
        'executed_price': 50000.0,
        'commission_paid': 20.0,
        'slippage_paid': 10.0
    }
    
    # Configure broker's calculate_liquidation_price
    mock_broker_instance.calculate_liquidation_price.return_value = 45000.0
    
    # Create environment
    env = TradingEnvironment(**sample_env_params)
    
    # Reset the environment
    env.reset(seed=42)
    
    # Test step with a buy action
    action = np.array([0.8])  # Action > action_threshold (0.5) should open a long position
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that the observation has the expected structure
    assert isinstance(obs, dict)
    assert 'market_features' in obs
    assert 'portfolio_features' in obs
    
    # Check that the step incremented
    assert env.current_step == 1
    
    # Check that a position was opened
    assert env.current_position_btc > 0.0
    assert env.current_position_entry_price > 0.0
    
    # Check that the broker was called to execute the order
    mock_broker_instance.execute_order.assert_called_once()
    
    # Check that history was updated
    assert len(env.portfolio_history) > 1
    assert len(env.action_history) == 1
    
    # Check that the reward is a float
    assert isinstance(reward, float)
    
    # Check that the episode is not terminated yet
    assert not terminated
    assert not truncated
    
    # Test another step with a sell action
    action = np.array([-0.8])  # Action < -action_threshold (-0.5) should close the position
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that the position was closed
    assert env.current_position_btc == 0.0
    
    # Check that the broker was called again
    assert mock_broker_instance.execute_order.call_count == 2
    
    # Check that history was updated
    assert len(env.portfolio_history) > 2
    assert len(env.action_history) == 2

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_episode_end(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test that the environment correctly terminates episodes."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Create environment with very short episodes
    short_params = sample_env_params.copy()
    short_params['episode_steps'] = 3
    
    env = TradingEnvironment(**short_params)
    
    # Reset the environment
    env.reset(seed=42)
    
    # Run steps until episode ends
    terminated = truncated = False
    step_count = 0
    
    while not (terminated or truncated) and step_count < 10:  # Safeguard
        action = np.array([0.0])  # Neutral action
        _, _, terminated, truncated, _ = env.step(action)
        step_count += 1
    
    # Check that the episode ended after the specified number of steps
    assert step_count == 3
    assert not terminated  # Not terminated by liquidation or other conditions
    assert truncated  # Truncated by max episode steps

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_bankrupt(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test that the environment terminates on bankruptcy."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Create environment
    env = TradingEnvironment(**sample_env_params)
    
    # Reset the environment
    env.reset(seed=42)
    
    # Manually set the balance close to zero
    env.current_balance_usd = 1.0
    
    # Configure broker's execute_order to simulate a loss
    mock_broker_instance.execute_order.return_value = {
        'executed_price': 50000.0,
        'commission_paid': 2.0,  # This should bankrupt the account
        'slippage_paid': 0.0
    }
    
    # Test step with any action
    action = np.array([0.8])
    _, _, terminated, truncated, info = env.step(action)
    
    # Check that the episode ended due to bankruptcy
    assert terminated
    assert not truncated
    assert info.get('termination_reason') == 'bankrupt'

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_liquidation(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test that the environment terminates on position liquidation."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Create environment
    env = TradingEnvironment(**sample_env_params)
    
    # Reset the environment
    env.reset(seed=42)
    
    # Open a position
    action = np.array([0.8])
    obs, _, _, _, _ = env.step(action)
    
    # Manually set the market price below the liquidation price
    env.current_position_entry_price = 50000.0
    liquidation_price = 49000.0
    mock_broker_instance.calculate_liquidation_price.return_value = liquidation_price
    
    # Modify the current market price to trigger liquidation
    env.current_market_price = 48000.0  # Below liquidation price
    
    # Take another step
    action = np.array([0.0])  # Neutral action
    _, _, terminated, truncated, info = env.step(action)
    
    # Check that the episode ended due to liquidation
    assert terminated
    assert not truncated
    assert info.get('termination_reason') == 'liquidated'

@patch('google.cloud.storage.Client')
@patch('src.environments.simulated_broker.SimulatedBroker')
def test_trading_env_render(mock_broker_class, mock_client_class, sample_env_params, mock_storage_client):
    """Test the render method of the trading environment."""
    # Configure mocks
    mock_broker_instance = MagicMock()
    mock_broker_class.return_value = mock_broker_instance
    mock_client_class.return_value = mock_storage_client
    
    # Create environment
    env = TradingEnvironment(**sample_env_params)
    
    # Reset the environment
    env.reset(seed=42)
    
    # Test render method
    result = env.render()
    
    # Render should return a string representation of the current state
    assert isinstance(result, str)
    assert "Step:" in result
    assert "Balance:" in result
    assert "Position:" in result
    assert "Market Price:" in result
