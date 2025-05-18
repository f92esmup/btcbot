"""
Unit tests for the backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

from src.backtesting.backtester import Backtester
from src.environments.trading_env import TradingEnvironment

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
        
        yield mock_client

@pytest.fixture
def mock_env():
    """Fixture providing a mock TradingEnvironment."""
    mock_env = MagicMock(spec=TradingEnvironment)
    
    # Configure mock environment behavior
    mock_env.reset.return_value = ({}, {})  # obs, info
    mock_env.step.return_value = ({}, 0.0, False, False, {})  # obs, reward, terminated, truncated, info
    
    # Mock portfolio history
    mock_env.portfolio_history = [
        {
            'timestamp': '2022-01-01T00:00:00',
            'balance_usd': 10000.0,
            'position_btc': 0.0,
            'market_price': 50000.0,
            'unrealized_pnl': 0.0
        },
        {
            'timestamp': '2022-01-01T01:00:00',
            'balance_usd': 10100.0,
            'position_btc': 0.1,
            'market_price': 51000.0,
            'unrealized_pnl': 100.0
        },
        {
            'timestamp': '2022-01-01T02:00:00',
            'balance_usd': 10200.0,
            'position_btc': 0.0,
            'market_price': 52000.0,
            'unrealized_pnl': 0.0
        }
    ]
    
    # Mock action history
    mock_env.action_history = [
        {
            'timestamp': '2022-01-01T00:00:00',
            'action': 0.7,
            'position_btc': 0.0,
            'market_price': 50000.0,
            'entry_price': 0.0
        },
        {
            'timestamp': '2022-01-01T01:00:00',
            'action': 0.7,
            'position_btc': 0.1,
            'market_price': 51000.0,
            'entry_price': 50000.0
        },
        {
            'timestamp': '2022-01-01T02:00:00',
            'action': 0.3,
            'position_btc': 0.0,
            'market_price': 52000.0,
            'entry_price': 0.0
        }
    ]
    
    yield mock_env

@pytest.fixture
def mock_model():
    """Fixture providing a mock Stable Baselines 3 SAC model."""
    with patch('stable_baselines3.SAC') as mock_sac:
        # Configure the mock model
        mock_model = mock_sac.load.return_value
        mock_model.predict.return_value = (np.array([0.5]), None)  # action, state
        
        yield mock_model

@pytest.fixture
def sample_backtester(mock_storage_client):
    """Fixture providing a sample Backtester instance."""
    env_params = {
        'project_id': 'test-project',
        'gcs_processed_data_uri': 'gs://bucket/path/to/data',
        'initial_balance_usd': 10000.0,
        'max_position_btc': 1.0,
        'commission_rate': 0.0004,
        'max_leverage': 20
    }
    
    backtester = Backtester(
        project_id='test-project',
        model_uri='gs://bucket/path/to/model.zip',
        test_data_uri='gs://bucket/path/to/test_data',
        env_params=env_params
    )
    
    return backtester

def test_backtester_initialization(sample_backtester):
    """Test initialization of Backtester class."""
    assert sample_backtester is not None
    assert sample_backtester.project_id == 'test-project'
    assert sample_backtester.model_uri == 'gs://bucket/path/to/model.zip'
    assert sample_backtester.test_data_uri == 'gs://bucket/path/to/test_data'
    assert sample_backtester.env_params['initial_balance_usd'] == 10000.0
    assert sample_backtester.model is None
    assert sample_backtester.env is None
    assert sample_backtester.results == {}

@patch('google.cloud.storage.Client')
def test_download_from_gcs(mock_client, sample_backtester):
    """Test downloading from GCS."""
    # Configure the mock
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Set up test data
    gcs_uri = 'gs://bucket/path/to/file.txt'
    local_path = '/tmp/file.txt'
    
    # Test the method
    sample_backtester.download_from_gcs(gcs_uri, local_path)
    
    # Verify the calls
    mock_client.return_value.bucket.assert_called_once_with('bucket')
    mock_bucket.blob.assert_called_once_with('path/to/file.txt')
    mock_blob.download_to_filename.assert_called_once_with(local_path)

@patch('google.cloud.storage.Client')
def test_upload_to_gcs(mock_client, sample_backtester):
    """Test uploading to GCS."""
    # Configure the mock
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Set up test data
    local_path = '/tmp/file.txt'
    gcs_uri = 'gs://bucket/path/to/file.txt'
    
    # Test the method
    sample_backtester.upload_to_gcs(local_path, gcs_uri)
    
    # Verify the calls
    mock_client.return_value.bucket.assert_called_once_with('bucket')
    mock_bucket.blob.assert_called_once_with('path/to/file.txt')
    mock_blob.upload_from_filename.assert_called_once_with(local_path)

@patch('stable_baselines3.SAC.load')
def test_load_model(mock_load, sample_backtester, mock_storage_client):
    """Test loading a model."""
    # Configure the mock
    mock_bucket = mock_storage_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    
    # Test the method
    sample_backtester.load_model()
    
    # Verify the calls
    mock_bucket.blob.assert_called_once()
    mock_blob.download_to_filename.assert_called_once()
    mock_load.assert_called_once()

@patch('src.environments.trading_env.TradingEnvironment')
def test_setup_environment(mock_env_class, sample_backtester):
    """Test setting up the environment."""
    # Configure the mock
    mock_env_instance = MagicMock()
    mock_env_class.return_value = mock_env_instance
    
    # Test the method
    sample_backtester.setup_environment()
    
    # Verify the calls
    mock_env_class.assert_called_once()
    assert sample_backtester.env == mock_env_instance
    assert sample_backtester.env_params['gcs_processed_data_uri'] == 'gs://bucket/path/to/test_data'

def test_run_backtest(sample_backtester, mock_model, mock_env):
    """Test running a backtest."""
    # Configure the test
    sample_backtester.model = mock_model
    sample_backtester.env = mock_env
    
    # Configure the mock environment's info dictionary
    mock_env.step.return_value = (
        {},  # obs
        0.1,  # reward
        False,  # terminated
        True,  # truncated (to end the episode)
        {
            'final_balance_usd': 10200.0,
            'sharpe_ratio': 1.5,
            'max_drawdown_pct': 2.0,
            'total_trades': 1,
            'winning_trades': 1
        }  # info
    )
    
    # Test the method
    results = sample_backtester.run_backtest(n_episodes=1)
    
    # Verify the calls
    mock_env.reset.assert_called_once()
    mock_model.predict.assert_called()
    mock_env.step.assert_called()
    
    # Check the results
    assert 'final_balance_usd' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown_pct' in results
    assert 'win_rate' in results
    assert 'total_trades' in results
    assert 'portfolio_history' in results
    assert 'action_history' in results
    
    # Check the values
    assert results['final_balance_usd']['mean'] == 10200.0
    assert results['sharpe_ratio']['mean'] == 1.5
    assert results['max_drawdown_pct']['mean'] == 2.0
    assert results['win_rate']['mean'] == 1.0
    assert results['total_trades']['mean'] == 1

def test_calculate_metrics(sample_backtester):
    """Test calculating additional metrics."""
    # Set up test data
    sample_backtester.results = {
        'final_balance_usd': {'mean': 10200.0, 'values': [10200.0]},
        'sharpe_ratio': {'mean': 1.5, 'values': [1.5]},
        'max_drawdown_pct': {'mean': 2.0, 'values': [2.0]},
        'total_trades': {'mean': 1, 'values': [1]},
        'win_rate': {'mean': 1.0, 'values': [1.0]},
        'portfolio_history': [
            [
                {'timestamp': '2022-01-01T00:00:00', 'balance_usd': 10000.0},
                {'timestamp': '2022-01-01T01:00:00', 'balance_usd': 10100.0},
                {'timestamp': '2022-01-01T02:00:00', 'balance_usd': 10200.0}
            ]
        ],
        'action_history': [
            [
                {'timestamp': '2022-01-01T00:00:00', 'position_btc': 0.0, 'market_price': 50000.0},
                {'timestamp': '2022-01-01T01:00:00', 'position_btc': 0.1, 'market_price': 51000.0, 'entry_price': 50000.0},
                {'timestamp': '2022-01-01T02:00:00', 'position_btc': 0.0, 'market_price': 52000.0}
            ]
        ]
    }
    
    # Test the method
    metrics = sample_backtester.calculate_metrics()
    
    # Check the metrics
    assert 'annualized_return' in metrics
    assert 'annualized_volatility' in metrics
    assert 'sortino_ratio' in metrics
    assert 'calmar_ratio' in metrics
    assert 'profit_factor' in metrics
    assert 'avg_trade_duration' in metrics
    assert 'largest_winning_trade' in metrics
    assert 'largest_losing_trade' in metrics

def test_generate_plots(sample_backtester):
    """Test generating plots."""
    # Set up test data
    sample_backtester.results = {
        'final_balance_usd': {'mean': 10200.0, 'values': [10200.0]},
        'sharpe_ratio': {'mean': 1.5, 'values': [1.5]},
        'max_drawdown_pct': {'mean': 2.0, 'values': [2.0]},
        'total_trades': {'mean': 1, 'values': [1]},
        'win_rate': {'mean': 1.0, 'values': [1.0]},
        'portfolio_history': [
            [
                {'timestamp': '2022-01-01T00:00:00', 'balance_usd': 10000.0, 'position_btc': 0.0},
                {'timestamp': '2022-01-01T01:00:00', 'balance_usd': 10100.0, 'position_btc': 0.1},
                {'timestamp': '2022-01-01T02:00:00', 'balance_usd': 10200.0, 'position_btc': 0.0}
            ]
        ],
        'action_history': [
            [
                {'timestamp': '2022-01-01T00:00:00', 'position_btc': 0.0, 'market_price': 50000.0},
                {'timestamp': '2022-01-01T01:00:00', 'position_btc': 0.1, 'market_price': 51000.0, 'entry_price': 50000.0},
                {'timestamp': '2022-01-01T02:00:00', 'position_btc': 0.0, 'market_price': 52000.0}
            ]
        ]
    }
    
    # Mock environment parameters
    sample_backtester.env_params = {'initial_balance_usd': 10000.0}
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test the method
        plot_files = sample_backtester.generate_plots(temp_dir)
        
        # Check that plot files were created
        assert len(plot_files) > 0
        for plot_file in plot_files:
            assert os.path.exists(plot_file)

def test_save_results(sample_backtester, mock_storage_client):
    """Test saving results to GCS."""
    # Set up test data
    sample_backtester.results = {
        'final_balance_usd': {'mean': 10200.0, 'values': [10200.0]},
        'sharpe_ratio': {'mean': 1.5, 'values': [1.5]},
        'max_drawdown_pct': {'mean': 2.0, 'values': [2.0]},
        'total_trades': {'mean': 1, 'values': [1]},
        'win_rate': {'mean': 1.0, 'values': [1.0]},
        'portfolio_history': [
            [
                {'timestamp': '2022-01-01T00:00:00', 'balance_usd': 10000.0, 'position_btc': 0.0},
                {'timestamp': '2022-01-01T01:00:00', 'balance_usd': 10100.0, 'position_btc': 0.1},
                {'timestamp': '2022-01-01T02:00:00', 'balance_usd': 10200.0, 'position_btc': 0.0}
            ]
        ],
        'action_history': [
            [
                {'timestamp': '2022-01-01T00:00:00', 'position_btc': 0.0, 'market_price': 50000.0},
                {'timestamp': '2022-01-01T01:00:00', 'position_btc': 0.1, 'market_price': 51000.0, 'entry_price': 50000.0},
                {'timestamp': '2022-01-01T02:00:00', 'position_btc': 0.0, 'market_price': 52000.0}
            ]
        ]
    }
    
    # Mock methods that interact with external systems
    sample_backtester.generate_plots = MagicMock(return_value=['/tmp/plot1.png', '/tmp/plot2.png'])
    sample_backtester.upload_to_gcs = MagicMock()
    
    # Test the method
    output_gcs_uri = 'gs://bucket/path/to/results'
    sample_backtester.save_results(output_gcs_uri)
    
    # Verify the calls
    sample_backtester.generate_plots.assert_called_once()
    
    # Check that upload_to_gcs was called at least 3 times
    # (1 for metrics.json, 2 for plot files)
    assert sample_backtester.upload_to_gcs.call_count >= 3
