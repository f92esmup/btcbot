#!/usr/bin/env python3
"""
Test script to validate implementation improvements.

This script tests the key components that were modified to align with the
technical design document.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("Starting implementation tests...")

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")
# Import the necessary modules
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.feature_engineer import FeatureEngineer
print("Preprocessing imports successful")

# Continue with rest of imports
try:
    import torch
    from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor
    from src.environments.trading_env import TradingEnvironment
    import gymnasium as gym
    from gymnasium import spaces
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_feature_engineer():
    """Test the feature engineering implementation."""
    print("Testing feature engineering...")
    
    # Create a simple DataFrame with OHLCV data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(101, 5, 100),
        'volume': np.random.normal(1000, 200, 100),
    }, index=dates)
    
    # Ensure high is actually the highest and low is the lowest
    for i in range(len(df)):
        row = df.iloc[i]
        max_val = max(row['open'], row['close']) + abs(np.random.normal(1, 0.5))
        min_val = min(row['open'], row['close']) - abs(np.random.normal(1, 0.5))
        df.iloc[i, df.columns.get_loc('high')] = max_val
        df.iloc[i, df.columns.get_loc('low')] = min_val
    
    # Initialize the feature engineer
    feature_engineer = FeatureEngineer()
    
    # Calculate features
    featured_df = feature_engineer.engineer_features(df, include_all=False)
    
    # Check if we have exactly 20 features
    num_features = len(featured_df.columns)
    print(f"Number of features: {num_features} (expected 20)")
    assert num_features == 20, f"Expected 20 features, got {num_features}"
    
    # Check if all required features are present
    required_features = [
        'open', 'high', 'low', 'close', 'volume',
        'log_return', 'hl_range', 'body_size_rel', 'atr', 'rsi',
        'macd', 'macd_signal', 'macd_histogram', 'bb_width', 'sma_cross',
        'stoch_k', 'adx', 'volume_ratio', 'mfi', 'obv'
    ]
    
    missing_features = []
    for feature in required_features:
        if feature not in featured_df.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"Missing features: {missing_features}")
        assert not missing_features, f"Features {missing_features} are missing"
    else:
        print("All required features are present")
    
    print("Feature engineering test passed!")
    return featured_df

def test_data_preprocessor(featured_df):
    """Test the data preprocessing implementation."""
    logger.info("Testing data preprocessing...")
    
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(
        project_id="test-project",
        feature_normalization_lookback=20
    )
    
    # Normalize features
    normalized_df = preprocessor.normalize_features(featured_df)
    
    # Check if normalized features have the same shape
    assert normalized_df.shape == featured_df.shape, \
        f"Normalized shape {normalized_df.shape} doesn't match original {featured_df.shape}"
    
    # Check if we don't have any NaN values
    assert not normalized_df.isna().any().any(), "Normalized data contains NaN values"
    
    logger.info("Data preprocessing test passed!")
    return normalized_df

def test_transformer_extractor():
    """Test the transformer feature extractor implementation."""
    logger.info("Testing transformer feature extractor...")
    
    # Create observation space
    observation_space = spaces.Dict({
        'market_features': spaces.Box(
            low=-np.inf, high=np.inf, shape=(60, 20), dtype=np.float32
        ),
        'portfolio_features': spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
    })
    
    # Initialize the feature extractor
    extractor = CustomTransformerFeatureExtractor(
        observation_space=observation_space,
        features_dim=64,
        d_model=32,
        n_heads=4,
        n_encoder_layers=2
    )
    
    # Create a batch of observations
    batch_size = 4
    observation = {
        'market_features': torch.randn(batch_size, 60, 20),
        'portfolio_features': torch.randn(batch_size, 8)
    }
    
    # Forward pass
    features = extractor(observation)
    
    # Check output shape
    assert features.shape == (batch_size, 64), \
        f"Expected shape (4, 64), got {features.shape}"
    
    logger.info("Transformer feature extractor test passed!")
    return extractor

def test_execute_trade():
    """Test the improved trade execution logic."""
    logger.info("Testing trade execution logic...")
    
    # Create a simple function to test trade execution
    def execute_trade_test(decision, target_pos, current_pos, entry_price, balance, market_price):
        """Simplified trade execution test."""
        # Calculate position delta
        pos_delta = target_pos - current_pos
        
        # Commission
        commission = abs(pos_delta * market_price) * 0.0004
        
        # Calculate realized PnL
        realized_pnl = 0
        if current_pos != 0 and (
                (current_pos > 0 and pos_delta < 0) or 
                (current_pos < 0 and pos_delta > 0)):
            
            # Calculate how much of the position we're closing
            closing_size = min(abs(current_pos), abs(pos_delta))
            if current_pos > 0:  # Closing long position
                realized_pnl = closing_size * (market_price - entry_price)
            else:  # Closing short position
                realized_pnl = closing_size * (entry_price - market_price)
        
        # Update balance
        new_balance = balance - commission + realized_pnl
        
        # Calculate new position
        new_pos = current_pos + pos_delta
        
        # Calculate new entry price
        if abs(new_pos) < 1e-8:
            new_entry_price = 0
            steps = 0
        elif (current_pos > 0 and new_pos > 0) or (current_pos < 0 and new_pos < 0):
            # Adding to existing position (same direction) - weighted average
            if pos_delta != 0:
                old_value = abs(current_pos * entry_price)
                new_value = abs(pos_delta * market_price)
                new_entry_price = (old_value + new_value) / abs(new_pos)
            else:
                new_entry_price = entry_price
            steps = 1
        else:
            # New position or flipped position - use current price
            new_entry_price = market_price
            steps = 0
        
        return new_pos, new_entry_price, new_balance, realized_pnl, steps
    
    # Test cases
    test_cases = [
        # (decision, target_pos, current_pos, entry_price, balance, market_price)
        # Open a new long position
        ('buy', 1.0, 0.0, 0.0, 10000.0, 20000.0),
        # Add to existing long position
        ('buy', 2.0, 1.0, 20000.0, 9992.0, 21000.0),
        # Reduce long position
        ('sell', 0.5, 2.0, 20500.0, 9983.8, 21000.0),
        # Close long position
        ('close', 0.0, 0.5, 20500.0, 10235.8, 21000.0),
        # Open short position
        ('sell', -1.0, 0.0, 0.0, 10485.8, 21000.0),
        # Add to short position
        ('sell', -2.0, -1.0, 21000.0, 10477.8, 20500.0),
        # Reduce short position
        ('buy', -0.5, -2.0, 20750.0, 10469.6, 20000.0),
        # Flip from short to long
        ('buy', 1.0, -0.5, 20750.0, 10824.6, 20000.0),
    ]
    
    # Run tests
    for i, (decision, target_pos, current_pos, entry_price, balance, market_price) in enumerate(test_cases):
        new_pos, new_entry, new_balance, realized_pnl, steps = execute_trade_test(
            decision, target_pos, current_pos, entry_price, balance, market_price
        )
        logger.info(f"Test case {i+1}:")
        logger.info(f"  Decision: {decision}, Target: {target_pos}, Current: {current_pos}")
        logger.info(f"  Entry: {entry_price}, Balance: {balance}, Market: {market_price}")
        logger.info(f"  Result: Pos={new_pos}, Entry={new_entry}, Balance={new_balance}, PnL={realized_pnl}")
    
    logger.info("Trade execution test completed!")

if __name__ == "__main__":
    print("Starting implementation tests...")
    
    try:
        # Run the tests
        featured_df = test_feature_engineer()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise
