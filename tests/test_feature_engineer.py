"""
Unit tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.preprocessing.feature_engineer import FeatureEngineer

@pytest.fixture
def sample_ohlcv_data():
    """Fixture providing sample OHLCV data for testing."""
    # Create sample data for testing
    dates = pd.date_range(start='2020-01-01', periods=100, freq='1h')
    data = {
        'timestamp': dates,
        'open': np.random.uniform(8000, 12000, 100),
        'high': np.random.uniform(9000, 12500, 100),
        'low': np.random.uniform(7500, 11000, 100),
        'close': np.random.uniform(8000, 12000, 100),
        'volume': np.random.uniform(1, 100, 100) * 1000000
    }
    
    # Ensure high is always higher than open, close, and low
    for i in range(len(data['high'])):
        data['high'][i] = max(data['open'][i], data['close'][i]) + np.random.uniform(100, 500)
    
    # Ensure low is always lower than open, close, and high
    for i in range(len(data['low'])):
        data['low'][i] = min(data['open'][i], data['close'][i]) - np.random.uniform(100, 500)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def test_feature_engineer_initialization():
    """Test initialization of FeatureEngineer class."""
    engineer = FeatureEngineer()
    assert engineer is not None
    
    # Test with custom parameters
    engineer = FeatureEngineer(
        sma_periods=[10, 50, 200],
        ema_periods=[5, 20, 100],
        rsi_period=14
    )
    assert engineer.sma_periods == [10, 50, 200]
    assert engineer.ema_periods == [5, 20, 100]
    assert engineer.rsi_period == 14

def test_calculate_features(sample_ohlcv_data):
    """Test the feature calculation functionality."""
    engineer = FeatureEngineer()
    
    # Calculate features
    features_df = engineer.calculate_features(sample_ohlcv_data)
    
    # Check that the result is a DataFrame
    assert isinstance(features_df, pd.DataFrame)
    
    # Check that the original columns are preserved
    for col in sample_ohlcv_data.columns:
        assert col in features_df.columns
    
    # Check that the expected features are added
    # SMAs
    for period in engineer.sma_periods:
        assert f'sma_{period}' in features_df.columns
    
    # EMAs
    for period in engineer.ema_periods:
        assert f'ema_{period}' in features_df.columns
    
    # RSI
    assert f'rsi_{engineer.rsi_period}' in features_df.columns
    
    # MACD
    assert 'macd' in features_df.columns
    assert 'macd_signal' in features_df.columns
    assert 'macd_histogram' in features_df.columns
    
    # Bollinger Bands
    assert 'bb_upper' in features_df.columns
    assert 'bb_lower' in features_df.columns
    assert 'bb_middle' in features_df.columns
    
    # ATR
    assert 'atr' in features_df.columns
    
    # Check that the number of rows in the result is the same as the input
    assert len(features_df) == len(sample_ohlcv_data)

def test_features_have_valid_values(sample_ohlcv_data):
    """Test that calculated features have valid (non-NaN) values where expected."""
    engineer = FeatureEngineer()
    features_df = engineer.calculate_features(sample_ohlcv_data)
    
    # Define the expected number of NaN values for each feature type
    # based on the lookback periods
    max_sma_period = max(engineer.sma_periods)
    max_ema_period = max(engineer.ema_periods)
    max_period = max(max_sma_period, max_ema_period, engineer.rsi_period, 
                    engineer.bb_period, engineer.atr_period, 
                    engineer.macd_fast_period + engineer.macd_signal_period)
    
    # Check for valid values after the maximum lookback period
    for col in features_df.columns:
        if col in sample_ohlcv_data.columns:
            # Original columns should have no NaN values
            assert features_df[col].isna().sum() == 0
        else:
            # Technical indicators will have NaN values at the beginning
            # Check that there are valid values after the lookback period
            assert features_df[col].iloc[max_period:].isna().sum() == 0

def test_feature_relationships(sample_ohlcv_data):
    """Test the logical relationships between calculated features."""
    engineer = FeatureEngineer()
    features_df = engineer.calculate_features(sample_ohlcv_data)
    
    # Define the index from which we expect valid values for all features
    valid_idx = max(
        max(engineer.sma_periods),
        max(engineer.ema_periods),
        engineer.rsi_period,
        engineer.bb_period,
        engineer.atr_period,
        engineer.macd_fast_period + engineer.macd_signal_period
    )
    
    valid_df = features_df.iloc[valid_idx:]
    
    # Test Bollinger Bands relationships
    assert (valid_df['bb_upper'] >= valid_df['bb_middle']).all()
    assert (valid_df['bb_middle'] >= valid_df['bb_lower']).all()
    
    # Test RSI is between 0 and 100
    assert (valid_df[f'rsi_{engineer.rsi_period}'] >= 0).all()
    assert (valid_df[f'rsi_{engineer.rsi_period}'] <= 100).all()
    
    # Test ATR is positive
    assert (valid_df['atr'] >= 0).all()
    
    # Test MACD histogram matches the difference between MACD and signal line
    np.testing.assert_almost_equal(
        valid_df['macd_histogram'].values,
        (valid_df['macd'] - valid_df['macd_signal']).values,
        decimal=6
    )
