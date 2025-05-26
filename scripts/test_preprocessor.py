#!/usr/bin/env python3
"""
Enhanced Test script for Preprocessor class to verify the KeyError fix.

This script tests:
1. That the Preprocessor class can be instantiated with a raw config dictionary
2. That the DictConfigAdapter correctly handles environment variables
3. That the _apply_feature_normalization method properly handles missing columns
4. That the KeyError issue is fixed

Both the original synthetic data test and a "break-and-fix" test are included.
"""
import os
import logging
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import yaml

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging with stdout handler to ensure output is shown
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("preprocessor_test")

# Print some diagnostics
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# Set required environment variables for testing
os.environ['GCP_PROJECT_ID'] = 'test-project'
os.environ['GCS_BUCKET_NAME'] = 'test-bucket'

# Import our modules
from src.utils.config import ConfigManager
from src.data.preprocessor import Preprocessor, DataPreprocessor, DictConfigAdapter

def create_sample_ohlcv_data(num_rows=500):
    """Create sample OHLCV data for testing"""
    # Create a date range
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=num_rows-1)
    dates = pd.date_range(start=start_date, end=end_date, periods=num_rows)
    
    # Create price data with some random walk
    close = 30000 + np.cumsum(np.random.normal(0, 100, num_rows))
    high = close + np.abs(np.random.normal(0, 50, num_rows))
    low = close - np.abs(np.random.normal(0, 50, num_rows))
    open_price = close - np.random.normal(0, 30, num_rows)
    volume = np.random.normal(100, 20, num_rows) * 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open_Time': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    df.set_index('Open_Time', inplace=True)
    return df

def test_environment_variables():
    """Test that DictConfigAdapter correctly handles environment variables"""
    logger.info("Testing environment variable handling in DictConfigAdapter...")
    
    config = ConfigManager.load_config('/workspaces/btcbot/src/config.yaml')
    adapter = DictConfigAdapter(config)
    
    # Test GCP_PROJECT_ID
    project_id = adapter.get_env_variable('GCP_PROJECT_ID')
    if project_id == 'test-project':
        logger.info("✓ GCP_PROJECT_ID retrieved correctly")
    else:
        logger.error(f"✗ GCP_PROJECT_ID incorrect: {project_id}")
        return False
    
    # Test GCS_BUCKET_NAME
    bucket_name = adapter.get_env_variable('GCS_BUCKET_NAME')
    if bucket_name == 'test-bucket':
        logger.info("✓ GCS_BUCKET_NAME retrieved correctly")
    else:
        logger.error(f"✗ GCS_BUCKET_NAME incorrect: {bucket_name}")
        return False
        
    # Set a custom env variable for testing
    os.environ['TEST_VARIABLE'] = 'test-value'
    test_var = adapter.get_env_variable('TEST_VARIABLE')
    if test_var == 'test-value':
        logger.info("✓ Custom environment variable retrieved correctly")
    else:
        logger.error(f"✗ Custom environment variable incorrect: {test_var}")
        return False
        
    return True

def test_broken_config():
    """
    Deliberately break the configuration and verify that error handling 
    can recover using the added fallback mechanisms.
    """
    logger.info("Testing error handling with broken configuration...")
    
    config = ConfigManager.load_config('/workspaces/btcbot/src/config.yaml')
    
    # Create a broken config by removing a key feature column
    broken_config = config.copy()
    
    # Remove a feature that should be in final_market_feature_columns
    feature_to_remove = 'log_ret_C_O_norm'
    broken_config['preprocessing']['final_market_feature_columns'] = [
        col for col in config['preprocessing']['final_market_feature_columns'] 
        if col != feature_to_remove
    ]
    
    logger.info(f"Removed '{feature_to_remove}' from final_market_feature_columns")
    
    # Create preprocessor with broken config and testing mode
    preprocessor = Preprocessor(broken_config, testing_mode=True)
    logger.info("Created preprocessor with broken config (in testing mode)")
    
    # Process sample data
    sample_data = create_sample_ohlcv_data(500)
    logger.info(f"Created sample OHLCV data with shape: {sample_data.shape}")
    
    try:
        # This should not raise KeyError due to our fix
        sequences = preprocessor.process_market_data(sample_data)
        if sequences is not None:
            logger.info(f"✓ Successfully processed data despite broken config! Sequences shape: {sequences.shape}")
            return True
        else:
            logger.error("✗ Preprocessing returned None with broken config")
            return False
    except KeyError as e:
        logger.error(f"✗ KeyError still occurring: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Other exception occurred: {e}", exc_info=True)
        return False

def main():
    """Main test function with enhanced diagnostics"""
    logger.info("Starting Enhanced Preprocessor test")
    
    # Track overall success
    overall_success = True
    
    # Test 1: Environment variables handling
    if test_environment_variables():
        logger.info("✓ Environment variable test passed")
    else:
        logger.error("✗ Environment variable test failed")
        overall_success = False
    
    # Test 2: Normal operation
    logger.info("\nTesting normal operation...")
    
    # Load configuration
    try:
        config = ConfigManager.load_config('/workspaces/btcbot/src/config.yaml')
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        overall_success = False
        return
    
    # Create Preprocessor instance with testing mode enabled
    try:
        # Set testing mode environment variable for extra safety
        os.environ['BTCBOT_TESTING_MODE'] = 'true'
        
        # Create preprocessor with testing mode
        preprocessor = Preprocessor(config, testing_mode=True)
        logger.info("✓ Preprocessor instance created successfully (in testing mode)")
    except Exception as e:
        logger.error(f"✗ Failed to create Preprocessor: {e}")
        overall_success = False
        return
    
    # Create sample data
    sample_data = create_sample_ohlcv_data(500)
    logger.info(f"Created sample OHLCV data with shape: {sample_data.shape}")
    
    # Process data
    try:
        sequences = preprocessor.process_market_data(sample_data)
        if sequences is not None:
            logger.info(f"✓ Successfully processed data! Sequences shape: {sequences.shape}")
        else:
            logger.error("✗ Preprocessing returned None")
            overall_success = False
    except Exception as e:
        logger.error(f"✗ Error processing market data: {e}", exc_info=True)
        overall_success = False
        return
    
    # Test 3: Broken configuration test
    logger.info("\nTesting handling of broken configuration...")
    if test_broken_config():
        logger.info("✓ Broken configuration test passed")
    else:
        logger.error("✗ Broken configuration test failed")
        overall_success = False
    
    # Final results
    if overall_success:
        logger.info("\n✅ All tests PASSED! The KeyError fix is working correctly!")
    else:
        logger.error("\n❌ Some tests FAILED. Fix needs improvement.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        print("Starting test script execution...")
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
