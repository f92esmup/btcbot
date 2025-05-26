#!/usr/bin/env python3
"""
Simple script to verify the data pipeline works with our enhanced Preprocessor.
This script directly tests the fix in a simple, controlled way.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
print("Starting simplified pipeline test")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
print("Loading modules...")
from src.utils.config import ConfigManager
from src.data.preprocessor import Preprocessor

print("Creating sample data...")
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

def main():
    """Main test function"""
    print("Setting up test environment...")
    os.environ['BTCBOT_TESTING_MODE'] = 'true'
    os.environ['GCP_PROJECT_ID'] = 'test-project'
    os.environ['GCS_BUCKET_NAME'] = 'test-bucket'
    
    print("Loading configuration...")
    config = ConfigManager.load_config('/workspaces/btcbot/src/config.yaml')
    print("Config loaded. Keys:", ", ".join(config.keys()))
    
    print("Creating broken config (removing a feature column)...")
    broken_config = config.copy()
    feature_to_remove = 'log_ret_C_O_norm'
    broken_config['preprocessing']['final_market_feature_columns'] = [
        col for col in config['preprocessing']['final_market_feature_columns']
        if col != feature_to_remove
    ]
    
    print(f"Creating Preprocessor with broken config (missing '{feature_to_remove}')...")
    preprocessor = Preprocessor(broken_config, testing_mode=True)
    
    print("Creating sample data...")
    sample_data = create_sample_ohlcv_data(500)
    print(f"Sample data shape: {sample_data.shape}")
    
    print("Processing data with modified config...")
    try:
        sequences = preprocessor.process_market_data(sample_data)
        if sequences is not None:
            print(f"SUCCESS! Sequences shape: {sequences.shape}")
            print("✅ The KeyError fix is working correctly!")
        else:
            print("❌ Processing returned None")
    except Exception as e:
        print(f"❌ Error processing market data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
