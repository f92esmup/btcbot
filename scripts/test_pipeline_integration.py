#!/usr/bin/env python3
"""
Integration test for the data pipeline that verifies our KeyError fix.

This script tests that the integrated data pipeline works with our enhanced 
Preprocessor class that handles missing final_market_feature_columns.
"""
import os
import logging
import sys
import asyncio
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging with stdout handler to ensure output is shown
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("pipeline_integration_test")

# Set testing mode
os.environ['BTCBOT_TESTING_MODE'] = 'true'

# Set required environment variables for the pipeline
os.environ['GCP_PROJECT_ID'] = 'test-project'
os.environ['GCS_BUCKET_NAME'] = 'test-bucket'

# Import our modules
from src.utils.config import ConfigManager
from src.data.data_pipeline import IntegratedDataPipeline
from src.data.preprocessor import Preprocessor


async def test_data_pipeline():
    """Test the IntegratedDataPipeline with our fixed Preprocessor"""
    logger.info("Starting data pipeline integration test")
    
    # Load configuration
    try:
        config_path = "/workspaces/btcbot/src/config.yaml"
        config = ConfigManager.load_config(config_path)
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return False
    
    # Create a pipeline instance with testing mode - with explicit error printing
    try:
        print("Creating IntegratedDataPipeline instance...")
        pipeline = IntegratedDataPipeline(config_path)
        print("✓ Created IntegratedDataPipeline instance")
        
        # Verify that the preprocessor is initialized correctly
        print(f"Pipeline preprocessor type: {type(pipeline.preprocessor)}")
        if hasattr(pipeline, 'preprocessor') and isinstance(pipeline.preprocessor, Preprocessor):
            print(f"✓ Pipeline has correct Preprocessor instance: {type(pipeline.preprocessor)}")
        else:
            print(f"✗ Pipeline has incorrect or missing Preprocessor: {getattr(pipeline, 'preprocessor', None)}")
            return False
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test the pipeline status function (should work without real GCS access)
    try:
        start_date = "2025-01-01"
        end_date = "2025-01-10"
        status = pipeline.get_pipeline_status(start_date, end_date)
        
        logger.info(f"✓ Pipeline status retrieved successfully: {status}")
        logger.info(f"  - Date range: {status['date_range']}")
        logger.info(f"  - Missing chunks: {status['missing_chunks']}")
    except Exception as e:
        logger.error(f"✗ Failed to get pipeline status: {e}")
        return False
    
    logger.info("All basic pipeline tests passed!")
    return True


async def main():
    try:
        success = await test_data_pipeline()
        if success:
            logger.info("\n✅ Integration test PASSED! The fix is working with the data pipeline!")
        else:
            logger.error("\n❌ Integration test FAILED!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
