#!/usr/bin/env python3
"""
Simple test script to diagnose module import issues
"""
import os
import sys
import logging

# Configure logging with stdout handler to ensure output is shown
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"sys.path: {sys.path}")

try:
    import yaml
    logger.info("Successfully imported yaml")
except ImportError as e:
    logger.error(f"Failed to import yaml: {e}")

try:
    from src.utils.config import ConfigManager
    logger.info("Successfully imported load_config")
except ImportError as e:
    logger.error(f"Failed to import load_config: {e}")

try:
    from src.data.preprocessor import Preprocessor
    logger.info("Successfully imported Preprocessor")
except ImportError as e:
    logger.error(f"Failed to import Preprocessor: {e}")

logger.info("Test script completed")
