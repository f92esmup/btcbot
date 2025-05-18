"""
Data acquisition component for the BTC Trading Bot pipeline.

This script serves as the entry point for the Vertex AI Pipeline component
responsible for downloading historical OHLCV data from Binance Futures API
and saving it to Google Cloud Storage.
"""

import os
import argparse
import logging
import json
import yaml
from datetime import datetime
from typing import Optional, Dict, Any

from google.cloud import storage
import kfp
from kfp.v2.dsl import Output, Dataset

from src.data_acquisition.binance_downloader import BinanceFuturesDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the data acquisition component.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Download historical data from Binance Futures API')
    
    # Required arguments
    parser.add_argument('--project_id', type=str, required=True,
                        help='Google Cloud project ID')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True,
                        help='Kline interval (e.g., 1h, 4h, 1d)')
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date in format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date in format YYYY-MM-DD')
    parser.add_argument('--gcs_bucket', type=str, required=True,
                        help='GCS bucket name for storing the data')
    
    # Optional arguments
    parser.add_argument('--gcs_prefix', type=str, default=None,
                        help='Prefix for GCS path (default: None)')
    parser.add_argument('--batch_size_days', type=int, default=30,
                        help='Number of days to download in each batch (default: 30)')
    parser.add_argument('--secret_name', type=str, default=None,
                        help='Name of the secret in Secret Manager containing API credentials')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Binance API key (if not using Secret Manager)')
    parser.add_argument('--api_secret', type=str, default=None,
                        help='Binance API secret (if not using Secret Manager)')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum number of retries for API calls (default: 5)')
    parser.add_argument('--retry_delay_base', type=int, default=10,
                        help='Base delay (in seconds) for exponential backoff (default: 10)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to YAML configuration file (can be a GCS path)')
    
    # KFP v2 Output artifacts
    parser.add_argument('--data_output_path', type=str, required=True,
                        help='Output path for the downloaded data artifact')
    
    return parser.parse_args()

def load_config_from_gcs(gcs_path: str, project_id: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file stored in GCS.
    
    Args:
        gcs_path (str): GCS path to the YAML configuration file.
        project_id (str): Google Cloud project ID.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
        
    Raises:
        Exception: If there's an error loading the configuration.
    """
    try:
        # Parse bucket and blob path from GCS URI
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path[5:]  # Remove 'gs://' prefix
        
        bucket_name, blob_path = gcs_path.split('/', 1)
        
        # Initialize GCS client and download the file
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download as string and parse YAML
        yaml_content = blob.download_as_text()
        config = yaml.safe_load(yaml_content)
        
        logger.info(f"Successfully loaded configuration from {gcs_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from GCS: {str(e)}")
        raise

def run_data_acquisition(
    project_id: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    gcs_bucket: str,
    gcs_prefix: Optional[str] = None,
    batch_size_days: int = 30,
    secret_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    max_retries: int = 5,
    retry_delay_base: int = 10,
    data_output: Output[Dataset] = None
) -> str:
    """
    Run the data acquisition process to download historical OHLCV data from Binance Futures.
    
    Args:
        project_id (str): Google Cloud project ID.
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        interval (str): Kline interval (e.g., '1h', '4h', '1d').
        start_date (str): Start date in format 'YYYY-MM-DD'.
        end_date (str): End date in format 'YYYY-MM-DD'.
        gcs_bucket (str): GCS bucket name.
        gcs_prefix (str, optional): Prefix for the GCS path. Defaults to None.
        batch_size_days (int, optional): Number of days to download in each batch. Defaults to 30.
        secret_name (str, optional): Name of the secret in Secret Manager containing API credentials.
        api_key (str, optional): Binance API key (if not using Secret Manager).
        api_secret (str, optional): Binance API secret (if not using Secret Manager).
        max_retries (int, optional): Maximum number of retries for API calls. Defaults to 5.
        retry_delay_base (int, optional): Base delay (in seconds) for exponential backoff. Defaults to 10.
        data_output (Output[Dataset], optional): KFP output parameter for the data artifact.
        
    Returns:
        str: GCS URI of the saved Parquet file.
        
    Raises:
        Exception: If there's an error in the data acquisition process.
    """
    try:
        logger.info(f"Starting data acquisition process for {symbol} {interval} from {start_date} to {end_date}")
        
        # Initialize the downloader
        downloader = BinanceFuturesDownloader(
            api_key=api_key,
            api_secret=api_secret,
            project_id=project_id,
            secret_name=secret_name,
            retry_delay_base=retry_delay_base,
            max_retries=max_retries
        )
        
        # Download historical data
        output_gcs_uri = downloader.download_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            batch_size_days=batch_size_days
        )
        
        logger.info(f"Data acquisition completed successfully. Data saved to: {output_gcs_uri}")
        
        # Store output for KFP
        if data_output:
            # Write metadata to the output path
            metadata = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'gcs_uri': output_gcs_uri,
                'download_timestamp': datetime.now().isoformat(),
                'record_count': 'unknown'  # Would need to read Parquet to know this
            }
            
            # Write metadata as JSON to the output location
            with open(data_output.path, 'w') as f:
                json.dump(metadata, f)
            
            # Also store the GCS URI in the metadata field
            data_output.metadata['gcs_uri'] = output_gcs_uri
        
        return output_gcs_uri
        
    except Exception as e:
        logger.error(f"Error in data acquisition process: {str(e)}")
        raise

def main():
    """
    Main entry point for the data acquisition component.
    """
    args = parse_arguments()
    
    try:
        # Load configuration from file if specified
        config = {}
        if args.config_file:
            if args.config_file.startswith('gs://'):
                config = load_config_from_gcs(args.config_file, args.project_id)
            else:
                with open(args.config_file, 'r') as f:
                    config = yaml.safe_load(f)
        
        # Create the output parameter
        data_output = Output(type=Dataset, path=args.data_output_path)
        
        # Command line arguments take precedence over config file
        # Use a dictionary with default values from args, then update with config
        params = vars(args)
        
        # Update with config values only where args are None or not specified
        for key in config:
            if key in params and params[key] is None:
                params[key] = config[key]
        
        # Run data acquisition
        output_gcs_uri = run_data_acquisition(
            project_id=params['project_id'],
            symbol=params['symbol'],
            interval=params['interval'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            gcs_bucket=params['gcs_bucket'],
            gcs_prefix=params['gcs_prefix'],
            batch_size_days=params['batch_size_days'],
            secret_name=params['secret_name'],
            api_key=params['api_key'],
            api_secret=params['api_secret'],
            max_retries=params['max_retries'],
            retry_delay_base=params['retry_delay_base'],
            data_output=data_output
        )
        
        logger.info(f"Data acquisition component completed successfully. Output: {output_gcs_uri}")
        
    except Exception as e:
        logger.error(f"Error in data acquisition component: {str(e)}")
        raise

if __name__ == '__main__':
    main()