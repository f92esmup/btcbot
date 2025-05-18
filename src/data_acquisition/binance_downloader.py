"""
Module for downloading historical OHLCV data from Binance Futures.

This module contains the BinanceFuturesDownloader class which connects to the Binance Futures API,
handles authentication, and downloads OHLCV data for a given symbol, interval, and date range.
The data is saved in Parquet format in Google Cloud Storage (GCS).
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List

# Third-party imports
from binance.client import Client
from binance.exceptions import BinanceAPIException
from google.cloud import storage
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import secretmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BinanceFuturesDownloader:
    """
    A class to download historical OHLCV data from Binance Futures API and save it to GCS.
    
    This class connects to the Binance Futures API, handles authentication using API keys 
    (read from Google Cloud Secret Manager), and downloads OHLCV data for a given symbol, 
    interval, and date range with robust error handling and retries.
    
    Attributes:
        api_key (str): Binance API key.
        api_secret (str): Binance API secret.
        client (binance.client.Client): Initialized Binance API client.
        project_id (str): Google Cloud project ID.
        retry_delay_base (int): Base delay (in seconds) for exponential backoff.
        max_retries (int): Maximum number of retries for API calls.
    """
    
    # Define constants
    KLINE_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                      'close_time', 'quote_asset_volume', 'number_of_trades', 
                      'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
    
    # Map Binance interval strings to milliseconds
    INTERVAL_TO_MS = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
        '1M': 30 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        project_id: str = None,
        secret_name: str = None,
        retry_delay_base: int = 10,
        max_retries: int = 5
    ):
        """
        Initialize the BinanceFuturesDownloader with API credentials and configuration.
        
        Args:
            api_key (str, optional): Binance API key. If not provided, it will be read from Secret Manager.
            api_secret (str, optional): Binance API secret. If not provided, it will be read from Secret Manager.
            project_id (str, optional): Google Cloud project ID for Secret Manager access.
            secret_name (str, optional): Name of the secret in Secret Manager containing API credentials.
            retry_delay_base (int, optional): Base delay (in seconds) for exponential backoff. Defaults to 10.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 5.
            
        Raises:
            ValueError: If neither direct API credentials nor Secret Manager details are provided.
        """
        self.logger = logging.getLogger(__name__)
        self.retry_delay_base = retry_delay_base
        self.max_retries = max_retries
        self.project_id = project_id
        
        # Initialize API credentials
        if api_key and api_secret:
            self.api_key = api_key
            self.api_secret = api_secret
        elif project_id and secret_name:
            self.api_key, self.api_secret = self._get_api_credentials_from_secret(project_id, secret_name)
        else:
            raise ValueError("Either provide API credentials directly or specify project_id and secret_name for Secret Manager.")
        
        # Initialize Binance client
        self.client = Client(self.api_key, self.api_secret)
        self.logger.info("BinanceFuturesDownloader initialized successfully.")
    
    def _get_api_credentials_from_secret(self, project_id: str, secret_name: str) -> Tuple[str, str]:
        """
        Retrieve Binance API credentials from Google Cloud Secret Manager.
        
        Args:
            project_id (str): Google Cloud project ID.
            secret_name (str): Name of the secret in Secret Manager.
            
        Returns:
            Tuple[str, str]: API key and API secret.
            
        Raises:
            Exception: If there's an error accessing the secret.
        """
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(name=secret_path)
            secret_data = response.payload.data.decode('UTF-8')
            
            # Parse the secret data - expected format is JSON string {"api_key": "...", "api_secret": "..."}
            import json
            credentials = json.loads(secret_data)
            api_key = credentials.get('api_key')
            api_secret = credentials.get('api_secret')
            
            if not api_key or not api_secret:
                raise ValueError(f"Secret {secret_name} does not contain required 'api_key' and 'api_secret' fields.")
            
            self.logger.info(f"Successfully retrieved API credentials from Secret Manager.")
            return api_key, api_secret
        
        except Exception as e:
            self.logger.error(f"Error accessing Secret Manager: {str(e)}")
            raise
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            Result of the function call.
            
        Raises:
            Exception: The last exception encountered after max_retries.
        """
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                last_exception = e
                
                # Don't retry on certain error codes (e.g., invalid symbol, authentication issues)
                if e.code in (-1121, -2013, -2014, -2015):
                    self.logger.error(f"Non-retriable Binance API error: {str(e)}")
                    raise
                
                # For rate limit errors, use the retry-after header if available
                retry_after = e.response.headers.get('Retry-After', None) if hasattr(e, 'response') else None
                
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = self.retry_delay_base * (2 ** (attempt - 1))  # Exponential backoff
                
                self.logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
        
        if last_exception:
            self.logger.error(f"Failed after {self.max_retries} attempts. Last error: {str(last_exception)}")
            raise last_exception
    
    def _get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1000
    ) -> List[List]:
        """
        Get klines (candlestick data) from Binance Futures API with retry logic.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            start_time (int): Start time in milliseconds.
            end_time (int): End time in milliseconds.
            limit (int, optional): Maximum number of klines to return (max 1000). Defaults to 1000.
            
        Returns:
            List[List]: List of klines data.
        """
        return self._retry_with_backoff(
            self.client.futures_klines,
            symbol=symbol,
            interval=interval,
            startTime=start_time,
            endTime=end_time,
            limit=limit
        )
    
    def _convert_klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Convert raw klines data to a pandas DataFrame with proper types.
        
        Args:
            klines (List[List]): Raw klines data from Binance API.
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data.
        """
        if not klines:
            return pd.DataFrame(columns=self.KLINE_COLUMNS)
            
        df = pd.DataFrame(klines, columns=self.KLINE_COLUMNS)
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_asset_volume', 'taker_buy_base_volume', 
                           'taker_buy_quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert number_of_trades to int
        df['number_of_trades'] = df['number_of_trades'].astype(int)
        
        # Drop the 'ignore' column
        if 'ignore' in df.columns:
            df = df.drop(columns=['ignore'])
        
        return df

    def _save_dataframe_to_gcs(
        self,
        df: pd.DataFrame,
        gcs_bucket: str,
        gcs_path: str
    ) -> str:
        """
        Save DataFrame to Google Cloud Storage as a Parquet file.
        
        Args:
            df (pd.DataFrame): DataFrame to save.
            gcs_bucket (str): GCS bucket name.
            gcs_path (str): Path within the bucket where to save the file.
            
        Returns:
            str: Full GCS URI of the saved file.
        """
        try:
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(gcs_bucket)
            
            # Convert DataFrame to Parquet format
            table = pa.Table.from_pandas(df)
            
            # Create an in-memory buffer
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer)
            
            # Get the finished buffer
            parquet_data = buffer.getvalue().to_pybytes()
            
            # Upload to GCS
            blob = bucket.blob(gcs_path)
            blob.upload_from_string(parquet_data, content_type='application/octet-stream')
            
            gcs_uri = f"gs://{gcs_bucket}/{gcs_path}"
            self.logger.info(f"Successfully saved data to {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            self.logger.error(f"Error saving data to GCS: {str(e)}")
            raise

    def download_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        gcs_bucket: str,
        gcs_prefix: str = None,
        batch_size_days: int = 30
    ) -> str:
        """
        Download historical OHLCV data from Binance Futures and save it to GCS.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            start_date (str): Start date in format 'YYYY-MM-DD'.
            end_date (str): End date in format 'YYYY-MM-DD'.
            gcs_bucket (str): GCS bucket name.
            gcs_prefix (str, optional): Prefix for the GCS path. Defaults to None.
            batch_size_days (int, optional): Number of days to download in each batch. Defaults to 30.
            
        Returns:
            str: GCS URI of the saved Parquet file.
            
        Raises:
            ValueError: If invalid parameters are provided.
            BinanceAPIException: If there's an error calling the Binance API.
            Exception: For other errors.
        """
        self.logger.info(f"Starting download of {symbol} {interval} data from {start_date} to {end_date}")
        
        # Validate input parameters
        if interval not in self.INTERVAL_TO_MS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(self.INTERVAL_TO_MS.keys())}")
        
        # Parse dates
        try:
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            
            if end_datetime < start_datetime:
                raise ValueError("End date must be greater than or equal to start date")
            
        except ValueError as e:
            self.logger.error(f"Date parsing error: {str(e)}")
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. Error: {str(e)}")
        
        # Convert dates to milliseconds timestamps
        start_ms = int(start_datetime.timestamp() * 1000)
        end_ms = int(end_datetime.timestamp() * 1000)
        
        # Calculate the interval duration in milliseconds
        interval_ms = self.INTERVAL_TO_MS[interval]
        
        # Calculate batch size in milliseconds
        batch_size_ms = batch_size_days * 24 * 60 * 60 * 1000
        
        # Create dataframe to store all data
        all_data = pd.DataFrame()
        
        # Fetch data in batches to handle API limitations
        current_start_ms = start_ms
        
        while current_start_ms < end_ms:
            # Calculate batch end time
            current_end_ms = min(current_start_ms + batch_size_ms, end_ms)
            
            # Maximum number of candles in a batch = batch_size_ms / interval_ms
            max_candles = batch_size_ms // interval_ms
            
            # Binance API limit is 1000 candles per request
            if max_candles > 1000:
                # If we need more than 1000 candles, split into sub-batches
                sub_batch_size_ms = 1000 * interval_ms
                sub_start_ms = current_start_ms
                
                while sub_start_ms < current_end_ms:
                    sub_end_ms = min(sub_start_ms + sub_batch_size_ms, current_end_ms)
                    
                    self.logger.info(f"Fetching sub-batch: {datetime.fromtimestamp(sub_start_ms/1000)} to {datetime.fromtimestamp(sub_end_ms/1000)}")
                    
                    # Fetch sub-batch
                    klines = self._get_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=sub_start_ms,
                        end_time=sub_end_ms,
                        limit=1000
                    )
                    
                    # Convert to DataFrame and append
                    batch_df = self._convert_klines_to_dataframe(klines)
                    all_data = pd.concat([all_data, batch_df], ignore_index=True)
                    
                    # Move to next sub-batch
                    sub_start_ms = sub_end_ms
                    
                    # Implement a small delay to avoid hitting API rate limits
                    time.sleep(1)
            else:
                # If less than 1000 candles, fetch in one request
                self.logger.info(f"Fetching batch: {datetime.fromtimestamp(current_start_ms/1000)} to {datetime.fromtimestamp(current_end_ms/1000)}")
                
                klines = self._get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start_ms,
                    end_time=current_end_ms,
                    limit=1000
                )
                
                batch_df = self._convert_klines_to_dataframe(klines)
                all_data = pd.concat([all_data, batch_df], ignore_index=True)
            
            # Move to next batch
            current_start_ms = current_end_ms
            
            # Log progress
            progress_pct = min(100, (current_start_ms - start_ms) / (end_ms - start_ms) * 100)
            self.logger.info(f"Download progress: {progress_pct:.2f}%")
        
        # Remove duplicates if any (can happen with batch overlaps)
        if not all_data.empty:
            all_data = all_data.drop_duplicates(subset=['open_time'])
            all_data = all_data.sort_values('open_time')
            
            self.logger.info(f"Downloaded {len(all_data)} rows of {symbol} {interval} data")
            
            # Construct GCS path
            symbol_clean = symbol.replace('/', '_')
            if gcs_prefix:
                gcs_path = f"{gcs_prefix}/{symbol_clean}/{interval}/{start_date}_to_{end_date}.parquet"
            else:
                gcs_path = f"binance_futures/{symbol_clean}/{interval}/{start_date}_to_{end_date}.parquet"
            
            # Save to GCS
            return self._save_dataframe_to_gcs(all_data, gcs_bucket, gcs_path)
        else:
            self.logger.warning("No data was downloaded. Please check your parameters.")
            return None