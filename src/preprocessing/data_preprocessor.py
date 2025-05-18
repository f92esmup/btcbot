"""
Data preprocessing module for Bitcoin trading bot.

This module contains the DataPreprocessor class, which is responsible for cleaning,
normalizing, and preparing market data sequences for the reinforcement learning agent.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

from google.cloud import storage
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DataPreprocessor:
    """
    A class for preprocessing market data for the reinforcement learning agent.
    
    This class is responsible for loading raw OHLCV data, cleaning it, calculating
    features using FeatureEngineer, normalizing the features causally, and creating
    sequence arrays suitable for the Transformer-based RL agent.
    
    Attributes:
        logger (logging.Logger): Class logger.
        project_id (str): Google Cloud project ID.
        feature_engineer (FeatureEngineer): Instance of FeatureEngineer for feature calculation.
        sequence_length (int): Length of sequences to create.
        ffill_limit_for_nans (int): Maximum number of consecutive NaN values to forward fill.
        feature_normalization_lookback (int): Lookback period for feature normalization.
        normalization_methods (Dict): Dictionary mapping features to their normalization methods.
    """
    
    def __init__(
        self,
        project_id: str,
        feature_params: Optional[Dict] = None,
        sequence_length: int = 60,
        ffill_limit_for_nans: int = 5,
        feature_normalization_lookback: int = 100,
        normalization_methods: Optional[Dict] = None
    ):
        """
        Initialize the DataPreprocessor with preprocessing parameters.
        
        Args:
            project_id (str): Google Cloud project ID.
            feature_params (Dict, optional): Parameters for feature calculation.
            sequence_length (int, optional): Length of sequences to create. Defaults to 60.
            ffill_limit_for_nans (int, optional): Maximum number of consecutive NaN values to forward fill.
                Defaults to 5.
            feature_normalization_lookback (int, optional): Lookback period for feature normalization.
                Defaults to 100.
            normalization_methods (Dict, optional): Dictionary mapping features to their normalization methods.
                If not provided, default methods will be used.
        """
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.feature_engineer = FeatureEngineer(feature_params)
        self.sequence_length = sequence_length
        self.ffill_limit_for_nans = ffill_limit_for_nans
        self.feature_normalization_lookback = feature_normalization_lookback
        
        # Default normalization methods for each feature
        self.default_normalization_methods = {
            # OHLCV
            'open': 'log_diff',
            'high': 'log_diff',
            'low': 'log_diff',
            'close': 'log_diff',
            'volume': 'log_diff',
            
            # Technical indicators
            'log_return': 'zscore',
            'hl_range': 'divide_by_close',
            'body_size_rel': 'identity',  # Already normalized
            'atr': 'divide_by_close',
            'rsi': 'identity_center',  # RSI is already in [0, 100], center to [-50, 50]
            'macd': 'divide_by_atr',
            'macd_histogram': 'divide_by_atr',
            'bb_width': 'zscore',
            'sma_cross': 'divide_by_atr',
            'stoch_k': 'identity_center',  # Stochastic is already in [0, 100], center to [-50, 50]
            'adx': 'identity_scale',  # ADX is in [0, 100], scale to [0, 1]
            'volume_ratio': 'zscore',
            'mfi': 'identity_center',  # MFI is already in [0, 100], center to [-50, 50]
            'obv': 'log_diff',
            'tenkan_sen': 'log_diff'
        }
        
        # Update with provided normalization methods
        self.normalization_methods = self.default_normalization_methods.copy()
        if normalization_methods:
            self.normalization_methods.update(normalization_methods)
        
        self.logger.info("DataPreprocessor initialized with parameters.")
        self.logger.info(f"Sequence length: {self.sequence_length}")
        self.logger.info(f"Feature normalization lookback: {self.feature_normalization_lookback}")
    
    def load_data_from_gcs(self, gcs_uri: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from a Parquet file in Google Cloud Storage.
        
        Args:
            gcs_uri (str): GCS URI of the Parquet file to load.
            
        Returns:
            pd.DataFrame: DataFrame with the loaded data.
            
        Raises:
            Exception: If there's an error loading the data.
        """
        try:
            self.logger.info(f"Loading data from GCS: {gcs_uri}")
            
            # Parse bucket and blob path from GCS URI
            if gcs_uri.startswith('gs://'):
                gcs_uri = gcs_uri[5:]  # Remove 'gs://' prefix
            
            bucket_name, blob_path = gcs_uri.split('/', 1)
            
            # Initialize GCS client and download the file
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download to a temporary file
            temp_file = '/tmp/temp_data.parquet'
            blob.download_to_filename(temp_file)
            
            # Read the Parquet file
            df = pd.read_parquet(temp_file)
            
            # Clean up
            os.remove(temp_file)
            
            # Set open_time as index if it exists
            if 'open_time' in df.columns:
                df.set_index('open_time', inplace=True)
            
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            self.logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from GCS: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLCV data, handling NaN values and ensuring data quality.
        
        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        self.logger.info("Cleaning data...")
        
        # Make a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Check for NaN values
        nan_counts = clean_df.isna().sum()
        if nan_counts.sum() > 0:
            self.logger.warning(f"Found NaN values in the data: {nan_counts}")
            
            # Forward fill NaN values up to the limit
            clean_df.fillna(method='ffill', limit=self.ffill_limit_for_nans, inplace=True)
            
            # Check if there are still NaN values
            remaining_nans = clean_df.isna().sum()
            if remaining_nans.sum() > 0:
                self.logger.warning(f"Remaining NaN values after forward fill: {remaining_nans}")
                
                # Drop rows with NaN values
                clean_df.dropna(inplace=True)
                self.logger.info(f"Dropped {len(df) - len(clean_df)} rows with NaN values.")
        
        # Check for duplicate indices
        if clean_df.index.duplicated().any():
            self.logger.warning("Found duplicate indices in the data.")
            clean_df = clean_df[~clean_df.index.duplicated(keep='first')]
            self.logger.info(f"Removed duplicate indices. New shape: {clean_df.shape}")
        
        # Ensure the data is sorted by index (datetime)
        clean_df.sort_index(inplace=True)
        
        # Check for gaps in the time series
        if len(clean_df) > 1:
            time_diffs = clean_df.index.to_series().diff().dropna()
            unique_diffs = time_diffs.unique()
            if len(unique_diffs) > 1:
                self.logger.warning(f"Found irregular time intervals: {unique_diffs}")
                # We don't interpolate or fill gaps, just log the warning
        
        self.logger.info(f"Data cleaning completed. Final shape: {clean_df.shape}")
        return clean_df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal normalization to features.
        
        This method applies different normalization techniques to each feature
        based on the specified normalization_methods. All normalizations are
        causal, meaning they only use information available up to the current point.
        
        Args:
            df (pd.DataFrame): DataFrame with features calculated by FeatureEngineer.
            
        Returns:
            pd.DataFrame: DataFrame with normalized features.
        """
        self.logger.info("Normalizing features...")
        
        # Create a new DataFrame for normalized features
        norm_df = pd.DataFrame(index=df.index)
        
        # Ensure OHLCV and ATR columns exist for normalization dependencies
        required_cols = ['close', 'atr', 'open']
        missing_deps = [col for col in required_cols if col not in df.columns]
        if missing_deps:
            self.logger.error(f"Missing required columns for normalization: {missing_deps}")
            raise ValueError(f"Required columns missing for normalization: {missing_deps}")
        
        # Process each feature with its specified normalization method
        for feature, norm_method in self.normalization_methods.items():
            try:
                if feature not in df.columns:
                    self.logger.warning(f"Feature {feature} not found in input DataFrame. Skipping.")
                    continue
                
                # Apply the specified normalization method
                if norm_method == 'log_diff':
                    # Log difference from previous value: log(x_t / x_{t-1})
                    # For market prices: log(C/C_prev), log(H/O), log(L/O), log(C/O)
                    if feature == 'open':
                        # log(O/O_prev)
                        norm_df[feature] = np.log(df[feature] / df[feature].shift(1))
                    elif feature == 'high':
                        # log(H/O) - log ratio of high to open
                        norm_df[feature] = np.log(df[feature] / df['open'])
                    elif feature == 'low':
                        # log(L/O) - log ratio of low to open
                        norm_df[feature] = np.log(df[feature] / df['open'])
                    elif feature == 'close':
                        # log(C/O) - log ratio of close to open (within candle)
                        norm_df[feature] = np.log(df[feature] / df['open'])
                    elif feature == 'volume':
                        # log(Vol/SMA(Vol,N)) - volume relative to its moving average
                        vol_sma = df[feature].rolling(window=20).mean().shift(1)
                        norm_df[feature] = np.log(df[feature] / vol_sma)
                    else:
                        # Standard log diff for other features
                        norm_df[feature] = np.log(df[feature] / df[feature].shift(1))
                
                elif norm_method == 'zscore':
                    # Z-score using rolling window: (x_t - mean_{t-1}) / std_{t-1}
                    # This is causal - only using past data for mean and std
                    mean = df[feature].rolling(window=self.feature_normalization_lookback).mean().shift(1)
                    std = df[feature].rolling(window=self.feature_normalization_lookback).std().shift(1)
                    # Handle zero std with a small epsilon
                    std = std.replace(0, 1e-8)
                    norm_df[feature] = (df[feature] - mean) / std
                
                elif norm_method == 'divide_by_close':
                    # Divide by close price: x_t / close_t
                    norm_df[feature] = df[feature] / df['close']
                
                elif norm_method == 'divide_by_atr':
                    # Normalize by ATR: (x_t - close_t) / atr_t
                    # This works better for oscillators and indicators that should be centered
                    norm_df[feature] = (df[feature] - df['close']) / df['atr']
                
                elif norm_method == 'pct_change':
                    # Percentage change: (x_t / x_{t-1}) - 1
                    norm_df[feature] = df[feature].pct_change()
                
                elif norm_method == 'identity':
                    # No normalization: x_t
                    norm_df[feature] = df[feature]
                
                elif norm_method == 'identity_center':
                    # Center to [-50, 50]: x_t - 50
                    # Perfect for indicators already in [0, 100] range like RSI, MFI, Stochastic
                    norm_df[feature] = df[feature] - 50
                
                elif norm_method == 'identity_scale':
                    # Scale to [0, 1]: x_t / 100
                    # For indicators in [0, 100] when we want [0, 1] range
                    norm_df[feature] = df[feature] / 100
                
                else:
                    self.logger.warning(f"Unknown normalization method: {norm_method} for feature: {feature}. Using identity.")
                    norm_df[feature] = df[feature]
                
            except Exception as e:
                self.logger.error(f"Error normalizing feature {feature} with method {norm_method}: {str(e)}")
                # Use identity normalization as fallback
                norm_df[feature] = df[feature]
        
        # Handle NaN values created by normalization
        nan_counts = norm_df.isna().sum()
        if nan_counts.sum() > 0:
            self.logger.warning(f"Normalization created NaN values: {nan_counts}")
            
            # Forward fill NaN values up to the limit
            norm_df.fillna(method='ffill', limit=self.ffill_limit_for_nans, inplace=True)
            
            # Fill remaining NaNs with 0
            norm_df.fillna(0, inplace=True)
        
        self.logger.info(f"Feature normalization completed. Final shape: {norm_df.shape}")
        return norm_df
    
    def create_sequences(
        self,
        normalized_df: pd.DataFrame,
        drop_incomplete: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for the RL agent from normalized features.
        
        Args:
            normalized_df (pd.DataFrame): DataFrame with normalized features.
            drop_incomplete (bool, optional): Whether to drop incomplete sequences
                at the beginning. Defaults to True.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X_market: Array of market feature sequences with shape (N_samples, sequence_length, N_features)
                - timestamps: Array of timestamps for each sequence
        """
        self.logger.info(f"Creating sequences with length {self.sequence_length}...")
        
        # Convert DataFrame to numpy array
        data = normalized_df.values
        timestamps = normalized_df.index.values
        
        # Get dimensions
        total_steps = len(data)
        n_features = data.shape[1]
        
        if total_steps <= self.sequence_length:
            self.logger.warning(f"Not enough data points ({total_steps}) to create sequences of length {self.sequence_length}.")
            if drop_incomplete:
                self.logger.error("No complete sequences available.")
                return np.array([]), np.array([])
        
        # Determine start index
        start_idx = self.sequence_length if drop_incomplete else 1
        
        # Calculate number of sequences
        n_sequences = total_steps - start_idx + 1
        
        self.logger.info(f"Creating {n_sequences} sequences from {total_steps} data points.")
        
        # Initialize arrays for sequences and timestamps
        X_market = np.zeros((n_sequences, self.sequence_length, n_features))
        sequence_timestamps = np.zeros(n_sequences, dtype='datetime64[ns]')
        
        # Create sequences
        for i in range(n_sequences):
            idx = i + start_idx - 1
            X_market[i] = data[idx - self.sequence_length + 1 : idx + 1]
            sequence_timestamps[i] = timestamps[idx]  # Use the timestamp of the last element in the sequence
        
        self.logger.info(f"Created sequences with shape: {X_market.shape}")
        
        # Check for NaN values in sequences
        if np.isnan(X_market).any():
            nan_count = np.isnan(X_market).sum()
            self.logger.warning(f"Found {nan_count} NaN values in sequences. Filling with 0.")
            X_market = np.nan_to_num(X_market)
        
        return X_market, sequence_timestamps
    
    def save_sequences_to_gcs(
        self,
        X_market: np.ndarray,
        timestamps: np.ndarray,
        feature_names: List[str],
        gcs_bucket: str,
        gcs_prefix: str = None
    ) -> str:
        """
        Save sequence arrays to Google Cloud Storage as a .npz file.
        
        Args:
            X_market (np.ndarray): Array of market feature sequences.
            timestamps (np.ndarray): Array of timestamps for each sequence.
            feature_names (List[str]): List of feature names.
            gcs_bucket (str): GCS bucket name.
            gcs_prefix (str, optional): Prefix for the GCS path. Defaults to None.
            
        Returns:
            str: GCS URI of the saved file.
            
        Raises:
            Exception: If there's an error saving the data.
        """
        try:
            self.logger.info(f"Saving sequences to GCS bucket: {gcs_bucket}")
            
            # Create a temporary file
            temp_file = '/tmp/sequences.npz'
            
            # Save arrays to the temporary file
            np.savez_compressed(
                temp_file,
                X_market=X_market,
                timestamps=timestamps,
                feature_names=feature_names
            )
            
            # Upload to GCS
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(gcs_bucket)
            
            # Construct GCS path
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            if gcs_prefix:
                gcs_path = f"{gcs_prefix}/sequences_{current_time}.npz"
            else:
                gcs_path = f"preprocessed/sequences_{current_time}.npz"
            
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(temp_file)
            
            # Clean up
            os.remove(temp_file)
            
            gcs_uri = f"gs://{gcs_bucket}/{gcs_path}"
            self.logger.info(f"Successfully saved sequences to {gcs_uri}")
            
            # Also save metadata about the sequences
            metadata = {
                'n_sequences': X_market.shape[0],
                'sequence_length': X_market.shape[1],
                'n_features': X_market.shape[2],
                'feature_names': feature_names,
                'timestamp_range': [str(timestamps[0]), str(timestamps[-1])],
                'creation_time': datetime.now().isoformat()
            }
            
            metadata_path = f"{gcs_path}.metadata.json"
            metadata_blob = bucket.blob(metadata_path)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )
            
            return gcs_uri
            
        except Exception as e:
            self.logger.error(f"Error saving sequences to GCS: {str(e)}")
            raise
    
    def process_data_pipeline(
        self,
        input_gcs_uri: str,
        output_gcs_bucket: str,
        output_gcs_prefix: str = None
    ) -> str:
        """
        Execute the full data preprocessing pipeline.
        
        This method orchestrates the entire preprocessing workflow:
        1. Load raw data from GCS
        2. Clean the data
        3. Calculate features using FeatureEngineer
        4. Normalize features
        5. Create sequences
        6. Save sequences to GCS
        
        Args:
            input_gcs_uri (str): GCS URI of the raw data file.
            output_gcs_bucket (str): GCS bucket for output.
            output_gcs_prefix (str, optional): Prefix for output GCS path. Defaults to None.
            
        Returns:
            str: GCS URI of the saved sequences file.
        """
        self.logger.info(f"Starting data preprocessing pipeline.")
        self.logger.info(f"Input GCS URI: {input_gcs_uri}")
        self.logger.info(f"Output GCS bucket: {output_gcs_bucket}")
        
        # 1. Load data from GCS
        raw_df = self.load_data_from_gcs(input_gcs_uri)
        
        # 2. Clean the data
        cleaned_df = self.clean_data(raw_df)
        
        # 3. Calculate features
        self.logger.info("Calculating features...")
        featured_df = self.feature_engineer.engineer_features(cleaned_df, include_all=False)
        
        # 4. Normalize features
        normalized_df = self.normalize_features(featured_df)
        
        # 5. Create sequences
        X_market, timestamps = self.create_sequences(normalized_df)
        
        if len(X_market) == 0:
            self.logger.error("Failed to create any valid sequences.")
            raise ValueError("No valid sequences were created.")
        
        # 6. Save sequences to GCS
        feature_names = featured_df.columns.tolist()
        output_uri = self.save_sequences_to_gcs(
            X_market,
            timestamps,
            feature_names,
            output_gcs_bucket,
            output_gcs_prefix
        )
        
        self.logger.info(f"Data preprocessing pipeline completed successfully.")
        self.logger.info(f"Output sequences saved to: {output_uri}")
        
        return output_uri
