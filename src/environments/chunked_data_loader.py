"""
Chunked Data Loader for Training Environment

This module implements a data loader for working with chunked sequence files, 
allowing efficient loading and concatenation of multiple data chunks.
"""

import os
import numpy as np
import io
import logging
from typing import List, Tuple, Dict, Any, Optional
from google.cloud import storage
from fnmatch import fnmatch

logger = logging.getLogger(__name__)

class ChunkedDataLoader:
    """
    Handles loading and concatenating multiple sequence data chunks from GCS or local storage.
    """
    
    def __init__(self, config_manager, sequence_length: int):
        """
        Initialize the chunked data loader.
        
        Args:
            config_manager: Configuration manager object
            sequence_length: Length of each sequence
        """
        self.config_manager = config_manager
        self.sequence_length = sequence_length
        self.config = config_manager.get_environment_config()
        
    def load_market_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load and concatenate market data from multiple sequence files.
        
        Returns:
            Tuple[np.ndarray, List[str]]: Processed market data and list of feature names
        """
        # Try to get GCS configuration
        try:
            gcs_bucket_name = os.environ.get('GCS_BUCKET_NAME')
            
            if gcs_bucket_name:
                return self._load_from_gcs(gcs_bucket_name)
            else:
                return self._load_from_local_storage()
                
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            # If loading fails, create dummy data for testing
            num_sequences = 1000
            num_timesteps = self.sequence_length
            num_features = 15  # Standard number of market features
            
            logger.warning(f"Creating {num_sequences} random test sequences")
            dummy_data = np.random.normal(0, 1, (num_sequences, num_timesteps, num_features))
            dummy_features = [f'feature_{i}' for i in range(num_features)]
            
            return dummy_data, dummy_features
    
    def _load_from_gcs(self, gcs_bucket_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load chunked sequence data from Google Cloud Storage.
        
        Args:
            gcs_bucket_name: Name of the GCS bucket
            
        Returns:
            Tuple[np.ndarray, List[str]]: Concatenated sequences and feature names
        """
        try:
            # New approach: Load from processed_sequences directory with chunked files
            processed_sequences_dir = self.config['processed_sequences_directory']
            processed_sequence_pattern = self.config['processed_sequence_file_pattern'].format(
                sequence_length=self.sequence_length
            )
            
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket_name)
            
            # List blobs with the given prefix
            blobs = list(bucket.list_blobs(prefix=processed_sequences_dir))
            
            # Find all matching sequence files
            matching_blobs = [
                blob for blob in blobs 
                if fnmatch(os.path.basename(blob.name), processed_sequence_pattern)
            ]
            
            if not matching_blobs:
                # Try fallback to legacy format
                logger.warning(f"No sequence files found in {processed_sequences_dir}, trying legacy format...")
                return self._load_legacy_format_from_gcs(gcs_bucket_name)
            
            # Sort blobs by name to maintain chronological order
            matching_blobs.sort(key=lambda b: b.name)
            
            logger.info(f"Found {len(matching_blobs)} sequence chunk files in GCS")
            
            # Load and concatenate all sequence files
            all_sequences = []
            feature_names = None
            close_prices_chunks = []
            atr_values_chunks = []
            
            for blob in matching_blobs:
                logger.info(f"Loading sequence chunk: {blob.name}")
                
                # Download the data to memory
                content = blob.download_as_bytes()
                with io.BytesIO(content) as f:
                    loaded_data = np.load(f)
                    
                    # Extract feature names from first chunk
                    if feature_names is None and 'feature_names' in loaded_data:
                        feature_names = loaded_data['feature_names'].tolist()
                    
                    # Extract sequences - check both naming conventions
                    if 'sequences' in loaded_data:
                        chunk_sequences = loaded_data['sequences']
                    elif 'market_sequences' in loaded_data:
                        chunk_sequences = loaded_data['market_sequences']
                    else:
                        logger.warning(f"No sequences found in {blob.name}, skipping")
                        continue
                        
                    all_sequences.append(chunk_sequences)
                    
                    # Extract optional auxiliary data if available
                    if 'close_prices' in loaded_data:
                        close_prices_chunks.append(loaded_data['close_prices'])
                    
                    if 'atr_values' in loaded_data:
                        atr_values_chunks.append(loaded_data['atr_values'])
            
            if not all_sequences:
                raise ValueError("No valid sequence data found in any chunk file")
            
            # Concatenate all sequences
            market_data = np.concatenate(all_sequences, axis=0)
            
            # Create default feature names if needed
            if feature_names is None:
                num_features = market_data.shape[2]
                feature_names = [f'feature_{i}' for i in range(num_features)]
            
            # Concatenate auxiliary data if available
            close_prices = np.concatenate(close_prices_chunks) if close_prices_chunks else None
            atr_values = np.concatenate(atr_values_chunks) if atr_values_chunks else None
            
            # Store auxiliary data as attributes that can be accessed by the environment
            self.close_prices = close_prices
            self.atr_values = atr_values
            
            logger.info(f"Loaded {len(market_data)} total sequences with {len(feature_names)} features")
            
            # Make sure data is float32
            if market_data.dtype != np.float32:
                market_data = market_data.astype(np.float32)
            
            return market_data, feature_names
            
        except Exception as e:
            logger.error(f"Error loading data from GCS: {e}")
            # Try fallback method
            return self._load_legacy_format_from_gcs(gcs_bucket_name)
    
    def _load_legacy_format_from_gcs(self, gcs_bucket_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load data using the legacy format as a fallback.
        
        Args:
            gcs_bucket_name: Name of the GCS bucket
            
        Returns:
            Tuple[np.ndarray, List[str]]: Market data and feature names
        """
        logger.info("Attempting to load using legacy data format from GCS")
        
        try:
            processed_data_dir = self.config['processed_data_directory']
            processed_file_identifier = self.config['processed_data_file_identifier']
            
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket_name)
            
            # List blobs with the given prefix
            blobs = list(bucket.list_blobs(prefix=processed_data_dir))
            matching_blobs = [blob for blob in blobs if processed_file_identifier in blob.name]
            
            if not matching_blobs:
                raise FileNotFoundError(
                    f"No files found with identifier {processed_file_identifier} in {processed_data_dir}"
                )
            
            # Use the most recent file if multiple matches
            target_blob = sorted(matching_blobs, key=lambda b: b.updated)[-1]
            logger.info(f"Loading market data from GCS: {target_blob.name}")
            
            # Download data to memory
            content = target_blob.download_as_bytes()
            with io.BytesIO(content) as f:
                loaded_data = np.load(f)
                
                # Determine which key contains the market data
                if 'market_sequences' in loaded_data:
                    market_data = loaded_data['market_sequences']
                elif 'X_market' in loaded_data:
                    market_data = loaded_data['X_market']
                else:
                    raise KeyError("No valid market data key found in file")
                
                # Extract feature names
                if 'feature_names' in loaded_data:
                    feature_names = loaded_data['feature_names'].tolist()
                else:
                    feature_names = [f'feature_{i}' for i in range(market_data.shape[2])]
                
                # Extract auxiliary data if available
                self.close_prices = loaded_data['close_prices'] if 'close_prices' in loaded_data else None
                self.atr_values = loaded_data['atr_values'] if 'atr_values' in loaded_data else None
                
                # Make sure data is float32
                if market_data.dtype != np.float32:
                    market_data = market_data.astype(np.float32)
                    
                logger.info(f"Loaded {len(market_data)} sequences with {len(feature_names)} features (legacy format)")
                return market_data, feature_names
                
        except Exception as e:
            logger.error(f"Error loading legacy format from GCS: {e}")
            raise
    
    def _load_from_local_storage(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load chunked sequence data from local storage.
        
        Returns:
            Tuple[np.ndarray, List[str]]: Concatenated sequences and feature names
        """
        logger.info("GCS not configured. Looking for local sequence files...")
        
        # Try to load from local processed_sequences directory
        local_sequences_dir = os.path.join('data', 'processed_sequences')
        
        if os.path.exists(local_sequences_dir):
            try:
                all_sequences = []
                feature_names = None
                close_prices_chunks = []
                atr_values_chunks = []
                
                # Find all sequence files for the current sequence length
                sequence_files = [
                    f for f in os.listdir(local_sequences_dir) 
                    if f.endswith('.npz') and f"L{self.sequence_length}_sequences" in f
                ]
                
                if not sequence_files:
                    logger.warning(f"No sequence files found in {local_sequences_dir}")
                    # Try legacy format
                    return self._load_legacy_format_local()
                
                # Sort files to maintain chronological order
                sequence_files.sort()
                
                # Load and concatenate each file
                for filename in sequence_files:
                    file_path = os.path.join(local_sequences_dir, filename)
                    logger.info(f"Loading local sequence file: {file_path}")
                    
                    loaded_data = np.load(file_path)
                    
                    # Extract feature names from first file
                    if feature_names is None and 'feature_names' in loaded_data:
                        feature_names = loaded_data['feature_names'].tolist()
                    
                    # Extract sequences - check both naming conventions
                    if 'sequences' in loaded_data:
                        chunk_sequences = loaded_data['sequences']
                    elif 'market_sequences' in loaded_data:
                        chunk_sequences = loaded_data['market_sequences']
                    else:
                        logger.warning(f"No sequences found in {filename}, skipping")
                        continue
                        
                    all_sequences.append(chunk_sequences)
                    
                    # Extract auxiliary data if available
                    if 'close_prices' in loaded_data:
                        close_prices_chunks.append(loaded_data['close_prices'])
                    
                    if 'atr_values' in loaded_data:
                        atr_values_chunks.append(loaded_data['atr_values'])
                
                if not all_sequences:
                    raise ValueError("No valid sequence data found in any local file")
                
                # Concatenate sequences
                market_data = np.concatenate(all_sequences, axis=0)
                
                # Create default feature names if needed
                if feature_names is None:
                    num_features = market_data.shape[2]
                    feature_names = [f'feature_{i}' for i in range(num_features)]
                
                # Concatenate auxiliary data if available
                if close_prices_chunks:
                    self.close_prices = np.concatenate(close_prices_chunks)
                else:
                    self.close_prices = None
                    
                if atr_values_chunks:
                    self.atr_values = np.concatenate(atr_values_chunks)
                else:
                    self.atr_values = None
                
                # Make sure data is float32
                if market_data.dtype != np.float32:
                    market_data = market_data.astype(np.float32)
                
                logger.info(f"Loaded {len(market_data)} total sequences with {len(feature_names)} features")
                return market_data, feature_names
                
            except Exception as e:
                logger.error(f"Error loading from local sequences: {e}")
                # Try legacy format
                return self._load_legacy_format_local()
        else:
            logger.warning(f"Processed sequences directory not found at {local_sequences_dir}")
            # Try legacy format
            return self._load_legacy_format_local()
    
    def _load_legacy_format_local(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load data using the legacy format locally.
        
        Returns:
            Tuple[np.ndarray, List[str]]: Market data and feature names
        """
        logger.info("Attempting to load using legacy data format locally")
        
        data_dir = os.path.join('data', 'processed')
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"No data directory found at {data_dir}")
        
        # Look for legacy format files
        for filename in os.listdir(data_dir):
            if filename.endswith('market_features.npz'):
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Loading legacy data format from: {file_path}")
                
                loaded_data = np.load(file_path)
                
                # Extract market data
                if 'market_sequences' in loaded_data:
                    market_data = loaded_data['market_sequences']
                elif 'X_market' in loaded_data:
                    market_data = loaded_data['X_market']
                else:
                    continue  # Try next file
                
                # Extract feature names
                if 'feature_names' in loaded_data:
                    feature_names = loaded_data['feature_names'].tolist()
                else:
                    feature_names = [f'feature_{i}' for i in range(market_data.shape[2])]
                
                # Extract auxiliary data if available
                self.close_prices = loaded_data['close_prices'] if 'close_prices' in loaded_data else None
                self.atr_values = loaded_data['atr_values'] if 'atr_values' in loaded_data else None
                
                # Make sure data is float32
                if market_data.dtype != np.float32:
                    market_data = market_data.astype(np.float32)
                    
                logger.info(f"Loaded {len(market_data)} sequences with {len(feature_names)} features (legacy format)")
                return market_data, feature_names
        
        raise FileNotFoundError("No market data files found locally in any format")
