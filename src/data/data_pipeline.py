"""
Integrated Data Pipeline for BTCBot

This module consolidates data acquisition and preprocessing into a single, efficient pipeline
that operates in chunks to manage memory with large datasets (5 years of 1-minute candles).

Key Features:
- Chunk-based processing (quarterly/semi-annually) to manage memory
- Smart checkpointing/resuming (only processes missing chunks)
- Saves only final processed sequences to GCS under processed_sequences/
- Replaces separate download_data.py and preprocess_data.py scripts
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import asyncio
from pathlib import Path
import os

from src.data.binance_futures_downloader import BinanceFuturesDownloader
from src.data.preprocessor import Preprocessor
from src.utils.gcs_utils import upload_to_gcs, download_from_gcs, file_exists_in_gcs, list_files_in_gcs
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


class IntegratedDataPipeline:
    """
    Integrated data acquisition and preprocessing pipeline that operates in chunks
    to efficiently handle large datasets while providing smart resuming capabilities.
    """
    
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the integrated data pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_manager = ConfigManager(config_path=config_path)
        self.config = self.config_manager.get_full_config()
        self.downloader = BinanceFuturesDownloader(self.config_manager)
        self.preprocessor = Preprocessor(self.config_manager)
        
        # Pipeline configuration
        self.bucket_name = os.getenv('GCS_BUCKET_NAME')
        self.processed_sequences_path = "data/processed_sequences"
        
        # Chunk configuration - quarterly chunks for efficient memory management
        self.chunk_duration_months = 3  # Quarterly chunks
        
        # Data parameters from config
        data_acq_config = self.config_manager.get_data_acquisition_defaults()
        preprocessing_config = self.config_manager.get_preprocessing_config()
        
        self.symbol = data_acq_config['symbol']
        self.interval = data_acq_config['interval']
        self.sequence_length = preprocessing_config['sequence_length_L']
        
        logger.info(f"Initialized IntegratedDataPipeline for {self.symbol} {self.interval}")
        logger.info(f"Chunk duration: {self.chunk_duration_months} months")
        logger.info(f"Sequence length: {self.sequence_length}")
    
    def generate_chunk_periods(
        self, 
        start_date: str, 
        end_date: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate quarterly chunk periods between start and end dates.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of (chunk_start, chunk_end) datetime tuples
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            # Calculate chunk end (3 months later)
            if current_start.month <= 9:
                chunk_end = current_start.replace(
                    month=current_start.month + self.chunk_duration_months
                )
            else:
                # Handle year rollover
                chunk_end = current_start.replace(
                    year=current_start.year + 1,
                    month=current_start.month + self.chunk_duration_months - 12
                )
            
            # Don't exceed the overall end date
            chunk_end = min(chunk_end, end_dt)
            
            chunks.append((current_start, chunk_end))
            current_start = chunk_end
        
        return chunks
    
    def get_chunk_filename(self, start_date: datetime, end_date: datetime) -> str:
        """
        Generate standardized filename for a data chunk.
        
        Args:
            start_date: Chunk start datetime
            end_date: Chunk end datetime
            
        Returns:
            Standardized chunk filename
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{self.symbol}_{self.interval}_L{self.sequence_length}_sequences_{start_str}_{end_str}.npz"
    
    def check_existing_chunks(
        self, 
        chunk_periods: List[Tuple[datetime, datetime]]
    ) -> Tuple[List[Tuple[datetime, datetime]], List[str]]:
        """
        Check which chunks already exist in GCS and which need to be processed.
        
        Args:
            chunk_periods: List of (start_date, end_date) tuples
            
        Returns:
            Tuple of (missing_chunks, existing_chunk_files)
        """
        missing_chunks = []
        existing_files = []
        
        for start_date, end_date in chunk_periods:
            chunk_filename = self.get_chunk_filename(start_date, end_date)
            chunk_gcs_path = f"{self.processed_sequences_path}/{chunk_filename}"
            
            if file_exists_in_gcs(self.bucket_name, chunk_gcs_path):
                existing_files.append(chunk_gcs_path)
                logger.info(f"Found existing chunk: {chunk_filename}")
            else:
                missing_chunks.append((start_date, end_date))
                logger.info(f"Missing chunk: {chunk_filename}")
        
        return missing_chunks, existing_files
    
    async def download_chunk_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Download raw data for a specific chunk period.
        
        Args:
            start_date: Chunk start datetime
            end_date: Chunk end datetime
            
        Returns:
            Raw OHLCV DataFrame for the chunk
        """
        logger.info(f"Downloading data chunk: {start_date.date()} to {end_date.date()}")
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Download data for this chunk
        raw_data = await self.downloader.download_historical_data(
            symbol=self.symbol,
            interval=self.interval,
            start_date=start_str,
            end_date=end_str
        )
        
        logger.info(f"Downloaded {len(raw_data)} candles for chunk {start_date.date()} to {end_date.date()}")
        return raw_data
    
    def preprocess_chunk_data(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess raw data chunk into sequences.
        
        Args:
            raw_data: Raw OHLCV DataFrame
            
        Returns:
            Processed sequences as numpy array
        """
        logger.info(f"Preprocessing chunk with {len(raw_data)} candles")
        
        # Add required buffer for indicators and normalization windows
        normalization_window = (
            self.sequence_length * 
            self.config['preprocessing']['normalization_window_multiplier_for_L']
        )
        
        # Get the longest indicator period from config
        indicators_config = self.config['preprocessing']['indicators']
        max_indicator_period = max([
            indicators_config.get('sma_long_period', 50),
            indicators_config.get('ema_long_period', 26),
            indicators_config.get('bollinger_period', 20),
            indicators_config.get('cci_period', 20),
            indicators_config.get('stochastic_k_period', 14)
        ])
        
        required_buffer = max(normalization_window, max_indicator_period)
        
        if len(raw_data) < required_buffer + self.sequence_length:
            logger.warning(
                f"Insufficient data for processing. "
                f"Need at least {required_buffer + self.sequence_length} candles, "
                f"got {len(raw_data)}"
            )
            return np.array([])
        
        # Process the data using the existing preprocessor
        processed_sequences = self.preprocessor.process_market_data(raw_data)
        
        logger.info(f"Generated {len(processed_sequences)} sequences from chunk")
        return processed_sequences
    
    def save_chunk_sequences(
        self, 
        sequences: np.ndarray, 
        start_date: datetime, 
        end_date: datetime
    ) -> str:
        """
        Save processed sequences for a chunk to GCS.
        
        Args:
            sequences: Processed sequences array
            start_date: Chunk start datetime
            end_date: Chunk end datetime
            
        Returns:
            GCS path where sequences were saved
        """
        chunk_filename = self.get_chunk_filename(start_date, end_date)
        local_temp_path = f"/tmp/{chunk_filename}"
        gcs_path = f"{self.processed_sequences_path}/{chunk_filename}"
        
        # Save sequences locally first
        np.savez_compressed(local_temp_path, sequences=sequences)
        
        # Upload to GCS
        upload_to_gcs(
            bucket_name=self.bucket_name,
            source_file_name=local_temp_path,
            destination_blob_name=gcs_path
        )
        
        # Clean up local temp file
        os.remove(local_temp_path)
        
        logger.info(f"Saved {len(sequences)} sequences to {gcs_path}")
        return gcs_path
    
    async def process_missing_chunks(
        self, 
        missing_chunks: List[Tuple[datetime, datetime]]
    ) -> List[str]:
        """
        Process all missing chunks and save to GCS.
        
        Args:
            missing_chunks: List of (start_date, end_date) tuples to process
            
        Returns:
            List of GCS paths for newly created chunk files
        """
        processed_files = []
        
        for i, (start_date, end_date) in enumerate(missing_chunks):
            logger.info(
                f"Processing chunk {i+1}/{len(missing_chunks)}: "
                f"{start_date.date()} to {end_date.date()}"
            )
            
            try:
                # Download raw data for chunk
                raw_data = await self.download_chunk_data(start_date, end_date)
                
                if raw_data.empty:
                    logger.warning(f"No data downloaded for chunk {start_date.date()} to {end_date.date()}")
                    continue
                
                # Preprocess into sequences
                sequences = self.preprocess_chunk_data(raw_data)
                
                if len(sequences) == 0:
                    logger.warning(f"No sequences generated for chunk {start_date.date()} to {end_date.date()}")
                    continue
                
                # Save to GCS
                gcs_path = self.save_chunk_sequences(sequences, start_date, end_date)
                processed_files.append(gcs_path)
                
                # Small delay between chunks to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing chunk {start_date.date()} to {end_date.date()}: {e}")
                raise
        
        return processed_files
    
    async def run_pipeline(
        self, 
        start_date: str, 
        end_date: str = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete integrated data pipeline.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            force_reprocess: If True, reprocess all chunks even if they exist
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting integrated data pipeline for {self.symbol}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Force reprocess: {force_reprocess}")
        
        # Generate chunk periods
        chunk_periods = self.generate_chunk_periods(start_date, end_date)
        logger.info(f"Generated {len(chunk_periods)} chunks to process")
        
        # Check existing chunks (unless force reprocessing)
        if force_reprocess:
            missing_chunks = chunk_periods
            existing_files = []
        else:
            missing_chunks, existing_files = self.check_existing_chunks(chunk_periods)
        
        logger.info(f"Found {len(existing_files)} existing chunks")
        logger.info(f"Need to process {len(missing_chunks)} missing chunks")
        
        # Process missing chunks
        newly_processed = []
        if missing_chunks:
            logger.info("Starting processing of missing chunks...")
            newly_processed = await self.process_missing_chunks(missing_chunks)
        else:
            logger.info("No missing chunks to process")
        
        # Get final list of all available chunk files
        all_chunk_files = existing_files + newly_processed
        
        # Gather statistics
        total_sequences = 0
        for chunk_file in all_chunk_files:
            try:
                # Download and check sequence count
                local_temp = f"/tmp/temp_check_{Path(chunk_file).name}"
                download_from_gcs(self.bucket_name, chunk_file, local_temp)
                data = np.load(local_temp)
                total_sequences += len(data['sequences'])
                os.remove(local_temp)
            except Exception as e:
                logger.warning(f"Could not check sequences in {chunk_file}: {e}")
        
        pipeline_results = {
            'status': 'completed',
            'total_chunks': len(chunk_periods),
            'existing_chunks': len(existing_files),
            'newly_processed_chunks': len(newly_processed),
            'total_sequence_files': len(all_chunk_files),
            'total_sequences': total_sequences,
            'chunk_files': all_chunk_files,
            'newly_processed_files': newly_processed,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Total chunks: {len(chunk_periods)}")
        logger.info(f"Existing chunks: {len(existing_files)}")
        logger.info(f"Newly processed: {len(newly_processed)}")
        logger.info(f"Total sequences: {total_sequences}")
        
        return pipeline_results
    
    def list_available_chunks(self) -> List[str]:
        """
        List all available processed sequence chunks in GCS.
        
        Returns:
            List of GCS paths for available chunk files
        """
        try:
            files = list_files_in_gcs(self.bucket_name, self.processed_sequences_path)
            chunk_files = [f for f in files if f.endswith('.npz')]
            logger.info(f"Found {len(chunk_files)} available chunk files")
            return chunk_files
        except Exception as e:
            logger.error(f"Error listing available chunks: {e}")
            return []
    
    def get_pipeline_status(self, start_date: str, end_date: str = None) -> Dict[str, Any]:
        """
        Get status of pipeline for a given date range without running it.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Dictionary with pipeline status information
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        chunk_periods = self.generate_chunk_periods(start_date, end_date)
        missing_chunks, existing_files = self.check_existing_chunks(chunk_periods)
        
        return {
            'date_range': {'start': start_date, 'end': end_date},
            'total_chunks_needed': len(chunk_periods),
            'existing_chunks': len(existing_files),
            'missing_chunks': len(missing_chunks),
            'completion_percentage': (len(existing_files) / len(chunk_periods)) * 100,
            'existing_files': existing_files,
            'missing_chunk_periods': [
                {
                    'start': start.strftime('%Y-%m-%d'),
                    'end': end.strftime('%Y-%m-%d')
                }
                for start, end in missing_chunks
            ]
        }


# CLI interface for running the pipeline
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Run BTCBot Integrated Data Pipeline")
    parser.add_argument(
        "--start-date", 
        required=True, 
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", 
        help="End date in YYYY-MM-DD format (defaults to today)"
    )
    parser.add_argument(
        "--force-reprocess", 
        action="store_true", 
        help="Reprocess all chunks even if they exist"
    )
    parser.add_argument(
        "--status-only", 
        action="store_true", 
        help="Only show pipeline status without processing"
    )
    parser.add_argument(
        "--list-chunks", 
        action="store_true", 
        help="List all available chunk files"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        pipeline = IntegratedDataPipeline()
        
        if args.list_chunks:
            chunks = pipeline.list_available_chunks()
            print(f"\nAvailable chunk files ({len(chunks)}):")
            for chunk in chunks:
                print(f"  {chunk}")
            return
        
        if args.status_only:
            status = pipeline.get_pipeline_status(args.start_date, args.end_date)
            print(f"\nPipeline Status:")
            print(f"Date range: {status['date_range']['start']} to {status['date_range']['end']}")
            print(f"Total chunks needed: {status['total_chunks_needed']}")
            print(f"Existing chunks: {status['existing_chunks']}")
            print(f"Missing chunks: {status['missing_chunks']}")
            print(f"Completion: {status['completion_percentage']:.1f}%")
            
            if status['missing_chunks'] > 0:
                print(f"\nMissing chunk periods:")
                for period in status['missing_chunk_periods']:
                    print(f"  {period['start']} to {period['end']}")
            return
        
        # Run the full pipeline
        results = await pipeline.run_pipeline(
            start_date=args.start_date,
            end_date=args.end_date,
            force_reprocess=args.force_reprocess
        )
        
        print(f"\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Total chunks: {results['total_chunks']}")
        print(f"Existing chunks: {results['existing_chunks']}")
        print(f"Newly processed: {results['newly_processed_chunks']}")
        print(f"Total sequences: {results['total_sequences']}")
    
    asyncio.run(main())
