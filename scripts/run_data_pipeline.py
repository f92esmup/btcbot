#!/usr/bin/env python3
# filepath: /workspaces/btcbot/scripts/run_data_pipeline.py
"""
Integrated Data Pipeline Runner

This script runs the integrated data pipeline that handles both data acquisition and preprocessing
in chunks to manage memory efficiently, and saves processed sequences to GCS.

Usage:
  python run_data_pipeline.py --start-date 2020-01-01 [--end-date 2023-12-31] [--force-reprocess]
  python run_data_pipeline.py --status  # Show pipeline status without processing
  python run_data_pipeline.py --list-chunks  # List all available chunks
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import IntegratedDataPipeline
from src.utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger("DataPipeline")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run BTCBot Integrated Data Pipeline")
    
    # Required start date (unless using --status or --list-chunks)
    parser.add_argument(
        "--start-date", 
        help="Start date in YYYY-MM-DD format"
    )
    
    # Optional end date (defaults to today)
    parser.add_argument(
        "--end-date", 
        help="End date in YYYY-MM-DD format (defaults to today)"
    )
    
    # Option to force reprocessing of all chunks
    parser.add_argument(
        "--force-reprocess", 
        action="store_true", 
        help="Reprocess all chunks even if they already exist"
    )
    
    # Option to only show pipeline status
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Show pipeline status without processing"
    )
    
    # Option to list all available chunks
    parser.add_argument(
        "--list-chunks", 
        action="store_true", 
        help="List all available chunks"
    )
    
    # Config path
    parser.add_argument(
        "--config", 
        help="Path to config file", 
        default="src/config.yaml"
    )
    
    return parser.parse_args()

async def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    # Check if GCS bucket is configured
    gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        logger.error("GCS_BUCKET_NAME environment variable is not set!")
        sys.exit(1)
    
    # Initialize the pipeline
    try:
        pipeline = IntegratedDataPipeline(config_path=args.config)
        
        # Handle list-chunks mode
        if args.list_chunks:
            chunks = pipeline.list_available_chunks()
            print(f"\nAvailable processed sequence chunks ({len(chunks)}):")
            for chunk in chunks:
                print(f"  {chunk}")
            return
        
        # Handle status-only mode
        if args.status:
            if not args.start_date:
                # Use default historical date from config for status check
                start_date = pipeline.config['data_acquisition_defaults']['historical_start_date']
                logger.info(f"No start date provided, using default from config: {start_date}")
            else:
                start_date = args.start_date
                
            end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
            
            status = pipeline.get_pipeline_status(start_date, end_date)
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
        
        # Regular pipeline run mode
        if not args.start_date:
            logger.error("--start-date is required for pipeline execution")
            sys.exit(1)
        
        # Run the pipeline
        logger.info(f"Starting integrated data pipeline from {args.start_date} to {args.end_date or 'today'}")
        
        results = await pipeline.run_pipeline(
            start_date=args.start_date,
            end_date=args.end_date,
            force_reprocess=args.force_reprocess
        )
        
        # Print summary of results
        print(f"\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Total chunks: {results['total_chunks']}")
        print(f"Existing chunks: {results['existing_chunks']}")
        print(f"Newly processed: {results['newly_processed_chunks']}")
        print(f"Total sequences: {results['total_sequences']}")
        print(f"\nDate range: {results['date_range']['start']} to {results['date_range']['end']}")
        
    except Exception as e:
        logger.error(f"Error running data pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
