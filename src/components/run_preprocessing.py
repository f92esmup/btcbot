"""
Preprocessing component for the BTC Trading Bot pipeline.

This script serves as the entry point for the Vertex AI Pipeline component
responsible for preprocessing raw OHLCV data, calculating technical indicators,
normalizing features, and creating sequences for the RL agent.
"""

import os
import argparse
import logging
import json
import yaml
from typing import Dict, Any

from google.cloud import storage
import kfp
from kfp.v2.dsl import Output, Dataset

from src.preprocessing.data_preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the preprocessing component.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Preprocess data and create sequences for RL agent')
    
    # Required arguments
    parser.add_argument('--project_id', type=str, required=True,
                        help='Google Cloud project ID')
    parser.add_argument('--input_data_uri', type=str, required=True,
                        help='GCS URI of the input Parquet file with OHLCV data')
    parser.add_argument('--output_gcs_bucket', type=str, required=True,
                        help='GCS bucket for output sequences')
    
    # Optional arguments
    parser.add_argument('--output_gcs_prefix', type=str, default=None,
                        help='Prefix for output GCS path')
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Length of sequences to create (default: 60)')
    parser.add_argument('--ffill_limit_for_nans', type=int, default=5,
                        help='Maximum number of consecutive NaN values to forward fill (default: 5)')
    parser.add_argument('--feature_normalization_lookback', type=int, default=100,
                        help='Lookback period for feature normalization (default: 100)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to YAML configuration file (can be a GCS path)')
    
    # Feature parameters as JSON string (optional)
    parser.add_argument('--feature_params_json', type=str, default=None,
                        help='JSON string with parameters for feature calculation')
    
    # Normalization methods as JSON string (optional)
    parser.add_argument('--normalization_methods_json', type=str, default=None,
                        help='JSON string with normalization methods for features')
    
    # KFP v2 Output artifacts
    parser.add_argument('--sequences_output_path', type=str, required=True,
                        help='Output path for the sequences dataset artifact')
    parser.add_argument('--preprocessing_summary_path', type=str, required=True,
                        help='Output path for the preprocessing summary metrics artifact')
    
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

def run_preprocessing(
    project_id: str,
    input_data_uri: str,
    output_gcs_bucket: str,
    output_gcs_prefix: str = None,
    sequence_length: int = 60,
    ffill_limit_for_nans: int = 5,
    feature_normalization_lookback: int = 100,
    feature_params: Dict[str, Any] = None,
    normalization_methods: Dict[str, str] = None,
    sequences_output: Output[Dataset] = None,
    preprocessing_summary: Output[Dataset] = None
) -> str:
    """
    Run the preprocessing pipeline to create sequences for the RL agent.
    
    Args:
        project_id (str): Google Cloud project ID.
        input_data_uri (str): GCS URI of the input Parquet file with OHLCV data.
        output_gcs_bucket (str): GCS bucket for output sequences.
        output_gcs_prefix (str, optional): Prefix for output GCS path.
        sequence_length (int, optional): Length of sequences to create. Defaults to 60.
        ffill_limit_for_nans (int, optional): Maximum number of consecutive NaN values to forward fill.
        feature_normalization_lookback (int, optional): Lookback period for feature normalization.
        feature_params (Dict[str, Any], optional): Parameters for feature calculation.
        normalization_methods (Dict[str, str], optional): Normalization methods for features.
        sequences_output (Output[Dataset], optional): KFP output parameter for the sequences artifact.
        preprocessing_summary (Output[Dataset], optional): KFP output parameter for the summary metrics.
        
    Returns:
        str: GCS URI of the saved sequences file.
        
    Raises:
        Exception: If there's an error in the preprocessing pipeline.
    """
    try:
        logger.info(f"Starting preprocessing pipeline for data: {input_data_uri}")
        
        # Initialize the preprocessor
        preprocessor = DataPreprocessor(
            project_id=project_id,
            feature_params=feature_params,
            sequence_length=sequence_length,
            ffill_limit_for_nans=ffill_limit_for_nans,
            feature_normalization_lookback=feature_normalization_lookback,
            normalization_methods=normalization_methods
        )
        
        # Run the preprocessing pipeline
        output_uri = preprocessor.process_data_pipeline(
            input_gcs_uri=input_data_uri,
            output_gcs_bucket=output_gcs_bucket,
            output_gcs_prefix=output_gcs_prefix
        )
        
        logger.info(f"Preprocessing pipeline completed successfully. Sequences saved to: {output_uri}")
        
        # Store output for KFP
        if sequences_output:
            # Create a file with the GCS URI of the sequences
            with open(sequences_output.path, 'w') as f:
                f.write(output_uri)
            
            # Set metadata
            sequences_output.metadata['gcs_uri'] = output_uri
        
        # Create preprocessing summary metrics
        if preprocessing_summary:
            # Load metadata from GCS to get sequence info
            if output_uri.endswith('.npz'):
                metadata_uri = f"{output_uri}.metadata.json"
            else:
                metadata_uri = f"{output_uri}.metadata.json"
            
            try:
                # Parse bucket and blob path from GCS URI
                if metadata_uri.startswith('gs://'):
                    metadata_uri = metadata_uri[5:]  # Remove 'gs://' prefix
                
                bucket_name, blob_path = metadata_uri.split('/', 1)
                
                # Initialize GCS client and download the metadata
                storage_client = storage.Client(project=project_id)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                # Check if metadata exists
                if blob.exists():
                    metadata_content = blob.download_as_text()
                    metadata = json.loads(metadata_content)
                    
                    # Write summary metrics
                    with open(preprocessing_summary.path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Set metadata fields
                    preprocessing_summary.metadata['n_sequences'] = metadata.get('n_sequences', 0)
                    preprocessing_summary.metadata['sequence_length'] = metadata.get('sequence_length', 0)
                    preprocessing_summary.metadata['n_features'] = metadata.get('n_features', 0)
                else:
                    logger.warning(f"Metadata file not found: {metadata_uri}")
                    
                    # Create a default summary
                    summary = {
                        'n_sequences': 'unknown',
                        'sequence_length': sequence_length,
                        'n_features': 'unknown',
                        'output_uri': output_uri
                    }
                    
                    with open(preprocessing_summary.path, 'w') as f:
                        json.dump(summary, f, indent=2)
            
            except Exception as e:
                logger.warning(f"Error loading sequence metadata: {str(e)}")
                
                # Create a default summary
                summary = {
                    'output_uri': output_uri,
                    'error': str(e)
                }
                
                with open(preprocessing_summary.path, 'w') as f:
                    json.dump(summary, f, indent=2)
        
        return output_uri
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        
        # Create error summary if possible
        if preprocessing_summary:
            error_summary = {
                'error': str(e),
                'input_data_uri': input_data_uri
            }
            
            with open(preprocessing_summary.path, 'w') as f:
                json.dump(error_summary, f, indent=2)
        
        raise

def main():
    """
    Main entry point for the preprocessing component.
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
        
        # Parse JSON strings if provided
        feature_params = None
        if args.feature_params_json:
            feature_params = json.loads(args.feature_params_json)
        elif 'feature_params' in config:
            feature_params = config.get('feature_params')
        
        normalization_methods = None
        if args.normalization_methods_json:
            normalization_methods = json.loads(args.normalization_methods_json)
        elif 'normalization_methods' in config:
            normalization_methods = config.get('normalization_methods')
        
        # Create the output parameters
        sequences_output = Output(type=Dataset, path=args.sequences_output_path)
        preprocessing_summary = Output(type=Dataset, path=args.preprocessing_summary_path)
        
        # Command line arguments take precedence over config file
        sequence_length = args.sequence_length if args.sequence_length else config.get('sequence_length', 60)
        ffill_limit = args.ffill_limit_for_nans if args.ffill_limit_for_nans else config.get('ffill_limit_for_nans', 5)
        norm_lookback = args.feature_normalization_lookback if args.feature_normalization_lookback else config.get('feature_normalization_lookback', 100)
        
        # Run preprocessing
        output_uri = run_preprocessing(
            project_id=args.project_id,
            input_data_uri=args.input_data_uri,
            output_gcs_bucket=args.output_gcs_bucket,
            output_gcs_prefix=args.output_gcs_prefix,
            sequence_length=sequence_length,
            ffill_limit_for_nans=ffill_limit,
            feature_normalization_lookback=norm_lookback,
            feature_params=feature_params,
            normalization_methods=normalization_methods,
            sequences_output=sequences_output,
            preprocessing_summary=preprocessing_summary
        )
        
        logger.info(f"Preprocessing component completed successfully. Output: {output_uri}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing component: {str(e)}")
        raise

if __name__ == '__main__':
    main()
