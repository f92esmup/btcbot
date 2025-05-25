# src/utils/bigquery_utils.py
import logging
import time
import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)

def sanitize_for_json(data):
    """
    Recursively sanitize data for JSON serialization by converting numpy types to native Python types,
    datetime objects to ISO strings, and handling NaN/infinity values.
    
    Args:
        data: The data to sanitize (can be dict, list, or primitive)
        
    Returns:
        The sanitized data with numpy types converted to native Python types and NaN/inf handled
    """
    import datetime
    
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, datetime.datetime):
        # Convert datetime to ISO format string for BigQuery JSON serialization
        return data.isoformat()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        # Handle NaN and infinity values
        if np.isnan(data):
            return 0.0
        elif np.isinf(data):
            return 0.0
        else:
            return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, float):
        # Handle native Python float NaN and infinity
        if np.isnan(data) or np.isinf(data):
            return 0.0
        else:
            return data
    else:
        return data

def stream_data_to_bigquery(
    project_id: str,
    dataset_id: str,
    table_id: str,
    rows_to_insert: list,
    client: bigquery.Client = None,
    schema: list = None,
    max_retries: int = 2,
    initial_retry_delay_seconds: float = 1.0
):
    """
    Streams data to a BigQuery table. Handles table creation if schema is provided.

    Args:
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        rows_to_insert (list): A list of dictionaries, where each dictionary is a row.
        client (bigquery.Client, optional): An existing BigQuery client. If None, a new one is created.
        schema (list, optional): A list of bigquery.SchemaField objects.
                                 Required if the table might not exist.
        max_retries (int): Maximum number of retries for transient errors.
        initial_retry_delay_seconds (float): Initial delay before the first retry.
    Returns:
        bool: True if successful, False otherwise.
    """
    if not rows_to_insert:
        logger.info(f"No rows to insert into {dataset_id}.{table_id}.")
        return True

    # Sanitize rows to handle numpy types that aren't JSON serializable
    rows_to_insert = sanitize_for_json(rows_to_insert)

    if client is None:
        client = bigquery.Client(project=project_id)

    table_ref = client.dataset(dataset_id).table(table_id)
    table_exists = False

    try:
        client.get_table(table_ref)
        table_exists = True
        logger.debug(f"Table {project_id}.{dataset_id}.{table_id} already exists.")
    except NotFound:
        if schema:
            logger.info(f"Table {project_id}.{dataset_id}.{table_id} not found. Attempting to create it.")
            bigquery_table = bigquery.Table(table_ref, schema=schema)
            try:
                client.create_table(bigquery_table)
                logger.info(f"Table {project_id}.{dataset_id}.{table_id} created successfully.")
                table_exists = True
            except Exception as e:
                logger.error(f"Failed to create table {project_id}.{dataset_id}.{table_id}: {e}", exc_info=True)
                return False # Cannot proceed if table creation fails
        else:
            logger.warning(
                f"Table {project_id}.{dataset_id}.{table_id} not found, and no schema provided to create it."
            )
            return False # Cannot proceed if table doesn't exist and no schema to create

    if not table_exists: # Should only happen if schema was not provided for a non-existent table
        logger.error(f"Cannot stream data: Table {project_id}.{dataset_id}.{table_id} does not exist and could not be created.")
        return False

    errors = None
    for attempt in range(max_retries + 1):
        try:
            errors = client.insert_rows_json(table_ref, rows_to_insert)
            if not errors:
                logger.info(
                    f"Successfully streamed {len(rows_to_insert)} rows to {project_id}.{dataset_id}.{table_id}."
                )
                return True
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1}: Encountered errors inserting rows into "
                    f"{project_id}.{dataset_id}.{table_id}: {errors}"
                )
                # Decide if errors are retryable - for simplicity, retrying on any BQ error for now
                # More specific error handling could be added here (e.g. check error reasons)

        except Exception as e: # Catch other potential exceptions like network issues
            logger.error(
                f"Attempt {attempt + 1}/{max_retries + 1}: Exception while trying to stream data to "
                f"{project_id}.{dataset_id}.{table_id}: {e}", exc_info=True
            )
            errors = [e] # Populate errors to trigger retry logic or final failure

        if attempt < max_retries:
            delay = initial_retry_delay_seconds * (2 ** attempt) # Exponential backoff
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        else: # Max retries reached
            logger.error(
                f"Failed to stream data to {project_id}.{dataset_id}.{table_id} after {max_retries + 1} attempts. Final errors: {errors}"
            )
            return False
    return False # Should not be reached if logic is correct
