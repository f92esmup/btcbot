"""
Utilidades para interactuar con Google Cloud Storage.
"""

import os
from google.cloud import storage, bigquery  # Added bigquery import
from pathlib import Path
import tempfile
import logging
import pandas as pd
import io
from typing import Dict, Any, List  # Added typing imports

logger = logging.getLogger("GCSUtils")

def upload_model_to_gcs(local_model_path: str, gcs_model_path: str) -> str:
    """
    Sube un modelo a Google Cloud Storage.
    
    Args:
        local_model_path: Ruta local al archivo del modelo
        gcs_model_path: Ruta en GCS donde guardar el modelo (sin gs://)
                        Formato: "bucket_name/path/to/model.zip"
    
    Returns:
        URL completa del modelo en GCS (gs://bucket/path/to/model.zip)
    """
    try:
        # Separar nombre del bucket y ruta del objeto
        parts = gcs_model_path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else os.path.basename(local_model_path)
        
        # Inicializar cliente de GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Subir el archivo
        blob.upload_from_filename(local_model_path)
        
        # Construir URL de GCS
        gcs_url = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Modelo subido exitosamente a: {gcs_url}")
        
        return gcs_url
        
    except Exception as e:
        logger.error(f"Error al subir el modelo a GCS: {str(e)}")
        raise

def download_model_from_gcs(gcs_model_path: str, local_dir: str = None) -> str:
    """
    Descarga un modelo desde Google Cloud Storage.
    
    Args:
        gcs_model_path: Ruta en GCS al modelo (sin gs://)
                        Formato: "bucket_name/path/to/model.zip"
        local_dir: Directorio local donde guardar el modelo (opcional)
    
    Returns:
        Ruta local al modelo descargado
    """
    try:
        # Separar nombre del bucket y ruta del objeto
        parts = gcs_model_path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        
        if not blob_name:
            raise ValueError("La ruta del modelo en GCS debe incluir el nombre del archivo")
        
        # Determinar ruta local
        if local_dir is None:
            # Usar directorio temporal si no se especifica uno
            local_dir = tempfile.gettempdir()
        
        # Asegurar que el directorio exista
        os.makedirs(local_dir, exist_ok=True)
        
        # Nombre del archivo local
        filename = os.path.basename(blob_name)
        local_model_path = os.path.join(local_dir, filename)
        
        # Inicializar cliente de GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Descargar el archivo
        blob.download_to_filename(local_model_path)
        
        logger.info(f"Modelo descargado exitosamente a: {local_model_path}")
        return local_model_path
        
    except Exception as e:
        logger.error(f"Error al descargar el modelo desde GCS: {str(e)}")
        raise

def upload_dataframe_to_gcs(df, bucket_name: str, blob_path: str, if_exists: str = 'replace') -> bool:
    """
    Sube un DataFrame de pandas a Google Cloud Storage.
    
    Args:
        df: DataFrame de pandas a subir
        bucket_name: Nombre del bucket de GCS
        blob_path: Ruta al archivo dentro del bucket (sin el nombre del bucket)
        if_exists: Comportamiento cuando el archivo ya existe:
                   - 'replace': Reemplaza el archivo existente (predeterminado)
                   - 'append': Añade los datos al final del archivo existente
    
    Returns:
        True si la operación fue exitosa, False en caso contrario
    """
    try:
        # Inicializar cliente de GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Verificar si se debe aplicar append
        if if_exists == 'append' and blob.exists():
            # Descargar archivo existente
            buffer = io.BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)
            
            # Leer archivo existente como DataFrame
            try:
                existing_df = pd.read_csv(buffer)
                # Concatenar con el nuevo DataFrame
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                logger.warning(f"No se pudo leer el archivo existente para append. Creando nuevo archivo: {str(e)}")
        
        # Convertir DataFrame a CSV en memoria
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Subir a GCS
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        
        # Construir URL de GCS para el log
        gcs_url = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"DataFrame subido exitosamente a: {gcs_url}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al subir DataFrame a GCS: {str(e)}")
        return False

def stream_row_to_bigquery(bigquery_client: bigquery.Client, table_id: str, row: Dict[str, Any]) -> bool:
    """
    Inserta una única fila (diccionario) en una tabla de BigQuery.

    Args:
        bigquery_client: Instancia del cliente de BigQuery.
        table_id: ID completo de la tabla de BigQuery (ej: "project_id.dataset_id.table_id").
        row: Un diccionario que representa la fila a insertar. Las claves deben coincidir
             con los nombres de las columnas de la tabla de BigQuery.

    Returns:
        True si la inserción fue exitosa (sin errores reportados por la API), False en caso contrario.
    """
    if not row:
        logger.warning("Fila vacía proporcionada para inserción en BigQuery. Saltando.")
        return False

    try:
        # La API insert_rows_json espera una lista de filas.
        errors = bigquery_client.insert_rows_json(table_id, [row])
        if not errors:
            logger.debug(f"Fila insertada correctamente en BigQuery table {table_id}: {row.get('timestamp_decision_madrid', 'No timestamp')}")
            return True
        else:
            logger.error(f"Errores al insertar fila en BigQuery table {table_id}: {errors}")
            for error_detail in errors:
                logger.error(f"  - Fila: {error_detail['index']}, Errores: {error_detail['errors']}")
            return False
    except Exception as e:
        logger.error(f"Excepción al insertar fila en BigQuery table {table_id}: {e}", exc_info=True)
        return False
