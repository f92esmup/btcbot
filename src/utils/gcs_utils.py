"""
Utilidades para interactuar con Google Cloud Storage.
"""

import os
from google.cloud import storage
from pathlib import Path
import tempfile
import logging

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
