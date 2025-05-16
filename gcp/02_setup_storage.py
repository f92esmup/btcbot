"""
Script para configurar buckets de Google Cloud Storage con versionado habilitado.
"""
import argparse
from google.cloud.exceptions import Conflict
from common import config, clients

def create_bucket_with_versioning(client, bucket_name, location):
    """
    Crea un bucket con versionado habilitado si no existe.
    
    Args:
        client: Cliente de GCS.
        bucket_name: Nombre del bucket a crear.
        location: Ubicación/región para el bucket.
    """
    try:
        bucket = client.create_bucket(bucket_name, location=location)
        print(f"Bucket {bucket_name} creado en {location}.")
    except Conflict:
        # El bucket ya existe, lo obtenemos
        bucket = client.get_bucket(bucket_name)
        print(f"El bucket {bucket_name} ya existe.")
    
    # Habilitar versionado en el bucket
    bucket.versioning_enabled = True
    bucket.patch()
    print(f"Versionado habilitado en el bucket {bucket_name}.")
    
    return bucket

def setup_storage():
    """
    Configura todos los buckets necesarios para el proyecto.
    """
    client = clients.get_storage_client()
    location = config.REGION
    
    # Crear buckets con versionado
    create_bucket_with_versioning(client, config.RAW_DATA_BUCKET, location)
    create_bucket_with_versioning(client, config.PROCESSED_DATA_BUCKET, location)
    create_bucket_with_versioning(client, config.MODELS_STAGING_BUCKET, location)
    create_bucket_with_versioning(client, config.EVALUATION_RESULTS_BUCKET, location)
    
    print("Todos los buckets han sido configurados correctamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurar buckets de GCS para el proyecto btcbot")
    # Sin argumentos adicionales, ya que los nombres de los buckets se definen en config.py
    
    args = parser.parse_args()
    setup_storage()
