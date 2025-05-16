"""
Script para configurar secretos en Google Secret Manager para las claves API de Binance.
"""
import argparse
from google.cloud import secretmanager
from google.cloud.secretmanager_v1.types import Secret, Replication
from google.api_core.exceptions import AlreadyExists
from common import config, clients

def create_secret_if_not_exists(client, secret_id):
    """
    Crea un secreto si no existe.
    
    Args:
        client: Cliente de Secret Manager.
        secret_id: ID del secreto a crear.
    """
    try:
        secret_path = f"projects/{config.PROJECT_ID}/secrets/{secret_id}"
        client.create_secret(
            request={
                "parent": f"projects/{config.PROJECT_ID}",
                "secret_id": secret_id,
                "secret": Secret(
                    replication=Replication(automatic=Replication.Automatic())
                ),
            }
        )
        print(f"Secreto {secret_id} creado exitosamente.")
    except AlreadyExists:
        print(f"El secreto {secret_id} ya existe.")

def add_secret_version(client, secret_id, secret_value):
    """
    Añade una nueva versión a un secreto existente.
    
    Args:
        client: Cliente de Secret Manager.
        secret_id: ID del secreto.
        secret_value: Valor del secreto.
    """
    parent = client.secret_path(config.PROJECT_ID, secret_id)
    response = client.add_secret_version(
        request={
            "parent": parent,
            "payload": {"data": secret_value.encode("UTF-8")},
        }
    )
    print(f"Versión {response.name} añadida al secreto {secret_id}.")

def setup_secrets(api_key, api_secret):
    """
    Configura los secretos necesarios para la API de Binance.
    
    Args:
        api_key: Clave API de Binance.
        api_secret: Secreto API de Binance.
    """
    client = clients.get_secret_manager_client()
    
    # Crear secretos si no existen
    create_secret_if_not_exists(client, config.BINANCE_API_KEY_SECRET_NAME)
    create_secret_if_not_exists(client, config.BINANCE_API_SECRET_SECRET_NAME)
    
    # Añadir valores a los secretos
    add_secret_version(client, config.BINANCE_API_KEY_SECRET_NAME, api_key)
    add_secret_version(client, config.BINANCE_API_SECRET_SECRET_NAME, api_secret)
    
    print("Secretos configurados exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurar secretos para la API de Binance en GCP Secret Manager")
    parser.add_argument("--api_key", required=True, help="Clave API de Binance")
    parser.add_argument("--api_secret", required=True, help="Secreto API de Binance")
    
    args = parser.parse_args()
    setup_secrets(args.api_key, args.api_secret)
