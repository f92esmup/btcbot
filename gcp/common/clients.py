"""
Clientes centralizados para acceder a los servicios de GCP.
"""
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import aiplatform
from google.api_core.client_options import ClientOptions
from . import config

def get_storage_client():
    """
    Obtiene un cliente para Google Cloud Storage.
    """
    return storage.Client(project=config.PROJECT_ID)

def get_secret_manager_client():
    """
    Obtiene un cliente para Google Secret Manager.
    """
    return secretmanager.SecretManagerServiceClient()

def get_aiplatform_client():
    """
    Obtiene un cliente para Google Cloud AI Platform.
    """
    client_options = ClientOptions(api_endpoint=f"{config.REGION}-aiplatform.googleapis.com")
    aiplatform.init(
        project=config.PROJECT_ID,
        location=config.REGION,
        client_options=client_options
    )
    return aiplatform

def access_secret(secret_name):
    """
    Accede al valor más reciente de un secreto.

    Args:
        secret_name: Nombre del secreto en Secret Manager.

    Returns:
        El valor del secreto como una cadena UTF-8.
    """
    client = get_secret_manager_client()
    name = f"projects/{config.PROJECT_ID}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
