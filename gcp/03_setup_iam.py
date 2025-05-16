"""
Script para configurar IAM y cuentas de servicio en GCP.
"""
import argparse
import time
import subprocess
from google.cloud import iam_admin_v1
from google.api_core.exceptions import AlreadyExists, PermissionDenied
from common import config, clients

def create_service_account(project_id, name, display_name, description):
    """
    Crea una cuenta de servicio si no existe.
    
    Args:
        project_id: ID del proyecto GCP.
        name: Nombre único de la cuenta de servicio.
        display_name: Nombre para mostrar de la cuenta de servicio.
        description: Descripción de la cuenta de servicio.
    """
    client = iam_admin_v1.IAMClient()
    try:
        request = iam_admin_v1.CreateServiceAccountRequest(
            name=f"projects/{project_id}",
            account_id=name,
            service_account=iam_admin_v1.ServiceAccount(
                display_name=display_name,
                description=description
            )
        )
        service_account = client.create_service_account(request=request)
        print(f"Cuenta de servicio {name} creada.")
        return service_account
    except AlreadyExists:
        print(f"La cuenta de servicio {name} ya existe.")
        request = iam_admin_v1.GetServiceAccountRequest(
            name=f"projects/{project_id}/serviceAccounts/{name}@{project_id}.iam.gserviceaccount.com"
        )
        return client.get_service_account(request=request)

def add_role_to_service_account(project_id, service_account_email, role):
    """
    Añade un rol a una cuenta de servicio usando gcloud.
    
    Args:
        project_id: ID del proyecto GCP.
        service_account_email: Email de la cuenta de servicio.
        role: Rol a añadir (ej: roles/storage.admin).
    """
    try:
        # Usar gcloud para asignar el rol a la cuenta de servicio
        cmd = [
            "gcloud", "projects", "add-iam-policy-binding", project_id,
            "--member", f"serviceAccount:{service_account_email}",
            "--role", role
        ]
        
        print(f"Asignando rol {role} a {service_account_email}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Rol {role} asignado a {service_account_email} correctamente.")
        
        # Esperar a que se propaguen los permisos
        time.sleep(2)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al asignar rol {role}: {e.stderr}")
        return False

def setup_iam():
    """
    Configura todas las cuentas de servicio y permisos necesarios.
    """
    # Crear cuenta de servicio principal
    service_account = create_service_account(
        config.PROJECT_ID,
        config.SERVICE_ACCOUNT_NAME,
        "BTCBot Service Account",
        "Cuenta de servicio principal para el proyecto BTCBot"
    )
    
    # Asignar roles necesarios
    roles = [
        "roles/storage.admin",           # Acceso completo a buckets GCS
        "roles/secretmanager.secretAccessor",  # Acceso a secretos
        "roles/aiplatform.user",         # Uso de Vertex AI
        "roles/artifactregistry.admin",  # Administración de imágenes Docker
        "roles/logging.logWriter",       # Escritura de logs
        "roles/monitoring.metricWriter"  # Escritura de métricas
    ]
    
    for role in roles:
        add_role_to_service_account(config.PROJECT_ID, config.SERVICE_ACCOUNT_EMAIL, role)
    
    # Roles específicos para Vertex AI Model Registry
    vertex_roles = [
        "roles/aiplatform.admin",        # Administración de recursos de Vertex AI
        "roles/ml.admin"                 # Administración de recursos de ML
    ]
    
    for role in vertex_roles:
        add_role_to_service_account(config.PROJECT_ID, config.SERVICE_ACCOUNT_EMAIL, role)
    
    print("Configuración de IAM completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurar IAM para el proyecto btcbot")
    # Sin argumentos adicionales, ya que la configuración se define en config.py
    
    args = parser.parse_args()
    setup_iam()
