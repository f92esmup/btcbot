#!/usr/bin/env python3
# filepath: /Users/f92esmup/btcbot/gcp/10_cleanup_resources.py
"""
Script para eliminar recursos de GCP asociados al proyecto BTCBot.

Este script permite eliminar de forma segura los recursos creados por los
scripts de configuración e implementación, proporcionando opciones para
eliminar todos los recursos o solo categorías específicas.
"""
import argparse
import logging
import sys
import time
from typing import List, Optional

from google.api_core.exceptions import NotFound
from google.cloud import storage, secretmanager, aiplatform
from google.cloud.devtools import artifactregistry_v1
from googleapiclient import discovery

# Importar la configuración centralizada
sys.path.append(".")
from gcp.common import config, clients

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Elimina recursos de GCP asociados con el proyecto BTCBot."
    )
    parser.add_argument(
        "--resources",
        type=str,
        choices=["all", "endpoints", "models", "jobs", "services", 
                 "storage", "secrets", "service_accounts", "docker_images"],
        default="all",
        help="Categoría de recursos a eliminar (por defecto: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar la eliminación sin solicitar confirmación"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=config.PROJECT_ID,
        help=f"ID del proyecto GCP (por defecto: {config.PROJECT_ID})"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=config.REGION,
        help=f"Región de GCP (por defecto: {config.REGION})"
    )
    
    return parser.parse_args()


def confirm_deletion(resource_type: str) -> bool:
    """Solicita confirmación del usuario para eliminar recursos."""
    confirm = input(f"¿Estás seguro de que quieres eliminar los recursos de {resource_type}? "
                   "Esta acción no se puede deshacer. (s/N): ")
    return confirm.lower() in ("s", "si", "sí", "y", "yes")


def delete_endpoints(project_id: str, region: str, force: bool = False) -> None:
    """Elimina los endpoints de Vertex AI."""
    if not force and not confirm_deletion("Endpoints de Vertex AI"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando endpoints de Vertex AI...")
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Listar todos los endpoints
    endpoints = aiplatform.Endpoint.list()
    
    if not endpoints:
        logger.info("No se encontraron endpoints para eliminar.")
        return
    
    for endpoint in endpoints:
        try:
            logger.info(f"Eliminando endpoint: {endpoint.name}")
            endpoint.delete(force=True)
            logger.info(f"Endpoint {endpoint.name} eliminado correctamente.")
        except Exception as e:
            logger.error(f"Error al eliminar el endpoint {endpoint.name}: {e}")


def delete_models(project_id: str, region: str, force: bool = False) -> None:
    """Elimina los modelos de Vertex AI."""
    if not force and not confirm_deletion("Modelos de Vertex AI"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando modelos de Vertex AI...")
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Listar todos los modelos
    models = aiplatform.Model.list()
    
    if not models:
        logger.info("No se encontraron modelos para eliminar.")
        return
    
    for model in models:
        try:
            logger.info(f"Eliminando modelo: {model.name}")
            model.delete()
            logger.info(f"Modelo {model.name} eliminado correctamente.")
        except Exception as e:
            logger.error(f"Error al eliminar el modelo {model.name}: {e}")


def delete_training_jobs(project_id: str, region: str, force: bool = False) -> None:
    """Elimina los jobs de entrenamiento de Vertex AI."""
    if not force and not confirm_deletion("Jobs de entrenamiento"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando jobs de entrenamiento de Vertex AI...")
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Listar todos los jobs de entrenamiento
    # Nota: Los jobs no se pueden eliminar, pero podemos listarlos para informar al usuario
    jobs = aiplatform.CustomJob.list()
    
    if not jobs:
        logger.info("No se encontraron jobs de entrenamiento.")
        return
    
    logger.info(f"Se encontraron {len(jobs)} jobs de entrenamiento.")
    logger.info("Nota: Los jobs de entrenamiento no se pueden eliminar, pero dejarán de incurrir en costos una vez completados.")
    
    # Listar los jobs en curso
    running_jobs = [job for job in jobs if job.state == "JOB_STATE_RUNNING"]
    
    if running_jobs and (force or confirm_deletion("Jobs de entrenamiento en ejecución")):
        for job in running_jobs:
            try:
                logger.info(f"Cancelando job en ejecución: {job.name}")
                job.cancel()
                logger.info(f"Job {job.name} cancelado correctamente.")
            except Exception as e:
                logger.error(f"Error al cancelar el job {job.name}: {e}")


def delete_cloud_run_services(project_id: str, region: str, force: bool = False) -> None:
    """
    Función para eliminar servicios Cloud Run.
    
    Nota: Esta función se mantiene como stub ya que el servicio de adquisición de datos
    ha sido integrado en el pipeline de Vertex AI y ya no se despliega en Cloud Run.
    """
    logger.info("No hay servicios de Cloud Run que eliminar (el servicio de adquisición de datos ahora es parte del pipeline).")
    return


def delete_storage_buckets(project_id: str, force: bool = False) -> None:
    """Elimina los buckets de Cloud Storage."""
    if not force and not confirm_deletion("Buckets de almacenamiento"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando buckets de Cloud Storage...")
    
    # Inicializar el cliente de Storage
    storage_client = storage.Client(project=project_id)
    
    # Buckets a eliminar según la configuración
    bucket_names = [
        config.RAW_DATA_BUCKET,
        config.PROCESSED_DATA_BUCKET,
        config.MODELS_STAGING_BUCKET,
        config.EVALUATION_RESULTS_BUCKET
    ]
    
    for bucket_name in bucket_names:
        try:
            # Verificar si el bucket existe
            bucket = storage_client.bucket(bucket_name)
            if not bucket.exists():
                logger.info(f"El bucket {bucket_name} no existe.")
                continue
            
            logger.info(f"Eliminando bucket: {bucket_name}")
            
            # Eliminar todos los objetos del bucket
            blobs = list(bucket.list_blobs())
            for blob in blobs:
                logger.debug(f"Eliminando objeto: {blob.name}")
                blob.delete()
            
            # Eliminar todas las versiones de los objetos si está habilitado el versionado
            if bucket.versioning_enabled:
                blobs = list(bucket.list_blobs(versions=True))
                for blob in blobs:
                    logger.debug(f"Eliminando versión de objeto: {blob.name} (generation: {blob.generation})")
                    blob.delete()
            
            # Eliminar el bucket
            bucket.delete()
            logger.info(f"Bucket {bucket_name} eliminado correctamente.")
        except Exception as e:
            logger.error(f"Error al eliminar el bucket {bucket_name}: {e}")


def delete_secrets(project_id: str, force: bool = False) -> None:
    """Elimina los secretos de Secret Manager."""
    if not force and not confirm_deletion("Secretos"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando secretos de Secret Manager...")
    
    # Inicializar el cliente de Secret Manager
    client = secretmanager.SecretManagerServiceClient()
    
    # Secretos a eliminar según la configuración
    secret_names = [
        config.BINANCE_API_KEY_SECRET_NAME,
        config.BINANCE_API_SECRET_SECRET_NAME
    ]
    
    for secret_name in secret_names:
        try:
            # Nombre completo del recurso
            name = f"projects/{project_id}/secrets/{secret_name}"
            
            logger.info(f"Eliminando secreto: {secret_name}")
            client.delete_secret(request={"name": name})
            logger.info(f"Secreto {secret_name} eliminado correctamente.")
        except NotFound:
            logger.info(f"El secreto {secret_name} no existe.")
        except Exception as e:
            logger.error(f"Error al eliminar el secreto {secret_name}: {e}")


def delete_service_accounts(project_id: str, force: bool = False) -> None:
    """Elimina las cuentas de servicio."""
    if not force and not confirm_deletion("Cuentas de servicio"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando cuentas de servicio...")
    
    # Crear cliente de IAM
    service = discovery.build('iam', 'v1')
    
    # Nombre de la cuenta de servicio según la configuración
    sa_name = config.SERVICE_ACCOUNT_NAME
    sa_email = config.SERVICE_ACCOUNT_EMAIL
    
    try:
        # Nombre completo del recurso
        name = f'projects/{project_id}/serviceAccounts/{sa_email}'
        
        logger.info(f"Eliminando cuenta de servicio: {sa_name}")
        service.projects().serviceAccounts().delete(name=name).execute()
        logger.info(f"Cuenta de servicio {sa_name} eliminada correctamente.")
    except Exception as e:
        logger.error(f"Error al eliminar la cuenta de servicio {sa_name}: {e}")


def delete_docker_images(project_id: str, region: str, force: bool = False) -> None:
    """Elimina las imágenes de Docker de Artifact Registry."""
    if not force and not confirm_deletion("Imágenes de Docker"):
        logger.info("Operación cancelada por el usuario.")
        return

    logger.info("Eliminando imágenes de Docker de Artifact Registry...")
    
    # Inicializar cliente de Artifact Registry
    client = artifactregistry_v1.ArtifactRegistryClient()
    
    # Nombre del repositorio según la configuración
    repo_name = config.ARTIFACT_REPO
    
    # Nombre completo del repositorio
    repository = f"projects/{project_id}/locations/{region}/repositories/{repo_name}"
    
    try:
        # Listar todos los paquetes
        packages = client.list_packages(parent=repository)
        
        if not list(packages):
            logger.info(f"No se encontraron paquetes en el repositorio {repo_name}.")
            return
        
        # Eliminar cada paquete y sus versiones
        for package in packages:
            package_name = package.name
            logger.info(f"Eliminando paquete: {package_name}")
            
            # Listar versiones del paquete
            versions = client.list_versions(parent=package_name)
            
            # Eliminar cada versión
            for version in versions:
                version_name = version.name
                logger.info(f"Eliminando versión: {version_name}")
                client.delete_version(name=version_name)
            
            # Eliminar el paquete
            client.delete_package(name=package_name)
            logger.info(f"Paquete {package_name} eliminado correctamente.")
        
        logger.info(f"Todas las imágenes del repositorio {repo_name} eliminadas correctamente.")
    except Exception as e:
        logger.error(f"Error al eliminar imágenes de Docker: {e}")


def main():
    """Función principal."""
    args = parse_args()
    
    # Validar que el usuario ha especificado un tipo de recurso válido
    valid_resources = ["all", "endpoints", "models", "jobs", "services", 
                        "storage", "secrets", "service_accounts", "docker_images"]
    if args.resources not in valid_resources:
        logger.error(f"Tipo de recurso inválido: {args.resources}")
        logger.info(f"Recursos válidos: {', '.join(valid_resources)}")
        return
    
    # Si no se especifica --force, mostrar advertencia general
    if not args.force:
        logger.warning("¡ADVERTENCIA! Estás a punto de eliminar recursos de GCP.")
        logger.warning("Esta acción no se puede deshacer y puede resultar en pérdida permanente de datos.")
        logger.warning(f"Proyecto: {args.project_id}, Región: {args.region}")
        
        if not confirm_deletion("BTCBot en GCP"):
            logger.info("Operación cancelada por el usuario.")
            return
    
    # Eliminar los recursos según la opción seleccionada
    if args.resources in ["all", "endpoints"]:
        delete_endpoints(args.project_id, args.region, args.force)
    
    if args.resources in ["all", "models"]:
        delete_models(args.project_id, args.region, args.force)
    
    if args.resources in ["all", "jobs"]:
        delete_training_jobs(args.project_id, args.region, args.force)
    
    if args.resources in ["all", "services"]:
        delete_cloud_run_services(args.project_id, args.region, args.force)
    
    if args.resources in ["all", "storage"]:
        delete_storage_buckets(args.project_id, args.force)
    
    if args.resources in ["all", "secrets"]:
        delete_secrets(args.project_id, args.force)
    
    if args.resources in ["all", "service_accounts"]:
        delete_service_accounts(args.project_id, args.force)
    
    if args.resources in ["all", "docker_images"]:
        delete_docker_images(args.project_id, args.region, args.force)
    
    logger.info("¡Proceso de limpieza completado!")


if __name__ == "__main__":
    main()