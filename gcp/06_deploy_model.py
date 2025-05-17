#!/usr/bin/env python
"""
Script para desplegar un modelo registrado en Vertex AI a un endpoint.
"""
import argparse
import time
from google.cloud import aiplatform

from common import config, clients

def deploy_model(model_id, endpoint_name=None, machine_type="n1-standard-2"):
    """
    Despliega un modelo registrado en Vertex AI a un endpoint.
    
    Args:
        model_id: ID del modelo en Vertex AI Model Registry.
        endpoint_name: Nombre para el endpoint (opcional).
        machine_type: Tipo de máquina para el endpoint.
        
    Returns:
        Endpoint: El objeto del endpoint donde se desplegó el modelo.
    """
    # Inicializar el cliente de AI Platform
    aiplatform_client = clients.get_aiplatform_client()
    
    # Cargar el modelo
    model = aiplatform.Model(model_id)
    
    # Definir nombre del endpoint si no se proporcionó
    if endpoint_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        endpoint_name = f"btcbot-endpoint-{timestamp}"
    
    # Crear o cargar el endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"',
        order_by="create_time desc",
        project=config.PROJECT_ID,
        location=config.REGION
    )
    
    if endpoints:
        endpoint = endpoints[0]
        print(f"Usando endpoint existente: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            project=config.PROJECT_ID,
            location=config.REGION
        )
        print(f"Endpoint creado: {endpoint.display_name}")
    
    # Desplegar el modelo al endpoint
    deployed_model = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model.display_name}-deployment",
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1,
        service_account=config.SERVICE_ACCOUNT_EMAIL
    )
    
    print(f"Modelo desplegado exitosamente a {endpoint.display_name}")
    print(f"Endpoint URI: {endpoint.resource_name}")
    
    # Imprimir URL para acceder al modelo en la consola
    print(f"Ver en la consola: https://console.cloud.google.com/vertex-ai/endpoints/{endpoint.name}?project={config.PROJECT_ID}")
    
    return endpoint

def main():
    """
    Función principal para desplegar un modelo a un endpoint.
    """
    parser = argparse.ArgumentParser(description="Desplegar modelo desde Vertex AI Model Registry a un endpoint")
    parser.add_argument("--model_id", required=True, help="ID completo del modelo en Vertex AI")
    parser.add_argument("--endpoint_name", help="Nombre para el endpoint (opcional)")
    parser.add_argument("--machine_type", default="n1-standard-2", help="Tipo de máquina para el endpoint")
    
    args = parser.parse_args()
    
    try:
        deploy_model(
            model_id=args.model_id,
            endpoint_name=args.endpoint_name,
            machine_type=args.machine_type
        )
        
        print("Modelo desplegado correctamente.")
    except Exception as e:
        print(f"Error al desplegar el modelo: {e}")
        raise

if __name__ == "__main__":
    main()
