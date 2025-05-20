import os
import json
import numpy as np
import requests
import time
import argparse
import logging
from typing import Dict, List, Any
from google.cloud import aiplatform

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EndpointTest")

def create_sample_data(sequence_length: int = 96, portfolio_features_size: int = 4) -> Dict[str, List]:
    """
    Crea datos de ejemplo para enviar al endpoint.
    
    Args:
        sequence_length: Longitud de la secuencia para las características del mercado
        portfolio_features_size: Número de características del portafolio
        
    Returns:
        Diccionario con datos de ejemplo
    """
    # Crear características del mercado de ejemplo (secuencia de L pasos temporales x F características)
    num_market_features = 15  # Según las columnas en config.yaml
    
    # Generar datos aleatorios normalizados para simular una secuencia de mercado
    market_features = np.random.randn(sequence_length, num_market_features).astype(np.float32)
    
    # Generar datos aleatorios para las características del portafolio
    # Normalmente: [equity_usd, position_size_btc, avg_entry_price, unrealized_pnl_pct]
    portfolio_features = np.random.randn(portfolio_features_size).astype(np.float32)
    
    # Convertir a listas para serialización JSON
    return {
        "market_features": market_features.tolist(),
        "portfolio_features": portfolio_features.tolist()
    }

def test_local_endpoint(url: str = "http://localhost:8080"):
    """
    Prueba un endpoint local.
    
    Args:
        url: URL base del endpoint local
    """
    # Verificar si el servidor está funcionando
    try:
        health_response = requests.get(f"{url}/health")
        if health_response.status_code != 200:
            logger.error(f"El servidor no está saludable. Status: {health_response.status_code}")
            return
        logger.info("Servidor saludable")
    except requests.exceptions.ConnectionError:
        logger.error(f"No se pudo conectar al servidor en {url}")
        return
    
    # Crear datos de ejemplo
    sample_data = create_sample_data()
    
    # Medir tiempo de respuesta
    start_time = time.time()
    
    # Enviar solicitud al endpoint
    try:
        response = requests.post(
            f"{url}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Predicción exitosa en {elapsed_time:.4f} segundos")
            logger.info(f"Acción predicha: {result['action']}")
            logger.info(f"Valor de acción: {result['action_value']}")
        else:
            logger.error(f"Error en la predicción. Status: {response.status_code}")
            logger.error(f"Respuesta: {response.text}")
    except Exception as e:
        logger.error(f"Error al enviar la solicitud: {str(e)}")

def test_vertex_endpoint(
    project_id: str,
    endpoint_id: str,
    location: str = "us-central1"
):
    """
    Prueba un endpoint desplegado en Vertex AI.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        endpoint_id: ID del endpoint en Vertex AI
        location: Región donde está desplegado el endpoint
    """
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Obtener el endpoint
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    # Crear datos de ejemplo
    sample_data = create_sample_data()
    
    # Medir tiempo de respuesta
    start_time = time.time()
    
    try:
        # Realizar la predicción
        response = endpoint.predict(instances=[sample_data])
        
        elapsed_time = time.time() - start_time
        
        # Procesar respuesta
        logger.info(f"Predicción exitosa en {elapsed_time:.4f} segundos")
        logger.info(f"Respuesta del endpoint: {response}")
        
        # La estructura exacta de la respuesta dependerá de cómo se configure el endpoint
        # y cómo el servidor procese y devuelva los resultados
        
    except Exception as e:
        logger.error(f"Error al realizar la predicción en Vertex AI: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de endpoint para el modelo de trading")
    parser.add_argument("--mode", choices=["local", "vertex"], default="local",
                        help="Modo de prueba: local o vertex")
    parser.add_argument("--url", default="http://localhost:8080",
                        help="URL del endpoint local (solo para modo local)")
    parser.add_argument("--project-id", help="ID del proyecto de Google Cloud (solo para modo vertex)")
    parser.add_argument("--endpoint-id", help="ID del endpoint en Vertex AI (solo para modo vertex)")
    parser.add_argument("--location", default="us-central1",
                        help="Región donde está desplegado el endpoint (solo para modo vertex)")
    
    args = parser.parse_args()
    
    if args.mode == "local":
        test_local_endpoint(url=args.url)
    else:  # vertex
        if not args.project_id or not args.endpoint_id:
            parser.error("Para el modo vertex, se requieren --project-id y --endpoint-id")
        test_vertex_endpoint(
            project_id=args.project_id,
            endpoint_id=args.endpoint_id,
            location=args.location
        )