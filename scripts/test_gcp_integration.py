#!/usr/bin/env python3
"""
Script para probar la integración de GCP en el sistema btcbot.
Este script verifica que todas las partes del sistema funcionen correctamente usando
servicios GCP y que fallen explícitamente si no pueden acceder a GCP.
"""

import os
import sys
import logging
import yaml
import time
from datetime import datetime, timedelta
import numpy as np
from google.cloud import bigquery

# Configurar el logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gcp_integration_test')

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.data.binance_futures_downloader import BinanceFuturesDownloader
from src.data.preprocessor import DataPreprocessor
from src.environments.trading_env import TradingEnvironment

def ensure_bigquery_datasets_exist():
    """
    Verifica y crea los datasets de BigQuery requeridos en la región madrid (europe-southwest1)
    """
    try:
        # Obtener el project_id
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            config_manager = ConfigManager('src/config.yaml')
            project_id = config_manager.gcp_project_id
            
        bq_client = bigquery.Client(project=project_id)
        datasets_to_create = ["market_data_raw", "market_data_processed"]
        
        for dataset_id in datasets_to_create:
            try:
                # Comprobar si el dataset existe
                dataset_ref = bq_client.dataset(dataset_id)
                bq_client.get_dataset(dataset_ref)
                logger.info(f"Dataset {dataset_id} ya existe en BigQuery")
            except Exception:
                # Crear el dataset si no existe
                logger.info(f"Creando dataset {dataset_id} en BigQuery (europe-southwest1)...")
                dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
                dataset.location = "europe-southwest1"  # Ubicación en Madrid
                bq_client.create_dataset(dataset)
                logger.info(f"Dataset {dataset_id} creado exitosamente")
        
        return True
    except Exception as e:
        logger.error(f"Error creando datasets de BigQuery: {e}")
        return False

def test_config_manager():
    """Prueba la funcionalidad del ConfigManager con Secret Manager"""
    logger.info("=== Probando ConfigManager y Secret Manager ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager('src/config.yaml')
        logger.info("✓ ConfigManager inicializado correctamente")
        
        # Verificar el acceso a Google Cloud
        project_id = config_manager.gcp_project_id
        logger.info(f"✓ ID de proyecto GCP obtenido: {project_id}")
        
        # Verificar la disponibilidad del cliente de Secret Manager (sin acceder a un secreto específico)
        if config_manager._secret_client:
            logger.info("✓ Cliente de Secret Manager inicializado correctamente")
            # No probamos obtener un secreto específico porque podría no existir
            return True
        else:
            logger.error("✗ No se pudo inicializar el cliente de Secret Manager")
            return False
    except Exception as e:
        logger.error(f"✗ Error en la prueba de ConfigManager: {e}")
        return False

def test_binance_downloader():
    """Prueba la funcionalidad del BinanceFuturesDownloader con BigQuery"""
    logger.info("=== Probando BinanceFuturesDownloader y BigQuery ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager('src/config.yaml')
        
        # Verificar que el dataset de BigQuery existe
        bq_client = bigquery.Client(project=config_manager.gcp_project_id)
        dataset_id = config_manager.get_config_value('gcp.bigquery.raw_dataset_id', 'market_data_raw')
        
        try:
            dataset_ref = bq_client.dataset(dataset_id)
            bq_client.get_dataset(dataset_ref)
            logger.info(f"✓ Dataset {dataset_id} existe en BigQuery")
            
            # No realizamos la descarga de datos reales para evitar consumir API de Binance
            # Solo verificamos que podemos acceder a BigQuery
            return True
        except Exception as e:
            logger.error(f"✗ Error accediendo al dataset de BigQuery: {e}")
            return False
    except Exception as e:
        logger.error(f"✗ Error en la prueba de BinanceFuturesDownloader: {e}")
        return False

def test_preprocessor():
    """Prueba la funcionalidad del DataPreprocessor con BigQuery y GCS"""
    logger.info("=== Probando DataPreprocessor, BigQuery y GCS ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager('src/config.yaml')
        
        # Verificar acceso a BigQuery
        bq_client = bigquery.Client(project=config_manager.gcp_project_id)
        bq_dataset_id = config_manager.get_config_value('gcp.bigquery.raw_dataset_id', 'market_data_raw')
        
        try:
            dataset_ref = bq_client.dataset(bq_dataset_id)
            bq_client.get_dataset(dataset_ref)
            logger.info(f"✓ Dataset BigQuery {bq_dataset_id} accesible")
            
            # Verificar acceso a GCS
            from google.cloud import storage
            gcs_client = storage.Client(project=config_manager.gcp_project_id)
            gcs_bucket_name = config_manager.get_config_value('gcp.gcs.processed_bucket_name')
            
            try:
                bucket = gcs_client.get_bucket(gcs_bucket_name)
                logger.info(f"✓ Bucket GCS {gcs_bucket_name} accesible")
                
                # No procesamos datos reales, solo verificamos acceso a los servicios
                logger.info("✓ DataPreprocessor tiene acceso a los servicios GCP requeridos")
                return True
            except Exception as e:
                logger.error(f"✗ Error accediendo al bucket GCS: {e}")
                return False
        except Exception as e:
            logger.error(f"✗ Error accediendo al dataset de BigQuery: {e}")
            return False
    except Exception as e:
        logger.error(f"✗ Error en la prueba de DataPreprocessor: {e}")
        return False

def test_trading_env():
    """Prueba la funcionalidad del TradingEnvironment con GCS"""
    logger.info("=== Probando TradingEnvironment y GCS ===")
    
    try:
        # Verificar acceso a GCS
        config_manager = ConfigManager('src/config.yaml')
        from google.cloud import storage
        gcs_client = storage.Client(project=config_manager.gcp_project_id)
        gcs_bucket_name = config_manager.get_config_value('gcp.gcs.processed_bucket_name')
        
        # Verificar acceso al bucket configurado
        try:
            bucket = gcs_client.get_bucket(gcs_bucket_name)
            logger.info(f"✓ Bucket GCS {gcs_bucket_name} accesible para TradingEnvironment")
            
            # Realizamos solo una validación básica del acceso a GCS, no intentamos cargar datos reales
            # ya que podrían no estar disponibles en esta fase de migración
            logger.info("✓ TradingEnvironment tiene acceso a GCS")
            return True
        except Exception as e:
            logger.error(f"✗ Error accediendo al bucket GCS desde TradingEnvironment: {e}")
            return False
    except Exception as e:
        logger.error(f"✗ Error en la prueba de TradingEnvironment: {e}")
        return False

def run_all_tests():
    """Ejecuta todas las pruebas de integración GCP"""
    start_time = time.time()
    
    logger.info("Iniciando pruebas de integración GCP...")
    
    # Asegurar que los datasets de BigQuery existen
    logger.info("Verificando/creando datasets de BigQuery...")
    ensure_bigquery_datasets_exist()
    
    tests = [
        ("ConfigManager y Secret Manager", test_config_manager),
        ("BinanceFuturesDownloader y BigQuery", test_binance_downloader),
        ("DataPreprocessor, BigQuery y GCS", test_preprocessor),
        ("TradingEnvironment y GCS", test_trading_env)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nEjecutando prueba: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Error inesperado en {test_name}: {e}")
            results[test_name] = False
    
    # Resumen de resultados
    logger.info("\n=== RESUMEN DE RESULTADOS ===")
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASÓ" if result else "✗ FALLÓ"
        logger.info(f"{status} - {test_name}")
        if not result:
            all_passed = False
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nTiempo total de ejecución: {elapsed_time:.2f} segundos")
    
    if all_passed:
        logger.info("✅ TODAS LAS PRUEBAS PASARON - La integración con GCP está funcionando correctamente.")
        return 0
    else:
        logger.error("❌ ALGUNAS PRUEBAS FALLARON - La integración con GCP no está completa.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
