#!/usr/bin/env python
"""
Script para descargar datos históricos de Binance Futures directamente a GCS.
Optimizado para ejecutarse como un servicio Cloud Run en GCP.
"""
import argparse
import logging
import os
from datetime import datetime, timedelta

from src.data.binance_futures_downloader_cloud import BinanceFuturesDownloaderCloud

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para descargar datos históricos."""
    parser = argparse.ArgumentParser(description="Descarga datos históricos de Binance Futures a GCS")
    
    # Parámetros básicos
    parser.add_argument("--symbol", type=str, default=os.getenv("DEFAULT_SYMBOL", "BTCUSDT"),
                        help="Símbolo a descargar (default: desde variable DEFAULT_SYMBOL o BTCUSDT)")
    parser.add_argument("--interval", type=str, default=os.getenv("DEFAULT_INTERVAL", "1h"),
                        help="Intervalo de velas (default: desde variable DEFAULT_INTERVAL o 1h)")
    parser.add_argument("--start-date", type=str, default=os.getenv("DEFAULT_HISTORICAL_START_DATE", "2020-01-01"),
                        help="Fecha de inicio (YYYY-MM-DD) (default: desde variable DEFAULT_HISTORICAL_START_DATE o 2020-01-01)")
    
    # Parámetros de GCP
    parser.add_argument("--project-id", type=str, required=False,
                        default=os.getenv("GCP_PROJECT_ID"),
                        help="ID del proyecto GCP (default: desde variable GCP_PROJECT_ID)")
    parser.add_argument("--api-key-secret-name", type=str, required=False,
                        default=os.getenv("BINANCE_API_KEY_SECRET_NAME", "binance-api-key"),
                        help="Nombre del secreto que contiene la API key (default: desde variable BINANCE_API_KEY_SECRET_NAME)")
    parser.add_argument("--api-secret-secret-name", type=str, required=False,
                        default=os.getenv("BINANCE_API_SECRET_SECRET_NAME", "binance-api-secret"),
                        help="Nombre del secreto que contiene el API secret (default: desde variable BINANCE_API_SECRET_SECRET_NAME)")
    parser.add_argument("--raw-data-bucket", type=str, required=False,
                        default=os.getenv("RAW_DATA_BUCKET"),
                        help="Nombre del bucket para datos crudos (default: desde variable RAW_DATA_BUCKET)")
    parser.add_argument("--output-gcs-prefix", type=str, required=False,
                        default=None,
                        help="Prefijo opcional para la ruta de salida en GCS (default: None)")
    
    args = parser.parse_args()
    
    # Verificar parámetros obligatorios
    if not args.project_id:
        raise ValueError("Se requiere --project-id o la variable de entorno GCP_PROJECT_ID")
    if not args.raw_data_bucket:
        raise ValueError("Se requiere --raw-data-bucket o la variable de entorno RAW_DATA_BUCKET")
    
    try:
        logger.info(f"Iniciando descarga para {args.symbol} ({args.interval}) desde {args.start_date}")
        
        # Inicializar el descargador
        downloader = BinanceFuturesDownloaderCloud(
            project_id=args.project_id,
            api_key_secret_name=args.api_key_secret_name,
            api_secret_secret_name=args.api_secret_secret_name,
            raw_data_bucket=args.raw_data_bucket
        )
        
        # Descargar los datos
        output_gcs_uri = downloader.fetch_historical_data(
            symbol=args.symbol,
            interval=args.interval,
            start_date_str=args.start_date,
            output_gcs_prefix=args.output_gcs_prefix
        )
        
        if output_gcs_uri:
            logger.info(f"Descarga completada exitosamente. Datos guardados en: {output_gcs_uri}")
            # En un entorno de Cloud Run, podríamos devolver esto como respuesta
            print(f"SUCCESS:{output_gcs_uri}")
            return 0
        else:
            logger.error("Error en la descarga de datos")
            return 1
            
    except Exception as e:
        logger.exception(f"Error no controlado: {e}")
        return 1

if __name__ == "__main__":
    exit(main())