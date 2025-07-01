#!/usr/bin/env python3
"""
Script principal para ejecutar el bot de trading en modo LIVE.

Este script inicializa y ejecuta el LiveTradingManager, que orquesta
todos los componentes necesarios para operar en tiempo real en testnet o producción.

Uso:
    python live.py --run-id <ID_DEL_RUN_ENTRENADO> [--mode {testnet,live}]
    
    Ejemplos:
    python live.py --run-id BTCUSDT_1h_12345_20250630 --mode testnet
    python live.py --run-id BTCUSDT_1h_12345_20250630 --mode live
"""

import argparse
import sys
import logging
from src.live.trading_manager import LiveTradingManager
from src.utils.system import setup_logging
from src.training.run_manager import RunManager # Importar RunManager

def parse_arguments():
    """Parsea los argumentos de línea de comandos para el modo live."""
    parser = argparse.ArgumentParser(
        description="Ejecuta el bot de trading en modo LIVE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="El ID del entrenamiento cuyo modelo se va a utilizar. El script extraerá el símbolo y la temporalidad de este ID."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['testnet', 'live'],
        default='testnet',
        help="El modo de operación: 'testnet' para operar con dinero de prueba, 'live' para operar con dinero real."
    )
    return parser.parse_args()

def main():
    """Función principal que lanza el bot en modo live."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_arguments()
        
        # Parsear el run_id para extraer parámetros
        try:
            parts = args.run_id.split('_')
            symbol = parts[0]
            interval = parts[1]
            
            logger.info("Parámetros extraídos del run_id:")
            logger.info(f"  - Símbolo: {symbol}")
            logger.info(f"  - Intervalo: {interval}")
            logger.info(f"  - Modo de operación: {args.mode}")

        except IndexError:
            logger.error(f"El formato del run-id '{args.run_id}' no es válido. Debe ser 'SYMBOL_INTERVAL_SEED_TIMESTAMP'.")
            sys.exit(1)

        logger.info(f"🚀 Lanzando bot en modo LIVE para el run-id: {args.run_id}")

        # Cargar la configuración específica del run como única fuente de verdad
        logger.info(f"Cargando configuración para el run_id: {args.run_id}...")
        
        # Step 1: Load the run configuration using the static method
        run_config = RunManager.load_run_config(args.run_id)
        if run_config is None:
            logger.error(f"No se pudo cargar la configuración para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        logger.info("Configuración del run cargada exitosamente.")
        
        # Step 2: Extract storage configuration from the loaded config
        main_config = run_config.get('config', {})
        if not main_config:
            logger.error(f"No se encontró la configuración principal en 'config' para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        
        # Extract storage mode and GCS configuration
        storage_mode = main_config.get('normalization', {}).get('storage_mode', 'local')
        gcs_bucket_name = None
        gcs_utils = None
        
        if storage_mode == "gcp":
            gcs_bucket_name = main_config.get('gcp', {}).get('storage', {}).get('bucket_name')
            if not gcs_bucket_name:
                logger.error("GCS bucket name not found in configuration but storage_mode is 'gcp'")
                sys.exit(1)
            
            # Initialize GCS utils for GCP mode
            from src.configuration.gcs_utils import gcs_utils
            logger.info("Usando instancia global de GCSUtils para modo GCP")
        
        # Step 3: Create the definitive RunManager with proper configuration
        run_manager = RunManager(
            run_id=args.run_id,
            storage_mode=storage_mode,
            gcs_bucket_name=gcs_bucket_name,
            gcs_utils=gcs_utils
        )

        # Crear e iniciar el gestor de trading, inyectando la configuración
        manager = LiveTradingManager(
            run_id=args.run_id,
            symbol=symbol,
            mode=args.mode,
            run_config=run_config # Inyección de la configuración
        )
        manager.run()

    except KeyboardInterrupt:
        logger.info("\n🛑 Bot detenido manualmente por el usuario. Cerrando...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error catastrófico en el bot: {e}")
        logger.exception("Detalles del error:")
        sys.exit(1)

if __name__ == "__main__":
    main()
