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
import os
import yaml
from src.live.trading_manager import LiveTradingManager
from src.utils.system import setup_logging
from src.training.run_manager import RunManager
from src.configuration.secret_utils import SecretManagerUtils

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
        help="El ID del entrenamiento cuyo modelo se va a utilizar."
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
        logger.info(f"🚀 Lanzando bot en modo LIVE para el run-id: {args.run_id}")

        # --- 1. Carga de Configuración del Run como Única Fuente de Verdad ---
        logger.info(f"Cargando configuración para el run_id: {args.run_id}...")
        
        # Cargar la configuración local solo para obtener los detalles de GCP
        try:
            with open('src/configuration/config.yaml', 'r') as f:
                local_config = yaml.safe_load(f)
            gcp_config_local = local_config.get('gcp')
        except FileNotFoundError:
            logger.warning("No se encontró config.yaml local. Se asumirá que no se necesita gcp_config.")
            gcp_config_local = None

        run_config = RunManager.load_training_run_config(args.run_id, gcp_config=gcp_config_local)
        if not run_config:
            logger.error(f"No se pudo cargar la configuración para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        logger.info("✅ Configuración del run cargada exitosamente.")

        # Extraer data_run_id desde la configuración del run
        try:
            data_run_id = run_config['lineage']['data_run_id']
            logger.info(f"  - Data Run ID: {data_run_id}")
            logger.info(f"  - Modo de operación: {args.mode}")
        except KeyError:
            logger.error(f"El config_run.yaml para '{args.run_id}' no contiene 'lineage.data_run_id'.")
            sys.exit(1)

        # --- 2. Carga de Credenciales desde Google Secret Manager ---
        logger.info("🔐 Cargando credenciales y secretos desde Google Secret Manager...")
        is_testnet = (args.mode == 'testnet')
        
        gcp_config = run_config.get('config', {}).get('gcp', {})
        project_id = gcp_config.get('project_id')
        if not project_id:
            logger.error("No se encontró 'project_id' en la configuración de GCP. Abortando.")
            sys.exit(1)
            
        secret_manager = SecretManagerUtils(project_id=project_id)
        secrets_config = run_config.get('config', {}).get('gcp', {}).get('secrets', {})

        try:
            if is_testnet:
                api_key_secret_id = secrets_config['testnet_binance_api_key_futures']
                api_secret_secret_id = secrets_config['testnet_binance_api_secret_futures']
            else:
                api_key_secret_id = secrets_config['binance_api_key_futures']
                api_secret_secret_id = secrets_config['binance_api_secret_futures']

            api_key = secret_manager.get_secret(api_key_secret_id)
            api_secret = secret_manager.get_secret(api_secret_secret_id)
            logger.info(f"✅ Credenciales de Binance {'testnet' if is_testnet else 'producción'} cargadas.")

            telegram_bot_token = secret_manager.get_secret(secrets_config['telegram_bot_token'])
            telegram_chat_id = secret_manager.get_secret(secrets_config['telegram_chat_id'])
            logger.info("✅ Credenciales de Telegram cargadas.")

        except (KeyError, RuntimeError) as e:
            logger.error(f"Error al cargar secretos: {e}")
            sys.exit(1)


        # --- 3. Inicialización y Ejecución del Trading Manager ---
        manager = LiveTradingManager(
            run_id=args.run_id,
            mode=args.mode,
            run_config=run_config, # Inyectar la configuración completa del run
            data_run_id=data_run_id,  # Pasar el data_run_id extraído
            api_key=api_key,
            api_secret=api_secret,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id
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
