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

        run_config = RunManager.load_run_config(args.run_id, gcp_config=gcp_config_local)
        if not run_config:
            logger.error(f"No se pudo cargar la configuración para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        logger.info("✅ Configuración del run cargada exitosamente.")

        # Extraer símbolo e intervalo desde la configuración del run para consistencia
        try:
            symbol = run_config['command_line_args']['symbol']
            interval = run_config['command_line_args']['interval']
            logger.info(f"  - Símbolo: {symbol}")
            logger.info(f"  - Intervalo: {interval}")
            logger.info(f"  - Modo de operación: {args.mode}")
        except KeyError:
            logger.error(f"El config_run.yaml para '{args.run_id}' no contiene 'symbol' o 'interval' en 'command_line_args'.")
            sys.exit(1)

        # --- 2. Carga de Credenciales desde Variables de Entorno ---
        logger.info("🔐 Cargando credenciales y secretos desde variables de entorno...")
        is_testnet = (args.mode == 'testnet')
        
        # Cargar credenciales de Binance
        api_key_env_var = 'TESTNET_BINANCE_API_KEY_FUTURES' if is_testnet else 'BINANCE_API_KEY_FUTURES'
        api_secret_env_var = 'TESTNET_BINANCE_API_SECRET_FUTURES' if is_testnet else 'BINANCE_API_SECRET_FUTURES'
        
        api_key = os.getenv(api_key_env_var)
        api_secret = os.getenv(api_secret_env_var)

        if not api_key or not api_secret:
            logger.error(f"No se encontraron las variables de entorno para las credenciales de Binance: {api_key_env_var}, {api_secret_env_var}")
            sys.exit(1)
        logger.info(f"✅ Credenciales de Binance {'testnet' if is_testnet else 'producción'} cargadas.")

        # Cargar credenciales de Telegram (opcional)
        telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if telegram_bot_token and telegram_chat_id:
            logger.info("✅ Credenciales de Telegram cargadas.")
        else:
            logger.warning("⚠️  Credenciales de Telegram no encontradas en variables de entorno. Notificaciones desactivadas.")

        # --- 3. Inicialización y Ejecución del Trading Manager ---
        manager = LiveTradingManager(
            run_id=args.run_id,
            symbol=symbol,
            mode=args.mode,
            run_config=run_config, # Inyectar la configuración completa del run
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
