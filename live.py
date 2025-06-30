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
        run_manager = RunManager()
        run_config = run_manager.download_and_load_yaml_config(args.run_id)
        if run_config is None:
            logger.error(f"No se pudo cargar la configuración para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        logger.info("Configuración del run cargada exitosamente.")

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
