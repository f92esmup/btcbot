#!/usr/bin/env python3
"""
Script principal para ejecutar el bot de trading en modo LIVE.

Este script inicializa y ejecuta el LiveTradingManager, que orquesta
todos los componentes necesarios para operar en tiempo real en testnet o producción.

Uso:
    python live.py --run-id <ID_DEL_RUN_ENTRENADO>
"""

import argparse
import sys
import logging
from src.live.trading_manager import LiveTradingManager
from src.utils.system import setup_logging

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
            # Podríamos extraer también la semilla y la fecha si fuera necesario
            
            logger.info("Parámetros extraídos del run_id:")
            logger.info(f"  - Símbolo: {symbol}")
            logger.info(f"  - Intervalo: {interval}")

        except IndexError:
            logger.error(f"El formato del run-id '{args.run_id}' no es válido. Debe ser 'SYMBOL_INTERVAL_SEED_TIMESTAMP'.")
            sys.exit(1)

        logger.info(f"🚀 Lanzando bot en modo LIVE para el run-id: {args.run_id}")

        # Crear e iniciar el gestor de trading
        manager = LiveTradingManager(
            run_id=args.run_id,
            symbol=symbol
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
