"""
Script principal de entrenamiento del bot de trading de Bitcoin.
Orquesta la adquisición de datos, cálculo de indicadores y entrenamiento del modelo.
"""

import argparse
import sys
import logging
from datetime import datetime
from src.data.Adquisicion import Adquisicion
from src.data.indicadores import Indicadores


def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            #logging.FileHandler('trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Bot de trading de Bitcoin')
    
    # Argumentos requeridos
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Símbolo del par de trading (ej: BTCUSDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Intervalo de tiempo para las velas'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Fecha de inicio en formato YYYY-MM-DD'
    )
    
    return parser.parse_args()


def validate_start_date(date_string: str) -> bool:
    """
    Valida que la fecha de inicio tenga el formato correcto.
    
    Args:
        date_string (str): Fecha en formato YYYY-MM-DD
        
    Returns:
        bool: True si es válida, False en caso contrario
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def main():
    """Función principal del script."""
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== Iniciando Bot de Trading de Bitcoin ===")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Validar fecha de inicio
    if not validate_start_date(args.start_date):
        logger.error(f"Formato de fecha inválido: {args.start_date}. Use YYYY-MM-DD")
        sys.exit(1)
    
    logger.info(f"Parámetros: Symbol={args.symbol}, Interval={args.interval}, Start Date={args.start_date}")
    
    try:
        # 1. Adquisición de datos
        logger.info("=== FASE 1: Adquisición de Datos ===")
        adquisicion = Adquisicion(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date
        )
        
        # Ejecutar proceso de adquisición
        dataframe = adquisicion.main()

        logger.info(f"Datos adquiridos exitosamente:")
        logger.info(f"  - Forma del DataFrame: {dataframe.shape}")
        logger.info(f"  - Rango temporal: {dataframe.index.min()} a {dataframe.index.max()}")
        logger.info(f"  - Columnas: {list(dataframe.columns)}")
        logger.info(f"  - Memoria utilizada: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar estadísticas básicas
        logger.info("Estadísticas básicas del DataFrame:")
        logger.info(f"\n{dataframe.describe()}")
        
        # 2. Cálculo de Indicadores Técnicos
        logger.info("=== FASE 2: Cálculo de Indicadores Técnicos ===")
        indicadores = Indicadores(dataframe)
        
        # Ejecutar proceso de cálculo de indicadores
        dataframe_with_indicators = indicadores.main()
        
        logger.info(f"Indicadores calculados exitosamente:")
        logger.info(f"  - Forma del DataFrame: {dataframe_with_indicators.shape}")
        logger.info(f"  - Columnas totales: {len(dataframe_with_indicators.columns)}")
        logger.info(f"  - Nuevas columnas de indicadores: {len(dataframe_with_indicators.columns) - len(dataframe.columns)}")
        logger.info(f"  - Memoria utilizada: {dataframe_with_indicators.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar las nuevas columnas
        original_columns = set(dataframe.columns)
        new_columns = [col for col in dataframe_with_indicators.columns if col not in original_columns]
        if new_columns:
            logger.info(f"  - Indicadores añadidos: {new_columns}")
        
        # Actualizar referencia al dataframe
        dataframe = dataframe_with_indicators
        
        # 3. TODO: Aquí se agregará el entrenamiento del modelo
        logger.info("=== FASE 3: Entrenamiento del Modelo (Pendiente) ===")
        logger.info("Esta fase se implementará más adelante...")
        
        logger.info("=== Proceso Completado Exitosamente ===")
        
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        logger.exception("Detalles del error:")
        sys.exit(1)


if __name__ == "__main__":
    main()