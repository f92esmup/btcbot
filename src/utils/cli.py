"""
Funciones de utilidad para el manejo de argumentos de línea de comandos.
"""

import argparse


def parse_arguments():
    """Parsea los argumentos de línea de comandos para el entrenamiento."""
    parser = argparse.ArgumentParser(
        description='Bot de trading de Bitcoin - Script de entrenamiento',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de operación:
  --data-run-id DATA_RUN_ID    Iniciar nuevo entrenamiento desde un dataset específico
  --checkpoint TRAINING_RUN_ID Reanudar entrenamiento desde un checkpoint existente

Ejemplos:
  %(prog)s --data-run-id BTCUSDT_1h_20250101_20250630-143022 --episodes 1000
  %(prog)s --checkpoint training_run_20250703_120000 --episodes 500
        """
    )
    
    # Grupo mutuamente excluyente para modo de inicio
    start_mode = parser.add_mutually_exclusive_group(required=True)
    
    start_mode.add_argument(
        '--data-run-id',
        type=str,
        help='ID del data_run desde el cual iniciar un nuevo entrenamiento. '
             'Debe corresponder a un dataset creado previamente con create_dataset.py'
    )
    
    start_mode.add_argument(
        '--checkpoint',
        type=str,
        help='ID del training_run desde el cual reanudar el entrenamiento. '
             'Carga el último checkpoint disponible del entrenamiento especificado.'
    )
    
    # Argumentos opcionales para entrenamiento
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Número de episodios de entrenamiento (default: 1000)'
    )
    
    parser.add_argument(
        '--eval-frequency',
        type=int,
        default=50,
        help='Frecuencia de evaluación en episodios (default: 50)'
    )
    
    parser.add_argument(
        '--save-frequency',
        type=int,
        default=100,
        help='Frecuencia de guardado en episodios (default: 100)'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Deshabilitar CUDA aunque esté disponible'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=5,
        help='Número de episodios para evaluación (default: 5)'
    )
    
    parser.add_argument(
        '--fine-tune-mode',
        action='store_true',
        help='Activa el modo de ajuste fino: carga los pesos del agente pero reinicia los optimizadores para aprender con nuevos hiperparámetros (ej: un learning rate más bajo).'
    )
    
    return parser.parse_args()


def parse_dataset_arguments():
    """Parsea los argumentos específicos para la creación de datasets."""
    parser = argparse.ArgumentParser(
        description='Script dedicado para la creación de datasets inmutables y versionados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --symbol BTCUSDT --interval 1m --start-date 2025-01-01
  %(prog)s --symbol BTCUSDT --interval 1m --start-date 2025-01-01 --end-date 2025-06-30
  %(prog)s --symbol ETHUSDT --interval 4h --start-date 2024-01-01
        """
    )
    
    # Argumentos requeridos
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Símbolo del par de trading (ej: BTCUSDT, ETHUSDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        help='Intervalo de tiempo para las velas (ej: 1m, 5m, 15m, 1h, 4h, 1d)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Fecha de inicio en formato YYYY-MM-DD (ej: 2025-01-01)'
    )
    
    # Argumentos opcionales
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Fecha de fin en formato YYYY-MM-DD (opcional, si no se especifica se usa hasta ahora)'
    )
    
    return parser.parse_args()
