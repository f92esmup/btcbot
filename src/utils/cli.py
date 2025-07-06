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
    
    # Grupo de hiperparámetros del agente (opcional)
    hyperparams_group = parser.add_argument_group('Hiperparámetros del Agente (Opcional)')
    
    hyperparams_group.add_argument(
        '--actor-learning-rate',
        type=float,
        default=None,
        help='Learning rate del actor. Sobrescribe el valor de config.yaml.'
    )
    
    hyperparams_group.add_argument(
        '--critic-learning-rate',
        type=float,
        default=None,
        help='Learning rate de los críticos. Sobrescribe el valor de config.yaml.'
    )
    
    hyperparams_group.add_argument(
        '--alpha-learning-rate',
        type=float,
        default=None,
        help='Learning rate de alpha. Sobrescribe el valor de config.yaml.'
    )
    
    hyperparams_group.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Tamaño del batch. Sobrescribe el valor de config.yaml.'
    )
    
    hyperparams_group.add_argument(
        '--tau',
        type=float,
        default=None,
        help='Factor de actualización suave (soft update). Sobrescribe el valor de config.yaml.'
    )
    
    hyperparams_group.add_argument(
        '--per-alpha',
        type=float,
        default=None,
        help='Exponente de prioridad para PER. Sobrescribe el valor de config.yaml.'
    )
    
    hyperparams_group.add_argument(
        '--per-beta',
        type=float,
        default=None,
        help='Exponente de muestreo por importancia para PER. Sobrescribe el valor de config.yaml.'
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


def parse_evaluation_arguments():
    """Parsea los argumentos específicos para la evaluación de modelos."""
    parser = argparse.ArgumentParser(
        description='Script para evaluación de modelos entrenados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --run-id training_run_20250703_120000
  %(prog)s --run-id training_run_20250703_120000 --model-type final_model
        """
    )
    
    # Argumentos requeridos
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='ID del training_run del modelo que se desea evaluar'
    )
    
    # Argumentos opcionales
    parser.add_argument(
        '--model-type',
        type=str,
        default='best_model',
        choices=['best_model', 'final_model'],
        help='Tipo de modelo a evaluar (default: best_model)'
    )
    
    return parser.parse_args()


def parse_hypertune_arguments():
    """Parsea los argumentos específicos para trials de Vertex AI Hypertune."""
    parser = argparse.ArgumentParser(
        description='Script para trials de optimización de hiperparámetros con Vertex AI Hypertune',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --train-data-run-id BTCUSDT_1h_bloque_1 --eval-data-run-id BTCUSDT_1h_bloque_2 --episodes 300 --actor-learning-rate 0.0003 --critic-learning-rate 0.0003 --alpha-learning-rate 0.0003 --batch-size 512 --tau 0.005 --per-alpha 0.6 --per-beta 0.4
        """
    )
    
    # Argumentos requeridos para datos
    parser.add_argument(
        '--train-data-run-id',
        type=str,
        required=True,
        help='ID del data_run para el entrenamiento del trial'
    )
    
    parser.add_argument(
        '--eval-data-run-id',
        type=str,
        required=True,
        help='ID del data_run para la evaluación del trial'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        required=True,
        help='Número de episodios para el entrenamiento del trial'
    )
    
    # Hiperparámetros requeridos (serán proporcionados por Vertex AI)
    parser.add_argument(
        '--actor-learning-rate',
        type=float,
        required=True,
        help='Learning rate del actor'
    )
    
    parser.add_argument(
        '--critic-learning-rate',
        type=float,
        required=True,
        help='Learning rate de los críticos'
    )
    
    parser.add_argument(
        '--alpha-learning-rate',
        type=float,
        required=True,
        help='Learning rate de alpha'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        required=True,
        help='Tamaño del batch'
    )
    
    parser.add_argument(
        '--tau',
        type=float,
        required=True,
        help='Factor de actualización suave (soft update)'
    )
    
    parser.add_argument(
        '--per-alpha',
        type=float,
        required=True,
        help='Exponente de prioridad para PER'
    )
    
    parser.add_argument(
        '--per-beta',
        type=float,
        required=True,
        help='Exponente de muestreo por importancia para PER'
    )
    
    return parser.parse_args()
