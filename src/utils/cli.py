"""
Funciones de utilidad para el manejo de argumentos de línea de comandos.
"""

import argparse


def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Bot de trading de Bitcoin')
    
    # Los argumentos que definen el experimento se han movido a config.yaml
    # y se cargan directamente en el script de entrenamiento.
    
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
        '--checkpoint',
        type=str,
        default=None,
        help='ID del run (run_id) desde el cual cargar el último checkpoint para continuar el entrenamiento. Si no se proporciona, el entrenamiento comienza desde cero.'
    )
    
    parser.add_argument(
        '--fine-tune-mode',
        action='store_true',
        help='Activa el modo de ajuste fino: carga los pesos del agente pero reinicia los optimizadores para aprender con nuevos hiperparámetros (ej: un learning rate más bajo).'
    )
    
    return parser.parse_args()
