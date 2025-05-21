#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para probar la configuración de logging en diferentes modos.
Este script te permitirá verificar el comportamiento del sistema de logging
con diferentes valores de la variable CLOUD_LOGGING_MODE.
"""
import os
import sys
import argparse

# Añadir directorio raíz al PYTHONPATH para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones locales
from src.utils.logging_utils import setup_logger

def test_logging_mode(mode):
    """
    Prueba la configuración de logging con un modo específico.
    
    Args:
        mode (str): Modo de logging a probar ('auto', 'enabled', 'disabled')
    """
    # Establecer el modo de Cloud Logging
    os.environ['CLOUD_LOGGING_MODE'] = mode
    
    print(f"\n{'='*80}")
    print(f"Probando modo de logging: {mode}")
    print(f"{'='*80}")
    
    try:
        # Configurar logger
        logger = setup_logger(f"test_logging_{mode}")
        
        # Enviar algunos mensajes de prueba
        logger.debug("Este es un mensaje de nivel DEBUG")
        logger.info("Este es un mensaje de nivel INFO")
        logger.warning("Este es un mensaje de nivel WARNING")
        logger.error("Este es un mensaje de nivel ERROR")
        
        print(f"\n✅ Modo '{mode}' configurado correctamente")
        print(f"Los logs deberían estar visibles en la consola{' y en Google Cloud Logging' if mode != 'disabled' else ''}")
        
    except Exception as e:
        print(f"\n❌ Error al probar el modo '{mode}': {str(e)}")
        if mode == 'enabled':
            print("Este error es esperado si no tienes permisos de Cloud Logging y usas el modo 'enabled'")

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Prueba la configuración de logging")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['auto', 'enabled', 'disabled', 'all'],
        default='all',
        help="Modo de logging a probar ('auto', 'enabled', 'disabled' o 'all' para probar todos)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.mode == 'all':
        # Probar todos los modos
        for mode in ['auto', 'enabled', 'disabled']:
            test_logging_mode(mode)
    else:
        # Probar solo el modo especificado
        test_logging_mode(args.mode)
    
    print("\nPrueba completada. Revisa los mensajes anteriores para verificar el comportamiento.")
