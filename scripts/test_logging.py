#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para probar la configuración de logging.
Este script te permitirá verificar el comportamiento del sistema de logging
simplificado que solo envía logs a la terminal.
"""
import os
import sys
import argparse

# Añadir directorio raíz al PYTHONPATH para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones locales
from src.utils.logging_utils import setup_logger

def test_logging():
    """
    Prueba la configuración de logging.
    """
    print(f"\n{'='*80}")
    print(f"Probando sistema de logging")
    print(f"{'='*80}")
    
    try:
        # Configurar logger
        logger = setup_logger("test_logging")
        
        # Enviar algunos mensajes de prueba
        logger.debug("Este es un mensaje de nivel DEBUG")
        logger.info("Este es un mensaje de nivel INFO")
        logger.warning("Este es un mensaje de nivel WARNING")
        logger.error("Este es un mensaje de nivel ERROR")
        
        print(f"\n✅ Logger configurado correctamente")
        print(f"Los logs deberían estar visibles en la consola")
        
    except Exception as e:
        print(f"\n❌ Error al probar el logger: {str(e)}")

if __name__ == "__main__":
    test_logging()
    
    print("\nPrueba completada. Revisa los mensajes anteriores para verificar el comportamiento.")
