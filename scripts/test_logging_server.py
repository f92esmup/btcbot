#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para probar el comportamiento de logging en un servidor web simple.
Simula el comportamiento de un servidor de inferencia para verificar
cómo se manejan los logs en un entorno similar a producción.
"""
import os
import sys
import argparse
import time
from flask import Flask, jsonify, request

# Añadir directorio raíz al PYTHONPATH para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones locales
from src.utils.logging_utils import setup_logger

# Crear app Flask
app = Flask(__name__)

# Configurar logger
logger = None

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar el estado del servidor.
    """
    logger.info("Health check solicitado")
    return jsonify({"status": "healthy"}), 200

@app.route('/log', methods=['POST'])
def log_message():
    """
    Endpoint para enviar mensajes de log.
    """
    try:
        data = request.json
        if not data or 'message' not in data or 'level' not in data:
            logger.warning("Solicitud de log inválida")
            return jsonify({"error": "Se requiere 'message' y 'level' en el JSON"}), 400
        
        message = data['message']
        level = data['level'].lower()
        
        if level == 'debug':
            logger.debug(message)
        elif level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)
        else:
            logger.warning(f"Nivel de log no reconocido: {level}")
            return jsonify({"error": f"Nivel de log no reconocido: {level}"}), 400
        
        logger.info(f"Mensaje de log procesado con nivel {level}")
        return jsonify({"success": True}), 200
        
    except Exception as e:
        logger.error(f"Error al procesar la solicitud: {str(e)}")
        return jsonify({"error": f"Error en el servidor: {str(e)}"}), 500

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Servidor de prueba para logging")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['auto', 'enabled', 'disabled'],
        default='auto',
        help="Modo de logging a utilizar ('auto', 'enabled', 'disabled')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto en el que escuchará el servidor"
    )
    return parser.parse_args()

def main():
    """
    Función principal.
    """
    args = parse_arguments()
    
    # Establecer el modo de Cloud Logging
    os.environ['CLOUD_LOGGING_MODE'] = args.mode
    
    # Configurar logger global
    global logger
    logger = setup_logger("test_server")
    
    # Mensaje inicial
    logger.info(f"Iniciando servidor de prueba con modo de logging: {args.mode}")
    logger.info(f"Escuchando en http://localhost:{args.port}")
    logger.info("Endpoints disponibles:")
    logger.info(f"  - GET http://localhost:{args.port}/health")
    logger.info(f"  - POST http://localhost:{args.port}/log")
    logger.info("Para probar el endpoint /log:")
    # Usamos dobles llaves para escapar las llaves en un f-string
    logger.info(f'  curl -X POST http://localhost:{args.port}/log -H "Content-Type: application/json" -d \'{{"message": "Mensaje de prueba", "level": "info"}}\'')
    
    # Iniciar servidor Flask
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == "__main__":
    main()
