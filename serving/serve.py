import os
import sys
import json
import numpy as np
import logging
import argparse
from flask import Flask, request, jsonify
import gunicorn.app.base
from typing import Dict, Any

# Añadir directorio raíz al PYTHONPATH para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.utils.logging_utils import setup_logger
from src.utils.config import ConfigManager
from src.utils.inference_utils import InferenceOptimizer

# Configurar logging
logger = setup_logger("Server")

# Inicializar Flask app
app = Flask(__name__)

# Variable global para el agente RL
agent_manager = None
config_path = "src/config.yaml"

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Servidor de inferencia para el modelo RL de trading")
    parser.add_argument(
        "--model_path",
        type=str,
        default="lofty-complex-460416-r6/models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip",
        help="Ruta completa en GCS donde se encuentra el modelo entrenado (gs://bucket/path/to/model)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/config.yaml",
        help="Ruta al archivo de configuración (predeterminado: src/config.yaml)"
    )
    return parser.parse_args()

def load_model(model_path, config_path):
    """
    Carga el modelo entrenado desde GCS al iniciar el servidor,
    utilizando optimizaciones para inferencia.
    
    Args:
        model_path (str): Ruta completa en GCS donde se encuentra el modelo
        config_path (str): Ruta al archivo de configuración
    """
    global agent_manager
    
    if not model_path:
        raise ValueError("Se debe proporcionar la ruta del modelo (--model_path)")
    
    logger.info(f"Cargando modelo desde: {model_path}")
    
    # Cargar el modelo optimizado para inferencia
    agent_manager = InferenceOptimizer.load_model_for_inference(model_path, config_path)
    
    # Log detallado de carga exitosa
    if model_path.startswith('gs://'):
        logger.info(f"Modelo cargado exitosamente desde Google Cloud Storage: {model_path}")
    else:
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
        
    logger.info("Modelo cargado exitosamente")
    return agent_manager

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar el estado del servidor.
    Vertex AI lo utiliza para comprobar que el servidor esté funcionando.
    """
    if agent_manager is None:
        return jsonify({"status": "error", "message": "El modelo no está cargado"}), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/ping', methods=['GET'])
def ping():
    """
    Endpoint alternativo para verificar el estado del servidor.
    """
    return health_check()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones con el modelo entrenado.
    Acepta datos del mercado y del portafolio en formato JSON y devuelve
    la acción predicha por el agente.
    """
    if agent_manager is None:
        return jsonify({"error": "El modelo no está cargado"}), 500
    
    # Obtener datos de la solicitud
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No se recibieron datos en la solicitud"}), 400
        
        # Verificar que se proporcionen las características necesarias
        required_keys = ['market_features', 'portfolio_features']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Falta la información de '{key}' en los datos"}), 400
        
        # Convertir los datos de entrada a arrays numpy
        observation = {}
        
        # Procesar características del mercado
        market_features = np.array(data['market_features'], dtype=np.float32)
        observation['market_features'] = market_features
        
        # Procesar características del portafolio
        portfolio_features = np.array(data['portfolio_features'], dtype=np.float32)
        observation['portfolio_features'] = portfolio_features
        
        # Log para diagnóstico de formas
        logger.info(f"Servidor recibió market_features con forma: {market_features.shape}")
        logger.info(f"Servidor recibió portfolio_features con forma: {portfolio_features.shape}")
        
        # Obtener la predicción del modelo
        action = agent_manager.predict_action(observation, deterministic=True)
        
        # Convertir la acción a un tipo serializable
        action_list = action.tolist()
        
        return jsonify({
            "action": action_list,
            "action_value": float(action[0])  # Extraer el valor numérico para mayor claridad
        })
        
    except Exception as e:
        logger.error(f"Error al procesar la predicción: {str(e)}")
        return jsonify({"error": f"Error en el servidor: {str(e)}"}), 500

class StandaloneApplication(gunicorn.app.base.BaseApplication):
    """
    Clase para ejecutar Gunicorn programáticamente.
    """
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == "__main__":
    # Parsear argumentos de línea de comandos
    args = parse_arguments()
    
    # Cargar el modelo al iniciar
    load_model(model_path=args.model_path, config_path=args.config_path)
    
    # Siempre usar Gunicorn para producción y consistencia
    options = {
        'bind': '0.0.0.0:8080',
        'workers': 1,  # Para modelos ML complejos, a menudo se usa solo 1 worker
        'timeout': 120,  # Timeout en segundos
        'preload_app': True,  # Precarga la aplicación para que el modelo se cargue una sola vez
    }
    
    # Iniciar Gunicorn
    logger.info("Iniciando servidor Gunicorn")
    StandaloneApplication(app, options).run()