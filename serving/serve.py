import os
import json
import numpy as np
import logging
from flask import Flask, request, jsonify
import gunicorn.app.base
from typing import Dict, Any

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.environments.trading_env import TradingEnvironment
from src.utils.logging_utils import setup_logger

# Configurar logging
logger = setup_logger("Server")

# Inicializar Flask app
app = Flask(__name__)

# Variable global para el agente RL
agent_manager = None
config_path = "src/config.yaml"

def load_model():
    """
    Carga el modelo entrenado desde GCS al iniciar el servidor.
    Utiliza la variable de entorno MODEL_PATH para determinar qué modelo cargar.
    """
    global agent_manager
    
    # Obtener ruta del modelo desde variables de entorno
    model_path = os.environ.get('MODEL_PATH')
    if not model_path:
        raise ValueError("La variable de entorno MODEL_PATH no está configurada. Debe especificar la ruta GCS del modelo.")
    
    logger.info(f"Cargando modelo desde: {model_path}")
    
    # Inicializar el administrador del agente
    agent_manager = RLAgentManager(config_path=config_path)
    
    # Crear un entorno de trading para poder cargar el modelo
    env = agent_manager.setup_environment()
    
    # Cargar el modelo
    agent_manager.setup_agent(env=env, load_model=True, model_path=model_path)
    
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
    # Cargar el modelo al iniciar
    load_model()
    
    # Configuración para producción con Gunicorn
    if os.environ.get('ENVIRONMENT') == 'production':
        # Configuración de Gunicorn
        options = {
            'bind': '0.0.0.0:8080',
            'workers': 1,  # Para modelos ML complejos, a menudo se usa solo 1 worker
            'timeout': 120,  # Timeout en segundos
            'preload_app': True,  # Precarga la aplicación para que el modelo se cargue una sola vez
        }
        
        # Iniciar Gunicorn
        logger.info("Iniciando servidor Gunicorn en modo producción")
        StandaloneApplication(app, options).run()
    else:
        # Para desarrollo, usar el servidor de Flask
        logger.info("Iniciando servidor Flask en modo desarrollo")
        app.run(host='0.0.0.0', port=8080, debug=False)