"""
Servidor de predicción para el modelo de trading en GCP.
Compatible con Vertex AI Prediction y Cloud Run.
"""
import os
import logging
import json
import numpy as np
import torch
from flask import Flask, request, jsonify
import tempfile
from google.cloud import storage
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('prediction_server')

# Inicializar app Flask
app = Flask(__name__)

# Variables globales para el modelo
MODEL = None
MODEL_METADATA = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("AIP_MODEL_DIR", "/tmp/model")

# Clase para cargar y ejecutar el modelo
class TradingModel:
    """Clase para cargar y ejecutar el modelo de trading."""
    
    def __init__(self, model_path, device="cpu"):
        """
        Inicializa el modelo de trading.
        
        Args:
            model_path: Ruta al modelo guardado
            device: Dispositivo para inferencia ('cpu' o 'cuda')
        """
        self.device = device
        self.model_path = model_path
        self.actor = None
        self.metadata = None
        
        # Cargar modelo y metadatos
        self._load_model()
        
    def _load_model(self):
        """Carga el modelo desde archivos locales o GCS."""
        # Determinar si la ruta es una URI de GCS
        if self.model_path.startswith("gs://"):
            logger.info(f"Cargando modelo desde GCS: {self.model_path}")
            self._load_from_gcs(self.model_path)
        else:
            logger.info(f"Cargando modelo desde archivo local: {self.model_path}")
            try:
                # Buscar archivos actor.pt y metadata.json
                if os.path.exists(os.path.join(self.model_path, "actor.pt")):
                    actor_path = os.path.join(self.model_path, "actor.pt")
                    self.actor = torch.load(actor_path, map_location=self.device)
                
                if os.path.exists(os.path.join(self.model_path, "metadata.json")):
                    with open(os.path.join(self.model_path, "metadata.json"), 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Metadatos del modelo cargados: {self.metadata}")
            except Exception as e:
                logger.error(f"Error cargando modelo local: {e}")
                raise
    
    def _load_from_gcs(self, gcs_path):
        """Carga el modelo desde Google Cloud Storage."""
        try:
            # Extraer bucket y blob
            gcs_path = gcs_path.replace("gs://", "")
            bucket_name = gcs_path.split("/")[0]
            prefix = "/".join(gcs_path.split("/")[1:])
            
            # Inicializar cliente GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Crear carpeta temporal
            temp_dir = tempfile.mkdtemp()
            
            # Descargar archivos
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                if blob.name.endswith("actor.pt"):
                    local_path = os.path.join(temp_dir, "actor.pt")
                    blob.download_to_filename(local_path)
                    logger.info(f"Modelo descargado a: {local_path}")
                    self.actor = torch.load(local_path, map_location=self.device)
                    
                elif blob.name.endswith("metadata.json"):
                    local_path = os.path.join(temp_dir, "metadata.json")
                    blob.download_to_filename(local_path)
                    with open(local_path, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Metadatos del modelo cargados: {self.metadata}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo desde GCS: {e}")
            raise
    
    def predict(self, market_features, portfolio_features):
        """
        Realiza una predicción con el modelo.
        
        Args:
            market_features: Tensor con características del mercado (secuencia)
            portfolio_features: Tensor con características de la cartera
            
        Returns:
            Acción del agente (posición objetivo)
        """
        if self.actor is None:
            raise ValueError("El modelo no ha sido cargado correctamente")
        
        with torch.no_grad():
            # Convertir a tensor PyTorch
            if not isinstance(market_features, torch.Tensor):
                market_features = torch.tensor(market_features, dtype=torch.float32).to(self.device)
            
            if not isinstance(portfolio_features, torch.Tensor):
                portfolio_features = torch.tensor(portfolio_features, dtype=torch.float32).to(self.device)
            
            # Añadir dimensión de batch si es necesario
            if len(market_features.shape) == 2:
                market_features = market_features.unsqueeze(0)
            
            if len(portfolio_features.shape) == 1:
                portfolio_features = portfolio_features.unsqueeze(0)
            
            # Pasar por el actor para obtener la acción determinista
            action = self.actor({"market_features": market_features, "portfolio_features": portfolio_features})
            
            # Convertir a numpy y recortar a rango válido [-1, 1]
            action_np = action.cpu().numpy()
            action_np = np.clip(action_np, -1.0, 1.0)
            
            return action_np[0]  # Retornar solo la primera acción (sin batch)

# Carga el modelo al iniciar el servidor
@app.before_first_request
def load_model():
    global MODEL, MODEL_METADATA
    
    # Verificar si el modelo ya está cargado
    if MODEL is not None:
        return
    
    try:
        # Intentar cargar el modelo
        model_path = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
        logger.info(f"Cargando modelo desde: {model_path}")
        
        # Cargar desde Vertex AI Model Registry o GCS
        if os.environ.get("AIP_STORAGE_URI"):
            model_path = os.environ.get("AIP_STORAGE_URI")
        
        # Instanciar el modelo
        MODEL = TradingModel(model_path, device=DEVICE)
        MODEL_METADATA = MODEL.metadata
        
        logger.info(f"Modelo cargado exitosamente en dispositivo: {DEVICE}")
        
    except Exception as e:
        logger.error(f"Error cargando el modelo: {e}")
        # En producción, no queremos que el servidor falle, solo loguear el error

# Endpoint de salud
@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar el estado del servidor."""
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

# Endpoint para metadata del modelo
@app.route('/metadata', methods=['GET'])
def metadata():
    """Endpoint para obtener metadatos del modelo."""
    if MODEL is None:
        load_model()
    
    return jsonify({
        "model_metadata": MODEL_METADATA,
        "server_info": {
            "device": DEVICE,
            "timestamp": datetime.now().isoformat()
        }
    })

# Endpoint principal de predicción
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    if MODEL is None:
        load_model()
    
    # Obtener datos de la solicitud
    request_json = request.get_json()
    
    if not request_json:
        return jsonify({"error": "No se proporcionaron datos de entrada"}), 400
    
    try:
        # Extraer características
        market_features = np.array(request_json.get("market_features"), dtype=np.float32)
        portfolio_features = np.array(request_json.get("portfolio_features"), dtype=np.float32)
        
        # Validar formato de entrada
        expected_market_shape = tuple(MODEL_METADATA.get("input_dim", {}).get("market_features", [0, 0]))
        if market_features.shape != expected_market_shape:
            return jsonify({
                "error": f"Formato incorrecto de market_features. Se esperaba {expected_market_shape}, pero se recibió {market_features.shape}"
            }), 400
        
        # Realizar predicción
        action = MODEL.predict(market_features, portfolio_features)
        
        # Construir respuesta
        response = {
            "action": float(action),  # Posición objetivo [-1, 1]
            "position_pct": float(action) * 100,  # Porcentaje más legible
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint para Vertex AI con formato específico
@app.route('/v1/models/trading-agent:predict', methods=['POST'])
def vertex_predict():
    """Endpoint compatible con Vertex AI Prediction."""
    if MODEL is None:
        load_model()
    
    # Obtener datos de la solicitud en formato Vertex AI
    request_json = request.get_json()
    
    if not request_json or "instances" not in request_json:
        return jsonify({"error": "Formato incorrecto. Se espera {'instances': [...]"}), 400
    
    try:
        predictions = []
        for instance in request_json["instances"]:
            # Extraer características
            market_features = np.array(instance.get("market_features"), dtype=np.float32)
            portfolio_features = np.array(instance.get("portfolio_features"), dtype=np.float32)
            
            # Realizar predicción
            action = MODEL.predict(market_features, portfolio_features)
            
            # Agregar a resultados
            predictions.append({
                "action": float(action),
                "position_pct": float(action) * 100
            })
        
        # Construir respuesta en formato Vertex AI
        response = {
            "predictions": predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en predicción Vertex AI: {e}")
        return jsonify({"error": str(e)}), 500

# Punto de entrada directo para probar localmente
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
