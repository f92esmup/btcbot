import logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from src.agente.agent import TransformerSACAgent
from src.training.run_manager import RunManager
from src.utils.system import setup_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelServiceState:
    def __init__(self):
        self.agent: Optional[TransformerSACAgent] = None
        self.run_config: Optional[Dict[str, Any]] = None
        self.device: torch.device = setup_device(no_cuda=True)
        self.run_manager: RunManager = RunManager()

model_service = ModelServiceState()

# --- Modelos de Datos Pydantic para la API ---

class LoadRequest(BaseModel):
    run_id: str = Field(..., description="El ID del run de entrenamiento a cargar.")
    model_type: str = Field("best", description="Tipo de modelo a cargar: 'best' o 'final'.")

class LoadResponse(BaseModel):
    message: str
    loaded_run_id: str

class PredictionRequest(BaseModel):
    state: List[float] = Field(..., description="Vector de estado ya procesado y normalizado.")

class PredictionResponse(BaseModel):
    action: List[float] = Field(..., description="La acción continua (-1 a 1) predicha por el agente.")

# --- Aplicación FastAPI ---

app = FastAPI(
    title="Trading Agent Prediction Service",
    description="Servicio que carga un modelo de trading de RL bajo demanda y sirve predicciones.",
    version="3.0.0"
)

# --- Endpoints de la API ---

@app.post("/load_model", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    """
    Endpoint de control para cargar un modelo en memoria.
    Saca al servicio de su estado de letargo y lo prepara para la inferencia.
    """
    try:
        logger.info(f"Recibida solicitud para cargar el modelo con RUN_ID: {request.run_id}, tipo: {request.model_type}")
        
        # Usar RunManager para obtener la configuración del run
        run_config = model_service.run_manager.download_and_load_yaml_config(request.run_id)
        if run_config is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontró la configuración para el run '{request.run_id}'"
            )
        
        # Extraer hiperparámetros del diccionario de configuración
        hyperparams = run_config.get('hyperparameters', {})
        
        # Obtener los parámetros de la configuración del run original
        hyperparams = run_config.get('hyperparameters', {})
        sequence_length = hyperparams.get('sequence_length', run_config.get('config_snapshot', {}).get('environment', {}).get('ventana_observacion_size', 24))
        market_features = hyperparams.get('market_features', 12)
        portfolio_features = hyperparams.get('portfolio_features', 4)
        action_dim = 1 # Nuestra acción es continua de dimensión 1

        # Calcular la forma PLANA del espacio de observación, tal como en el entrenamiento
        obs_dim_flat = (sequence_length * market_features) + portfolio_features

        # Crear instancia vacía del agente con las formas correctas
        agent = TransformerSACAgent(
            observation_space_shape=(obs_dim_flat,),
            action_space_shape=(action_dim,),
            market_features=market_features,
            portfolio_features=portfolio_features,
            sequence_length=sequence_length,
            device=model_service.device,
            is_distributed=False # El despliegue nunca es distribuido
        )

        # Definir el prefijo del modelo a cargar
        model_prefix = f"{request.run_id}/{request.model_type}_model"
        
        # Cargar los pesos del modelo usando RunManager
        model_service.run_manager.load_agent_from_checkpoint(
            agent=agent,
            checkpoint_prefix=model_prefix,
            reset_optimizers=True  # No necesarios para inferencia
        )
        
        # Poner el agente en modo de evaluación
        agent.eval_mode()
        
        # Actualizar el estado global del servicio
        model_service.agent = agent
        model_service.run_config = run_config
        
        logger.info(f"Modelo '{request.run_id}' (tipo: {request.model_type}) cargado exitosamente")
        
        return LoadResponse(
            message=f"Modelo cargado y listo para predecir (tipo: {request.model_type})",
            loaded_run_id=request.run_id
        )
        
    except FileNotFoundError as e:
        error_msg = f"Archivos del modelo no encontrados para el run '{request.run_id}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=404, detail=error_msg)
        
    except Exception as e:
        error_msg = f"Error al cargar el modelo '{request.run_id}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint de inferencia. Usa el modelo actualmente en memoria para predecir una acción.
    """
    # Verificar que hay un modelo cargado
    if model_service.agent is None:
        raise HTTPException(
            status_code=409,
            detail="Ningún modelo cargado. Por favor, llame a /load_model primero."
        )
    
    try:
        # Extraer parámetros de la configuración cargada
        hyperparams = model_service.run_config.get('hyperparameters', {})
        sequence_length = hyperparams.get('sequence_length', model_service.run_config.get('config_snapshot', {}).get('environment', {}).get('ventana_observacion_size', 24))
        market_features = hyperparams.get('market_features', 12)
        portfolio_features = hyperparams.get('portfolio_features', 4)
        
        # Validar la longitud del estado de entrada
        expected_length = (sequence_length * market_features) + portfolio_features
        if len(request.state) != expected_length:
            raise HTTPException(
                status_code=400,
                detail=f"La dimensión del vector de estado es incorrecta. Se esperaba {expected_length}, pero se recibió {len(request.state)}."
            )
        
        # Convertir a numpy array para facilitar el procesamiento
        state_array = np.array(request.state)
        
        # Calcular punto de división entre datos de mercado y portfolio
        market_features_total = sequence_length * market_features
        
        # Dividir el array en market_data y portfolio_data
        market_data_flat = state_array[:market_features_total]
        portfolio_data_flat = state_array[market_features_total:]
        
        # Reformar market_data a (sequence_length, market_features)
        market_data = market_data_flat.reshape(sequence_length, market_features)
        
        # Convertir a tensores de PyTorch y añadir dimensión de batch
        market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(model_service.device)
        portfolio_tensor = torch.FloatTensor(portfolio_data_flat).unsqueeze(0).to(model_service.device)
        
        # Obtener la acción del agente
        action = model_service.agent.select_action(
            market_tensor, 
            portfolio_tensor, 
            deterministic=True
        )
        
        # Convertir la acción a lista de Python
        action_list = action.tolist()
        
        logger.info(f"Predicción completada: acción = {action_list}")
        
        return PredictionResponse(action=action_list)
        
    except Exception as e:
        error_msg = f"Error durante la predicción: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/status")
def get_status():
    """
    Endpoint de estado para verificar qué modelo está cargado actualmente.
    """
    # Verificar si hay un modelo cargado
    if model_service.run_config is None:
        return {"status": "En letargo. Ningún modelo cargado."}
    
    # Extraer información del modelo activo
    hyperparams = model_service.run_config.get('hyperparameters', {})
    
    # Obtener parámetros relevantes
    sequence_length = hyperparams.get('sequence_length', model_service.run_config.get('config_snapshot', {}).get('environment', {}).get('ventana_observacion_size', 24))
    market_features = hyperparams.get('market_features', 12)
    portfolio_features = hyperparams.get('portfolio_features', 4)
    
    # Extraer run_id de la configuración
    run_id = model_service.run_config.get('run_id', 'unknown')
    
    return {
        "status": "Activo",
        "loaded_run_id": run_id,
        "model_config": {
            "market_features": market_features,
            "portfolio_features": portfolio_features,
            "sequence_length": sequence_length,
            "device": str(model_service.device),
            "observation_dim": (sequence_length * market_features) + portfolio_features,
            "action_dim": 1
        }
    }