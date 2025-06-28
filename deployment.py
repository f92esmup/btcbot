import os
import yaml
import tempfile
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List
import torch

# Importar las clases necesarias desde la estructura del proyecto
# Se asume que la estructura de `src` es accesible.
from src.agente.agent import Agent
from src.configuration.config import Config
from src.configuration.gcs_utils import GCSUtils

# --- Modelos de Datos Pydantic para la API ---

class LoadRequest(BaseModel):
    run_id: str = Field(..., description="El ID del experimento (ej. 'sac-transformer-20250628-103000') a cargar.")

class LoadResponse(BaseModel):
    message: str
    loaded_run_id: str

class PredictionRequest(BaseModel):
    state: List[float] = Field(..., description="Vector de estado ya procesado y normalizado.")

class PredictionResponse(BaseModel):
    action: int

# --- Aplicación FastAPI ---

app = FastAPI(
    title="Trading Agent Prediction Service",
    description="Un servicio que carga modelos de trading bajo demanda y sirve predicciones.",
    version="2.0.0"
)

# --- Estado Global del Servicio ---

# Estas variables mantendrán el modelo actualmente cargado en memoria.
# Empiezan en None, indicando el estado de "letargo".
agent: Agent = None
config: Config = None
current_run_id: str = None

# --- Endpoints de la API ---

@app.post("/load_model", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    """
    Endpoint de control para cargar un modelo en memoria.
    Saca al servicio de su estado de letargo y lo prepara para la inferencia.
    """
    global agent, config, current_run_id
    
    print(f"Recibida solicitud para cargar el modelo con RUN_ID: {request.run_id}")

    try:
        # Inicializar las utilidades de GCS
        gcs_utils = GCSUtils()

        # Crear un directorio temporal para descargar los artefactos
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            model_path = os.path.join(tmpdir, "agent.pth")

            # Definir las rutas de los blobs en GCS
            gcs_config_blob = f"{request.run_id}/config.yaml"
            gcs_model_blob = f"{request.run_id}/agent.pth"

            # Descargar los archivos usando la utilidad de GCS
            gcs_utils.download_file_from_gcs(gcs_config_blob, config_path)
            gcs_utils.download_file_from_gcs(gcs_model_blob, model_path)

            # Cargar la configuración
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            loaded_config = Config(config_dict)
            
            # Extraer parámetros y crear el agente
            state_size = loaded_config.get('agent.state_size')
            action_size = loaded_config.get('agent.action_size')
            
            loaded_agent = Agent(state_size=state_size, action_size=action_size, config=loaded_config)
            loaded_agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            loaded_agent.eval() # Modo de evaluación es crucial

            # Actualizar el estado global de forma atómica al final
            config = loaded_config
            agent = loaded_agent
            current_run_id = request.run_id
            
            print(f"Modelo '{current_run_id}' cargado exitosamente.")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No se encontraron los artefactos para el RUN_ID '{request.run_id}' en el bucket '{bucket_name}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {e}")

    return LoadResponse(
        message="Modelo cargado y listo para predecir.",
        loaded_run_id=current_run_id
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint de inferencia. Usa el modelo actualmente en memoria para predecir una acción.
    """
    if not agent or not config:
        raise HTTPException(
            status_code=409, # 409 Conflict: el estado del servidor impide la petición
            detail="Ningún modelo cargado. Por favor, llame a /load_model primero."
        )

    # Validación de la dimensión del estado usando la config cargada
    expected_size = config.get('agent.state_size')
    if len(request.state) != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"La dimensión del vector de estado es incorrecta. El modelo '{current_run_id}' espera {expected_size}, pero se recibió {len(request.state)}."
        )

    # Realizar la predicción
    try:
        state_tensor = torch.FloatTensor(request.state).unsqueeze(0)
        with torch.no_grad(): # Desactiva el cálculo de gradientes para inferencia
            action = agent.select_action(state_tensor, deterministic=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    return PredictionResponse(action=action)

@app.get("/status")
def get_status():
    """
    Endpoint de estado para verificar qué modelo está cargado actualmente.
    """
    if not current_run_id:
        return {"status": "En letargo. Ningún modelo cargado."}
    
    return {
        "status": "Activo",
        "loaded_run_id": current_run_id,
        "model_config": {
            "state_size": config.get('agent.state_size'),
            "action_size": config.get('agent.action_size')
        }
    }