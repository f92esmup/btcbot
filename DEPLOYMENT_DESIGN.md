# Diseño del Servicio de Despliegue de Modelos (`deployment.py`)

## 1. Objetivo y Principios de Diseño

El objetivo de `deployment.py` es proporcionar un microservicio robusto, flexible y eficiente para servir predicciones de cualquier modelo de trading entrenado. El diseño se fundamenta en tres principios clave:

- **Principio de Responsabilidad Única (SRP):** El servicio tiene una única responsabilidad: la inferencia. No se ocupa de la obtención de datos, el cálculo de indicadores o la normalización. Es una "calculadora de acciones" pura.
- **Carga Dinámica y Bajo Demanda:** El servicio no está atado a un único modelo. Arranca en un estado de "letargo" y puede cargar cualquier modelo entrenado (`run_id`) que se le solicite, sin necesidad de reiniciar o redesplegar.
- **No Repetir Código (DRY):** El servicio reutiliza la lógica y las clases ya existentes en el proyecto (`src.agente.Agent`, `src.configuration.Config`, `src.utils.gcs_utils`) para mantener la consistencia y reducir la duplicación de código.

---

## 2. Arquitectura General: Dos Endpoints

Para cumplir con los requisitos de flexibilidad y eficiencia (evitando la alta latencia de cargar el modelo en cada predicción), la arquitectura se basa en dos endpoints principales que separan la **gestión del modelo** de la **inferencia**.

1.  **Endpoint de Control (`/load_model`):** Se encarga de cargar un modelo específico desde GCS a la memoria del servicio.
2.  **Endpoint de Inferencia (`/predict`):** Utiliza el modelo actualmente en memoria para realizar predicciones de forma instantánea.

Adicionalmente, se incluye un endpoint de `/status` para observar el estado actual del servicio.

---

## 3. Detalle de los Endpoints

### 3.1. Endpoint de Control: `POST /load_model`

- **Propósito:** Cargar (o reemplazar) un modelo de trading en la memoria del servicio, preparándolo para la inferencia.
- **Petición (Request Body):**
  ```json
  {
    "run_id": "sac-transformer-20250628-103000"
  }
  ```
- **Lógica de Funcionamiento:**
  1.  Recibe el `run_id` del modelo a cargar.
  2.  Utiliza la función `download_file_from_gcs` para descargar `config.yaml` y `agent.pth` desde el bucket de GCS (la ruta se construye con el `run_id`).
  3.  Carga el `config.yaml` y extrae los parámetros `state_size` y `action_size`.
  4.  Instancia la clase `Agent` con la configuración cargada.
  5.  Carga los pesos del modelo (`agent.pth`) en la instancia del agente.
  6.  Pone al agente en modo de evaluación (`agent.eval()`), crucial para la inferencia determinista.
  7.  Almacena el agente y la configuración en variables globales, listos para ser usados por `/predict`.
- **Respuesta (Response Body):**
  ```json
  {
    "message": "Modelo cargado y listo para predecir.",
    "loaded_run_id": "sac-transformer-20250628-103000"
  }
  ```

### 3.2. Endpoint de Inferencia: `POST /predict`

- **Propósito:** Realizar una predicción rápida usando el modelo actualmente en memoria.
- **Petición (Request Body):**
  ```json
  {
    "state": [0.12, -0.45, 0.88, ..., -0.19]
  }
  ```
- **Lógica de Funcionamiento:**
  1.  Verifica si hay un agente cargado. Si no (estado de letargo), devuelve un error `409 Conflict`.
  2.  Recibe el vector de estado, que **debe venir ya procesado y normalizado**.
  3.  **Validación Dinámica:** Compara la longitud del vector `state` recibido con el `state_size` de la configuración del modelo cargado. Si no coinciden, devuelve un error `400 Bad Request`.
  4.  Convierte la lista de estado a un tensor de PyTorch.
  5.  Llama al método `agent.select_action(state_tensor, deterministic=True)`.
  6.  Devuelve la acción resultante.
- **Respuesta (Response Body):**
  ```json
  {
    "action": 1
  }
  ```

### 3.3. Endpoint de Observabilidad: `GET /status`

- **Propósito:** Consultar el estado actual del servicio.
- **Petición (Request Body):** Ninguna.
- **Lógica de Funcionamiento:**
  - Si no hay ningún modelo cargado, informa que está en estado de letargo.
  - Si hay un modelo cargado, informa su `run_id` y su configuración básica (`state_size`, `action_size`).
- **Respuesta (Response Body):**
  ```json
  {
    "status": "Activo",
    "loaded_run_id": "sac-transformer-20250628-103000",
    "model_config": {
      "state_size": 600,
      "action_size": 3
    }
  }
  ```

---

## 4. Flujo de Trabajo Típico

1.  **Desplegar** el contenedor con `deployment.py` en Vertex AI. El servicio inicia en estado de "letargo".
2.  **Cargar un Modelo:** Un cliente (u operador) realiza una única llamada a `POST /load_model` con el `run_id` deseado.
3.  **Predecir:** El cliente puede ahora realizar miles de llamadas a `POST /predict`. Cada llamada será respondida con baja latencia, ya que el modelo está en memoria.
4.  **Cambiar de Modelo:** Para usar un modelo diferente, el cliente simplemente vuelve a llamar a `POST /load_model` con un nuevo `run_id`. El servicio reemplazará el modelo anterior "en caliente".

---

## 5. Requisitos y Dependencias

- **Variables de Entorno:** El contenedor debe tener acceso a la variable `GCS_BUCKET_NAME` para saber de dónde descargar los artefactos.
- **Librerías Python:** `fastapi`, `uvicorn`, `pydantic`, `torch`, `PyYAML`, `google-cloud-storage`.
