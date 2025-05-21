# BTCBot - Sistema de Trading Automático con RL

BTCBot es un sistema completo de trading automático para Bitcoin basado en Reinforcement Learning (RL). El sistema incluye componentes para entrenamiento de modelos, inferencia, y trading en vivo.

## Estructura del Proyecto

El proyecto se ha organizado en las siguientes carpetas principales:

- `src/`: Código fuente principal del sistema
  - `agent/`: Implementación del agente RL y extractor de características
  - `data/`: Descarga y procesamiento de datos
  - `environments/`: Entorno de simulación para entrenamiento RL
  - `live/`: Componentes para trading en vivo
  - `utils/`: Utilidades compartidas

- `scripts/`: Scripts ejecutables para varias tareas
  - `train_rl_agent.py`: Entrena el agente RL
  - `evaluate_rl_agent.py`: Evalúa el rendimiento del agente
  - `run_live_trader.py`: Ejecuta el trading en vivo

- `serving/`: Servidor de inferencia para despliegue en producción
  - `serve.py`: Implementación del servidor Flask/Gunicorn

- `gcp/`: Archivos de configuración para Google Cloud Platform
  - `cloudbuild/`: Configuración de CI/CD con Cloud Build
  - `kubernetes/`: Configuración para despliegue en GKE

## Despliegue en GKE

El sistema está diseñado para desplegarse en Google Kubernetes Engine (GKE) con dos componentes principales:

1. **Servidor de Inferencia**: Carga el modelo RL desde GCS y expone un endpoint para predicciones
2. **Trader en Vivo**: Conecta con Binance y utiliza el servidor de inferencia para tomar decisiones

Para obtener instrucciones detalladas sobre el despliegue, consulta:
- `gcp/kubernetes/README.md`: Guía para despliegue manual en GKE
- `gcp/cloudbuild/README.md`: Guía para CI/CD automatizado con Cloud Build

## Diagrama de Arquitectura

```
                   +-------------------+
                   |  Binance Futures  |
                   +-------------------+
                          ^   |
                          |   | Datos de mercado (WebSocket)
                          |   v
+----------------+    +-------------------+    +-------------------+
|                |    |                   |    |                   |
| GCS            |    | Trading en Vivo   |    | Servidor de       |
| - Modelos RL   |<-->| (run_live_trader) |<-->| Inferencia RL     |
| - Logs         |    |                   |    | (serve.py)        |
|                |    +-------------------+    +-------------------+
+----------------+           |
                             | Logs y métricas
                             v
                   +-------------------+
                   |   BigQuery        |
                   |   (opcional)      |
                   +-------------------+
```

## Requisitos

- Python 3.9+
- PyTorch
- Stable Baselines3
- Flask/Gunicorn
- Google Cloud SDK
- docker
- kubectl

## Configuración

La configuración centralizada se encuentra en `src/config.yaml`. Este archivo contiene:

- Parámetros del agente RL
- Configuración del entorno de trading
- Configuración de preprocesamiento de datos
- Parámetros para trading en vivo

## Licencia

[Incluir información de licencia]




YO UTILIZO:
BIGQUERY
STORAGE
SECRETS
GKE
LOOCKER STUDIO
CLOUD BUILD
ARTIFACT REGESTRY
