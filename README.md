# Trading Bot con Reinforcement Learning para Futuros de Criptomonedas

## Descripción

Este proyecto implementa un agente de trading basado en Reinforcement Learning para operar en el mercado de futuros de criptomonedas. Utiliza el algoritmo SAC (Soft Actor-Critic) con una arquitectura de red neuronal basada en Transformers para procesar datos históricos del mercado y tomar decisiones de trading.

**Estado Actual del Proyecto**: Esta es una versión de desarrollo para entrenamiento local funcional. **No está lista para trading en vivo**.

## Características Principales

- Descarga de datos históricos de Binance Futures API
- Preprocesamiento y cálculo de características técnicas de trading usando pandas-ta
- Simulación de un entorno de trading de futuros BTCUSDT con Gymnasium
- Agente de RL (Soft Actor-Critic) con extractor de características basado en Transformers
- Visualización de resultados y métricas de rendimiento

## Estructura del Proyecto

```
btcbot/
├── logs/                      # Logs de entrenamiento y evaluación
├── models/                    # Modelos guardados
├── results/                   # Resultados de evaluación
├── scripts/                   # Scripts ejecutables
└── src/                       # Código fuente
    ├── config.yaml            # Configuración centralizada
    ├── agent/                 # Implementación del agente RL
    ├── data/                  # Código para adquisición y preprocesamiento
    ├── environments/          # Entorno de simulación
    └── utils/                 # Utilidades generales
```

## Almacenamiento en la Nube (Obligatorio)

El proyecto ahora funciona exclusivamente con almacenamiento en la nube:

- **Google Cloud Storage (GCS)**: Almacenamiento tanto para datos brutos como procesados
- **Google Cloud Secret Manager**: Gestión segura de credenciales de la API de Binance

> **Nota importante**: Ya no se utiliza almacenamiento local para datos. Todos los datos se cargan y guardan directamente desde/hacia Google Cloud Storage.

## Requisitos Previos

- Python 3.9+
- Git

## Instalación

1. **Clonar el repositorio**:
   ```
   git clone https://github.com/yourusername/btcbot.git
   cd btcbot
   ```

2. **Crear y activar un entorno virtual**:
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```
   pip install -r requirements.txt
   ```

4. **Configurar Variables de Entorno**:
   
   Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:
   
   ```
   # Configuración de Google Cloud (OBLIGATORIO)
   GCP_PROJECT_ID="tu-proyecto-id"
   GCS_BUCKET_NAME="tu-bucket-nombre"
   
   # Configuración regional de Google Cloud
   GCP_REGION="tu-region-preferida"
   ```
   
   > **Nota**: Las credenciales de Binance deben estar almacenadas en Google Cloud Secret Manager como `BINANCE_API_KEY_FUTURES` y `BINANCE_API_SECRET_FUTURES`. La autenticación con GCP se realiza mediante Application Default Credentials (ejecuta `gcloud auth application-default login`).

5. **Configurar Google Cloud**:

   ```
   # Iniciar sesión en Google Cloud
   gcloud auth login
   
   # Configurar credenciales de aplicación por defecto
   gcloud auth application-default login
   
   # Habilitar APIs necesarias
   gcloud services enable secretmanager.googleapis.com storage.googleapis.com
   
   # Crear bucket de GCS (si no existe)
   gsutil mb -p tu-proyecto-id -l tu-region gs://tu-bucket-nombre
   ```

## Uso

### 1. Descarga de Datos

```
python scripts/download_data.py
```

Este script descargará datos históricos de BTCUSDT de Binance Futures y los guardará directamente en Google Cloud Storage.

### 2. Preprocesamiento de Datos

```
python scripts/preprocess_data.py
```

Procesa los datos descargados para calcular características técnicas y los guarda en Google Cloud Storage.

Para procesar un archivo específico:
```
python scripts/preprocess_data.py --file BTCUSDT_FUTURES_1h_20200101_20250516.csv
```

### 3. Prueba del Entorno

```
python scripts/test_environment.py
```

Verifica que el entorno de trading funcione correctamente, simulando un agente aleatorio.

### 4. Entrenamiento del Agente

Para un entrenamiento de prueba rápido:
```
python scripts/train_rl_agent.py --config src/config.yaml --timesteps 1000
```

Para un entrenamiento más extenso:
```
python scripts/train_rl_agent.py --config src/config.yaml --timesteps 1000000
```

### 5. Evaluación del Agente

```
python scripts/evaluate_rl_agent.py --config src/config.yaml --model-path models/sac_transformer_trading_agent_final_1000_steps.zip --episodes 1
```

## Archivos de Configuración

- `src/config.yaml`: Configuración centralizada para todo el proyecto
  - Incluye configuración de rutas, API de Binance, preprocesamiento, entorno y agente RL
  - Toda la configuración se gestiona desde este único archivo

## Almacenamiento de Datos y Credenciales

- **Datos**: Google Cloud Storage (GCS) [Obligatorio]
  - Bucket: Definido en `.env` como `GCS_BUCKET_NAME`
  - Rutas configuradas en `src/config.yaml`
  - Todo el procesamiento de datos se realiza directamente en la nube, sin almacenamiento local

- **Credenciales API**: Google Cloud Secret Manager [Obligatorio]
  - Las credenciales de Binance se almacenan como secretos
  - Proyecto GCP: Definido en `.env` como `GCP_PROJECT_ID`
  - Secretos requeridos: `BINANCE_API_KEY_FUTURES` y `BINANCE_API_SECRET_FUTURES`

## Tecnologías Clave

- PyTorch: Framework de aprendizaje profundo
- Stable Baselines3: Implementaciones de algoritmos de RL
- Gymnasium: Framework para entornos de RL
- Pandas & NumPy: Procesamiento de datos
- Matplotlib: Visualización
- python-binance: API para conexión con Binance

## Documentación Adicional

Para información más detallada sobre cada componente, consulta los archivos en el directorio `docs/`:

- [Adquisición de Datos](docs/adquisición_datos.md)
- [Preprocesamiento](docs/preprocesamiento.md)
- [Entorno de Simulación](docs/entorno.md)
- [Agente RL](docs/agente.md)

## Limitaciones Conocidas

- Esta versión es solo para entrenamiento local, no está preparada para trading en vivo
- No incluye optimización avanzada de hiperparámetros
- La simulación no tiene en cuenta la profundidad del mercado ni el impacto de las operaciones
- El rendimiento del modelo no está garantizado en mercados reales

## Próximos Pasos

- Implementar backtest más rigurosos
- Añadir optimización automática de hiperparámetros
- Desarrollar módulo para trading en vivo (paper trading)
- Mejorar la arquitectura del modelo con atención a múltiples timeframes

## Licencia

MIT
