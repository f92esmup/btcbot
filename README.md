# Trading Bot con Reinforcement Learning para Futuros de Criptomonedas

## Descripción

Este proyecto implementa un agente de trading basado en Reinforcement Learning para operar en el mercado de futuros de criptomonedas. Utiliza el algoritmo SAC (Soft Actor-Critic) con una arquitectura de red neuronal basada en Transformers para procesar datos históricos del mercado y tomar decisiones de trading.

**Estado Actual del Proyecto**: Esta es una versión de desarrollo para entrenamiento local funcional. **No está lista para trading en vivo**.

## Características Principales

- Descarga de datos históricos de Binance Futures API
- Preprocesamiento y cálculo de características técnicas de trading
- Simulación de un entorno de trading de futuros BTCUSDT con Gymnasium
- Agente de RL (Soft Actor-Critic) con extractor de características basado en Transformers
- Visualización de resultados y métricas de rendimiento
- Infraestructura en Google Cloud Platform para entrenamiento y despliegue a escala
- **Configuración Flexible**: Sistema centralizado basado en variables de entorno para fácil adaptación a diferentes contextos
- **Pipeline de MLOps Completo**: Automatización de entrenamiento, evaluación y despliegue en GCP

## Estructura del Proyecto

```
btcbot/
├── data/                      # Datos del mercado
│   ├── raw/                   # Datos sin procesar de Binance
│   └── processed/             # Datos preprocesados con features
├── docs/                      # Documentación detallada
├── gcp/                       # Scripts para Google Cloud Platform
├── logs/                      # Logs de entrenamiento y evaluación
├── models/                    # Modelos guardados
├── results/                   # Resultados de evaluación
├── scripts/                   # Scripts ejecutables
└── src/                       # Código fuente
    ├── agent/                 # Implementación del agente RL
    ├── data/                  # Código para adquisición y preprocesamiento
    ├── environments/          # Entorno de simulación
    └── utils/                 # Utilidades generales
```

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
   
   Copiar el archivo `.env.example` a `.env` y modificarlo con tus valores:
   
   ```bash
   cp .env.example .env
   # Editar el archivo .env con tus valores
   ```
   
   Variables esenciales para ejecución local:
   ```
   BINANCE_API_KEY_FUTURES="TU_CLAVE_AQUI"
   BINANCE_API_SECRET_FUTURES="TU_SECRETO_AQUI"
   ```
   
   > **Nota**: Las claves API reales de Binance solo son necesarias para `download_data.py`. Para el resto de los scripts (preprocesamiento, entrenamiento, evaluación) puedes usar valores de placeholder si ya tienes datos descargados.

## Uso Local

### 1. Descarga de Datos

```bash
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start_date 2022-01-01
```

### 2. Preprocesamiento

```bash
python scripts/preprocess_data.py --input_file data/raw/BTCUSDT_FUTURES_1h_20220101_*.csv
```

### 3. Entrenamiento del Modelo

```bash
# Opción 1: Usar valores por defecto desde config.yaml
python scripts/train_rl_agent.py --timesteps 100000

# Opción 2: Usar variables de entorno para configuración
export AGENT_LEARNING_RATE=0.0003
export AGENT_BUFFER_SIZE=100000
python scripts/train_rl_agent.py --timesteps 100000
```

### 4. Evaluación del Modelo

```bash
python scripts/evaluate_rl_agent.py --model_path models/sac_transformer_trading_agent_*.zip
```

## Implementación en Google Cloud Platform

Este proyecto incluye implementación completa en Google Cloud Platform (GCP) para entrenamiento a escala, versionado de modelos y despliegue automatizado con configuración centralizada basada en variables de entorno. 

### Características de la Implementación en GCP

- **Configuración centralizada** mediante variables de entorno
- **Automatización completa** con pipelines en Vertex AI
- **Entrenamiento distribuido** con soporte para GPUs
- **Registro y versionado de modelos** en Vertex AI Model Registry
- **Despliegue condicional** basado en métricas de rendimiento
- **Monitorización** y logging integrados

Consulta el directorio `gcp/` y su [README](gcp/README.md) para obtener instrucciones detalladas sobre la implementación paso a paso.

Para más información sobre la migración a GCP y la configuración centralizada, consulta [Google Cloud Integration](docs/googlecloud.md).

### Requisitos para GCP

- Cuenta de Google Cloud Platform
- Google Cloud SDK instalado y configurado
- Proyecto GCP con facturación habilitada
- Variables de entorno configuradas (ver `.env.example`)

3. **Instalar dependencias**:
   ```
   pip install -r requirements.txt
   ```

4. **Configurar Variables de Entorno**:
   
   Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:
   
   ```
   BINANCE_API_KEY_FUTURES="TU_CLAVE_AQUI"
   BINANCE_API_SECRET_FUTURES="TU_SECRETO_AQUI"
   ```
   
   > **Nota**: Las claves API reales de Binance solo son necesarias para `download_data.py`. Para el resto de los scripts (preprocesamiento, entrenamiento, evaluación) puedes usar valores de placeholder si ya tienes datos descargados.

## Uso

### 1. Descarga de Datos

```
python scripts/download_data.py
```

Este script descargará datos históricos de BTCUSDT de Binance Futures y los guardará en `data/raw/`.

### 2. Preprocesamiento de Datos

```
python scripts/preprocess_data.py
```

Procesa los datos descargados para calcular características técnicas y los guarda en `data/processed/`.

### 3. Prueba del Entorno

```
python scripts/test_environment.py
```

Verifica que el entorno de trading funcione correctamente, simulando un agente aleatorio.

### 4. Entrenamiento del Agente

Para un entrenamiento de prueba rápido:
```
python scripts/train_rl_agent.py --timesteps 1000
```

Para un entrenamiento más extenso:
```
python scripts/train_rl_agent.py --timesteps 1000000
```

### 5. Evaluación del Agente

```
python scripts/evaluate_rl_agent.py --model-path models/sac_transformer_trading_agent_final_1000_steps.zip --episodes 1
```

## Archivos de Configuración

- `src/config.yaml`: Configuración general del proyecto
- `src/data/preprocessing_config.yaml`: Parámetros para el preprocesamiento de datos
- `src/environments/environment_config.yaml`: Configuración del entorno de simulación
- `src/agent/agent_config.yaml`: Hiperparámetros del agente RL

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


## Próximos Pasos y Mejoras Futuras

Si bien la versión actual del bot se centra en la toma de decisiones del agente de Reinforcement Learning basadas en velas completas de un timeframe específico, existen varias vías interesantes para futuras mejoras y una mayor sofisticación:

### 1. Gestión de Riesgo Avanzada y Monitorización Continua de Precios (Recomendado)

* **Objetivo:** Implementar una capa de gestión de riesgo más granular que opere entre las decisiones estratégicas del agente de RL.
* **Enfoque Propuesto:**
    * El bot de ejecución principal (desplegado, por ejemplo, en una VM de Google Compute Engine) mantendría una conexión **WebSocket** con Binance para recibir actualizaciones de precios en tiempo real (ej., a través del stream `@trade` o `@bookTicker`).
    * Paralelamente a las decisiones del agente RL (que seguirían basándose en velas completas del timeframe principal y se obtendrían del endpoint de Vertex AI), este bot monitorizaría continuamente el precio actual.
    * Si una posición abierta alcanza un nivel de **Stop-Loss (SL)** o **Take-Profit (TP)** predefinido, el bot ejecutaría inmediatamente una orden de cierre, sin esperar a la siguiente señal del agente RL.
    * Esto combina las decisiones estratégicas del RL con una gestión de riesgo táctica y reactiva.

### 2. Transición a un Agente de RL Basado en Datos de Mercado de Mayor Frecuencia (Investigación Advan