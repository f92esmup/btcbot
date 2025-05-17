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
python scripts/train_rl_agent.py --timesteps 100000
```

### 4. Evaluación del Modelo

```bash
python scripts/evaluate_rl_agent.py --model_path models/sac_transformer_trading_agent_*.zip
```

## Implementación en Google Cloud Platform

Este proyecto incluye implementación completa en Google Cloud Platform (GCP) para entrenamiento a escala, versionado de modelos y despliegue automatizado. Consulta el directorio `gcp/` y su [README](gcp/README.md) para obtener instrucciones detalladas sobre:

- Configuración de la infraestructura en GCP
- Ejecución de trabajos de entrenamiento en Vertex AI
- Registro y versionado de modelos en Vertex AI Model Registry  
- Evaluación de modelos con métricas de trading
- Despliegue de modelos a endpoints para inferencia
- Pipelines de MLOps automatizados

### Requisitos para GCP

- Cuenta de Google Cloud Platform
- Google Cloud SDK instalado y configurado
- APIs necesarias habilitadas (consulta la documentación)
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

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

### 2. Transición a un Agente de RL Basado en Datos de Mercado de Mayor Frecuencia (Investigación Avanzada)

* **Objetivo:** Explorar si el agente de RL puede tomar decisiones directamente basadas en datos de mercado de mayor frecuencia, como datos de trades (tick data) o actualizaciones del libro de órdenes, en lugar de velas OHLCV.
* **Implicaciones y Desafíos:**
    * **Adquisición y Almacenamiento de Datos Históricos:** Se requerirían grandes volúmenes de datos históricos de trades o del libro de órdenes para el entrenamiento, lo cual puede ser un desafío obtener de forma gratuita y completa desde las APIs públicas para periodos extensos.
    * **Ingeniería de Características:** Las características de entrada para el agente de RL necesitarían ser rediseñadas completamente para reflejar la naturaleza de estos datos de alta frecuencia (ej., desequilibrios del libro de órdenes, flujo de órdenes, micro-volatilidad).
    * **Diseño del Entorno de Simulación:** El `TradingEnvironment` necesitaría una simulación mucho más compleja y computacionalmente intensiva para operar a nivel de tick o evento del libro de órdenes, modelando con precisión la latencia y el slippage.
    * **Arquitectura del Agente:** Aunque el Transformer es adaptable, la forma exacta de procesar secuencias de eventos de mercado (en lugar de velas) podría requerir ajustes.
    * **Costos Computacionales:** El entrenamiento y la simulación serían significativamente más demandantes.
* **Estado Actual:** Esta es una línea de investigación considerablemente más compleja y se considera una mejora a muy largo plazo, una vez que el sistema actual esté completamente validado y operativo.

### 3. Optimización Avanzada de Hiperparámetros

* Utilizar técnicas como la optimización bayesiana o algoritmos genéticos (con herramientas como Optuna o Ray Tune) para encontrar conjuntos de hiperparámetros óptimos tanto para el agente de RL como para la arquitectura del Transformer.

### 4. Incorporación de Múltiples Timeframes o Fuentes de Datos

* Mejorar el extractor de características para que el Transformer pueda procesar y fusionar información de múltiples timeframes de velas (ej., 15min, 1h, 4h) simultáneamente.
* Explorar la incorporación de otras fuentes de datos relevantes (ej., análisis de sentimiento, datos on-chain, datos macroeconómicos) si se pueden cuantificar y alinear temporalmente.

### 5. Desarrollo de un Módulo de Paper Trading y Transición a Trading en Vivo

* Crear un módulo robusto de "paper trading" que simule operaciones con una cuenta ficticia pero utilizando datos de mercado en tiempo real y latencias de ejecución realistas.
* Tras una validación exhaustiva en paper trading, planificar cuidadosamente la transición a trading en vivo con capital real, comenzando con tamaños de posición muy pequeños.

### 6. Monitorización y Alertas en Profundidad

* Expandir la monitorización en GCP para incluir métricas de negocio específicas del bot (ej., P&L realizado, drawdown, número de operaciones, slippage promedio) y configurar alertas para desviaciones significativas o errores del sistema.