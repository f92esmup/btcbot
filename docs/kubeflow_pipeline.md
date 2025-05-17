# Kubeflow Pipelines en Vertex AI

Este documento describe la implementación del pipeline de MLOps utilizando Kubeflow Pipelines (KFP) en Vertex AI para el proyecto BTCBot.

## Introducción a Kubeflow Pipelines

Kubeflow Pipelines (KFP) es una plataforma para crear y ejecutar flujos de trabajo de machine learning. En este proyecto, utilizamos KFP para automatizar todo el proceso de entrenamiento, evaluación y despliegue de nuestros modelos de trading con RL.

## Arquitectura del Pipeline

Nuestro pipeline de Kubeflow consta de cinco componentes principales:

1. **Componente de Adquisición de Datos**
   - Descarga datos históricos de Binance Futures
   - Guarda datos en Cloud Storage
   - Parametrizable por símbolo, intervalo y rango de fechas

2. **Componente de Preprocesamiento**
   - Preprocesa los datos históricos
   - Aplica normalización y cálculo de features
   - Divide datos en conjuntos de entrenamiento y evaluación
   - Guarda datos procesados en Cloud Storage

3. **Componente de Entrenamiento**
   - Entrena el agente RL en los datos procesados
   - Parametrizable por algoritmo y hiperparámetros
   - Guarda el modelo entrenado en Cloud Storage
   - Exporta el modelo en formato compatible con TensorFlow Serving

4. **Componente de Evaluación**
   - Evalúa el rendimiento del modelo en datos de prueba
   - Calcula métricas de trading (Sharpe, Sortino, drawdown, win rate)
   - Determina si el modelo cumple criterios de despliegue
   - Guarda resultados de evaluación en Cloud Storage

5. **Componente de Despliegue** (condicional)
   - Solo se ejecuta si el modelo supera los umbrales de calidad
   - Despliega el modelo en un endpoint de Vertex AI
   - Configura escalado automático y otros parámetros

## Implementación Técnica

### Definición del Pipeline

El pipeline se define en el archivo `gcp/05_create_training_pipeline.py` utilizando la API de KFP v2:

```python
@pipeline(
    name=TRAINING_JOB_NAME_PREFIX,
    description="Pipeline de entrenamiento completo para el agente de trading",
    pipeline_root=f"gs://{MODELS_STAGING_BUCKET}/pipelines"
)
def crypto_trading_pipeline(...):
    # Definición del pipeline
    ...
```

### Componentes Modulares

Cada componente del pipeline se define como una función decorada con `@component` que especifica:
- La imagen Docker base
- Paquetes adicionales a instalar
- Parámetros de entrada y salida
- Lógica de ejecución

```python
@component(
    base_image=PIPELINE_IMAGE_NAME,
    packages_to_install=[...]
)
def acquire_data_component(...):
    # Lógica del componente
    ...
```

### Ejecución Condicional

Utilizamos condicionales para el despliegue automático:

```python
with Condition(
    auto_deploy == True,
    name="auto_deploy_condition"
) as deploy_condition:
    with Condition(
        evaluate_task.outputs["deploy_recommendation"] == True,
        name="quality_check_condition"
    ) as quality_condition:
        deploy_task = deploy_comp(...)
```

## Ventajas de Kubeflow Pipelines

1. **Reproducibilidad**: Cada ejecución del pipeline es trazable y reproducible
2. **Modularidad**: Componentes independientes que se pueden modificar sin afectar al resto
3. **Parametrización**: Configuración flexible mediante parámetros explícitos
4. **Automatización**: Flujo completo desde datos hasta despliegue
5. **Escalabilidad**: Aprovecha la infraestructura de GCP para entrenar a escala
6. **Monitorización**: Interfaz visual para seguir el progreso y resultados

## Cómo Funciona la Transferencia de Datos

Los componentes del pipeline se comunican entre sí mediante:

1. **Artefactos**: Objetos como Dataset o Model que se pasan entre componentes
2. **Parámetros de salida**: Valores que un componente produce y otro consume
3. **Cloud Storage**: Almacenamiento persistente para datos y modelos

Ejemplo de transferencia de datos:

```python
# El componente A produce un Dataset
output_dataset = Output[Dataset]

# El componente B consume ese Dataset
input_dataset = Input[Dataset]

# En el pipeline, conectamos la salida de A con la entrada de B
b_task = component_B(input_dataset=a_task.outputs["output_dataset"])
```

## Ejecución del Pipeline

El pipeline se puede ejecutar de dos formas:

1. **Script de conveniencia**:
   ```bash
   ./run_pipeline.sh --symbol BTCUSDT --timeframe 1h --deploy --wait
   ```

2. **Directamente vía Python**:
   ```python
   pipeline_job = aiplatform.PipelineJob.create(
       display_name=PIPELINE_NAME,
       template_path=pipeline_filename,
       pipeline_root=PIPELINE_ROOT_BUCKET,
       parameter_values={...}
   )
   pipeline_job.submit()
   ```

## Visualización y Debugging

Para visualizar y depurar el pipeline:

1. Ve a la consola de Google Cloud
2. Navega a Vertex AI > Pipelines
3. Selecciona la ejecución del pipeline
4. Explora el DAG (Directed Acyclic Graph) del pipeline
5. Accede a los logs de cada componente
6. Examina los artefactos generados

## Mejores Prácticas Implementadas

1. **Parametrización Explícita**: Todos los componentes aceptan parámetros explícitos
2. **Manejo de Errores**: Cada componente implementa manejo robusto de errores
3. **Logging Detallado**: Logs informativos para facilitar el debugging
4. **Control de Versiones**: Timestamps y metadatos para versionar artefactos
5. **Integración con GCP**: Uso eficiente de servicios nativos de Google Cloud
6. **Validación de Modelos**: Criterios claros para determinar si un modelo es desplegable

## Mejoras Futuras

- Implementar A/B testing automático para modelos
- Añadir monitorización de drift de datos y modelos
- Integrar pipeline con sistemas de notificación (Slack, email)
- Optimización automática de hiperparámetros
- Implementar canary deployments para reducir riesgo
