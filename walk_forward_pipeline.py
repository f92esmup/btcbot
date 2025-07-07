"""
Pipeline de validación walk-forward para optimización de hiperparámetros con Vertex AI.
Orquesta la ejecución secuencial de trials de Hypertune simulando una validación walk-forward.
"""

import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Artifact
from kfp import compiler
from typing import List


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def hypertune_step(
    project_id: str,
    location: str,
    train_data_run_id: str,
    eval_data_run_id: str,
    container_uri: str,
    staging_bucket: str,
    best_hyperparameters: Output[Artifact],
    service_account: str = None
) -> str:
    """
    Componente que ejecuta un paso de optimización de hiperparámetros con Vertex AI Hypertune.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        location: Región de Vertex AI (ej: us-central1)
        train_data_run_id: ID del data_run para entrenamiento
        eval_data_run_id: ID del data_run para evaluación
        container_uri: URI del contenedor Docker que ejecuta hypertune_trial.py
        staging_bucket: Bucket de GCS para staging
        best_hyperparameters: Artefacto de salida con los mejores hiperparámetros encontrados
        service_account: Service account para el job (opcional)
        
    Returns:
        str: ID del HyperparameterTuningJob ejecutado
    """
    import google.cloud.aiplatform as aip
    from google.cloud.aiplatform import hyperparameter_tuning as hpt
    import time
    
    # Inicializar Vertex AI
    aip.init(project=project_id, location=location, staging_bucket=staging_bucket)
    
    # Definir las especificaciones del worker
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["python", "hypertune_trial.py"],
                "args": [
                    f"--train-data-run-id={train_data_run_id}",
                    f"--eval-data-run-id={eval_data_run_id}",
                    "--episodes=10",  # Entrenamiento corto para trials
                    # Los hiperparámetros serán inyectados por Hypertune
                ]
            }
        }
    ]
    
    # Crear el Custom Job base
    custom_job = aip.CustomJob(
        display_name=f"hypertune-trial-{train_data_run_id}-to-{eval_data_run_id}",
        worker_pool_specs=worker_pool_specs
    )
    
    # Definir el espacio de búsqueda de hiperparámetros
    parameter_spec = {
        "actor-learning-rate": hpt.DoubleParameterSpec(
            min=1e-5, max=1e-2, scale="log"
        ),
        "critic-learning-rate": hpt.DoubleParameterSpec(
            min=1e-5, max=1e-2, scale="log"
        ),
        "alpha-learning-rate": hpt.DoubleParameterSpec(
            min=1e-5, max=1e-2, scale="log"
        ),
        "batch-size": hpt.DiscreteParameterSpec(
            values=[256, 512, 1024, 2048, 4096, 8192], scale="linear"
        ),
        "tau": hpt.DoubleParameterSpec(
            min=0.001, max=0.01, scale="log"
        ),
        "per-alpha": hpt.DoubleParameterSpec(
            min=0.4, max=0.8, scale="linear"
        ),
        "per-beta": hpt.DoubleParameterSpec(
            min=0.3, max=0.7, scale="linear"
        )
    }
    
    # Definir la métrica objetivo (formato moderno de diccionario)
    metric_spec = {
        "sortino_ratio": "maximize"
    }
    
    # Crear el HyperparameterTuningJob
    hpt_job = aip.HyperparameterTuningJob(
        display_name=f"walk-forward-hpt-{train_data_run_id}-{eval_data_run_id}",
        custom_job=custom_job,
        parameter_spec=parameter_spec,
        metric_spec=metric_spec,
        max_trial_count=4,  # Número máximo de trials
        parallel_trial_count=1,  # Ejecución secuencial para optimización bayesiana
        search_algorithm=None  # Algoritmo bayesiano por defecto de Vertex AI
    )
    
    # Ejecutar el HyperparameterTuningJob
    print(f"🚀 Iniciando optimización de hiperparámetros para {train_data_run_id} -> {eval_data_run_id}")
    hpt_job.run(
        service_account=service_account,
        sync=True,  # Esperar a que termine
        timeout=86400  # Timeout de 24 horas
    )
    
    print(f"✅ Optimización completada. Job ID: {hpt_job.name}")
    
    # Obtener los mejores hiperparámetros
    best_trial = hpt_job.trials[0]  # El primero es el mejor
    print(f"📈 Mejor Sortino Ratio: {best_trial.final_measurement.metrics[0].value}")
    print("🏆 Mejores hiperparámetros:")
    for param in best_trial.parameters:
        print(f"  - {param.parameter_id}: {param.value}")
    
    # Guardar los mejores hiperparámetros en el artefacto de salida
    import json
    
    # Crear diccionario con los mejores parámetros
    best_params_dict = {}
    for param in best_trial.parameters:
        best_params_dict[param.parameter_id] = param.value
    
    # Añadir información adicional del trial
    best_params_dict["sortino_ratio"] = best_trial.final_measurement.metrics[0].value
    best_params_dict["trial_id"] = best_trial.id
    best_params_dict["train_data_run_id"] = train_data_run_id
    best_params_dict["eval_data_run_id"] = eval_data_run_id
    
    # Escribir al archivo del artefacto de salida
    with open(best_hyperparameters.path, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    
    print(f"💾 Mejores hiperparámetros guardados en: {best_hyperparameters.path}")
    
    return hpt_job.name


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def consolidate_results(
    project_id: str,
    location: str,
    hpt_job_id_1: str,
    hpt_job_id_2: str,
    hpt_job_id_3: str
) -> str:
    """
    Componente que consolida los resultados de todos los pasos walk-forward.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        location: Región de Vertex AI
        hpt_job_id_1: ID del primer HyperparameterTuningJob
        hpt_job_id_2: ID del segundo HyperparameterTuningJob
        hpt_job_id_3: ID del tercer HyperparameterTuningJob
        
    Returns:
        str: Resumen de los resultados consolidados
    """
    import google.cloud.aiplatform as aip
    import json
    
    # Inicializar Vertex AI
    aip.init(project=project_id, location=location)
    
    # Construir lista de job IDs manualmente
    hpt_job_ids = [hpt_job_id_1, hpt_job_id_2, hpt_job_id_3]
    
    results_summary = []
    total_avg_sortino = 0.0
    
    print("📊 === CONSOLIDANDO RESULTADOS WALK-FORWARD ===")
    
    for i, job_id in enumerate(hpt_job_ids):
        print(f"\n🔍 Analizando paso {i+1}: {job_id}")
        
        # Cargar el HyperparameterTuningJob
        hpt_job = aip.HyperparameterTuningJob.get(job_id)
        
        # Obtener el mejor trial
        best_trial = hpt_job.trials[0]
        best_sortino = best_trial.final_measurement.metrics[0].value
        
        # Extraer hiperparámetros
        best_params = {}
        for param in best_trial.parameters:
            best_params[param.parameter_id] = param.value
        
        step_result = {
            "step": i + 1,
            "job_id": job_id,
            "best_sortino_ratio": best_sortino,
            "best_hyperparameters": best_params,
            "total_trials": len(hpt_job.trials)
        }
        
        results_summary.append(step_result)
        total_avg_sortino += best_sortino
        
        print(f"  ✅ Mejor Sortino: {best_sortino:.6f}")
        print(f"  🎯 Total trials: {len(hpt_job.trials)}")
    
    # Calcular estadísticas globales
    avg_sortino = total_avg_sortino / len(hpt_job_ids)
    
    final_summary = {
        "walk_forward_steps": len(hpt_job_ids),
        "average_sortino_ratio": avg_sortino,
        "individual_results": results_summary
    }
    
    print(f"\n🏆 === RESUMEN FINAL ===")
    print(f"📈 Sortino Ratio Promedio: {avg_sortino:.6f}")
    print(f"🔢 Pasos Walk-Forward: {len(hpt_job_ids)}")
    
    return json.dumps(final_summary, indent=2)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def full_training_step(
    project_id: str,
    location: str,
    train_data_run_id: str,
    container_uri: str,
    staging_bucket: str,
    best_hyperparameters: Input[Artifact],
    service_account: str = None
) -> str:
    """
    Componente que ejecuta un entrenamiento completo con los mejores hiperparámetros encontrados.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        location: Región de Vertex AI (ej: us-central1)
        train_data_run_id: ID del data_run para entrenamiento completo
        container_uri: URI del contenedor Docker que ejecuta train.py
        staging_bucket: Bucket de GCS para staging
        best_hyperparameters: Artefacto de entrada con los mejores hiperparámetros
        service_account: Service account para el job (opcional)
        
    Returns:
        str: ID único del entrenamiento completo ejecutado
    """
    import google.cloud.aiplatform as aip
    import json
    import time
    from datetime import datetime
    
    # Inicializar Vertex AI
    aip.init(project=project_id, location=location, staging_bucket=staging_bucket)
    
    # Leer y cargar los mejores hiperparámetros desde el artefacto
    with open(best_hyperparameters.path, 'r') as f:
        params_dict = json.load(f)
    
    print(f"📂 Hiperparámetros cargados desde artefacto:")
    for key, value in params_dict.items():
        print(f"  - {key}: {value}")
    
    # Generar un training_run_id único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_run_id = f"full_training_{train_data_run_id}_{timestamp}"
    
    # Construir argumentos dinámicamente desde los hiperparámetros
    training_args = [
        f"--data-run-id={train_data_run_id}",
        f"--run-id={training_run_id}",
        "--episodes=15",  # <-- CAMBIO: Un entrenamiento muy corto para validar.
        "--eval-frequency=10", # <-- AÑADIDO (Opcional): Evaluar más frecuentemente.
        "--save-frequency=10"  # <-- AÑADIDO (Opcional): Guardar más frecuentemente.
    ]
    
    # Añadir hiperparámetros optimizados si están disponibles
    hyperparameter_mappings = {
        "actor-learning-rate": "--actor-learning-rate",
        "critic-learning-rate": "--critic-learning-rate", 
        "alpha-learning-rate": "--alpha-learning-rate",
        "batch-size": "--batch-size",
        "tau": "--tau",
        "per-alpha": "--per-alpha",
        "per-beta": "--per-beta"
    }
    
    for param_key, arg_name in hyperparameter_mappings.items():
        if param_key in params_dict:
            training_args.append(f"{arg_name}={params_dict[param_key]}")
    
    print(f"🎯 Argumentos de entrenamiento: {training_args}")
    
    # Definir las especificaciones del worker con 4 GPUs (replicando vertexai.yaml)
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",  # 8 vCPUs, 30 GB RAM - coincide con vertexai.yaml
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 4,  # 4 GPUs para entrenamiento distribuido
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["torchrun", "--nproc_per_node=4", "train.py"],
                "args": training_args
            }
        }
    ]
    
    # Crear el Custom Job para entrenamiento completo
    custom_job = aip.CustomJob(
        display_name=f"full-training-{training_run_id}",
        worker_pool_specs=worker_pool_specs
    )
    
    # Ejecutar el Custom Job de entrenamiento
    print(f"🚀 Iniciando entrenamiento completo: {training_run_id}")
    print(f"📊 Usando datos de entrenamiento: {train_data_run_id}")
    print(f"🔧 Hiperparámetros optimizados desde trial: {params_dict.get('trial_id', 'N/A')}")
    
    custom_job.run(
        service_account=service_account,
        sync=True,  # Esperar a que termine el entrenamiento
        timeout=172800  # Timeout de 48 horas para entrenamiento completo
    )
    
    print(f"✅ Entrenamiento completo finalizado: {training_run_id}")
    print(f"📈 Job ID: {custom_job.name}")
    
    return training_run_id


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def evaluation_step(
    project_id: str,
    location: str,
    container_uri: str,
    staging_bucket: str,
    training_run_id_to_eval: str,
    eval_data_run_id: str,
    service_account: str = None
) -> str:
    """
    Componente que ejecuta la evaluación de un modelo entrenado.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        location: Región de Vertex AI (ej: us-central1)
        container_uri: URI del contenedor Docker que ejecuta evaluate.py
        staging_bucket: Bucket de GCS para staging
        training_run_id_to_eval: ID del entrenamiento a evaluar
        eval_data_run_id: ID del data_run para evaluación
        service_account: Service account para el job (opcional)
        
    Returns:
        str: ID único de la evaluación ejecutada
    """
    import google.cloud.aiplatform as aip
    from datetime import datetime
    
    # Inicializar Vertex AI
    aip.init(project=project_id, location=location, staging_bucket=staging_bucket)
    
    # Generar un evaluation_id único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_id = f"eval_{training_run_id_to_eval}_{timestamp}"
    
    # Construir argumentos para la evaluación
    evaluation_args = [
        f"--run-id={training_run_id_to_eval}",
        f"--eval-data-run-id={eval_data_run_id}"
    ]
    
    print(f"🎯 Argumentos de evaluación: {evaluation_args}")
    
    # Definir las especificaciones del worker (CPU solamente, no necesita GPU)
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",  # CPU solamente para evaluación
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["python", "evaluate.py"],
                "args": evaluation_args
            }
        }
    ]
    
    # Crear el Custom Job para evaluación
    custom_job = aip.CustomJob(
        display_name=f"evaluation-{evaluation_id}",
        worker_pool_specs=worker_pool_specs
    )
    
    # Ejecutar el Custom Job de evaluación
    print(f"🔍 Iniciando evaluación: {evaluation_id}")
    print(f"🤖 Evaluando modelo: {training_run_id_to_eval}")
    print(f"📊 Usando datos de evaluación: {eval_data_run_id}")
    
    custom_job.run(
        service_account=service_account,
        sync=True,  # Esperar a que termine la evaluación
        timeout=7200  # Timeout de 2 horas para evaluación
    )
    
    print(f"✅ Evaluación completada: {evaluation_id}")
    print(f"📈 Job ID: {custom_job.name}")
    
    return evaluation_id


@dsl.pipeline(
    name="btcbot-walk-forward-hypertuning",
    description="Pipeline de validación walk-forward con optimización de hiperparámetros para btcbot",
    pipeline_root="gs://btcbot-training-2762/pipeline_runs"
)
def walk_forward_pipeline(
    project_id: str,
    location: str,
    data_run_id_1: str,
    data_run_id_2: str,
    data_run_id_3: str,
    data_run_id_4: str,
    container_uri: str,
    staging_bucket: str,
    service_account: str = None
):
    """
    Pipeline principal que orquesta la validación walk-forward.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        location: Región de Vertex AI (ej: us-central1)
        data_run_id_1: Primer bloque de datos (entrenamiento paso 1)
        data_run_id_2: Segundo bloque de datos (evaluación paso 1, entrenamiento paso 2)
        data_run_id_3: Tercer bloque de datos (evaluación paso 2, entrenamiento paso 3)
        data_run_id_4: Cuarto bloque de datos (evaluación paso 3)
        container_uri: URI del contenedor Docker con el código del bot
        staging_bucket: Bucket de GCS para staging
        service_account: Service account para los jobs (opcional)
    """
    
    # Variable de control para encadenar tareas secuencialmente
    previous_step = None
    
    # Paso 1: data_run_id_1 -> data_run_id_2
    hpt_task_1 = hypertune_step(
        project_id=project_id,
        location=location,
        train_data_run_id=data_run_id_1,
        eval_data_run_id=data_run_id_2,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        service_account=service_account
    )
    
    train_task_1 = full_training_step(
        project_id=project_id,
        location=location,
        train_data_run_id=data_run_id_1,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        best_hyperparameters=hpt_task_1.outputs["best_hyperparameters"],
        service_account=service_account
    )
    train_task_1.after(hpt_task_1)
    
    eval_task_1 = evaluation_step(
        project_id=project_id,
        location=location,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        training_run_id_to_eval=train_task_1.output,
        eval_data_run_id=data_run_id_2,
        service_account=service_account
    )
    eval_task_1.after(train_task_1)
    
    # Paso 2: data_run_id_2 -> data_run_id_3
    hpt_task_2 = hypertune_step(
        project_id=project_id,
        location=location,
        train_data_run_id=data_run_id_2,
        eval_data_run_id=data_run_id_3,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        service_account=service_account
    )
    hpt_task_2.after(eval_task_1)  # Secuencial: esperar al paso anterior
    
    train_task_2 = full_training_step(
        project_id=project_id,
        location=location,
        train_data_run_id=data_run_id_2,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        best_hyperparameters=hpt_task_2.outputs["best_hyperparameters"],
        service_account=service_account
    )
    train_task_2.after(hpt_task_2)
    
    eval_task_2 = evaluation_step(
        project_id=project_id,
        location=location,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        training_run_id_to_eval=train_task_2.output,
        eval_data_run_id=data_run_id_3,
        service_account=service_account
    )
    eval_task_2.after(train_task_2)
    
    # Paso 3: data_run_id_3 -> data_run_id_4
    hpt_task_3 = hypertune_step(
        project_id=project_id,
        location=location,
        train_data_run_id=data_run_id_3,
        eval_data_run_id=data_run_id_4,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        service_account=service_account
    )
    hpt_task_3.after(eval_task_2)  # Secuencial: esperar al paso anterior
    
    train_task_3 = full_training_step(
        project_id=project_id,
        location=location,
        train_data_run_id=data_run_id_3,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        best_hyperparameters=hpt_task_3.outputs["best_hyperparameters"],
        service_account=service_account
    )
    train_task_3.after(hpt_task_3)
    
    eval_task_3 = evaluation_step(
        project_id=project_id,
        location=location,
        container_uri=container_uri,
        staging_bucket=staging_bucket,
        training_run_id_to_eval=train_task_3.output,
        eval_data_run_id=data_run_id_4,
        service_account=service_account
    )
    eval_task_3.after(train_task_3)
    
    # Paso final: consolidar resultados de los 3 pasos walk-forward
    consolidate_task = consolidate_results(
        project_id=project_id,
        location=location,
        hpt_job_id_1=hpt_task_1.outputs["Output"],
        hpt_job_id_2=hpt_task_2.outputs["Output"],
        hpt_job_id_3=hpt_task_3.outputs["Output"]
    )
    
    # La consolidación debe ejecutarse después del último paso de evaluación
    consolidate_task.after(eval_task_3)
    
    print("✅ Pipeline walk-forward definido correctamente")


def create_sample_data_run_ids():
    """
    Función helper para generar IDs de ejemplo para testing.
    En producción, estos vendrían de tus datos reales.
    """
    base_symbol = "BTCUSDT_1h"
    base_date = "20250101"
    
    # Generar 6 bloques de datos secuenciales (para 5 pasos walk-forward)
    sample_ids = []
    for i in range(6):
        month = f"{1 + i:02d}"  # Meses 01 al 06
        data_run_id = f"{base_symbol}_{base_date}_{month}_bloque_{i+1}"
        sample_ids.append(data_run_id)
    
    return sample_ids


if __name__ == "__main__":
    # Configuración del pipeline
    PROJECT_ID = "btcbot-2762"
    LOCATION = "us-central1"
    CONTAINER_URI = "gcr.io/btcbot-2762/btcbot:latest"  # Cambiar por tu imagen
    STAGING_BUCKET = "gs://btcbot-training-2762"
    
    # Compilar el pipeline
    print("🔧 Compilando pipeline walk-forward...")
    
    pipeline_compiler = compiler.Compiler()
    pipeline_compiler.compile(
        pipeline_func=walk_forward_pipeline,
        package_path="walk_forward_pipeline.json"
    )
    
    print("✅ Pipeline compilado exitosamente -> walk_forward_pipeline.json")
    print("\n📋 Para ejecutar el pipeline:")
    print(f"1. Sube 'walk_forward_pipeline.json' a Vertex AI Pipelines")
    print(f"2. Configura los parámetros:")
    print(f"   - project_id: {PROJECT_ID}")
    print(f"   - location: {LOCATION}")
    print(f"   - container_uri: {CONTAINER_URI}")
    print(f"   - staging_bucket: {STAGING_BUCKET}")
    print(f"   - data_run_ids: {create_sample_data_run_ids()}")
    print(f"3. Ejecuta el pipeline en la interfaz de Vertex AI")
    
    # Ejemplo de cómo se vería la lista de data_run_ids
    sample_ids = create_sample_data_run_ids()
    print(f"\n🔍 Ejemplo de data_run_ids para 5 pasos walk-forward:")
    for i, run_id in enumerate(sample_ids):
        if i < len(sample_ids) - 1:
            print(f"   Paso {i+1}: {run_id} -> {sample_ids[i+1]}")
