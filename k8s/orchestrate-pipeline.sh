#!/bin/bash
# k8s/orchestrate-pipeline.sh
# Script para orquestar la ejecución secuencial de los jobs de ML pipeline

set -e

# Configuración
NAMESPACE="btcbot"
DATA_ACQUISITION_JOB="btcbot-data-acquisition"
PREPROCESSING_JOB="btcbot-data-preprocessing"
TRAINING_JOB="btcbot-model-training"

# Función para esperar a que un job termine exitosamente
wait_for_job_completion() {
    local job_name=$1
    local namespace=$2
    
    echo "Esperando a que el job '$job_name' complete..."
    
    # Esperar hasta que el job termine (exitoso o fallido)
    kubectl wait --for=condition=complete job/$job_name -n $namespace --timeout=3600s
    
    # Verificar que el job fue exitoso
    job_status=$(kubectl get job $job_name -n $namespace -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
    
    if [ "$job_status" = "True" ]; then
        echo "✅ Job '$job_name' completado exitosamente"
        return 0
    else
        echo "❌ Job '$job_name' falló"
        kubectl describe job $job_name -n $namespace
        kubectl logs -l job-name=$job_name -n $namespace --tail=50
        return 1
    fi
}

# Función para limpiar jobs anteriores
cleanup_previous_jobs() {
    echo "Limpiando jobs anteriores..."
    
    # Eliminar jobs anteriores si existen (ignorar errores)
    kubectl delete job $PREPROCESSING_JOB -n $NAMESPACE --ignore-not-found=true
    kubectl delete job $TRAINING_JOB -n $NAMESPACE --ignore-not-found=true
    
    echo "✅ Limpieza completada"
}

# Función principal de orquestación
run_ml_pipeline() {
    echo "🚀 Iniciando pipeline de ML de btcbot..."
    
    # Limpiar jobs anteriores
    cleanup_previous_jobs
    
    # 1. Verificar que el CronJob de adquisición existe (se ejecuta automáticamente)
    if ! kubectl get cronjob $DATA_ACQUISITION_JOB -n $NAMESPACE > /dev/null 2>&1; then
        echo "❌ CronJob de adquisición de datos no encontrado. Aplicar k8s/data-acquisition-job.yaml primero"
        exit 1
    fi
    
    # Opcionalmente, ejecutar el job de adquisición manualmente para testing
    if [ "${1:-}" = "--run-acquisition" ]; then
        echo "Ejecutando job de adquisición de datos manualmente..."
        kubectl create job --from=cronjob/$DATA_ACQUISITION_JOB $DATA_ACQUISITION_JOB-manual -n $NAMESPACE
        wait_for_job_completion "$DATA_ACQUISITION_JOB-manual" $NAMESPACE
        kubectl delete job $DATA_ACQUISITION_JOB-manual -n $NAMESPACE
    fi
    
    # 2. Ejecutar job de preprocesamiento
    echo "📊 Iniciando job de preprocesamiento de datos..."
    kubectl apply -f k8s/data-preprocessing-job.yaml -n $NAMESPACE
    wait_for_job_completion $PREPROCESSING_JOB $NAMESPACE
    
    # 3. Ejecutar job de entrenamiento
    echo "🧠 Iniciando job de entrenamiento del modelo..."
    kubectl apply -f k8s/model-training-job.yaml -n $NAMESPACE
    wait_for_job_completion $TRAINING_JOB $NAMESPACE
    
    echo "🎉 Pipeline de ML completado exitosamente!"
    echo "El modelo entrenado está disponible en GCS y el bot de trading se actualizará automáticamente"
}

# Mostrar ayuda
show_help() {
    echo "Uso: $0 [--run-acquisition] [--help]"
    echo ""
    echo "Opciones:"
    echo "  --run-acquisition  Ejecutar también el job de adquisición manualmente"
    echo "  --help            Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0                    # Ejecutar solo preprocesamiento y entrenamiento"
    echo "  $0 --run-acquisition # Ejecutar pipeline completo incluyendo adquisición"
}

# Procesar argumentos
case "${1:-}" in
    --help)
        show_help
        exit 0
        ;;
    --run-acquisition|"")
        run_ml_pipeline "$1"
        ;;
    *)
        echo "❌ Argumento desconocido: $1"
        show_help
        exit 1
        ;;
esac
