#!/bin/bash
# Script de inicio para el contenedor en GCP
# Soporta múltiples modos de operación basados en la variable MODE

set -e

echo "Iniciando contenedor en modo: $MODE"

case "$MODE" in
  train)
    echo "Ejecutando entrenamiento"
    exec python scripts/train_rl_agent.py "$@"
    ;;
  
  serve)
    echo "Iniciando servidor para servir el modelo"
    # Para Vertex AI Prediction o Cloud Run
    exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 server:app
    ;;
  
  download)
    echo "Ejecutando descarga de datos"
    exec python scripts/download_data.py "$@"
    ;;
  
  preprocess)
    echo "Ejecutando preprocesamiento de datos"
    exec python scripts/preprocess_data.py "$@"
    ;;
    
  evaluate)
    echo "Ejecutando evaluación del modelo"
    exec python scripts/evaluate_rl_agent.py "$@"
    ;;
    
  *)
    echo "Modo no reconocido: $MODE"
    echo "Opciones válidas: train, serve, download, preprocess, evaluate"
    exit 1
    ;;
esac
