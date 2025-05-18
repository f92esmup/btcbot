# Guía de Despliegue del Bitcoin Trading Bot en GCP

Esta guía proporciona instrucciones detalladas para desplegar el Bitcoin Trading Bot en Google Cloud Platform (GCP) utilizando Vertex AI Pipelines y Cloud Build.

## Prerrequisitos

Antes de comenzar, asegúrate de tener:

1. **Una cuenta de Google Cloud Platform** con facturación habilitada
2. **gcloud CLI** instalado y configurado
3. **Terraform** instalado (versión 1.0.0 o superior)
4. **Python 3.10 o superior**
5. **Una cuenta de Binance** con API key y API secret

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **src/**: Código fuente del bot de trading
  - **agent/**: Implementación del agente de RL y modelo transformer
  - **environments/**: Entorno de trading simulado para entrenamiento
  - **preprocessing/**: Preprocesamiento de datos y feature engineering
  - **backtesting/**: Framework para backtesting
  - **components/**: Componentes independientes del pipeline
- **scripts/**: Scripts para despliegue y pruebas
- **terraform/**: Configuración de infraestructura como código
- **tests/**: Pruebas unitarias

## Pasos para el Despliegue

### 1. Configuración Inicial

Primero, clona el repositorio y configura las variables de entorno:

```bash
# Clonar el repositorio (si aún no lo has hecho)
git clone <url-del-repositorio>
cd btcbot

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Despliegue Completo

Para realizar un despliegue completo en un solo paso, utiliza el script maestro:

```bash
bash scripts/deploy_complete_pipeline.sh
```

Este script ejecutará todos los pasos necesarios en secuencia, solicitando la información requerida en cada etapa.

### 3. Despliegue Paso a Paso

Si prefieres hacer el despliegue paso a paso, sigue estas instrucciones:

#### 3.1. Configurar Terraform

```bash
bash scripts/setup_terraform.sh
```

Este script inicializará Terraform, creará el bucket para el estado y configurará las variables necesarias.

#### 3.2. Actualizar Secretos

```bash
bash scripts/update_secrets.sh
```

Utilizarás este script para guardar de forma segura las credenciales de la API de Binance en Secret Manager.

#### 3.3. Desplegar Infraestructura

```bash
bash scripts/deploy_infrastructure.sh
```

Este script aplica la configuración de Terraform para crear todos los recursos necesarios en GCP.

#### 3.4. Configurar el Trigger de Cloud Build

```bash
bash scripts/setup_cloud_build_trigger.sh
```

Configura la integración continua y despliegue continuo con GitHub.

#### 3.5. Desplegar el Pipeline en Vertex AI

```bash
bash scripts/deploy_to_vertex_ai.sh
```

Compila y despliega el pipeline de machine learning en Vertex AI.

### 4. Monitoreo y Verificación

Para verificar el estado de tu despliegue en cualquier momento:

```bash
bash scripts/check_deployment_status.sh
```

Este script comprobará el estado de todos los componentes desplegados en GCP.

## Recursos Creados

El despliegue crea los siguientes recursos en GCP:

- **Service Account** dedicada con los permisos mínimos necesarios
- **Buckets de Storage** para datos brutos, procesados, modelos y artefactos
- **Repositorio en Artifact Registry** para imágenes Docker
- **Secretos en Secret Manager** para las credenciales de Binance API
- **Vertex AI Tensorboard** para seguimiento de métricas de entrenamiento
- **Trigger de Cloud Build** para CI/CD automático
- **Pipeline de Vertex AI** para orquestar todo el flujo de trabajo

## Siguientes Pasos

Una vez desplegado el pipeline, puedes:

1. **Ver los resultados del backtest** en el bucket de artefactos
2. **Monitorear el entrenamiento** a través de Vertex AI Tensorboard
3. **Modificar parámetros** y volver a ejecutar el pipeline
4. **Implementar mejoras** y dejar que el trigger de CI/CD despliegue los cambios

## Solución de Problemas

Si encuentras problemas durante el despliegue:

1. Verifica el estado con `scripts/check_deployment_status.sh`
2. Consulta los logs en la consola de GCP
3. Asegúrate de que todas las APIs necesarias estén habilitadas
4. Comprueba que las credenciales de Binance sean válidas
