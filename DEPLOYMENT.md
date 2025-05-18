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

### 2. Uso del Makefile

El proyecto incluye un Makefile completo para facilitar todas las operaciones de despliegue, entrenamiento y monitoreo. Puedes ver los comandos disponibles con:

```bash
make help
```

Ejemplos de comandos útiles:

```bash
# Desplegar toda la infraestructura en GCP
make deploy-all

# Entrenar modelo localmente
make train-local

# Verificar estado del despliegue
make check-status

# Monitorear el despliegue en tiempo real
make monitor
```

### 3. Despliegue Completo

Para realizar un despliegue completo en un solo paso, puedes utilizar:

```bash
make deploy-all
```

O ejecutar el script maestro directamente:

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

1. Verifica el estado con `make check-status` o `scripts/check_deployment_status.sh`
2. Consulta los logs en la consola de GCP
3. Asegúrate de que todas las APIs necesarias estén habilitadas
4. Comprueba que las credenciales de Binance sean válidas

## Scripts Adicionales

El proyecto incluye varios scripts adicionales para ayudarte a gestionar y monitorear la implementación:

### Monitor Interactivo

```bash
make monitor
# o directamente
bash scripts/monitor_deployment.sh
```

Este script proporciona una interfaz interactiva para monitorear todos los aspectos del despliegue, incluyendo:
- Estado del despliegue
- Listado de recursos en GCP
- Pipelines en ejecución
- Logs de Cloud Build
- Contenido de los buckets

### Trading Programado

Para configurar el trading programado en producción:

```bash
make scheduled-trading
# o directamente
bash scripts/scheduled_trading.sh
```

Este script está diseñado para ser ejecutado periódicamente mediante un cron job, realizando operaciones de trading automáticamente según el modelo entrenado.

### Alertas de Trading

Para configurar alertas cuando se realizan operaciones:

```bash
make alerts
# o directamente
bash scripts/send_trading_alerts.sh
```

Este script envía alertas a servicios como Slack o Telegram cuando el modelo realiza operaciones importantes.
- Logs de Cloud Build
- Contenido de los buckets de almacenamiento
- Verificación de secretos

### Listado de Recursos

```bash
bash scripts/list_gcp_resources.sh
```

Genera un listado detallado de todos los recursos desplegados en GCP para el Bitcoin Trading Bot.

### Trading Programado

```bash
bash scripts/scheduled_trading.sh
```

Este script está diseñado para ejecutarse periódicamente (por ejemplo, mediante un cron job) para realizar operaciones de trading automáticas utilizando el modelo entrenado. Descarga el modelo desde GCS, obtiene los datos más recientes de Binance, y ejecuta el trading según la predicción del modelo.

Para configurarlo como un cron job:

```bash
# Editar la tabla de cron
crontab -e

# Añadir una entrada para ejecutar el script cada hora
0 * * * * cd /ruta/a/btcbot && bash scripts/scheduled_trading.sh
```

### Ejecución Directa del Modelo

```bash
bash scripts/run_trained_model.sh
```

Ejecuta el modelo entrenado localmente con datos simulados para verificar su funcionamiento.

### Alertas de Trading

```bash
bash scripts/send_trading_alerts.sh
```

Este script lee los resultados del último trading y envía alertas a servicios externos como Slack o Telegram cuando se detectan operaciones importantes (compras o ventas). Para utilizarlo:

1. Configura los webhooks o tokens necesarios:
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
   export TELEGRAM_BOT_TOKEN="0000000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
   export TELEGRAM_CHAT_ID="-0000000000000"
   ```

2. Ejecuta el script después del trading automatizado:
   ```bash
   bash scripts/scheduled_trading.sh && bash scripts/send_trading_alerts.sh
   ```

Para una integración completa, puedes configurar estos scripts en cron para que se ejecuten periódicamente.
