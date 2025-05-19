# Migración a GCP

Este documento describe la migración del sistema btcbot a una arquitectura basada en Google Cloud Platform (GCP). La migración abarca la integración con servicios clave de GCP, eliminación de fallbacks locales y la configuración del sistema para que falle explícitamente si no puede acceder a los servicios de GCP.

## Cambios Implementados

### 1. Gestión de Configuración y Secretos

El sistema ahora utiliza exclusivamente **Secret Manager** para almacenar y recuperar credenciales sensibles:

- Se eliminaron las opciones de fallback a variables de entorno o archivos locales
- Se agregó verificación explícita de acceso a Secret Manager en `ConfigManager`
- Todos los secretos (como API keys de Binance) ahora deben estar almacenados en Secret Manager

### 2. Almacenamiento de Datos Crudos

Los datos de mercado crudos ahora se almacenan exclusivamente en **BigQuery**:

- Se modificó `BinanceFuturesDownloader` para guardar directamente en BigQuery
- Se eliminó el almacenamiento primario en CSV local (solo se guardan copias como log)
- Se implementó la creación automática de datasets si no existen
- El sistema falla explícitamente si no puede cargar datos a BigQuery

### 3. Almacenamiento de Datos Procesados

Los datos procesados ahora se almacenan exclusivamente en **Google Cloud Storage (GCS)**:

- `DataPreprocessor` ahora guarda los datos procesados en GCS
- Se eliminó el almacenamiento primario en archivos locales
- Se implementó la creación automática de buckets si no existen
- El sistema falla explícitamente si no puede cargar datos a GCS

### 4. Entorno de Trading

El entorno de trading ahora carga datos exclusivamente desde **GCS**:

- Se eliminó la opción de fallback a archivos locales
- Se agregó verificación explícita de acceso a GCS
- El sistema falla explícitamente si no puede cargar datos desde GCS

## Estructura de Almacenamiento en GCP

### BigQuery

- **Dataset**: `market_data_raw`
  - **Tablas**: `klines_BTCUSDT_1h`, `klines_ETHUSDT_1h`, etc.
  - Cada tabla contiene los datos OHLCV para un par específico e intervalo

- **Dataset**: `market_data_processed`
  - **Tablas**: `features_BTCUSDT_1h`, `features_ETHUSDT_1h`, etc.
  - Contienen datos procesados con indicadores y características

### Google Cloud Storage

- **Bucket**: `btcbot-raw-data`
  - Respaldo opcional de datos crudos

- **Bucket**: `btcbot-processed-data`
  - Almacenamiento principal de secuencias procesadas para RL
  - Formato: `processed/SYMBOL_INTERVAL_L96_market_features.npz`

- **Bucket**: `btcbot-models`
  - Almacenamiento de modelos entrenados y checkpoints

## Requisitos para ejecución

El sistema ahora requiere:

1. Acceso a GCP con los siguientes servicios habilitados:
   - Secret Manager
   - BigQuery
   - Cloud Storage

2. Las siguientes credenciales deben estar configuradas:
   - `GOOGLE_APPLICATION_CREDENTIALS` (archivo de credenciales de servicio)
   - `GOOGLE_CLOUD_PROJECT` (ID del proyecto GCP)

3. Los siguientes secretos deben estar almacenados en Secret Manager:
   - `binance_api_key`
   - `binance_api_secret`

4. Permisos de IAM adecuados para:
   - Crear y modificar datasets en BigQuery
   - Crear y modificar buckets en GCS
   - Acceder a secretos en Secret Manager

## Script de Prueba

Se ha creado un script de prueba (`scripts/test_gcp_integration.py`) para verificar que todos los componentes del sistema estén correctamente integrados con GCP. Este script prueba:

1. La conexión a Secret Manager
2. La descarga y almacenamiento de datos en BigQuery
3. El procesamiento y almacenamiento de datos en GCS
4. La carga de datos desde GCS para el entorno de trading

Para ejecutar el script:

```bash
python scripts/test_gcp_integration.py
```

## Comportamiento de Error

En caso de que el sistema no pueda acceder a los servicios de GCP:

1. No se intentará usar alternativas locales
2. Se lanzará una excepción explícita indicando el error de acceso a GCP
3. El proceso terminará sin proceder con operaciones locales

Este comportamiento asegura que el sistema o bien opere completamente en GCP o falle de manera clara y explícita.
