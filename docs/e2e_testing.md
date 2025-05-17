# Prueba End-to-End del Pipeline en GCP

Este documento describe cómo ejecutar una prueba end-to-end (E2E) del pipeline completo de BTCBot en Google Cloud Platform.

## Objetivo de la Prueba E2E

La prueba end-to-end tiene como objetivo verificar que todos los componentes del sistema funcionan correctamente juntos en el entorno de GCP. Esto incluye:

1. Verificar que la configuración de GCP es correcta
2. Asegurar que las APIs necesarias están habilitadas
3. Comprobar que la imagen Docker se construye y publica correctamente
4. Validar que el pipeline de Kubeflow se ejecuta sin errores
5. Confirmar que los datos fluyen correctamente entre componentes
6. Verificar que el modelo se entrena y evalúa correctamente

## Requisitos Previos

Antes de ejecutar la prueba E2E, asegúrate de:

1. Tener una cuenta de GCP con facturación habilitada
2. Tener instalado y configurado Google Cloud SDK
3. Tener permisos adecuados en el proyecto de GCP
4. Tener configuradas las variables de entorno necesarias

## Configuración de Variables de Entorno

Exporta las siguientes variables de entorno o configúralas en un archivo `.env`:

```bash
export GCP_PROJECT_ID="tu-proyecto-id"
export GCP_REGION="europe-southwest1"
export BINANCE_API_KEY="tu-clave-api"
export BINANCE_API_SECRET="tu-secreto-api"
```

## Ejecutando la Prueba E2E

La prueba E2E está automatizada mediante el script `test_e2e_pipeline.sh`:

```bash
cd gcp
./test_e2e_pipeline.sh
```

Este script realiza las siguientes acciones:

1. Verifica la configuración de GCP
2. Habilita las APIs necesarias utilizando `enable_apis.sh`
3. Construye y publica la imagen Docker con tag `e2e-test`
4. Ejecuta el pipeline de entrenamiento con parámetros simplificados:
   - Período de tiempo reducido (solo 1 mes de datos)
   - Ventana de lookback más pequeña (24 en lugar de 96)
   - Menos pasos de entrenamiento (10,000 en lugar de 500,000)
   - Menos episodios de evaluación (2 en lugar de 10)

## Parámetros de la Prueba E2E

La prueba utiliza valores reducidos para completarse rápidamente:

- **Símbolo**: BTCUSDT
- **Timeframe**: 1h
- **Rango de fechas**: 2023-01-01 a 2023-02-01 (solo 1 mes)
- **Lookback window**: 24 (reducido)
- **Total timesteps**: 10,000 (reducido)
- **Episodios de evaluación**: 2 (reducido)

## Interpretación de Resultados

Al finalizar la prueba, se mostrará un mensaje de éxito o fracaso:

- **Éxito**: "✅ PRUEBA E2E COMPLETADA EXITOSAMENTE"
- **Fracaso**: "❌ PRUEBA E2E FALLIDA"

En caso de éxito, puedes verificar los resultados en:
- La consola de GCP > Vertex AI > Pipelines
- Los buckets de GCS donde se almacenan los datos y modelos

## Solución de Problemas

Si la prueba falla, puedes:

1. Revisar los logs del script para identificar dónde ocurrió el error
2. Ver los logs de los componentes individuales en la consola de GCP
3. Verificar que todas las APIs necesarias están habilitadas
4. Comprobar que tienes permisos suficientes en el proyecto
5. Asegurarte de que las credenciales de Binance son válidas

## Limpieza

Después de ejecutar la prueba, puedes limpiar los recursos creados para evitar cargos innecesarios:

```bash
python 07_cleanup_resources.py --force
```

Alternativamente, puedes eliminar solo los recursos creados durante la prueba E2E:

```bash
# Eliminar solo la imagen con tag e2e-test
gcloud artifacts docker images delete \
  ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/btcbot-docker-repo/btcbot-pipeline:e2e-test \
  --quiet
```

## Automatización en CI/CD

La prueba E2E también puede integrarse en un flujo de CI/CD para validar automáticamente cambios en el código antes de desplegarlos a producción. Esto asegura que las modificaciones no rompen el funcionamiento del sistema completo.

En un entorno de CI/CD, podrías ejecutar la prueba E2E después de que las pruebas unitarias y de integración hayan pasado exitosamente.
