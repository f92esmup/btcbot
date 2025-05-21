# Prompt para Interactuar con GitHub Copilot - Proyecto BTCBot

Este documento contiene prompts para utilizar con GitHub Copilot cuando desees realizar operaciones específicas en tu sistema BTCBot desplegado en GKE.

## Iniciar Entrenamiento

```
Me gustaría lanzar un entrenamiento de mi modelo RL para trading con [NÚMERO DE TIMESTEPS] pasos. El modelo está configurado en mi proyecto BTCBot en GKE. Por favor, genera los comandos de Kubernetes necesarios y verifica el estado del entrenamiento después de iniciarlo. No uses scripts, genera directamente los comandos de kubectl.
```

## Actualizar Modelo de Inferencia

```
Necesito actualizar el modelo que está utilizando mi servidor de inferencia en GKE. El nuevo modelo está ubicado en gs://lofty-complex-460416-r6/models/sac_transformer_trading_agent/[NOMBRE_DEL_MODELO.zip]. Por favor, genera los comandos para actualizar el despliegue y verificar que el cambio se haya aplicado correctamente.
```

## Verificar Estado del Sistema

```
Por favor, verifica el estado actual de mi sistema BTCBot en GKE. Necesito saber el estado de todos los pods, servicios y jobs en el namespace btcbot. También muéstrame los logs recientes del servidor de inferencia y el trader.
```

## Escalar Componentes

```
Necesito escalar [el servidor de inferencia/el trader] a [NÚMERO] réplicas. Por favor, genera los comandos necesarios para realizar esta operación y luego verifica que se haya aplicado correctamente.
```

## Solucionar Problemas

```
Estoy teniendo problemas con mi sistema BTCBot en GKE. Por favor, ayúdame a diagnosticar el problema ejecutando comandos para verificar el estado de los pods, servicios y logs. Luego, sugiere posibles soluciones basadas en lo que encuentres.
```

## Crear Despliegue desde Cero

```
Necesito desplegar mi sistema BTCBot completo en GKE desde cero. El sistema consta de un servidor de inferencia y un trader en vivo. Por favor, genera los comandos necesarios para crear el namespace, aplicar los manifiestos de Kubernetes y verificar que todo esté funcionando correctamente.
```

## Ver Rendimiento y Métricas

```
Me gustaría ver las métricas de rendimiento de mi sistema BTCBot en GKE. Por favor, genera comandos para verificar el uso de CPU, memoria y otros recursos de los pods, así como cualquier otra métrica útil para evaluar el rendimiento del sistema.
```

## Reconstruir y Actualizar Imágenes

```
Necesito reconstruir mis imágenes Docker y actualizar los despliegues de BTCBot en GKE. Por favor, genera los comandos para reconstruir las imágenes usando Cloud Build y luego actualizar los despliegues existentes para usar las nuevas imágenes.
```

## Limpiar Recursos No Utilizados

```
Por favor, ayúdame a limpiar recursos no utilizados del sistema BTCBot en GKE. Necesito eliminar jobs completados, pods fallidos y cualquier otro recurso que esté consumiendo espacio innecesariamente.
```

## Configurar Node Pool con GPU

```
Necesito configurar un node pool con GPU en mi clúster GKE para entrenar mi modelo RL de BTCBot. Por favor, genera los comandos para crear un node pool con GPU NVIDIA T4 que pueda escalar de 0 a 2 nodos según la demanda.
```

## Asignar Permisos a Cuenta de Servicio

```
Necesito asignar permisos a la cuenta de servicio btcbot-inference-sa para que pueda acceder a mi bucket de Google Cloud Storage. Por favor, genera los comandos necesarios para otorgar permisos de lectura y escritura en el bucket.
```

## Notas importantes

Al usar estos prompts con GitHub Copilot:

1. Reemplaza los valores entre [CORCHETES] con tus valores específicos
2. Revisa los comandos generados antes de ejecutarlos
3. Ten en cuenta que GitHub Copilot podría generar comandos que necesiten ajustes específicos para tu entorno
4. Si los comandos generados no funcionan, proporciona el error específico a Copilot para obtener ayuda en la solución del problema
5. Para entornos de producción, siempre respalda tus datos antes de realizar cambios significativos
