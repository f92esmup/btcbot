# Plan de Desarrollo y Tareas (Basado en Merges)

## Backlog / Ideas Futuras
- Relplay-Buffer usando un método PER.
- Evaluate.py probablemente se pueda mejorar recibiendo todos los datos del modelo desde el run_id. Absolutamente todo.
- La Optimización Profesional (El Siguiente Nivel)
- Si en el futuro tu fase de carga y preprocesamiento de datos se volviera muy lenta y quisieras optimizarla,implementarías el patrón "Chief Prepara, Workers Consumen".Es decir, no se descarga una vez por cada máquina.
- Crear un GEM de gemini que me permita trabajar de forma eficiente en mi proyecto y lo actualicemos.
- Muy a larga vista: Mas de una operacion a la vez -> Más de un simbolo a la vez- >  Capaz de diversificar por si mismo. Creación de un ejambre de agentes de varios niveles:-Nivel 1: son los especialistas (Se corresponde con mi versión actual)-Nivel 2: Metaagente dedicado a la gestión de riesgo y decisión de tomar operaciónes y restringir el acceso de los especialistas.-Nivel 3: Una lógica explicita que cada x tiempo ajusta los parámetros del meta-agente.
- Mi train.py ya esta funcional para todos los casos. Pero para varias máquinas se debe establecer una conexión de red entre ambos y no se permite en el trabajo si no se da acceso explícito. Hay un proboema de red que hay que solucionar.

---

## Merge 4: Deployment de modelo entrenado en vertex ai
*Objetivo: Lanzar un deployment en vertex ai para recibir peticiones y enviar las predicciones.*
- [ ] Tarea 4.1: Crear un `deployment.py` que sirva como un endpoint de inferencia puro en Vertex AI.
  - [ ] 4.1.1: Actualizar `requirements.txt` con `fastapi`, `uvicorn`, `pydantic`, y `google-cloud-storage`.
  - [ ] 4.1.2: Implementar el script `deployment.py` con FastAPI.
    - El endpoint cargará el agente y la configuración desde GCS al iniciar.
    - Expondrá un endpoint `/predict` que recibe un vector de estado ya procesado y normalizado.
    - Validará la dimensión del estado de entrada.
    - Devolverá la acción predicha por el agente en modo determinista.
  - [ ] 4.1.3: Crear un `Dockerfile.deploy` para contenerizar la aplicación de inferencia.
  - [ ] 4.1.4: Desplegar el contenedor en un Vertex AI Endpoint.
- [ ] Tarea 4.2: Configurar o corregir la instancia de Tensorboard.
- [ ] Tarea 4.3: Solucionar el que no funciona la configuración de varias máquinas, crea el acceso vpc que solicita.

## Merge 5: Construcción del modo Live. (Falta discutir)
*Objetivo: Crear la parte que despliesga el código en modo real y realiza las predicciones en tiempo real para realizar operaciones 24/7.*
- [ ] Tarea 5.3: Incluir el telegram bot en el modo live como lugar informativo
- [ ] Tarea 5.2: Configurar el modo live en cloud run. Osea que hay que crear el endpoint.
- [ ] Tarea 5.1: Crear el modo live 

---

## Completado
*Esta sección registrará los merges ya integrados en `main`.*

### Merge 0:
## Merge 1:
## Merge 2: 
## Merge 3: Entrenamiento universal
*Objetivo: Conseguir un train.py capaz de amoldarse a cualquier situación.*
- [X] Tarea 3.1: Renovar mi código para añadir la opicón de end_date en la descarga de datos y así poder hacer posible el entrenamiento de Walkforward.
- [X] Tarea 3.2: Crear un evaluate.py
- [X] Tarea 3.3: Aplicar un fine-tunning para el walk-forward. Este creo que funcionaría con un fin-tunning. Osea, el scaler es en los nuevos datos? 
- [X] Tarea 3.4:  La Optimización de JIT produce error. Hay que eliminarla.
- [X] Tarea 3.5:Tengo que unificar los procesos en un solo script y confirmar que el entrenamiento funciona para cualquier caso, x máquinas con x gpus. 
- [X] Tarea 3.6:  Crear un script que haga entrenamientos mucho más rápidos usando ray on vertex ai. entrenamiento distribuido. No uso ray, uso una forma nativa de pytorch que tiene todos los mismos beneficios y que solo usa el customjob de vertex ai.
- [X] Tarea 3.7:  Incluir una instancia de Tensorboard vertex ai Experiments.



