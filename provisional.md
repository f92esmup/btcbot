Plan de Implementación Acordado


  El objetivo es refactorizar el proyecto para separar la creación de conjuntos de datos (data_runs) del proceso de entrenamiento (training_runs), mejorando la trazabilidad y la reproducibilidad.

  1. Creación de un Script Dedicado para Datos (`create_dataset.py`)


   * Responsabilidad: Crear un nuevo script en la raíz del proyecto, create_dataset.py, cuya única función es ejecutar el DataPipeline.
   * Proceso:
       1. Genera un data_run_id único y una carpeta correspondiente en data_runs/.
       2. Ejecuta el DataPipeline, que procesa los datos según la configuración de config.yaml.
       3. Guarda los tres artefactos inmutables dentro de la carpeta del data_run:
           * normalized_dataframe.parquet
           * scaler.pkl
           * price_scaler.pkl
   * Resultado: Un data_run autocontenido y versionado, listo para ser consumido por múltiples entrenamientos.

  2. Adaptación de la Interfaz de Línea de Comandos (CLI)


   * Responsabilidad: Modificar src/utils/cli.py para que la intención del usuario sea explícita y sin ambigüedades, utilizando los argumentos que ya conoces.
   * Cambios:
       1. Argumentos Mutuamente Excluyentes: Se establece que --data-run-id y --checkpoint no pueden usarse al mismo tiempo.
       2. `--data-run-id <ID>`: Se utiliza exclusivamente para iniciar un entrenamiento nuevo desde cero, consumiendo los artefactos del data_run especificado.
       3. `--checkpoint <training_run_id>`: Se mantiene como el argumento para reanudar un entrenamiento existente o para hacer fine-tuning.
       4. `--fine-tune-mode`: Se conserva y se utiliza en combinación con --checkpoint para indicar que los optimizadores del agente deben ser reiniciados.

  3. Refactorización del Orquestador (`train.py`)


   * Responsabilidad: Convertir train.py en el gestor de training_runs.
   * Lógica Principal:
       1. Modo "Nuevo Entrenamiento":
           * Se activa con --data-run-id.
           * Genera un training_run_id nuevo y su carpeta en training_runs/.
           * Utiliza RunManager para cargar los artefactos del data_run especificado.
           * Crea y guarda un config_training_run.yaml en la nueva carpeta, que incluye una sección de lineage apuntando al data_run_id original.
           * Inicia el entrenamiento desde el episodio 0.
       2. Modo "Reanudar / Fine-Tune":
           * Se activa con --checkpoint <training_run_id>.
           * Genera un nuevo training_run_id para la sesión reanudada (ej: ..._resumed_1).
           * Usa RunManager para leer el config_training_run.yaml del training_run que se va a reanudar.
           * Extrae el `data_run_id` de la sección `lineage` de ese archivo.
           * Carga los artefactos de datos correctos usando el data_run_id recuperado.
           * Carga el último checkpoint del training_run original.
           * La bandera --fine-tune-mode decide si se resetean los optimizadores.
           * Continúa el entrenamiento desde el episodio guardado en el checkpoint.


  4. Flexibilización del Gestor de Archivos (`src/training/run_manager.py`)


   * Responsabilidad: Adaptar RunManager para que pueda operar sobre las dos estructuras de directorios (data_runs/ y training_runs/).
   * Cambios:
       * Sus métodos (load_data_artifacts, load_training_config, find_latest_checkpoint, etc.) se modifican para aceptar IDs específicos de data_run o training_run, permitiéndole apuntar a la ruta correcta
         para cada operación de lectura o escritura.


  ---
  Este es el plan que hemos implementado, logrando una arquitectura robusta y manteniendo una interfaz de usuario clara y potente.
