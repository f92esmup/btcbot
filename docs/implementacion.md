
# Guía de Implementación Detallada: BTC-Transformer-RL-Trader

**Versión del Documento:** 1.0
**Fecha de Creación:** 15 de mayo de 2025
**Autor del Proyecto:** Pedro Escudero Murcia
**Autor de la Guía:** Gemini AI

## Propósito

Esta guía proporciona una hoja de ruta exhaustiva y detallada para la implementación del proyecto "BTC-Transformer-RL-Trader". Está diseñada para ser interpretada por desarrolladores y herramientas de asistencia de IA (como GitHub Copilot) para facilitar un desarrollo preciso y paso a paso. Cada aspecto del `README.md` original y las consideraciones adicionales se traducen aquí en acciones concretas de implementación.

## Estructura General del Proyecto (Consideraciones Adicionales)

Antes de comenzar con las fases, establecemos la estructura de directorios principal que se utilizará, incorporando tus consideraciones:

```

btc-transformer-rl-trader/
├── .git/
├── .github/                    \# Opcional: para workflows de GitHub Actions
│   └── workflows/
│       └── python\_lint\_test.yml \# Ejemplo de CI
├── config/                     \# Configuración de módulos y parámetros
│   ├── module1\_data\_acquisition/
│   │   └── params.yaml
│   ├── module2\_preprocessing/
│   │   └── params.yaml
│   ├── module3\_environment/
│   │   └── params.yaml
│   ├── module4\_agent\_sac/
│   │   └── params.yaml
│   ├── module7\_backtesting\_wfo/
│   │   └── params.yaml
│   └── logging\_config.yaml     \# Configuración global de logging
├── data\_host/                  \# Datos brutos, procesados, etc. (mapeado desde Docker)
│   ├── raw/                    \# Datos brutos descargados
│   │   └── btc\_usdt\_futures/
│   ├── processed/              \# Datos procesados listos para el entrenamiento
│   │   └── btc\_usdt\_futures/
│   └── ...
├── results\_host/               \# Resultados de backtesting, modelos, logs (mapeado desde Docker)
│   ├── backtesting\_reports/
│   │   └── wfo/
│   ├── trained\_models/
│   │   └── sac\_transformer/
│   └── logs/
├── scripts/                    \# Scripts ejecutables para orquestar tareas
│   ├── download\_data.py
│   ├── preprocess\_data.py
│   ├── train\_agent.py          \# Podría ser parte de run\_backtest.py
│   ├── run\_backtest.py
│   └── (otros scripts de utilidad o ejecución)
├── src/                        \# Código fuente principal (clases y lógica modular)
│   ├── **init**.py
│   ├── common/                 \# Utilidades comunes (ej. logging, helpers)
│   │   ├── **init**.py
│   │   └── utils.py
│   ├── config\_loader.py        \# Módulo 9 Lógica
│   ├── data\_acquisition/       \# Módulo 1 Lógica
│   │   ├── **init**.py
│   │   └── binance\_downloader.py
│   ├── preprocessing/          \# Módulo 2 Lógica
│   │   ├── **init**.py
│   │   └── feature\_engineer.py
│   ├── environment/            \# Módulo 3 Lógica
│   │   ├── **init**.py
│   │   └── trading\_env.py
│   ├── agent/                  \# Módulo 4 Lógica
│   │   ├── **init**.py
│   │   ├── models/             \# Definiciones de arquitecturas (Transformer, MLP)
│   │   │   ├── **init**.py
│   │   │   └── transformer\_model.py
│   │   └── sac\_agent.py
│   ├── backtesting/            \# Módulo 7 Lógica
│   │   ├── **init**.py
│   │   └── wfo\_framework.py
│   └── main\_orchestrator.py    \# Opcional: una clase que los scripts pueden usar para flujos complejos
├── tests/                      \# Pruebas unitarias e de integración
│   ├── **init**.py
│   ├── common/
│   │   └── test\_utils.py
│   ├── data\_acquisition/
│   │   └── test\_binance\_downloader.py
│   └── (más tests por módulo...)
├── .dockerignore
├── .env.example                \# Plantilla para el archivo .env
├── .gitignore
├── Dockerfile                  \# Para la aplicación workhorse\_app
├── docker-compose.yml
├── LICENSE
├── README.md                   \# El README original del proyecto (puede ser este mismo documento o el original)
└── requirements.txt            \# Dependencias de Python

````

---

## Fase de Implementación 0: Configuración Inicial del Proyecto y Entorno de Desarrollo

**Nombre Descriptivo de la Fase:** Establecimiento de la Fundación del Proyecto y Entorno de Desarrollo con Docker.

Esta fase se centra en crear la estructura básica del proyecto, configurar el control de versiones, definir las dependencias iniciales y preparar el entorno de Docker para el desarrollo y la ejecución.

---

### Paso 1: Inicialización del Repositorio y Estructura de Directorios Base

* **Descripción Exhaustiva**: Crear el directorio raíz del proyecto, inicializar un repositorio Git, y crear la estructura de carpetas principal según lo definido anteriormente (`config`, `data_host`, `results_host`, `scripts`, `src`, `tests`).
* **Acciones Específicas**:
    * **1.1. Crear Directorio Raíz**:
        ```bash
        mkdir btc-transformer-rl-trader
        cd btc-transformer-rl-trader
        ```
    * **1.2. Inicializar Repositorio Git**:
        ```bash
        git init
        ```
    * **1.3. Crear Archivo `.gitignore`**:
        * Crear un archivo `.gitignore` en la raíz del proyecto.
        * Contenido inicial sugerido para `.gitignore`:
            ```gitignore
            # Byte-compiled / optimized / DLL files
            __pycache__/
            *.py[cod]
            *$py.class

            # C extensions
            *.so

            # Distribution / packaging
            .Python
            build/
            develop-eggs/
            dist/
            downloads/
            eggs/
            .eggs/
            lib/
            lib64/
            parts/
            sdist/
            var/
            wheels/
            pip-wheel-metadata/
            share/python-wheels/
            *.egg-info/
            .installed.cfg
            *.egg
            MANIFEST

            # PyInstaller
            #  Usually these files are written by a script go generate PE file or bundle
            #  Alternatively, PyInstaller dynamically generates these files during build process
            #  Data specific to PyInstaller runtime needed to find resources.
            *_MEI*

            # Environments
            .env
            .venv
            env/
            venv/
            ENV/
            env.bak
            venv.bak

            # Docker
            .dockerignore
            docker-compose.yml  # Puede ser versionado si no contiene secretos
            # Si docker-compose.yml SÍ contiene secretos o configuraciones específicas del host, añádelo.
            # De lo contrario, es mejor versionarlo y usar overrides o .env para personalización.

            # Data files
            data_host/
            results_host/

            # IDE / Editor specific
            .idea/
            .vscode/
            *.suo
            *.ntvs*
            *.njsproj
            *.sln
            *.sw?

            # Jupyter Notebook
            .ipynb_checkpoints

            # Log files
            logs/
            *.log

            # Test reports
            htmlcov/
            .tox/
            .coverage
            .coverage.*
            .cache
            nosetests.xml
            coverage.xml
            *.cover
            .hypothesis/
            ```
    * **1.4. Crear Estructura de Directorios Base**:
        ```bash
        mkdir config data_host results_host scripts src tests
        mkdir src/common src/data_acquisition src/preprocessing src/environment src/agent src/agent/models src/backtesting
        mkdir tests/common tests/data_acquisition tests/preprocessing tests/environment tests/agent tests/agent/models tests/backtesting
        touch src/__init__.py src/common/__init__.py src/data_acquisition/__init__.py src/preprocessing/__init__.py src/environment/__init__.py src/agent/__init__.py src/agent/models/__init__.py src/backtesting/__init__.py
        touch tests/__init__.py tests/common/__init__.py tests/data_acquisition/__init__.py tests/preprocessing/__init__.py tests/environment/__init__.py tests/agent/__init__.py tests/agent/models/__init__.py tests/backtesting/__init__.py
        ```
    * **1.5. Crear archivos README.md y LICENSE (Opcional pero recomendado)**:
        * Crear un `README.md` básico (puede ser este mismo documento una vez finalizado).
        * Añadir un archivo `LICENSE` (ej. MIT, Apache 2.0).

---

### Paso 2: Definición de Dependencias Iniciales (`requirements.txt`)

* **Descripción Exhaustiva**: Crear un archivo `requirements.txt` listando las bibliotecas Python principales que se utilizarán en el proyecto, según el `README.md`. Se pueden añadir versiones específicas si se conocen o dejar que pip instale las últimas compatibles.
* **Acciones Específicas**:
    * **2.1. Crear `requirements.txt`**:
        * En la raíz del proyecto, crear el archivo `requirements.txt`.
        * Contenido inicial (versiones son ejemplos, ajustar según necesidad o usar `>=` para versiones mínimas, la fecha del proyecto es Q2 2025, por lo que se asumen versiones relativamente recientes):
            ```txt
            # Core Data Science & Numerics
            python>=3.9,<3.12 # Especificar un rango compatible
            numpy~=1.26.0
            pandas~=2.2.0
            scikit-learn~=1.4.0

            # Deep Learning - PyTorch
            # Visitar [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) para obtener el comando exacto según CUDA si se usa GPU.
            # Ejemplo para CPU / o una versión de CUDA específica:
            torch~=2.2.0
            torchvision~=0.17.0 # Si se necesita, aunque no explícito para este proyecto
            torchaudio~=2.2.0 # Si se necesita

            # Reinforcement Learning - Stable Baselines3
            stable-baselines3[extra]~=2.3.0 # [extra] incluye soporte para gymnasium, ale, etc.

            # Gymnasium (antes OpenAI Gym)
            gymnasium~=0.29.0
            gymnasium[atari] # Si se necesitan entornos Atari, no para este proyecto
            gymnasium[box2d] # Si se necesitan entornos Box2D, no para este proyecto

            # Binance API
            python-binance~=1.0.19

            # TA-Lib (Technical Analysis Library)
            # La instalación de TA-Lib puede ser compleja.
            # Asegurar que las dependencias de sistema de TA-Lib estén instaladas antes de 'pip install TA-Lib'
            # Ver: [https://github.com/mrjbq7/ta-lib](https://github.com/mrjbq7/ta-lib)
            TA-Lib~=0.4.28

            # Redis client
            redis~=5.0.0

            # Configuration Management
            PyYAML~=6.0.0
            python-dotenv~=1.0.0

            # Backtesting Analysis & Plotting
            matplotlib~=3.8.0
            seaborn~=0.13.0
            plotly~=5.20.0
            quantstats~=0.0.61

            # Linting and Formatting (Desarrollo)
            flake8~=7.0.0
            black~=24.3.0
            isort~=5.12.0

            # Testing (Desarrollo)
            pytest~=8.0.0
            pytest-cov~=5.0.0

            # Jupyter (Opcional, para experimentación)
            notebook~=7.0.0
            jupyterlab~=4.0.0
            ```
    * **2.2. (Opcional) Crear un Entorno Virtual y Probar Instalación**:
        ```bash
        python -m venv .venv
        source .venv/bin/activate # En Linux/macOS
        # .venv\Scripts\activate # En Windows

        pip install --upgrade pip
        pip install -r requirements.txt
        ```

---

### Paso 3: Configuración de Archivo `.env.example`

* **Descripción Exhaustiva**: Crear un archivo `.env.example` en la raíz del proyecto. Este archivo servirá como plantilla para que los usuarios creen su propio archivo `.env` con las credenciales y configuraciones específicas del entorno.
* **Acciones Específicas**:
    * **3.1. Crear `.env.example`**:
        * Contenido basado en la Sección 5 del `README.md`:
            ```env
            # Binance API Credentials
            BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
            BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"

            # Redis Configuration
            REDIS_HOST="localhost" # o "redis" si se ejecuta en Docker Compose con ese nombre de servicio
            REDIS_PORT="6379"
            # REDIS_PASSWORD="YOUR_REDIS_PASSWORD" # Descomentar si Redis tiene contraseña

            # Host Directory Mappings (para Docker Compose)
            # Estas rutas son desde la perspectiva del HOST donde se ejecuta 'docker-compose up'
            # Deben apuntar a las carpetas creadas en el Paso 1.
            DATA_DIR_HOST="./data_host"
            RESULTS_DIR_HOST="./results_host"
            CONFIG_DIR_HOST="./config" # Para montar la configuración en el contenedor
            SRC_DIR_HOST="./src"       # Para montar el código fuente en el contenedor para desarrollo

            # Logging Configuration (opcional, puede ser manejado por logging_config.yaml)
            LOG_LEVEL="INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

            # Otros parámetros globales si los hubiera
            PROJECT_NAME="BTC_Transformer_RL_Trader"
            ```
    * **3.2. Instruir al usuario para crear su `.env`**:
        * Añadir una nota en el `README.md` principal o en un `CONTRIBUTING.md` para que los usuarios copien `.env.example` a `.env` y completen los valores.
        * Asegurarse de que `.env` esté en `.gitignore`.

---

### Paso 4: Creación del `Dockerfile` Base para la Aplicación `workhorse_app`

* **Descripción Exhaustiva**: Crear un `Dockerfile` que defina la imagen de Docker para la aplicación principal (`workhorse_app`). Esta imagen contendrá Python, las dependencias del proyecto y el código fuente.
* **Acciones Específicas**:
    * **4.1. Crear `Dockerfile`**:
        * En la raíz del proyecto.
        * Contenido inicial:
            ```dockerfile
            # Fase 1: Build TA-Lib (si es necesario de forma separada para algunas arquitecturas o para mayor control)
            # O, más comúnmente, instalar dependencias de sistema en la imagen principal.

            # Usar una imagen base oficial de Python.
            # Escoger una versión de Python que coincida con la especificada en requirements.txt (ej. 3.10)
            FROM python:3.10-slim-bullseye AS base

            # Establecer variables de entorno
            ENV PYTHONDONTWRITEBYTECODE 1
            ENV PYTHONUNBUFFERED 1
            ENV PIP_NO_CACHE_DIR off
            ENV PIP_DISABLE_PIP_VERSION_CHECK on

            # Instalar dependencias del sistema necesarias para TA-Lib y otras bibliotecas
            # Las dependencias exactas pueden variar ligeramente según la imagen base y las bibliotecas
            RUN apt-get update && \
                apt-get install -y --no-install-recommends \
                build-essential \
                wget \
                unzip \
                # Dependencias para TA-Lib (ejemplo para Debian/Ubuntu based)
                libta-lib0 \
                # Si libta-lib0 no está disponible o se requiere compilar:
                # curl ca-certificates gnupg software-properties-common make gcc g++ dpkg-dev
                # Para OpenCV (si se añadiera después, no en requirements.txt actual)
                # libgl1-mesa-glx libglib2.0-0
                && rm -rf /var/lib/apt/lists/*

            # (Opcional, si se necesita compilar TA-Lib desde fuente si el paquete libta-lib0 no funciona bien o no está)
            # RUN wget [http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz) && \
            #     tar -xvzf ta-lib-0.4.0-src.tar.gz && \
            #     cd ta-lib/ && \
            #     ./configure --prefix=/usr && \
            #     make && \
            #     make install && \
            #     cd .. && \
            #     rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

            # Crear directorio de trabajo
            WORKDIR /app

            # Copiar archivo de requerimientos e instalar dependencias de Python
            COPY requirements.txt requirements.txt
            RUN pip install --no-cache-dir -r requirements.txt

            # Copiar el resto de la aplicación (código fuente, configuración)
            # Esto se hará en docker-compose.yml mediante volúmenes para desarrollo,
            # pero para una imagen de "producción" o una imagen autocontenida, se copiaría aquí.
            # COPY ./src /app/src
            # COPY ./scripts /app/scripts
            # COPY ./config /app/config

            # Establecer el directorio de trabajo para los scripts
            # WORKDIR /app/scripts

            # Comando por defecto (puede ser sobrescrito en docker-compose.yml)
            # CMD ["python", "main.py"] # Suponiendo que tendrás un main.py en scripts
            CMD ["tail", "-f", "/dev/null"] # Mantiene el contenedor corriendo para desarrollo/exec

            # Exponer puertos si la aplicación sirve algo (no es el caso para el backtester MVP directamente)
            # EXPOSE 8000
            ```
    * **4.2. Crear Archivo `.dockerignore`**:
        * En la raíz del proyecto.
        * Para evitar copiar archivos innecesarios al contexto de build de Docker.
            ```dockerignore
            .git
            .gitignore
            .idea
            .vscode
            __pycache__
            *.pyc
            *.pyo
            *.pyd
            .Python
            build/
            develop-eggs/
            dist/
            downloads/
            eggs/
            .eggs/
            lib/
            lib64/
            parts/
            sdist/
            var/
            wheels/
            *.egg-info/
            .installed.cfg
            *.egg
            .env
            .venv
            env/
            venv/
            ENV/
            env.bak
            venv.bak
            data_host/
            results_host/
            tests/
            *.log
            *.db
            *.sqlite3
            docker-compose.yml
            Dockerfile
            README.md
            # Añade cualquier otro archivo o directorio que no deba ir al contexto de build
            ```

---

### Paso 5: Creación del Archivo `docker-compose.yml` Base

* **Descripción Exhaustiva**: Crear un archivo `docker-compose.yml` para definir y orquestar los servicios necesarios para el MVP: la aplicación `workhorse_app` y `redis`. Se configurarán volúmenes para el código fuente, datos, resultados y configuración.
* **Acciones Específicas**:
    * **5.1. Crear `docker-compose.yml`**:
        * En la raíz del proyecto.
        * Contenido basado en Módulo 10 y consideraciones de desarrollo:
            ```yaml
            version: '3.8' # Usar una versión reciente de Docker Compose

            services:
              redis:
                image: "redis:7.2-alpine" # Usar una versión específica y ligera de Redis
                container_name: btc_trader_redis
                ports:
                  - "${REDIS_PORT:-6379}:6379" # Mapea el puerto de Redis al host, usa variable de .env con default
                volumes:
                  - redis_data:/data # Volumen persistente para datos de Redis (opcional para MVP si solo es caché temporal)
                # command: redis-server --save 20 1 --loglevel warning # Ejemplo de configuración de Redis
                restart: unless-stopped
                healthcheck:
                  test: ["CMD", "redis-cli", "ping"]
                  interval: 10s
                  timeout: 5s
                  retries: 5

              workhorse_app:
                build:
                  context: . # El contexto es el directorio raíz del proyecto
                  dockerfile: Dockerfile # Especifica el Dockerfile a usar
                container_name: btc_trader_app
                env_file:
                  - .env # Carga variables de entorno desde el archivo .env
                environment:
                  # Se pueden añadir o sobreescribir variables de entorno aquí si es necesario
                  PYTHONPATH: "/app/src:/app" # Asegura que Python pueda encontrar módulos en src
                  # REDIS_HOST: "redis" # Ya debería estar en .env, pero importante para la comunicación entre contenedores
                volumes:
                  # Montar código fuente para desarrollo en vivo
                  # La ruta del host viene de .env o se define directamente
                  - ${SRC_DIR_HOST:-./src}:/app/src # Mapea src local a /app/src en el contenedor
                  - ${CONFIG_DIR_HOST:-./config}:/app/config # Mapea config local a /app/config
                  - ${SCRIPTS_DIR_HOST:-./scripts}:/app/scripts # Mapea scripts local a /app/scripts
                  # Montar directorios de datos y resultados
                  - ${DATA_DIR_HOST:-./data_host}:/app/data_persistent # Datos persistentes
                  - ${RESULTS_DIR_HOST:-./results_host}:/app/results_persistent # Resultados persistentes
                depends_on:
                  redis: # Asegura que Redis esté disponible (al menos iniciado) antes que la app
                    condition: service_healthy # Espera a que el healthcheck de Redis pase
                working_dir: /app/scripts # Directorio de trabajo por defecto al hacer docker-compose exec/run
                # El comando por defecto del Dockerfile es `tail -f /dev/null` para mantenerlo corriendo.
                # Los scripts se ejecutarán con `docker-compose exec workhorse_app python nombre_script.py`
                # o `docker-compose run workhorse_app python nombre_script.py`
                stdin_open: true # Para permitir interacción si es necesario (ej. PDB)
                tty: true        # Para permitir interacción
                restart: unless-stopped # O 'no' si solo se usa para ejecuciones puntuales

            volumes:
              redis_data: # Define el volumen nombrado para la persistencia de Redis
                driver: local
            ```
    * **5.2. Probar la Configuración de Docker Compose**:
        * Asegurarse de que Docker y Docker Compose estén instalados.
        * Crear un archivo `.env` a partir de `.env.example` y rellenar `BINANCE_API_KEY` y `BINANCE_API_SECRET` (aunque no se usarán todavía).
        * Ejecutar:
            ```bash
            docker-compose up --build -d # Construye y levanta en segundo plano
            ```
        * Verificar logs:
            ```bash
            docker-compose logs redis
            docker-compose logs workhorse_app
            ```
        * Verificar que los contenedores están corriendo:
            ```bash
            docker-compose ps
            ```
        * Probar entrar al contenedor de la app:
            ```bash
            docker-compose exec workhorse_app bash
            # Dentro del contenedor, verificar la estructura de archivos y Python:
            # ls /app
            # ls /app/src
            # python --version
            # pip list
            # exit
            ```
        * Bajar los servicios:
            ```bash
            docker-compose down -v # El -v elimina los volúmenes anónimos y el nombrado redis_data
            ```

---

### Paso 6: Implementación del Esqueleto de `src/config_loader.py` (Módulo 9 Parcial)

* **Descripción Exhaustiva**: Crear la estructura inicial del módulo encargado de cargar las configuraciones desde los archivos `.yaml` y las variables de entorno del archivo `.env`.
* **Acciones Específicas**:
    * **6.1. Crear `src/config_loader.py`**:
        * Contenido inicial con la lógica para cargar `.env` y un archivo YAML de ejemplo.
            ```python
            # src/config_loader.py
            import os
            import yaml
            from dotenv import load_dotenv
            from pathlib import Path
            import logging

            # Configurar un logger básico para este módulo
            logger = logging.getLogger(__name__)

            # Definir la ruta base del proyecto y la ruta de configuración
            # Asumimos que config_loader.py está en src/
            # Por lo tanto, BASE_DIR es el directorio padre de src/
            BASE_DIR = Path(__file__).resolve().parent.parent
            CONFIG_DIR = BASE_DIR / "config" # Esto es dentro del contenedor si se monta en /app/config

            # Cargar variables de entorno desde .env en el directorio base del proyecto
            # Esta ruta debe ser accesible desde donde se ejecute el script que importa este módulo.
            # Si los scripts se ejecutan desde /app/scripts, y .env está en /app, BASE_DIR debería ser /app
            # Reajustar BASE_DIR si es necesario o pasar la ruta del .env explícitamente.
            # Para Docker, .env está en la raíz del contexto de build, y se carga por docker-compose.
            # load_dotenv() buscará un .env en el directorio actual o uno superior.
            # Si se ejecuta desde /app/scripts, buscará en /app/scripts, luego /app
            
            # Corregimos la ruta para que sea relativa a la raíz del proyecto (/app en el contenedor)
            # Asumiendo que el .env está en la raíz del proyecto /app/
            # y que los scripts en /app/scripts/ pueden accederlo.
            # load_dotenv(BASE_DIR / ".env") # Esta es una opción
            load_dotenv() # Otra opción es dejar que python-dotenv lo encuentre automáticamente
                          # si el workdir es /app o si está en una ruta superior

            # Alternativamente, para mayor robustez en Docker, las variables de .env ya están cargadas
            # en el entorno del contenedor por Docker Compose. Así que load_dotenv() podría no ser
            # estrictamente necesario si se confía en Docker Compose.
            # Sin embargo, es buena práctica para desarrollo local fuera de Docker.

            def get_env_variable(var_name: str, default_value: str = None) -> str:
                """
                Obtiene una variable de entorno.
                Lanza un error si no se encuentra y no se provee un valor por defecto.
                """
                value = os.getenv(var_name, default_value)
                if value is None and default_value is None:
                    logger.error(f"La variable de entorno '{var_name}' no está configurada y no tiene valor por defecto.")
                    raise EnvironmentError(f"Variable de entorno requerida '{var_name}' no encontrada.")
                return value

            def load_yaml_config(module_name: str, file_name: str = "params.yaml") -> dict:
                """
                Carga un archivo de configuración YAML específico de un módulo.
                Ejemplo: module_name='module1_data_acquisition'
                """
                # La ruta dentro del contenedor será relativa a /app/config
                # si config/ local se monta en /app/config
                # O si CONFIG_DIR_HOST se define como ./config y se monta en /app/config
                
                # Asumimos que CONFIG_DIR está correctamente definido como /app/config dentro del contenedor
                # que es el montaje de la carpeta ./config del host.
                
                # El README indica config/module_name/params.yaml
                # config_file_path = CONFIG_DIR / module_name / file_name
                # Ajustamos la ruta de CONFIG_DIR para que sea relativa al directorio de ejecución o absoluta en el contenedor
                
                # Usaremos /app/config como la ruta absoluta dentro del contenedor
                effective_config_dir = Path(get_env_variable("CONFIG_DIR_CONTAINER", "/app/config"))
                config_file_path = effective_config_dir / module_name / file_name

                if not config_file_path.exists():
                    logger.error(f"Archivo de configuración YAML no encontrado en: {config_file_path}")
                    raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_file_path}")

                try:
                    with open(config_file_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    logger.info(f"Configuración cargada exitosamente desde: {config_file_path}")
                    return config_data if config_data else {}
                except yaml.YAMLError as e:
                    logger.error(f"Error al parsear el archivo YAML {config_file_path}: {e}")
                    raise ValueError(f"Error al parsear YAML: {config_file_path}") from e
                except Exception as e:
                    logger.error(f"Error inesperado al cargar el archivo YAML {config_file_path}: {e}")
                    raise RuntimeError(f"Error al cargar YAML: {config_file_path}") from e

            # Ejemplo de cómo obtener variables de .env (ya cargadas por Docker Compose o load_dotenv())
            # BINANCE_API_KEY = get_env_variable("BINANCE_API_KEY") # Descomentar para probar
            # REDIS_HOST_ENV = get_env_variable("REDIS_HOST", "localhost")

            if __name__ == "__main__":
                # Configuración básica de logging para prueba directa del script
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                
                logger.info(f"BASE_DIR: {BASE_DIR}")
                logger.info(f"CONFIG_DIR (calculado inicialmente): {CONFIG_DIR}")
                logger.info(f"Ruta efectiva de CONFIG_DIR en contenedor (desde env var CONFIG_DIR_CONTAINER o default): {Path(get_env_variable('CONFIG_DIR_CONTAINER', '/app/config'))}")

                # Crear archivos de config de ejemplo para probar
                sample_module_name = "module_test"
                sample_config_dir_path = Path(get_env_variable("CONFIG_DIR_CONTAINER", "/app/config")) / sample_module_name
                sample_config_dir_path.mkdir(parents=True, exist_ok=True)
                sample_config_file = sample_config_dir_path / "params.yaml"
                with open(sample_config_file, 'w') as f:
                    yaml.dump({"param1": "value1", "param2": 123}, f)
                
                logger.info(f"Archivo de config de prueba creado en: {sample_config_file}")

                try:
                    # Probar cargar una variable de entorno
                    api_key = get_env_variable("BINANCE_API_KEY", "clave_api_default_para_prueba") # Usar default para prueba si no está en .env
                    logger.info(f"Binance API Key (prueba): {api_key}")

                    # Probar cargar un archivo YAML
                    test_config = load_yaml_config(sample_module_name)
                    logger.info(f"Configuración de prueba cargada: {test_config}")
                    
                    # Intentar cargar un archivo que no existe para probar el error
                    # load_yaml_config("module_non_existent")

                except Exception as e:
                    logger.error(f"Error durante la prueba de config_loader: {e}")
                finally:
                    # Limpiar archivo de prueba
                    if sample_config_file.exists():
                        sample_config_file.unlink()
                    if sample_config_dir_path.exists():
                        # Comprobar si el directorio está vacío antes de borrarlo
                        if not any(sample_config_dir_path.iterdir()):
                             sample_config_dir_path.rmdir()
                        else:
                            logger.warning(f"El directorio de prueba {sample_config_dir_path} no está vacío, no se eliminará.")


            print("Prueba de src/config_loader.py completada. Revisa los logs.")
            ```
    * **6.2. Añadir `CONFIG_DIR_CONTAINER` a `.env.example` y `.env`**:
        * En `.env.example` (y el `.env` local):
            ```env
            # ... otras variables ...
            CONFIG_DIR_CONTAINER="/app/config" # Ruta absoluta de la carpeta de configuración dentro del contenedor
            ```
    * **6.3. Crear directorios de config y archivos `params.yaml` de ejemplo**:
        * Basado en `README.md`, Sección 3 y 5.
        * `config/module1_data_acquisition/params.yaml`:
            ```yaml
            kline_interval: "15m"
            order_book_depth: 5
            data_download_start_date: "2021-01-01"
            data_download_end_date: "2024-12-31"
            trading_pair: "BTCUSDT"
            ```
        * Crear las demás carpetas y archivos `params.yaml` vacíos o con valores por defecto según el README:
            * `config/module2_preprocessing/params.yaml`
            * `config/module3_environment/params.yaml`
            * `config/module4_agent_sac/params.yaml`
            * `config/module7_backtesting_wfo/params.yaml`
    * **6.4. (Opcional) Crear `config/logging_config.yaml`**:
        * Este archivo centralizará la configuración de logging de Python.
            ```yaml
            # config/logging_config.yaml
            version: 1
            disable_existing_loggers: False

            formatters:
              simple:
                format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
              detailed:
                format: '%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'

            handlers:
              console:
                class: logging.StreamHandler
                level: DEBUG # Nivel del handler, puede ser sobreescrito por el logger
                formatter: simple
                stream: ext://sys.stdout

              file_info:
                class: logging.handlers.RotatingFileHandler
                level: INFO
                formatter: detailed
                filename: /app/results_persistent/logs/app_info.log # Ruta dentro del contenedor
                maxBytes: 10485760 # 10MB
                backupCount: 5
                encoding: utf8

              file_error:
                class: logging.handlers.RotatingFileHandler
                level: ERROR
                formatter: detailed
                filename: /app/results_persistent/logs/app_error.log # Ruta dentro del contenedor
                maxBytes: 10485760 # 10MB
                backupCount: 3
                encoding: utf8

            loggers:
              src.config_loader: # Logger específico para config_loader
                level: INFO
                handlers: [console, file_info, file_error]
                propagate: no # No propagar a root si se maneja específicamente
              
              src.data_acquisition:
                level: INFO
                handlers: [console, file_info, file_error]
                propagate: no

              # Añadir más loggers específicos para otros módulos de src/ aquí...
              # src.preprocessing:
              #   level: INFO
              #   handlers: [console, file_info, file_error]
              #   propagate: no

            root: # Logger raíz, captura todo lo no capturado por loggers específicos
              level: INFO # Nivel por defecto para todo si no se especifica
              handlers: [console, file_info, file_error]
            ```
    * **6.5. Crear `src/common/utils.py` para cargar `logging_config.yaml`**:
            ```python
            # src/common/utils.py
            import logging
            import logging.config
            import yaml
            from pathlib import Path

            # Asumimos que /app/config es la ruta de config en el contenedor
            DEFAULT_LOGGING_CONFIG_PATH = Path("/app/config/logging_config.yaml") 

            def setup_logging(config_path: Path = DEFAULT_LOGGING_CONFIG_PATH) -> None:
                """
                Configura el logging usando un archivo YAML.
                """
                if config_path.exists():
                    try:
                        with open(config_path, 'rt') as f:
                            config = yaml.safe_load(f.read())
                        logging.config.dictConfig(config)
                        logging.info(f"Configuración de logging cargada desde {config_path}")
                    except Exception as e:
                        logging.basicConfig(level=logging.INFO) # Fallback a config básica
                        logging.error(f"Error al cargar la configuración de logging desde {config_path}: {e}. Usando logging básico.")
                else:
                    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    logging.warning(f"Archivo de configuración de logging no encontrado en {config_path}. Usando logging básico.")

            # Puedes añadir otras utilidades comunes aquí.
            ```
    * **6.6. Crear un script de prueba en `scripts/test_config.py`**:
            ```python
            # scripts/test_config.py
            import sys
            from pathlib import Path
            import logging

            # Añadir src al PYTHONPATH para que se puedan importar módulos de src
            # Esto es crucial si ejecutas el script directamente desde la carpeta 'scripts'
            # y 'src' no está instalado como un paquete.
            # En Docker, PYTHONPATH se establece en docker-compose.yml
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent 
            src_path = project_root / "src"
            sys.path.append(str(src_path)) # Añade src al path

            # Ahora podemos importar desde src
            from common.utils import setup_logging
            from config_loader import load_yaml_config, get_env_variable

            # Configurar logging usando la utilidad
            setup_logging() # Usará la ruta por defecto a logging_config.yaml

            # Obtener el logger para este script
            logger = logging.getLogger(__name__) # Usará la config root o una específica si la defines

            def main():
                logger.info("Iniciando prueba de carga de configuración...")

                try:
                    # Probar la carga de variables de entorno
                    binance_key = get_env_variable("BINANCE_API_KEY", "NO_KEY_DEFAULT")
                    logger.info(f"BINANCE_API_KEY (desde .env o default): {binance_key}")

                    redis_host = get_env_variable("REDIS_HOST")
                    logger.info(f"REDIS_HOST (desde .env): {redis_host}")

                    # Probar la carga de configuraciones YAML de módulos
                    m1_config = load_yaml_config("module1_data_acquisition")
                    logger.info(f"Configuración de Módulo 1 (Data Acquisition): {m1_config}")
                    if m1_config:
                        logger.info(f"Intervalo de Kline de M1: {m1_config.get('kline_interval')}")

                    # (Añadir pruebas para otros módulos si ya tienen params.yaml)
                    # m2_config = load_yaml_config("module2_preprocessing")
                    # logger.info(f"Configuración de Módulo 2 (Preprocessing): {m2_config}")

                    logger.info("Prueba de carga de configuración completada exitosamente.")

                except FileNotFoundError as e:
                    logger.error(f"Error de archivo no encontrado: {e}")
                except EnvironmentError as e:
                    logger.error(f"Error de variable de entorno: {e}")
                except Exception as e:
                    logger.error(f"Ocurrió un error inesperado durante la prueba de configuración: {e}", exc_info=True)

            if __name__ == "__main__":
                main()
            ```
    * **6.7. Ejecutar la prueba de configuración dentro de Docker**:
        * Asegurarse que `.env` existe y `docker-compose.yml` está configurado con `CONFIG_DIR_HOST`.
        * Asegurarse que las carpetas `config/module_name` y sus `params.yaml` existen.
        * Asegurarse que `config/logging_config.yaml` existe.
        * Asegurarse que se ha creado la carpeta `./results_host/logs` en el host o que el script de logging la crea (o que el servicio tiene permisos para crearla).
            ```bash
            # Desde la raíz del proyecto en el host
            mkdir -p ./results_host/logs # Crear el directorio de logs si no existe
            docker-compose up -d --build workhorse_app redis
            docker-compose exec workhorse_app python scripts/test_config.py
            # Revisar la salida en la consola y el contenido de results_host/logs/
            docker-compose logs workhorse_app # Para ver si hay errores de Python al iniciar también
            ```

---

### Paso 7: Compromiso Inicial en Git

* **Descripción Exhaustiva**: Añadir todos los archivos creados al control de versiones Git y realizar el primer commit.
* **Acciones Específicas**:
    * **7.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add .
        git commit -m "Fase 0: Initial project structure, dependencies, Docker setup, and basic config loading module"
        # Opcional: git branch -M main # Si la rama por defecto es 'master'
        # Opcional: git remote add origin <URL_DEL_REPOSITORIO_REMOTO>
        # Opcional: git push -u origin main
        ```

**Fin de la Fase de Implementación 0.**

---


## Fase de Implementación 1: Módulo 9 - Gestión Centralizada de Configuración (Finalización y Robustecimiento)

**Nombre Descriptivo de la Fase:** Consolidación y Pruebas del Sistema de Carga de Configuración.

Esta fase se centra en finalizar y robustecer el Módulo 9, asegurando que la carga de configuraciones desde archivos `.yaml` y variables de entorno (`.env`) sea fiable, manejando errores adecuadamente y estableciendo las bases para que todos los demás módulos puedan acceder a sus parámetros de forma consistente. También incluye la creación de pruebas unitarias para el cargador de configuración.

-----

### Paso 1: Refinamiento de `src/config_loader.py`

  * **Descripción Exhaustiva**: Mejorar el script `src/config_loader.py` creado en la Fase 0. Se añadirán más detalles en docstrings, se mejorará el manejo de errores y se asegurará que las rutas a los archivos de configuración sean gestionadas de forma robusta, especialmente en el contexto de Docker.
  * **Acciones Específicas**:
      * **1.1. Revisar y Actualizar `src/config_loader.py`**:
          * **Objetivos**:
              * Mejorar la claridad de los mensajes de error.
              * Añadir type hinting más específico si es posible.
              * Asegurar que `load_dotenv()` funcione de manera predecible (Docker Compose ya carga las variables de `.env` en el entorno del servicio, pero `load_dotenv()` es útil para desarrollo local fuera de Docker o para tests). `find_dotenv()` puede usarse para localizar el archivo `.env` de forma más fiable.
              * Confirmar que la ruta para `load_yaml_config` (usando `CONFIG_DIR_CONTAINER`) es la forma definitiva de acceder a los YAMLs dentro del contenedor.
          * **Código Actualizado para `src/config_loader.py`**:
            ```python
            # src/config_loader.py
            import os
            import yaml
            from dotenv import load_dotenv, find_dotenv
            from pathlib import Path
            import logging
            from typing import Any, Dict, Optional, Union

            # Configurar un logger básico para este módulo
            logger = logging.getLogger(__name__)

            # Cargar variables de entorno desde .env. find_dotenv() busca el archivo .env
            # subiendo desde el directorio actual o el directorio del script.
            # Es buena práctica llamarlo una vez al inicio.
            # Docker Compose ya inyecta las variables de .env, pero esto es útil para
            # ejecución local/tests fuera de Docker.
            env_path = find_dotenv(usecwd=True) # Prioriza .env en CWD, luego sube
            if env_path:
                logger.info(f"Cargando variables de entorno desde: {env_path}")
                load_dotenv(dotenv_path=env_path, override=True) # Override para que .env tenga precedencia sobre vars del sistema
            else:
                logger.info("Archivo .env no encontrado por find_dotenv(). Las variables de entorno deben estar preconfiguradas si es necesario (ej. en Docker).")


            def get_env_variable(var_name: str, default_value: Optional[str] = None, required: bool = True) -> Optional[str]:
                """
                Obtiene una variable de entorno.

                Args:
                    var_name (str): El nombre de la variable de entorno.
                    default_value (Optional[str]): El valor por defecto a retornar si la variable no se encuentra.
                                                  Si es None y required es True, se lanzará un error.
                    required (bool): Si es True y la variable no se encuentra y no hay default_value,
                                     lanza EnvironmentError.

                Returns:
                    Optional[str]: El valor de la variable de entorno o el valor por defecto.
                                   None si no es requerida, no se encuentra y no hay default.

                Raises:
                    EnvironmentError: Si la variable es requerida, no se encuentra y no tiene valor por defecto.
                """
                value = os.getenv(var_name)
                if value is not None:
                    return value
                
                if default_value is not None:
                    logger.debug(f"Variable de entorno '{var_name}' no encontrada, usando valor por defecto.")
                    return default_value
                
                if required:
                    logger.error(f"Variable de entorno requerida '{var_name}' no está configurada y no tiene valor por defecto.")
                    raise EnvironmentError(f"Variable de entorno requerida '{var_name}' no encontrada.")
                
                logger.debug(f"Variable de entorno opcional '{var_name}' no encontrada, retornando None.")
                return None

            def load_yaml_config(module_identifier: str, file_name: str = "params.yaml") -> Dict[str, Any]:
                """
                Carga un archivo de configuración YAML específico de un módulo o una configuración global.

                El `module_identifier` puede ser el nombre de una subcarpeta de módulo
                (ej. 'module1_data_acquisition') o el nombre de un archivo YAML global
                directamente en la carpeta de configuración (ej. 'logging_config.yaml',
                en cuyo caso `file_name` debería ser '').

                Args:
                    module_identifier (str): Identificador del módulo (subcarpeta) o nombre del archivo YAML global (sin extensión).
                    file_name (str): Nombre del archivo YAML dentro de la carpeta del módulo (default: "params.yaml").
                                     Si se carga un archivo global, esto puede ser ignorado o ser el nombre completo del archivo.

                Returns:
                    Dict[str, Any]: Un diccionario con la configuración cargada. Retorna un diccionario vacío si hay errores
                                    y no se puede cargar, aunque se prioriza lanzar excepciones.

                Raises:
                    FileNotFoundError: Si el archivo de configuración no se encuentra.
                    ValueError: Si hay un error al parsear el archivo YAML.
                    RuntimeError: Para otros errores inesperados durante la carga.
                """
                # Ruta base de la configuración dentro del contenedor, obtenida de variable de entorno.
                # Esta variable es establecida en .env y usada por docker-compose.yml
                # Default a /app/config si no está definida.
                config_dir_str = get_env_variable("CONFIG_DIR_CONTAINER", "/app/config", required=True)
                if not config_dir_str: # Asegurar que no sea None o vacío si get_env_variable se modifica
                    raise EnvironmentError("CONFIG_DIR_CONTAINER no está configurado apropiadamente.")
                
                base_config_path = Path(config_dir_str)

                # Determinar la ruta completa del archivo de configuración
                # Si file_name se proporciona explícitamente y module_identifier es solo un directorio
                if file_name and module_identifier:
                    config_file_path = base_config_path / module_identifier / file_name
                # Si module_identifier es el nombre completo de un archivo (ej. "logging_config.yaml")
                elif module_identifier.endswith(".yaml") or module_identifier.endswith(".yml"):
                     config_file_path = base_config_path / module_identifier
                # Caso por defecto: module_identifier es un directorio, file_name es params.yaml
                else:
                    config_file_path = base_config_path / module_identifier / file_name


                if not config_file_path.exists():
                    logger.error(f"Archivo de configuración YAML no encontrado en: {config_file_path}")
                    raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_file_path}")
                if not config_file_path.is_file():
                    logger.error(f"La ruta de configuración especificada no es un archivo: {config_file_path}")
                    raise FileNotFoundError(f"La ruta de configuración no es un archivo: {config_file_path}")

                try:
                    with open(config_file_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    if config_data is None: # Archivo YAML vacío
                        logger.warning(f"El archivo de configuración {config_file_path} está vacío.")
                        return {}
                    logger.info(f"Configuración cargada exitosamente desde: {config_file_path}")
                    return config_data
                except yaml.YAMLError as e:
                    logger.error(f"Error al parsear el archivo YAML {config_file_path}: {e}")
                    raise ValueError(f"Error al parsear YAML desde {config_file_path}: {e}") from e
                except Exception as e:
                    logger.error(f"Error inesperado al cargar el archivo YAML {config_file_path}: {e}")
                    raise RuntimeError(f"Error inesperado al cargar YAML desde {config_file_path}: {e}") from e

            if __name__ == "__main__":
                # Este bloque es solo para pruebas directas del script,
                # la configuración de logging debería ser llamada por la app principal.
                # Para una prueba más robusta, usa el script en `scripts/test_config.py`.
                
                # Setup básico de logging si se ejecuta directamente
                logging.basicConfig(
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                )

                logger.info("Ejecutando prueba directa de src/config_loader.py...")
                try:
                    # Prueba de get_env_variable
                    test_api_key = get_env_variable("BINANCE_API_KEY", "test_default_api_key")
                    logger.info(f"BINANCE_API_KEY (prueba directa): {test_api_key}")
                    
                    non_existent_var = get_env_variable("NON_EXISTENT_VAR_12345", "default_for_non_existent", required=False)
                    logger.info(f"NON_EXISTENT_VAR_12345 (prueba directa, no requerida): {non_existent_var}")

                    try:
                        get_env_variable("ABSOLUTELY_NON_EXISTENT_VAR", required=True)
                    except EnvironmentError as e:
                        logger.info(f"Capturado error esperado al obtener variable requerida no existente: {e}")

                    # Crear archivos de config de ejemplo para probar load_yaml_config
                    # Asumiendo que .env tiene CONFIG_DIR_CONTAINER="/app/config"
                    # y que este script se ejecuta en un contexto donde /app/config es escribible (ej. Docker)
                    # Para pruebas locales, ajusta esta ruta o asegúrate de que el directorio exista.
                    temp_config_dir_base_str = get_env_variable("CONFIG_DIR_CONTAINER", default_value="./temp_test_configs")
                    temp_config_dir_base = Path(temp_config_dir_base_str) # ./temp_test_configs si no está en .env
                    temp_config_dir_base.mkdir(parents=True, exist_ok=True)
                    
                    sample_module_name = "test_module_direct"
                    sample_module_path = temp_config_dir_base / sample_module_name
                    sample_module_path.mkdir(parents=True, exist_ok=True)
                    
                    sample_params_file = sample_module_path / "params.yaml"
                    with open(sample_params_file, 'w', encoding='utf-8') as f:
                        yaml.dump({"paramA": "valueA", "paramB": 456}, f)
                    
                    logger.info(f"Archivo de config de prueba creado en: {sample_params_file}")

                    loaded_test_config = load_yaml_config(f"{sample_module_name}", "params.yaml")
                    logger.info(f"Configuración de prueba cargada (directa): {loaded_test_config}")
                    assert loaded_test_config.get("paramA") == "valueA"

                except Exception as e:
                    logger.error(f"Error durante la prueba directa de config_loader: {e}", exc_info=True)
                finally:
                    # Limpieza (opcional, puede ser útil para no dejar basura)
                    if 'sample_params_file' in locals() and sample_params_file.exists():
                       sample_params_file.unlink()
                    if 'sample_module_path' in locals() and sample_module_path.exists():
                       try:
                           sample_module_path.rmdir() # Solo si está vacío
                       except OSError:
                           logger.warning(f"No se pudo eliminar {sample_module_path}, puede no estar vacío.")
                    # if temp_config_dir_base.exists() and not any(temp_config_dir_base.iterdir()):
                    #    temp_config_dir_base.rmdir()


                logger.info("Prueba directa de src/config_loader.py completada. Revisa los logs.")
            ```
      * **1.2. Verificar `PYTHONPATH` en `docker-compose.yml`**:
          * La línea `PYTHONPATH: "/app/src:/app"` en `docker-compose.yml` ayuda a que los módulos en `src/` sean importables. Confirmar que está presente y es correcta. (Sí, se añadió en Fase 0).

-----

### Paso 2: Refinamiento de `src/common/utils.py` (Función `setup_logging`)

  * **Descripción Exhaustiva**: Asegurar que la función `setup_logging` en `src/common/utils.py` sea robusta, maneje la ausencia del archivo de configuración de logging y use rutas de forma consistente.
  * **Acciones Específicas**:
      * **2.1. Revisar y Actualizar `src/common/utils.py`**:
          * **Objetivos**: Confirmar que el fallback a `logging.basicConfig` es adecuado y que los mensajes de log son claros.
          * **Código (sin cambios significativos respecto a Fase 0, pero se revisa su robustez)**:
            ```python
            # src/common/utils.py
            import logging
            import logging.config
            import yaml
            from pathlib import Path
            from typing import Union # Necesario para Path | str en versiones antiguas de Python

            # Importar desde config_loader para obtener la ruta base de config
            # Esto crea una dependencia, asegurarse que sea manejable.
            # Alternativamente, pasar la ruta completa de logging_config.yaml como argumento.
            from config_loader import get_env_variable # Asumiendo que config_loader.py está en src/

            # Ruta por defecto para el archivo de configuración de logging.
            # Se construye usando la misma lógica que para otros archivos de config.
            CONFIG_DIR_CONTAINER_STR = get_env_variable("CONFIG_DIR_CONTAINER", "/app/config", required=True)
            if not CONFIG_DIR_CONTAINER_STR: # Defensa por si get_env_variable cambia
                 raise EnvironmentError("CONFIG_DIR_CONTAINER no está configurado.")
            DEFAULT_LOGGING_CONFIG_PATH = Path(CONFIG_DIR_CONTAINER_STR) / "logging_config.yaml"

            def setup_logging(config_path: Union[Path, str] = DEFAULT_LOGGING_CONFIG_PATH) -> None:
                """
                Configura el logging de la aplicación usando un archivo de configuración YAML.

                Si el archivo no se encuentra o hay un error al cargarlo,
                se recurre a una configuración básica de logging.

                Args:
                    config_path (Union[Path, str]): Ruta al archivo de configuración de logging YAML.
                                                     Por defecto usa DEFAULT_LOGGING_CONFIG_PATH.
                """
                effective_config_path = Path(config_path)
                if effective_config_path.exists() and effective_config_path.is_file():
                    try:
                        with open(effective_config_path, 'rt', encoding='utf-8') as f:
                            logging_config_dict = yaml.safe_load(f.read())
                        
                        # Asegurarse de que los directorios de logs para file handlers existan
                        # (Esto es importante, ya que logging.config.dictConfig no los crea)
                        if 'handlers' in logging_config_dict:
                            for handler_name, handler_config in logging_config_dict['handlers'].items():
                                if handler_config.get('class') in ['logging.FileHandler', 'logging.handlers.RotatingFileHandler', 'logging.handlers.TimedRotatingFileHandler']:
                                    log_filename = handler_config.get('filename')
                                    if log_filename:
                                        log_dir = Path(log_filename).parent
                                        log_dir.mkdir(parents=True, exist_ok=True)
                                        # logger.debug(f"Asegurando que el directorio de log exista: {log_dir}") # Ojo, el logger no está configurado aún.

                        logging.config.dictConfig(logging_config_dict)
                        # Este log usará la nueva configuración si tiene éxito
                        logging.info(f"Configuración de logging cargada exitosamente desde {effective_config_path}")
                    except Exception as e:
                        # Fallback a config básica si hay cualquier error
                        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
                        logging.error(f"Error al cargar la configuración de logging desde {effective_config_path}: {e}. Usando logging básico.", exc_info=True)
                else:
                    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
                    if not effective_config_path.exists():
                        logging.warning(f"Archivo de configuración de logging no encontrado en {effective_config_path}. Usando logging básico.")
                    else: # Existe pero no es un archivo
                        logging.warning(f"La ruta de configuración de logging {effective_config_path} no es un archivo. Usando logging básico.")

            # Otras utilidades comunes pueden ir aquí...
            # Ejemplo:
            # def some_other_utility_function():
            #    pass
            ```
          * **Nota**: La creación automática de directorios para los manejadores de archivos de log es una mejora importante.

-----

### Paso 3: Completar los Archivos `params.yaml` para Todos los Módulos

  * **Descripción Exhaustiva**: Asegurar que todos los archivos `params.yaml` dentro de las subcarpetas de `config/` estén creados y contengan todos los parámetros especificados en el `README.md` para cada módulo (M1, M2, M3, M4, M7).
  * **Acciones Específicas**:
      * **3.1. Crear/Actualizar `config/module1_data_acquisition/params.yaml`**: (Ya creado en Fase 0, verificar completitud)
        ```yaml
        # config/module1_data_acquisition/params.yaml
        kline_interval: "15m" # default
        order_book_depth: 5
        data_download_start_date: "2021-01-01" # ej. "YYYY-MM-DD"
        data_download_end_date: "2024-12-31"   # ej. "YYYY-MM-DD"
        trading_pair: "BTCUSDT"
        # (Añadir otros parámetros si se identifican más adelante para este módulo)
        ```
      * **3.2. Crear/Actualizar `config/module2_preprocessing/params.yaml`**:
        ```yaml
        # config/module2_preprocessing/params.yaml
        sequence_length_L: 96
        normalization_window_multiplier_for_L: 2 # para Z-score sobre L*multiplicador periodos
        # Parámetros para indicadores técnicos
        sma_short_period: 20
        sma_long_period: 50
        ema_short_period: 12
        ema_long_period: 26
        rsi_period: 14
        atr_period: 14
        macd_fast_period: 12
        macd_slow_period: 26
        macd_signal_period: 9
        bollinger_period: 20
        bollinger_std_dev: 2 # Desviaciones estándar
        cci_period: 20
        stochastic_k_period: 14
        stochastic_d_period: 3
        stochastic_slowing_period: 3 # A veces llamado 'smooth_k'
        # (Añadir otros parámetros si se identifican más adelante para este módulo)
        ```
      * **3.3. Crear/Actualizar `config/module3_environment/params.yaml`**:
        ```yaml
        # config/module3_environment/params.yaml
        initial_equity: 10000.0 # en USD
        leverage: 10 # Apalancamiento fijo (ej. 10x)
        position_size_pct_equity: 0.05 # 5% del equity por operación
        taker_fee_rate: 0.0004 # Comisión de Taker (ej. ~0.04% de Binance)
        slippage_atr_multiplier: 0.1 # Slippage como 0.1 * ATR(14) por lado
        action_threshold: 0.15 # Umbral para abrir/cerrar posiciones basado en la señal del agente
        equity_drawdown_threshold_episode_end: -0.20 # -20% de drawdown máximo de equity para terminar episodio
        liquidation_safety_factor: 0.8 # Factor de seguridad para simular liquidación (Precio_Liq = Precio_Entrada * (1 +/- (1/Apalancamiento) * Factor_Seguridad))
                                         # Un movimiento del 8% en contra con apalancamiento 10x (1/10 = 10% -> 10% * 0.8 = 8%)
        max_episode_steps_equals_dataset_length: true # Si el episodio termina cuando se agotan los datos del dataset de entrenamiento.
        # (Añadir otros parámetros si se identifican más adelante para este módulo)
        ```
      * **3.4. Crear/Actualizar `config/module4_agent_sac/params.yaml`**:
        ```yaml
        # config/module4_agent_sac/params.yaml
        # Arquitectura del Modelo Transformer
        d_model_transformer: 128
        transformer_layers: 3
        transformer_heads: 4
        # mlp_hidden_dims_transformer_ffn: 512 # Internamente es 4 * d_model, no necesita ser param aquí si es fijo

        # Arquitectura Redes Actor/Críticos (MLPs post-Transformer)
        actor_critic_hidden_dims: [256, 256] # Capas ocultas para Actor y Críticos después del Transformer

        # Hiperparámetros de SAC (Soft Actor-Critic) de Stable Baselines3
        learning_rate: 0.0003 # Adam optimizer learning rate
        buffer_size: 100000   # Tamaño del Replay Buffer
        batch_size: 256
        gamma: 0.99           # Factor de descuento
        tau: 0.005            # Coeficiente de actualización suave (polyak) para redes objetivo
        train_freq_steps: 1   # Frecuencia de entrenamiento (en pasos del entorno)
        gradient_steps: 1     # Número de pasos de gradiente a realizar en cada train_freq
        ent_coef: 'auto'      # Coeficiente de entropía ('auto' o un float)
        learning_starts: 1000 # Cuántos pasos recolectar antes de empezar a entrenar
        # policy_kwargs: null # Para pasar kwargs adicionales a la política, se puede definir aquí si es complejo
        # verbose: 0 # Nivel de verbosidad de SB3
        # seed: null # Semilla para reproducibilidad
        # device: 'auto' # 'cpu', 'cuda', 'auto'
        # (Añadir otros parámetros específicos de SB3 SAC si son necesarios)
        ```
      * **3.5. Crear/Actualizar `config/module7_backtesting_wfo/params.yaml`**:
        ```yaml
        # config/module7_backtesting_wfo/params.yaml
        # Parámetros de Walk-Forward Optimization (WFO)
        wfo_is_window_months: 18      # Longitud de la ventana In-Sample (Entrenamiento) en meses
        wfo_oos_window_months: 3       # Longitud de la ventana Out-of-Sample (Prueba) en meses
        wfo_step_months: 3             # Paso de avance (igual a OOS para WFO contiguo)
        wfo_window_type: "rolling"     # Tipo de ventana IS: "rolling" (deslizante) o "expanding" (anclada)

        # Métricas de Backtesting
        risk_free_rate_for_sharpe_sortino: 0.0 # Tasa libre de riesgo para Sharpe/Sortino (anualizada)
        # (Añadir otros parámetros para la generación de reportes o métricas si es necesario)
        ```

-----

### Paso 4: Escritura de Pruebas Unitarias para `src/config_loader.py`

  * **Descripción Exhaustiva**: Crear pruebas unitarias para las funciones `get_env_variable` y `load_yaml_config` en `src/config_loader.py`. Estas pruebas deben cubrir casos de éxito, manejo de errores (archivos no encontrados, YAML malformado, variables de entorno faltantes) y valores por defecto.
  * **Acciones Específicas**:
      * **4.1. Crear archivo de prueba `tests/test_config_loader.py`**:
          * (Si `config_loader.py` está en `src/`, el test debería estar en `tests/test_config_loader.py`. Si estuviera en `src/common/`, entonces `tests/common/test_config_loader.py`). Asumimos `src/config_loader.py`.
      * **4.2. Escribir Pruebas Unitarias**:
          * Usar `pytest` y `unittest.mock` para mockear variables de entorno y el sistema de archivos si es necesario.
            ```python
            # tests/test_config_loader.py
            import pytest
            import os
            import yaml
            from pathlib import Path
            from unittest import mock

            # Asegurarse que src esté en el path para importar config_loader
            # (pytest usualmente maneja esto si se ejecuta desde la raíz con `python -m pytest`)
            # import sys
            # project_root_for_test = Path(__file__).resolve().parent.parent
            # sys.path.insert(0, str(project_root_for_test / "src"))

            from src.config_loader import get_env_variable, load_yaml_config

            @pytest.fixture
            def mock_env_vars(monkeypatch):
                """Fixture para mockear variables de entorno."""
                monkeypatch.setenv("TEST_VAR_EXISTS", "test_value")
                monkeypatch.setenv("OTHER_TEST_VAR", "another_value")
                # No establecer TEST_VAR_MISSING
                return monkeypatch

            @pytest.fixture
            def temp_config_dir(tmp_path: Path):
                """Fixture para crear un directorio de configuración temporal para pruebas."""
                config_dir = tmp_path / "config"
                config_dir.mkdir()
                
                # Crear un params.yaml de prueba
                module1_dir = config_dir / "test_module1"
                module1_dir.mkdir()
                with open(module1_dir / "params.yaml", "w") as f:
                    yaml.dump({"key1": "value1", "key2": 123}, f)

                # Crear un params.yaml vacío
                module2_dir = config_dir / "test_module2"
                module2_dir.mkdir()
                with open(module2_dir / "params.yaml", "w") as f:
                    f.write("") # Archivo vacío

                # Crear un archivo YAML malformado
                module3_dir = config_dir / "test_module3"
                module3_dir.mkdir()
                with open(module3_dir / "params.yaml", "w") as f:
                    f.write("key1: value1\nkey2: [1, 2\nmalformed_yaml") # Error de sintaxis

                # Crear un archivo global de config
                with open(config_dir / "global_config.yaml", "w") as f:
                    yaml.dump({"global_param": "global_value"}, f)

                return config_dir

            # Pruebas para get_env_variable
            def test_get_env_variable_exists(mock_env_vars):
                assert get_env_variable("TEST_VAR_EXISTS") == "test_value"

            def test_get_env_variable_missing_required(mock_env_vars):
                with pytest.raises(EnvironmentError, match="Variable de entorno requerida 'TEST_VAR_MISSING' no encontrada"):
                    get_env_variable("TEST_VAR_MISSING", required=True)

            def test_get_env_variable_missing_with_default(mock_env_vars):
                assert get_env_variable("TEST_VAR_MISSING", default_value="default_val") == "default_val"

            def test_get_env_variable_missing_not_required(mock_env_vars):
                assert get_env_variable("TEST_VAR_MISSING", required=False) is None

            def test_get_env_variable_missing_not_required_with_default(mock_env_vars):
                 assert get_env_variable("TEST_VAR_MISSING", default_value="default_val", required=False) == "default_val"


            # Pruebas para load_yaml_config
            def test_load_yaml_config_success(temp_config_dir, monkeypatch):
                # Mockear CONFIG_DIR_CONTAINER para que apunte a temp_config_dir
                monkeypatch.setenv("CONFIG_DIR_CONTAINER", str(temp_config_dir))
                
                config = load_yaml_config("test_module1", "params.yaml")
                assert config == {"key1": "value1", "key2": 123}

            def test_load_yaml_config_global_file(temp_config_dir, monkeypatch):
                monkeypatch.setenv("CONFIG_DIR_CONTAINER", str(temp_config_dir))
                config = load_yaml_config("global_config.yaml") # Asume que la función maneja esto
                assert config == {"global_param": "global_value"}

            def test_load_yaml_config_empty_file(temp_config_dir, monkeypatch):
                monkeypatch.setenv("CONFIG_DIR_CONTAINER", str(temp_config_dir))
                config = load_yaml_config("test_module2", "params.yaml")
                assert config == {} # Un YAML vacío debería resultar en un dict vacío

            def test_load_yaml_config_file_not_found(temp_config_dir, monkeypatch):
                monkeypatch.setenv("CONFIG_DIR_CONTAINER", str(temp_config_dir))
                with pytest.raises(FileNotFoundError):
                    load_yaml_config("non_existent_module", "params.yaml")

            def test_load_yaml_config_malformed_yaml(temp_config_dir, monkeypatch):
                monkeypatch.setenv("CONFIG_DIR_CONTAINER", str(temp_config_dir))
                with pytest.raises(ValueError, match="Error al parsear YAML"): # O yaml.YAMLError si la función no lo envuelve
                    load_yaml_config("test_module3", "params.yaml")

            def test_load_yaml_config_env_var_not_set(monkeypatch):
                # Asegurar que CONFIG_DIR_CONTAINER no esté seteado para esta prueba
                monkeypatch.delenv("CONFIG_DIR_CONTAINER", raising=False)
                with pytest.raises(EnvironmentError, match="CONFIG_DIR_CONTAINER no está configurado apropiadamente."):
                    load_yaml_config("any_module")
            ```
      * **4.3. Ejecutar las Pruebas**:
          * Añadir `pytest` y `pytest-cov` a `requirements.txt` si no están (ya se añadieron en Fase 0).
          * Desde la raíz del proyecto:
            ```bash
            # Opción 1: Ejecutar directamente con python -m pytest
            python -m pytest tests/test_config_loader.py --cov=src/config_loader --cov-report=html

            # Opción 2: Dentro de Docker si se prefiere testear en el entorno del contenedor
            # (Asegurar que tests/ esté montado o copiado en el Dockerfile/docker-compose si es necesario para este enfoque)
            # Por ahora, es más simple ejecutar tests en el host si el código es Python puro y mocks manejan E/S.
            # Si los tests necesitan servicios como Redis, Docker es mejor.
            # Para este módulo, tests en host son suficientes.
            ```
              * Revisar el reporte de cobertura en `htmlcov/index.html`.

-----

### Paso 5: Actualización del Script de Prueba `scripts/test_config.py`

  * **Descripción Exhaustiva**: Revisar y actualizar el script `scripts/test_config.py` creado en Fase 0 para asegurar que utiliza las versiones refinadas de las funciones de carga, prueba la carga de todos los `params.yaml` y el `logging_config.yaml`.
  * **Acciones Específicas**:
      * **5.1. Revisar `scripts/test_config.py`**:
          * Ajustar las llamadas a `get_env_variable` y `load_yaml_config` según las nuevas firmas (si cambiaron significativamente, aunque el cambio principal fue interno y en manejo de errores).
          * Asegurar que `setup_logging()` se llama al inicio.
          * Añadir pruebas para cargar la configuración de todos los módulos (M1, M2, M3, M4, M7).
          * **Código de `scripts/test_config.py` (actualizado)**:
            ```python
            # scripts/test_config.py
            import sys
            from pathlib import Path
            import logging

            # Añadir src al PYTHONPATH
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            src_path = project_root / "src"
            if str(src_path) not in sys.path: # Evitar duplicados si ya está
                sys.path.insert(0, str(src_path))

            from common.utils import setup_logging
            from config_loader import load_yaml_config, get_env_variable

            # --- Configurar logging primero ---
            # Esto es crítico: setup_logging() DEBE llamarse antes de obtener cualquier logger.
            # Asume que common.utils y config_loader ya están disponibles
            # y que .env y logging_config.yaml están accesibles.
            try:
                setup_logging() # Usa la ruta por defecto a logging_config.yaml desde common.utils
            except Exception as e:
                # Fallback muy básico si setup_logging falla catastróficamente ANTES de configurarse
                logging.basicConfig(level=logging.ERROR)
                logging.critical(f"Fallo CRÍTICO al configurar logging: {e}", exc_info=True)
                # Podría ser necesario salir si el logging es indispensable para continuar
                # sys.exit(1) 

            logger = logging.getLogger(__name__) # Ahora __name__ será 'scripts.test_config' o similar

            def test_all_module_configs():
                """Intenta cargar la configuración para todos los módulos definidos."""
                module_configs_to_test = [
                    "module1_data_acquisition",
                    "module2_preprocessing",
                    "module3_environment",
                    "module4_agent_sac",
                    "module7_backtesting_wfo"
                ]
                all_successful = True
                for module_name in module_configs_to_test:
                    try:
                        config = load_yaml_config(module_name)
                        logger.info(f"Configuración para '{module_name}' cargada exitosamente: {config if config else '{}'}")
                        if not config: # Si el YAML está vacío pero es válido
                             logger.warning(f"El archivo de configuración para '{module_name}' está vacío pero es válido.")
                    except FileNotFoundError:
                        logger.error(f"Archivo de configuración NO ENCONTRADO para el módulo: {module_name}")
                        all_successful = False
                    except ValueError as ve:
                        logger.error(f"Error de parseo YAML para el módulo {module_name}: {ve}")
                        all_successful = False
                    except Exception as e:
                        logger.error(f"Error inesperado al cargar config para {module_name}: {e}", exc_info=True)
                        all_successful = False
                
                if not all_successful:
                    logger.error("Una o más configuraciones de módulo fallaron al cargar.")
                else:
                    logger.info("Todas las configuraciones de módulos principales cargadas exitosamente (o son válidamente vacías).")
                return all_successful

            def main():
                logger.info("======================================================================")
                logger.info("Iniciando prueba de carga de configuración (scripts/test_config.py)...")
                logger.info("======================================================================")

                try:
                    # Probar la carga de variables de entorno
                    logger.info("--- Probando variables de entorno ---")
                    binance_key = get_env_variable("BINANCE_API_KEY", "NO_KEY_DEFAULT_FOR_TEST", required=False)
                    logger.info(f"BINANCE_API_KEY (desde .env o default): '{binance_key}'")

                    redis_host = get_env_variable("REDIS_HOST", required=True) # Asumir que REDIS_HOST debe estar en .env
                    logger.info(f"REDIS_HOST (desde .env): '{redis_host}'")
                    
                    config_dir_container = get_env_variable("CONFIG_DIR_CONTAINER", required=True)
                    logger.info(f"CONFIG_DIR_CONTAINER (desde .env): '{config_dir_container}'")
                    logger.info("--- Variables de entorno probadas ---")

                    # Probar la carga de configuraciones YAML de módulos
                    logger.info("--- Probando carga de configuraciones YAML de módulos ---")
                    test_all_module_configs()
                    logger.info("--- Configuraciones YAML de módulos probadas ---")

                    # Probar la carga del logging_config.yaml explícitamente (aunque setup_logging ya lo hizo)
                    logger.info("--- Probando carga de logging_config.yaml explícitamente ---")
                    try:
                        log_conf = load_yaml_config("logging_config.yaml")
                        logger.info(f"logging_config.yaml cargado explícitamente con éxito (contenido parcial): level de root = {log_conf.get('root', {}).get('level')}")
                    except Exception as e:
                        logger.error(f"Error al cargar logging_config.yaml explícitamente: {e}")
                    logger.info("--- logging_config.yaml probado ---")


                    logger.info("========================================================================")
                    logger.info("Prueba de carga de configuración (scripts/test_config.py) completada.")
                    logger.info("Por favor, revisa los logs para detalles y posibles errores.")
                    logger.info("========================================================================")


                except FileNotFoundError as e:
                    logger.error(f"Error de archivo no encontrado durante la prueba principal: {e}", exc_info=True)
                except EnvironmentError as e:
                    logger.error(f"Error de variable de entorno requerida durante la prueba principal: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Ocurrió un error inesperado durante la prueba principal de configuración: {e}", exc_info=True)

            if __name__ == "__main__":
                main()
            ```
      * **5.2. Ejecutar el Script de Prueba Actualizado**:
          * Desde la raíz del proyecto, dentro del entorno Docker:
            ```bash
            # Asegurarse que los servicios estén corriendo (Fase 0, Paso 6.7)
            # docker-compose up -d --build workhorse_app redis 
            docker-compose exec workhorse_app python scripts/test_config.py
            ```
          * Revisar la salida en la consola y los archivos de log en `results_host/logs/` para confirmar que todo se carga como se espera y que los logs se generan correctamente.

-----

### Paso 6: Commit de los Cambios de la Fase 1

  * **Descripción Exhaustiva**: Añadir todos los cambios realizados durante esta fase al control de versiones Git.
  * **Acciones Específicas**:
      * **6.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add src/config_loader.py src/common/utils.py
        git add config/module1_data_acquisition/params.yaml # y los demás params.yaml
        git add config/module2_preprocessing/params.yaml
        git add config/module3_environment/params.yaml
        git add config/module4_agent_sac/params.yaml
        git add config/module7_backtesting_wfo/params.yaml
        git add tests/test_config_loader.py
        git add scripts/test_config.py
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 1: Robustecer y finalizar Módulo 9 (Configuración). Añadir pruebas unitarias para config_loader y completar params.yaml."
        ```

**Fin de la Fase de Implementación 1.**

-----


## Fase de Implementación 2: Módulo 1 - Adquisición y Gestión de Datos de Mercado

**Nombre Descriptivo de la Fase:** Implementación del Sistema de Descarga de Datos Históricos de Binance y Almacenamiento en Redis y Disco.

Esta fase se centra en desarrollar el Módulo 1, responsable de conectarse a la API de Binance, descargar datos históricos de futuros BTCUSDT (Klines, Libro de Órdenes, Trades Agregados), publicar estos datos en Redis y persistirlos en disco para su uso futuro y evitar descargas repetidas.

-----

### Paso 1: Creación de la Clase `BinanceDownloader` y Archivos Necesarios

  * **Descripción Exhaustiva**: Crear el archivo `src/data_acquisition/binance_downloader.py` que contendrá la clase `BinanceDownloader`. Esta clase encapsulará toda la lógica para la adquisición de datos. También se crearán archivos auxiliares si son necesarios.
  * **Acciones Específicas**:
      * **1.1. Crear `src/data_acquisition/__init__.py`** (si no existe):
        ```python
        # src/data_acquisition/__init__.py
        # Este archivo puede estar vacío o exportar clases principales.
        from .binance_downloader import BinanceDownloader

        __all__ = ['BinanceDownloader']
        ```
      * **1.2. Crear el Esqueleto de `src/data_acquisition/binance_downloader.py`**:
        ```python
        # src/data_acquisition/binance_downloader.py
        import pandas as pd
        import time
        import json
        import logging
        from datetime import datetime, timezone
        from pathlib import Path
        from typing import List, Dict, Any, Optional

        from binance.client import Client
        from binance.exceptions import BinanceAPIException, BinanceRequestException
        import redis # type: ignore[import-untyped] # Para compatibilidad con mypy si redis no tiene stubs completos

        # Suponiendo que config_loader y utils están en src/ y PYTHONPATH está configurado
        from config_loader import load_yaml_config, get_env_variable
        from common.utils import setup_logging # Asumiendo que setup_logging ya está configurado para ser llamado globalmente

        # Configurar logger para este módulo
        # setup_logging() debería ser llamado una vez al inicio de la aplicación (ej. en el script principal)
        # Por ahora, para desarrollo del módulo, podemos obtener el logger.
        logger = logging.getLogger(__name__) # o logging.getLogger(f"src.{__name__}")

        class BinanceDownloader:
            """
            Clase para descargar datos históricos de Binance (Futuros USDT-M).
            Incluye Klines, snapshots del libro de órdenes y trades agregados.
            Los datos se publican en Redis y se guardan en disco.
            """

            # Constantes para reintentos de API
            MAX_RETRIES = 5
            RETRY_DELAY_SECONDS = 5 # Aumentar con backoff exponencial

            def __init__(self):
                """
                Inicializa el BinanceDownloader.
                Carga la configuración del módulo y las credenciales API.
                Establece la conexión con el cliente de Binance y Redis.
                """
                logger.info("Inicializando BinanceDownloader...")
                try:
                    # Cargar configuración del módulo
                    self.module_config = load_yaml_config("module1_data_acquisition")
                    self.api_key = get_env_variable("BINANCE_API_KEY", required=True)
                    self.api_secret = get_env_variable("BINANCE_API_SECRET", required=True)
                    
                    self.redis_host = get_env_variable("REDIS_HOST", "localhost")
                    self.redis_port = int(get_env_variable("REDIS_PORT", "6379"))
                    # self.redis_password = get_env_variable("REDIS_PASSWORD", required=False) # Si se usa contraseña

                    # Rutas de datos persistentes (desde .env, mapeadas a /app/data_persistent en Docker)
                    self.data_dir_host_str = get_env_variable("DATA_DIR_HOST_FOR_APP", "/app/data_persistent")
                    self.raw_data_path = Path(self.data_dir_host_str) / "raw" / self.module_config.get("trading_pair", "BTCUSDT")
                    self.raw_data_path.mkdir(parents=True, exist_ok=True) # Asegurar que el directorio exista

                    # Inicializar cliente de Binance
                    # Para futuros USDT-M, se puede usar Client() normal. 
                    # Si se requiere especificar tld='com' o testnet, se puede añadir aquí.
                    self.binance_client = Client(self.api_key, self.api_secret)
                    self._test_binance_connection()

                    # Inicializar cliente de Redis
                    self.redis_client = redis.Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        # password=self.redis_password, # Descomentar si se usa
                        decode_responses=True # Para que las claves y valores se decodifiquen de bytes a str
                    )
                    self._test_redis_connection()
                    logger.info("BinanceDownloader inicializado correctamente.")

                except EnvironmentError as e:
                    logger.error(f"Error de variable de entorno durante la inicialización de BinanceDownloader: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error inesperado durante la inicialización de BinanceDownloader: {e}", exc_info=True)
                    raise

            def _test_binance_connection(self):
                """Prueba la conexión con la API de Binance."""
                try:
                    self.binance_client.ping()
                    server_time = self.binance_client.get_server_time()
                    logger.info(f"Conexión a Binance API exitosa. Hora del servidor: {datetime.fromtimestamp(server_time['serverTime']/1000, tz=timezone.utc)}")
                except Exception as e:
                    logger.error(f"Fallo al conectar con Binance API: {e}")
                    raise ConnectionError(f"No se pudo conectar a Binance API: {e}")

            def _test_redis_connection(self):
                """Prueba la conexión con Redis."""
                try:
                    self.redis_client.ping()
                    logger.info(f"Conexión a Redis ({self.redis_host}:{self.redis_port}) exitosa.")
                except redis.exceptions.ConnectionError as e:
                    logger.error(f"Fallo al conectar con Redis ({self.redis_host}:{self.redis_port}): {e}")
                    raise ConnectionError(f"No se pudo conectar a Redis: {e}")

            # --- Métodos de descarga (se implementarán en los siguientes pasos) ---
            
            def _make_api_request(self, api_method, *args, **kwargs) -> Any:
                """
                Realiza una solicitud a la API de Binance con manejo de reintentos y errores.
                """
                for attempt in range(self.MAX_RETRIES):
                    try:
                        return api_method(*args, **kwargs)
                    except BinanceAPIException as e:
                        logger.warning(f"Binance API Exception (Intento {attempt + 1}/{self.MAX_RETRIES}): Status {e.status_code}, Mensaje: {e.message}")
                        if e.status_code == 429 or e.status_code == 418: # Rate limit
                            delay = self.RETRY_DELAY_SECONDS * (2 ** attempt) # Backoff exponencial
                            logger.info(f"Rate limit alcanzado. Reintentando en {delay} segundos...")
                            time.sleep(delay)
                        elif e.status_code >= 500: # Error del servidor
                            logger.warning(f"Error del servidor de Binance. Reintentando en {self.RETRY_DELAY_SECONDS} segundos...")
                            time.sleep(self.RETRY_DELAY_SECONDS)
                        else: # Otros errores de API (ej. 400 Bad Request) no suelen resolverse con reintentos.
                            logger.error(f"Error de API no recuperable: {e}")
                            raise # Relanzar la excepción
                    except BinanceRequestException as e:
                        logger.error(f"Binance Request Exception (Intento {attempt + 1}/{self.MAX_RETRIES}): {e}")
                        time.sleep(self.RETRY_DELAY_SECONDS) # Reintentar en caso de problemas de red, etc.
                    except Exception as e:
                        logger.error(f"Excepción inesperada durante la llamada API (Intento {attempt + 1}/{self.MAX_RETRIES}): {e}", exc_info=True)
                        time.sleep(self.RETRY_DELAY_SECONDS)
                
                logger.error(f"Fallaron todos los {self.MAX_RETRIES} intentos para la llamada API: {api_method.__name__}")
                raise ConnectionError(f"Fallaron todos los {self.MAX_RETRIES} intentos para {api_method.__name__}")


            def _ms_to_datetime_str(self, ms: int) -> str:
                """Convierte milisegundos UTC a string ISO 8601 YYYY-MM-DDTHH:MM:SSZ."""
                return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

            def _datetime_to_ms_str(self, dt_str: str) -> str:
                """Convierte string de fecha/hora (YYYY-MM-DD o YYYY-MM-DD HH:MM:SS) a milisegundos UTC string."""
                # Intenta parsear varios formatos comunes
                try:
                    dt_obj = pd.to_datetime(dt_str, utc=True) # pandas es bueno parseando fechas
                except ValueError:
                     # Intenta con formato específico si pd.to_datetime falla
                    try:
                        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    except ValueError:
                        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                
                return str(int(dt_obj.timestamp() * 1000))

            def _get_data_save_path(self, data_type: str, interval: Optional[str] = None) -> Path:
                """Construye la ruta de guardado para un tipo de dato e intervalo."""
                path = self.raw_data_path / data_type
                if interval:
                    path = path / interval
                path.mkdir(parents=True, exist_ok=True)
                return path

            # Los métodos principales de descarga se definirán a continuación
            # def download_historical_klines(self, symbol: str, interval: str, start_date_str: str, end_date_str: str) -> None:
            # def download_order_book_snapshots(self, symbol: str, klines_df: pd.DataFrame) -> None:
            # def download_aggregated_trades(self, symbol: str, klines_df: pd.DataFrame) -> None:
            # def run_download_all(self) -> None: # Método principal para orquestar descargas

        if __name__ == '__main__':
            # Este bloque es para pruebas directas del módulo.
            # Asegúrate de que .env está configurado y Redis está disponible.
            # La configuración de logging debe ser inicializada por el script que llama a este.
            # Para pruebas, lo configuramos aquí de forma básica.
            
            # setup_logging() # Debería ser llamado por el script de ejecución principal
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger.info("Ejecutando prueba directa de BinanceDownloader...")
            
            # Crear una variable de entorno para el directorio de datos de prueba si no existe
            if "DATA_DIR_HOST_FOR_APP" not in os.environ:
                os.environ["DATA_DIR_HOST_FOR_APP"] = "./temp_test_data_downloader"
                logger.info(f"DATA_DIR_HOST_FOR_APP no estaba en .env, usando default temporal: {os.environ['DATA_DIR_HOST_FOR_APP']}")

            try:
                downloader = BinanceDownloader()
                logger.info("BinanceDownloader instanciado para prueba.")
                # Aquí se podrían llamar a los métodos de descarga una vez implementados.
                # Ejemplo:
                # downloader.download_historical_klines(
                #     symbol=downloader.module_config.get("trading_pair", "BTCUSDT"),
                #     interval=downloader.module_config.get("kline_interval", "15m"),
                #     start_date_str=downloader.module_config.get("data_download_start_date"),
                #     end_date_str=downloader.module_config.get("data_download_end_date")
                # )

            except ConnectionError as e:
                logger.error(f"Error de conexión durante la prueba: {e}")
            except EnvironmentError as e:
                logger.error(f"Error de configuración de entorno durante la prueba: {e}")
            except Exception as e:
                logger.error(f"Error inesperado durante la prueba de BinanceDownloader: {e}", exc_info=True)
            finally:
                 # Limpiar el directorio temporal si se creó
                if os.environ.get("DATA_DIR_HOST_FOR_APP") == "./temp_test_data_downloader":
                    temp_data_path = Path("./temp_test_data_downloader")
                    # (Considerar añadir lógica para borrar de forma segura si es necesario)
                    # import shutil
                    # if temp_data_path.exists():
                    #     shutil.rmtree(temp_data_path)
                    #     logger.info(f"Directorio de datos temporal {temp_data_path} eliminado.")
                    pass

        ```
      * **1.3. Añadir `DATA_DIR_HOST_FOR_APP` a `.env.example` y `.env`**:
          * Esta variable especifica la ruta *dentro del contenedor* donde se montan los datos persistentes. `DATA_DIR_HOST` era para la ruta en el host.
          * En `.env.example` (y el `.env` local):
            ```env
            # ... otras variables ...
            DATA_DIR_HOST_FOR_APP="/app/data_persistent" # Ruta absoluta de datos persistentes DENTRO del contenedor
            ```
          * Asegurarse que el volumen en `docker-compose.yml` mapee a esta ruta:
            ```yaml
            # En docker-compose.yml, servicio workhorse_app:
            # ...
            volumes:
              # ...
              - ${DATA_DIR_HOST:-./data_host}:${DATA_DIR_HOST_FOR_APP:-/app/data_persistent} # Mapea data_host a /app/data_persistent
            # ...
            ```
            *Corrección*: La Fase 0 tenía `- ${DATA_DIR_HOST:-./data_host}:/app/data_persistent`. Esto está bien. `DATA_DIR_HOST_FOR_APP` será la forma en que el código Python dentro del contenedor se refiere a `/app/data_persistent`.

-----

### Paso 2: Implementación de la Descarga de Klines Históricos

  * **Descripción Exhaustiva**: Implementar el método `download_historical_klines` en `BinanceDownloader`. Este método descargará los datos de klines/candlesticks para el par y el intervalo especificados, dentro del rango de fechas configurado. Los datos se guardarán en formato Parquet (un archivo por día o mes para manejabilidad) y se publicarán en Redis.
  * **Acciones Específicas**:
      * **2.1. Definir el método `download_historical_klines`**:
        ```python
        # En src/data_acquisition/binance_downloader.py, dentro de la clase BinanceDownloader

        def download_historical_klines(self, symbol: str, interval: str, start_date_str: str, end_date_str: str) -> Optional[pd.DataFrame]:
            """
            Descarga klines históricos para un símbolo, intervalo y rango de fechas.
            Guarda los datos en archivos Parquet (diarios) y publica en Redis.

            Args:
                symbol (str): El par de trading (ej. "BTCUSDT").
                interval (str): El intervalo de kline (ej. "15m", "1h", "1d").
                start_date_str (str): Fecha de inicio en formato "YYYY-MM-DD" o "YYYY-MM-DD HH:MM:SS".
                end_date_str (str): Fecha de fin en formato "YYYY-MM-DD" o "YYYY-MM-DD HH:MM:SS".
            
            Returns:
                Optional[pd.DataFrame]: DataFrame con todos los klines descargados, o None si falla.
            """
            logger.info(f"Iniciando descarga de klines históricos para {symbol}, intervalo {interval}, desde {start_date_str} hasta {end_date_str}.")
            
            klines_path = self._get_data_save_path("klines", interval)
            klines_redis_key_prefix = f"data:raw:klines:{symbol}:{interval}"

            # Convertir fechas a milisegundos para la API de Binance
            start_ms = int(self._datetime_to_ms_str(start_date_str))
            end_ms = int(self._datetime_to_ms_str(end_date_str))

            all_klines_list = []
            current_start_ms = start_ms

            # Columnas para el DataFrame de klines
            kline_columns = [
                'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
                'kline_close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]

            while current_start_ms < end_ms:
                limit = 1000 # Límite máximo de klines por solicitud de Binance (1500 para futuros, 1000 para spot)
                             # Verificar documentación de python-binance para futuros. Usaremos 1000 por seguridad.
                
                logger.debug(f"Descargando klines para {symbol} desde {self._ms_to_datetime_str(current_start_ms)} (limit: {limit})")
                
                # Usar el wrapper _make_api_request
                klines_batch = self._make_api_request(
                    self.binance_client.get_historical_klines,
                    symbol=symbol,
                    interval=interval,
                    start_str=str(current_start_ms),
                    end_str=str(end_ms), # La API usa end_str para limitar el rango total, no solo de este batch
                    limit=limit
                )

                if not klines_batch:
                    logger.info(f"No se recibieron más klines para {symbol} después de {self._ms_to_datetime_str(current_start_ms)}. Fin de la descarga.")
                    break

                all_klines_list.extend(klines_batch)
                
                # Actualizar current_start_ms al tiempo de apertura del último kline + 1 unidad de intervalo
                # para evitar solapamiento y asegurar la progresión.
                last_kline_open_time_ms = klines_batch[-1][0]
                current_start_ms = last_kline_open_time_ms + self._interval_to_milliseconds(interval)

                # Pequeña pausa para ser cortés con la API, aunque _make_api_request maneja rate limits
                time.sleep(0.2) # 200 ms

            if not all_klines_list:
                logger.warning(f"No se descargaron klines para {symbol} en el rango especificado.")
                return None

            # Convertir lista de klines a DataFrame
            klines_df = pd.DataFrame(all_klines_list, columns=kline_columns)
            # Convertir columnas numéricas
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            for col in numeric_cols:
                klines_df[col] = pd.to_numeric(klines_df[col])
            klines_df['number_of_trades'] = klines_df['number_of_trades'].astype(int)
            
            # Convertir timestamps a datetime y añadir columna legible
            klines_df['kline_open_time_dt'] = pd.to_datetime(klines_df['kline_open_time'], unit='ms', utc=True)
            klines_df['kline_close_time_dt'] = pd.to_datetime(klines_df['kline_close_time'], unit='ms', utc=True)
            
            # Guardar en disco por día y publicar en Redis
            # El guardado por día es para manejar grandes datasets.
            # La publicación en Redis podría ser de cada kline individual o batches.
            # Por simplicidad, publicaremos cada kline como un JSON string.
            
            logger.info(f"Total klines descargados para {symbol} ({interval}): {len(klines_df)}")
            if klines_df.empty:
                return klines_df # Retorna DataFrame vacío

            # Guardar y publicar
            grouped_by_day = klines_df.groupby(klines_df['kline_open_time_dt'].dt.date)
            total_published_redis = 0
            for date_obj, daily_klines_df in grouped_by_day:
                date_str = date_obj.strftime('%Y-%m-%d')
                file_path = klines_path / f"{symbol}_{interval}_klines_{date_str}.parquet"
                try:
                    daily_klines_df.to_parquet(file_path, index=False)
                    logger.info(f"Klines para {date_str} guardados en: {file_path}")
                except Exception as e:
                    logger.error(f"Error al guardar klines para {date_str} en Parquet: {e}", exc_info=True)

                # Publicar en Redis (ejemplo: cada kline como un mensaje JSON)
                # El canal podría ser más específico, ej., incluir la fecha.
                # O un stream de Redis. Por ahora, un canal por kline.
                # Esto podría generar MUCHAS claves si los datos son extensos.
                # Considerar publicar el DataFrame diario como JSON o usar streams.
                # Para este MVP, publicaremos individualmente para ilustrar.
                for _, kline_row in daily_klines_df.iterrows():
                    # Usar kline_open_time (timestamp ms) como parte de la clave o un score si es un ZSET
                    redis_key = f"{klines_redis_key_prefix}:{kline_row['kline_open_time']}"
                    try:
                        # Convertir la fila a JSON. pd.Timestamp necesita manejo especial.
                        kline_data_json = kline_row.astype(str).to_json() # Convertir todo a str para JSON simple
                        self.redis_client.set(redis_key, kline_data_json) # O usar LPUSH a una lista, o XADD a un stream
                        total_published_redis += 1
                    except Exception as e:
                        logger.error(f"Error al publicar kline {redis_key} en Redis: {e}")
            
            logger.info(f"Total de {total_published_redis} klines individuales publicados en Redis (prefijo: {klines_redis_key_prefix}:<timestamp>)")
            logger.info(f"Descarga de klines para {symbol} ({interval}) completada.")
            return klines_df

        def _interval_to_milliseconds(self, interval_str: str) -> int:
            """Convierte un string de intervalo de Binance a milisegundos."""
            multipliers = {'m': 60, 'h': 60*60, 'd': 24*60*60, 'w': 7*24*60*60} # Segundos
            try:
                value = int(interval_str[:-1])
                unit = interval_str[-1]
                if unit not in multipliers:
                    raise ValueError(f"Unidad de intervalo desconocida: {unit}")
                return value * multipliers[unit] * 1000 # Convertir a milisegundos
            except Exception as e:
                logger.error(f"Error al convertir intervalo '{interval_str}' a milisegundos: {e}")
                # Fallback a un valor común (ej. 15m) o relanzar
                raise ValueError(f"Formato de intervalo inválido: {interval_str}") from e
        ```

-----

### Paso 3: Implementación de la Descarga de Snapshots del Libro de Órdenes

  * **Descripción Exhaustiva**: Implementar el método `download_order_book_snapshots`. Este método tomará los timestamps de cierre de los klines descargados previamente y, para cada uno, obtendrá un snapshot del libro de órdenes (top N niveles). Los datos se guardarán en Parquet y se publicarán en Redis.
  * **Acciones Específicas**:
      * **3.1. Definir el método `download_order_book_snapshots`**:
        ```python
        # En src/data_acquisition/binance_downloader.py, dentro de la clase BinanceDownloader

        def download_order_book_snapshots(self, symbol: str, klines_df: pd.DataFrame) -> None:
            """
            Descarga snapshots del libro de órdenes para cada kline en el DataFrame proporcionado.
            Guarda los datos en archivos Parquet y publica en Redis.

            Args:
                symbol (str): El par de trading (ej. "BTCUSDT").
                klines_df (pd.DataFrame): DataFrame de klines previamente descargados.
                                          Debe contener 'kline_close_time' y 'kline_close_time_dt'.
            """
            if klines_df.empty:
                logger.warning(f"No hay klines para procesar para snapshots de libro de órdenes de {symbol}.")
                return

            logger.info(f"Iniciando descarga de snapshots de libro de órdenes para {symbol}, correlacionados con {len(klines_df)} klines.")
            
            # Obtener parámetros de configuración
            depth_limit = self.module_config.get("order_book_depth", 5) # Top N niveles
            # El intervalo se infiere de los klines, pero es útil para nombrar archivos/canales.
            # Se asume que todos los klines son del mismo intervalo.
            # Podríamos obtenerlo del nombre del archivo de klines o pasarlo como arg.
            # Por ahora, lo omitimos de los nombres de archivo de order_book si es general.
            
            order_book_path = self._get_data_save_path("order_book") # No se especifica intervalo aquí
            order_book_redis_prefix = f"data:raw:order_book:{symbol}"

            all_snapshots_data = [] # Lista para acumular datos para guardado en batch

            for index, kline_row in klines_df.iterrows():
                kline_close_time_ms = kline_row['kline_close_time'] # Timestamp en ms
                kline_close_time_dt_str = kline_row['kline_close_time_dt'].strftime('%Y-%m-%dT%H%M%SZ')
                
                logger.debug(f"Obteniendo snapshot de libro de órdenes para {symbol} al cierre de kline: {kline_close_time_dt_str} (ts: {kline_close_time_ms})")

                try:
                    # La API de Binance para order book no toma timestamp. Devuelve el actual.
                    # Esto es una limitación: NO PODEMOS obtener snapshots históricos del libro de órdenes
                    # directamente a través de la API pública estándar de Binance para momentos específicos del pasado.
                    # El README dice "Snapshots al cierre de cada KLine". Esto implica que o bien:
                    # 1. Se usa una API de datos históricos especializada (no python-binance estándar).
                    # 2. Se capturan en tiempo real y se correlacionan (no es lo que hacemos aquí para datos históricos).
                    # 3. Hay un malentendido de lo que la API pública puede hacer.
                    #
                    # Para el MVP, si la API pública es la única opción, NO PODEMOS obtener snapshots históricos.
                    # LO QUE SÍ PODEMOS HACER es, si estuviéramos en un sistema de TRADING EN VIVO,
                    # tomar un snapshot DESPUÉS de que una kline se cierra.
                    #
                    # Asumiré para este MVP que el requerimiento del README es idealista para la API pública histórica.
                    # Procederé con una advertencia y no implementaré la descarga real de snapshots históricos
                    # ya que `client.get_order_book()` da el estado actual.
                    #
                    # Si el objetivo es SIMULAR el tipo de datos que se tendría:
                    # Se podría generar datos sintéticos o usar una fuente de datos premium que sí los provea.
                    #
                    # **ACCIÓN CORRECTIVA PARA MVP:** Emitir una advertencia y saltar esta parte si no hay fuente.
                    # Si el usuario tiene una forma alternativa de obtener estos datos (ej. archivos preexistentes, API diferente),
                    # esta función necesitaría adaptarse.

                    # ---- INICIO DE BLOQUE ILUSTRATIVO SI LA API LO PERMITIERA ----
                    # # ESTO ES PSEUDOCÓDIGO - la API actual no soporta `time` o similar para histórico.
                    # # order_book_snapshot = self._make_api_request(
                    # # self.binance_client.get_historical_order_book, # MÉTODO FICTICIO
                    # # symbol=symbol,
                    # # timestamp=kline_close_time_ms, # FICTICIO
                    # # limit=depth_limit
                    # # )
                    #
                    # # Ejemplo de estructura de datos que se esperaría (si fuera posible):
                    # # order_book_snapshot = {'lastUpdateId': 12345, 'bids': [['price', 'qty'], ...], 'asks': [['price', 'qty'], ...]}
                    #
                    # snapshot_record = {
                    # 'kline_close_time_ms': kline_close_time_ms,
                    # 'kline_close_time_dt_str': kline_close_time_dt_str,
                    # 'symbol': symbol,
                    # 'lastUpdateId': order_book_snapshot.get('lastUpdateId'),
                    # 'bids': order_book_snapshot.get('bids', [])[:depth_limit], # Tomar solo N niveles
                    # 'asks': order_book_snapshot.get('asks', [])[:depth_limit], # Tomar solo N niveles
                    # }
                    # all_snapshots_data.append(snapshot_record)
                    #
                    # # Publicar en Redis
                    # redis_key = f"{order_book_redis_prefix}:{kline_close_time_ms}"
                    # self.redis_client.set(redis_key, json.dumps(snapshot_record))
                    # ---- FIN DE BLOQUE ILUSTRATIVO ----

                    # Simulación de no disponibilidad para API pública histórica:
                    if index == 0: # Mostrar advertencia solo una vez
                        logger.warning(f"ADVERTENCIA: La API pública estándar de Binance no provee snapshots históricos del libro de órdenes para momentos específicos del pasado.")
                        logger.warning(f"La funcionalidad de 'download_order_book_snapshots' no puede implementarse como se describe en el README con la API pública.")
                        logger.warning(f"Se omitirá la descarga de snapshots del libro de órdenes. Considera usar una fuente de datos especializada o ajustar los requerimientos del MVP.")
                    # Rompemos el bucle o simplemente no hacemos nada más aquí.
                    # Para evitar que el log se llene, retornamos.
                    return

                except BinanceAPIException as e:
                    logger.error(f"Error de API al intentar obtener libro de órdenes para {symbol} (correlacionado con kline {kline_close_time_dt_str}): {e}")
                    # Continuar con el siguiente kline si es un error puntual
                    time.sleep(1) # Pequeña pausa
                except Exception as e:
                    logger.error(f"Error inesperado al procesar libro de órdenes para {symbol} (kline {kline_close_time_dt_str}): {e}", exc_info=True)
                    time.sleep(1)
                
                # Pausa para no sobrecargar (si la descarga fuera real)
                if (index + 1) % 10 == 0: # Cada 10 klines
                    time.sleep(0.5)

            if not all_snapshots_data: # Si el bloque ilustrativo se ejecutara y no hubiera datos
                logger.info(f"No se generaron datos de snapshots de libro de órdenes para {symbol}.")
                return

            # Guardar en disco (si se hubieran recolectado datos)
            # snapshots_df = pd.DataFrame(all_snapshots_data)
            # date_str_for_file = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') # O agrupar por día de kline
            # file_path = order_book_path / f"{symbol}_order_book_snapshots_{date_str_for_file}.parquet"
            # try:
            # snapshots_df.to_parquet(file_path, index=False)
            #     logger.info(f"Todos los snapshots de libro de órdenes ({len(snapshots_df)}) guardados en: {file_path}")
            # except Exception as e:
            #     logger.error(f"Error al guardar snapshots de libro de órdenes en Parquet: {e}", exc_info=True)

            logger.info(f"Procesamiento de 'download_order_book_snapshots' para {symbol} completado (con las limitaciones mencionadas).")

        ```
      * **Nota Importante sobre Libro de Órdenes Histórico**: La API pública de Binance (`python-binance`) **no permite** descargar snapshots del libro de órdenes para momentos específicos en el pasado. `get_order_book()` devuelve el estado actual. Esto es una limitación crucial. La implementación anterior refleja esta limitación y emite una advertencia. Si se dispone de una fuente de datos histórica para libros de órdenes (ej. archivos CSV/Parquet de un proveedor de datos, o una API de pago), esta función debería modificarse para leer de esa fuente en lugar de intentar llamar a la API de Binance para datos históricos inexistentes. **Para el MVP, se debe asumir que estos datos no están disponibles a través de la descarga directa a menos que se aclare la fuente.**

-----

### Paso 4: Implementación de la Descarga de Trades Agregados

  * **Descripción Exhaustiva**: Implementar el método `download_aggregated_trades`. Este método descargará trades agregados por Binance para cada intervalo de kline descargado. Los datos se guardarán en Parquet y se publicarán en Redis.
  * **Acciones Específicas**:
      * **4.1. Definir el método `download_aggregated_trades`**:
        ```python
        # En src/data_acquisition/binance_downloader.py, dentro de la clase BinanceDownloader

        def download_aggregated_trades(self, symbol: str, klines_df: pd.DataFrame) -> None:
            """
            Descarga trades agregados para cada intervalo de kline en el DataFrame proporcionado.
            Guarda los datos en archivos Parquet y publica en Redis.

            Args:
                symbol (str): El par de trading (ej. "BTCUSDT").
                klines_df (pd.DataFrame): DataFrame de klines previamente descargados.
                                          Debe contener 'kline_open_time' y 'kline_close_time'.
            """
            if klines_df.empty:
                logger.warning(f"No hay klines para procesar para trades agregados de {symbol}.")
                return

            logger.info(f"Iniciando descarga de trades agregados para {symbol}, correlacionados con {len(klines_df)} klines.")
            
            # El intervalo se infiere de los klines.
            # interval_ms = self._interval_to_milliseconds(klines_df.iloc[1]['kline_open_time'] - klines_df.iloc[0]['kline_open_time']) # Si es uniforme
            # De forma más segura, se pasa el intervalo o se calcula consistentemente.
            # Por ahora, asumimos que el intervalo es el mismo que el de los klines.
            
            agg_trades_path = self._get_data_save_path("agg_trades") # No se especifica intervalo aquí
            agg_trades_redis_prefix = f"data:raw:agg_trades:{symbol}"

            all_agg_trades_data = [] # Lista para acumular datos para guardado en batch
            
            # Columnas para DataFrame de trades agregados de Binance
            # a = tradeId, p = price, q = quantity, f = firstTradeId, l = lastTradeId, T = timestamp, m = isMaker
            agg_trade_columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 
                                 'timestamp_ms', 'is_maker_buyer', 'is_best_price_match'] # 'is_best_price_match' no siempre está.

            for index, kline_row in klines_df.iterrows():
                kline_open_time_ms = kline_row['kline_open_time']
                kline_close_time_ms = kline_row['kline_close_time'] # El kline_close_time es exclusivo para la API get_historical_klines
                                                                    # Para get_aggregate_trades, el end_time es inclusivo.
                                                                    # Usaremos kline_open_time como start_time y kline_open_time + interval_duration - 1ms como end_time.
                
                # Calcular la duración del intervalo de kline
                if index > 0:
                    interval_duration_ms = kline_open_time_ms - klines_df.iloc[index-1]['kline_open_time']
                elif len(klines_df) > 1: # Para el primer kline, usar la diferencia con el segundo
                    interval_duration_ms = klines_df.iloc[index+1]['kline_open_time'] - kline_open_time_ms
                else: # Solo un kline, no podemos determinar el intervalo así. Usar config.
                    interval_str_cfg = self.module_config.get("kline_interval", "15m") # Fallback
                    interval_duration_ms = self._interval_to_milliseconds(interval_str_cfg)

                effective_end_time_ms = kline_open_time_ms + interval_duration_ms -1 # end_time es inclusivo

                kline_open_time_dt_str = kline_row['kline_open_time_dt'].strftime('%Y-%m-%dT%H%M%SZ')
                logger.debug(f"Obteniendo trades agregados para {symbol} durante kline que abre en: {kline_open_time_dt_str}")

                try:
                    # La API get_aggregate_trades puede devolver hasta 1000 trades.
                    # Si hay más en un intervalo de kline, se necesitaría paginación por fromId.
                    # Para MVP, asumimos que 1000 es suficiente por kline de 15m.
                    # Si no, se necesitaría un bucle con fromId.
                    agg_trades_batch = self._make_api_request(
                        self.binance_client.get_aggregate_trades,
                        symbol=symbol,
                        startTime=kline_open_time_ms,
                        endTime=effective_end_time_ms, # Binance API: endTime es inclusivo
                        limit=1000 # Max limit
                    )

                    if agg_trades_batch:
                        for trade in agg_trades_batch:
                            trade_record = {
                                'kline_open_time_ms': kline_open_time_ms, # Para correlacionar
                                'agg_trade_id': trade['a'],
                                'price': float(trade['p']),
                                'quantity': float(trade['q']),
                                'first_trade_id': trade['f'],
                                'last_trade_id': trade['l'],
                                'timestamp_ms': trade['T'],
                                'is_maker_buyer': trade['m'],
                                # 'is_best_price_match': trade['M'] # 'M' no siempre está, verificar
                            }
                            all_agg_trades_data.append(trade_record)
                            
                            # Publicar en Redis (individualmente o en batch)
                            redis_key = f"{agg_trades_redis_prefix}:{kline_open_time_ms}:{trade['a']}" # kline_ts:trade_id
                            self.redis_client.set(redis_key, json.dumps(trade_record))
                    
                except BinanceAPIException as e:
                    logger.error(f"Error de API al obtener trades agregados para kline {kline_open_time_dt_str}: {e}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error inesperado al procesar trades agregados para kline {kline_open_time_dt_str}: {e}", exc_info=True)
                    time.sleep(1)
                
                if (index + 1) % 5 == 0: # Pausa cada 5 klines
                    time.sleep(0.3)

            if not all_agg_trades_data:
                logger.info(f"No se descargaron datos de trades agregados para {symbol}.")
                return

            # Guardar en disco
            agg_trades_df = pd.DataFrame(all_agg_trades_data)
            agg_trades_df['timestamp_dt'] = pd.to_datetime(agg_trades_df['timestamp_ms'], unit='ms', utc=True)
            
            # Guardar por día (basado en el kline_open_time_ms para agrupar)
            agg_trades_df['kline_open_time_dt'] = pd.to_datetime(agg_trades_df['kline_open_time_ms'], unit='ms', utc=True)
            grouped_by_day = agg_trades_df.groupby(agg_trades_df['kline_open_time_dt'].dt.date)
            
            for date_obj, daily_agg_trades_df in grouped_by_day:
                date_str = date_obj.strftime('%Y-%m-%d')
                # Crear una copia para evitar SettingWithCopyWarning al quitar la columna auxiliar
                daily_df_to_save = daily_agg_trades_df.copy()
                daily_df_to_save.drop(columns=['kline_open_time_dt'], inplace=True)

                file_path = agg_trades_path / f"{symbol}_agg_trades_{date_str}.parquet"
                try:
                    daily_df_to_save.to_parquet(file_path, index=False)
                    logger.info(f"Trades agregados para klines del {date_str} guardados en: {file_path} ({len(daily_df_to_save)} trades)")
                except Exception as e:
                    logger.error(f"Error al guardar trades agregados para {date_str} en Parquet: {e}", exc_info=True)

            logger.info(f"Descarga de trades agregados para {symbol} completada. Total trades: {len(agg_trades_df)}")
        ```

-----

### Paso 5: Método Principal de Orquestación y Script de Ejecución

  * **Descripción Exhaustiva**: Crear un método `run_download_all` en `BinanceDownloader` que orqueste la descarga de todos los tipos de datos. Crear un script `scripts/download_data.py` que use esta clase para iniciar el proceso.
  * **Acciones Específicas**:
      * **5.1. Definir `run_download_all` en `BinanceDownloader`**:
        ```python
        # En src/data_acquisition/binance_downloader.py, dentro de la clase BinanceDownloader

        def run_download_all(self) -> None:
            """
            Orquesta la descarga de todos los tipos de datos configurados (klines, y luego otros).
            """
            logger.info("Iniciando proceso de descarga de todos los datos...")
            
            symbol = self.module_config.get("trading_pair", "BTCUSDT")
            kline_interval = self.module_config.get("kline_interval", "15m")
            start_date = self.module_config.get("data_download_start_date")
            end_date = self.module_config.get("data_download_end_date")

            if not all([symbol, kline_interval, start_date, end_date]):
                logger.error("Configuración incompleta para la descarga de datos (símbolo, intervalo, fechas). Abortando.")
                return

            # 1. Descargar Klines
            klines_df = None
            try:
                klines_df = self.download_historical_klines(symbol, kline_interval, start_date, end_date)
            except Exception as e:
                logger.error(f"Fallo crítico durante la descarga de klines: {e}", exc_info=True)
                # Decidir si continuar con otros datos o abortar. Por ahora, abortamos si klines falla.
                return

            if klines_df is None or klines_df.empty:
                logger.warning(f"No se pudieron descargar klines para {symbol}. No se procederá con datos dependientes (libro de órdenes, trades).")
                return

            # 2. Descargar Snapshots del Libro de Órdenes (con las advertencias ya mencionadas)
            try:
                self.download_order_book_snapshots(symbol, klines_df)
            except Exception as e:
                logger.error(f"Fallo durante el procesamiento de snapshots del libro de órdenes: {e}", exc_info=True)
                # Continuar de todas formas, ya que no es crítico si falla dado el problema de la API

            # 3. Descargar Trades Agregados
            try:
                self.download_aggregated_trades(symbol, klines_df)
            except Exception as e:
                logger.error(f"Fallo crítico durante la descarga de trades agregados: {e}", exc_info=True)

            logger.info("Proceso de descarga de todos los datos completado.")
        ```
      * **5.2. Crear `scripts/download_data.py`**:
        ```python
        # scripts/download_data.py
        import sys
        import logging
        from pathlib import Path

        # Añadir src al PYTHONPATH
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from common.utils import setup_logging
        from data_acquisition.binance_downloader import BinanceDownloader

        # Configurar logging (debe hacerse antes de que cualquier módulo lo use)
        try:
            setup_logging() # Esto cargará la configuración desde logging_config.yaml
        except Exception as e:
            # Configuración de logging de emergencia si setup_logging falla
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logging.critical(f"Fallo CRÍTICO al configurar logging centralizado: {e}. Usando logging básico.", exc_info=True)

        logger = logging.getLogger(__name__) # Obtener logger después de la configuración

        def main():
            logger.info("=======================================================")
            logger.info("Iniciando script de descarga de datos de mercado...")
            logger.info("=======================================================")
            
            try:
                downloader = BinanceDownloader()
                downloader.run_download_all()
                logger.info("Script de descarga de datos de mercado finalizado.")
            except ConnectionError as e:
                logger.error(f"Error de conexión en el script principal: {e}", exc_info=True)
            except EnvironmentError as e:
                logger.error(f"Error de configuración de entorno en el script principal: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error inesperado en el script principal de descarga: {e}", exc_info=True)
            finally:
                logger.info("=======================================================")
                logger.info("Finalización del script de descarga de datos.")
                logger.info("=======================================================")

        if __name__ == "__main__":
            main()
        ```
      * **5.3. Ejecutar el Script de Descarga (Prueba Inicial)**:
          * Asegurarse que `.env` tiene credenciales válidas de Binance (reales o de testnet si `python-binance` lo soporta fácilmente para futuros).
          * Para una prueba inicial, modificar `config/module1_data_acquisition/params.yaml` para un rango de fechas corto (ej. 1-2 días).
            ```yaml
            # config/module1_data_acquisition/params.yaml (para prueba corta)
            kline_interval: "15m"
            order_book_depth: 5
            data_download_start_date: "2024-05-01" # Un día reciente
            data_download_end_date: "2024-05-02"   # Un día después
            trading_pair: "BTCUSDT"
            ```
          * Ejecutar dentro de Docker:
            ```bash
            # Reconstruir si hay cambios en Dockerfile o requirements.txt
            # docker-compose up -d --build workhorse_app redis 

            # Asegurar que el directorio de datos del host exista
            mkdir -p ./data_host/raw/BTCUSDT/klines/15m
            mkdir -p ./data_host/raw/BTCUSDT/order_book
            mkdir -p ./data_host/raw/BTCUSDT/agg_trades
            # (La clase BinanceDownloader ya crea subdirectorios, pero raw/BTCUSDT debe existir)

            docker-compose exec workhorse_app python scripts/download_data.py
            ```
          * Revisar los logs en la consola y en `results_host/logs/`.
          * Verificar que se creen archivos Parquet en `data_host/raw/BTCUSDT/...`
          * Verificar (opcionalmente usando `redis-cli`) que se creen claves en Redis.

-----

### Paso 6: Pruebas Unitarias (Básicas) para `BinanceDownloader`

  * **Descripción Exhaustiva**: Crear pruebas unitarias básicas para `BinanceDownloader`. Dado que interactúa con APIs externas, se usarán mocks para simular las respuestas de Binance API y Redis.
  * **Acciones Específicas**:
      * **6.1. Crear archivo `tests/data_acquisition/test_binance_downloader.py`**:
        ```python
        # tests/data_acquisition/test_binance_downloader.py
        import pytest
        from unittest import mock
        from pathlib import Path
        import pandas as pd
        import os

        # Asegurar que src esté en el path
        # (pytest usualmente maneja esto)

        from src.data_acquisition.binance_downloader import BinanceDownloader
        # from src.config_loader import get_env_variable # Para mockearlo si es necesario

        @pytest.fixture
        def mock_binance_client():
            """Mockea el cliente de Binance."""
            client_mock = mock.Mock()
            client_mock.ping.return_value = None # Simula éxito
            client_mock.get_server_time.return_value = {'serverTime': 1678886400000} # Ejemplo de timestamp
            
            # Mock para get_historical_klines
            # Formato: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume, ignore]
            client_mock.get_historical_klines.return_value = [
                [1678886400000, "20000", "20100", "19900", "20050", "100", 1678886400000 + 15*60*1000 -1, "2000000", 500, "50", "1000000", "0"],
                [1678886400000 + 15*60*1000, "20050", "20150", "20000", "20100", "120", 1678886400000 + 30*60*1000 -1, "2400000", 600, "60", "1200000", "0"]
            ]
            # Mock para get_aggregate_trades
            client_mock.get_aggregate_trades.return_value = [
                {'a': 1, 'p': '20010', 'q': '0.1', 'f': 100, 'l': 102, 'T': 1678886400100, 'm': True},
                {'a': 2, 'p': '20015', 'q': '0.2', 'f': 103, 'l': 105, 'T': 1678886400200, 'm': False},
            ]
            # (get_order_book no se usa para histórico, así que no se mockea aquí a menos que la lógica cambie)
            return client_mock

        @pytest.fixture
        def mock_redis_client():
            """Mockea el cliente de Redis."""
            redis_mock = mock.Mock()
            redis_mock.ping.return_value = True # Simula éxito
            redis_mock.set.return_value = True
            return redis_mock

        @pytest.fixture
        def mock_config_and_env(monkeypatch, tmp_path):
            """Mockea la carga de configuración y variables de entorno necesarias."""
            # Mockear get_env_variable para retornar valores fijos
            def mock_get_env(var_name, default_value=None, required=True):
                if var_name == "BINANCE_API_KEY": return "test_api_key"
                if var_name == "BINANCE_API_SECRET": return "test_api_secret"
                if var_name == "REDIS_HOST": return "localhost"
                if var_name == "REDIS_PORT": return "6379"
                if var_name == "DATA_DIR_HOST_FOR_APP": return str(tmp_path / "test_data_downloader")
                if var_name == "CONFIG_DIR_CONTAINER": return str(tmp_path / "test_config_downloader") # Si es necesario
                if default_value is not None: return default_value
                if required: raise EnvironmentError(f"Mocked env var {var_name} is required but not set in mock.")
                return None
            
            monkeypatch.setattr("src.data_acquisition.binance_downloader.get_env_variable", mock_get_env)

            # Mockear load_yaml_config
            def mock_load_yaml(module_name, file_name="params.yaml"):
                if module_name == "module1_data_acquisition":
                    return {
                        "kline_interval": "15m",
                        "order_book_depth": 5,
                        "data_download_start_date": "2023-03-15",
                        "data_download_end_date": "2023-03-16",
                        "trading_pair": "BTCUSDT"
                    }
                return {}
            monkeypatch.setattr("src.data_acquisition.binance_downloader.load_yaml_config", mock_load_yaml)
            
            # Crear el directorio de datos de prueba que DATA_DIR_HOST_FOR_APP apuntaría
            (tmp_path / "test_data_downloader").mkdir(parents=True, exist_ok=True)


        @mock.patch("src.data_acquisition.binance_downloader.Client") # Mockear la clase Client de Binance
        @mock.patch("src.data_acquisition.binance_downloader.redis.Redis") # Mockear la clase Redis
        def test_downloader_initialization(self, MockRedis, MockBinanceClient, mock_config_and_env, mock_binance_client, mock_redis_client):
            """Testea la inicialización de BinanceDownloader."""
            MockBinanceClient.return_value = mock_binance_client
            MockRedis.return_value = mock_redis_client
            
            downloader = BinanceDownloader()
            
            assert downloader.binance_client is not None
            assert downloader.redis_client is not None
            mock_binance_client.ping.assert_called_once()
            mock_redis_client.ping.assert_called_once()
            assert downloader.module_config["trading_pair"] == "BTCUSDT"

        @mock.patch("src.data_acquisition.binance_downloader.Client")
        @mock.patch("src.data_acquisition.binance_downloader.redis.Redis")
        def test_download_historical_klines_success(self, MockRedis, MockBinanceClient, mock_config_and_env, mock_binance_client, mock_redis_client, tmp_path):
            MockBinanceClient.return_value = mock_binance_client
            MockRedis.return_value = mock_redis_client

            downloader = BinanceDownloader()
            
            # Asegurar que la ruta de guardado exista (tmp_path es el base para DATA_DIR_HOST_FOR_APP)
            symbol = downloader.module_config["trading_pair"]
            interval = downloader.module_config["kline_interval"]
            # (Path(tmp_path / "test_data_downloader" / "raw" / symbol / "klines" / interval)).mkdir(parents=True, exist_ok=True) # Creado por _get_data_save_path

            klines_df = downloader.download_historical_klines(
                symbol=symbol,
                interval=interval,
                start_date_str=downloader.module_config["data_download_start_date"],
                end_date_str=downloader.module_config["data_download_end_date"]
            )

            assert klines_df is not None
            assert len(klines_df) == 2 # Basado en el mock_binance_client
            mock_binance_client.get_historical_klines.assert_called()
            # Verificar que se haya llamado a redis set (número de llamadas = número de klines)
            assert mock_redis_client.set.call_count == 2 
            
            # Verificar que el archivo parquet se haya creado
            # La fecha viene de los datos mockeados (1678886400000 -> 2023-03-15)
            expected_parquet_path = downloader.raw_data_path / "klines" / interval / f"{symbol}_{interval}_klines_2023-03-15.parquet"
            assert expected_parquet_path.exists()
            
            # Leer el parquet y verificar contenido
            df_from_parquet = pd.read_parquet(expected_parquet_path)
            assert len(df_from_parquet) == 2


        # (Añadir más tests para download_aggregated_trades con mocks apropiados)
        # (No se pueden testear snapshots de libro de órdenes históricos directamente con la API pública)

        ```
      * **6.2. Ejecutar las Pruebas**:
        ```bash
        python -m pytest tests/data_acquisition/test_binance_downloader.py --cov=src/data_acquisition
        ```

-----

### Paso 7: Commit de los Cambios de la Fase 2

  * **Descripción Exhaustiva**: Añadir todos los cambios realizados durante esta fase al control de versiones Git.
  * **Acciones Específicas**:
      * **7.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add src/data_acquisition/binance_downloader.py
        git add src/data_acquisition/__init__.py
        git add scripts/download_data.py
        git add tests/data_acquisition/test_binance_downloader.py
        git add .env.example # Por si se añadió DATA_DIR_HOST_FOR_APP
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 2: Implementar Módulo 1 (Adquisición de Datos). Añadida clase BinanceDownloader para klines y trades agregados, con persistencia en disco/Redis y script de ejecución. Incluye tests básicos."
        ```

**Fin de la Fase de Implementación 2.**

-----


## Fase de Implementación 3: Módulo 2 - Preprocesamiento y Gestión de Datos

**Nombre Descriptivo de la Fase:** Ingeniería y Normalización de Características para el Agente RL.

Esta fase se centra en desarrollar el Módulo 2. Este módulo consumirá los datos brutos descargados en la fase anterior (principalmente Klines y, si disponibles y relevantes, Trades Agregados), realizará la ingeniería de características para calcular indicadores técnicos y otras features relevantes, normalizará estos datos y finalmente los estructurará en secuencias `(L, N_features)` listas para ser consumidas por el entorno de trading y el agente. Se prestará especial atención a la causalidad para evitar el lookahead bias.

**Consideraciones Importantes sobre las Características (Features):**

  * **Features de Libro de Órdenes:** Como se identificó en la Fase 2, la API pública de Binance no provee snapshots históricos del libro de órdenes. Por lo tanto, las 8 features dependientes del libro de órdenes detalladas en el `README.md` (`Spread L1 norm., Retorno Precio Medio L1, OBI L1`, etc.) **no se implementarán en esta fase** debido a la falta de datos de origen. Si en el futuro se dispone de esta data, se podrán añadir.
  * **Features de Cartera y Portafolio:** Las 8 features relativas a la cartera (`Estado Posición, Tamaño Posición norm., Precio Entrada norm.`, etc.) son dinámicas y dependen del estado de la simulación y las acciones del agente. Estas serán calculadas y gestionadas por el **Módulo 3 (Entorno de Trading)** y formarán parte de la observación que el entorno proporciona al agente. No serán precalculadas estáticamente por este Módulo 2.
  * **Features a Implementar por Módulo 2:** Este módulo se centrará en las **20 features** derivables directamente de los datos de mercado (Klines y Trades Agregados):
      * 5 Features de Klines OHLCV Procesados.
      * 15 Features de Indicadores Técnicos.
        El parámetro `N_features` en la configuración y en el código reflejará este número (20).

-----

### Paso 1: Creación de la Clase `DataPreprocessor` y Archivos Necesarios

  * **Descripción Exhaustiva**: Crear el archivo `src/preprocessing/feature_engineer.py` que contendrá la clase `DataPreprocessor`. Esta clase manejará la carga de datos, cálculo de features, normalización y creación de secuencias.
  * **Acciones Específicas**:
      * **1.1. Crear `src/preprocessing/__init__.py`** (si no existe):
        ```python
        # src/preprocessing/__init__.py
        from .feature_engineer import DataPreprocessor

        __all__ = ['DataPreprocessor']
        ```
      * **1.2. Crear el Esqueleto de `src/preprocessing/feature_engineer.py`**:
        ```python
        # src/preprocessing/feature_engineer.py
        import pandas as pd
        import numpy as np
        import talib # type: ignore[import-untyped]
        from pathlib import Path
        import logging
        from typing import Tuple, List, Optional, Dict, Any
        from sklearn.preprocessing import StandardScaler # Para Z-score si no es con ventana móvil

        from config_loader import load_yaml_config, get_env_variable
        # from common.utils import setup_logging # Asumir que ya se llamó globalmente

        logger = logging.getLogger(__name__)

        class DataPreprocessor:
            """
            Clase para preprocesar datos de mercado brutos (Klines) y transformarlos
            en secuencias de características listas para el agente de RL.
            """
            
            EXPECTED_N_FEATURES = 20 # Klines (5) + Indicadores Técnicos (15)

            def __init__(self):
                logger.info("Inicializando DataPreprocessor...")
                try:
                    self.module_config = load_yaml_config("module2_preprocessing")
                    self.data_acquisition_config = load_yaml_config("module1_data_acquisition") # Para saber el par, intervalo

                    self.trading_pair = self.data_acquisition_config.get("trading_pair", "BTCUSDT")
                    self.kline_interval = self.data_acquisition_config.get("kline_interval", "15m")

                    # Rutas de datos (desde .env, mapeadas a /app/data_persistent en Docker)
                    self.data_dir_host_str = get_env_variable("DATA_DIR_HOST_FOR_APP", "/app/data_persistent")
                    self.raw_data_base_path = Path(self.data_dir_host_str) / "raw" / self.trading_pair
                    self.processed_data_path = Path(self.data_dir_host_str) / "processed" / self.trading_pair
                    self.processed_data_path.mkdir(parents=True, exist_ok=True)

                    # Parámetros de preprocesamiento
                    self.sequence_L = self.module_config.get("sequence_length_L", 96)
                    self.norm_window_multiplier = self.module_config.get("normalization_window_multiplier_for_L", 2)
                    self.normalization_window = self.sequence_L * self.norm_window_multiplier
                    
                    logger.info("DataPreprocessor inicializado correctamente.")
                    logger.info(f"Sequence Length (L): {self.sequence_L}, Normalization Window: {self.normalization_window}")
                    logger.info(f"Directorio de datos brutos base: {self.raw_data_base_path}")
                    logger.info(f"Directorio de datos procesados: {self.processed_data_path}")

                except EnvironmentError as e:
                    logger.error(f"Error de variable de entorno durante la inicialización de DataPreprocessor: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error inesperado durante la inicialización de DataPreprocessor: {e}", exc_info=True)
                    raise

            def _load_raw_klines_from_disk(self, start_date_str: str, end_date_str: str) -> Optional[pd.DataFrame]:
                """
                Carga datos de klines brutos desde archivos Parquet guardados por Módulo 1.
                Concatena datos de múltiples archivos diarios dentro del rango de fechas.
                """
                klines_dir = self.raw_data_base_path / "klines" / self.kline_interval
                logger.info(f"Cargando klines desde: {klines_dir} para el rango {start_date_str} a {end_date_str}")

                start_date = pd.to_datetime(start_date_str, utc=True).date()
                end_date = pd.to_datetime(end_date_str, utc=True).date() # Inclusivo para el día final

                all_daily_klines_dfs = []
                current_date = start_date
                while current_date <= end_date:
                    date_file_str = current_date.strftime('%Y-%m-%d')
                    file_path = klines_dir / f"{self.trading_pair}_{self.kline_interval}_klines_{date_file_str}.parquet"
                    if file_path.exists():
                        try:
                            daily_df = pd.read_parquet(file_path)
                            all_daily_klines_dfs.append(daily_df)
                            logger.debug(f"Cargado {file_path} con {len(daily_df)} klines.")
                        except Exception as e:
                            logger.error(f"Error al cargar el archivo Parquet {file_path}: {e}", exc_info=True)
                    else:
                        logger.warning(f"Archivo de klines no encontrado para la fecha {date_file_str}: {file_path}")
                    current_date += pd.Timedelta(days=1)

                if not all_daily_klines_dfs:
                    logger.error(f"No se encontraron datos de klines en el disco para el rango {start_date_str} a {end_date_str} en {klines_dir}")
                    return None
                
                combined_klines_df = pd.concat(all_daily_klines_dfs, ignore_index=True)
                # Asegurar orden y eliminar duplicados (basado en tiempo de apertura de kline)
                combined_klines_df.sort_values(by='kline_open_time', inplace=True)
                combined_klines_df.drop_duplicates(subset=['kline_open_time'], keep='first', inplace=True)
                
                # Convertir columnas a tipos correctos si no lo están ya
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                for col in numeric_cols:
                    if col in combined_klines_df.columns:
                        combined_klines_df[col] = pd.to_numeric(combined_klines_df[col])
                if 'number_of_trades' in combined_klines_df.columns:
                     combined_klines_df['number_of_trades'] = combined_klines_df['number_of_trades'].astype(int)
                if 'kline_open_time_dt' not in combined_klines_df.columns and 'kline_open_time' in combined_klines_df.columns:
                     combined_klines_df['kline_open_time_dt'] = pd.to_datetime(combined_klines_df['kline_open_time'], unit='ms', utc=True)


                logger.info(f"Total de klines cargados y combinados desde disco: {len(combined_klines_df)}")
                return combined_klines_df
            
            # Otros métodos se definirán a continuación
            # def _calculate_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
            # def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
            # def _handle_nans(self, df: pd.DataFrame) -> pd.DataFrame:
            # def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
            # def _create_sequences(self, df: pd.DataFrame) -> np.ndarray:
            # def process_data(self, start_date_str: str, end_date_str: str, save_to_disk: bool = True) -> Optional[np.ndarray]:

        if __name__ == '__main__':
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger.info("Ejecutando prueba directa de DataPreprocessor...")

            # Configurar variables de entorno para prueba si no existen
            os.environ.setdefault("DATA_DIR_HOST_FOR_APP", "./temp_test_data_preprocessor")
            
            # Crear datos de klines de prueba (simulando la salida de Módulo 1)
            temp_data_dir = Path(get_env_variable("DATA_DIR_HOST_FOR_APP", "./temp_test_data_preprocessor"))
            pair = "BTCUSDT"
            interval = "15m"
            raw_klines_path = temp_data_dir / "raw" / pair / "klines" / interval
            raw_klines_path.mkdir(parents=True, exist_ok=True)

            # Generar algunos datos de klines de prueba para 2 días
            sample_klines_data_day1 = []
            start_ts_day1 = int(datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
            for i in range(96): # 96 klines de 15m en un día
                open_time = start_ts_day1 + i * (15 * 60 * 1000)
                sample_klines_data_day1.append([
                    open_time, 20000+i, 20100+i, 19900+i, 20050+i, 100+i,
                    open_time + 15*60*1000 -1, 2000000, 500, 50, 1000000, 0
                ])
            kl_cols = ['kline_open_time', 'open', 'high', 'low', 'close', 'volume', 'kline_close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            pd.DataFrame(sample_klines_data_day1, columns=kl_cols).to_parquet(raw_klines_path / f"{pair}_{interval}_klines_2023-01-01.parquet")
            
            # (Añadir otro día si se quieren probar concatenaciones más complejas)

            try:
                preprocessor = DataPreprocessor()
                logger.info("DataPreprocessor instanciado para prueba.")
                
                # Probar la carga de datos
                test_start_date = "2023-01-01"
                test_end_date = "2023-01-01" # Probar con un solo día
                klines_df = preprocessor._load_raw_klines_from_disk(test_start_date, test_end_date)
                
                if klines_df is not None:
                    logger.info(f"Klines de prueba cargados exitosamente, {len(klines_df)} filas.")
                    logger.info(f"Columnas: {klines_df.columns.tolist()}")
                    logger.info(f"Primeras filas:\n{klines_df.head()}")
                    # Aquí se llamarían a las otras funciones de preprocesamiento una vez implementadas
                    # features_df = preprocessor._calculate_ohlcv_features(klines_df.copy())
                    # features_df = preprocessor._calculate_technical_indicators(features_df)
                    # ...etc.
                else:
                    logger.error("Fallo al cargar klines de prueba.")

            except Exception as e:
                logger.error(f"Error durante la prueba de DataPreprocessor: {e}", exc_info=True)
            finally:
                # Limpiar datos de prueba (opcional)
                # import shutil
                # if temp_data_dir.exists():
                #     shutil.rmtree(temp_data_dir)
                #     logger.info(f"Directorio de datos de prueba temporal {temp_data_dir} eliminado.")
                pass
        ```

-----

### Paso 2: Implementación del Cálculo de Features de Klines OHLCV

  * **Descripción Exhaustiva**: Añadir el método `_calculate_ohlcv_features` a `DataPreprocessor` para calcular las 5 features basadas en OHLCV y volumen, como se especifica en el `README.md` (log returns, etc.).
  * **Acciones Específicas**:
      * **2.1. Definir `_calculate_ohlcv_features`**:
        ```python
        # En src/preprocessing/feature_engineer.py, dentro de la clase DataPreprocessor

        def _calculate_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Calcula características basadas en OHLCV y volumen.
            - log_ret(C/O)
            - log_ret(H/O)
            - log_ret(L/O)
            - log_ret(C/C_prev)
            - log_ret(Vol/SMA(Vol,20)) (o alguna variación si SMA(Vol,20) es cero o NaN)
            """
            logger.debug(f"Calculando features OHLCV para DataFrame con {len(df)} filas.")
            if df.empty:
                return df

            # Asegurar que las columnas 'open', 'high', 'low', 'close', 'volume' sean numéricas y existan
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Columna requerida '{col}' no encontrada en el DataFrame para calcular features OHLCV.")
                    raise ValueError(f"Columna faltante: {col}")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Columna '{col}' no es numérica, intentando convertir...")
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception as e:
                        logger.error(f"No se pudo convertir la columna '{col}' a numérica: {e}")
                        raise TypeError(f"Columna {col} debe ser numérica.") from e
            
            # Reemplazar ceros en 'open' y 'volume' con un valor muy pequeño para evitar errores de división por cero o log(0)
            # Esto es una heurística; una mejor aproximación podría ser ffill o bfill si tiene sentido.
            df['open'] = df['open'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').replace(0, 1e-9) # Evitar 0 en open
            df['volume_proc'] = df['volume'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').replace(0, 1e-9) # Usar una columna procesada para volumen

            # log_ret(C/O) = log(close/open)
            df['feat_log_ret_co'] = np.log(df['close'] / df['open'])

            # log_ret(H/O) = log(high/open)
            df['feat_log_ret_ho'] = np.log(df['high'] / df['open'])

            # log_ret(L/O) = log(low/open)
            # Aquí, si low es 0 o igual a open, podría dar -inf o 0. Low no debería ser 0 si open no lo es.
            # Low es siempre <= High. Low debería ser > 0.
            df['low_proc'] = df['low'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').replace(0, 1e-9)
            df['feat_log_ret_lo'] = np.log(df['low_proc'] / df['open'])
            
            # log_ret(C/C_prev) = log(close / close.shift(1))
            df['close_prev'] = df['close'].shift(1)
            df['feat_log_ret_ccp'] = np.log(df['close'] / df['close_prev'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').replace(0, 1e-9))

            # log_ret(Vol/SMA(Vol,20))
            sma_vol_period = self.module_config.get("sma_short_period", 20) # Usar un param de config si está
            df['sma_volume'] = talib.SMA(df['volume_proc'], timeperiod=sma_vol_period)
            # Reemplazar 0 o NaN en sma_volume para evitar división por cero o log(0)
            df['sma_volume_proc'] = df['sma_volume'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').replace(0, 1e-9)
            
            df['feat_log_ret_vol_sma'] = np.log(df['volume_proc'] / df['sma_volume_proc'])
            
            # Limpiar columnas auxiliares
            df.drop(columns=['volume_proc', 'low_proc', 'close_prev', 'sma_volume', 'sma_volume_proc'], inplace=True, errors='ignore')
            
            logger.debug("Features OHLCV calculadas.")
            return df
        ```

-----

### Paso 3: Implementación del Cálculo de Indicadores Técnicos

  * **Descripción Exhaustiva**: Añadir el método `_calculate_technical_indicators` a `DataPreprocessor`. Este método utilizará la biblioteca `TA-Lib` para calcular los 15 indicadores técnicos definidos.
  * **Acciones Específicas**:
      * **3.1. Definir `_calculate_technical_indicators`**:
        ```python
        # En src/preprocessing/feature_engineer.py, dentro de la clase DataPreprocessor

        def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Calcula los 15 indicadores técnicos especificados usando TA-Lib.
            """
            logger.debug(f"Calculando indicadores técnicos para DataFrame con {len(df)} filas.")
            if df.empty:
                return df

            # Extraer parámetros de configuración del módulo
            cfg = self.module_config

            # Asegurar que las columnas OHLCV existan y sean del tipo correcto (float64 para TA-Lib)
            ohlcv_cols = {'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64, 'volume': np.float64}
            for col, dtype in ohlcv_cols.items():
                if col not in df.columns:
                    raise ValueError(f"Columna requerida '{col}' no encontrada para indicadores técnicos.")
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    raise TypeError(f"No se pudo convertir la columna {col} a {dtype} para TA-Lib: {e}") from e

            # Indicadores:
            # 1, 2. SMA(20), SMA(50)
            df['feat_sma_short'] = talib.SMA(df['close'], timeperiod=cfg.get("sma_short_period", 20))
            df['feat_sma_long'] = talib.SMA(df['close'], timeperiod=cfg.get("sma_long_period", 50))

            # 3, 4. EMA(12), EMA(26)
            df['feat_ema_short'] = talib.EMA(df['close'], timeperiod=cfg.get("ema_short_period", 12))
            df['feat_ema_long'] = talib.EMA(df['close'], timeperiod=cfg.get("ema_long_period", 26))

            # 5. RSI(14)
            df['feat_rsi'] = talib.RSI(df['close'], timeperiod=cfg.get("rsi_period", 14))

            # 6. ATR(14)
            df['feat_atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=cfg.get("atr_period", 14))

            # 7, 8, 9. MACD(12,26,9) - Línea MACD, Línea Señal, Histograma
            macd, macdsignal, macdhist = talib.MACD(
                df['close'],
                fastperiod=cfg.get("macd_fast_period", 12),
                slowperiod=cfg.get("macd_slow_period", 26),
                signalperiod=cfg.get("macd_signal_period", 9)
            )
            df['feat_macd_line'] = macd
            df['feat_macd_signal'] = macdsignal
            df['feat_macd_hist'] = macdhist
            
            # 10, 11, 12. Bandas de Bollinger(20,2) - Dist. a Superior, Dist. a Inferior, Ancho de Bandas
            upperband, middleband, lowerband = talib.BBANDS(
                df['close'],
                timeperiod=cfg.get("bollinger_period", 20),
                nbdevup=cfg.get("bollinger_std_dev", 2),
                nbdevdn=cfg.get("bollinger_std_dev", 2),
                matype=0 # SMA
            )
            # Distancias normalizadas por ATR (o precio si ATR es 0/NaN)
            # Primero, asegurar que ATR no sea cero para evitar división por cero
            atr_for_norm = df['feat_atr'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').replace(0, df['close']) # Fallback a close
            atr_for_norm = atr_for_norm.replace(0, 1e-9) # Último recurso si close también fue 0

            df['feat_bb_dist_upper'] = (df['close'] - upperband) / atr_for_norm
            df['feat_bb_dist_lower'] = (df['close'] - lowerband) / atr_for_norm
            df['feat_bb_width'] = (upperband - lowerband) / atr_for_norm # Ancho normalizado

            # 13. CCI(20)
            df['feat_cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=cfg.get("cci_period", 20))
            
            # 14, 15. Stochastic Oscillator(14,3,3) - %K lento, %D
            slowk, slowd = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=cfg.get("stochastic_k_period", 14), # fastk_period en TA-Lib se usa para %K
                slowk_period=cfg.get("stochastic_slowing_period", 3), # slowk_period es el suavizado de %K
                slowk_matype=0, # SMA
                slowd_period=cfg.get("stochastic_d_period", 3), # slowd_period es el suavizado de %D (slowk)
                slowd_matype=0  # SMA
            )
            df['feat_stoch_k'] = slowk
            df['feat_stoch_d'] = slowd
            
            logger.debug("Indicadores técnicos calculados.")
            return df
        ```

-----

### Paso 4: Implementación del Manejo de NaNs

  * **Descripción Exhaustiva**: Añadir el método `_handle_nans` para gestionar los valores NaN generados por los cálculos de indicadores (que tienen periodos de calentamiento) y por operaciones como `.shift()`. Se puede optar por `ffill`, `bfill`, o eliminar las filas iniciales que contienen NaNs. Eliminar filas es más simple si se tiene suficientes datos.
  * **Acciones Específicas**:
      * **4.1. Definir `_handle_nans`**:
        ```python
        # En src/preprocessing/feature_engineer.py, dentro de la clase DataPreprocessor

        def _handle_nans(self, df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
            """
            Maneja los valores NaN en el DataFrame de características.

            Args:
                df (pd.DataFrame): DataFrame con características calculadas.
                strategy (str): "drop" para eliminar filas con cualquier NaN,
                                "ffill_bfill" para rellenar con forward-fill y luego backward-fill.
            Returns:
                pd.DataFrame: DataFrame sin NaNs o con NaNs rellenados.
            """
            logger.debug(f"Manejando NaNs. Estrategia: {strategy}. NaNs antes: {df.isnull().sum().sum()}")
            
            # Columnas de features (asumiendo que empiezan con 'feat_')
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            if not feature_cols:
                logger.warning("No se encontraron columnas de features (prefijo 'feat_') para manejo de NaN.")
                return df

            if strategy == "drop":
                # Eliminar filas que tengan AL MENOS UN NaN en las columnas de features
                # Esto asegura que las secuencias no tengan NaNs.
                # Se perderán los primeros N datos debido al calentamiento de los indicadores.
                # El número de filas a perder es aproximadamente el periodo del indicador más largo,
                # o la ventana de normalización más larga.
                df_cleaned = df.dropna(subset=feature_cols)
            elif strategy == "ffill_bfill":
                # Rellenar NaNs: primero hacia adelante, luego hacia atrás para los iniciales.
                # ¡PRECAUCIÓN! Esto puede introducir sesgos si se usa indiscriminadamente,
                # especialmente al inicio del dataset. 'drop' es generalmente más seguro
                # si la pérdida de datos iniciales es aceptable.
                df_cleaned = df.copy()
                df_cleaned[feature_cols] = df_cleaned[feature_cols].fillna(method='ffill').fillna(method='bfill')
            else:
                logger.warning(f"Estrategia de manejo de NaN desconocida: {strategy}. No se aplicaron cambios.")
                df_cleaned = df
            
            logger.info(f"NaNs después de manejo ({strategy}): {df_cleaned.isnull().sum().sum()} (en {len(df_cleaned)} filas)")
            if strategy == "drop" and len(df_cleaned) < len(df):
                logger.info(f"Se eliminaron {len(df) - len(df_cleaned)} filas debido a NaNs.")

            return df_cleaned
        ```

-----

### Paso 5: Implementación de la Normalización de Features

  * **Descripción Exhaustiva**: Añadir el método `_normalize_features`. Para la mayoría de las features, se aplicará un Z-score utilizando una ventana móvil causal (rollling) de `L * normalization_window_multiplier_for_L` periodos. Indicadores como RSI y Stochastic Oscillator se escalarán a rangos específicos (`[0,1]` o `[-1,1]`).
  * **Acciones Específicas**:
      * **5.1. Definir `_normalize_features`**:
        ```python
        # En src/preprocessing/feature_engineer.py, dentro de la clase DataPreprocessor

        def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Normaliza/escala las características.
            - Z-score causal con ventana móvil para la mayoría de las features.
            - Escalado específico para RSI ([0,100] -> [-1,1] o [0,1]) y Stochastic ([0,100] -> [-1,1] o [0,1]).
            """
            logger.debug(f"Normalizando features para DataFrame con {len(df)} filas. Ventana Z-score: {self.normalization_window}")
            if df.empty:
                return df

            # Columnas de features (asumiendo que empiezan con 'feat_')
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            if not feature_cols:
                logger.warning("No se encontraron columnas de features para normalizar.")
                return df

            df_normalized = df.copy()

            for col in feature_cols:
                if col == 'feat_rsi':
                    # RSI está entre 0 y 100. Escalar a [-1, 1]
                    # (RSI - 50) / 50
                    df_normalized[col] = (df_normalized[col] - 50.0) / 50.0
                elif col in ['feat_stoch_k', 'feat_stoch_d']:
                    # Stochastic K y D están entre 0 y 100. Escalar a [-1, 1]
                    # (Stoch - 50) / 50
                    df_normalized[col] = (df_normalized[col] - 50.0) / 50.0
                # Podríamos añadir escalado para CCI si su rango típico es conocido y acotado,
                # o dejarlo para Z-score. CCI puede ser [-inf, +inf] teóricamente, pero usualmente [-100,100] o [-200,200].
                # Por ahora, CCI irá a Z-score.
                elif col == 'feat_macd_hist':
                     # MACD Histograma puede tener valores muy variables. Z-score es apropiado.
                     # Pero si es muy "peaky", Z-score puede amplificar outliers.
                     # No se requiere escalado especial aquí, se usará Z-score.
                     pass # Dejar para Z-score

                # Aplicar Z-score con ventana móvil causal para las demás features
                # (excluyendo las ya escaladas especialmente)
                if col not in ['feat_rsi', 'feat_stoch_k', 'feat_stoch_d']:
                    # Z-score = (valor - media_ventana) / std_ventana
                    # .rolling() es causal por defecto (usa puntos anteriores incluyendo el actual)
                    # min_periods: para evitar NaNs al inicio si la ventana no está llena.
                    # Si la ventana es 192, necesitaremos al menos 192 puntos para un Z-score "completo".
                    # Los primeros (ventana-1) puntos tendrán NaN si no se usa min_periods o si es muy alto.
                    
                    # Usaremos min_periods = self.sequence_L para tener al menos L valores para calcular std/mean.
                    # O un valor más pequeño si L es muy grande.
                    # Para que la std no sea 0, min_periods debe ser al menos 2.
                    min_p = max(2, self.sequence_L // 2) # Heurística
                    
                    rolling_mean = df_normalized[col].rolling(window=self.normalization_window, min_periods=min_p).mean()
                    rolling_std = df_normalized[col].rolling(window=self.normalization_window, min_periods=min_p).std()

                    # Reemplazar std=0 con un valor pequeño para evitar división por cero.
                    # Esto puede ocurrir si todos los valores en la ventana son idénticos.
                    rolling_std = rolling_std.replace(0, 1e-9) 
                    
                    df_normalized[col] = (df_normalized[col] - rolling_mean) / rolling_std
            
            # Después de la normalización con ventana móvil, pueden aparecer NaNs al inicio
            # debido a min_periods y la longitud de la ventana.
            # Estos deben ser manejados. Se pueden eliminar estas filas.
            logger.debug(f"NaNs después de normalización por ventana (antes del drop): {df_normalized[feature_cols].isnull().sum().sum()}")
            df_normalized.dropna(subset=feature_cols, inplace=True) # Eliminar filas con NaNs en features
            logger.info(f"Filas después de normalización y drop de NaNs: {len(df_normalized)}. NaNs restantes: {df_normalized[feature_cols].isnull().sum().sum()}")
            
            # Verificar que el número de features sea el esperado
            final_feature_cols = [col for col in df_normalized.columns if col.startswith('feat_')]
            if len(final_feature_cols) != self.EXPECTED_N_FEATURES:
                logger.warning(f"El número de features finales ({len(final_feature_cols)}) no coincide con el esperado ({self.EXPECTED_N_FEATURES}). Features encontradas: {final_feature_cols}")

            logger.debug("Normalización de features completada.")
            return df_normalized
        ```

-----

### Paso 6: Implementación de la Creación de Secuencias

  * **Descripción Exhaustiva**: Añadir el método `_create_sequences` que tomará el DataFrame de features normalizadas y lo convertirá en un array de NumPy de secuencias superpuestas de forma `(num_sequences, L, N_features)`.
  * **Acciones Específicas**:
      * **6.1. Definir `_create_sequences`**:
        ```python
        # En src/preprocessing/feature_engineer.py, dentro de la clase DataPreprocessor

        def _create_sequences(self, df_features: pd.DataFrame) -> Optional[np.ndarray]:
            """
            Crea secuencias superpuestas (L, N_features) a partir del DataFrame de features.

            Args:
                df_features (pd.DataFrame): DataFrame con todas las features calculadas y normalizadas,
                                            y sin NaNs. Las columnas de features deben empezar con 'feat_'.
            Returns:
                Optional[np.ndarray]: Array de NumPy de forma (num_samples, sequence_L, N_features),
                                      o None si no se pueden crear secuencias.
            """
            feature_cols = sorted([col for col in df_features.columns if col.startswith('feat_')]) # Ordenar para consistencia
            if not feature_cols:
                logger.error("No se encontraron columnas de features ('feat_') para crear secuencias.")
                return None
            
            if len(feature_cols) != self.EXPECTED_N_FEATURES:
                 logger.warning(f"Creando secuencias con {len(feature_cols)} features, pero se esperaban {self.EXPECTED_N_FEATURES}.")

            data_values = df_features[feature_cols].values # Convertir a array NumPy
            num_samples_total = len(data_values)

            if num_samples_total < self.sequence_L:
                logger.error(f"No hay suficientes datos ({num_samples_total}) para crear ni una secuencia de longitud {self.sequence_L}.")
                return None

            # Usar sliding_window_view de NumPy para crear secuencias eficientemente (disponible en NumPy >= 1.20)
            # Si se usa una versión anterior de NumPy, se necesitaría una implementación manual con bucles o strides.
            # Asumimos NumPy moderno.
            # La forma de la ventana es (sequence_L, N_features)
            # El resultado de sliding_window_view será (num_sequences, sequence_L, N_features)
            try:
                # np.lib.stride_tricks.as_strided es más general pero más complejo de usar correctamente.
                # sliding_window_view es más seguro y directo para este caso de uso.
                # El array debe ser C-contiguo o F-contiguo para sliding_window_view sin copia.
                # data_values ya debería ser C-contiguo desde .values de Pandas.
                
                # window_shape es (sequence_length, num_features)
                # axis=0 para deslizar a lo largo de las filas (tiempo)
                # El resultado es una VISTA, no una copia, si es posible. Para guardar, es mejor copiar.
                
                # Para crear (num_sequences, L, N_features)
                # data_values tiene forma (num_samples_total, N_features)
                # Necesitamos crear ventanas de L timesteps.
                
                # Implementación manual si sliding_window_view no está o por claridad:
                num_sequences = num_samples_total - self.sequence_L + 1
                sequences_shape = (num_sequences, self.sequence_L, len(feature_cols))
                
                # Usar np.lib.stride_tricks.as_strided para eficiencia (evita bucles Python)
                # itemsize es el tamaño en bytes de un elemento del array
                itemsize = data_values.itemsize
                # strides: (bytes_para_siguiente_secuencia, bytes_para_siguiente_elemento_en_secuencia, bytes_para_siguiente_feature)
                # strides: (N_features * itemsize, N_features * itemsize, itemsize)
                # Esta es una forma común de crear ventanas deslizantes.
                # ¡CUIDADO! El array resultante comparte memoria. Si se modifica una secuencia, se modifica el original.
                # Para nuestro caso (solo lectura y luego guardado), puede estar bien, pero una copia es más segura.
                # sequences = np.lib.stride_tricks.as_strided(
                #     data_values,
                #     shape=sequences_shape,
                #     strides=(data_values.strides[0], data_values.strides[0], data_values.strides[1])
                # )
                # La forma anterior de strides no es correcta para (num_seq, L, N_feat). Sería:
                # strides=(data_values.strides[0], data_values.strides[0], data_values.strides[1])
                # NO, eso es para (num_seq, L) de un array 1D.
                # Para (num_samples, N_features) -> (num_sequences, L, N_features)
                # strides = (data_values.strides[0], # Salto para la siguiente secuencia (1 fila en data_values)
                #            data_values.strides[0], # Salto para el siguiente timestep dentro de una secuencia (1 fila en data_values)
                #            data_values.strides[1])  # Salto para la siguiente feature (1 columna en data_values)

                # Usando un bucle para mayor claridad y seguridad (copia los datos):
                # Esto es menos eficiente que as_strided o sliding_window_view para datasets grandes.
                # Pero para L=96, N_feat=20, y unos cientos de miles de klines, podría ser aceptable.
                # Considerar np.sliding_window_view si el rendimiento es crítico y NumPy >= 1.20.0
                
                # Implementación con bucle (más segura, hace copia):
                sequences_list = []
                for i in range(num_sequences):
                    sequences_list.append(data_values[i : i + self.sequence_L, :])
                
                sequences_np = np.array(sequences_list, dtype=np.float32) # Usar float32 para ahorrar espacio

            except Exception as e:
                logger.error(f"Error al crear secuencias: {e}", exc_info=True)
                return None
            
            logger.info(f"Secuencias creadas con forma: {sequences_np.shape}") # (num_sequences, L, N_features)
            return sequences_np
        ```

-----

### Paso 7: Implementación del Método Principal `process_data` y Guardado

  * **Descripción Exhaustiva**: Crear el método principal `process_data` que orquesta todos los pasos de preprocesamiento: carga de datos, cálculo de features, manejo de NaNs, normalización y creación de secuencias. También guardará las secuencias procesadas en disco.
  * **Acciones Específicas**:
      * **7.1. Definir `process_data`**:
        ```python
        # En src/preprocessing/feature_engineer.py, dentro de la clase DataPreprocessor

        def process_data(self, start_date_str: str, end_date_str: str, save_to_disk: bool = True) -> Optional[np.ndarray]:
            """
            Orquesta el proceso completo de preprocesamiento de datos.

            Args:
                start_date_str (str): Fecha de inicio para cargar datos brutos.
                end_date_str (str): Fecha de fin para cargar datos brutos.
                save_to_disk (bool): Si es True, guarda las secuencias procesadas en disco.

            Returns:
                Optional[np.ndarray]: Array de NumPy con las secuencias procesadas,
                                      o None si el proceso falla.
            """
            logger.info(f"Iniciando preprocesamiento de datos para {self.trading_pair} ({self.kline_interval})")
            logger.info(f"Rango de fechas para datos brutos: {start_date_str} a {end_date_str}")

            # 1. Cargar datos brutos (klines)
            klines_df = self._load_raw_klines_from_disk(start_date_str, end_date_str)
            if klines_df is None or klines_df.empty:
                logger.error("No se pudieron cargar datos brutos de klines. Abortando preprocesamiento.")
                return None
            
            # Mantener una copia de las columnas originales que podríamos necesitar (ej. kline_open_time para WFO)
            # o solo seleccionar las necesarias para features.
            # Por ahora, trabajaremos sobre el DataFrame y luego seleccionaremos 'feat_' para secuencias.
            df_processed = klines_df.copy()

            # 2. Calcular features OHLCV
            try:
                df_processed = self._calculate_ohlcv_features(df_processed)
            except Exception as e:
                logger.error(f"Error al calcular features OHLCV: {e}", exc_info=True)
                return None
            
            # 3. Calcular indicadores técnicos
            try:
                df_processed = self._calculate_technical_indicators(df_processed)
            except Exception as e:
                logger.error(f"Error al calcular indicadores técnicos: {e}", exc_info=True)
                return None

            # 4. Manejar NaNs (después de todos los cálculos de features que pueden introducirlos)
            # Es importante hacer esto ANTES de la normalización por ventana si esta usa min_periods
            # o si queremos una base limpia. O DESPUÉS si la normalización también puede generar NaNs.
            # Los indicadores TA-Lib generan NaNs al inicio. Las features OHLCV también (por .shift()).
            # La normalización por ventana también genera NaNs al inicio.
            # Estrategia: Calcular features -> Handle NaNs de features -> Normalizar -> Handle NaNs de normalización
            df_processed = self._handle_nans(df_processed, strategy="drop") # Eliminar filas con NaNs de features
            if df_processed.empty:
                logger.error("DataFrame vacío después del manejo de NaNs de features. No se puede continuar.")
                return None

            # 5. Normalizar features
            try:
                df_processed = self._normalize_features(df_processed) # Esto también hace un dropna interno
            except Exception as e:
                logger.error(f"Error al normalizar features: {e}", exc_info=True)
                return None
            
            if df_processed.empty:
                logger.error("DataFrame vacío después de la normalización de features. No se pueden crear secuencias.")
                return None

            # Columnas de features finales para las secuencias
            final_feature_cols = sorted([col for col in df_processed.columns if col.startswith('feat_')])
            if len(final_feature_cols) != self.EXPECTED_N_FEATURES:
                logger.warning(f"El número final de columnas de features ({len(final_feature_cols)}) es {len(final_feature_cols)}, se esperaban {self.EXPECTED_N_FEATURES}. Features: {final_feature_cols}")
                # Considerar abortar si esto es crítico
                if not final_feature_cols : # No hay ninguna feature
                    logger.error("No quedaron features después del preprocesamiento. Abortando.")
                    return None
            
            logger.info(f"Preprocesamiento de columnas de features completado. {len(df_processed)} filas listas para secuenciación.")
            logger.info(f"Columnas de features finales que se usarán para secuencias: {final_feature_cols}")

            # 6. Crear secuencias
            sequences_np = self._create_sequences(df_processed[final_feature_cols]) # Pasar solo las columnas de features
            
            if sequences_np is None or sequences_np.size == 0:
                logger.error("Fallo al crear secuencias o resultaron vacías.")
                return None

            # 7. Guardar secuencias procesadas
            if save_to_disk:
                # Determinar un nombre de archivo descriptivo
                # Podría incluir el rango de fechas, L, N_features.
                # Para el MVP, un nombre general para todo el dataset procesado.
                # Si se procesa por chunks para WFO, el nombre cambiaría.
                # Por ahora, un solo archivo para todos los datos procesados.
                processed_filename = f"{self.trading_pair}_{self.kline_interval}_L{self.sequence_L}_N{sequences_np.shape[2]}_processed_sequences.npz"
                # Usar .npz para guardar arrays de NumPy, o convertir a DataFrame y guardar como Parquet si se prefiere
                # y si se quiere mantener información de timestamps asociada (no directamente en este array).
                # Para el input del agente, el array NumPy es directo.
                
                save_path = self.processed_data_path / self.kline_interval
                save_path.mkdir(parents=True, exist_ok=True)
                full_file_path_npz = save_path / processed_filename
                
                try:
                    # Podríamos también guardar los timestamps correspondientes al inicio de cada secuencia.
                    # df_processed contiene 'kline_open_time' o 'kline_open_time_dt'.
                    # Las secuencias empiezan en el índice `i` y terminan en `i + L - 1`.
                    # El timestamp de la secuencia `s` correspondería al kline_open_time del primer elemento de esa secuencia.
                    # Los datos en `df_processed` ya están filtrados por NaNs de normalización,
                    # por lo que los índices pueden no ser contiguos respecto al DF original.
                    # Necesitamos los timestamps de `df_processed` ANTES de pasar solo las `final_feature_cols` a `_create_sequences`.
                    
                    # Para obtener los timestamps de inicio de cada secuencia:
                    # Si `_create_sequences` usa `data_values = df_features[final_feature_cols].values`,
                    # entonces los timestamps corresponderían a los de `df_features`.
                    # Número de secuencias = len(df_features) - self.sequence_L + 1
                    # Timestamps de inicio: df_features['kline_open_time'].iloc[0 : num_sequences].values
                    
                    # Asegurémonos de que 'kline_open_time' esté disponible en df_processed
                    if 'kline_open_time' in df_processed.columns:
                        num_total_rows_for_seq = len(df_processed)
                        num_sequences_generated = num_total_rows_for_seq - self.sequence_L + 1
                        if num_sequences_generated == sequences_np.shape[0] and num_sequences_generated > 0 :
                             sequence_start_timestamps_ms = df_processed['kline_open_time'].iloc[0 : num_sequences_generated].values.astype(np.int64)
                             np.savez_compressed(full_file_path_npz, sequences=sequences_np, timestamps=sequence_start_timestamps_ms)
                             logger.info(f"Secuencias procesadas y timestamps guardados en: {full_file_path_npz}")
                        else:
                            logger.warning("Discrepancia en número de secuencias y timestamps, guardando solo secuencias.")
                            np.savez_compressed(full_file_path_npz, sequences=sequences_np)
                            logger.info(f"Secuencias procesadas guardadas en: {full_file_path_npz}")
                    else:
                        np.savez_compressed(full_file_path_npz, sequences=sequences_np)
                        logger.info(f"Secuencias procesadas guardadas en: {full_file_path_npz} (sin timestamps asociados en el archivo).")

                except Exception as e:
                    logger.error(f"Error al guardar las secuencias procesadas en {full_file_path_npz}: {e}", exc_info=True)
            
            logger.info("Preprocesamiento de datos completado exitosamente.")
            return sequences_np
        ```

-----

### Paso 8: Script de Orquestación `scripts/preprocess_data.py`

  * **Descripción Exhaustiva**: Crear un script que utilice la clase `DataPreprocessor` para ejecutar el proceso completo de preprocesamiento para el rango de datos configurado.
  * **Acciones Específicas**:
      * **8.1. Crear `scripts/preprocess_data.py`**:
        ```python
        # scripts/preprocess_data.py
        import sys
        import logging
        from pathlib import Path

        # Añadir src al PYTHONPATH
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from common.utils import setup_logging
        from preprocessing.feature_engineer import DataPreprocessor
        from config_loader import load_yaml_config # Para obtener rango de fechas de descarga

        try:
            setup_logging()
        except Exception as e:
            logging.basicConfig(level=logging.ERROR)
            logging.critical(f"Fallo CRÍTICO al configurar logging: {e}", exc_info=True)

        logger = logging.getLogger(__name__)

        def main():
            logger.info("===========================================================")
            logger.info("Iniciando script de preprocesamiento de datos...")
            logger.info("===========================================================")

            try:
                # Cargar configuración de data_acquisition para obtener el rango de fechas
                # de los datos brutos que se deben procesar.
                # Asumimos que procesaremos todo el rango descargado.
                data_acq_config = load_yaml_config("module1_data_acquisition")
                start_date = data_acq_config.get("data_download_start_date")
                end_date = data_acq_config.get("data_download_end_date")

                if not start_date or not end_date:
                    logger.error("Fechas de inicio/fin no encontradas en la configuración de module1_data_acquisition. Abortando.")
                    return

                preprocessor = DataPreprocessor()
                processed_sequences = preprocessor.process_data(
                    start_date_str=start_date,
                    end_date_str=end_date,
                    save_to_disk=True
                )

                if processed_sequences is not None:
                    logger.info(f"Preprocesamiento completado. Forma de las secuencias generadas: {processed_sequences.shape}")
                else:
                    logger.error("El preprocesamiento de datos falló o no generó secuencias.")
                
                logger.info("Script de preprocesamiento de datos finalizado.")

            except EnvironmentError as e:
                logger.error(f"Error de configuración de entorno en el script principal de preprocesamiento: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error inesperado en el script principal de preprocesamiento: {e}", exc_info=True)
            finally:
                logger.info("===========================================================")
                logger.info("Finalización del script de preprocesamiento de datos.")
                logger.info("===========================================================")

        if __name__ == "__main__":
            main()
        ```
      * **8.2. Ejecutar el Script de Preprocesamiento (Prueba)**:
          * Asegurarse que existan datos brutos de Klines en `data_host/raw/...` (generados por Fase 2, o los datos de prueba de `DataPreprocessor.__main__`).
          * Ejecutar dentro de Docker:
            ```bash
            # docker-compose up -d --build workhorse_app redis # Si no está corriendo
            docker-compose exec workhorse_app python scripts/preprocess_data.py
            ```
          * Revisar logs y verificar la creación del archivo `.npz` en `data_host/processed/BTCUSDT/15m/`.

-----

### Paso 9: Pruebas Unitarias para `DataPreprocessor`

  * **Descripción Exhaustiva**: Crear pruebas unitarias para los métodos clave de `DataPreprocessor`, especialmente para el cálculo de features y la creación de secuencias. Usar DataFrames de prueba pequeños y mockear dependencias si es necesario.
  * **Acciones Específicas**:
      * **9.1. Crear `tests/preprocessing/test_feature_engineer.py`**:
        ```python
        # tests/preprocessing/test_feature_engineer.py
        import pytest
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from unittest import mock
        import os

        from src.preprocessing.feature_engineer import DataPreprocessor
        # from src.config_loader import get_env_variable # Para mockear

        @pytest.fixture
        def sample_raw_klines_df() -> pd.DataFrame:
            """Genera un DataFrame de klines de prueba."""
            # Necesita suficientes datos para que los indicadores y la normalización funcionen.
            # Ej: 200 periodos para una ventana de normalización de 192.
            num_rows = 300 
            start_ts = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
            interval_ms = 15 * 60 * 1000
            data = {
                'kline_open_time': [start_ts + i * interval_ms for i in range(num_rows)],
                'open': np.random.uniform(19000, 21000, num_rows),
                'high': np.random.uniform(21000, 22000, num_rows),
                'low': np.random.uniform(18000, 19000, num_rows),
                'close': np.random.uniform(19000, 21000, num_rows),
                'volume': np.random.uniform(50, 200, num_rows),
                'kline_close_time': [(start_ts + i * interval_ms) + interval_ms - 1 for i in range(num_rows)],
                'quote_asset_volume': np.random.uniform(1e6, 4e6, num_rows),
                'number_of_trades': np.random.randint(100, 1000, num_rows),
                'taker_buy_base_asset_volume': np.random.uniform(25, 100, num_rows),
                'taker_buy_quote_asset_volume': np.random.uniform(0.5e6, 2e6, num_rows)
            }
            df = pd.DataFrame(data)
            # Asegurar que high >= open, high >= close, low <= open, low <= close
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            df['kline_open_time_dt'] = pd.to_datetime(df['kline_open_time'], unit='ms', utc=True)
            return df

        @pytest.fixture
        def mock_config_for_preprocessor(monkeypatch, tmp_path: Path):
            """Mockea la carga de configuración para DataPreprocessor."""
            # Mock get_env_variable
            processed_data_dir = tmp_path / "test_processed_data"
            raw_data_dir = tmp_path / "test_raw_data" # Para _load_raw_klines_from_disk
            
            def mock_get_env(var_name, default_value=None, required=True):
                if var_name == "DATA_DIR_HOST_FOR_APP": return str(tmp_path) # Base para raw y processed
                # Añadir otros si son necesarios para la inicialización
                if default_value is not None: return default_value
                return None # Simular no encontrado si no es uno de los anteriores
            
            monkeypatch.setattr("src.preprocessing.feature_engineer.get_env_variable", mock_get_env)
            monkeypatch.setattr("src.config_loader.get_env_variable", mock_get_env) # Si config_loader es usado por otro módulo importado

            # Mock load_yaml_config
            def mock_load_yaml(module_name, file_name="params.yaml"):
                if module_name == "module2_preprocessing":
                    return {
                        "sequence_length_L": 96,
                        "normalization_window_multiplier_for_L": 2,
                        "sma_short_period": 20, "sma_long_period": 50,
                        "ema_short_period": 12, "ema_long_period": 26,
                        "rsi_period": 14, "atr_period": 14,
                        "macd_fast_period": 12, "macd_slow_period": 26, "macd_signal_period": 9,
                        "bollinger_period": 20, "bollinger_std_dev": 2,
                        "cci_period": 20,
                        "stochastic_k_period": 14, "stochastic_slowing_period": 3, "stochastic_d_period": 3
                    }
                if module_name == "module1_data_acquisition": # Necesario para init de Preprocessor
                    return {"trading_pair": "BTCUSDT", "kline_interval": "15m"}
                return {}
            
            monkeypatch.setattr("src.preprocessing.feature_engineer.load_yaml_config", mock_load_yaml)
            
            # Crear directorios que el preprocesador esperaría (basado en el DATA_DIR_HOST_FOR_APP mockeado)
            (tmp_path / "processed" / "BTCUSDT" / "15m").mkdir(parents=True, exist_ok=True)
            (tmp_path / "raw" / "BTCUSDT" / "klines" / "15m").mkdir(parents=True, exist_ok=True)


        def test_preprocessor_initialization(mock_config_for_preprocessor):
            """Testea la inicialización de DataPreprocessor."""
            preprocessor = DataPreprocessor()
            assert preprocessor.sequence_L == 96
            assert preprocessor.EXPECTED_N_FEATURES == 20
            assert "BTCUSDT" in str(preprocessor.processed_data_path)

        def test_calculate_ohlcv_features(mock_config_for_preprocessor, sample_raw_klines_df):
            preprocessor = DataPreprocessor()
            features_df = preprocessor._calculate_ohlcv_features(sample_raw_klines_df.copy())
            assert 'feat_log_ret_co' in features_df.columns
            assert not features_df['feat_log_ret_co'].isnull().all() # No todos deben ser NaN
            # Verificar que las 5 features OHLCV estén presentes
            ohlcv_feat_names = ['feat_log_ret_co', 'feat_log_ret_ho', 'feat_log_ret_lo', 'feat_log_ret_ccp', 'feat_log_ret_vol_sma']
            for fname in ohlcv_feat_names:
                assert fname in features_df.columns

        def test_calculate_technical_indicators(mock_config_for_preprocessor, sample_raw_klines_df):
            preprocessor = DataPreprocessor()
            # Se necesitan features OHLCV para algunos indicadores (aunque TA-Lib usa OHLC directamente)
            # Pasamos el DF crudo como lo haría el flujo normal
            features_df = preprocessor._calculate_technical_indicators(sample_raw_klines_df.copy())
            # Verificar algunas features clave
            assert 'feat_sma_short' in features_df.columns
            assert 'feat_rsi' in features_df.columns
            assert 'feat_macd_hist' in features_df.columns
            assert 'feat_stoch_k' in features_df.columns
            assert not features_df['feat_rsi'].isnull().all()
            
            # Contar features de indicadores (deberían ser 15)
            indicator_feature_count = 0
            expected_indicator_features = [ # Basado en la implementación
                'feat_sma_short', 'feat_sma_long', 'feat_ema_short', 'feat_ema_long',
                'feat_rsi', 'feat_atr', 'feat_macd_line', 'feat_macd_signal', 'feat_macd_hist',
                'feat_bb_dist_upper', 'feat_bb_dist_lower', 'feat_bb_width',
                'feat_cci', 'feat_stoch_k', 'feat_stoch_d'
            ]
            for col in features_df.columns:
                if col in expected_indicator_features:
                    indicator_feature_count +=1
            assert indicator_feature_count == 15


        def test_handle_nans_drop(mock_config_for_preprocessor, sample_raw_klines_df):
            preprocessor = DataPreprocessor()
            # Simular que ya tiene algunas features con NaNs al inicio
            df_with_nans = sample_raw_klines_df.copy()
            df_with_nans['feat_dummy'] = np.nan # Añadir una columna de features con todo NaN
            df_with_nans.loc[0:10, 'feat_dummy'] = 1 # Quitar NaN de algunos para que no se borre todo
            df_with_nans.loc[0:50, 'feat_another'] = np.nan # Esta sí causará drop de filas iniciales
            
            # Crear algunas NaNs a propósito para probar
            df_with_nans.loc[0, 'close'] = np.nan 
            df_with_nans = preprocessor._calculate_ohlcv_features(df_with_nans) # Esto generará NaNs
            
            cleaned_df = preprocessor._handle_nans(df_with_nans.copy(), strategy="drop")
            feature_cols = [col for col in cleaned_df.columns if col.startswith('feat_')]
            if feature_cols: # Solo si hay features después del cálculo
                 assert cleaned_df[feature_cols].isnull().sum().sum() == 0
            assert len(cleaned_df) < len(df_with_nans) # Se deben haber eliminado filas

        def test_normalize_features(mock_config_for_preprocessor, sample_raw_klines_df):
            preprocessor = DataPreprocessor()
            # Pipeline parcial para obtener features antes de normalizar
            df = sample_raw_klines_df.copy()
            df = preprocessor._calculate_ohlcv_features(df)
            df = preprocessor._calculate_technical_indicators(df)
            df = preprocessor._handle_nans(df, strategy="drop") # Quitar NaNs de indicadores
            
            if df.empty:
                pytest.skip("DataFrame vacío después del manejo de NaNs, no se puede probar normalización.")

            normalized_df = preprocessor._normalize_features(df.copy())
            feature_cols = [col for col in normalized_df.columns if col.startswith('feat_')]

            if not feature_cols or normalized_df.empty:
                 pytest.skip("No features o DataFrame vacío después de normalización, no se puede probar más.")

            # Verificar que RSI y Stoch estén en rango aprox. [-1, 1] (pueden ser ligeramente fuera por float precision)
            assert normalized_df['feat_rsi'].min() >= -1.1 and normalized_df['feat_rsi'].max() <= 1.1
            assert normalized_df['feat_stoch_k'].min() >= -1.1 and normalized_df['feat_stoch_k'].max() <= 1.1
            
            # Otras features (Z-score) deben tener media cercana a 0 y std cercana a 1 a lo largo del tiempo
            # (esto es difícil de testear con precisión sin mirar las ventanas móviles completas)
            # Por ahora, solo verificamos que no haya NaNs
            assert normalized_df[feature_cols].isnull().sum().sum() == 0


        def test_create_sequences(mock_config_for_preprocessor, sample_raw_klines_df):
            preprocessor = DataPreprocessor()
            preprocessor.sequence_L = 10 # Usar una L más pequeña para la prueba
            
            # Crear un DataFrame de features ya procesadas y normalizadas (simulado)
            num_rows_test_seq = 50
            num_features_test_seq = preprocessor.EXPECTED_N_FEATURES # 20
            
            # Generar datos dummy para las features ya normalizadas
            feature_data = np.random.rand(num_rows_test_seq, num_features_test_seq)
            feature_names = [f"feat_{i}" for i in range(num_features_test_seq)]
            processed_df = pd.DataFrame(feature_data, columns=feature_names)

            sequences_np = preprocessor._create_sequences(processed_df)
            
            assert sequences_np is not None
            expected_num_sequences = num_rows_test_seq - preprocessor.sequence_L + 1
            assert sequences_np.shape == (expected_num_sequences, preprocessor.sequence_L, num_features_test_seq)

        # Test completo del método process_data (más como un test de integración)
        @mock.patch("src.preprocessing.feature_engineer.DataPreprocessor._load_raw_klines_from_disk")
        def test_process_data_e2e(self, mock_load_klines, mock_config_for_preprocessor, sample_raw_klines_df, tmp_path):
            mock_load_klines.return_value = sample_raw_klines_df # Mockear la carga desde disco
            
            preprocessor = DataPreprocessor()
            # Reducir L para que el test sea más rápido y requiera menos datos iniciales
            preprocessor.sequence_L = 10 
            preprocessor.normalization_window = preprocessor.sequence_L * preprocessor.norm_window_multiplier

            # Definir dónde se guardarán los datos procesados (basado en el mock de DATA_DIR_HOST_FOR_APP)
            expected_save_dir = tmp_path / "processed" / "BTCUSDT" / "15m"
            
            sequences = preprocessor.process_data(
                start_date_str="2023-01-01", 
                end_date_str="2023-01-10", # Rango de prueba
                save_to_disk=True
            )

            assert sequences is not None
            assert sequences.ndim == 3
            assert sequences.shape[1] == preprocessor.sequence_L
            assert sequences.shape[2] == preprocessor.EXPECTED_N_FEATURES
            
            # Verificar que el archivo .npz se haya creado
            # El nombre del archivo depende de L y N_features.
            # preprocessor.trading_pair, preprocessor.kline_interval
            filename = f"{preprocessor.trading_pair}_{preprocessor.kline_interval}_L{preprocessor.sequence_L}_N{preprocessor.EXPECTED_N_FEATURES}_processed_sequences.npz"
            expected_file = expected_save_dir / filename
            assert expected_file.exists()
            
            # Cargar y verificar el contenido
            loaded_data = np.load(expected_file)
            assert 'sequences' in loaded_data
            assert 'timestamps' in loaded_data # Asumiendo que se guardan timestamps
            assert loaded_data['sequences'].shape == sequences.shape
        ```
      * **9.2. Ejecutar las Pruebas**:
        ```bash
        python -m pytest tests/preprocessing/test_feature_engineer.py --cov=src/preprocessing
        ```

-----

### Paso 10: Commit de los Cambios de la Fase 3

  * **Descripción Exhaustiva**: Añadir todos los cambios realizados durante esta fase al control de versiones Git.
  * **Acciones Específicas**:
      * **10.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add src/preprocessing/feature_engineer.py
        git add src/preprocessing/__init__.py
        git add scripts/preprocess_data.py
        git add tests/preprocessing/test_feature_engineer.py
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 3: Implementar Módulo 2 (Preprocesamiento de Datos). Clase DataPreprocessor para ingeniería de features (OHLCV, Técnicos), normalización y creación de secuencias. Incluye script y tests."
        ```

**Fin de la Fase de Implementación 3.**

-----


## Fase de Implementación 4: Módulo 3 - Entorno de Trading Simulado (Gymnasium)

**Nombre Descriptivo de la Fase:** Construcción del Entorno de Simulación de Trading para el Agente.

Esta fase se dedica a implementar el Módulo 3, que es el entorno de trading simulado. Este entorno, compatible con la API de `gymnasium.Env`, permitirá al agente de Reinforcement Learning interactuar con el mercado (datos históricos procesados), tomar decisiones de trading, y recibir observaciones y recompensas. Gestionará la cartera, simulará la ejecución de órdenes con costes (comisiones, slippage) y determinará las condiciones de finalización del episodio.

**Decisión de Diseño Clave para la Observación del Agente:**

  * El Módulo 2 (Preprocesador) genera secuencias de características de mercado de forma `(num_total_steps, L, N_market_features)`, donde `N_market_features = 20` (5 OHLCV + 15 Técnicos).
  * El `README.md` (Módulo 4) indica que el Transformer del agente espera una entrada de `(L, N_features=36)`.
  * Las 8 features de "Cartera y Portafolio" son gestionadas por este Módulo 3 (Entorno).
  * Las 8 features de "Libro de Órdenes" fueron omitidas por falta de datos históricos.
  * Por lo tanto, para aproximarnos a la especificación del `README.md` con los datos disponibles, el entorno construirá una observación para el agente de forma `(L, N_effective_features)`, donde `N_effective_features = 20 (market) + 8 (portfolio) = 28`.
  * Las 8 features de portafolio (que son un snapshot en el tiempo `t`) se replicarán a lo largo de la dimensión de secuencia `L` y se concatenarán con las `N_market_features` para cada uno de los `L` timesteps. El agente recibirá entonces una observación NumPy de `(L, 28)`.

-----

### Paso 1: Creación de la Clase `TradingEnv` y Archivos Necesarios

  * **Descripción Exhaustiva**: Crear el archivo `src/environment/trading_env.py` que contendrá la clase `TradingEnv`, heredando de `gymnasium.Env`.
  * **Acciones Específicas**:
      * **1.1. Crear `src/environment/__init__.py`** (si no existe):
        ```python
        # src/environment/__init__.py
        from .trading_env import TradingEnv

        __all__ = ['TradingEnv']
        ```
      * **1.2. Crear el Esqueleto de `src/environment/trading_env.py`**:
        ```python
        # src/environment/trading_env.py
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        import pandas as pd
        from pathlib import Path
        import logging
        from typing import Tuple, Dict, Any, Optional, Union, List

        from config_loader import load_yaml_config, get_env_variable
        # from common.utils import setup_logging # Asumir que ya se llamó globalmente

        logger = logging.getLogger(__name__)

        # Constantes para el estado de la posición
        POSITION_NEUTRAL = 0
        POSITION_LONG = 1
        POSITION_SHORT = -1

        # Número de features de portafolio que se añadirán a la observación
        NUM_PORTFOLIO_FEATURES = 8
        # Número de features de mercado (provenientes de M2)
        NUM_MARKET_FEATURES = 20 # Actualizado según M2
        # Número total de features por timestep en la secuencia para el agente
        # (Market Features + Portfolio Features). Las de Order Book (8) se omiten.
        TOTAL_FEATURES_PER_TIMESTEP_IN_SEQ = NUM_MARKET_FEATURES + NUM_PORTFOLIO_FEATURES # 20 + 8 = 28


        class TradingEnv(gym.Env):
            """
            Entorno de Trading de Futuros BTCUSDT simulado, compatible con Gymnasium.
            """
            metadata = {'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 4}

            def __init__(self, data_npz_path: Union[str, Path], config_override: Optional[Dict[str, Any]] = None):
                """
                Inicializa el Entorno de Trading.

                Args:
                    data_npz_path (Union[str, Path]): Ruta al archivo .npz que contiene las secuencias
                                                      preprocesadas ('sequences') y opcionalmente 'timestamps'.
                                                      Este archivo es la salida del Módulo 2.
                    config_override (Optional[Dict[str, Any]]): Permite sobreescribir parámetros de configuración
                                                                del entorno para pruebas o experimentación.
                """
                super().__init__()
                logger.info(f"Inicializando TradingEnv con datos de: {data_npz_path}")

                try:
                    self.env_config = load_yaml_config("module3_environment")
                    if config_override:
                        self.env_config.update(config_override)
                    
                    self.preproc_config = load_yaml_config("module2_preprocessing") # Para L
                    self.sequence_L = self.preproc_config.get("sequence_length_L", 96)

                    # Cargar datos preprocesados (secuencias y timestamps)
                    self._load_market_data(data_npz_path) # Define self.market_sequences y self.market_timestamps

                    # Espacios de acción y observación
                    # Acción: señal continua [-1, 1]
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                    
                    # Observación: Secuencia (L, N_total_features) donde N_total_features = N_market + N_portfolio
                    # N_market_features se infiere de self.market_sequences.shape[2]
                    if self.market_sequences.shape[2] != NUM_MARKET_FEATURES:
                        logger.warning(f"El número de features de mercado cargadas ({self.market_sequences.shape[2]})"
                                       f" no coincide con el esperado ({NUM_MARKET_FEATURES}). Usando el cargado.")
                    
                    self.effective_num_market_features = self.market_sequences.shape[2]
                    self.total_features_in_obs = self.effective_num_market_features + NUM_PORTFOLIO_FEATURES

                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, # Normalización Z-score puede ir más allá de ciertos rangos
                        shape=(self.sequence_L, self.total_features_in_obs), # (L, 20_market + 8_portfolio)
                        dtype=np.float32
                    )
                    
                    # Parámetros del entorno
                    self.initial_equity = float(self.env_config.get("initial_equity", 10000.0))
                    self.leverage = float(self.env_config.get("leverage", 10.0))
                    self.position_size_pct_equity = float(self.env_config.get("position_size_pct_equity", 0.05))
                    self.taker_fee_rate = float(self.env_config.get("taker_fee_rate", 0.0004))
                    self.slippage_atr_multiplier = float(self.env_config.get("slippage_atr_multiplier", 0.1))
                    self.action_threshold = float(self.env_config.get("action_threshold", 0.15))
                    self.equity_drawdown_threshold = float(self.env_config.get("equity_drawdown_threshold_episode_end", -0.20))
                    self.liquidation_safety_factor = float(self.env_config.get("liquidation_safety_factor", 0.8))
                    
                    # Estado interno del entorno (se reinicia en reset())
                    self.current_step_index = 0 # Índice para las secuencias de mercado
                    self.equity = self.initial_equity
                    self.balance = self.initial_equity # Similar a equity si no hay P&L flotante complejo
                    self.position = POSITION_NEUTRAL # -1 (corto), 0 (neutral), 1 (largo)
                    self.position_size_contracts = 0.0 # Tamaño de la posición en unidades del activo base (ej. BTC)
                    self.entry_price = 0.0
                    self.unrealized_pnl = 0.0
                    self.realized_pnl_episode = 0.0 # P&L realizado durante el episodio actual
                    self.margin_used = 0.0 # Puede ser calculado o simplificado
                    self.steps_since_position_opened = 0
                    self.total_trades_episode = 0
                    
                    # Para modo de evaluación/backtesting (no aleatorio)
                    self.eval_mode = False 
                    self.eval_start_index = 0

                    logger.info("TradingEnv inicializado.")
                    logger.info(f"Observation space: {self.observation_space}")
                    logger.info(f"Action space: {self.action_space}")

                except FileNotFoundError as e:
                    logger.error(f"Error: Archivo de datos no encontrado en {data_npz_path}. {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error inesperado durante la inicialización de TradingEnv: {e}", exc_info=True)
                    raise

            def _load_market_data(self, data_npz_path: Union[str, Path]):
                """Carga las secuencias de mercado y timestamps desde un archivo .npz."""
                logger.info(f"Cargando datos de mercado desde {data_npz_path}...")
                try:
                    data = np.load(data_npz_path)
                    self.market_sequences = data['sequences'].astype(np.float32) # (num_total_samples, L, N_market_features)
                    if 'timestamps' in data:
                        self.market_timestamps = data['timestamps'] # (num_total_samples,) Timestamps de inicio de cada secuencia
                    else:
                        logger.warning("No se encontraron 'timestamps' en el archivo .npz. Se generarán índices secuenciales.")
                        self.market_timestamps = np.arange(len(self.market_sequences))
                    
                    if self.market_sequences.shape[1] != self.sequence_L:
                        raise ValueError(f"La longitud de secuencia L en los datos cargados ({self.market_sequences.shape[1]}) "
                                         f"no coincide con la configuración ({self.sequence_L}).")
                    
                    self.total_steps_in_dataset = len(self.market_sequences)
                    logger.info(f"Datos de mercado cargados. Forma de las secuencias: {self.market_sequences.shape}. Total de pasos disponibles: {self.total_steps_in_dataset}")
                    if self.total_steps_in_dataset <= self.sequence_L : # O algún umbral mínimo
                        raise ValueError(f"No hay suficientes datos en el dataset ({self.total_steps_in_dataset} pasos) para operar el entorno.")

                except Exception as e:
                    logger.error(f"Error al cargar o validar datos de mercado desde {data_npz_path}: {e}", exc_info=True)
                    raise
            
            # --- Métodos de Gymnasium (reset, step, render, close) ---
            # def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
            # def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
            # def render(self) -> Optional[Union[np.ndarray, str]]:
            # def close(self):

            # --- Métodos auxiliares de la lógica del entorno ---
            # def _get_current_market_data_sequence(self) -> np.ndarray:
            # def _get_portfolio_state_features(self) -> np.ndarray:
            # def _construct_observation(self) -> np.ndarray:
            # def _interpret_action(self, action_signal: float) -> int: # Devuelve tipo de orden
            # def _execute_trade(self, order_type: int, current_price: float, current_atr: float):
            # def _calculate_slippage(self, current_atr: float) -> float:
            # def _calculate_commission(self, trade_value: float) -> float:
            # def _update_pnl_and_equity(self, current_price: float):
            # def _check_liquidation(self, current_price: float) -> bool:
            # def _calculate_reward(self) -> float:

        if __name__ == '__main__':
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger.info("Ejecutando prueba directa de TradingEnv...")

            # Crear un archivo .npz de prueba
            test_data_dir = Path("./temp_test_env_data")
            test_data_dir.mkdir(parents=True, exist_ok=True)
            
            L_test = 96
            N_market_feat_test = NUM_MARKET_FEATURES # 20
            num_samples_test = 500
            
            dummy_sequences = np.random.rand(num_samples_test, L_test, N_market_feat_test).astype(np.float32)
            dummy_timestamps = np.arange(num_samples_test).astype(np.int64)
            test_npz_path = test_data_dir / f"test_processed_sequences_L{L_test}_N{N_market_feat_test}.npz"
            np.savez_compressed(test_npz_path, sequences=dummy_sequences, timestamps=dummy_timestamps)

            # Mockear config_loader para que no dependa de archivos YAML externos en esta prueba directa
            # Esto es opcional si los archivos YAML ya están configurados.
            with mock.patch('src.environment.trading_env.load_yaml_config') as mock_load_cfg:
                def get_mock_config(module_name, file_name="params.yaml"):
                    if module_name == "module3_environment":
                        return { # Valores de ejemplo para la prueba
                            "initial_equity": 10000.0, "leverage": 10.0, 
                            "position_size_pct_equity": 0.05, "taker_fee_rate": 0.0004,
                            "slippage_atr_multiplier": 0.1, "action_threshold": 0.15,
                            "equity_drawdown_threshold_episode_end": -0.20,
                            "liquidation_safety_factor": 0.8
                        }
                    if module_name == "module2_preprocessing":
                        return {"sequence_length_L": L_test}
                    return {}
                mock_load_cfg.side_effect = get_mock_config
                
                # Mockear get_env_variable
                with mock.patch('src.environment.trading_env.get_env_variable') as mock_get_env:
                    mock_get_env.return_value = "dummy_value_not_used_in_this_test_path" 

                    try:
                        env = TradingEnv(data_npz_path=test_npz_path)
                        logger.info(f"TradingEnv instanciado para prueba. Observation space: {env.observation_space.shape}")
                        
                        # Probar reset
                        # obs, info = env.reset()
                        # logger.info(f"Reset completado. Observación inicial shape: {obs.shape}")
                        # logger.info(f"Info inicial: {info}")

                        # Probar un step (requiere que reset y step estén implementados)
                        # action = env.action_space.sample()
                        # obs, reward, terminated, truncated, info = env.step(action)
                        # logger.info(f"Step completado. Nueva obs shape: {obs.shape}, Recompensa: {reward}")

                    except Exception as e:
                        logger.error(f"Error durante la prueba de TradingEnv: {e}", exc_info=True)
                    finally:
                        # Limpiar archivo de prueba
                        if test_npz_path.exists():
                            test_npz_path.unlink()
                        if test_data_dir.exists():
                            try:
                                test_data_dir.rmdir() # Solo si está vacío
                            except OSError:
                                logger.warning(f"No se pudo eliminar el directorio temporal {test_data_dir}")
        ```

-----

### Paso 2: Implementación del Método `reset`

  * **Descripción Exhaustiva**: Implementar el método `reset` que reinicia el entorno a un estado inicial. Esto incluye seleccionar un punto de partida para los datos de mercado (aleatorio para entrenamiento, fijo para evaluación), reiniciar la cartera y devolver la primera observación.
  * **Acciones Específicas**:
      * **2.1. Definir `reset` en `TradingEnv`**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
            """
            Reinicia el entorno a un estado inicial.

            Args:
                seed (Optional[int]): Semilla para la generación de números aleatorios (opcional).
                options (Optional[dict]): Opciones adicionales, como 'start_index' para evaluación.

            Returns:
                Tuple[np.ndarray, dict]: La observación inicial y un diccionario de información.
            """
            super().reset(seed=seed) # Importante para gestionar la semilla del generador de números aleatorios de Gymnasium
            
            # Reiniciar estado de la cartera
            self.equity = self.initial_equity
            self.balance = self.initial_equity
            self.position = POSITION_NEUTRAL
            self.position_size_contracts = 0.0
            self.entry_price = 0.0
            self.unrealized_pnl = 0.0
            self.realized_pnl_episode = 0.0
            self.margin_used = 0.0 
            self.steps_since_position_opened = 0
            self.total_trades_episode = 0
            self.episode_start_equity = self.initial_equity # Para calcular drawdown del episodio

            # Determinar el punto de inicio para los datos de mercado
            # El dataset tiene `self.total_steps_in_dataset` secuencias disponibles.
            # Cada secuencia ya tiene longitud L. El índice apunta al inicio de una secuencia.
            if self.eval_mode and options and 'start_index' in options:
                 self.current_step_index = options.get('start_index', 0)
                 # Asegurar que el índice de evaluación sea válido
                 self.current_step_index = max(0, min(self.current_step_index, self.total_steps_in_dataset - 1))
                 logger.info(f"Modo Evaluación: Entorno reseteado. Iniciando en step_index: {self.current_step_index}")
            elif self.env_config.get("max_episode_steps_equals_dataset_length", True): # Si el episodio recorre todo el dataset (no random start)
                # En este caso, el "punto aleatorio" se maneja a nivel de WFO, y cada episodio aquí es un "pass"
                # o si se quiere un inicio aleatorio incluso en este modo, se puede añadir lógica
                self.current_step_index = 0 # Siempre empieza desde el inicio del chunk de datos actual
                logger.info(f"Entorno reseteado (modo dataset completo). Iniciando en step_index: 0")
            else: # Inicio aleatorio dentro del dataset disponible (para entrenamiento típico de RL)
                # Dejar suficientes datos para que un episodio pueda correr una cantidad razonable de pasos
                # o hasta el final del dataset.
                # El índice puede ir de 0 a self.total_steps_in_dataset - 1
                max_start_index = self.total_steps_in_dataset - 1 # Puede empezar en el último step posible
                if max_start_index < 0: max_start_index = 0 # Si el dataset es muy corto
                
                # Usar el generador de números aleatorios de Gymnasium para reproducibilidad
                self.current_step_index = self.np_random.integers(0, max_start_index + 1)
                logger.info(f"Entorno reseteado. Iniciando aleatoriamente en step_index: {self.current_step_index}")

            # Construir la observación inicial
            observation = self._construct_observation()
            info = self._get_info()
            
            return observation, info

        def _get_current_market_data_sequence(self) -> np.ndarray:
            """
            Obtiene la secuencia de datos de mercado (L, N_market_features) para el paso actual.
            """
            if not (0 <= self.current_step_index < self.total_steps_in_dataset):
                logger.error(f"Índice de paso actual ({self.current_step_index}) fuera de rango ({self.total_steps_in_dataset} pasos disponibles).")
                # Esto no debería ocurrir si la lógica de reset/step es correcta.
                # Devolver una secuencia de ceros o lanzar error.
                # Por ahora, lanzamos error para detectar problemas temprano.
                raise IndexError("current_step_index fuera de los límites del dataset de mercado.")
            
            # market_sequences tiene forma (num_total_samples, L, N_market_features)
            # El current_step_index apunta a cuál de las 'num_total_samples' usar.
            # Cada una de estas ya es una secuencia de longitud L.
            return self.market_sequences[self.current_step_index]

        def _get_portfolio_state_features(self) -> np.ndarray:
            """
            Construye el vector de características del estado de la cartera (8 features).
            Normaliza algunas de estas features.
            """
            # 1. Estado Posición (-1,0,1)
            f_pos_state = float(self.position)

            # 2. Tamaño Posición norm. (por equity inicial, o equity actual?)
            #    Normalizar por equity inicial podría ser más estable para la red.
            #    Si position_size_contracts es BTC, el valor es BTC * precio_actual.
            #    Si es 0, el tamaño es 0.
            #    El README dice "Tamaño Posición norm."
            #    Asumamos normalizado por el equity inicial (o un valor de capital de referencia).
            #    Si es 0, f_pos_size_norm es 0.
            current_price = self._get_current_price() # Precio de cierre del último kline en la secuencia actual
            position_value = self.position_size_contracts * current_price
            f_pos_size_norm = (position_value / self.initial_equity) * self.leverage if self.initial_equity > 0 else 0.0
            # Multiplicar por apalancamiento para reflejar el tamaño nocional.

            # 3. Precio Entrada norm. (por precio actual)
            f_entry_price_norm = (self.entry_price / current_price) - 1.0 if current_price > 0 and self.position != POSITION_NEUTRAL else 0.0

            # 4. P&L No Realizado norm. (por equity actual o inicial)
            #    self.unrealized_pnl ya está calculado. Normalizarlo por equity actual.
            f_unrealized_pnl_norm = self.unrealized_pnl / self.equity if self.equity > 0 else 0.0

            # 5. Retorno Log Equity Cuenta (log(equity_t / equity_{t-1}))
            #    Este es más una recompensa o un cambio de estado.
            #    Para la observación, podría ser log(equity_actual / equity_inicial_episodio).
            f_log_ret_equity_episode = np.log(self.equity / self.episode_start_equity) if self.episode_start_equity > 0 and self.equity > 0 else 0.0

            # 6. Margen Disponible norm. (Margen disponible / Equity total)
            #    Simplificación: Equity - Margen Usado (nocional) / Equity
            #    Margen Usado (nocional) = Pos_Value / Apalancamiento (pero ya incluimos apalancamiento en f_pos_size_norm)
            #    Margen Usado Real = position_value / self.leverage (si es positivo)
            #    Margen Disponible = Equity - (position_value / self.leverage)
            #    Por ahora, una simplificación: 1.0 si no hay posición, o % de equity no usado por el margen nocional.
            #    Si el margen usado es `position_value / leverage`, entonces `free_margin = equity - position_value / leverage`.
            #    `f_margin_avail_norm = free_margin / equity`
            #    Esta es una aproximación. Binance tiene cálculos de margen más complejos (aislado/cruzado).
            #    Para el MVP, si `self.margin_used` se calcula como `abs(position_value) / self.leverage`,
            #    entonces `f_margin_avail_norm = (self.equity - self.margin_used) / self.equity` if self.equity > 0 else 0.0
            #    Si no hay posición, margen usado es 0, f_margin_avail_norm = 1.0
            self.margin_used = abs(position_value) / self.leverage if self.position != POSITION_NEUTRAL else 0.0
            f_margin_avail_norm = (self.equity - self.margin_used) / self.equity if self.equity > 0 else 0.0
            if self.position == POSITION_NEUTRAL: f_margin_avail_norm = 1.0


            # 7. Pasos desde Apertura Posición norm. (por L, o un max_hold_steps?)
            #    Normalizar por self.sequence_L podría ser una opción.
            f_steps_since_open_norm = float(self.steps_since_position_opened) / self.sequence_L if self.position != POSITION_NEUTRAL else 0.0
            
            # 8. Apalancamiento Configurado (constante, pero útil para el agente)
            #    Normalizarlo? Si es fijo (10x), podría ser 10.0 / MAX_LEVERAGE_POSSIBLE (ej. 125x)
            #    O simplemente el valor 10.0. Si es solo para informar, el valor directo.
            #    Si se usa como feature numérica, mejor normalizar.
            #    Asumamos un MAX_LEVERAGE_HYPOTHETICAL = 125.0 (Binance)
            MAX_LEVERAGE_HYPOTHETICAL = 125.0
            f_leverage_norm = self.leverage / MAX_LEVERAGE_HYPOTHETICAL

            return np.array([
                f_pos_state, f_pos_size_norm, f_entry_price_norm, f_unrealized_pnl_norm,
                f_log_ret_equity_episode, f_margin_avail_norm, f_steps_since_open_norm, f_leverage_norm
            ], dtype=np.float32)

        def _construct_observation(self) -> np.ndarray:
            """
            Construye la observación completa para el agente.
            Combina la secuencia de datos de mercado con las features de estado de la cartera.
            Las features de cartera se replican a lo largo de la dimensión L.
            Forma final: (L, N_market_features + N_portfolio_features)
            """
            market_data_seq = self._get_current_market_data_sequence() # (L, N_market_features)
            portfolio_state_features = self._get_portfolio_state_features() # (N_portfolio_features,)

            # Replicar portfolio_state_features L veces para tener forma (L, N_portfolio_features)
            portfolio_state_seq = np.tile(portfolio_state_features, (self.sequence_L, 1))

            # Concatenar a lo largo del eje de características (axis=1)
            observation = np.concatenate((market_data_seq, portfolio_state_seq), axis=1)
            
            if observation.shape != (self.sequence_L, self.total_features_in_obs):
                 logger.error(f"Forma de observación incorrecta. Esperada: {(self.sequence_L, self.total_features_in_obs)}, Obtenida: {observation.shape}")
                 # Esto podría indicar un problema con N_market_features o N_portfolio_features
                 # O con la concatenación.
                 raise ValueError("Forma de observación construida incorrecta.")
            
            return observation.astype(np.float32)

        def _get_info(self) -> dict:
            """Retorna información adicional sobre el estado del entorno."""
            return {
                "current_step_index": self.current_step_index,
                "equity": self.equity,
                "balance": self.balance,
                "position": self.position,
                "position_size_contracts": self.position_size_contracts,
                "entry_price": self.entry_price,
                "unrealized_pnl": self.unrealized_pnl,
                "realized_pnl_episode": self.realized_pnl_episode,
                "margin_used": self.margin_used,
                "steps_since_position_opened": self.steps_since_position_opened,
                "total_trades_episode": self.total_trades_episode,
                "current_market_timestamp_ms": self.market_timestamps[self.current_step_index] # Timestamp de inicio de la secuencia actual
            }

        def _get_current_price(self) -> float:
            """
            Obtiene el precio de cierre actual del mercado.
            El "actual" es el cierre del último kline en la secuencia de mercado actual.
            market_sequences[self.current_step_index] es (L, N_market_features).
            El último kline es la última fila. Necesitamos el índice de la columna 'close'
            original en los datos brutos de M2 antes de la normalización.
            Esto es un desafío si M2 solo guarda features normalizadas sin el precio original.

            **Decisión de Diseño:** M2 debe guardar, además de las secuencias de features normalizadas,
            una serie temporal de precios de cierre (close_prices) y ATRs (atr_values) no normalizados
            correspondientes a cada paso del dataset (no por secuencia, sino por kline original).
            El archivo .npz de M2 debería contener:
            - `sequences`: (num_total_samples, L, N_market_features)
            - `timestamps`: (num_total_samples,)
            - `close_prices`: (num_total_original_klines,)
            - `atr_values`: (num_total_original_klines,)
            Y un mapeo de `market_timestamps` (inicio de secuencia) a índices en `close_prices`.
            O, más simple: `market_sequences` incluye 'close' y 'atr' *antes* de normalizar,
            o M2 provee un archivo separado de `raw_kline_metrics_per_step.parquet` con (timestamp, close, atr).

            **Alternativa más simple para MVP (usada aquí):**
            Asumir que M2 ha incluido 'feat_close_price_unnorm' y 'feat_atr_unnorm' en sus secuencias,
            o que podemos "desnormalizar" el 'close' y 'feat_atr' de la secuencia actual.
            O, más directo, que la última fila de la secuencia de mercado `market_data_seq[-1, :]`
            contiene las features del kline actual. Si 'feat_close' es una de ellas (normalizada),
            necesitamos el precio de cierre original.

            **Suposición para esta implementación:**
            `self.market_sequences` (shape `num_steps, L, N_market_features`) contiene las features
            normalizadas. Para obtener el precio de cierre y ATR actuales para la mecánica del entorno
            (slippage, P&L), necesitamos los valores *no normalizados* del kline que acaba de cerrar.
            Estos deberían estar en un array alineado con `self.market_timestamps`.
            El `npz` de M2 debe proveerlos.
            Por ahora, asumimos que `self.raw_close_prices` y `self.raw_atr_values` son cargados en `_load_market_data`.
            Estos tendrían longitud `self.total_steps_in_dataset`.
            """
            if hasattr(self, 'raw_close_prices') and self.raw_close_prices is not None:
                # El current_step_index se refiere al *inicio* de la secuencia de longitud L.
                # El precio actual para tomar decisiones y calcular P&L es el cierre del *último* kline
                # en esa secuencia, o el precio de apertura del *siguiente* kline si la acción se toma al cierre.
                # Por convención de RL, la acción se toma, el estado cambia, y luego se observa el nuevo estado.
                # Si la observación es la secuencia hasta el kline T (inclusive), la acción se aplica
                # para operar al inicio del kline T+1 (o al precio de cierre de T).
                # Usaremos el precio de cierre del último kline de la secuencia actual como referencia.
                # El índice `self.current_step_index` en `raw_close_prices` corresponde al timestamp de inicio de la secuencia.
                # El último kline de la secuencia actual corresponde a un índice en `raw_close_prices` que es
                # `self.current_step_index` (si timestamps[i] es el kline i, y la secuencia [i] es klines i..i+L-1)
                # Entonces el cierre del kline i+L-1 es el relevante.
                # Esto se complica. Es más fácil si `raw_close_prices` está alineado con los timesteps principales,
                # y `current_step_index` apunta al kline actual (el último de la ventana de observación).

                # Simplificación: _load_market_data carga `kline_close_prices_for_env_steps`
                # que tiene un precio de cierre para cada `current_step_index`.
                if not (0 <= self.current_step_index < len(self.kline_close_prices_for_env_steps)):
                     raise IndexError(f"current_step_index {self.current_step_index} fuera de rango para kline_close_prices_for_env_steps.")
                return self.kline_close_prices_for_env_steps[self.current_step_index]
            else:
                # Fallback si no hay precios crudos: usar el último 'close' de la secuencia de mercado (normalizado)
                # ¡ESTO ES INCORRECTO PARA CÁLCULOS REALES DE P&L! Necesita desnormalización.
                # Se debe modificar _load_market_data para cargar precios de cierre no normalizados.
                logger.warning("Usando precio de cierre normalizado de la secuencia. ¡Esto es incorrecto para P&L! Se necesitan precios crudos.")
                market_data_seq = self._get_current_market_data_sequence() # (L, N_market_features)
                # Asumir que la feature 'close' está en un índice conocido, ej. 0 después de la normalización.
                # Esto es muy problemático. Requiere una solución en M2 o _load_market_data.
                # Por ahora, placeholder:
                return 20000.0 # Placeholder - ¡REQUERIRÁ CORRECCIÓN URGENTE!

        def _get_current_atr(self) -> float:
            """Obtiene el valor ATR actual (no normalizado). Similar a _get_current_price."""
            if hasattr(self, 'kline_atr_values_for_env_steps') and self.kline_atr_values_for_env_steps is not None:
                if not (0 <= self.current_step_index < len(self.kline_atr_values_for_env_steps)):
                     raise IndexError(f"current_step_index {self.current_step_index} fuera de rango para kline_atr_values_for_env_steps.")
                return self.kline_atr_values_for_env_steps[self.current_step_index]
            else:
                logger.warning("Usando valor ATR placeholder. Se necesitan valores ATR crudos.")
                return 50.0 # Placeholder - ¡REQUERIRÁ CORRECCIÓN URGENTE!
        ```

      * **Modificación Necesaria en `_load_market_data` (y por ende, en la salida de M2)**:

          * M2 debe guardar, además de `sequences` (normalizadas) y `timestamps` (de inicio de secuencia), dos arrays adicionales en el archivo `.npz`:
              * `env_step_close_prices`: Un array 1D con los precios de cierre *no normalizados* para cada paso/kline que el entorno puede tomar. Su longitud debe ser `num_total_samples`. `env_step_close_prices[i]` es el precio de cierre que se usa cuando `current_step_index = i`.
              * `env_step_atr_values`: Un array 1D con los valores ATR *no normalizados* para cada paso. Misma longitud.
          * `_load_market_data` debe cargar estos y asignarlos a `self.kline_close_prices_for_env_steps` y `self.kline_atr_values_for_env_steps`.
          * Esto significa que la Fase 3 (M2) necesita una pequeña adición para guardar estos datos. (Se hará una nota para revisitar M2).

        **Actualización de `_load_market_data` (conceptual):**

        ```python
        # Dentro de TradingEnv, en _load_market_data:
        # ... cargar 'sequences' y 'timestamps' ...
        # if 'env_step_close_prices' in data and 'env_step_atr_values' in data:
        #     self.kline_close_prices_for_env_steps = data['env_step_close_prices'].astype(np.float32)
        #     self.kline_atr_values_for_env_steps = data['env_step_atr_values'].astype(np.float32)
        #     if len(self.kline_close_prices_for_env_steps) != self.total_steps_in_dataset or \
        #        len(self.kline_atr_values_for_env_steps) != self.total_steps_in_dataset:
        #         raise ValueError("Las longitudes de env_step_close_prices/atr_values no coinciden con el número de secuencias.")
        # else:
        #     logger.error("Faltan 'env_step_close_prices' o 'env_step_atr_values' en el archivo de datos .npz. Son necesarios para la mecánica del entorno.")
        #     raise FileNotFoundError("Datos cruciales (precios de cierre, ATRs no normalizados) no encontrados en el archivo .npz.")
        ```

        *Por ahora, se usarán los placeholders en `_get_current_price` y `_get_current_atr`, pero se marcará como deuda técnica.*

-----

\*\*(Continuará con el método `step` y los auxiliares en la siguiente respuesta por límite de longitud)\*\*Okay, Pedro. Esta fase es densa, así que la dividiremos. Aquí está la continuación para el Módulo 3.

Continuamos con la **Fase de Implementación 4: Módulo 3 - Entorno de Trading Simulado (Gymnasium)**.

-----

### Paso 3: Implementación del Método `step` (Lógica Principal)

  * **Descripción Exhaustiva**: Implementar el método `step` que procesa la acción del agente, actualiza el estado del entorno y de la cartera, calcula la recompensa y determina si el episodio ha terminado.
  * **Acciones Específicas**:
      * **3.1. Definir `step` en `TradingEnv`**:
        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
            """
            Ejecuta un paso en el entorno basado en la acción del agente.

            Args:
                action (np.ndarray): Array de NumPy con la señal de acción del agente (shape (1,)).
                                     Valor continuo entre -1 y 1.

            Returns:
                Tuple[np.ndarray, float, bool, bool, dict]:
                    - observation (np.ndarray): La nueva observación del estado.
                    - reward (float): La recompensa obtenida en este paso.
                    - terminated (bool): True si el episodio ha terminado por una condición de fin (liquidación, drawdown).
                    - truncated (bool): True si el episodio ha terminado por alcanzar un límite de tiempo/pasos.
                    - info (dict): Información adicional.
            """
            action_signal = float(action[0]) # Extraer la señal del array
            terminated = False
            truncated = False

            # Guardar equity anterior para cálculo de recompensa
            previous_equity = self.equity

            # Obtener datos de mercado actuales (necesarios para P&L, slippage, etc.)
            # Estos son los valores NO normalizados.
            current_market_price = self._get_current_price() 
            current_market_atr = self._get_current_atr()

            # 1. Actualizar P&L no realizado si hay una posición abierta (antes de cualquier acción)
            if self.position != POSITION_NEUTRAL:
                self._update_pnl_and_equity(current_market_price)
                self.steps_since_position_opened += 1

            # 2. Interpretar la acción y ejecutar la lógica de trading
            order_decision = self._interpret_action(action_signal) # HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_POSITION
            
            if order_decision != "HOLD": # Si la decisión es hacer algo
                self._execute_trade(order_decision, current_market_price, current_market_atr)

            # 3. Después del trade (si hubo), actualizar P&L y equity de nuevo
            # (Si se cerró posición, el P&L se realizó. Si se abrió, unrealized_pnl es 0 inicialmente).
            self._update_pnl_and_equity(current_market_price)


            # 4. Calcular recompensa (basada en el cambio de equity)
            reward = self._calculate_reward(previous_equity, self.equity)
            
            # 5. Avanzar al siguiente paso de tiempo en los datos de mercado
            self.current_step_index += 1

            # 6. Comprobar condiciones de fin de episodio
            # 6.1. Liquidación
            if self.position != POSITION_NEUTRAL and self._check_liquidation(current_market_price):
                logger.info(f"¡LIQUIDACIÓN! Posición: {self.position}, Precio Entrada: {self.entry_price}, "
                            f"Precio Actual: {current_market_price}, Equity: {self.equity}")
                # El P&L de la liquidación ya se debería haber aplicado en _check_liquidation o _execute_trade (al cerrar)
                # Por ahora, _check_liquidation solo retorna bool, y _execute_trade maneja el cierre.
                # Si _check_liquidation implica forzar cierre, se debe reflejar en P&L y equity aquí.
                # Asumimos que si _check_liquidation es true, la posición ya fue efectivamente cerrada
                # y el equity refleja la pérdida por liquidación (o se ajusta aquí).
                # Por simplicidad, la liquidación causa una gran pérdida y resetea la posición.
                # (La lógica de cierre real en _execute_trade manejaría esto)
                terminated = True
                # Aquí se podría aplicar una penalización adicional por liquidación si se desea
                # reward -= self.env_config.get("liquidation_penalty", 0.5) # Ejemplo
            
            # 6.2. Drawdown máximo de equity del episodio
            current_episode_drawdown = (self.equity - self.episode_start_equity) / self.episode_start_equity
            if current_episode_drawdown < self.equity_drawdown_threshold: # threshold es negativo
                logger.info(f"Drawdown máximo del episodio alcanzado ({current_episode_drawdown*100:.2f}%). Equity: {self.equity}")
                terminated = True

            # 6.3. Agotamiento del conjunto de datos
            # Si current_step_index apunta al *siguiente* estado a observar,
            # entonces el episodio termina si ya no hay más datos para formar una observación completa.
            # self.total_steps_in_dataset es el número de secuencias (L) disponibles.
            if self.current_step_index >= self.total_steps_in_dataset:
                logger.info("Se agotaron los datos del dataset.")
                truncated = True # O terminated, según la convención de Gymnasium. Truncated es más para límites de tiempo.
                                 # Si es el fin natural de los datos, puede ser terminated.
                                 # El README dice: "Agotamiento del Conjunto de Datos de Entrenamiento: Se alcanza max_episode_steps igual a la longitud del dataset (truncated = True)"
                                 # Asumimos que max_episode_steps es self.total_steps_in_dataset.
                if not terminated : # Solo si no fue terminated por otra razón
                    truncated = True 
            
            # 7. Construir la nueva observación
            # Si es terminated o truncated, la observación devuelta puede no ser usada por el agente
            # para tomar decisiones, pero Gymnasium requiere que se devuelva una.
            if terminated or truncated:
                # Devolver una observación "dummy" o la última válida, o una de ceros.
                # Por ahora, intentamos construirla. Si falla por falta de datos (current_step_index out of bounds),
                # se manejará en _construct_observation o se devolverá una de ceros.
                try:
                    observation = self._construct_observation()
                except IndexError: # Si current_step_index ya está fuera de rango para el *siguiente* estado
                    logger.warning("Índice fuera de rango al final del episodio, devolviendo observación de ceros.")
                    observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                observation = self._construct_observation()

            # 8. Obtener información adicional
            info = self._get_info()
            info["reward_raw"] = reward # Guardar recompensa antes de cualquier modificación/clipping

            # (Opcional) Clipping de recompensa si se especifica en config
            # reward_clip_min = self.env_config.get("reward_clip_min", -1.0)
            # reward_clip_max = self.env_config.get("reward_clip_max", 1.0)
            # reward = np.clip(reward, reward_clip_min, reward_clip_max)

            if terminated or truncated:
                logger.info(f"Episodio finalizado. Terminated: {terminated}, Truncated: {truncated}. "
                            f"Equity Final: {self.equity:.2f}, P&L Realizado Episodio: {self.realized_pnl_episode:.2f}, "
                            f"Total Trades: {self.total_trades_episode}")

            return observation, reward, terminated, truncated, info
        ```

-----

### Paso 4: Implementación de Métodos Auxiliares (Lógica del Entorno)

  * **Descripción Exhaustiva**: Implementar los métodos privados que manejan la lógica de trading, P\&L, comisiones, slippage, liquidación, y cálculo de recompensas.
  * **Acciones Específicas**:
      * **4.1. Definir `_interpret_action`**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def _interpret_action(self, action_signal: float) -> str:
            """
            Interpreta la señal de acción continua del agente en una decisión de orden discreta.
            Devuelve: "HOLD", "TRY_OPEN_LONG", "TRY_OPEN_SHORT", "TRY_CLOSE_POSITION"
            """
            threshold = self.action_threshold

            if action_signal > threshold: # Señal para comprar/abrir largo
                if self.position == POSITION_SHORT: # Si está corto, primero cerrar y luego intentar abrir largo
                    return "TRY_CLOSE_THEN_OPEN_LONG" 
                elif self.position == POSITION_NEUTRAL:
                    return "TRY_OPEN_LONG"
                else: # Ya está largo, mantener
                    return "HOLD" 
            elif action_signal < -threshold: # Señal para vender/abrir corto
                if self.position == POSITION_LONG: # Si está largo, primero cerrar y luego intentar abrir corto
                    return "TRY_CLOSE_THEN_OPEN_SHORT"
                elif self.position == POSITION_NEUTRAL:
                    return "TRY_OPEN_SHORT"
                else: # Ya está corto, mantener
                    return "HOLD"
            else: # Señal neutral (-threshold <= action_signal <= threshold)
                if self.position != POSITION_NEUTRAL: # Si hay posición abierta, cerrarla
                    return "TRY_CLOSE_POSITION"
                else: # Neutral y señal neutral, mantener
                    return "HOLD"
        ```

      * **4.2. Definir `_calculate_slippage` y `_calculate_commission`**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def _calculate_slippage(self, current_atr: float, order_side: str) -> float:
            """
            Calcula el coste de slippage para una operación.
            Se aplica un slippage de `0.1 * ATR(14)` por cada lado.
            Args:
                current_atr (float): Valor actual del ATR(14) no normalizado.
                order_side (str): "BUY" o "SELL". El slippage siempre es adverso.
            Returns:
                float: Deslizamiento en precio por unidad del activo.
            """
            slippage_per_unit = self.slippage_atr_multiplier * current_atr
            # El slippage siempre es adverso: si compras, pagas más; si vendes, recibes menos.
            # El método _execute_trade aplicará esto al precio de ejecución.
            # Aquí solo retornamos la magnitud.
            return slippage_per_unit

        def _calculate_commission(self, trade_notional_value: float) -> float:
            """
            Calcula la comisión de Taker para una operación.
            Args:
                trade_notional_value (float): Valor nocional de la operación (Precio * Cantidad_Contratos * Apalancamiento implícito en futuros).
                                             O más simple: Precio * Cantidad_Contratos (valor real de los activos base transaccionados).
                                             Para futuros, la comisión se calcula sobre el valor nocional (precio * cantidad * multiplicador_contrato).
                                             Asumimos que `position_size_contracts` es en activo base (ej. BTC).
                                             Entonces `trade_notional_value` es `precio_ejecucion * cantidad_contratos_transaccionados`.
            Returns:
                float: Coste de la comisión.
            """
            return abs(trade_notional_value) * self.taker_fee_rate
        ```

      * **4.3. Definir `_execute_trade` (Complejo, maneja la lógica de abrir/cerrar)**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def _execute_trade(self, order_decision: str, current_price: float, current_atr: float):
            """
            Ejecuta la lógica de trading basada en la decisión de orden.
            Actualiza el estado de la cartera (posición, precio de entrada, equity, P&L realizado).
            """
            if order_decision == "HOLD":
                return # No hacer nada

            initial_equity_before_trade = self.equity # Para loguear costes
            trade_executed_info = {"type": order_decision, "executed": False, "reason": ""}

            # --- Lógica para CERRAR una posición existente ---
            if order_decision in ["TRY_CLOSE_POSITION", "TRY_CLOSE_THEN_OPEN_LONG", "TRY_CLOSE_THEN_OPEN_SHORT"]:
                if self.position != POSITION_NEUTRAL:
                    closing_qty_contracts = self.position_size_contracts
                    
                    # Determinar precio de cierre con slippage
                    slippage_amount = self._calculate_slippage(current_atr, "SELL" if self.position == POSITION_LONG else "BUY")
                    execution_price_close = current_price - slippage_amount if self.position == POSITION_LONG else current_price + slippage_amount
                    
                    # Calcular P&L de esta posición
                    pnl_gross = 0
                    if self.position == POSITION_LONG:
                        pnl_gross = (execution_price_close - self.entry_price) * closing_qty_contracts
                    elif self.position == POSITION_SHORT:
                        pnl_gross = (self.entry_price - execution_price_close) * closing_qty_contracts
                    
                    # Calcular comisión de cierre
                    trade_notional_value_close = execution_price_close * closing_qty_contracts
                    commission_close = self._calculate_commission(trade_notional_value_close)
                    
                    pnl_net = pnl_gross - commission_close
                    
                    # Actualizar equity y P&L realizado
                    self.equity += pnl_net
                    self.balance = self.equity # Asumimos que balance = equity después de P&L realizado
                    self.realized_pnl_episode += pnl_net
                    self.unrealized_pnl = 0.0 # Ya no hay P&L no realizado para esta posición
                    
                    logger.info(f"Posición CERRADA: {'LONG' if self.position == POSITION_LONG else 'SHORT'} "
                                f"Qty: {closing_qty_contracts:.4f}, Entry: {self.entry_price:.2f}, "
                                f"MarketPrice: {current_price:.2f}, ExecPrice: {execution_price_close:.2f}, "
                                f"Slippage: {slippage_amount:.2f}, PnL Gross: {pnl_gross:.2f}, "
                                f"Commission: {commission_close:.2f}, PnL Net: {pnl_net:.2f}, Equity: {self.equity:.2f}")

                    self.position = POSITION_NEUTRAL
                    self.position_size_contracts = 0.0
                    self.entry_price = 0.0
                    self.steps_since_position_opened = 0
                    self.margin_used = 0.0
                    self.total_trades_episode += 1 # Contar como una operación completa (apertura+cierre)
                                                  # O contar aperturas y cierres como operaciones separadas (contar aquí como 1 trade de cierre)
                                                  # Aquí lo contamos como 1 evento de cierre.
                    trade_executed_info["executed"] = True
                    trade_executed_info["closed_position"] = True
                else: # No había posición que cerrar
                    if order_decision == "TRY_CLOSE_POSITION":
                         trade_executed_info["reason"] = "No hay posición para cerrar."

            # --- Lógica para ABRIR una nueva posición ---
            # Esto ocurre si la decisión era solo abrir, o abrir después de un cierre.
            open_new_position_type = POSITION_NEUTRAL
            if order_decision == "TRY_OPEN_LONG" or order_decision == "TRY_CLOSE_THEN_OPEN_LONG":
                open_new_position_type = POSITION_LONG
            elif order_decision == "TRY_OPEN_SHORT" or order_decision == "TRY_CLOSE_THEN_OPEN_SHORT":
                open_new_position_type = POSITION_SHORT

            if open_new_position_type != POSITION_NEUTRAL:
                if self.position == POSITION_NEUTRAL: # Solo abrir si no hay ya una posición (la anterior se cerró)
                    # Calcular tamaño de la posición
                    # Capital a arriesgar = equity * position_size_pct_equity
                    # Valor nocional de la posición = Capital a arriesgar * apalancamiento
                    # Cantidad de contratos = Valor nocional / Precio actual
                    capital_at_risk = self.equity * self.position_size_pct_equity
                    if capital_at_risk <= 0: # No abrir si no hay capital o es negativo
                        logger.warning(f"Intento de abrir posición con capital <= 0 ({capital_at_risk:.2f}). No se abre.")
                        trade_executed_info["reason"] += " Capital insuficiente para abrir."
                        return # No hacer más nada en _execute_trade
                        
                    notional_position_value = capital_at_risk * self.leverage
                    qty_contracts_to_open = notional_position_value / current_price if current_price > 0 else 0
                    
                    # (Opcional) Considerar mínimos de orden de Binance (ej. 0.001 BTC)
                    # min_order_qty_btc = 0.0001 # Ejemplo, obtener de config o API info
                    # if qty_contracts_to_open < min_order_qty_btc:
                    #    logger.warning(f"Cantidad calculada ({qty_contracts_to_open}) menor que mínimo de orden. No se abre.")
                    #    return

                    # Determinar precio de entrada con slippage
                    slippage_amount = self._calculate_slippage(current_atr, "BUY" if open_new_position_type == POSITION_LONG else "SELL")
                    execution_price_open = current_price + slippage_amount if open_new_position_type == POSITION_LONG else current_price - slippage_amount

                    # Calcular comisión de apertura
                    trade_notional_value_open = execution_price_open * qty_contracts_to_open
                    commission_open = self._calculate_commission(trade_notional_value_open)

                    # Actualizar equity (se reduce por la comisión)
                    self.equity -= commission_open
                    self.realized_pnl_episode -= commission_open # Comisiones son P&L realizado negativo

                    # Establecer nueva posición
                    self.position = open_new_position_type
                    self.position_size_contracts = qty_contracts_to_open
                    self.entry_price = execution_price_open
                    self.steps_since_position_opened = 0 # Reseteado al abrir
                    self.total_trades_episode +=1 # Contar apertura como una operación/evento

                    # Margen usado (aproximado)
                    self.margin_used = abs(trade_notional_value_open) / self.leverage 

                    logger.info(f"Posición ABIERTA: {'LONG' if self.position == POSITION_LONG else 'SHORT'} "
                                f"Qty: {self.position_size_contracts:.4f}, MarketPrice: {current_price:.2f}, "
                                f"ExecPrice: {self.entry_price:.2f}, Slippage: {slippage_amount:.2f}, "
                                f"Commission: {commission_open:.2f}, Equity: {self.equity:.2f}, Margin Used: {self.margin_used:.2f}")
                    trade_executed_info["executed"] = True
                    trade_executed_info["opened_position"] = True
                else: # Ya hay una posición, y la señal era para abrir (ej. ya largo y señal de abrir largo)
                    trade_executed_info["reason"] += " Ya existe una posición."
            
            if not trade_executed_info["executed"] and not trade_executed_info["reason"]:
                trade_executed_info["reason"] = "Decisión no llevó a ninguna acción de trade (ej. Hold o ya en estado deseado)."
            
            # logger.debug(f"Trade execution info: {trade_executed_info}")
            # Aquí se podría añadir trade_executed_info al self.info del entorno, pero step() lo regenera.
            # Se podría acumular en una lista de trades del episodio si es necesario.
        ```

      * **4.4. Definir `_update_pnl_and_equity`**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def _update_pnl_and_equity(self, current_price: float):
            """
            Actualiza el P&L no realizado y el equity basado en el precio de mercado actual.
            El equity aquí refleja el valor de liquidación de la posición si se cerrara AHORA MISMO
            SIN costes adicionales (estos se aplican en _execute_trade al cerrar).
            """
            if self.position == POSITION_NEUTRAL:
                self.unrealized_pnl = 0.0
                # self.balance = self.equity # No es necesario si se actualiza al cerrar
                return

            pnl_per_contract = 0
            if self.position == POSITION_LONG:
                pnl_per_contract = current_price - self.entry_price
            elif self.position == POSITION_SHORT:
                pnl_per_contract = self.entry_price - current_price
            
            self.unrealized_pnl = pnl_per_contract * self.position_size_contracts
            
            # Equity = Balance (P&L realizado) + P&L No Realizado
            # self.balance se actualiza cuando se realiza P&L (cierre de posición, comisiones)
            self.equity = self.balance + self.unrealized_pnl
        ```

      * **4.5. Definir `_check_liquidation`**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def _check_liquidation(self, current_price: float) -> bool:
            """
            Comprueba si la posición actual debe ser liquidada.
            Liquidación si el precio se mueve un X% en contra (ej. 8% para apalancamiento 10x con factor seguridad 0.8).
            El X% se calcula como (1 / Apalancamiento) * Factor_Seguridad_Liquidacion.
            Este método solo comprueba; el cierre y ajuste de equity se hace en _execute_trade.
            Devuelve True si debe liquidarse, False en caso contrario.
            """
            if self.position == POSITION_NEUTRAL:
                return False

            # Porcentaje de movimiento adverso para liquidación
            liquidation_threshold_pct = (1.0 / self.leverage) * self.liquidation_safety_factor
            
            price_change_pct = 0
            if self.position == POSITION_LONG:
                # Pérdida si current_price < entry_price
                price_change_pct = (self.entry_price - current_price) / self.entry_price if self.entry_price > 0 else 0
            elif self.position == POSITION_SHORT:
                # Pérdida si current_price > entry_price
                price_change_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
            
            if price_change_pct >= liquidation_threshold_pct:
                # Forzar cierre de posición debido a liquidación.
                # Esto debería ocurrir dentro de _execute_trade o llamar a una función de cierre forzado.
                # Por ahora, esta función solo identifica la condición.
                # El `step` method checkeará esto y marcará `terminated = True`.
                # La pérdida real se calculará cuando la posición se cierre (posiblemente en el mismo `step`).
                logger.warning(f"CONDICIÓN DE LIQUIDACIÓN DETECTADA. Pos: {self.position}, "
                               f"Entry: {self.entry_price:.2f}, CurrPrice: {current_price:.2f}, "
                               f"PriceChange: {price_change_pct*100:.2f}%, LiqThreshold: {liquidation_threshold_pct*100:.2f}%")
                return True
            
            return False
        ```

      * **4.6. Definir `_calculate_reward`**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def _calculate_reward(self, prev_equity: float, current_equity: float) -> float:
            """
            Calcula la recompensa para el paso actual.
            Recompensa = log(equity_t / equity_{t-1})
            """
            if prev_equity <= 0 or current_equity <= 0: # Evitar log(0) o log(negativo)
                # Si el equity se vuelve no positivo, es un estado muy malo.
                # Podría ser una recompensa muy negativa o simplemente 0 si el episodio termina.
                # Si current_equity es 0 después de ser positivo, la recompensa log sería -inf.
                return -1.0 if current_equity <=0 else 0.0 # Penalización fuerte si se quiebra.
            
            # El log return puede ser muy pequeño. A veces se escala o se usan otras funciones de recompensa.
            # Para el MVP, seguimos la especificación.
            reward = np.log(current_equity / prev_equity)
            
            # (Opcional) Añadir modelado de recompensa (reward shaping) si es necesario:
            # - Penalización por over-trading (si self.total_trades_episode es alto)
            # - Bonus por mantener una posición rentable por X tiempo
            # Por ahora, solo el log return del equity.
            
            return float(reward)
        ```

      * **4.7. Implementar `render` y `close` (básicos para Gymnasium)**:

        ```python
        # En src/environment/trading_env.py, dentro de la clase TradingEnv

        def render(self) -> Optional[Union[np.ndarray, str]]:
            """
            Renderiza el estado actual del entorno (opcional).
            Para 'human' o 'ansi', podría imprimir información de la cartera.
            """
            mode = self.render_mode # Establecido por Gymnasium o al hacer gym.make(..., render_mode=...)
            
            if mode == 'ansi':
                return (f"Step: {self.current_step_index}, Equity: {self.equity:.2f}, "
                        f"Position: {self.position}, Entry: {self.entry_price:.2f}, "
                        f"Size: {self.position_size_contracts:.4f}, UPL: {self.unrealized_pnl:.2f}, "
                        f"RPL Ep: {self.realized_pnl_episode:.2f}, Trades: {self.total_trades_episode}")
            elif mode == 'human':
                # Podría usar Matplotlib para graficar equity, etc., si se ejecuta interactivamente.
                # Por ahora, igual que ansi.
                print(f"Step: {self.current_step_index}, Equity: {self.equity:.2f}, "
                      f"Position: {self.position}, Entry: {self.entry_price:.2f}, "
                      f"Size: {self.position_size_contracts:.4f}, UPL: {self.unrealized_pnl:.2f}, "
                      f"RPL Ep: {self.realized_pnl_episode:.2f}, Trades: {self.total_trades_episode}")
                return None # Opcional, podría devolver un array RGB si se implementa renderizado gráfico.
            elif mode == 'rgb_array':
                # Implementar la generación de un array RGB que represente el estado
                # (ej. un gráfico simple de la curva de equity o features).
                # Esto es más complejo y opcional para la mayoría de los entornos de trading.
                logger.warning("Modo de render 'rgb_array' no implementado completamente.")
                return np.zeros((100,100,3), dtype=np.uint8) # Placeholder
            return None

        def close(self):
            """
            Limpia recursos del entorno (si es necesario).
            """
            logger.info("Cerrando TradingEnv.")
            # No hay recursos abiertos explícitos aquí (archivos, conexiones de red, etc.)
            # que no se manejen ya localmente en los métodos.
            pass

        def set_eval_mode(self, start_index: int = 0):
            """Configura el entorno para modo de evaluación (inicio fijo, sin aleatoriedad)."""
            self.eval_mode = True
            self.eval_start_index = start_index
            logger.info(f"Entorno configurado para modo EVALUACIÓN. Inicio fijo en índice: {start_index}")

        def set_train_mode(self):
            """Configura el entorno para modo de entrenamiento (inicio aleatorio)."""
            self.eval_mode = False
            logger.info("Entorno configurado para modo ENTRENAMIENTO.")

        ```

-----


Continuamos con la **Fase de Implementación 4: Módulo 3 - Entorno de Trading Simulado (Gymnasium)**.

-----

### Paso 5: Script de Prueba para el Entorno en `scripts/test_environment.py`

  * **Descripción Exhaustiva**: Crear un script que instancie `TradingEnv` y ejecute algunos pasos con acciones aleatorias (o una política simple) para verificar que el flujo principal del entorno (reset, step) funciona y que los estados/recompensas parecen razonables.
  * **Acciones Específicas**:
      * **5.1. Crear `scripts/test_environment.py`**:

        ```python
        # scripts/test_environment.py
        import sys
        import logging
        from pathlib import Path
        import numpy as np
        import time # Para medir rendimiento

        # Añadir src al PYTHONPATH
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from common.utils import setup_logging
        from environment.trading_env import TradingEnv # Asumiendo que la clase está en trading_env.py
        from config_loader import get_env_variable, load_yaml_config # Para obtener rutas y configs

        try:
            setup_logging()
        except Exception as e:
            logging.basicConfig(level=logging.ERROR)
            logging.critical(f"Fallo CRÍTICO al configurar logging: {e}", exc_info=True)

        logger = logging.getLogger(__name__)

        def run_random_agent_test(env: TradingEnv, num_episodes: int = 1, num_steps_per_episode: int = 200):
            """Ejecuta una prueba con un agente aleatorio."""
            logger.info(f"Iniciando prueba con agente aleatorio: {num_episodes} episodios, {num_steps_per_episode} pasos/ep.")

            for episode in range(num_episodes):
                logger.info(f"--- Iniciando Episodio {episode + 1}/{num_episodes} ---")
                # Para entrenamiento, resetear sin opciones específicas para inicio aleatorio (si está configurado así el env)
                # Para prueba consistente, podemos pasar un `start_index` si el env lo soporta en options.
                # env.set_train_mode() # O set_eval_mode(start_index=...)
                obs, info = env.reset(options={"start_index": 0}) # Forzar inicio en 0 para esta prueba
                
                total_reward_episode = 0
                start_time_episode = time.time()

                for step in range(num_steps_per_episode):
                    action = env.action_space.sample() # Acción aleatoria
                    
                    new_obs, reward, terminated, truncated, info = env.step(action)
                    total_reward_episode += reward

                    # Imprimir información del paso (puede ser muy verboso)
                    if (step + 1) % 50 == 0 or terminated or truncated : # Imprimir cada 50 pasos o al final
                        logger.debug(f"Ep {episode+1}, Step {step+1}: Action: {action[0]:.2f}, Reward: {reward:.6f}, "
                                     f"Term: {terminated}, Trunc: {truncated}, Equity: {info.get('equity', 0):.2f}, "
                                     f"Pos: {info.get('position', 0)}")
                                     # f"Obs shape: {new_obs.shape}")


                    obs = new_obs
                    if terminated or truncated:
                        logger.info(f"Episodio {episode + 1} finalizado en paso {step + 1}. "
                                    f"Causa: {'Terminated' if terminated else 'Truncated'}. "
                                    f"Recompensa Total: {total_reward_episode:.4f}, "
                                    f"Equity Final: {info.get('equity', 0):.2f}")
                        break
                
                end_time_episode = time.time()
                duration_episode = end_time_episode - start_time_episode
                steps_taken = step + 1
                sps = steps_taken / duration_episode if duration_episode > 0 else float('inf')
                logger.info(f"Episodio {episode+1} resumen: Duración: {duration_episode:.2f}s, Pasos: {steps_taken}, SPS: {sps:.2f}")
                logger.info(f"--- Fin Episodio {episode + 1}/{num_episodes} ---")

        def main():
            logger.info("==========================================================")
            logger.info("Iniciando script de prueba del Entorno de Trading...")
            logger.info("==========================================================")

            try:
                # Obtener la ruta al archivo de datos procesados (.npz)
                # Asumimos que la configuración del preprocesador puede darnos el nombre base
                preproc_cfg = load_yaml_config("module2_preprocessing")
                data_acq_cfg = load_yaml_config("module1_data_acquisition")
                
                trading_pair = data_acq_cfg.get("trading_pair", "BTCUSDT")
                kline_interval = data_acq_cfg.get("kline_interval", "15m")
                sequence_L = preproc_cfg.get("sequence_length_L", 96)
                # N_features se infiere del archivo .npz, pero podemos construir el nombre esperado
                # N_effective_features = NUM_MARKET_FEATURES + NUM_PORTFOLIO_FEATURES # Definido en trading_env.py
                # Para obtener el N_features real del archivo, necesitamos saberlo o inferirlo
                # DataPreprocessor.EXPECTED_N_FEATURES era 20 (solo market)
                # El entorno espera N_market_features + N_portfolio_features = 20 + 8 = 28 en la obs
                # El archivo .npz tiene N_market_features = 20 en sus secuencias.
                
                # Construir el nombre del archivo .npz esperado
                # (Debe coincidir con cómo se guardó en Fase 3, Paso 7)
                # processed_filename = f"{trading_pair}_{kline_interval}_L{sequence_L}_N{N_market_features_from_m2}_processed_sequences.npz"
                # Donde N_market_features_from_m2 es 20.
                
                data_dir_host_str = get_env_variable("DATA_DIR_HOST_FOR_APP", "/app/data_persistent")
                processed_data_base_path = Path(data_dir_host_str) / "processed" / trading_pair / kline_interval
                
                # El nombre exacto del archivo puede variar si N_features en el nombre es el N_market
                # o el N_total. Asumamos que es N_market (20)
                processed_filename = f"{trading_pair}_{kline_interval}_L{sequence_L}_N{20}_processed_sequences.npz" # N20 -> N_market_features
                npz_file_path = processed_data_base_path / processed_filename

                logger.info(f"Intentando cargar datos del entorno desde: {npz_file_path}")
                if not npz_file_path.exists():
                    logger.error(f"Archivo de datos procesados no encontrado: {npz_file_path}")
                    logger.error("Por favor, ejecuta primero el script de preprocesamiento (Fase 3).")
                    
                    # Como fallback para la prueba, intentar usar el archivo de prueba generado en TradingEnv.__main__
                    # Esto es solo para desarrollo/prueba del script test_environment.py
                    logger.warning("Intentando usar archivo de datos de prueba dummy si existe...")
                    dummy_L_test = 96 # Coincidir con config
                    dummy_N_market_feat_test = 20
                    num_samples_test_env = 500
                    dummy_test_data_dir = Path("./temp_test_env_data") # Creado por TradingEnv.__main__
                    dummy_npz_path = dummy_test_data_dir / f"test_processed_sequences_L{dummy_L_test}_N{dummy_N_market_feat_test}.npz"
                    if dummy_npz_path.exists():
                        npz_file_path = dummy_npz_path
                        logger.info(f"Usando archivo de datos dummy: {npz_file_path}")
                    else:
                        logger.error("No se pudo encontrar ni el archivo de datos procesados real ni el dummy. Abortando prueba.")
                        return

                # Instanciar el entorno
                trading_environment = TradingEnv(data_npz_path=npz_file_path)
                
                # (Opcional) Comprobar con gymnasium.utils.env_checker
                # from gymnasium.utils.env_checker import check_env
                # try:
                #     logger.info("Comprobando el entorno con env_checker de Gymnasium...")
                #     check_env(trading_environment)
                #     logger.info("check_env completado sin errores.")
                # except Exception as e_check:
                #     logger.error(f"Error durante check_env: {e_check}", exc_info=True)

                # Ejecutar prueba con agente aleatorio
                run_random_agent_test(trading_environment, num_episodes=2, num_steps_per_episode=300)

                logger.info("Script de prueba del Entorno de Trading finalizado.")

            except FileNotFoundError as e_fnf:
                logger.error(f"Error de archivo no encontrado: {e_fnf}")
            except Exception as e:
                logger.error(f"Error inesperado en el script de prueba del entorno: {e}", exc_info=True)
            finally:
                logger.info("==========================================================")
                logger.info("Finalización del script de prueba del Entorno.")
                logger.info("==========================================================")

        if __name__ == "__main__":
            main()
        ```

      * **Importante sobre `_get_current_price` y `_get_current_atr`**:
        El script de prueba anterior fallará o usará placeholders si el archivo `.npz` de M2 no contiene `env_step_close_prices` y `env_step_atr_values`. Es **CRUCIAL** que la Fase 3 (M2) se revise para asegurar que estos arrays (con datos *no normalizados* y alineados con los `timestamps` de inicio de secuencia) se guarden en el archivo `.npz`.

        **Acción Requerida (Deuda Técnica de Fase 3):**
        Modificar `src/preprocessing/feature_engineer.py` (Módulo 2, Fase 3, Paso 7.1 `process_data`) para que, además de las `sequences` normalizadas, guarde:

          * `env_step_close_prices`: Array de precios de cierre originales correspondientes a cada `timestamp` de inicio de secuencia.
          * `env_step_atr_values`: Array de valores ATR originales correspondientes a cada `timestamp` de inicio de secuencia.
            Estos se extraerían del DataFrame `df_processed` *antes* de la normalización de estas columnas específicas, o del DataFrame original `klines_df` alineándolos con los `sequence_start_timestamps_ms`.

        Por ejemplo, en `process_data` de `feature_engineer.py`, antes de guardar:

        ```python
        # ... (después de calcular features y ANTES de normalizar todo para las secuencias)
        # df_with_features_before_norm = df_processed.copy() # Si df_processed ya tiene todo
        # ...
        # (después de que sequences_np y sequence_start_timestamps_ms estén listos)
        # Se necesita un DF que tenga 'kline_open_time', 'close' (original), 'feat_atr' (original)
        # y que esté alineado con los `sequence_start_timestamps_ms`.
        # El DataFrame `df_processed` justo antes de `_normalize_features` podría tener esto,
        # o `klines_df` original si se puede mapear.

        # Solución más simple:
        # En process_data, después de cargar klines_df y antes de cualquier modificación que quite columnas:
        # df_for_env_mechanics = klines_df[klines_df['kline_open_time'].isin(sequence_start_timestamps_ms)].copy()
        # env_closes = df_for_env_mechanics['close'].values
        # env_atrs = df_for_env_mechanics['ATR_columna_calculada_antes_norm'].values # Requiere que ATR se calcule y guarde.
        # np.savez_compressed(..., env_step_close_prices=env_closes, env_step_atr_values=env_atrs)
        ```

        Luego, en `TradingEnv._load_market_data`, cargar estos arrays.

      * **5.2. Ejecutar el Script de Prueba**:

          * Asegurar que exista un archivo `.npz` (real de M2, o el dummy si `test_environment.py` lo crea).
          * Ejecutar dentro de Docker:
            ```bash
            docker-compose exec workhorse_app python scripts/test_environment.py
            ```
          * Revisar logs para el comportamiento del agente aleatorio.

-----

### Paso 6: Pruebas Unitarias (Básicas) para `TradingEnv`

  * **Descripción Exhaustiva**: Crear pruebas unitarias para los métodos clave de `TradingEnv`, como la lógica de `reset`, `step` (con acciones específicas para abrir/cerrar), cálculo de P\&L, comisiones y recompensas. Usar un dataset `.npz` de prueba pequeño y determinista.
  * **Acciones Específicas**:
      * **6.1. Crear `tests/environment/test_trading_env.py`**:
        ```python
        # tests/environment/test_trading_env.py
        import pytest
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from unittest import mock

        from src.environment.trading_env import TradingEnv, POSITION_LONG, POSITION_SHORT, POSITION_NEUTRAL
        from src.environment.trading_env import NUM_MARKET_FEATURES, NUM_PORTFOLIO_FEATURES # Para verificar formas

        @pytest.fixture
        def mock_env_configs(monkeypatch):
            """Mockea load_yaml_config para el entorno."""
            def get_mock_config(module_name, file_name="params.yaml"):
                if module_name == "module3_environment":
                    return {
                        "initial_equity": 10000.0, "leverage": 10.0,
                        "position_size_pct_equity": 0.05, "taker_fee_rate": 0.0004,
                        "slippage_atr_multiplier": 0.1, "action_threshold": 0.15,
                        "equity_drawdown_threshold_episode_end": -0.20,
                        "liquidation_safety_factor": 0.8,
                        "max_episode_steps_equals_dataset_length": False # Para permitir inicios aleatorios en tests
                    }
                if module_name == "module2_preprocessing":
                    return {"sequence_length_L": 10} # Usar L pequeña para tests
                return {}
            monkeypatch.setattr("src.environment.trading_env.load_yaml_config", get_mock_config)
            
            # Mock get_env_variable si es necesario para rutas u otros
            monkeypatch.setattr("src.environment.trading_env.get_env_variable", lambda x, y=None, required=True: "dummy_val_for_env_tests")


        @pytest.fixture
        def sample_npz_data_for_env(tmp_path: Path) -> Path:
            """Crea un archivo .npz de prueba para el entorno."""
            L_test = 10 # Debe coincidir con sequence_length_L del mock_env_configs
            N_market_feat_test = NUM_MARKET_FEATURES # 20
            num_samples_test = 50 # Suficientes pasos para algunos trades

            sequences = np.random.rand(num_samples_test, L_test, N_market_feat_test).astype(np.float32)
            timestamps = np.arange(num_samples_test).astype(np.int64) * (15 * 60 * 1000) # Timestamps de inicio de secuencia
            
            # Datos cruciales para la mecánica del entorno (NO NORMALIZADOS)
            # Longitud num_samples_test (uno por cada posible 'current_step_index')
            close_prices = np.linspace(20000, 21000, num_samples_test).astype(np.float32)
            atr_values = np.full(num_samples_test, 50.0).astype(np.float32) # ATR constante para simplificar

            npz_file = tmp_path / "test_env_data.npz"
            np.savez_compressed(
                npz_file, 
                sequences=sequences, 
                timestamps=timestamps,
                env_step_close_prices=close_prices, # Clave para _get_current_price
                env_step_atr_values=atr_values      # Clave para _get_current_atr
            )
            return npz_file

        def test_env_initialization(mock_env_configs, sample_npz_data_for_env):
            """Testea la inicialización del entorno."""
            env = TradingEnv(data_npz_path=sample_npz_data_for_env)
            assert env.sequence_L == 10
            assert env.total_features_in_obs == NUM_MARKET_FEATURES + NUM_PORTFOLIO_FEATURES
            assert env.observation_space.shape == (10, NUM_MARKET_FEATURES + NUM_PORTFOLIO_FEATURES)
            assert env.action_space.shape == (1,)
            assert env.initial_equity == 10000.0

        def test_env_reset(mock_env_configs, sample_npz_data_for_env):
            """Testea el método reset."""
            env = TradingEnv(data_npz_path=sample_npz_data_for_env)
            obs, info = env.reset()
            
            assert obs.shape == env.observation_space.shape
            assert isinstance(info, dict)
            assert info['equity'] == env.initial_equity
            assert info['position'] == POSITION_NEUTRAL
            # current_step_index debería ser aleatorio si max_episode_steps_equals_dataset_length es False
            assert 0 <= info['current_step_index'] < env.total_steps_in_dataset

        def test_env_step_hold_action(mock_env_configs, sample_npz_data_for_env):
            """Testea un paso con acción de mantener (cercana a 0)."""
            env = TradingEnv(data_npz_path=sample_npz_data_for_env)
            env.reset(options={'start_index':0}) # Inicio fijo
            
            initial_equity = env.equity
            action = np.array([0.0], dtype=np.float32) # Acción Hold
            obs, reward, terminated, truncated, info = env.step(action)

            assert obs.shape == env.observation_space.shape
            assert info['position'] == POSITION_NEUTRAL # No debería abrir posición
            assert info['total_trades_episode'] == 0
            # La recompensa podría ser cercana a 0 si el equity no cambió mucho (solo por P&L no realizado si hubiera posición)
            # En este caso, sin posición, el equity no debería cambiar, recompensa = log(1) = 0.
            assert np.isclose(reward, 0.0) 
            assert info['equity'] == initial_equity # Equity no cambia si no hay P&L ni costes
            assert not terminated
            assert not truncated

        def test_env_step_open_long_and_close(mock_env_configs, sample_npz_data_for_env):
            """Testea abrir una posición larga y luego cerrarla."""
            env = TradingEnv(data_npz_path=sample_npz_data_for_env)
            env.reset(options={'start_index':0})
            
            initial_equity = env.equity
            action_open_long = np.array([0.5], dtype=np.float32) # Señal para abrir largo (> threshold 0.15)
            
            # --- Paso 1: Abrir Largo ---
            obs, reward_open, term_open, trunc_open, info_open = env.step(action_open_long)
            
            assert info_open['position'] == POSITION_LONG
            assert info_open['total_trades_episode'] == 1
            assert info_open['entry_price'] > 0 # Se estableció un precio de entrada
            # Equity debería haber disminuido por la comisión de apertura
            assert info_open['equity'] < initial_equity 
            commission_expected = (info_open['entry_price'] * info_open['position_size_contracts']) * env.taker_fee_rate
            assert np.isclose(initial_equity - info_open['equity'], commission_expected, atol=1e-4) # atol por float precision

            # --- Paso 2: Mantener (opcional, para ver P&L no realizado) ---
            # Asumir que el precio sube para P&L positivo
            # Esto es difícil de forzar en el test sin mockear _get_current_price de forma más compleja
            # Por ahora, solo hacemos un hold
            action_hold = np.array([0.0], dtype=np.float32)
            obs_hold, _, _, _, info_hold = env.step(action_hold)
            # El P&L no realizado y el equity deberían cambiar según el movimiento de precio mockeado en sample_npz_data_for_env

            # --- Paso 3: Cerrar Posición ---
            action_close = np.array([0.0], dtype=np.float32) # Señal neutral para cerrar posición existente
            obs_close, reward_close, term_close, trunc_close, info_close = env.step(action_close)

            assert info_close['position'] == POSITION_NEUTRAL
            assert info_close['total_trades_episode'] == 2 # Cierre cuenta como otra "media" operación, o la segunda parte.
                                                           # La implementación actual incrementa por cada _execute_trade con acción.
            # El P&L realizado del episodio debería reflejar el resultado del trade (neto de 2 comisiones)
            # Equity final vs equity después de la 1ra comisión.
            # Esto es complejo de verificar sin controlar el movimiento de precios.

        # Añadir más tests: abrir corto, liquidación, drawdown, fin de dataset.
        ```
      * **6.2. Ejecutar las Pruebas**:
        ```bash
        python -m pytest tests/environment/test_trading_env.py --cov=src/environment
        ```

-----

### Paso 7: Commit de los Cambios de la Fase 4

  * **Descripción Exhaustiva**: Añadir todos los cambios realizados durante esta fase al control de versiones Git.
  * **Acciones Específicas**:
      * **7.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add src/environment/trading_env.py
        git add src/environment/__init__.py
        git add scripts/test_environment.py
        git add tests/environment/test_trading_env.py
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 4: Implementar Módulo 3 (Entorno de Trading). Clase TradingEnv con reset, step, gestión de cartera, P&L, costes y recompensas. Incluye script de prueba y tests unitarios básicos."
        ```

**Fin de la Fase de Implementación 4.**

-----

**Nota sobre Deuda Técnica de Fase 3 (Módulo 2):**
Como se mencionó, es crucial que el Módulo 2 (`DataPreprocessor`) guarde los precios de cierre y valores ATR **no normalizados** (`env_step_close_prices`, `env_step_atr_values`) en el archivo `.npz`. Sin esto, el entorno no puede calcular correctamente el P\&L, slippage, y otras mecánicas que dependen de los valores de mercado brutos. Esta tarea debe abordarse antes de que el entrenamiento del agente sea significativo. Por ahora, el entorno `TradingEnv` tiene placeholders o advertencias para cuando estos datos faltan.


-----
## Fase de Implementación 5: Módulo 4 - Agente de Reinforcement Learning (SAC + Transformer)

**Nombre Descriptivo de la Fase:** Desarrollo del Agente SAC con Arquitectura de Política Basada en Transformer.

Esta fase se enfoca en implementar el Módulo 4. Desarrollaremos un agente de Aprendizaje por Refuerzo utilizando el algoritmo Soft Actor-Critic (SAC), aprovechando la biblioteca Stable Baselines3 (SB3). La característica distintiva de este agente será su política, que incorporará un codificador Transformer para procesar las secuencias de estado del mercado antes de alimentar las redes del Actor y los Críticos.

**Aclaración sobre las Características de Entrada del Agente (N\_features):**

  * El Módulo 3 (Entorno) proporciona una observación de forma `(L, N_effective_features)`, donde `L` es la longitud de la secuencia (ej. 96) y `N_effective_features = 28` (20 características de mercado + 8 características de portafolio replicadas).
  * El `README.md` (Módulo 4) menciona una entrada al Transformer de `(L, N_features=36)`. Dada la omisión de las 8 características del libro de órdenes, el Transformer operará con las `N_effective_features=28` disponibles. Los parámetros de configuración del Transformer (como `d_model`) se aplicarán a estas 28 características por paso de tiempo en la secuencia.

-----

### Paso 1: Creación de Archivos y Estructura para el Agente

  * **Descripción Exhaustiva**: Establecer la estructura de directorios y los archivos Python necesarios para el agente y sus componentes (modelo Transformer, política custom para SB3).
  * **Acciones Específicas**:
      * **1.1. Crear `src/agent/__init__.py`**:
        ```python
        # src/agent/__init__.py
        from .sac_agent_trainer import SACAgentTrainer
        from .policy import TransformerFeaturesExtractor # Si es la clase principal a exponer
        # from .models.transformer_model import TransformerEncoderModel # Si es necesario

        __all__ = ['SACAgentTrainer', 'TransformerFeaturesExtractor']
        ```
      * **1.2. Crear `src/agent/models/__init__.py`**:
        ```python
        # src/agent/models/__init__.py
        from .transformer_model import TransformerEncoderModel

        __all__ = ['TransformerEncoderModel']
        ```
      * **1.3. Crear Esqueleto de `src/agent/models/transformer_model.py`** (Definición del Transformer):
        ```python
        # src/agent/models/transformer_model.py
        import torch
        import torch.nn as nn
        import math
        import logging
        from typing import Tuple

        logger = logging.getLogger(__name__)

        class PositionalEncoding(nn.Module):
            # Implementación estándar de Positional Encoding Sinusoidal
            # ... (se detallará en el Paso 2) ...
            pass

        class TransformerEncoderModel(nn.Module):
            # Implementación del Transformer Encoder
            # ... (se detallará en el Paso 2) ...
            pass
        ```
      * **1.4. Crear Esqueleto de `src/agent/policy.py`** (Extractor de Características para SB3):
        ```python
        # src/agent/policy.py
        import gymnasium as gym
        import torch
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from typing import Dict, Any

        # from .models.transformer_model import TransformerEncoderModel # Se importará aquí

        class TransformerFeaturesExtractor(BaseFeaturesExtractor):
            # Implementación del extractor de características usando el TransformerEncoderModel
            # ... (se detallará en el Paso 3) ...
            pass
        ```
      * **1.5. Crear Esqueleto de `src/agent/sac_agent_trainer.py`** (Lógica del Agente y Entrenamiento con SB3):
        ```python
        # src/agent/sac_agent_trainer.py
        import logging
        from pathlib import Path
        from typing import Optional, Dict, Any

        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv # Para envolver el entorno
        from stable_baselines3.common.callbacks import BaseCallback # Para callbacks custom
        from gymnasium import Env # Para type hinting

        from config_loader import load_yaml_config
        # from .policy import TransformerFeaturesExtractor # Se importará aquí

        logger = logging.getLogger(__name__)

        class SACAgentTrainer:
            # Lógica para configurar, entrenar, guardar y cargar el modelo SAC de SB3
            # ... (se detallará en el Paso 4 y 5) ...
            pass
        ```

-----

### Paso 2: Implementación del Codificador Transformer (`transformer_model.py`)

  * **Descripción Exhaustiva**: Implementar la clase `TransformerEncoderModel` que contendrá la arquitectura del Transformer (capa de embedding de entrada, codificación posicional, y capas de Transformer Encoder de PyTorch).
  * **Acciones Específicas**:
      * **2.1. Implementar `PositionalEncoding` en `src/agent/models/transformer_model.py`**:
        ```python
        # En src/agent/models/transformer_model.py

        class PositionalEncoding(nn.Module):
            """Implementa la codificación posicional sinusoidal fija."""
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (d_model/2)
                
                pe = torch.zeros(max_len, 1, d_model) # (max_len, 1, d_model)
                pe[:, 0, 0::2] = torch.sin(position * div_term)
                pe[:, 0, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe) # No es un parámetro entrenable

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    x: Tensor, shape [seq_len, batch_size, embedding_dim]
                       (PyTorch Transformer espera batch_first=False por defecto)
                       O [batch_size, seq_len, embedding_dim] si batch_first=True en nn.Transformer layers.
                       Nuestra entrada al features extractor será (B, L, N_feat).
                       Después del embedding será (B, L, d_model).
                       El Transformer de PyTorch por defecto espera (L, B, d_model).
                       Así que podríamos necesitar permutar antes de pasar a nn.TransformerEncoder.
                """
                # Asumiendo que x llega como (batch, seq_len, d_model)
                # y self.pe es (max_len, 1, d_model) -> (max_len, d_model)
                # Necesitamos pe de (seq_len, d_model) para sumar a x (después de permutar x)
                # o pe de (1, seq_len, d_model) para sumar a x (batch_first).
                
                # Si x es (batch_size, seq_len, d_model)
                x = x + self.pe[:x.size(1), :].squeeze(1) # self.pe[:x.size(1), :] da (seq_len, 1, d_model)
                                                         # .squeeze(1) -> (seq_len, d_model)
                                                         # Esto se puede broadcast a (batch_size, seq_len, d_model)
                return self.dropout(x)
        ```
      * **2.2. Implementar `TransformerEncoderModel` en `src/agent/models/transformer_model.py`**:
        ```python
        # En src/agent/models/transformer_model.py (continuación)

        class TransformerEncoderModel(nn.Module):
            """
            Modelo Transformer Encoder para extraer características de secuencias de estado.
            """
            def __init__(self, 
                         input_features_dim: int,  # N_effective_features (ej. 28)
                         d_model: int,             # Dimensión del modelo Transformer (ej. 128)
                         n_heads: int,             # Número de cabezas de atención (ej. 4)
                         n_encoder_layers: int,    # Número de capas de Encoder (ej. 3)
                         dim_feedforward: int,     # Dimensión FFN interna del Encoder (ej. 512)
                         dropout: float = 0.1,
                         max_seq_len: int = 100    # Max L (ej. 96), para PositionalEncoding
                        ):
                super().__init__()
                self.d_model = d_model

                # 1. Capa de Embedding de Entrada: Proyecta N_features a d_model
                self.input_embedder = nn.Linear(input_features_dim, d_model)
                
                # 2. Codificación Posicional
                self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

                # 3. Capas de Transformer Encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True # IMPORTANTE: SB3 y la mayoría de los datos vienen como (batch, seq, feat)
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_encoder_layers
                )
                
                self.init_weights()
                logger.info(f"TransformerEncoderModel inicializado: input_dim={input_features_dim}, d_model={d_model}, "
                            f"n_heads={n_heads}, n_layers={n_encoder_layers}, dim_ffn={dim_feedforward}")

            def init_weights(self):
                """Inicializa los pesos."""
                initrange = 0.1
                self.input_embedder.weight.data.uniform_(-initrange, initrange)
                self.input_embedder.bias.data.zero_()
                # Los pesos de nn.TransformerEncoderLayer ya se inicializan bien por defecto.

            def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                """
                Forward pass del Transformer Encoder.

                Args:
                    src (torch.Tensor): Tensor de entrada, shape (batch_size, seq_len, input_features_dim)
                    src_mask (Optional[torch.Tensor]): Máscara para evitar atención a ciertos elementos (ej. padding).
                                                      Shape (seq_len, seq_len) o (batch_size * n_heads, seq_len, seq_len).
                                                      Para nuestro caso de secuencias de longitud fija L, usualmente no se necesita
                                                      una máscara a menos que haya padding dentro de las secuencias.
                                                      Por ahora, asumimos que no hay padding y no se usa src_mask.

                Returns:
                    torch.Tensor: Salida del Transformer, shape (batch_size, seq_len, d_model)
                """
                if src.dim() != 3:
                    raise ValueError(f"Entrada 'src' debe tener 3 dimensiones (batch, seq_len, features), obtuvo {src.dim()}")
                
                # 1. Embedding de entrada
                # src: (B, L, N_feat) -> embedded: (B, L, d_model)
                embedded = self.input_embedder(src) * math.sqrt(self.d_model) # Escalar por sqrt(d_model) es común
                
                # 2. Añadir codificación posicional
                # embedded: (B, L, d_model) -> pos_encoded: (B, L, d_model)
                pos_encoded = self.pos_encoder(embedded)
                
                # 3. Pasar por el Transformer Encoder
                # pos_encoded: (B, L, d_model) -> output: (B, L, d_model)
                # nn.TransformerEncoder con batch_first=True espera (B, L, d_model) y devuelve (B, L, d_model)
                output = self.transformer_encoder(pos_encoded, mask=src_mask) # `mask` es para atención causal si se requiere
                                                                             # `src_key_padding_mask` es para padding.
                                                                             # Por ahora, sin máscaras explícitas.
                return output
        ```

-----

### Paso 3: Implementación del Extractor de Características Custom para SB3 (`policy.py`)

  * **Descripción Exhaustiva**: Crear la clase `TransformerFeaturesExtractor` que hereda de `BaseFeaturesExtractor` de SB3. Esta clase usará el `TransformerEncoderModel` para procesar la observación y producir un vector de características aplanado para las redes del Actor y los Críticos.
  * **Acciones Específicas**:
      * **3.1. Implementar `TransformerFeaturesExtractor` en `src/agent/policy.py`**:
        ```python
        # src/agent/policy.py
        import gymnasium as gym
        import torch
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from typing import Dict, Any, Tuple # Tuple añadido

        from .models.transformer_model import TransformerEncoderModel # Importar el modelo Transformer

        class TransformerFeaturesExtractor(BaseFeaturesExtractor):
            """
            Extractor de características custom que usa un Transformer Encoder.
            Toma observaciones de forma (batch_size, seq_len, num_input_features)
            y produce un vector de características de forma (batch_size, features_dim).
            `features_dim` será igual a `d_model` del Transformer si se toma el output del último timestep.
            """
            def __init__(self, 
                         observation_space: gym.spaces.Box, 
                         # Parámetros para el TransformerEncoderModel, pasados desde policy_kwargs
                         d_model: int = 128,
                         n_heads: int = 4,
                         n_encoder_layers: int = 3,
                         dim_feedforward: int = 512, # Usualmente 4 * d_model
                         dropout: float = 0.1,
                         aggregation_method: str = "last" # "last", "mean"
                        ):
                
                # El `features_dim` que se pasa al constructor de BaseFeaturesExtractor
                # es la dimensión de la salida de este extractor.
                # Si agregamos tomando el último timestep, la salida es d_model.
                # Si agregamos por mean pooling, la salida también es d_model.
                super().__init__(observation_space, features_dim=d_model)

                if not isinstance(observation_space, gym.spaces.Box):
                    raise ValueError(f"TransformerFeaturesExtractor solo soporta gym.spaces.Box, "
                                     f"obtuvo {type(observation_space)}")
                if len(observation_space.shape) != 3: # Espera (L, N_effective_features) + batch dim
                    raise ValueError(f"Forma del espacio de observación debe ser (L, N_features), "
                                     f"obtuvo {observation_space.shape}")

                # L (seq_len) y N_effective_features (input_features_dim para Transformer)
                self.seq_len = observation_space.shape[1] # L
                self.num_input_features = observation_space.shape[2] # N_effective_features (ej. 28)
                
                self.aggregation_method = aggregation_method.lower()
                if self.aggregation_method not in ["last", "mean"]:
                    raise ValueError(f"Método de agregación desconocido: {aggregation_method}. Usar 'last' o 'mean'.")

                # Instanciar el modelo Transformer Encoder
                self.transformer_encoder = TransformerEncoderModel(
                    input_features_dim=self.num_input_features,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_encoder_layers=n_encoder_layers,
                    dim_feedforward=dim_feedforward if dim_feedforward else 4 * d_model,
                    dropout=dropout,
                    max_seq_len=self.seq_len + 10 # Un poco más por si acaso, aunque L es fijo aquí
                )
                
                self._d_model = d_model # Guardar d_model para referencia

                logger.info(f"TransformerFeaturesExtractor inicializado. Observación: (L={self.seq_len}, N_in_feat={self.num_input_features}). "
                            f"Salida del extractor (features_dim): {self._features_dim}. Agregación: {self.aggregation_method}.")


            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                """
                Forward pass del extractor.
                Args:
                    observations (torch.Tensor): Tensor de observaciones, 
                                                 shape (batch_size, seq_len, num_input_features)
                Returns:
                    torch.Tensor: Vector de características extraídas, shape (batch_size, self.features_dim)
                """
                # observations: (B, L, N_in_feat)
                # Salida del transformer: (B, L, d_model)
                transformer_output = self.transformer_encoder(observations) 

                # Agregar la salida del Transformer a un solo vector por muestra en el batch
                if self.aggregation_method == "last":
                    # Tomar la salida del último timestep de la secuencia
                    # transformer_output shape es (batch_size, seq_len, d_model)
                    extracted_features = transformer_output[:, -1, :] # (batch_size, d_model)
                elif self.aggregation_method == "mean":
                    # Media sobre la dimensión de la secuencia
                    extracted_features = transformer_output.mean(dim=1) # (batch_size, d_model)
                else: # No debería llegar aquí por el check en __init__
                    raise ValueError(f"Método de agregación no soportado: {self.aggregation_method}")
                
                if extracted_features.shape[1] != self._features_dim:
                     raise ValueError(f"Dimensión de salida del extractor ({extracted_features.shape[1]}) "
                                      f"no coincide con self.features_dim ({self._features_dim}).")

                return extracted_features
        ```

-----

### Paso 4: Implementación de la Lógica del Agente y Entrenamiento (`sac_agent_trainer.py`)

  * **Descripción Exhaustiva**: Crear la clase `SACAgentTrainer` que se encargará de configurar el modelo SAC de SB3 utilizando el `TransformerFeaturesExtractor` y los hiperparámetros definidos. Incluirá métodos para entrenar, guardar y cargar el modelo.
  * **Acciones Específicas**:
      * **4.1. Implementar `SACAgentTrainer` en `src/agent/sac_agent_trainer.py`**:
        ```python
        # En src/agent/sac_agent_trainer.py (continuación)
        from .policy import TransformerFeaturesExtractor # Importar el extractor

        class SACAgentTrainer:
            """
            Clase helper para entrenar un agente SAC con una política basada en Transformer usando Stable Baselines3.
            """
            def __init__(self, 
                         env: Env, # El entorno Gymnasium ya instanciado
                         log_dir: Union[str, Path] = "./results_host/sac_tensorboard_logs/",
                         model_save_dir: Union[str, Path] = "./results_host/trained_models/sac_transformer/"
                        ):
                self.env = env
                self.log_dir = Path(log_dir)
                self.model_save_dir = Path(model_save_dir)
                
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.model_save_dir.mkdir(parents=True, exist_ok=True)

                try:
                    self.agent_config = load_yaml_config("module4_agent_sac")
                    logger.info(f"Configuración del agente SAC cargada: {self.agent_config}")
                except Exception as e:
                    logger.error(f"Error al cargar la configuración del agente SAC: {e}", exc_info=True)
                    raise

                self.model: Optional[SAC] = None # El modelo SB3 SAC se inicializará en setup_model

            def setup_model(self, custom_objects: Optional[Dict[str, Any]] = None):
                """
                Configura e instancia el modelo SAC de Stable Baselines3.
                """
                logger.info("Configurando el modelo SAC de SB3...")
                
                # Parámetros para el TransformerFeaturesExtractor
                # Estos vienen de la config module4_agent_sac/params.yaml
                transformer_kwargs = {
                    "d_model": self.agent_config.get("d_model_transformer", 128),
                    "n_heads": self.agent_config.get("transformer_heads", 4),
                    "n_encoder_layers": self.agent_config.get("transformer_layers", 3),
                    "dim_feedforward": self.agent_config.get("d_model_transformer", 128) * 4, # Asumir 4*d_model si no está explícito
                    "dropout": self.agent_config.get("transformer_dropout", 0.1),
                    "aggregation_method": self.agent_config.get("transformer_aggregation", "last")
                }
                if "dim_feedforward_transformer" in self.agent_config: # Permitir override
                    transformer_kwargs["dim_feedforward"] = self.agent_config["dim_feedforward_transformer"]


                # policy_kwargs para SB3:
                # - features_extractor_class: Nuestra clase custom.
                # - features_extractor_kwargs: Args para el constructor de nuestra clase.
                # - net_arch: Define las capas MLP para Actor y Críticos DESPUÉS del extractor.
                #   Si es una lista simple como [256, 256], se usa para ambos.
                #   O un dict: dict(pi=[256, 256], qf=[256, 256])
                policy_kwargs = dict(
                    features_extractor_class=TransformerFeaturesExtractor,
                    features_extractor_kwargs=transformer_kwargs,
                    net_arch=self.agent_config.get("actor_critic_hidden_dims", [256, 256]) # ej. [256, 256]
                    # (Opcional) Se pueden añadir normalización de features de observación aquí si es necesario
                    # normalize_images=False (default)
                    # share_features_extractor=True (default, actor y crítico comparten el extractor)
                )

                # Hiperparámetros de SAC
                sac_hyperparams = {
                    "learning_rate": self.agent_config.get("learning_rate", 0.0003),
                    "buffer_size": self.agent_config.get("buffer_size", 100000),
                    "batch_size": self.agent_config.get("batch_size", 256),
                    "gamma": self.agent_config.get("gamma", 0.99),
                    "tau": self.agent_config.get("tau", 0.005),
                    "train_freq": tuple(self.agent_config.get("train_freq", [1, "step"])), # ej. (1, "step") o 4
                    "gradient_steps": self.agent_config.get("gradient_steps", 1), # -1 para igual a train_freq
                    "ent_coef": self.agent_config.get("ent_coef", 'auto'), # 'auto' o un float
                    "learning_starts": self.agent_config.get("learning_starts", 1000),
                    "use_sde": self.agent_config.get("use_sde", False), # State-Dependent Exploration
                    # "sde_sample_freq": -1 (default)
                    # "target_entropy": 'auto' (default)
                    # "target_update_interval": 1 (default)
                }
                
                # Envolver el entorno en DummyVecEnv si no es ya un VecEnv
                # SB3 espera un VecEnv.
                if not isinstance(self.env, gym.vector.いわゆるVectorEnv): # No usar isinstance con str
                     vec_env = DummyVecEnv([lambda: self.env])
                else:
                     vec_env = self.env


                self.model = SAC(
                    policy="MlpPolicy", # Se usa MlpPolicy pero se le pasa el extractor custom
                    env=vec_env, 
                    policy_kwargs=policy_kwargs,
                    verbose=self.agent_config.get("sb3_verbose", 1), # 0: nada, 1: progreso, 2: debug
                    tensorboard_log=str(self.log_dir), # Ruta para logs de TensorBoard
                    seed=self.agent_config.get("seed", None), # Para reproducibilidad
                    device=self.agent_config.get("device", "auto"), # "cpu", "cuda", "auto"
                    **sac_hyperparams # Desempaquetar el resto de hiperparámetros
                )
                
                # Para cargar custom objects (como el extractor) al cargar un modelo guardado
                if custom_objects:
                    self.model.custom_objects = custom_objects
                else: # Registrar el extractor para que se pueda cargar después
                    self.model.custom_objects = {
                        "policy_kwargs": policy_kwargs # Esto puede no ser suficiente, SB3 a veces necesita más para cargar.
                                                      # Es mejor pasar la clase directamente si es posible.
                                                      # O guardar el modelo de forma que incluya todo.
                    }


                logger.info("Modelo SAC de SB3 configurado e instanciado.")
                logger.info(f"Política del modelo:\n{self.model.policy}")

            def train(self, total_timesteps: int, 
                      callback: Optional[BaseCallback] = None, 
                      log_interval: int = 10, # Frecuencia de logueo de progreso
                      tb_log_name: str = "SAC_Transformer_Run",
                      reset_num_timesteps: bool = True
                     ):
                """
                Entrena el modelo SAC.
                """
                if self.model is None:
                    logger.error("El modelo no está configurado. Llama a setup_model() primero.")
                    return

                logger.info(f"Iniciando entrenamiento del modelo SAC por {total_timesteps} timesteps...")
                try:
                    self.model.learn(
                        total_timesteps=total_timesteps,
                        callback=callback,
                        log_interval=log_interval, # Cada cuántos episodios loguear info
                        tb_log_name=tb_log_name, # Nombre para la run en TensorBoard
                        reset_num_timesteps=reset_num_timesteps, # Si es True, el contador de timesteps del modelo se resetea
                        progress_bar=True # Mostrar barra de progreso
                    )
                    logger.info("Entrenamiento completado.")
                except Exception as e:
                    logger.error(f"Error durante el entrenamiento del modelo: {e}", exc_info=True)
                    raise

            def save_model(self, model_name: str = "sac_transformer_final"):
                """Guarda el modelo entrenado."""
                if self.model is None:
                    logger.error("No hay modelo para guardar.")
                    return
                
                save_path = self.model_save_dir / f"{model_name}.zip"
                try:
                    self.model.save(save_path)
                    logger.info(f"Modelo guardado exitosamente en: {save_path}")
                    # Opcional: guardar replay buffer también si es necesario para continuar entrenamiento
                    # self.model.save_replay_buffer(self.model_save_dir / f"{model_name}_replay_buffer.pkl")
                except Exception as e:
                    logger.error(f"Error al guardar el modelo en {save_path}: {e}", exc_info=True)

            @classmethod
            def load_model(cls, model_path: Union[str, Path], env: Optional[Env] = None, 
                           custom_feature_extractor_class: Optional[type] = None,
                           custom_feature_extractor_kwargs: Optional[Dict[str,Any]] = None):
                """
                Carga un modelo SAC entrenado.
                Es crucial que el custom_objects se configure correctamente si se usó un extractor custom.
                """
                model_path = Path(model_path)
                if not model_path.exists():
                    logger.error(f"Archivo de modelo no encontrado en: {model_path}")
                    raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

                logger.info(f"Cargando modelo SAC desde: {model_path}")
                custom_objects_for_load = {}
                if custom_feature_extractor_class:
                    # Esto es necesario para que SB3 pueda reconstruir la política
                    # policy_kwargs = {"features_extractor_class": custom_feature_extractor_class}
                    # if custom_feature_extractor_kwargs:
                    #     policy_kwargs["features_extractor_kwargs"] = custom_feature_extractor_kwargs
                    # custom_objects_for_load["policy_kwargs"] = policy_kwargs
                    
                    # Mejor práctica: pasar el espacio de observación y acción si es posible,
                    # y el SB3 lo reconstruirá. A veces, se necesita registrar la clase:
                    custom_objects_for_load = {
                        "observation_space": env.observation_space if env else None,
                        "action_space": env.action_space if env else None,
                        # SB3 puede necesitar que el nombre de la clase se mapee a la clase real.
                        # 'TransformerFeaturesExtractor': TransformerFeaturesExtractor # Esto se hace globalmente con register_namespace
                        # O se pasa a través de policy_kwargs dentro de custom_objects al cargar.
                    }
                    if custom_feature_extractor_kwargs and custom_feature_extractor_class : # Para reconstruir el extractor
                         policy_kwargs = dict(
                            features_extractor_class=custom_feature_extractor_class, # La clase en sí
                            features_extractor_kwargs=custom_feature_extractor_kwargs
                         )
                         custom_objects_for_load["policy_kwargs"] = policy_kwargs


                try:
                    # Si el entorno (env) se pasa, SB3 puede inferir espacios.
                    # Si no, necesita los custom_objects para reconstruir la política correctamente.
                    loaded_model = SAC.load(model_path, env=env, custom_objects=custom_objects_for_load if custom_objects_for_load else None)
                    logger.info("Modelo cargado exitosamente.")
                    
                    # Crear una instancia de SACAgentTrainer para devolver (opcional, o solo devolver el modelo SB3)
                    # trainer = cls(env=env if env else loaded_model.get_env()) # Cuidado, get_env() puede ser None
                    # trainer.model = loaded_model
                    # return trainer
                    return loaded_model # Devolver directamente el modelo SB3

                except Exception as e:
                    logger.error(f"Error al cargar el modelo desde {model_path}: {e}", exc_info=True)
                    raise
            
            def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
                """
                Realiza una predicción usando el modelo entrenado.
                Args:
                    observation: La observación del entorno.
                    deterministic: Si la acción debe ser determinista (True) o estocástica (False).
                Returns:
                    Tuple de (acción, None) ya que SAC usualmente no devuelve estados recurrentes.
                """
                if self.model is None:
                    logger.error("El modelo no está cargado o configurado. No se puede predecir.")
                    return np.array([0.0]), None # Acción neutral por defecto

                # Asegurar que la observación tenga el shape esperado por el modelo (batch_size, ...)
                # El modelo SB3 espera un batch, incluso si es de 1.
                # Si la obs es (L, N_feat), necesita ser (1, L, N_feat)
                if observation.ndim == self.model.observation_space. सोपान: # .rank para TF, .shape para gym.spaces
                    #  self.model.observation_space es el espacio del VecEnv, puede ser diferente
                    #  del env.observation_space si hay wrappers.
                    #  Es más seguro chequear con el espacio original del entorno.
                    obs_shape_expected_no_batch = self.env.observation_space.shape
                    if observation.shape == obs_shape_expected_no_batch:
                        observation = np.expand_dims(observation, axis=0)
                    else:
                        logger.error(f"Shape de observación para predicción ({observation.shape}) no coincide con esperado sin batch ({obs_shape_expected_no_batch}) ni con batch.")
                        # Podría intentar reformar si es un error común, o fallar.

                action, _states = self.model.predict(observation, deterministic=deterministic)
                return action, _states

        ```

-----

### Paso 5: Script de Entrenamiento del Agente (`scripts/train_agent.py`)

  * **Descripción Exhaustiva**: Crear un script que instancie el `TradingEnv` (Módulo 3) y el `SACAgentTrainer` (Módulo 4). Este script cargará los datos preprocesados, configurará el agente y comenzará el proceso de entrenamiento, guardando finalmente el modelo entrenado.
  * **Acciones Específicas**:
      * **5.1. Crear `scripts/train_agent.py`**:
        ```python
        # scripts/train_agent.py
        import sys
        import logging
        from pathlib import Path
        import argparse # Para argumentos de línea de comandos

        # Añadir src al PYTHONPATH
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from common.utils import setup_logging
        from environment.trading_env import TradingEnv
        from agent.sac_agent_trainer import SACAgentTrainer
        from config_loader import load_yaml_config, get_env_variable
        from agent.policy import TransformerFeaturesExtractor # Importante para cargar modelos guardados

        # (Opcional) Callbacks para SB3
        # from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

        try:
            setup_logging() # Configurar logging globalmente al inicio
        except Exception as e:
            logging.basicConfig(level=logging.ERROR) # Fallback
            logging.critical(f"Fallo CRÍTICO al configurar logging: {e}", exc_info=True)

        logger = logging.getLogger(__name__)

        def main(args):
            logger.info("========================================================")
            logger.info("Iniciando script de entrenamiento del Agente SAC...")
            logger.info("========================================================")
            logger.info(f"Argumentos recibidos: {args}")

            try:
                # --- 1. Cargar Configuraciones ---
                agent_cfg = load_yaml_config("module4_agent_sac")
                env_cfg = load_yaml_config("module3_environment") # Podría ser necesario para configurar el env
                preproc_cfg = load_yaml_config("module2_preprocessing")
                data_acq_cfg = load_yaml_config("module1_data_acquisition")

                # --- 2. Determinar la ruta del archivo de datos preprocesados ---
                trading_pair = data_acq_cfg.get("trading_pair", "BTCUSDT")
                kline_interval = data_acq_cfg.get("kline_interval", "15m")
                sequence_L = preproc_cfg.get("sequence_length_L", 96)
                
                # N_features en el nombre del archivo .npz es el N_market_features (ej. 20)
                # proveniente del Módulo 2.
                n_market_features_from_m2 = 20 # Según diseño de M2 y M3.
                
                processed_filename = f"{trading_pair}_{kline_interval}_L{sequence_L}_N{n_market_features_from_m2}_processed_sequences.npz"
                
                data_dir_str = get_env_variable("DATA_DIR_HOST_FOR_APP", "/app/data_persistent")
                processed_data_base_path = Path(data_dir_str) / "processed" / trading_pair / kline_interval
                npz_file_path = processed_data_base_path / processed_filename

                logger.info(f"Usando archivo de datos preprocesados: {npz_file_path}")
                if not npz_file_path.exists():
                    logger.error(f"Archivo de datos preprocesados NO encontrado: {npz_file_path}. "
                                 "Por favor, ejecuta el script de preprocesamiento (Fase 3) primero.")
                    return

                # --- 3. Inicializar el Entorno de Trading ---
                # El entorno se usará para entrenamiento, así que debería tener inicios aleatorios (si se configuró así)
                train_env = TradingEnv(data_npz_path=npz_file_path)
                # train_env.set_train_mode() # Asegurar modo entrenamiento si hay lógica de eval/train separada

                # --- 4. Inicializar el Entrenador del Agente SAC ---
                # Rutas para logs y modelos guardados (pueden venir de args o config)
                log_dir_base = Path(args.log_dir if args.log_dir else agent_cfg.get("tensorboard_log_path", "./results_host/sac_tensorboard_logs/"))
                model_save_dir_base = Path(args.model_save_dir if args.model_save_dir else agent_cfg.get("model_save_path", "./results_host/trained_models/sac_transformer/"))
                
                # Crear un nombre de run único para TensorBoard y guardado de modelos
                run_name = f"SAC_Transformer_{trading_pair}_{kline_interval}_L{sequence_L}_N{train_env.total_features_in_obs}_{Path(npz_file_path).stem}_{int(time.time())}"
                
                agent_trainer = SACAgentTrainer(
                    env=train_env,
                    log_dir = log_dir_base / run_name,
                    model_save_dir = model_save_dir_base / run_name
                )

                # --- 5. Configurar el Modelo SAC ---
                # (Opcional) Pasar custom_objects si se planea cargar modelos que lo requieran explícitamente
                # custom_objects_for_setup = {
                #     "policy_kwargs": {
                #         "features_extractor_class": TransformerFeaturesExtractor,
                #         "features_extractor_kwargs": { # Estos deberían ser los mismos que en setup_model
                #                "d_model": agent_cfg.get("d_model_transformer", 128),
                #                # ... otros kwargs del extractor ...
                #         }
                #     }
                # }
                agent_trainer.setup_model() # custom_objects se maneja internamente si es simple

                # --- 6. (Opcional) Definir Callbacks para SB3 ---
                # Ejemplo: CheckpointCallback para guardar el modelo periódicamente
                # checkpoint_callback = CheckpointCallback(
                #     save_freq=max(args.save_freq // num_cpu, 1), # num_cpu si se usa SubprocVecEnv
                #     save_path=str(agent_trainer.model_save_dir / "checkpoints/"),
                #     name_prefix="sac_checkpoint"
                # )
                # Ejemplo: EvalCallback para evaluación periódica (requiere un env de evaluación separado)
                # eval_env = TradingEnv(data_npz_path=npz_file_path_eval) # Necesitaría datos de eval
                # eval_callback = EvalCallback(eval_env, best_model_save_path=str(agent_trainer.model_save_dir / "best_model/"),
                #                            log_path=str(agent_trainer.log_dir / "eval_logs/"), eval_freq=args.eval_freq,
                #                            deterministic=True, render=False)
                # callbacks_list = [checkpoint_callback] # Añadir eval_callback si se usa


                # --- 7. Entrenar el Modelo ---
                total_timesteps_to_train = args.total_timesteps if args.total_timesteps else agent_cfg.get("total_training_timesteps", 1_000_000)
                
                agent_trainer.train(
                    total_timesteps=total_timesteps_to_train,
                    # callback=callbacks_list if callbacks_list else None,
                    tb_log_name=run_name # Nombre de la run en TensorBoard (ya se pasó al constructor de SAC)
                                         # o "SAC_Transformer" como nombre general de la métrica.
                )

                # --- 8. Guardar el Modelo Final Entrenado ---
                final_model_name = f"sac_transformer_final_{total_timesteps_to_train}steps"
                agent_trainer.save_model(model_name=final_model_name)

                logger.info(f"Entrenamiento y guardado del modelo '{final_model_name}' completados.")
                logger.info(f"Logs de TensorBoard en: {agent_trainer.log_dir}")
                logger.info(f"Modelos guardados en: {agent_trainer.model_save_dir}")

            except FileNotFoundError as e_fnf:
                logger.error(f"Error de archivo no encontrado: {e_fnf}", exc_info=True)
            except EnvironmentError as e_env:
                logger.error(f"Error de configuración de entorno: {e_env}", exc_info=True)
            except Exception as e:
                logger.error(f"Error inesperado durante el script de entrenamiento: {e}", exc_info=True)
            finally:
                logger.info("========================================================")
                logger.info("Finalización del script de entrenamiento del Agente.")
                logger.info("========================================================")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description="Script de entrenamiento para el Agente SAC con Transformer.")
            parser.add_argument("--total_timesteps", type=int, default=None,
                                help="Número total de timesteps para entrenar. Sobreescribe config.")
            parser.add_argument("--log_dir", type=str, default=None,
                                help="Directorio base para logs de TensorBoard. Sobreescribe config.")
            parser.add_argument("--model_save_dir", type=str, default=None,
                                help="Directorio base para guardar modelos entrenados. Sobreescribe config.")
            # Añadir más argumentos si son necesarios (ej. save_freq, eval_freq, etc.)
            
            script_args = parser.parse_args()
            main(script_args)
        ```
      * **5.2. Añadir Parámetros de Rutas y Entrenamiento a `config/module4_agent_sac/params.yaml`**:
        ```yaml
        # En config/module4_agent_sac/params.yaml (añadir/actualizar)
        # ... (parámetros de arquitectura y SAC ya definidos) ...

        total_training_timesteps: 1000000 # Ejemplo, ajustar según necesidad
        sb3_verbose: 1 # Nivel de verbosidad de SB3 (0, 1, o 2)
        seed: null # Semilla para reproducibilidad (null para aleatorio, o un entero)
        device: "auto" # "cpu", "cuda", "auto"
        transformer_dropout: 0.1
        transformer_aggregation: "last" # "last" o "mean"

        # Rutas (pueden ser sobreescritas por argumentos de línea de comando)
        tensorboard_log_path: "./results_host/sac_tensorboard_logs/"
        model_save_path: "./results_host/trained_models/sac_transformer/"

        # (Opcional) Parámetros para callbacks
        # save_freq_timesteps: 50000 # Frecuencia para CheckpointCallback
        # eval_freq_timesteps: 25000 # Frecuencia para EvalCallback
        ```
      * **5.3. Ejecutar el Script de Entrenamiento (Prueba Corta)**:
          * Asegurar que el entorno (M3) y los datos preprocesados (M2) funcionan.
          * Modificar `total_training_timesteps` en `params.yaml` o pasar `--total_timesteps` con un valor pequeño (ej. 1000-5000) para una prueba rápida.
          * Ejecutar dentro de Docker:
            ```bash
            # docker-compose up -d --build workhorse_app redis # Si no está corriendo
            docker-compose exec workhorse_app python scripts/train_agent.py --total_timesteps 2000 
            ```
          * Monitorear la salida. Debería mostrar el progreso de SB3 y los logs.
          * Verificar la creación de logs de TensorBoard en `results_host/sac_tensorboard_logs/` y el modelo guardado en `results_host/trained_models/`.
          * Iniciar TensorBoard para ver los logs: `tensorboard --logdir ./results_host/sac_tensorboard_logs/` (ejecutar en el host, apuntando al directorio mapeado).

-----

### Paso 6: Pruebas Unitarias (Básicas)

  * **Descripción Exhaustiva**: Crear pruebas unitarias para los componentes del agente, como `TransformerEncoderModel` (verificar shapes de salida) y `TransformerFeaturesExtractor`. Probar el flujo de entrenamiento completo es más un test de integración y puede ser complejo de unitarizar completamente sin muchos mocks.
  * **Acciones Específicas**:
      * **6.1. Crear `tests/agent/models/test_transformer_model.py`**:
        ```python
        # tests/agent/models/test_transformer_model.py
        import torch
        import pytest
        from src.agent.models.transformer_model import TransformerEncoderModel, PositionalEncoding

        def test_positional_encoding():
            d_model = 128
            max_len = 50
            seq_len = 20
            batch_size = 4
            
            pe = PositionalEncoding(d_model, max_len=max_len)
            dummy_input = torch.zeros(batch_size, seq_len, d_model) # (B, L, d_model)
            output = pe(dummy_input)
            
            assert output.shape == (batch_size, seq_len, d_model)
            # Verificar que algo se añadió (no todos ceros si input era ceros)
            assert not torch.all(output == 0).item() 

        def test_transformer_encoder_model_output_shape():
            batch_size = 4
            seq_len = 96 # L
            input_features = 28 # N_effective_features
            d_model_test = 64 # Usar d_model más pequeño para test
            n_heads_test = 2
            n_layers_test = 1
            dim_ffn_test = 128

            model = TransformerEncoderModel(
                input_features_dim=input_features,
                d_model=d_model_test,
                n_heads=n_heads_test,
                n_encoder_layers=n_layers_test,
                dim_feedforward=dim_ffn_test,
                max_seq_len=seq_len + 10
            )
            
            dummy_src = torch.rand(batch_size, seq_len, input_features) # (B, L, N_in)
            output = model(dummy_src) # (B, L, d_model)
            
            assert output.shape == (batch_size, seq_len, d_model_test)
        ```
      * **6.2. Crear `tests/agent/test_policy.py`**:
        ```python
        # tests/agent/test_policy.py
        import torch
        import pytest
        import gymnasium as gym
        from gymnasium import spaces

        from src.agent.policy import TransformerFeaturesExtractor
        # from src.agent.models.transformer_model import TransformerEncoderModel # Ya testeado

        def test_transformer_features_extractor_output_shape():
            seq_len = 96
            num_input_features = 28 # N_effective_features
            d_model_test = 64 # Dimensión de salida del extractor
            
            # Espacio de observación como lo esperaría el extractor
            obs_space = spaces.Box(low=-1.0, high=1.0, shape=(1, seq_len, num_input_features), dtype=np.float32)
            # El extractor de SB3 recibe (N_envs, L, N_feat) pero el obs_space se define como (L,N_feat) para un solo env.
            # Al construirlo, el constructor de BaseFeaturesExtractor toma obs_space.shape, que es (1, L, N_feat).
            # El forward recibe (batch_size, L, N_feat).
            # Corregimos: obs_space debe ser (L, N_feat)
            single_env_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, num_input_features), dtype=np.float32)


            extractor = TransformerFeaturesExtractor(
                observation_space=single_env_obs_space, # Pasar el espacio de un solo entorno
                d_model=d_model_test,
                n_heads=2,
                n_encoder_layers=1,
                dim_feedforward=128,
                aggregation_method="last"
            )
            
            assert extractor.features_dim == d_model_test # La salida del extractor debe ser d_model

            batch_size = 4
            # Observaciones de entrada para el forward pass del extractor
            dummy_observations = torch.rand(batch_size, seq_len, num_input_features)
            
            extracted_features = extractor(dummy_observations) # (B, features_dim)
            
            assert extracted_features.shape == (batch_size, d_model_test)
        ```
      * **6.3. Ejecutar las Pruebas**:
        ```bash
        python -m pytest tests/agent/models/test_transformer_model.py tests/agent/test_policy.py --cov=src/agent
        ```

-----

### Paso 7: Commit de los Cambios de la Fase 5

  * **Descripción Exhaustiva**: Añadir todos los cambios realizados durante esta fase al control de versiones Git.
  * **Acciones Específicas**:
      * **7.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add src/agent/models/transformer_model.py src/agent/models/__init__.py
        git add src/agent/policy.py
        git add src/agent/sac_agent_trainer.py src/agent/__init__.py
        git add scripts/train_agent.py
        git add config/module4_agent_sac/params.yaml # Por los nuevos parámetros
        git add tests/agent/models/test_transformer_model.py tests/agent/test_policy.py
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 5: Implementar Módulo 4 (Agente SAC+Transformer). Modelo Transformer, Extractor de Features para SB3, Lógica de Agente SAC y script de entrenamiento. Incluye tests."
        ```

**Fin de la Fase de Implementación 5.**

-----

El agente SAC con la arquitectura basada en Transformer está ahora definido y listo para ser entrenado. La integración con Stable Baselines3 permite flexibilidad y el uso de un algoritmo robusto.

**Recordatorio sobre Deuda Técnica de Fase 3 (Módulo 2):**
Sigue siendo crucial que el Módulo 2 (`DataPreprocessor`) guarde los precios de cierre y valores ATR **no normalizados** (`env_step_close_prices`, `env_step_atr_values`) en el archivo `.npz` para el correcto funcionamiento del `TradingEnv` (Módulo 3).

Ahora que tenemos el agente, el siguiente paso lógico sería finalizar la orquestación con Docker para asegurar que todo el pipeline (descarga, preprocesamiento, entrenamiento) se pueda ejecutar de forma contenida, y luego implementar el framework de backtesting (WFO).

----

## Fase de Implementación 6: Módulo 10 - Orquestación con Docker (MVP - Finalización)

**Nombre Descriptivo de la Fase:** Consolidación del Entorno Dockerizado y Creación de un Pipeline de Ejecución.

Esta fase se centra en finalizar y validar la configuración de Docker y Docker Compose (Módulo 10) para asegurar que toda la aplicación (`workhorse_app`) y sus dependencias se puedan construir y ejecutar de manera fiable y reproducible. También se creará un script principal para orquestar el flujo de trabajo del MVP (descarga, preprocesamiento, y entrenamiento del agente) dentro del entorno Docker.

-----

### Paso 1: Revisión y Finalización del `Dockerfile`

  * **Descripción Exhaustiva**: Revisar el `Dockerfile` creado en la Fase 0 para asegurar que todas las dependencias del sistema y de Python estén correctamente especificadas, que la imagen se construya eficientemente y que el entorno del contenedor sea adecuado para ejecutar los scripts del proyecto.
  * **Acciones Específicas**:
      * **1.1. Revisar Dependencias del Sistema**:
          * Confirmar que `libta-lib0` (o el método de instalación de TA-Lib elegido) es robusto. En Fase 0, se usó `libta-lib0`. Esto es generalmente adecuado para imágenes basadas en Debian/Ubuntu.
          * Asegurar que `build-essential`, `wget`, `unzip` (si se usara para descargar algo manualmente, aunque no es el caso actual para TA-Lib si se usa `libta-lib0`) y otras dependencias necesarias para las bibliotecas de Python estén presentes. La lista actual en el `Dockerfile` de Fase 0 parece un buen punto de partida.
      * **1.2. Optimización de Capas (Opcional para MVP)**:
          * Para MVP, la optimización extensiva de capas no es crítica, pero combinar múltiples comandos `RUN apt-get install` y limpiar cachés de `apt` en la misma capa (`&& rm -rf /var/lib/apt/lists/*`) ya es una buena práctica implementada.
      * **1.3. Usuario No-Root (Opcional Avanzado para MVP)**:
          * Por seguridad, en entornos de producción se recomienda ejecutar aplicaciones con un usuario no-root. Para el MVP de backtesting, esto es menos crítico pero es una buena práctica a considerar para el futuro.
          * *Acción para MVP*: Mantener la ejecución como root por simplicidad, pero añadir un comentario sobre esto para futuras mejoras.
            ```dockerfile
            # ... (después de instalar dependencias) ...
            # WORKDIR /app

            # COPY requirements.txt requirements.txt
            # RUN pip install --no-cache-dir -r requirements.txt

            # (Opcional - Mejora Futura) Crear y usar un usuario no-root
            # RUN useradd -ms /bin/bash appuser
            # USER appuser
            # WORKDIR /home/appuser/app # Ajustar WORKDIR y rutas de copia si se usa appuser

            # Por ahora, continuamos como root para simplicidad del MVP.
            # ...
            ```
      * **1.4. `WORKDIR` y `CMD`**:
          * `WORKDIR /app` está bien.
          * `CMD ["tail", "-f", "/dev/null"]` es adecuado para mantener el contenedor `workhorse_app` corriendo y permitir `docker-compose exec` para ejecutar scripts. Esto ya está en el `Dockerfile` de Fase 0.
      * **1.5. Revisar `requirements.txt` y `pip install`**:
          * Confirmar que `requirements.txt` está completo.
          * El comando `RUN pip install --no-cache-dir -r requirements.txt` es eficiente.
      * **1.6. Comentario sobre Copia de Código Fuente en `Dockerfile`**:
          * En Fase 0, la copia de `src`, `scripts`, `config` se comentó en el `Dockerfile` porque se maneja mediante volúmenes en `docker-compose.yml` para desarrollo. Esto es correcto. Para una imagen de "producción" o independiente, se descomentarían estas líneas `COPY`.
      * **No se requieren cambios significativos al `Dockerfile` de Fase 0 para este paso, solo revisión y confirmación.**

-----

### Paso 2: Revisión y Finalización del `docker-compose.yml`

  * **Descripción Exhaustiva**: Revisar el archivo `docker-compose.yml` para confirmar que los servicios (`workhorse_app`, `redis`), volúmenes, variables de entorno y dependencias entre servicios están configurados correctamente para el flujo de trabajo del MVP.
  * **Acciones Específicas**:
      * **2.1. Definiciones de Servicios**:
          * **`redis`**: La configuración actual con `redis:7.2-alpine`, mapeo de puertos, volumen persistente opcional (`redis_data`) y `healthcheck` es robusta.
          * **`workhorse_app`**:
              * `build: context: ., dockerfile: Dockerfile` es correcto.
              * `env_file: - .env` carga las variables del archivo `.env`.
              * `environment: PYTHONPATH: "/app/src:/app"` es crucial para que los scripts puedan importar módulos de `src/`. Esta ruta asume que `src` y `scripts` se montan directamente bajo `/app`.
                  * *Revisión*: Si los volúmenes montan `src` en `/app/src` y `scripts` en `/app/scripts` (como se configuró en Fase 0), entonces `PYTHONPATH="/app/src:/app/scripts"` podría ser más preciso, o simplemente `/app` si los scripts importan como `from src...`. El actual `PYTHONPATH: "/app/src:/app"` permitiría `import src.module` y `import scripts.module` desde `/app` como CWD, o `import module` si CWD es `/app/src` o `/app/scripts`. La configuración de `working_dir: /app/scripts` hace que los imports desde `/app/src` necesiten que `/app` (o `/app/src`) esté en `PYTHONPATH`.
                  * **Acción**: Mantener `PYTHONPATH: "/app/src:/app"`. Esto debería cubrir los casos si los scripts en `/app/scripts` importan `from src...` o si se trabaja con CWD en `/app`.
      * **2.2. Montaje de Volúmenes**:
          * Confirmar las rutas de montaje para desarrollo en vivo:
            ```yaml
            volumes:
              - ${SRC_DIR_HOST:-./src}:/app/src
              - ${CONFIG_DIR_HOST:-./config}:/app/config
              - ${SCRIPTS_DIR_HOST:-./scripts}:/app/scripts
              - ${DATA_DIR_HOST:-./data_host}:${DATA_DIR_HOST_FOR_APP:-/app/data_persistent}
              - ${RESULTS_DIR_HOST:-./results_host}:${RESULTS_DIR_HOST_FOR_APP:-/app/results_persistent} # Añadido RESULTS_DIR_HOST_FOR_APP
            ```
          * En Fase 0, se usó `/app/data_persistent` y `/app/results_persistent`. Esto es consistente.
          * *Añadir mapeo para `results_host` en `.env.example` y `docker-compose.yml` si no está explícitamente con `RESULTS_DIR_HOST_FOR_APP`*. (Ya estaba implícito, pero hacerlo explícito en `.env.example` es bueno).
              * Actualizar `.env.example`:
            <!-- end list -->
            ```env
            # ...
            RESULTS_DIR_HOST_FOR_APP="/app/results_persistent" # Ruta de resultados DENTRO del contenedor
            # ...
            ```
      * **2.3. `depends_on` y `working_dir`**:
          * `depends_on: redis: condition: service_healthy` para `workhorse_app` es correcto.
          * `working_dir: /app/scripts` para `workhorse_app` es una buena elección, ya que la mayoría de las ejecuciones serán scripts.
      * **No se requieren cambios estructurales importantes al `docker-compose.yml` de Fase 0, principalmente confirmaciones y asegurar consistencia con `.env` para las rutas internas del contenedor.**

-----

### Paso 3: Creación de un Script de Orquestación Principal (`scripts/run_pipeline.py`)

  * **Descripción Exhaustiva**: Crear un script principal `scripts/run_pipeline.py` que orqueste la ejecución secuencial de los scripts de las fases anteriores (descarga de datos, preprocesamiento, entrenamiento del agente). Esto simplificará la ejecución de todo el flujo de trabajo del MVP.
  * **Acciones Específicas**:
      * **3.1. Crear `scripts/run_pipeline.py`**:
        ```python
        # scripts/run_pipeline.py
        import sys
        import logging
        from pathlib import Path
        import subprocess # Para llamar a otros scripts
        import argparse

        # Añadir src al PYTHONPATH (aunque los sub-scripts también lo hacen, es bueno tenerlo aquí)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from common.utils import setup_logging # Para configurar logging una vez

        try:
            setup_logging()
        except Exception as e:
            logging.basicConfig(level=logging.ERROR)
            logging.critical(f"Fallo CRÍTICO al configurar logging en pipeline: {e}", exc_info=True)

        logger = logging.getLogger("PipelineOrchestrator") # Logger específico para el pipeline

        # Definir las rutas a los scripts que se van a ejecutar
        # Asumimos que este script está en la carpeta 'scripts/'
        SCRIPTS_DIR = Path(__file__).resolve().parent
        DOWNLOAD_SCRIPT = SCRIPTS_DIR / "download_data.py"
        PREPROCESS_SCRIPT = SCRIPTS_DIR / "preprocess_data.py"
        TRAIN_SCRIPT = SCRIPTS_DIR / "train_agent.py"
        # TEST_ENV_SCRIPT = SCRIPTS_DIR / "test_environment.py" # Para pruebas

        def run_script(script_path: Path, script_args: list = None):
            """Ejecuta un script de Python y maneja errores."""
            if not script_path.exists():
                logger.error(f"Script no encontrado: {script_path}")
                raise FileNotFoundError(f"Script no encontrado: {script_path}")
            
            command = [sys.executable, str(script_path)] # sys.executable es el intérprete de Python actual
            if script_args:
                command.extend(script_args)
                
            logger.info(f"Ejecutando script: {' '.join(command)}")
            try:
                # capture_output=True para capturar stdout/stderr si es necesario
                # text=True para decodificar output como texto
                # check=True para lanzar CalledProcessError si el script devuelve un código de error
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                logger.info(f"Salida de {script_path.name}:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"Errores/Advertencias de {script_path.name}:\n{result.stderr}")
                logger.info(f"Script {script_path.name} completado exitosamente.")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al ejecutar {script_path.name}. Código de retorno: {e.returncode}")
                logger.error(f"Stdout:\n{e.stdout}")
                logger.error(f"Stderr:\n{e.stderr}")
                return False
            except Exception as e:
                logger.error(f"Excepción inesperada al ejecutar {script_path.name}: {e}", exc_info=True)
                return False

        def main(args):
            logger.info("############################################################")
            logger.info("#          INICIANDO PIPELINE COMPLETO DEL MVP             #")
            logger.info("############################################################")

            if args.skip_download:
                logger.info("Omitiendo Fase de Descarga de Datos según argumento.")
            else:
                logger.info("--- Fase 1: Descarga de Datos ---")
                if not run_script(DOWNLOAD_SCRIPT):
                    logger.error("La descarga de datos falló. Abortando pipeline.")
                    return
                logger.info("--- Fase 1: Descarga de Datos Completada ---")

            if args.skip_preprocess:
                logger.info("Omitiendo Fase de Preprocesamiento de Datos según argumento.")
            else:
                logger.info("--- Fase 2: Preprocesamiento de Datos ---")
                if not run_script(PREPROCESS_SCRIPT):
                    logger.error("El preprocesamiento de datos falló. Abortando pipeline.")
                    return
                logger.info("--- Fase 2: Preprocesamiento de Datos Completado ---")

            if args.skip_train:
                logger.info("Omitiendo Fase de Entrenamiento del Agente según argumento.")
            else:
                logger.info("--- Fase 3: Entrenamiento del Agente ---")
                train_args = []
                if args.training_timesteps:
                    train_args.extend(["--total_timesteps", str(args.training_timesteps)])
                # Añadir más argumentos para train_agent.py si es necesario
                
                if not run_script(TRAIN_SCRIPT, script_args=train_args):
                    logger.error("El entrenamiento del agente falló.")
                    # No necesariamente abortar todo el pipeline si solo el entrenamiento falla.
                else:
                    logger.info("--- Fase 3: Entrenamiento del Agente Completado ---")
            
            # (Opcional) Ejecutar otras pruebas o scripts, como test_environment.py
            # if args.run_env_test:
            #     logger.info("--- Prueba del Entorno de Trading ---")
            #     run_script(TEST_ENV_SCRIPT)

            logger.info("############################################################")
            logger.info("#            PIPELINE COMPLETO DEL MVP FINALIZADO          #")
            logger.info("############################################################")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description="Orquestador del Pipeline MVP para BTC-Transformer-RL-Trader.")
            parser.add_argument("--skip_download", action="store_true", help="Omitir la fase de descarga de datos.")
            parser.add_argument("--skip_preprocess", action="store_true", help="Omitir la fase de preprocesamiento.")
            parser.add_argument("--skip_train", action="store_true", help="Omitir la fase de entrenamiento.")
            parser.add_argument("--training_timesteps", type=int, default=None,
                                help="Número de timesteps para el entrenamiento del agente (sobrescribe config).")
            # parser.add_argument("--run_env_test", action="store_true", help="Ejecutar prueba del entorno después del pipeline.")
            
            script_args = parser.parse_args()
            main(script_args)
        ```
      * **3.2. Hacer el Script Ejecutable (Opcional, o usar `python ...`)**:
          * En sistemas Linux/macOS: `chmod +x scripts/run_pipeline.py`

-----

### Paso 4: Documentación del Proceso de Ejecución con Docker

  * **Descripción Exhaustiva**: Actualizar el archivo `README.md` (o crear una sección/archivo nuevo como `RUNNING.md`) con instrucciones claras sobre cómo construir la imagen de Docker, iniciar los servicios con Docker Compose, y ejecutar el pipeline completo o scripts individuales.
  * **Acciones Específicas**:
      * **4.1. Actualizar `README.md` - Sección "Cómo Empezar" o similar**:
          * Añadir instrucciones para:
            1.  **Construir la Imagen y Levantar Servicios**:
                ```bash
                # Desde la raíz del proyecto
                docker-compose up --build -d
                ```
            2.  **Ejecutar el Pipeline Completo**:
                ```bash
                # Ejecutar el pipeline completo (descarga, preproceso, entrenamiento con timesteps por defecto)
                docker-compose exec workhorse_app python scripts/run_pipeline.py

                # Ejecutar con opciones (ej. omitir descarga y preproceso, N timesteps para entrenar)
                docker-compose exec workhorse_app python scripts/run_pipeline.py --skip_download --skip_preprocess --training_timesteps 10000
                ```
            3.  **Ejecutar Scripts Individuales**:
                ```bash
                # Descargar datos (usará config de module1)
                docker-compose exec workhorse_app python scripts/download_data.py

                # Preprocesar datos (usará config de module1 y module2)
                docker-compose exec workhorse_app python scripts/preprocess_data.py

                # Entrenar agente (usará config de module4 y datos preprocesados)
                # (Para una prueba corta de entrenamiento)
                docker-compose exec workhorse_app python scripts/train_agent.py --total_timesteps 5000

                # Probar el entorno (requiere datos preprocesados)
                docker-compose exec workhorse_app python scripts/test_environment.py

                # Probar la carga de configuración
                docker-compose exec workhorse_app python scripts/test_config.py
                ```
            4.  **Acceder a Logs de TensorBoard**:
                  * "Los logs de TensorBoard se guardan en `results_host/sac_tensorboard_logs/` (mapeado desde `./results_host/sac_tensorboard_logs/` en el host). Puedes visualizarlos ejecutando `tensorboard --logdir ./results_host/sac_tensorboard_logs/` en tu máquina host (no dentro de Docker) y abriendo el enlace en tu navegador."
            5.  **Detener Servicios**:
                ```bash
                docker-compose down # -v para eliminar volúmenes (incluyendo redis_data si no es necesario)
                ```
      * **4.2. (Opcional) Crear `Makefile` para Comandos Comunes**:
          * Un `Makefile` puede simplificar los comandos de Docker.
            ```makefile
            # Makefile
            .PHONY: help build up down logs shell pipeline run_download run_preprocess run_train clean_pyc

            help:
            	@echo "Comandos disponibles:"
            	@echo "  build          : Construye o reconstruye los servicios de Docker."
            	@echo "  up             : Levanta los servicios en segundo plano."
            	@echo "  down           : Detiene y elimina los contenedores. Añade v=1 para volúmenes (down v=1)."
            	@echo "  logs           : Muestra los logs de los servicios. Añade service=<nombre> para específico."
            	@echo "  shell          : Abre un shell bash en el contenedor workhorse_app."
            	@echo "  pipeline       : Ejecuta el pipeline completo (download, preprocess, train)."
            	@echo "  pipeline-notrain: Ejecuta download y preprocess."
            	@echo "  run_download   : Ejecuta solo el script de descarga de datos."
            	@echo "  run_preprocess : Ejecuta solo el script de preprocesamiento."
            	@echo "  run_train      : Ejecuta solo el script de entrenamiento (ej. TIMESTEPS=1000 make run_train)."
            	@echo "  tensorboard    : Inicia TensorBoard (ejecutar en host)."
            	@echo "  clean_pyc      : Elimina archivos .pyc y __pycache__."

            build:
            	docker-compose build

            up:
            	docker-compose up -d

            down:
            ifeq ($(v),1)
            	docker-compose down -v
            else
            	docker-compose down
            endif

            logs:
            ifeq ($(service),)
            	docker-compose logs -f
            else
            	docker-compose logs -f $(service)
            endif

            shell:
            	docker-compose exec workhorse_app bash

            pipeline: up
            	docker-compose exec workhorse_app python scripts/run_pipeline.py $(ARGS)

            pipeline-notrain: up
            	docker-compose exec workhorse_app python scripts/run_pipeline.py --skip_train $(ARGS)

            run_download: up
            	docker-compose exec workhorse_app python scripts/download_data.py $(ARGS)

            run_preprocess: up
            	docker-compose exec workhorse_app python scripts/preprocess_data.py $(ARGS)

            # Ejemplo: TIMESTEPS=5000 make run_train
            run_train: up
            ifeq ($(TIMESTEPS),)
            	docker-compose exec workhorse_app python scripts/train_agent.py $(ARGS)
            else
            	docker-compose exec workhorse_app python scripts/train_agent.py --total_timesteps $(TIMESTEPS) $(ARGS)
            endif

            tensorboard:
            	@echo "Abre tu navegador en http://localhost:6006 después de ejecutar esto desde el host:"
            	@echo "tensorboard --logdir ./results_host/sac_tensorboard_logs/"


            clean_pyc:
            	find . -name "*.pyc" -exec rm -f {} \;
            	find . -name "__pycache__" -exec rm -rf {} \;

            # ARGS se puede usar para pasar argumentos adicionales, ej: make pipeline ARGS="--skip_download"
            ```
          * Si se añade un `Makefile`, referenciarlo en el `README.md`.

-----

### Paso 5: Commit de los Cambios de la Fase 6

  * **Descripción Exhaustiva**: Añadir todos los cambios y la documentación actualizada al control de versiones Git.
  * **Acciones Específicas**:
      * **5.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add Dockerfile docker-compose.yml .env.example
        git add scripts/run_pipeline.py
        git add README.md # Si se actualizó con instrucciones de Docker
        # git add Makefile # Si se creó
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 6: Finalizar Módulo 10 (Orquestación Docker). Revisar Dockerfile/docker-compose, añadir script de pipeline principal y actualizar documentación de ejecución."
        ```

**Fin de la Fase de Implementación 6.**

-----

Con esto, el entorno Docker está consolidado y tenemos un script para ejecutar el pipeline principal del MVP. Esto facilita enormemente la reproducibilidad y la ejecución de las diferentes etapas del proyecto.

**Recordatorio sobre Deuda Técnica de Fase 3 (Módulo 2):**
Sigue siendo importante que el Módulo 2 (`DataPreprocessor`) guarde `env_step_close_prices` y `env_step_atr_values` (no normalizados) en el archivo `.npz` para el correcto funcionamiento del `TradingEnv` (Módulo 3) en cálculos de P\&L y slippage. Si esto no se ha hecho, el entrenamiento del agente podría basarse en mecánicas de entorno incorrectas.

La siguiente fase sería el Módulo 7: Framework de Backtesting (Walk-Forward Optimization), que es un componente clave del MVP.

----

## Fase de Implementación 7: Módulo 7 - Framework de Backtesting (Walk-Forward Optimization)

**Nombre Descriptivo de la Fase:** Implementación del Sistema de Validación Walk-Forward Optimization y Análisis de Rendimiento.

Esta fase se centra en desarrollar el Módulo 7, que orquestará el entrenamiento y la evaluación del agente de RL (Módulo 4) utilizando el entorno de trading (Módulo 3) sobre los datos históricos procesados (Módulo 2). La pieza central será la implementación de una metodología de Walk-Forward Optimization (WFO) para evaluar la robustez del modelo y evitar el sobreajuste a todo el conjunto de datos históricos. Al final, se calcularán métricas de rendimiento detalladas y se generarán visualizaciones.

**Recordatorio Importante sobre Datos para el Entorno (Deuda Técnica de M2/M3):**
El correcto funcionamiento del `TradingEnv` (Módulo 3) y, por ende, de este módulo de backtesting, depende crucialmente de que el archivo `.npz` generado por el Módulo 2 (`DataPreprocessor`) contenga no solo las secuencias de features normalizadas, sino también arrays alineados con los `env_step_close_prices` (precios de cierre no normalizados) y `env_step_atr_values` (valores ATR no normalizados). Estos son esenciales para que el entorno calcule P\&L, slippage, etc., de forma precisa. **Se asume que Módulo 2 ha sido o será actualizado para proveer estos datos.**

-----

### Paso 1: Creación de la Clase `WalkForwardOptimizer` y Archivos Necesarios

  * **Descripción Exhaustiva**: Crear el archivo `src/backtesting/wfo_framework.py` que contendrá la clase `WalkForwardOptimizer`. Esta clase gestionará la división de datos, los bucles de entrenamiento/evaluación de WFO y la recolección de resultados.
  * **Acciones Específicas**:
      * **1.1. Crear `src/backtesting/__init__.py`** (si no existe):
        ```python
        # src/backtesting/__init__.py
        from .wfo_framework import WalkForwardOptimizer

        __all__ = ['WalkForwardOptimizer']
        ```
      * **1.2. Crear el Esqueleto de `src/backtesting/wfo_framework.py`**:
        ```python
        # src/backtesting/wfo_framework.py
        import pandas as pd
        import numpy as np
        from pathlib import Path
        import logging
        from datetime import datetime, timezone
        from dateutil.relativedelta import relativedelta # Para sumar meses a fechas
        from typing import List, Dict, Any, Tuple, Optional

        import quantstats as qs # type: ignore[import-untyped]

        from config_loader import load_yaml_config, get_env_variable
        from environment.trading_env import TradingEnv # Módulo 3
        from agent.sac_agent_trainer import SACAgentTrainer # Módulo 4
        # from agent.policy import TransformerFeaturesExtractor # Para cargar modelos

        logger = logging.getLogger(__name__)

        class WalkForwardOptimizer:
            """
            Gestiona el proceso de Walk-Forward Optimization (WFO) para entrenar y evaluar
            un agente de trading de RL.
            """
            def __init__(self, 
                         wfo_config_name: str = "module7_backtesting_wfo",
                         agent_config_name: str = "module4_agent_sac",
                         preproc_config_name: str = "module2_preprocessing",
                         data_acq_config_name: str = "module1_data_acquisition"
                        ):
                logger.info("Inicializando WalkForwardOptimizer...")
                try:
                    self.wfo_config = load_yaml_config(wfo_config_name)
                    self.agent_config = load_yaml_config(agent_config_name) # Para total_timesteps por walk, etc.
                    self.preproc_config = load_yaml_config(preproc_config_name)
                    self.data_acq_config = load_yaml_config(data_acq_config_name)

                    # Parámetros de WFO
                    self.is_window_months = self.wfo_config.get("wfo_is_window_months", 18)
                    self.oos_window_months = self.wfo_config.get("wfo_oos_window_months", 3)
                    self.step_months = self.wfo_config.get("wfo_step_months", 3) # Debe ser igual a oos_window_months para WFO contiguo
                    self.window_type = self.wfo_config.get("wfo_window_type", "rolling") # "rolling" o "expanding"

                    # Rutas
                    self.data_dir_str = get_env_variable("DATA_DIR_HOST_FOR_APP", "/app/data_persistent")
                    self.results_dir_str = get_env_variable("RESULTS_DIR_HOST_FOR_APP", "/app/results_persistent")
                    
                    self.trading_pair = self.data_acq_config.get("trading_pair", "BTCUSDT")
                    self.kline_interval = self.data_acq_config.get("kline_interval", "15m")
                    self.sequence_L = self.preproc_config.get("sequence_length_L", 96)
                    
                    # N_features en el nombre del archivo .npz es el N_market_features (ej. 20)
                    self.n_market_features_from_m2 = 20 # Según diseño de M2 y M3.
                    
                    self.processed_filename = (f"{self.trading_pair}_{self.kline_interval}_"
                                               f"L{self.sequence_L}_N{self.n_market_features_from_m2}_processed_sequences.npz")
                    self.npz_file_path = (Path(self.data_dir_str) / "processed" / 
                                          self.trading_pair / self.kline_interval / self.processed_filename)

                    # Directorios de salida para WFO
                    self.wfo_base_output_dir = Path(self.results_dir_str) / "wfo_runs" / f"{self.trading_pair}_{self.kline_interval}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                    self.wfo_base_output_dir.mkdir(parents=True, exist_ok=True)
                    self.wfo_models_dir = self.wfo_base_output_dir / "models_per_walk"
                    self.wfo_models_dir.mkdir(parents=True, exist_ok=True)
                    self.wfo_oos_results_dir = self.wfo_base_output_dir / "oos_results_per_walk"
                    self.wfo_oos_results_dir.mkdir(parents=True, exist_ok=True)

                    # Cargar el dataset completo una vez
                    self._load_full_dataset()

                    logger.info("WalkForwardOptimizer inicializado.")
                    logger.info(f"Config WFO: IS={self.is_window_months}m, OOS={self.oos_window_months}m, Step={self.step_months}m, Type={self.window_type}")
                    logger.info(f"Dataset cargado: {self.npz_file_path} con {self.num_total_data_points} puntos (secuencias).")
                    logger.info(f"Directorio de salida WFO: {self.wfo_base_output_dir}")

                except FileNotFoundError as e:
                    logger.error(f"Error: Archivo de datos procesados no encontrado en {self.npz_file_path}. {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error inesperado durante la inicialización de WalkForwardOptimizer: {e}", exc_info=True)
                    raise

            def _load_full_dataset(self):
                """Carga el dataset completo (.npz) que contiene secuencias, timestamps y datos para el entorno."""
                if not self.npz_file_path.exists():
                    raise FileNotFoundError(f"Archivo de datos .npz no encontrado: {self.npz_file_path}")
                
                data = np.load(self.npz_file_path, allow_pickle=True) # allow_pickle si hay objetos (aunque no debería)
                self.full_market_sequences = data['sequences'].astype(np.float32)
                self.full_market_timestamps_ms = data['timestamps'].astype(np.int64) # Timestamps de inicio de cada secuencia
                
                # Datos necesarios para la mecánica del entorno (NO NORMALIZADOS)
                if 'env_step_close_prices' in data and 'env_step_atr_values' in data:
                    self.full_env_step_close_prices = data['env_step_close_prices'].astype(np.float32)
                    self.full_env_step_atr_values = data['env_step_atr_values'].astype(np.float32)
                else:
                    msg = "Faltan 'env_step_close_prices' y/o 'env_step_atr_values' en el archivo .npz. Son cruciales."
                    logger.error(msg)
                    raise ValueError(msg)

                self.num_total_data_points = len(self.full_market_sequences)
                if self.num_total_data_points != len(self.full_market_timestamps_ms) or \
                   self.num_total_data_points != len(self.full_env_step_close_prices) or \
                   self.num_total_data_points != len(self.full_env_step_atr_values):
                    raise ValueError("Discrepancia en la longitud de los arrays cargados del archivo .npz.")

                # Convertir timestamps (ms) a datetime objetos para facilitar el slicing por fechas
                self.full_market_datetimes = pd.to_datetime(self.full_market_timestamps_ms, unit='ms', utc=True)


            # --- Métodos para slicing de datos, bucle WFO, entrenamiento, evaluación, métricas ---
            # def _get_walk_forward_windows(self) -> List[Dict[str, pd.Timestamp]]:
            # def run_optimization(self) -> pd.DataFrame:
            # def _train_on_is_window(self, is_data_slice: Dict, walk_num: int) -> Path:
            # def _evaluate_on_oos_window(self, oos_data_slice: Dict, model_path: Path, walk_num: int) -> pd.DataFrame:
            # def _calculate_performance_metrics(self, all_oos_trades: pd.DataFrame, all_oos_equity_curves: pd.DataFrame) -> Dict:
            # def _generate_report_and_visualizations(self, metrics: Dict, concatenated_oos_equity: pd.Series):

        if __name__ == '__main__':
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger.info("Ejecutando prueba directa de WalkForwardOptimizer...")
            
            # Mockear get_env_variable para rutas si es necesario para prueba local
            # os.environ.setdefault("DATA_DIR_HOST_FOR_APP", "./temp_test_wfo_data/data_persistent")
            # os.environ.setdefault("RESULTS_DIR_HOST_FOR_APP", "./temp_test_wfo_data/results_persistent")

            # Crear un archivo .npz de prueba mínimo
            temp_data_root = Path("./temp_test_wfo_data")
            temp_data_dir_str = str(temp_data_root / "data_persistent") # Simula DATA_DIR_HOST_FOR_APP
            temp_results_dir_str = str(temp_data_root / "results_persistent")
            
            # Configurar variables de entorno para que get_env_variable las encuentre
            # (o mockear get_env_variable directamente)
            with mock.patch('src.backtesting.wfo_framework.get_env_variable') as mock_getenv:
                def get_mock_env_wfo(var_name, default_value=None, required=True):
                    if var_name == "DATA_DIR_HOST_FOR_APP": return temp_data_dir_str
                    if var_name == "RESULTS_DIR_HOST_FOR_APP": return temp_results_dir_str
                    return "dummy_val_not_used" # Para otras vars de .env
                mock_getenv.side_effect = get_mock_env_wfo

                L_test_wfo = 96
                N_market_feat_test_wfo = 20
                # Simular datos para 2 años (24 meses) para tener al menos un walk
                num_months_sim = 24 
                approx_days = num_months_sim * 30
                klines_per_day = (24 * 60) // 15 # Klines de 15min
                num_samples_test_wfo = approx_days * klines_per_day

                sequences_arr = np.random.rand(num_samples_test_wfo, L_test_wfo, N_market_feat_test_wfo).astype(np.float32)
                
                # Generar timestamps para 2 años, empezando desde hace 2 años
                start_date_sim = datetime.now(timezone.utc) - relativedelta(years=2)
                timestamps_arr = np.array([
                    int((start_date_sim + relativedelta(minutes=i*15)).timestamp() * 1000) 
                    for i in range(num_samples_test_wfo)
                ]).astype(np.int64)
                
                close_prices_arr = np.linspace(20000, 25000, num_samples_test_wfo).astype(np.float32)
                atr_values_arr = np.full(num_samples_test_wfo, 50.0).astype(np.float32)

                # Crear la estructura de directorios esperada para el archivo .npz
                pair_test = "BTCUSDT"
                interval_test = "15m"
                npz_save_path_dir = Path(temp_data_dir_str) / "processed" / pair_test / interval_test
                npz_save_path_dir.mkdir(parents=True, exist_ok=True)
                
                test_npz_filename = f"{pair_test}_{interval_test}_L{L_test_wfo}_N{N_market_feat_test_wfo}_processed_sequences.npz"
                test_npz_full_path = npz_save_path_dir / test_npz_filename
                
                np.savez_compressed(
                    test_npz_full_path, 
                    sequences=sequences_arr, timestamps=timestamps_arr,
                    env_step_close_prices=close_prices_arr, env_step_atr_values=atr_values_arr
                )
                logger.info(f"Archivo .npz de prueba creado en: {test_npz_full_path}")

                try:
                    wfo = WalkForwardOptimizer() # Usará las configs mockeadas por get_env_variable
                    logger.info("WalkForwardOptimizer instanciado para prueba.")
                    # Aquí se llamaría a wfo.run_optimization() una vez implementado
                    # Ejemplo:
                    # windows = wfo._get_walk_forward_windows()
                    # logger.info(f"Ventanas WFO generadas: {windows}")
                    # if windows:
                    #    logger.info(f"Primera ventana: IS {windows[0]['is_start']} a {windows[0]['is_end']}, OOS {windows[0]['oos_start']} a {windows[0]['oos_end']}")


                except Exception as e:
                    logger.error(f"Error durante la prueba de WalkForwardOptimizer: {e}", exc_info=True)
                finally:
                    # Limpiar (opcional)
                    # import shutil
                    # if temp_data_root.exists():
                    #     shutil.rmtree(temp_data_root)
                    pass
        ```

-----

### Paso 2: Implementación de la Lógica de División de Datos (Ventanas WFO)

  * **Descripción Exhaustiva**: Añadir el método `_get_walk_forward_windows` a `WalkForwardOptimizer`. Este método calculará los rangos de fechas (y los índices correspondientes en el dataset de timestamps) para cada ventana In-Sample (IS) y Out-of-Sample (OOS) según la configuración de WFO.
  * **Acciones Específicas**:
      * **2.1. Definir `_get_walk_forward_windows`**:
        ```python
        # En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer

        def _get_walk_forward_windows(self) -> List[Dict[str, Any]]:
            """
            Calcula y devuelve una lista de diccionarios, cada uno representando una ventana
            de Walk-Forward con sus fechas/índices de inicio y fin para IS y OOS.
            """
            windows = []
            
            # Fecha de inicio del dataset completo (primer timestamp de secuencia)
            dataset_start_date = pd.Timestamp(self.full_market_datetimes[0])
            # Fecha de fin del dataset completo (último timestamp de secuencia)
            dataset_end_date = pd.Timestamp(self.full_market_datetimes[-1])
            
            logger.info(f"Calculando ventanas WFO. Dataset total: {dataset_start_date} a {dataset_end_date}")

            current_is_start_date = dataset_start_date
            walk_num = 0

            while True:
                walk_num += 1
                # Calcular fin de la ventana In-Sample (IS)
                current_is_end_date = current_is_start_date + relativedelta(months=self.is_window_months) - relativedelta(days=1) # Fin del día
                
                # Calcular inicio y fin de la ventana Out-of-Sample (OOS)
                current_oos_start_date = current_is_end_date + relativedelta(days=1)
                current_oos_end_date = current_oos_start_date + relativedelta(months=self.oos_window_months) - relativedelta(days=1)

                # Asegurar que la ventana OOS no exceda el final del dataset
                if current_oos_end_date > dataset_end_date:
                    current_oos_end_date = dataset_end_date # Ajustar al final del dataset
                    if current_oos_start_date > dataset_end_date : # Si OOS empieza después del fin de datos
                        logger.info(f"No hay suficientes datos para la ventana OOS #{walk_num} (OOS start: {current_oos_start_date}). Finalizando generación de ventanas.")
                        break 
                
                # Convertir fechas a índices en nuestros arrays de datos
                # Encontrar el índice para is_start_date (primer timestamp >= is_start_date)
                is_start_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(current_is_start_date, tz='UTC'), side='left')
                # Encontrar el índice para is_end_date (último timestamp <= is_end_date)
                is_end_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(current_is_end_date, tz='UTC').replace(hour=23,minute=59,second=59), side='right') -1
                
                oos_start_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(current_oos_start_date, tz='UTC'), side='left')
                oos_end_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(current_oos_end_date, tz='UTC').replace(hour=23,minute=59,second=59), side='right') -1

                # Validar índices y duración mínima
                min_data_points_for_env = self.sequence_L + 50 # Mínimo arbitrario para que un env pueda correr un poco
                if (is_end_idx - is_start_idx + 1) < min_data_points_for_env or \
                   (oos_end_idx - oos_start_idx + 1) < 1: # OOS debe tener al menos 1 punto
                    logger.warning(f"Ventana WFO #{walk_num} demasiado corta o inválida. "
                                   f"IS: {current_is_start_date.date()} ({is_start_idx}) a {current_is_end_date.date()} ({is_end_idx}). "
                                   f"OOS: {current_oos_start_date.date()} ({oos_start_idx}) a {current_oos_end_date.date()} ({oos_end_idx}). "
                                   f"Omitiendo y finalizando.")
                    break
                
                # Asegurar que no haya solapamiento de índices si algo salió mal con las fechas
                if oos_start_idx <= is_end_idx :
                    logger.warning(f"Solapamiento detectado entre IS y OOS para walk {walk_num}, ajustando OOS start index.")
                    oos_start_idx = is_end_idx + 1
                    if oos_start_idx > oos_end_idx: # Si el ajuste invalida OOS
                         logger.warning(f"Ventana OOS para walk {walk_num} inválida después de ajuste de solapamiento. Finalizando.")
                         break


                window_info = {
                    "walk_num": walk_num,
                    "is_start_date": current_is_start_date, "is_end_date": current_is_end_date,
                    "is_start_idx": is_start_idx, "is_end_idx": is_end_idx,
                    "oos_start_date": current_oos_start_date, "oos_end_date": current_oos_end_date,
                    "oos_start_idx": oos_start_idx, "oos_end_idx": oos_end_idx,
                }
                windows.append(window_info)
                logger.info(f"Ventana WFO #{walk_num}: "
                            f"IS [{current_is_start_date.date()} ({is_start_idx}) - {current_is_end_date.date()} ({is_end_idx})], "
                            f"OOS [{current_oos_start_date.date()} ({oos_start_idx}) - {current_oos_end_date.date()} ({oos_end_idx})]")

                # Preparar para la siguiente iteración
                if self.window_type == "rolling":
                    current_is_start_date = current_is_start_date + relativedelta(months=self.step_months)
                elif self.window_type == "expanding":
                    # current_is_start_date no cambia, solo se extiende is_end_date (manejado por el bucle y oos_start_date)
                    # El is_end_date se recalcula basado en el nuevo oos_start_date - 1 día, menos is_window_months.
                    # Esto es más complejo. La forma más simple de "expanding" es que IS siempre empiece en dataset_start_date.
                    # En este caso, solo el oos_start_date avanza.
                    # Si IS siempre empieza en `dataset_start_date`:
                    # current_is_start_date = dataset_start_date (fijo)
                    # current_is_end_date = (current_oos_start_date - relativedelta(days=1)) # Esta es la nueva IS_end
                    # Es más simple si el `step_months` define el avance de OOS_start.
                    
                    # Para expanding, IS start es fijo. IS end es OOS start - 1 day.
                    # El avance es por OOS.
                    next_oos_start_candidate = current_oos_start_date + relativedelta(months=self.step_months)
                    if next_oos_start_candidate > dataset_end_date: # Si el siguiente paso OOS ya está fuera de rango
                        break
                    # `current_is_start_date` se mantiene como `dataset_start_date` para expanding.
                    # No se actualiza aquí si es "expanding", el bucle recalculará `current_is_end_date`
                    # basado en el nuevo `current_oos_start_date`.
                    # El `while True` y el chequeo de `current_oos_start_date > dataset_end_date` deberían manejarlo.
                    # Para "expanding", el `current_is_start_date` efectivo para la siguiente iteración
                    # se mantiene, pero el `is_end_date` se calculará como `next_oos_start - 1 day`.
                    # El `current_is_start_date` para la SIGUIENTE ventana debe ser el OOS_start_date de la ventana ACTUAL
                    # si la ventana IS se mueve.
                    #
                    # Si es "rolling": current_is_start_date avanza por `step_months`.
                    # Si es "expanding": current_is_start_date es siempre `dataset_start_date`.
                    # El `while True` se romperá cuando `current_oos_start_date` supere `dataset_end_date`.
                    
                    # La forma en que se actualiza current_is_start_date debe ser condicional al tipo:
                    if current_oos_start_date + relativedelta(months=self.step_months) > dataset_end_date :
                        break # Salir si el siguiente paso ya nos saca de los datos.

                    # No se actualiza `current_is_start_date` si es expanding.
                    # Solo necesitamos que el loop continue y `current_oos_start_date` avance.
                    # Esto se complica. Vamos a simplificar el "expanding":
                    # El bucle principal avanza como si fuera "rolling" para el inicio de OOS.
                    # Dentro del bucle, si es "expanding", is_start_date siempre es dataset_start_date.
                    # Esto ya está implícito si no actualizamos current_is_start_date para "expanding".
                    
                    # El avance del OOS_start_date es el que realmente mueve el WFO.
                    potential_next_oos_start = current_oos_start_date + relativedelta(months=self.step_months)
                    if potential_next_oos_start > dataset_end_date:
                         break
                    # Para la siguiente iteración, si es expanding, is_start_date será dataset_start_date.
                    # Si es rolling, será el step.

                else: # Tipo de ventana desconocido
                    logger.error(f"Tipo de ventana WFO desconocido: {self.window_type}. Abortando.")
                    return []
                
                # Condición de salida principal: si la próxima ventana OOS ya no tiene sentido.
                if current_oos_end_date >= dataset_end_date :
                    logger.info("Se alcanzó el final del dataset con la última ventana OOS.")
                    break
            
            if not windows:
                logger.warning("No se pudieron generar ventanas de Walk-Forward. Verificar configuración y longitud del dataset.")
            return windows

        def _get_data_slice(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
            """
            Extrae un slice de todos los arrays de datos necesarios para el entorno.
            end_idx es inclusivo para Python slicing (ej. array[:end_idx+1]).
            """
            if start_idx < 0 or end_idx >= self.num_total_data_points or start_idx > end_idx:
                raise ValueError(f"Índices de slice inválidos: start={start_idx}, end={end_idx}, total_points={self.num_total_data_points}")

            # El end_idx de las ventanas es el índice final de la secuencia *que se puede usar como inicio*.
            # Si queremos datos HASTA ese punto, es `[:end_idx+1]`.
            return {
                'sequences': self.full_market_sequences[start_idx : end_idx + 1],
                'timestamps': self.full_market_timestamps_ms[start_idx : end_idx + 1],
                'env_step_close_prices': self.full_env_step_close_prices[start_idx : end_idx + 1],
                'env_step_atr_values': self.full_env_step_atr_values[start_idx : end_idx + 1]
            }

        def _prepare_temp_npz_for_env(self, data_slice: Dict[str, np.ndarray], file_name_suffix: str) -> Path:
            """
            Guarda un slice de datos en un archivo .npz temporal para ser usado por TradingEnv.
            """
            temp_npz_dir = self.wfo_base_output_dir / "temp_env_data"
            temp_npz_dir.mkdir(parents=True, exist_ok=True)
            temp_npz_path = temp_npz_dir / f"temp_data_slice_{file_name_suffix}.npz"
            
            np.savez_compressed(
                temp_npz_path,
                sequences=data_slice['sequences'],
                timestamps=data_slice['timestamps'],
                env_step_close_prices=data_slice['env_step_close_prices'],
                env_step_atr_values=data_slice['env_step_atr_values']
            )
            return temp_npz_path
        ```
      * **Ajuste en `_get_walk_forward_windows` para "expanding"**:
        La lógica de "expanding" necesita que `current_is_start_date` se resetee a `dataset_start_date` en cada iteración si `self.window_type == "expanding"`.
        Se modifica el bucle:
        ```python
        # ... dentro de _get_walk_forward_windows
            # walk_num = 0 # Ya está
            effective_is_start_date = dataset_start_date # Para la primera iteración o rolling
            
            while True:
                walk_num += 1
                
                # Determinar el inicio real de IS para esta ventana
                actual_is_start_this_walk = dataset_start_date if self.window_type == "expanding" else effective_is_start_date

                if actual_is_start_this_walk >= dataset_end_date: # No más datos para IS
                    break

                current_is_end_date = actual_is_start_this_walk + relativedelta(months=self.is_window_months) - relativedelta(days=1)
                # ... (resto de la lógica de cálculo de OOS y fechas/índices como antes) ...
                # ... (is_start_idx se calculará desde actual_is_start_this_walk) ...

                # ... (al final del bucle, antes de `break` o `continue`)
                # Avanzar effective_is_start_date para la *siguiente* ventana (solo si es rolling)
                if self.window_type == "rolling":
                    effective_is_start_date = effective_is_start_date + relativedelta(months=self.step_months)
                # Para "expanding", effective_is_start_date no se usa para determinar el inicio de IS.
                # El avance es implícito por cómo se calcula OOS y la condición de salida.
                # La condición de salida principal se basa en que current_oos_end_date alcance dataset_end_date.

                if current_oos_end_date >= dataset_end_date:
                    logger.info("Se alcanzó el final del dataset con la última ventana OOS.")
                    break
                
                # Asegurar que el bucle avance incluso para expanding, controlando por el inicio de OOS
                if self.window_type == "expanding":
                    # Si el siguiente OOS start date (basado en el step desde el OOS start actual)
                    # supera el fin del dataset, entonces paramos.
                    next_oos_start_candidate = window_info["oos_start_date"] + relativedelta(months=self.step_months)
                    if next_oos_start_candidate > dataset_end_date:
                        break
        # ...
        ```
        Esta lógica de "expanding" necesita ser cuidadosa. Una forma más común es que el OOS\_start\_date avance y el IS\_end\_date sea `OOS_start_date - 1 día`, mientras que IS\_start\_date es fijo. El bucle iteraría mientras `OOS_start_date < dataset_end_date`.

-----


Primero, refinaré ligeramente la lógica de `_get_walk_forward_windows` para mayor claridad, especialmente para el manejo de ventanas "expanding", y luego procederemos con los siguientes pasos.

```python
# En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer

# (Reemplazar o fusionar con la versión anterior de _get_walk_forward_windows)
def _get_walk_forward_windows(self) -> List[Dict[str, Any]]:
    """
    Calcula y devuelve una lista de diccionarios, cada uno representando una ventana
    de Walk-Forward con sus fechas/índices de inicio y fin para IS y OOS.
    """
    windows = []
    dataset_start_date = pd.Timestamp(self.full_market_datetimes[0])
    dataset_end_date = pd.Timestamp(self.full_market_datetimes[-1])
    
    logger.info(f"Calculando ventanas WFO. Dataset total: {dataset_start_date.date()} a {dataset_end_date.date()}")
    logger.info(f"Config WFO: IS={self.is_window_months}m, OOS={self.oos_window_months}m, Step={self.step_months}m, Type={self.window_type}")

    # El primer OOS no puede empezar antes de que haya suficientes datos para una ventana IS.
    current_oos_period_start_date = dataset_start_date + relativedelta(months=self.is_window_months)
    walk_num = 0

    while current_oos_period_start_date <= dataset_end_date:
        walk_num += 1
        
        # Determinar fechas de la ventana Out-of-Sample (OOS)
        oos_start_date = current_oos_period_start_date
        oos_end_date = oos_start_date + relativedelta(months=self.oos_window_months) - relativedelta(days=1)
        oos_end_date = min(oos_end_date, dataset_end_date) # No exceder el final del dataset

        if oos_start_date > oos_end_date: # No hay suficientes datos para este OOS
            logger.info(f"No hay suficientes datos para la ventana OOS #{walk_num} (OOS start: {oos_start_date.date()}). Finalizando.")
            break

        # Determinar fechas de la ventana In-Sample (IS)
        is_end_date = oos_start_date - relativedelta(days=1)
        if self.window_type == "rolling":
            is_start_date = is_end_date - relativedelta(months=self.is_window_months) + relativedelta(days=1)
            is_start_date = max(is_start_date, dataset_start_date) # Asegurar que no empiece antes del dataset
        elif self.window_type == "expanding":
            is_start_date = dataset_start_date
        else:
            logger.error(f"Tipo de ventana WFO desconocido: {self.window_type}. Abortando.")
            return []

        if is_start_date > is_end_date: # No hay suficientes datos para este IS
            logger.warning(f"No hay suficientes datos para la ventana IS #{walk_num} (IS start: {is_start_date.date()} > IS end: {is_end_date.date()}). Omitiendo y finalizando.")
            break
            
        # Convertir fechas a índices
        is_start_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(is_start_date, tz='UTC'), side='left')
        is_end_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(is_end_date, tz='UTC').replace(hour=23,minute=59,second=59), side='right') -1
        
        oos_start_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(oos_start_date, tz='UTC'), side='left')
        oos_end_idx = np.searchsorted(self.full_market_datetimes, pd.Timestamp(oos_end_date, tz='UTC').replace(hour=23,minute=59,second=59), side='right') -1

        # Validar que los índices sean válidos y las ventanas tengan una longitud mínima
        min_is_len = max(self.sequence_L + 50, self.is_window_months * 20 * 4) # Heurística (50 pasos extra o 4 klines/hr * 20 días/mes)
        min_oos_len = 1 # OOS debe tener al menos un punto

        if not (is_start_idx <= is_end_idx and (is_end_idx - is_start_idx + 1) >= min_is_len and \
                oos_start_idx <= oos_end_idx and (oos_end_idx - oos_start_idx + 1) >= min_oos_len and \
                is_end_idx < oos_start_idx): # IS debe terminar antes de que OOS comience
            logger.warning(f"Ventana WFO #{walk_num} inválida o demasiado corta. "
                           f"IS: {is_start_date.date()}[{is_start_idx}]-{is_end_date.date()}[{is_end_idx}] ({is_end_idx-is_start_idx+1} pts). "
                           f"OOS: {oos_start_date.date()}[{oos_start_idx}]-{oos_end_date.date()}[{oos_end_idx}] ({oos_end_idx-oos_start_idx+1} pts). "
                           "Finalizando generación de ventanas.")
            break
            
        window_info = {
            "walk_num": walk_num,
            "is_start_date": is_start_date, "is_end_date": is_end_date,
            "is_start_idx": is_start_idx, "is_end_idx": is_end_idx,
            "oos_start_date": oos_start_date, "oos_end_date": oos_end_date,
            "oos_start_idx": oos_start_idx, "oos_end_idx": oos_end_idx,
        }
        windows.append(window_info)
        logger.info(f"Ventana WFO #{walk_num}: "
                    f"IS [{is_start_date.date()}({is_start_idx})-{is_end_date.date()}({is_end_idx})], "
                    f"OOS [{oos_start_date.date()}({oos_start_idx})-{oos_end_date.date()}({oos_end_idx})]")

        # Avanzar al inicio del siguiente periodo OOS
        current_oos_period_start_date = current_oos_period_start_date + relativedelta(months=self.step_months)
        
        if oos_end_date >= dataset_end_date and self.step_months > 0 : # Evitar bucle infinito si step es 0
            logger.info("Se procesó la última ventana OOS posible que cubre el final del dataset.")
            break
            
    if not windows:
        logger.warning("No se pudieron generar ventanas de Walk-Forward. Verificar config y longitud del dataset.")
    return windows

```

Continuamos con la **Fase de Implementación 7: Módulo 7 - Framework de Backtesting (Walk-Forward Optimization)**.

-----

### Paso 3: Implementación del Bucle Principal de Walk-Forward (`run_optimization`)

  * **Descripción Exhaustiva**: Añadir el método `run_optimization` que itera sobre las ventanas IS/OOS generadas. En cada "walk", orquesta el entrenamiento del agente en los datos IS y su posterior evaluación en los datos OOS, guardando los modelos y resultados.
  * **Acciones Específicas**:
      * **3.1. Definir `run_optimization` en `WalkForwardOptimizer`**:
        ```python
        # En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer

        def run_optimization(self):
            """
            Ejecuta el proceso completo de Walk-Forward Optimization.
            Entrena y evalúa el agente en cada ventana.
            Retorna un DataFrame con los resultados OOS concatenados.
            """
            logger.info(" Iniciando proceso de Walk-Forward Optimization ".center(80, "="))
            
            windows = self._get_walk_forward_windows()
            if not windows:
                logger.error("No se generaron ventanas WFO. Abortando optimización.")
                return None

            all_oos_walk_results_dfs = [] # Lista para guardar DataFrames de trades/equity de cada OOS walk
            
            # Total de timesteps de entrenamiento por walk (desde config del agente)
            training_timesteps_per_walk = self.agent_config.get("total_training_timesteps", 100000) # Default
            # Podría ser más pequeño para WFO para acelerar, ej. self.agent_config.get("wfo_training_timesteps", 50000)

            for window_info in windows:
                walk_num = window_info["walk_num"]
                logger.info(f"--- Iniciando Walk #{walk_num}/{len(windows)} ---")
                logger.info(f"IS: {window_info['is_start_date'].date()} a {window_info['is_end_date'].date()} "
                            f"(Índices: {window_info['is_start_idx']} a {window_info['is_end_idx']})")
                logger.info(f"OOS: {window_info['oos_start_date'].date()} a {window_info['oos_end_date'].date()} "
                            f"(Índices: {window_info['oos_start_idx']} a {window_info['oos_end_idx']})")

                # 1. Preparar datos IS y crear archivo .npz temporal para el entorno IS
                is_data_slice = self._get_data_slice(window_info['is_start_idx'], window_info['is_end_idx'])
                is_temp_npz_path = self._prepare_temp_npz_for_env(is_data_slice, f"is_walk{walk_num}")
                
                # 2. Entrenar modelo en la ventana IS
                logger.info(f"[Walk {walk_num}] Entrenando modelo en datos IS...")
                trained_model_path = self._train_on_is_window(
                    is_env_data_path=is_temp_npz_path, 
                    walk_num=walk_num,
                    training_timesteps=training_timesteps_per_walk
                )
                if not trained_model_path:
                    logger.error(f"[Walk {walk_num}] Entrenamiento falló. Saltando a la siguiente ventana.")
                    if is_temp_npz_path.exists(): is_temp_npz_path.unlink() # Limpiar temp
                    continue
                logger.info(f"[Walk {walk_num}] Modelo entrenado y guardado en: {trained_model_path}")
                
                # Limpiar archivo .npz temporal IS
                if is_temp_npz_path.exists(): is_temp_npz_path.unlink()

                # 3. Preparar datos OOS y crear archivo .npz temporal para el entorno OOS
                oos_data_slice = self._get_data_slice(window_info['oos_start_idx'], window_info['oos_end_idx'])
                oos_temp_npz_path = self._prepare_temp_npz_for_env(oos_data_slice, f"oos_walk{walk_num}")

                # 4. Evaluar modelo en la ventana OOS
                logger.info(f"[Walk {walk_num}] Evaluando modelo en datos OOS...")
                oos_results_df = self._evaluate_on_oos_window(
                    oos_env_data_path=oos_temp_npz_path,
                    model_path=trained_model_path,
                    walk_num=walk_num,
                    oos_start_idx_global=window_info['oos_start_idx'] # Para alinear con timestamps globales
                )
                if oos_results_df is None or oos_results_df.empty:
                    logger.warning(f"[Walk {walk_num}] Evaluación OOS no produjo resultados o falló.")
                else:
                    logger.info(f"[Walk {walk_num}] Evaluación OOS completada. {len(oos_results_df)} pasos OOS registrados.")
                    all_oos_walk_results_dfs.append(oos_results_df)
                    # Guardar resultados OOS de este walk (ej. CSV)
                    oos_walk_results_file = self.wfo_oos_results_dir / f"oos_results_walk_{walk_num}.csv"
                    oos_results_df.to_csv(oos_walk_results_file, index=False)
                    logger.info(f"[Walk {walk_num}] Resultados OOS guardados en: {oos_walk_results_file}")

                # Limpiar archivo .npz temporal OOS
                if oos_temp_npz_path.exists(): oos_temp_npz_path.unlink()
                logger.info(f"--- Fin Walk #{walk_num} ---")

            if not all_oos_walk_results_dfs:
                logger.error("No se generaron resultados OOS de ningún walk. Abortando análisis.")
                return None

            # 5. Concatenar todos los resultados OOS
            concatenated_oos_results = pd.concat(all_oos_walk_results_dfs, ignore_index=True)
            concatenated_oos_results_file = self.wfo_base_output_dir / "concatenated_oos_results.csv"
            concatenated_oos_results.to_csv(concatenated_oos_results_file, index=False)
            logger.info(f"Resultados OOS de todos los walks concatenados y guardados en: {concatenated_oos_results_file}")
            
            # 6. Calcular métricas de rendimiento globales y generar reporte
            logger.info("Calculando métricas de rendimiento finales y generando reporte...")
            performance_summary = self._calculate_and_report_performance(concatenated_oos_results)
            
            logger.info(" Walk-Forward Optimization Completado ".center(80, "="))
            return concatenated_oos_results, performance_summary
        ```

-----

### Paso 4: Implementación del Método de Entrenamiento (`_train_on_is_window`)

  * **Descripción Exhaustiva**: Implementar el método `_train_on_is_window` que toma los datos In-Sample, instancia el entorno y el entrenador del agente, entrena el modelo SAC, y guarda el modelo entrenado para ese "walk".
  * **Acciones Específicas**:
      * **4.1. Definir `_train_on_is_window`**:
        ```python
        # En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer

        def _train_on_is_window(self, is_env_data_path: Path, walk_num: int, training_timesteps: int) -> Optional[Path]:
            """
            Entrena un modelo SAC en la ventana In-Sample especificada.

            Args:
                is_env_data_path (Path): Ruta al archivo .npz con los datos IS para el entorno.
                walk_num (int): Número del walk actual (para nombrar logs/modelos).
                training_timesteps (int): Número de timesteps para entrenar.

            Returns:
                Optional[Path]: Ruta al modelo entrenado guardado, o None si falla.
            """
            try:
                # 1. Crear entorno de entrenamiento con datos IS
                #    El entorno IS no necesita modo eval, puede tener inicios aleatorios si está configurado así
                is_env = TradingEnv(data_npz_path=is_env_data_path)
                # is_env.set_train_mode() # Asegurar que esté en modo entrenamiento

                # 2. Configurar SACAgentTrainer
                #    Los logs y modelos se guardarán en subdirectorios específicos del walk
                is_train_log_dir = self.wfo_base_output_dir / f"walk_{walk_num}_is_train_logs"
                is_model_save_dir = self.wfo_models_dir # Guardar todos los modelos de walk aquí

                agent_trainer = SACAgentTrainer(
                    env=is_env,
                    log_dir=is_train_log_dir,
                    model_save_dir=is_model_save_dir # El nombre del archivo incluirá el walk_num
                )
                agent_trainer.setup_model()

                # 3. Entrenar el modelo
                # (Opcional) Añadir callbacks específicos para el entrenamiento de WFO si es necesario
                # Por ejemplo, un callback para detener temprano si el rendimiento en un validation_set_is no mejora.
                run_name_tb = f"SAC_Walk{walk_num}_IS_Train"
                agent_trainer.train(total_timesteps=training_timesteps, tb_log_name=run_name_tb)
                
                # 4. Guardar el modelo entrenado
                model_name = f"sac_model_walk_{walk_num}_ts{training_timesteps}"
                agent_trainer.save_model(model_name=model_name)
                
                # Cerrar el entorno IS
                is_env.close()
                
                return is_model_save_dir / f"{model_name}.zip"

            except Exception as e:
                logger.error(f"Error durante el entrenamiento en IS para Walk #{walk_num}: {e}", exc_info=True)
                # Asegurar que el entorno se cierre si se creó
                if 'is_env' in locals() and hasattr(is_env, 'close'):
                    is_env.close() # type: ignore[possibly-undefined]
                return None
        ```

-----

### Paso 5: Implementación del Método de Evaluación (`_evaluate_on_oos_window`)

  * **Descripción Exhaustiva**: Implementar el método `_evaluate_on_oos_window` que toma los datos Out-of-Sample y el modelo entrenado, ejecuta la evaluación en modo determinista, y recolecta los resultados (P\&L diario, curva de equity, lista de trades).
  * **Acciones Específicas**:
      * **5.1. Definir `_evaluate_on_oos_window`**:
        ```python
        # En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer

        def _evaluate_on_oos_window(self, 
                                    oos_env_data_path: Path, 
                                    model_path: Path, 
                                    walk_num: int,
                                    oos_start_idx_global: int # Para alinear timestamps con el dataset completo
                                   ) -> Optional[pd.DataFrame]:
            """
            Evalúa el modelo entrenado en la ventana Out-of-Sample.

            Args:
                oos_env_data_path (Path): Ruta al archivo .npz con los datos OOS para el entorno.
                model_path (Path): Ruta al modelo SAC entrenado (.zip).
                walk_num (int): Número del walk actual.
                oos_start_idx_global (int): Índice global de inicio de esta ventana OOS, para alinear timestamps.


            Returns:
                Optional[pd.DataFrame]: DataFrame con los resultados de la evaluación OOS
                                        (ej. timestamp, equity, pnl_diario, detalles de trades),
                                        o None si falla.
            """
            try:
                # 1. Crear entorno de evaluación con datos OOS
                oos_env = TradingEnv(data_npz_path=oos_env_data_path)
                # Configurar el entorno para evaluación: inicio fijo en el primer paso de OOS
                oos_env.set_eval_mode(start_index=0) 

                # 2. Cargar el modelo entrenado
                #    Necesitamos pasarle el custom_objects con el extractor para que SB3 lo reconstruya.
                #    Los kwargs del extractor se leen de la config del agente.
                transformer_kwargs_load = {
                    "d_model": self.agent_config.get("d_model_transformer", 128),
                    "n_heads": self.agent_config.get("transformer_heads", 4),
                    "n_encoder_layers": self.agent_config.get("transformer_layers", 3),
                    "dim_feedforward": self.agent_config.get("d_model_transformer", 128) * 4,
                    "dropout": self.agent_config.get("transformer_dropout", 0.1),
                    "aggregation_method": self.agent_config.get("transformer_aggregation", "last")
                }
                if "dim_feedforward_transformer" in self.agent_config:
                    transformer_kwargs_load["dim_feedforward"] = self.agent_config["dim_feedforward_transformer"]
                
                # Importar aquí para evitar dependencia circular en la parte superior si no es necesario
                from agent.policy import TransformerFeaturesExtractor 
                
                # El SACAgentTrainer.load_model ya maneja esto, pero si llamamos a SAC.load directamente:
                # custom_objects_load = {
                #     "policy_kwargs": {
                #         "features_extractor_class": TransformerFeaturesExtractor,
                #         "features_extractor_kwargs": transformer_kwargs_load
                #     }
                # }
                # loaded_model = SAC.load(model_path, env=oos_env, custom_objects=custom_objects_load)
                
                # Usar el método de clase de SACAgentTrainer si está disponible y es conveniente
                # O simplemente SAC.load()
                from stable_baselines3 import SAC # Asegurar importación
                # No es necesario pasar policy_kwargs si el modelo se guardó correctamente
                # y las clases están disponibles en el entorno de ejecución.
                # SB3 intenta guardar los kwargs de la política, pero a veces es más robusto
                # reconstruir explícitamente o asegurar que las clases estén registradas.
                # Para máxima robustez, podríamos re-crear el policy_kwargs y pasarlo.
                policy_kwargs_for_load = dict(
                    features_extractor_class=TransformerFeaturesExtractor,
                    features_extractor_kwargs=transformer_kwargs_load,
                    net_arch=self.agent_config.get("actor_critic_hidden_dims", [256, 256])
                )
                custom_objects_for_sb3_load = {"policy_kwargs": policy_kwargs_for_load,
                                               "observation_space": oos_env.observation_space, # Ayuda a SB3
                                               "action_space": oos_env.action_space}


                loaded_model = SAC.load(model_path, env=oos_env, custom_objects=custom_objects_for_sb3_load)
                logger.info(f"[Walk {walk_num}] Modelo {model_path.name} cargado para evaluación OOS.")

                # 3. Ejecutar la evaluación
                obs, info = oos_env.reset(options={'start_index': 0}) # Asegurar inicio desde el principio de OOS
                terminated = False
                truncated = False
                
                # Listas para recolectar datos del episodio OOS
                episode_timestamps = [] # Timestamps globales de inicio de cada secuencia evaluada
                episode_equity = []
                episode_realized_pnl = [] # P&L realizado en cada paso (generalmente 0 hasta cierre)
                episode_position_changes = [] # Lista de diccionarios de trades

                # El timestamp del primer paso OOS
                current_global_timestamp_idx = oos_start_idx_global

                while not (terminated or truncated):
                    action, _states = loaded_model.predict(obs, deterministic=True)
                    prev_position = info.get("position", POSITION_NEUTRAL) # Posición antes del step

                    obs, reward, terminated, truncated, info = oos_env.step(action)
                    
                    # Guardar datos del paso
                    # El timestamp en `info` es el del inicio de la secuencia actual.
                    # Corresponde a self.full_market_timestamps_ms[current_global_timestamp_idx]
                    episode_timestamps.append(self.full_market_timestamps_ms[current_global_timestamp_idx])
                    episode_equity.append(info.get("equity"))
                    episode_realized_pnl.append(info.get("realized_pnl_episode")) # Esto es acumulativo del episodio
                                                                               # Mejor calcular P&L del paso si es posible.
                    
                    # Detectar y registrar trades (cambios de posición)
                    current_position = info.get("position")
                    if current_position != prev_position:
                        trade_info = {
                            "timestamp_ms": self.full_market_timestamps_ms[current_global_timestamp_idx],
                            "walk_num": walk_num,
                            "prev_position": prev_position,
                            "current_position": current_position,
                            "entry_price": info.get("entry_price") if current_position != POSITION_NEUTRAL else None,
                            "size": info.get("position_size_contracts") if current_position != POSITION_NEUTRAL else None,
                            "equity_after_trade": info.get("equity")
                            # Se podrían añadir detalles de P&L del trade específico si el info lo proveyera
                        }
                        episode_position_changes.append(trade_info)
                        logger.debug(f"[Walk {walk_num} OOS] Trade: {trade_info}")

                    current_global_timestamp_idx += 1 # Avanzar timestamp global
                    if current_global_timestamp_idx >= self.num_total_data_points: # Seguridad
                        logger.warning("Índice global de timestamp excedió los datos disponibles durante OOS.")
                        break

                # 4. Preparar DataFrame de resultados para este walk OOS
                results_data = {
                    "timestamp_ms": episode_timestamps,
                    "equity": episode_equity,
                    # "realized_pnl_step": np.diff(np.array([0.0] + episode_realized_pnl), prepend=0) # P&L del paso
                }
                oos_df = pd.DataFrame(results_data)
                oos_df['datetime_utc'] = pd.to_datetime(oos_df['timestamp_ms'], unit='ms', utc=True)
                
                # Guardar trades (opcional, o incluirlos en el df principal)
                if episode_position_changes:
                    trades_df = pd.DataFrame(episode_position_changes)
                    trades_df.to_csv(self.wfo_oos_results_dir / f"trades_walk_{walk_num}.csv", index=False)
                    logger.info(f"[Walk {walk_num}] {len(trades_df)} trades guardados.")

                oos_env.close()
                return oos_df

            except Exception as e:
                logger.error(f"Error durante la evaluación OOS para Walk #{walk_num}: {e}", exc_info=True)
                if 'oos_env' in locals() and hasattr(oos_env, 'close'): # type: ignore[possibly-undefined]
                    oos_env.close() # type: ignore[possibly-undefined]
                return None
        ```

-----




-----

### Paso 6: Agregación de Resultados OOS y Cálculo de Métricas de Rendimiento

  * **Descripción Exhaustiva**: Implementar el método `_calculate_and_report_performance` (anteriormente mencionado como parte de `run_optimization`). Este método tomará los resultados OOS concatenados de todos los "walks", calculará un conjunto exhaustivo de métricas de rendimiento utilizando `quantstats` y otras lógicas personalizadas si es necesario, y guardará estas métricas.
  * **Acciones Específicas**:
      * **6.1. Definir `_calculate_and_report_performance` en `WalkForwardOptimizer`**:
        ```python
        # En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer

        def _calculate_and_report_performance(self, concatenated_oos_results: pd.DataFrame) -> Dict[str, Any]:
            """
            Calcula métricas de rendimiento detalladas sobre los resultados OOS concatenados
            y genera un reporte HTML con quantstats.

            Args:
                concatenated_oos_results (pd.DataFrame): DataFrame con columnas como 'datetime_utc', 'equity'.
                                                         Debe tener el índice como datetime para quantstats.

            Returns:
                Dict[str, Any]: Un diccionario con las métricas clave calculadas.
            """
            if concatenated_oos_results.empty:
                logger.error("El DataFrame de resultados OOS concatenados está vacío. No se pueden calcular métricas.")
                return {}

            logger.info("Calculando métricas de rendimiento sobre resultados OOS concatenados...")
            
            # Preparar datos para QuantStats
            # QuantStats espera una Serie de retornos diarios, o una Serie de valores de equity.
            # Si tenemos equity, podemos calcular retornos diarios.
            # Asegurar que el índice sea DatetimeIndex
            if not isinstance(concatenated_oos_results.index, pd.DatetimeIndex):
                if 'datetime_utc' in concatenated_oos_results.columns:
                    concatenated_oos_results['datetime_utc'] = pd.to_datetime(concatenated_oos_results['datetime_utc'], utc=True)
                    concatenated_oos_results.set_index('datetime_utc', inplace=True)
                else:
                    logger.error("Columna 'datetime_utc' no encontrada o índice no es DatetimeIndex. No se puede usar QuantStats.")
                    return {}
            
            # Asegurar que no haya duplicados en el índice (podría ocurrir si los walks se solapan o por errores)
            concatenated_oos_results = concatenated_oos_results[~concatenated_oos_results.index.duplicated(keep='first')]
            concatenated_oos_results.sort_index(inplace=True) # Asegurar orden cronológico

            equity_curve = concatenated_oos_results['equity']
            
            if equity_curve.empty:
                logger.error("La curva de equity está vacía. No se pueden calcular métricas.")
                return {}

            # Calcular retornos diarios para QuantStats
            # (QuantStats también puede tomar la curva de equity directamente con qs.stats.daily(equity_curve))
            # daily_returns = equity_curve.resample('D').last().pct_change().fillna(0) # Resamplear a diario
            # O, si los datos ya son por paso y queremos usar la serie de equity directamente:
            # QuantStats puede inferir retornos de una serie de equity.
            
            # Usar una copia para evitar modificar el original si qs lo hace
            qs_equity_series = equity_curve.copy()

            # Generar reporte HTML de QuantStats
            report_title = f"WFO Performance Report - {self.trading_pair} ({self.kline_interval}) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            report_file_path = self.wfo_base_output_dir / f"wfo_quantstats_report_{self.trading_pair}.html"
            
            try:
                logger.info(f"Generando reporte de QuantStats en: {report_file_path}")
                # qs.reports.html(returns=daily_returns, output=str(report_file_path), title=report_title)
                # O usando la serie de equity directamente (más simple si los datos no son estrictamente diarios)
                # Asegurarse que el índice de qs_equity_series sea un DatetimeIndex
                if not isinstance(qs_equity_series.index, pd.DatetimeIndex):
                     qs_equity_series.index = pd.to_datetime(qs_equity_series.index, utc=True)

                qs.reports.html(
                    returns=qs_equity_series, # Pasar la serie de equity directamente
                    output=str(report_file_path), 
                    title=report_title,
                    # benchmark_prices=None, # Opcional: añadir un benchmark como SPY
                    # compounded=True (default)
                )
                logger.info(f"Reporte de QuantStats generado: {report_file_path}")
            except Exception as e:
                logger.error(f"Error al generar el reporte de QuantStats: {e}", exc_info=True)

            # Calcular y extraer métricas específicas
            # QuantStats devuelve muchas métricas; podemos seleccionar algunas clave.
            # metrics_series = qs.stats.all(daily_returns) # Si se usan retornos
            # O con la serie de equity:
            metrics_dict = {}
            try:
                metrics_dict['CAGR'] = qs.stats.cagr(qs_equity_series)
                metrics_dict['Sharpe Ratio'] = qs.stats.sharpe(qs_equity_series, rf=self.wfo_config.get("risk_free_rate_for_sharpe_sortino", 0.0))
                metrics_dict['Sortino Ratio'] = qs.stats.sortino(qs_equity_series, rf=self.wfo_config.get("risk_free_rate_for_sharpe_sortino", 0.0))
                metrics_dict['Max Drawdown'] = qs.stats.max_drawdown(qs_equity_series)
                # metrics_dict['Win Rate'] = qs.stats.win_rate(daily_returns) # Requiere retornos o trades
                metrics_dict['Profit Factor'] = qs.stats.profit_factor(qs_equity_series.pct_change().fillna(0)) # Requiere retornos
                # ...y otras métricas que se deseen del output de qs.reports.metrics(returns=..., display=False)

                # Calcular Win Rate y Avg Win/Loss si tenemos info de trades
                # Esto requeriría que _evaluate_on_oos_window retorne o guarde los trades individuales con P&L.
                # Por ahora, estas métricas se omiten si no se han recolectado trades detallados.
                # El README las pide, así que se debería mejorar la recolección de datos en _evaluate_on_oos_window.

                logger.info("Métricas de rendimiento calculadas:")
                for key, value in metrics_dict.items():
                    logger.info(f"  {key}: {value:.4f}")
                
                # Guardar métricas en un archivo JSON
                metrics_file_path = self.wfo_base_output_dir / "wfo_performance_metrics.json"
                import json
                with open(metrics_file_path, 'w') as f:
                    # Convertir tipos NumPy a Python nativo para JSON
                    serializable_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics_dict.items()}
                    json.dump(serializable_metrics, f, indent=4)
                logger.info(f"Métricas guardadas en: {metrics_file_path}")

            except Exception as e:
                logger.error(f"Error al calcular métricas específicas con QuantStats: {e}", exc_info=True)

            return metrics_dict
        ```
      * **Nota sobre Métricas de Trades (Win Rate, Avg Win/Loss, Profit Factor):** Para calcular estas métricas con precisión, se necesita una lista detallada de cada trade individual (entrada, salida, P\&L por trade). El método `_evaluate_on_oos_window` actual guarda un log de `episode_position_changes`. Este log debería procesarse para extraer el P\&L de cada trade cerrado y luego calcular estas métricas. `quantstats` puede calcular `Profit Factor` a partir de retornos, pero `Win Rate` y `Avg Win/Loss` son mejores con datos de trades.

-----

### Paso 7: Generación de Visualizaciones Adicionales (Opcional)

  * **Descripción Exhaustiva**: Además del reporte de `quantstats`, se pueden generar gráficos específicos usando Matplotlib/Seaborn/Plotly si se requieren visualizaciones personalizadas (ej. distribución de retornos OOS, drawdown sobre tiempo).
  * **Acciones Específicas**:
      * **7.1. Añadir a `_calculate_and_report_performance` o un método separado**:
        ```python
        # En src/backtesting/wfo_framework.py, dentro de la clase WalkForwardOptimizer
        # (Puede ser parte de _calculate_and_report_performance o un nuevo método)

        # ... después de calcular la curva de equity_curve ...
        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # # 1. Curva de Equity OOS
        # plt.figure(figsize=(12, 6))
        # equity_curve.plot(title='Out-of-Sample Equity Curve')
        # plt.xlabel('Date')
        # plt.ylabel('Equity')
        # equity_curve_path = self.wfo_base_output_dir / "wfo_oos_equity_curve.png"
        # plt.savefig(equity_curve_path)
        # plt.close()
        # logger.info(f"Gráfico de curva de equity guardado en: {equity_curve_path}")

        # # 2. Distribución de Retornos Diarios OOS (si se calcularon)
        # if 'daily_returns' in locals() and not daily_returns.empty: # type: ignore[possibly-undefined]
        #     plt.figure(figsize=(10, 6))
        #     sns.histplot(daily_returns, kde=True, bins=50) # type: ignore[possibly-undefined]
        #     plt.title('Distribution of OOS Daily Returns')
        #     plt.xlabel('Daily Return')
        #     plt.ylabel('Frequency')
        #     returns_dist_path = self.wfo_base_output_dir / "wfo_oos_returns_distribution.png"
        #     plt.savefig(returns_dist_path)
        #     plt.close()
        #     logger.info(f"Gráfico de distribución de retornos guardado en: {returns_dist_path}")
        ```
      * **Nota**: El reporte HTML de `quantstats` ya incluye muchas de estas visualizaciones, por lo que generar gráficos separados podría ser redundante a menos que se necesite un formato o personalización específicos. Por simplicidad del MVP, se puede confiar principalmente en el reporte de `quantstats`.

-----

### Paso 8: Script de Orquestación para Backtesting WFO (`scripts/run_backtest.py`)

  * **Descripción Exhaustiva**: Crear un script que instancie y ejecute el `WalkForwardOptimizer`.
  * **Acciones Específicas**:
      * **8.1. Crear `scripts/run_backtest.py`**:
        ```python
        # scripts/run_backtest.py
        import sys
        import logging
        from pathlib import Path
        import argparse

        # Añadir src al PYTHONPATH
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from common.utils import setup_logging
        from backtesting.wfo_framework import WalkForwardOptimizer
        # (Otras importaciones si son necesarias, ej. para configurar algo específico)

        try:
            setup_logging()
        except Exception as e:
            logging.basicConfig(level=logging.ERROR)
            logging.critical(f"Fallo CRÍTICO al configurar logging: {e}", exc_info=True)

        logger = logging.getLogger(__name__)

        def main(args):
            logger.info("===========================================================")
            logger.info("Iniciando script de Backtesting Walk-Forward Optimization...")
            logger.info("===========================================================")
            logger.info(f"Argumentos recibidos: {args}") # Para futuras extensiones

            try:
                # Instanciar y ejecutar el optimizador WFO
                # Las configuraciones se cargan dentro de WalkForwardOptimizer desde los YAMLs
                wfo_optimizer = WalkForwardOptimizer()
                
                # Ejecutar la optimización WFO
                # Esto entrenará modelos, evaluará en OOS, y generará reportes/métricas.
                concatenated_oos_results, performance_summary = wfo_optimizer.run_optimization()

                if concatenated_oos_results is not None and performance_summary:
                    logger.info("Backtesting WFO completado exitosamente.")
                    logger.info(f"Resumen de Rendimiento: {performance_summary}")
                    logger.info(f"Resultados OOS concatenados disponibles en el directorio de salida: {wfo_optimizer.wfo_base_output_dir}")
                else:
                    logger.error("El backtesting WFO falló o no produjo resultados.")
                
            except FileNotFoundError as e_fnf:
                logger.error(f"Error de archivo no encontrado durante el backtesting: {e_fnf}. "
                             "Asegúrate de que los datos procesados (.npz) existan.", exc_info=True)
            except EnvironmentError as e_env:
                logger.error(f"Error de configuración de entorno: {e_env}", exc_info=True)
            except Exception as e:
                logger.error(f"Error inesperado durante el script de backtesting WFO: {e}", exc_info=True)
            finally:
                logger.info("===========================================================")
                logger.info("Finalización del script de Backtesting WFO.")
                logger.info("===========================================================")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description="Ejecuta el framework de Walk-Forward Optimization.")
            # Añadir argumentos si WFO necesita parámetros en tiempo de ejecución,
            # por ejemplo, para seleccionar un conjunto específico de configuraciones YAML
            # o para sobreescribir algún parámetro del WFO.
            # parser.add_argument("--config_set", type=str, default="default", help="Conjunto de configuración a usar.")
            
            script_args = parser.parse_args()
            main(script_args)
        ```
      * **8.2. Integrar en `scripts/run_pipeline.py` (Opcional)**:
          * Se puede añadir una opción a `run_pipeline.py` para ejecutar `run_backtest.py` después del entrenamiento (o en lugar de un entrenamiento simple).
          * Ejemplo en `run_pipeline.py`:
            ```python
            # ... (en run_pipeline.py)
            # TRAIN_SCRIPT = SCRIPTS_DIR / "train_agent.py"
            BACKTEST_SCRIPT = SCRIPTS_DIR / "run_backtest.py"
            # ...
            # parser.add_argument("--run_wfo_backtest", action="store_true", help="Ejecutar WFO backtesting en lugar de entrenamiento simple.")
            # ...
            # if args.run_wfo_backtest:
            #     logger.info("--- Fase 3b: Backtesting Walk-Forward Optimization ---")
            #     if not run_script(BACKTEST_SCRIPT, script_args=backtest_args_list): # Pasar args si es necesario
            #         logger.error("El Backtesting WFO falló.")
            #     else:
            #         logger.info("--- Fase 3b: Backtesting WFO Completado ---")
            # elif not args.skip_train:
            #     # ... lógica de entrenamiento simple ...
            ```
      * **8.3. Ejecutar el Script de Backtesting**:
          * Asegurar que los datos procesados (`.npz`) existan.
          * Ejecutar dentro de Docker:
            ```bash
            # docker-compose up -d --build workhorse_app redis # Si no está corriendo
            docker-compose exec workhorse_app python scripts/run_backtest.py
            ```
          * Este proceso será largo, ya que implica múltiples ciclos de entrenamiento y evaluación.
          * Monitorear logs. Verificar la creación de directorios y archivos en `results_host/wfo_runs/`.

-----

### Paso 9: Pruebas Unitarias (Básicas) para `WalkForwardOptimizer`

  * **Descripción Exhaustiva**: Crear pruebas unitarias para los aspectos clave de `WalkForwardOptimizer`, como la lógica de generación de ventanas (`_get_walk_forward_windows`) y, si es posible, simular un "walk" muy pequeño con mocks para verificar el flujo.
  * **Acciones Específicas**:
      * **9.1. Crear `tests/backtesting/test_wfo_framework.py`**:
        ```python
        # tests/backtesting/test_wfo_framework.py
        import pytest
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from unittest import mock
        from datetime import datetime, timezone
        from dateutil.relativedelta import relativedelta

        from src.backtesting.wfo_framework import WalkForwardOptimizer
        # Mockear dependencias como TradingEnv, SACAgentTrainer si se testea run_optimization

        @pytest.fixture
        def mock_wfo_configs(monkeypatch, tmp_path: Path):
            """Mockea configs y variables de entorno para WalkForwardOptimizer."""
            # Mock get_env_variable
            test_data_dir = tmp_path / "wfo_test_data_persistent"
            test_results_dir = tmp_path / "wfo_test_results_persistent"
            
            def mock_get_env(var_name, default_value=None, required=True):
                if var_name == "DATA_DIR_HOST_FOR_APP": return str(test_data_dir)
                if var_name == "RESULTS_DIR_HOST_FOR_APP": return str(test_results_dir)
                return "dummy_env_val"
            monkeypatch.setattr("src.backtesting.wfo_framework.get_env_variable", mock_get_env)

            # Mock load_yaml_config
            def mock_load_yaml(module_name, file_name="params.yaml"):
                if module_name == "module7_backtesting_wfo":
                    return {"wfo_is_window_months": 6, "wfo_oos_window_months": 2, 
                            "wfo_step_months": 2, "wfo_window_type": "rolling",
                            "risk_free_rate_for_sharpe_sortino": 0.0}
                if module_name == "module4_agent_sac": # Usado por _train_on_is_window
                    return {"total_training_timesteps": 100} # Timesteps muy bajos para test
                if module_name == "module2_preprocessing":
                    return {"sequence_length_L": 10} # L pequeña
                if module_name == "module1_data_acquisition":
                    return {"trading_pair": "TESTBTC", "kline_interval": "1h"}
                return {}
            monkeypatch.setattr("src.backtesting.wfo_framework.load_yaml_config", mock_load_yaml)

            # Crear estructura de directorios y archivo .npz de prueba
            pair_test = "TESTBTC"
            interval_test = "1h"
            L_test = 10
            N_market_feat_test = 20
            
            npz_save_path_dir = test_data_dir / "processed" / pair_test / interval_test
            npz_save_path_dir.mkdir(parents=True, exist_ok=True)
            
            # Simular datos para 12 meses para WFO
            num_months_sim_wfo = 12
            start_date_wfo = datetime(2022, 1, 1, tzinfo=timezone.utc)
            samples_per_month_approx = 30 * 24 # Para klines horarios
            num_samples_wfo = num_months_sim_wfo * samples_per_month_approx

            sequences = np.random.rand(num_samples_wfo, L_test, N_market_feat_test).astype(np.float32)
            timestamps = np.array([
                int((start_date_wfo + relativedelta(hours=i)).timestamp() * 1000) 
                for i in range(num_samples_wfo)
            ]).astype(np.int64)
            env_closes = np.linspace(100, 150, num_samples_wfo).astype(np.float32)
            env_atrs = np.full(num_samples_wfo, 1.0).astype(np.float32)

            npz_file = npz_save_path_dir / f"{pair_test}_{interval_test}_L{L_test}_N{N_market_feat_test}_processed_sequences.npz"
            np.savez_compressed(npz_file, sequences=sequences, timestamps=timestamps,
                                env_step_close_prices=env_closes, env_step_atr_values=env_atrs)
            return npz_file # Devuelve la ruta al archivo npz creado

        def test_wfo_initialization(mock_wfo_configs, sample_npz_data_for_env): # sample_npz_data_for_env es el fixture del test de TradingEnv
            # Aquí sample_npz_data_for_env no es el fixture correcto, es el de arriba
            # Necesitamos que el fixture mock_wfo_configs ya cree el archivo.
            # El fixture se llama `mock_wfo_configs` y devuelve la ruta al npz.
            _ = WalkForwardOptimizer() # Debería inicializar sin errores
            # Se pueden añadir asserts sobre los parámetros cargados si se desea

        def test_get_walk_forward_windows_rolling(mock_wfo_configs):
            # mock_wfo_configs ya crea el archivo .npz necesario.
            wfo = WalkForwardOptimizer()
            windows = wfo._get_walk_forward_windows()
            
            assert len(windows) > 0 # Debería generar al menos algunas ventanas
            # Ejemplo: IS=6m, OOS=2m, Step=2m, Total=12m
            # Walk 1: IS [M1-M6], OOS [M7-M8]
            # Walk 2: IS [M3-M8], OOS [M9-M10]
            # Walk 3: IS [M5-M10], OOS [M11-M12]
            # Deberían ser 3 walks
            assert len(windows) == 3 
            
            # Verificar la primera ventana
            walk1 = windows[0]
            assert walk1['walk_num'] == 1
            assert (walk1['is_end_date'].year == 2022 and walk1['is_end_date'].month == 6) # Fin Junio 2022
            assert (walk1['oos_start_date'].year == 2022 and walk1['oos_start_date'].month == 7) # Inicio Julio 2022
            assert (walk1['oos_end_date'].year == 2022 and walk1['oos_end_date'].month == 8) # Fin Agosto 2022

            # Verificar la última ventana
            walk_last = windows[-1]
            assert (walk_last['is_start_date'].year == 2022 and walk_last['is_start_date'].month == 5) # Inicio Mayo 2022
            assert (walk_last['oos_end_date'].year == 2022 and walk_last['oos_end_date'].month == 12) # Fin Dic 2022

        def test_get_walk_forward_windows_expanding(mock_wfo_configs, monkeypatch):
            # Modificar la config para que sea expanding
            def mock_load_yaml_expanding(module_name, file_name="params.yaml"):
                if module_name == "module7_backtesting_wfo":
                    return {"wfo_is_window_months": 6, "wfo_oos_window_months": 2, 
                            "wfo_step_months": 2, "wfo_window_type": "expanding", # CAMBIADO
                            "risk_free_rate_for_sharpe_sortino": 0.0}
                if module_name == "module4_agent_sac": return {"total_training_timesteps": 100}
                if module_name == "module2_preprocessing": return {"sequence_length_L": 10}
                if module_name == "module1_data_acquisition": return {"trading_pair": "TESTBTC", "kline_interval": "1h"}
                return {}
            monkeypatch.setattr("src.backtesting.wfo_framework.load_yaml_config", mock_load_yaml_expanding)
            
            wfo = WalkForwardOptimizer()
            windows = wfo._get_walk_forward_windows()

            assert len(windows) == 3 # Misma cantidad de walks
            
            # Verificar la primera ventana (igual que rolling)
            walk1 = windows[0]
            assert (walk1['is_start_date'].year == 2022 and walk1['is_start_date'].month == 1) # Siempre Enero 2022
            assert (walk1['is_end_date'].year == 2022 and walk1['is_end_date'].month == 6) 
            assert (walk1['oos_start_date'].year == 2022 and walk1['oos_start_date'].month == 7) 
            assert (walk1['oos_end_date'].year == 2022 and walk1['oos_end_date'].month == 8)

            # Verificar la última ventana (IS start debe ser Enero 2022)
            walk_last = windows[-1]
            assert (walk_last['is_start_date'].year == 2022 and walk_last['is_start_date'].month == 1) # Expanding
            assert (walk_last['is_end_date'].year == 2022 and walk_last['is_end_date'].month == 10) # IS end es OOS_start (Nov) - 1 dia
            assert (walk_last['oos_start_date'].year == 2022 and walk_last['oos_start_date'].month == 11) # OOS Start Nov 2022
            assert (walk_last['oos_end_date'].year == 2022 and walk_last['oos_end_date'].month == 12) # OOS End Dic 2022

        # Testear run_optimization es más un test de integración.
        # Se podría mockear _train_on_is_window y _evaluate_on_oos_window.
        @mock.patch("src.backtesting.wfo_framework.WalkForwardOptimizer._train_on_is_window")
        @mock.patch("src.backtesting.wfo_framework.WalkForwardOptimizer._evaluate_on_oos_window")
        @mock.patch("src.backtesting.wfo_framework.WalkForwardOptimizer._calculate_and_report_performance")
        def test_run_optimization_flow(self, mock_report, mock_evaluate, mock_train, mock_wfo_configs):
            mock_train.return_value = Path("/fake/model/path.zip") # Simular que el entrenamiento fue exitoso
            # Simular que la evaluación devuelve un DataFrame
            mock_evaluate.return_value = pd.DataFrame({"timestamp_ms": [1,2,3], "equity": [100,101,102]})
            mock_report.return_value = {"Sharpe Ratio": 1.5} # Simular métricas

            wfo = WalkForwardOptimizer()
            results_df, performance = wfo.run_optimization()

            assert mock_train.call_count == 3 # 3 walks esperados
            assert mock_evaluate.call_count == 3
            assert mock_report.called_once
            assert results_df is not None
            assert not results_df.empty
            assert "Sharpe Ratio" in performance
        ```
      * **9.2. Ejecutar las Pruebas**:
        ```bash
        python -m pytest tests/backtesting/test_wfo_framework.py --cov=src/backtesting
        ```

-----

### Paso 10: Commit de los Cambios de la Fase 7

  * **Descripción Exhaustiva**: Añadir todos los cambios realizados durante esta fase al control de versiones Git.
  * **Acciones Específicas**:
      * **10.1. Añadir Archivos y Hacer Commit**:
        ```bash
        git add src/backtesting/wfo_framework.py src/backtesting/__init__.py
        git add scripts/run_backtest.py
        git add config/module7_backtesting_wfo/params.yaml # Si se modificó
        git add tests/backtesting/test_wfo_framework.py
        # git add . # Si se prefiere añadir todo lo modificado
        git commit -m "Fase 7: Implementar Módulo 7 (Backtesting WFO). Clase WalkForwardOptimizer con división de datos, bucle WFO, entrenamiento/evaluación por walk, cálculo de métricas y reporte. Incluye script y tests."
        ```

**Fin de la Fase de Implementación 7.**

-----