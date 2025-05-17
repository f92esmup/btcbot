FROM python:3.12-slim

WORKDIR /app

# Instalar herramientas de compilación y dependencias para ta-lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    cmake \
    libssl-dev \
    autoconf \
    automake \
    pkg-config \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    ldconfig # Actualiza el caché del enlazador dinámico

# Crear enlaces simbólicos para asegurar que ta-lib se encuentre correctamente
RUN ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so

# Variables de entorno para ayudar a CMake a encontrar OpenSSL (para la compilación del paquete Python 'cmake')
ENV OPENSSL_ROOT_DIR=/usr
ENV OPENSSL_INCLUDE_DIRS=/usr/include/openssl
ENV OPENSSL_LIBRARIES=/usr/lib/x86_64-linux-gnu

# Copiar archivos necesarios
COPY requirements.txt /app/
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    echo "INFO: Iniciando instalación de requirements.txt" && \
    # Primero instala numpy que es necesario para la compilación de TA-Lib
    pip install --no-cache-dir numpy==1.26.4 && \
    (pip install --no-cache-dir -r requirements.txt || \
    (echo "ERROR: Falló la instalación directa de requirements.txt. Intentando TA-Lib por separado..." && \
     echo "INFO: Instalando TA-Lib (Python wrapper)..." && \
     pip install --no-cache-dir --no-binary :all: TA-Lib==0.6.3 && \
     echo "INFO: Reintentando instalar requirements.txt..." && \
     pip install --no-cache-dir -r requirements.txt))

# Variables de entorno predeterminadas
ENV PYTHONPATH=/app
ENV AIP_MODEL_DIR=/tmp/model

# Punto de entrada para entrenar
ENTRYPOINT ["python", "scripts/train_rl_agent.py"]
