FROM python:3.12-slim

WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt /app/
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Variables de entorno predeterminadas
ENV PYTHONPATH=/app
ENV AIP_MODEL_DIR=/tmp/model

# Punto de entrada para entrenar
ENTRYPOINT ["python", "scripts/train_rl_agent.py"]
