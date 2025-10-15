# Imagen base ligera con Python 3.10
FROM python:3.10-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Paquetes del sistema y ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copia proyecto
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Render asigna el puerto en la var de entorno PORT
CMD ["bash", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT"]
