FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libs del SO para audio
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 👇 copiamos *el nuevo nombre* y mostramos su contenido
COPY app.requirements.txt /app/requirements.txt
RUN echo "===== REQUIREMENTS THAT WILL BE INSTALLED =====" \
 && cat /app/requirements.txt \
 && python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY server.py transcribe.py ./

# logs de acceso + proxy headers
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "${PORT}", "--proxy-headers", "--forwarded-allow-ips", "*", "--log-level", "info", "--access-log"]
