FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Usa el requirements.txt normal y muéstralo en el build
COPY requirements.txt /app/requirements.txt
RUN echo "===== REQUIREMENTS INSTALADOS =====" \
 && cat /app/requirements.txt \
 && python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY server.py transcribe.py ./

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "${PORT}", "--proxy-headers", "--forwarded-allow-ips", "*", "--log-level", "info", "--access-log"]
