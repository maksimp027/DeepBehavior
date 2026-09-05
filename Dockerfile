FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY config.yaml /app/config.yaml
COPY src/ /app/src/
COPY data/extract.py /app/data/extract.py
COPY data/build_features.py /app/data/build_features.py

RUN mkdir -p /app/data/raw /app/data/interim /app/data/processed /app/models

CMD ["sh", "-c", "python -m src.data.extract && python -m src.data.build_features && python -m src.models.train && python -m src.recsys.cluster"]
