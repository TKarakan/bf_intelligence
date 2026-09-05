FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest pandas jupyter notebook
RUN pip install --no-cache-dir mlflow

EXPOSE 5000


RUN mkdir -p data/processed models logs outputs/results


COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY app/ ./app/
COPY notebooks/ ./notebooks/


ENV PYTHONPATH="/app"


CMD ["pytest", "app/testing/test_utils.py"]