FROM python:3.10-slim

LABEL maintainer="AIMLCZG523-Student"
LABEL description="Heart Disease Prediction API with MLflow Model & Prometheus Monitoring"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, model artifacts, and preprocessor
COPY app/ app/
COPY src/ src/
COPY models/ models/

# Set Environment Variables
ENV MODEL_PATH="models/heart_model.pkl"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
