import json
import logging
import os
import time
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import HeartDiseaseInput, PredictionResponse

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [app] %(message)s'
)
logger = logging.getLogger("heart_disease_api")

# Custom Prometheus Metrics
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of predictions served",
    ["predicted_class", "risk_level"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time spent running model prediction pipeline",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/heart_model.pkl")
model = None


def load_model():
    """Helper function to load the serialized machine learning model."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Successfully loaded production model artifact from {MODEL_PATH}")
        else:
            logger.warning(f"Model artifact not found at {MODEL_PATH}.")
    except Exception as e:
        logger.error(f"Failed to load model artifact: {e}")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    load_model()
    yield
    # Shutdown
    logger.info("Application shutdown.")


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production MLOps Service for Heart Disease Classification with Prometheus Monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument Prometheus Metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        json.dumps({
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 4),
            "client_host": request.client.host if request.client else "unknown"
        })
    )
    return response


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Heart Disease Prediction Service",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: HeartDiseaseInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Please train or deploy model artifact."
        )

    t0 = time.time()
    try:
        input_data = payload.model_dump()
        input_df = pd.DataFrame([input_data])

        # Inference
        pred_class = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        prob_disease = float(probabilities[1])
        confidence = float(max(probabilities))

        risk_level = "HIGH RISK" if pred_class == 1 else "LOW RISK"

        # Record custom metrics
        latency = time.time() - t0
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(predicted_class=str(pred_class), risk_level=risk_level).inc()

        logger.info(
            f"Prediction completed | Class: {pred_class} | Probability: {prob_disease:.4f} | "
            f"Risk: {risk_level} | Latency: {latency*1000:.2f}ms"
        )

        return PredictionResponse(
            prediction=pred_class,
            confidence=round(confidence, 4),
            probability_disease=round(prob_disease, 4),
            risk_level=risk_level,
            status="success"
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
