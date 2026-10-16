# MLOps End-to-End Pipeline — Heart Disease Risk Prediction
**Course:** AIMLCZG523 — Machine Learning Operations  
**Architecture:** MLflow + FastAPI + Docker + Kubernetes + Prometheus/Grafana + GitHub Actions CI/CD

---

## 📌 Project Overview
This repository implements an end-to-end, production-ready MLOps pipeline for predicting heart disease risk from patient clinical parameters using the UCI Heart Disease dataset.

### Core Features
* **Data & Feature Engineering:** Modular `ColumnTransformer` preprocessing with median imputation and standard scaling.
* **Experiment Tracking:** Integrated **MLflow** logging for hyperparameters, cross-validation metrics, confusion matrices, and serialized model artifacts.
* **Packaging & Reproducibility:** Full sklearn `Pipeline` exported as `models/heart_model.pkl` and MLflow model flavor.
* **Automated CI/CD:** GitHub Actions workflow executing code linting (`flake8`), unit tests with coverage (`pytest`), automated training, and Docker smoke test.
* **Containerization:** Isolated, production-grade `Dockerfile` packaging FastAPI with health probes.
* **Kubernetes Deployment:** Declarative manifests for Deployment, Service (NodePort/LoadBalancer), Ingress, and HorizontalPodAutoscaler (HPA).
* **Monitoring & Observability:** Structured JSON request logging + Prometheus metrics (`/metrics`) and Grafana monitoring stack.

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD pipeline
├── data/
│   ├── download_data.py          # Data acquisition script
│   └── heart.csv                 # UCI Heart Disease dataset
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Feature preprocessing & ColumnTransformer
│   └── train.py                  # MLflow experiment training script
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI serving application & /metrics
│   └── schemas.py                # Pydantic request/response schemas
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py   # Unit tests for preprocessing & model
│   └── test_api.py               # Unit tests for FastAPI endpoints
├── k8s/
│   ├── deployment.yaml           # Kubernetes Deployment (2 replicas + probes)
│   ├── service.yaml              # Kubernetes NodePort/LoadBalancer Service
│   ├── ingress.yaml              # Ingress routing configuration
│   └── hpa.yaml                  # Horizontal Pod Autoscaler
├── monitoring/
│   ├── prometheus.yml            # Prometheus scrape configuration
│   └── docker-compose.monitoring.yml # Prometheus & Grafana stack
├── models/
│   └── heart_model.pkl           # Trained scikit-learn pipeline artifact
├── Dockerfile                    # Container definition
├── requirements.txt              # Pinned dependencies
└── README.md
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data & Train with MLflow
```bash
python data/download_data.py
PYTHONPATH=. python src/train.py
```
* **MLflow UI:** Launch MLflow dashboard to view experiment runs and parameters:
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```
  Open `http://localhost:5000` to inspect runs, metrics (`ROC-AUC`, `Accuracy`, `F1-Score`), and model artifacts.

### 3. Run Automated Unit Tests & Linting
```bash
# Code Style Check
flake8 src/ app/ tests/ data/ --count --max-line-length=120 --statistics

# Unit Tests with Coverage Report
PYTHONPATH=. pytest tests/ -v --cov=src --cov=app
```

### 4. Run the API Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Swagger Documentation:** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/health`
* **Prometheus Metrics:** `http://localhost:8000/metrics`

---

## 🐳 Docker Containerization (Task 6)

### Build and Run Container
```bash
# Build Docker Image
docker build -t heart-disease-api:latest .

# Run Container
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

# Test Endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

---

## ☸️ Kubernetes Deployment (Task 7)

```bash
# Apply manifests to cluster (Minikube / GKE / EKS / AKS)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify Deployment
kubectl get pods -l app=heart-disease-api
kubectl get svc heart-disease-api-service
kubectl get hpa heart-disease-api-hpa
```

---

## 📊 Monitoring & Observability (Task 8)

Start Prometheus & Grafana stack:
```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```
* **Prometheus:** `http://localhost:9090` (Target: `heart-disease-prediction-api`)
* **Grafana:** `http://localhost:3000` (Credentials: `admin` / `admin`)
* **Tracked Metrics:**
  * `model_predictions_total`: Number of predictions served per risk class
  * `model_prediction_latency_seconds`: Latency distribution of model inference
  * `http_requests_total`: Request volume and HTTP response codes
