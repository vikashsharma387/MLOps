from fastapi.testclient import TestClient
from app.main import app, load_model

# Ensure model is initialized
load_model()
client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "service" in data
    assert data["metrics"] == "/metrics"


def test_health_check_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "http_requests" in res.text or "process_cpu_seconds" in res.text


def test_predict_success_positive_case():
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_disease"] <= 1.0
    assert data["risk_level"] in ["HIGH RISK", "LOW RISK"]


def test_predict_success_negative_case():
    payload = {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "confidence" in data


def test_predict_validation_error_missing_field():
    incomplete_payload = {
        "age": 52,
        "sex": 1
    }
    res = client.post("/predict", json=incomplete_payload)
    assert res.status_code == 422


def test_predict_validation_error_out_of_bounds():
    invalid_payload = {
        "age": -10,  # Invalid age
        "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    }
    res = client.post("/predict", json=invalid_payload)
    assert res.status_code == 422
