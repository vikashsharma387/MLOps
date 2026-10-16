import os
import joblib
import numpy as np
import pandas as pd
import pytest
from src.preprocess import build_preprocessor, load_and_split_data, FEATURE_COLUMNS


@pytest.fixture
def sample_features():
    return pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }])


def test_data_loader_valid():
    data_path = os.path.join("data", "heart.csv")
    if os.path.exists(data_path):
        X, y = load_and_split_data(data_path)
        assert len(X) == len(y)
        assert list(X.columns) == FEATURE_COLUMNS
        assert set(y.unique()).issubset({0, 1})


def test_preprocessor_pipeline(sample_features):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_features)
    assert transformed.shape == (1, len(FEATURE_COLUMNS))
    assert not np.isnan(transformed).any()


def test_model_artifact_load_and_predict(sample_features):
    model_path = os.path.join("models", "heart_model.pkl")
    assert os.path.exists(model_path), "Trained model artifact heart_model.pkl must exist"

    model = joblib.load(model_path)
    pred = model.predict(sample_features)
    prob = model.predict_proba(sample_features)

    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0][0] <= 1.0
    assert 0.0 <= prob[0][1] <= 1.0
    assert np.isclose(prob[0].sum(), 1.0)
