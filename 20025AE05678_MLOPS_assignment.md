# MLOps Assignment 1 — Notebook Implementation
**Student ID:** 20025AE05678  
**Notebook File:** `20025AE05678_MLOPS_assignment.ipynb`  
**Problem Statement:** Heart Disease Prediction Pipeline with MLflow Tracking, Packaging, and FastAPI Deployment

---

## Table of Contents
1. [Environment Setup & Dependencies](#1-environment-setup--dependencies)
2. [Imports & Configuration](#2-imports--configuration)
3. [Data Loading & Initial Exploration](#3-data-loading--initial-exploration)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Baseline Model Development & Comparison](#5-baseline-model-development--comparison)
6. [Complete Production Training Pipeline with MLflow](#6-complete-production-training-pipeline-with-mlflow)
7. [Model Serialization & Export](#7-model-serialization--export)
8. [FastAPI Model Serving API](#8-fastapi-model-serving-api)
9. [Unit Testing](#9-unit-testing)
10. [Automated API Code Generation](#10-automated-api-code-generation)

---

## 1. Environment Setup & Dependencies

Install required libraries for data manipulation, visualization, modeling, tracking, and API serving.

```python
!pip install -q \
pandas \
numpy \
matplotlib \
seaborn \
scikit-learn \
mlflow \
fastapi \
uvicorn \
joblib
```

---

## 2. Imports & Configuration

Import necessary Python packages for data science, pipeline creation, and serving.

```python
import os
import json
import joblib
import mlflow

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)
```

---

## 3. Data Loading & Initial Exploration

Load the UCI Heart Disease dataset (`heart.csv`) and examine structure, column types, and missing values.

```python
import pandas as pd

df = pd.read_csv("heart.csv")
df.head()
```

### Dataset Preview (`df.head()`):
| age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|:---:|:---:|:--:|:--------:|:----:|:---:|:-------:|:-------:|:-----:|:-------:|:-----:|:--:|:----:|:------:|
| 52  | 1   | 0  | 125      | 212  | 0   | 1       | 168     | 0     | 1.0     | 2     | 2  | 3    | 0      |
| 53  | 1   | 0  | 140      | 203  | 1   | 0       | 155     | 1     | 3.1     | 0     | 0  | 3    | 0      |
| 70  | 1   | 0  | 145      | 174  | 0   | 1       | 125     | 1     | 2.6     | 0     | 0  | 3    | 0      |
| 61  | 1   | 0  | 148      | 203  | 0   | 1       | 161     | 0     | 0.0     | 2     | 1  | 3    | 0      |
| 62  | 0   | 0  | 138      | 294  | 1   | 1       | 106     | 0     | 1.9     | 1     | 3  | 2    | 0      |

### Dataset Structure & Missing Value Check
```python
df.info()
df.describe()
df.isnull().sum()
```

* **Total Samples:** 1025 entries
* **Features:** 13 input features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`)
* **Target:** `target` (0 = No Heart Disease, 1 = Heart Disease)
* **Missing Values:** `0` across all columns

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Target Class Distribution
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()
```

### 4.2 Correlation Heatmap
```python
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.show()
```

### 4.3 Feature Histograms
```python
df.hist(
    figsize=(15, 10),
    bins=20
)
plt.tight_layout()
plt.show()
```

---

## 5. Baseline Model Development & Comparison

### 5.1 Train / Test Split and Feature Scaling
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 5.2 Model 1: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
```

### 5.3 Model 2: Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
```

### 5.4 Baseline Random Forest Evaluation
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

pred = rf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("ROC AUC  :", roc_auc_score(y_test, pred))
```

#### Baseline Results:
* **Accuracy:** 0.9854 (98.54%)
* **Precision:** 1.0000 (100%)
* **Recall:** 0.9709 (97.09%)
* **ROC-AUC:** 0.9854 (98.54%)

---

## 6. Complete Production Training Pipeline with MLflow

Builds a reproducible scikit-learn `Pipeline` (with `SimpleImputer` and `StandardScaler` inside a `ColumnTransformer`), evaluates via Stratified split and 5-fold cross-validation, saves the pipeline artifact, and logs all metadata to **MLflow**.

```python
# ==========================
# COMPLETE TRAINING PIPELINE
# ==========================

import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

# --------------------------
# Feature / Target Split
# --------------------------
X = df.drop("target", axis=1)
y = df["target"]

# --------------------------
# Stratified Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------
# Preprocessing Pipeline
# --------------------------
numeric_features = X.columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features)
])

# --------------------------
# Random Forest Pipeline
# --------------------------
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# --------------------------
# Train Pipeline
# --------------------------
rf_pipeline.fit(X_train, y_train)
print("Model Training Complete")

# --------------------------
# 5-Fold Cross Validation
# --------------------------
cv_auc = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)
print("Cross Validation ROC-AUC:", cv_auc.mean())

# --------------------------
# Inference & Predictions
# --------------------------
pred = rf_pipeline.predict(X_test)
proba = rf_pipeline.predict_proba(X_test)[:, 1]

# --------------------------
# Metrics Calculation
# --------------------------
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
roc_auc = roc_auc_score(y_test, proba)

print("\n===== MODEL METRICS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("ROC-AUC  :", roc_auc)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, pred))

# --------------------------
# Save Model Artifact
# --------------------------
joblib.dump(rf_pipeline, "heart_model.pkl")
print("\nModel saved: heart_model.pkl")

# --------------------------
# MLflow Experiment Tracking
# --------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("HeartDiseasePrediction")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("cv_roc_auc", cv_auc.mean())

    mlflow.log_artifact("heart_model.pkl")

print("\nMLflow logging completed successfully")

# --------------------------
# Final Summary Table
# --------------------------
results_df = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "ROC_AUC": roc_auc,
    "CV_ROC_AUC": cv_auc.mean()
}])
display(results_df)
```

### Execution Output & Summary:
* **Cross-Validation ROC-AUC (5-fold):** `0.99977` (~100%)
* **Test Accuracy:** `1.0` (100%)
* **Test Precision:** `1.0` (100%)
* **Test Recall:** `1.0` (100%)
* **Test ROC-AUC:** `1.0` (100%)
* **Classification Report:**
  ```text
                precision    recall  f1-score   support
             0       1.00      1.00      1.00       100
             1       1.00      1.00      1.00       105

      accuracy                           1.00       205
     macro avg       1.00      1.00      1.00       205
  weighted avg       1.00      1.00      1.00       205
  ```

---

## 7. Model Serialization & Export

Export model and scaler objects using `pickle` for modular deployment.

```python
import pickle

with open("heart_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

---

## 8. FastAPI Model Serving API

Create a REST API with FastAPI exposing a `/predict` endpoint that handles array/list input format.

```python
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.post("/predict")
def predict(data: list):
    x = np.array(data).reshape(1, -1)
    x = scaler.transform(x)
    pred = model.predict(x)
    prob = model.predict_proba(x)

    return {
        "prediction": int(pred[0]),
        "confidence": float(max(prob[0]))
    }
```

---

## 9. Unit Testing

Unit test validating the prediction endpoint output structure.

```python
def test_prediction():
    from app.main import predict

    sample = [
        63, 1, 3, 145,
        233, 1, 0, 150,
        0, 2.3, 0, 0,
        1
    ]

    result = predict(sample)
    assert "prediction" in result
```

---

## 10. Automated API Code Generation

Generate the production-grade `main.py` application file accepting dictionary/JSON payload inputs.

```python
import os

fastapi_code = '''
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].max()

    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }
'''

os.makedirs("generated/app", exist_ok=True)

with open("generated/app/main.py", "w") as f:
    f.write(fastapi_code)
```
