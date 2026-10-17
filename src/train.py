import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.preprocess import build_preprocessor, load_and_split_data


def train_and_track():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "heart.csv")
    models_dir = os.path.join(base_dir, "models")
    mlruns_dir = os.path.join(base_dir, "mlruns")
    mlflow_db = os.path.join(base_dir, "mlflow.db")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(mlruns_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    X, y = load_and_split_data(data_path)

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Configure SQLite MLflow Tracking backend
    tracking_uri = f"sqlite:///{mlflow_db}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "HeartDiseasePrediction"
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=mlruns_dir
        )
    mlflow.set_experiment(experiment_name)

    preprocessor = build_preprocessor()

    # 1. Baseline Model: Logistic Regression
    with mlflow.start_run(run_name="Baseline_LogisticRegression"):
        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        lr_pipeline.fit(X_train, y_train)

        lr_preds = lr_pipeline.predict(X_test)
        lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]

        lr_acc = accuracy_score(y_test, lr_preds)
        lr_prec = precision_score(y_test, lr_preds)
        lr_rec = recall_score(y_test, lr_preds)
        lr_f1 = f1_score(y_test, lr_preds)
        lr_roc = roc_auc_score(y_test, lr_proba)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", lr_acc)
        mlflow.log_metric("precision", lr_prec)
        mlflow.log_metric("recall", lr_rec)
        mlflow.log_metric("f1_score", lr_f1)
        mlflow.log_metric("roc_auc", lr_roc)
        print(f"Logistic Regression - Accuracy: {lr_acc:.4f}, ROC-AUC: {lr_roc:.4f}")

    # 2. Production Model: Random Forest
    with mlflow.start_run(run_name="Production_RandomForest"):
        n_estimators = 200
        max_depth = 8
        random_state = 42

        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            ))
        ])

        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc")
        cv_roc_auc_mean = float(np.mean(cv_scores))

        rf_pipeline.fit(X_train, y_train)

        rf_preds = rf_pipeline.predict(X_test)
        rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

        rf_acc = accuracy_score(y_test, rf_preds)
        rf_prec = precision_score(y_test, rf_preds)
        rf_rec = recall_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds)
        rf_roc = roc_auc_score(y_test, rf_proba)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        mlflow.log_metric("accuracy", rf_acc)
        mlflow.log_metric("precision", rf_prec)
        mlflow.log_metric("recall", rf_rec)
        mlflow.log_metric("f1_score", rf_f1)
        mlflow.log_metric("roc_auc", rf_roc)
        mlflow.log_metric("cv_roc_auc_mean", cv_roc_auc_mean)

        # Save model artifact
        model_out_path = os.path.join(models_dir, "heart_model.pkl")
        joblib.dump(rf_pipeline, model_out_path)
        mlflow.log_artifact(model_out_path)

        # Log scikit-learn model flavor in MLflow with cloudpickle
        mlflow.sklearn.log_model(
            rf_pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        print("\n================ PRODUCTION MODEL RESULTS ================")
        print(f"Accuracy         : {rf_acc:.4f}")
        print(f"Precision        : {rf_prec:.4f}")
        print(f"Recall           : {rf_rec:.4f}")
        print(f"F1-Score         : {rf_f1:.4f}")
        print(f"ROC-AUC          : {rf_roc:.4f}")
        print(f"5-Fold CV ROC-AUC: {cv_roc_auc_mean:.4f}")
        print("==========================================================")
        print("\nClassification Report:\n", classification_report(y_test, rf_preds))
        print(f"Artifact saved: {model_out_path}")
        print("MLflow logging completed successfully.")


if __name__ == "__main__":
    train_and_track()
