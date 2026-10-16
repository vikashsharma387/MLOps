import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "target"


def build_preprocessor() -> ColumnTransformer:
    """
    Build a reusable scikit-learn preprocessing ColumnTransformer.
    Applies median imputation and standard scaling to all numeric medical features.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS)
        ],
        remainder="drop"
    )

    return preprocessor


def load_and_split_data(data_path: str):
    """Load dataset from CSV and separate features from target."""
    df = pd.read_csv(data_path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
