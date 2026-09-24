"""
preprocessing.py
-----------------
Single source of truth for feature preprocessing so that training and
inference (inside the Streamlit app) apply IDENTICAL transformations.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "fasting_glucose",
    "bmi",
]

CATEGORICAL_FEATURES = [
    "sex",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity",
    "family_history_cvd",
]

TARGET = "risk_category"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfit ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Returns human-readable feature names after transformation (for SHAP labels)."""
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names
