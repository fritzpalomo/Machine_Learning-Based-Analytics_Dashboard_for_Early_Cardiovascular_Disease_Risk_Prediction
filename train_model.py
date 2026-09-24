"""
train_model.py
---------------
Trains candidate models (Logistic Regression, Decision Tree, Random
Forest, SVM, XGBoost), selects the best performer on held-out data,
fits a SHAP explainer on it, and serializes everything the Streamlit
app needs to data/patients.csv -> models/*.joblib

Run: python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

from preprocessing import build_preprocessor, get_feature_names, ALL_FEATURES, TARGET

DATA_PATH = "data/patients.csv"
MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"
EXPLAINER_PATH = "models/shap_explainer.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return X, y


def get_candidate_models():
    """Each entry: (name, estimator, param_grid for GridSearchCV)."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"clf__max_depth": [4, 8, None]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [6, None]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"clf__C": [1.0, 10.0], "clf__kernel": ["rbf"]},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="mlogloss", random_state=42
            ),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6]},
        ),
    }


def main():
    print("Loading data...")
    X, y = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Low/Moderate/High -> 0/1/2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    preprocessor = build_preprocessor()

    best_score = -np.inf
    best_name = None
    best_pipeline = None

    for name, (estimator, param_grid) in get_candidate_models().items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])
        search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)

        preds = search.predict(X_test)
        score = f1_score(y_test, preds, average="macro")
        print(f"{name} macro-F1 on test set: {score:.4f}")
        print(classification_report(y_test, preds, target_names=label_encoder.classes_))

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = search.best_estimator_

    print(f"\n=== Best model: {best_name} (macro-F1={best_score:.4f}) ===")

    # Refit preprocessor standalone so we can transform data for SHAP
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_clf = best_pipeline.named_steps["clf"]

    X_train_transformed = fitted_preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    feature_names = get_feature_names(fitted_preprocessor)

    # Build a SHAP explainer appropriate to the model type
    tree_based = best_name in ("DecisionTree", "RandomForest", "XGBoost")
    if tree_based:
        explainer = shap.TreeExplainer(fitted_clf)
    else:
        # KernelExplainer works for any model but is slower; sample background data
        background = shap.sample(X_train_transformed, 100, random_state=42)
        explainer = shap.KernelExplainer(fitted_clf.predict_proba, background)

    # Persist everything the dashboard needs
    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print(f"\nSaved model pipeline -> {MODEL_PATH}")
    print(f"Saved preprocessor -> {PREPROCESSOR_PATH}")
    print(f"Saved label encoder -> {LABEL_ENCODER_PATH}")
    print(f"Saved SHAP explainer -> {EXPLAINER_PATH}")
    print(f"Saved feature names -> {FEATURE_NAMES_PATH}")
    print(f"\nBest model was: {best_name}")


if __name__ == "__main__":
    main()
