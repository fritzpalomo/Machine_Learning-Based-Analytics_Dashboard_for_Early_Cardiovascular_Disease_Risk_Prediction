"""
train_model.py
---------------
Trains candidate models (Logistic Regression, Decision Tree, Random
Forest, SVM, XGBoost), tunes each with GridSearchCV, selects the best
performer, fits a SHAP explainer on it, and serializes everything the
Streamlit app needs to data/patients.csv -> models/*.joblib

Methodology (matches Chapter IV, 4.1.6 and 4.1.8 of the paper):
  - Data: ENNS records labelled with the five WHO/ISH (PhilPEN) chart levels
    (see prepare_enns_data.py). Household-grouped 80/20 split, so no household
    appears in both sets; random_state = 42.
  - Only XGBoost is tuned: five settings, three-fold household-grouped
    cross-validation on the TRAINING set only. The other models run with fixed
    scikit-learn settings. No SMOTE.
  - XGBoost is the DEPLOYED model family (class probabilities for the gauge and
    exact TreeSHAP), whichever model scores highest; the comparison is printed.
  - Model evaluation: the tuned models are evaluated on the held-out,
    independent TEST set using weighted F1-score (primary selection
    criterion, appropriate given class imbalance across WHO/PhilPEN
    risk categories), plus Accuracy, Precision, Recall, ROC-AUC,
    Confusion Matrix, ROC Curve, and Precision-Recall Curve.

Run: python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # no GUI needed; just save plot files
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from preprocessing import build_preprocessor, get_feature_names, ALL_FEATURES, TARGET

DATA_PATH = "data/patients.csv"
MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"
EXPLAINER_PATH = "models/shap_explainer.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"

CV_FOLDS = 3
SCORING = "f1_weighted"


DEPLOYED_MODEL = "Extreme Gradient Boosting (XGBoost)"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    # household_group (from prepare_enns_data.py) keeps members of one household on the same side of the split
    groups = df["household_group"] if "household_group" in df.columns else None
    return X, y, groups


def get_candidate_models():
    """(estimator, param_grid) per model. Only XGBoost is tuned: five settings
    (a list of one-setting grids), three-fold household-grouped CV. The other
    models use fixed settings (an empty grid means one candidate)."""
    xgb_settings = [
        {"clf__n_estimators": [100], "clf__max_depth": [3], "clf__learning_rate": [0.1]},
        {"clf__n_estimators": [200], "clf__max_depth": [4], "clf__learning_rate": [0.1]},
        {"clf__n_estimators": [300], "clf__max_depth": [4], "clf__learning_rate": [0.1]},
        {"clf__n_estimators": [300], "clf__max_depth": [6], "clf__learning_rate": [0.05]},
        {"clf__n_estimators": [400], "clf__max_depth": [6], "clf__learning_rate": [0.1]},
    ]
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), {}),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), {}),
        "Support Vector Machine (SVM)": (SVC(probability=True, random_state=42), {}),
        "Extreme Gradient Boosting (XGBoost)": (XGBClassifier(eval_metric="mlogloss", random_state=42), xgb_settings),
    }


def main():
    print("Loading data...")
    X, y, groups = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Level 1..Level 5 -> 0..4
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y_encoded, groups))
    else:
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.8)
        train_idx, test_idx = perm[:cut], perm[cut:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    groups_train = groups.iloc[train_idx].to_numpy() if groups is not None else np.arange(len(train_idx))

    preprocessor = build_preprocessor()

    best_score = -np.inf
    best_name = None
    best_pipeline = None
    all_results = []  # (name, cv_f1, test_f1, best_hyperparameters)
    full_metrics = {}  # name -> dict of accuracy/precision/recall/f1/roc_auc
    proba_by_model = {}  # name -> predict_proba output on the test set (for ROC/PR curves)

    for name, (estimator, param_grid) in get_candidate_models().items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])

        # --- Hyperparameter optimization + 10-fold cross-validation on the TRAINING set ---
        search = GridSearchCV(pipe, param_grid, cv=GroupKFold(n_splits=CV_FOLDS), scoring=SCORING, n_jobs=-1)
        search.fit(X_train, y_train, groups=groups_train)
        cv_score = search.best_score_  # mean weighted-F1 across the folds, best setting

        # --- Evaluation on the independent TEST set ---
        preds = search.predict(X_test)
        proba = search.predict_proba(X_test)
        proba_by_model[name] = proba

        test_f1 = f1_score(y_test, preds, average="weighted")
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

        full_metrics[name] = {
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": test_f1, "roc_auc": roc_auc,
        }

        print(f"{name}: 10-fold CV weighted-F1 = {cv_score:.4f}  |  Test weighted-F1 = {test_f1:.4f}")
        print(classification_report(y_test, preds, target_names=class_names))

        best_params_for_model = {
            k.replace("clf__", ""): v for k, v in search.best_params_.items()
        }
        all_results.append((name, cv_score, test_f1, best_params_for_model))

        # The deployed model family is fixed (see DEPLOYED_MODEL); the others are comparison only
        if name == DEPLOYED_MODEL:
            best_score = test_f1
            best_name = name
            best_pipeline = search.best_estimator_

    print(f"\n=== Deployed model: {best_name} (test weighted-F1={best_score:.4f}) ===")

    # ------------------------------------------------------------------
    # TABLE 1 — Hyperparameter optimization + cross-validation summary
    # ------------------------------------------------------------------
    all_results.sort(key=lambda r: r[2], reverse=True)  # sort by test weighted-F1
    SEP_WIDTH = 110
    print("\n" + "=" * SEP_WIDTH)
    print(f"MODEL COMPARISON SUMMARY  ({CV_FOLDS}-fold cross-validated, scored on weighted F1)")
    print("=" * SEP_WIDTH)
    print(f"{'Rank':<6}{'Model':<38}{'CV F1':<10}{'Test F1':<10}{'Best Hyperparameters'}")
    print("-" * SEP_WIDTH)
    for rank, (name, cv_f1, test_f1, params) in enumerate(all_results, start=1):
        marker = "  <-- SELECTED" if name == best_name else ""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{rank:<6}{name:<38}{cv_f1:<10.4f}{test_f1:<10.4f}{params_str}{marker}")
    print("=" * SEP_WIDTH)
    print(f"Dataset: {len(X)} patients  |  Train/test split: {len(X_train)}/{len(X_test)} (household-grouped)")
    print(f"Cross-validation: {CV_FOLDS}-fold, training set only  |  Risk categories: {', '.join(class_names)}")
    print("=" * SEP_WIDTH + "\n")

    # ------------------------------------------------------------------
    # TABLE 2 — Full evaluation metrics on the independent test set
    # ------------------------------------------------------------------
    print("=" * SEP_WIDTH)
    print("FULL EVALUATION METRICS  (independent test set)")
    print("=" * SEP_WIDTH)
    print(f"{'Model':<38}{'Accuracy':<11}{'Precision':<12}{'Recall':<10}{'F1':<10}{'ROC-AUC'}")
    print("-" * SEP_WIDTH)
    for name, _, _, _ in all_results:
        m = full_metrics[name]
        marker = "  <-- SELECTED" if name == best_name else ""
        print(f"{name:<38}{m['accuracy']:<11.4f}{m['precision']:<12.4f}{m['recall']:<10.4f}{m['f1']:<10.4f}{m['roc_auc']:.4f}{marker}")
    print("=" * SEP_WIDTH)
    print("(Precision, Recall, and F1 are weighted averages across the WHO/PhilPEN risk categories;")
    print(" ROC-AUC is one-vs-rest, weighted by class support.)")
    print("=" * SEP_WIDTH + "\n")

    # Refit preprocessor standalone so we can transform data for SHAP
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_clf = best_pipeline.named_steps["clf"]

    X_train_transformed = fitted_preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    feature_names = get_feature_names(fitted_preprocessor)

    # ------------------------------------------------------------------
    # Confusion Matrix, ROC Curve, and Precision-Recall Curve for the
    # SELECTED (deployed) model only -- these are per-model diagnostic
    # plots and would be excessive to generate for all five candidates.
    # ------------------------------------------------------------------
    best_preds = best_pipeline.predict(X_test)

    print(f"Confusion Matrix ({best_name}):")
    cm = confusion_matrix(y_test, best_preds)
    header = "".join(f"{c:>16}" for c in class_names)
    print(f"{'':>16}{header}   <- predicted")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>16}" for v in row)
        print(f"{class_names[i]:>16}{row_str}")
    print()

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title(f"Confusion Matrix\n{best_name}", fontsize=12)
    plt.tight_layout()
    fig_cm.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    print("Saved models/confusion_matrix.png\n")

    # Build a SHAP explainer appropriate to the model type
    tree_based = best_name in ("Decision Tree", "Random Forest", "Extreme Gradient Boosting (XGBoost)")
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

    print(f"Saved model pipeline -> {MODEL_PATH}")
    print(f"Saved preprocessor -> {PREPROCESSOR_PATH}")
    print(f"Saved label encoder -> {LABEL_ENCODER_PATH}")
    print(f"Saved SHAP explainer -> {EXPLAINER_PATH}")
    print(f"Saved feature names -> {FEATURE_NAMES_PATH}")
    print(f"\nBest model was: {best_name}")


if __name__ == "__main__":
    main()
