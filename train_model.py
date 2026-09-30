"""
train_model.py
---------------
Trains candidate models (Logistic Regression, Decision Tree, Random
Forest, SVM, XGBoost), tunes each with GridSearchCV, selects the best
performer, fits a SHAP explainer on it, and serializes everything the
Streamlit app needs to data/patients.csv -> models/*.joblib

Methodology (matches the paper's Hyperparameter Optimization,
Cross-Validation, and Model Evaluation sections):
  - Hyperparameter optimization: GridSearchCV over the parameter grids
    below, one per candidate algorithm.
  - Cross-validation: 10-fold CV on the TRAINING set only, used both
    to select each model's best hyperparameters and to report a
    cross-validated performance estimate.
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
from sklearn.model_selection import GridSearchCV, train_test_split
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

CV_FOLDS = 10
SCORING = "f1_weighted"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return X, y


def get_candidate_models():
    """Each entry: (name, estimator, param_grid for GridSearchCV).
    Parameter grids match the specific parameters named in the paper's
    Hyperparameter Optimization section for each algorithm. Grid sizes
    are kept modest (2-3 values per parameter) so a full run completes
    in reasonable time on a laptop with the current dataset size; widen
    them once training on the real, larger DOH/PhilPEN dataset if desired."""
    return {
        "Logistic Regression": (
            # solver='saga' supports both l1 and l2 penalties for multinomial classification
            LogisticRegression(max_iter=2000, solver="saga", random_state=42),
            {
                "clf__penalty": ["l1", "l2"],           # regularization type
                "clf__C": [0.1, 1.0, 10.0],             # regularization strength
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "clf__max_depth": [4, 8, None],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [6, None],
                "clf__min_samples_split": [2, 5],
            },
        ),
        "Support Vector Machine (SVM)": (
            SVC(probability=True, random_state=42),
            {
                "clf__kernel": ["rbf", "linear"],
                "clf__C": [1.0, 10.0],
                "clf__gamma": ["scale", "auto"],
            },
        ),
        "Extreme Gradient Boosting (XGBoost)": (
            XGBClassifier(eval_metric="mlogloss", random_state=42),
            {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 6],
                "clf__n_estimators": [200, 400],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        ),
    }


def main():
    print("Loading data...")
    X, y = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Low/Moderate/High -> 0/1/2
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

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
        search = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1)
        search.fit(X_train, y_train)
        cv_score = search.best_score_  # mean weighted-F1 across the 10 folds, best combo

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

        # Best model selected by weighted F1-score on the independent test set
        if test_f1 > best_score:
            best_score = test_f1
            best_name = name
            best_pipeline = search.best_estimator_

    print(f"\n=== Best model: {best_name} (test weighted-F1={best_score:.4f}) ===")

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
    print(f"Dataset: {len(X)} patients  |  Train/test split: {len(X_train)}/{len(X_test)} (stratified)")
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
    best_proba = proba_by_model[best_name]

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

    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], best_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{cname} (AUC = {roc_auc_i:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve (One-vs-Rest)\n{best_name}", fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig_roc.savefig("models/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(7, 5.5))
    for i, cname in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i], best_proba[:, i])
        ax_pr.plot(rec, prec, label=f"{cname}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall Curve (One-vs-Rest)\n{best_name}", fontsize=12)
    ax_pr.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    fig_pr.savefig("models/pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pr)

    print("Saved diagnostic plots for the selected model:")
    print("  models/confusion_matrix.png")
    print("  models/roc_curve.png")
    print("  models/pr_curve.png\n")

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
