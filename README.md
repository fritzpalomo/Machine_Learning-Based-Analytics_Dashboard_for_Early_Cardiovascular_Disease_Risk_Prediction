# Explainable ML Analytics Dashboard — WHO/PhilPEN CVD Risk Classification

Working scaffold for the dashboard in your wireframe: patient input → ML
prediction → local SHAP explanation → educational guidance, backed by
SQLite for record-keeping.

## Project structure

```
cvd_dashboard/
├── app.py                      # Streamlit dashboard (the 4-module UI)
├── train_model.py              # Trains 5 candidate models, saves the best one
├── generate_synthetic_data.py  # Stand-in dataset generator (see below)
├── preprocessing.py            # Shared preprocessing (training + inference)
├── shap_utils.py                # SHAP computation + chart builder
├── db.py                       # SQLite data access layer
├── requirements.txt
├── data/                       # patients.csv goes here
└── models/                     # trained model artifacts land here
```

## Setup

```bash
cd cvd_dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get your data in

`generate_synthetic_data.py` creates a **fake but realistically-shaped**
dataset (`data/patients.csv`) so you can run the whole pipeline today:

```bash
python generate_synthetic_data.py
```

**Replace this with your real, anonymized DOH/PhilPEN dataset** — just
match the column names in `preprocessing.py` (`ALL_FEATURES` /
`TARGET`), or edit that file to match your real column names instead.

## 2. Train the model

```bash
python train_model.py
```

This trains Logistic Regression, Decision Tree, Random Forest, SVM, and
XGBoost with `GridSearchCV`, picks the best by macro-F1, fits a SHAP
explainer on it, and saves everything to `models/`.

## 3. Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Notes

- **No PII is stored.** `db.py` only stores whatever a clinician enters
  at prediction time (already de-identified for training purposes) —
  review this against your institution's data governance policy before
  any real deployment.
- **Swap the SHAP explainer type** if you change models: `train_model.py`
  auto-selects `TreeExplainer` for tree-based models (RF/XGBoost/Decision
  Tree) and falls back to `KernelExplainer` for Logistic Regression/SVM.
  KernelExplainer is much slower — expect a delay on first prediction.
  Consider caching, or swapping SVM/LogReg out during actual PhilPEN
  cardiovascular fieldwork if latency becomes an issue.
  For the current wireframe (which flags "Random Forest / XGBoost / Best
  Performing Model"), tree-based models are the expected path so this
  should rarely bite you.
- **Synthetic data caveat**: the fake dataset's risk labels are derived
  from a made-up formula purely so training/SHAP have something
  sensible to work with. Model accuracy on synthetic data is not
  meaningful — retrain on your real dataset before drawing any
  conclusions.
- This was built and syntax/logic-tested using Python 3, scikit-learn,
  and pandas in this environment. `streamlit`, `xgboost`, `shap`, and
  `plotly` should be installed in your own environment via
  `requirements.txt` — they weren't available to test against here, so
  double-check versions if you hit import errors.
