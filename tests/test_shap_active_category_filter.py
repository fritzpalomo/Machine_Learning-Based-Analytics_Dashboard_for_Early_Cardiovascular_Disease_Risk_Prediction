"""
test_shap_active_category_filter.py
-------------------------------------
Regression tests for a display bug found during live testing: the SHAP
chart was showing BOTH one-hot columns of a categorical feature (e.g.
smoking_status_Never AND smoking_status_Current) even though a patient
can only actually be one or the other. This meant a patient who answered
"Never" could see "Smoking Status Current" listed as a contributing factor,
which is misleading since that column was inactive (0) for them.

_keep_only_active_categories() fixes this by dropping the one-hot column
for any category the patient did NOT select.
"""

import pandas as pd
import pytest
from shap_utils import _keep_only_active_categories


def make_patient(**overrides):
    base = {
        "age": 54, "sex": "Male", "systolic_bp": 132, "diastolic_bp": 84,
        "total_cholesterol": 210, "fasting_glucose": 110, "bmi": 27.30,
        "smoking_status": "Former", "alcohol_consumption": "Current",
        "physical_activity": "High",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_inactive_category_is_dropped():
    patient_df = make_patient(smoking_status="Never")
    raw_shap = {"smoking_status_Current": 0.01, "smoking_status_Never": 0.06}
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert "smoking_status_Current" not in filtered
    assert filtered["smoking_status_Never"] == 0.06


def test_active_category_flips_with_patient_answer():
    patient_df = make_patient(smoking_status="Current")
    raw_shap = {"smoking_status_Current": 0.12, "smoking_status_Never": 0.00}
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert "smoking_status_Never" not in filtered
    assert filtered["smoking_status_Current"] == 0.12


def test_filters_across_multiple_categorical_features_simultaneously():
    patient_df = make_patient(smoking_status="Current", alcohol_consumption="Former", physical_activity="Low")
    raw_shap = {
        "smoking_status_Never": 0.0, "smoking_status_Former": 0.0, "smoking_status_Current": 0.08,
        "alcohol_consumption_Never": 0.0, "alcohol_consumption_Former": 0.03, "alcohol_consumption_Current": 0.0,
        "physical_activity_High": 0.0, "physical_activity_Low": 0.02,
    }
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert set(filtered.keys()) == {"smoking_status_Current", "alcohol_consumption_Former", "physical_activity_Low"}


def test_numeric_features_are_never_filtered():
    patient_df = make_patient()
    raw_shap = {"age": 0.05, "bmi": -0.02, "systolic_bp": 0.10}
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert filtered == raw_shap
