"""
test_shap_active_category_filter.py
-------------------------------------
Regression tests for a display bug found during live testing: the SHAP
chart was showing BOTH one-hot columns of a categorical feature (e.g.
family_history_cvd_Yes AND family_history_cvd_No) even though a patient
can only actually be one or the other. This meant a patient who answered
"No" could see "Family History Cvd Yes" listed as a contributing factor,
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
        "smoking_status": "Former", "alcohol_consumption": "Occasional",
        "physical_activity": "Yes", "family_history_cvd": "No",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_inactive_category_is_dropped():
    # Regression test for the exact bug found in live testing: patient
    # answered "No" to family history, but the raw SHAP output also
    # included a value for the "Yes" column (which was never selected).
    patient_df = make_patient(family_history_cvd="No")
    raw_shap = {
        "family_history_cvd_Yes": 0.01,
        "family_history_cvd_No": 0.06,
    }
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert "family_history_cvd_Yes" not in filtered
    assert filtered["family_history_cvd_No"] == 0.06


def test_active_category_flips_with_patient_answer():
    # Same feature, opposite answer — the kept column should flip too.
    patient_df = make_patient(family_history_cvd="Yes")
    raw_shap = {
        "family_history_cvd_Yes": 0.12,
        "family_history_cvd_No": 0.00,
    }
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert "family_history_cvd_No" not in filtered
    assert filtered["family_history_cvd_Yes"] == 0.12


def test_filters_across_multiple_categorical_features_simultaneously():
    patient_df = make_patient(
        smoking_status="Current", alcohol_consumption="Regular",
        physical_activity="No", family_history_cvd="Yes",
    )
    raw_shap = {
        "smoking_status_Never": 0.0, "smoking_status_Former": 0.0, "smoking_status_Current": 0.08,
        "alcohol_consumption_None": 0.0, "alcohol_consumption_Occasional": 0.0, "alcohol_consumption_Regular": 0.03,
        "physical_activity_Yes": 0.0, "physical_activity_No": 0.02,
        "family_history_cvd_Yes": 0.06, "family_history_cvd_No": 0.0,
    }
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert set(filtered.keys()) == {
        "smoking_status_Current", "alcohol_consumption_Regular",
        "physical_activity_No", "family_history_cvd_Yes",
    }


def test_numeric_features_are_never_filtered():
    patient_df = make_patient()
    raw_shap = {"age": 0.05, "bmi": -0.02, "systolic_bp": 0.10}
    filtered = _keep_only_active_categories(raw_shap, patient_df)
    assert filtered == raw_shap
