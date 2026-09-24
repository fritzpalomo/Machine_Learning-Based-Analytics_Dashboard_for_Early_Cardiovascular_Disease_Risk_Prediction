"""
test_preprocessing.py
----------------------
Unit tests for preprocessing.py — the shared transformation pipeline
used by both training and the dashboard at inference time.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    build_preprocessor,
    get_feature_names,
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def make_sample_df(n=20, with_missing=False):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(25, 80, size=n),
        "sex": rng.choice(["Male", "Female"], size=n),
        "systolic_bp": rng.normal(130, 15, size=n),
        "diastolic_bp": rng.normal(82, 10, size=n),
        "total_cholesterol": rng.normal(200, 30, size=n),
        "fasting_glucose": rng.normal(100, 20, size=n),
        "bmi": rng.normal(26, 4, size=n),
        "smoking_status": rng.choice(["Never", "Former", "Current"], size=n),
        "alcohol_consumption": rng.choice(["None", "Occasional", "Regular"], size=n),
        "physical_activity": rng.choice(["Yes", "No"], size=n),
        "family_history_cvd": rng.choice(["Yes", "No"], size=n),
    })
    if with_missing:
        df.loc[0, "age"] = np.nan
        df.loc[1, "sex"] = None
    return df


def test_preprocessor_fits_and_transforms():
    df = make_sample_df()
    pre = build_preprocessor()
    transformed = pre.fit_transform(df[ALL_FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    # rows preserved, columns = numeric + one-hot encoded categorical
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_preprocessor_handles_missing_values():
    df = make_sample_df(with_missing=True)
    pre = build_preprocessor()
    transformed = pre.fit_transform(df[ALL_FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    assert not np.isnan(transformed).any(), "Imputation should remove all NaNs"


def test_feature_names_match_categories():
    df = make_sample_df()
    pre = build_preprocessor()
    pre.fit(df[ALL_FEATURES])
    names = get_feature_names(pre)
    # every numeric feature appears verbatim
    for feat in NUMERIC_FEATURES:
        assert feat in names
    # every categorical feature contributes at least one one-hot column
    for feat in CATEGORICAL_FEATURES:
        assert any(name.startswith(feat + "_") for name in names)


def test_unknown_category_does_not_crash_at_inference():
    df = make_sample_df()
    pre = build_preprocessor()
    pre.fit(df[ALL_FEATURES])
    # simulate a category never seen during training
    new_row = df.iloc[[0]].copy()
    new_row["smoking_status"] = "UnseenCategory"
    # should not raise, thanks to handle_unknown="ignore"
    pre.transform(new_row[ALL_FEATURES])
