"""
test_db.py
----------
Unit tests for db.py — the local SQLite storage layer.
Uses a temporary database file per test so tests don't interfere
with a real cvd_dashboard.db or each other.
"""

import os
import importlib

import pytest


@pytest.fixture
def db_module(tmp_path, monkeypatch):
    """Reload db.py pointed at a throwaway temp database file."""
    import db as db_mod
    test_db_path = str(tmp_path / "test_cvd.db")
    monkeypatch.setattr(db_mod, "DB_PATH", test_db_path)
    db_mod.init_db()
    yield db_mod
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


SAMPLE_RECORD = {
    "age": 54, "sex": "Male", "systolic_bp": 132, "diastolic_bp": 84,
    "total_cholesterol": 210, "fasting_glucose": 110, "bmi": 27.3,
    "smoking_status": "Former", "alcohol_consumption": "Occasional",
    "physical_activity": "Yes", "family_history_cvd": "Yes",
    "predicted_risk_category": "Moderate Risk", "confidence_score": 72.0,
}


def test_init_db_creates_tables(db_module):
    with db_module.get_connection() as conn:
        tables = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert {"users", "predictions", "system_config"}.issubset(tables)


def test_insert_and_retrieve_prediction(db_module):
    db_module.insert_prediction(SAMPLE_RECORD)
    recent = db_module.get_recent_predictions(5)
    assert len(recent) == 1
    assert recent[0]["predicted_risk_category"] == "Moderate Risk"
    assert recent[0]["confidence_score"] == 72.0


def test_recent_predictions_ordered_newest_first(db_module):
    for i in range(3):
        record = dict(SAMPLE_RECORD)
        record["age"] = 40 + i
        db_module.insert_prediction(record)
    recent = db_module.get_recent_predictions(10)
    ages = [r["age"] for r in recent]
    assert ages == sorted(ages, reverse=True), "Expected newest insert (highest age here) first"


def test_recent_predictions_respects_limit(db_module):
    for i in range(5):
        db_module.insert_prediction(SAMPLE_RECORD)
    recent = db_module.get_recent_predictions(2)
    assert len(recent) == 2


def test_no_pii_columns_exist(db_module):
    """Guard against accidentally storing identifying fields not in the schema."""
    with db_module.get_connection() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    forbidden = {"full_name", "national_id", "philhealth_no", "address", "contact_number"}
    assert forbidden.isdisjoint(cols)


def test_numpy_float32_confidence_is_stored_as_a_real_number(db_module):
    # Regression test for DF-05: XGBoost's predict_proba() returns
    # numpy.float32 values. sqlite3 treats numpy.float64 as a plain float
    # (it subclasses float) but does NOT recognize numpy.float32, and
    # silently stores it as a raw BLOB instead of raising an error. A
    # patient's confidence score was corrupted into unreadable bytes
    # (e.g. b'\n\xd7^B') the moment the deployed model returned float32.
    np = pytest.importorskip("numpy")
    record = dict(SAMPLE_RECORD)
    record["confidence_score"] = np.float32(49.7)
    db_module.insert_prediction(record)
    recent = db_module.get_recent_predictions(1)
    stored_value = recent[0]["confidence_score"]
    assert isinstance(stored_value, float), (
        f"Expected a plain float, got {type(stored_value)} ({stored_value!r}) — "
        "confidence_score was likely stored as a corrupt BLOB."
    )
    assert stored_value == pytest.approx(49.7, abs=0.01)


def test_numpy_scalar_numeric_fields_are_stored_as_real_numbers(db_module):
    # Broader regression check: any numpy scalar type (not just float32
    # confidence scores) passed into a numeric column should be coerced
    # to a native Python type, not silently stored as a BLOB.
    np = pytest.importorskip("numpy")
    record = dict(SAMPLE_RECORD)
    record["bmi"] = np.float32(27.3)
    record["age"] = np.int64(54)
    db_module.insert_prediction(record)
    recent = db_module.get_recent_predictions(1)
    assert isinstance(recent[0]["bmi"], float)
    assert isinstance(recent[0]["age"], int)
