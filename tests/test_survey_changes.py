import bmi
import db
import explain
import shap_utils


def test_bmi_from_height_weight_and_category():
    assert bmi.compute_bmi(170, 70) == 24.2 and bmi.compute_bmi(0, 70) is None
    assert bmi.bmi_category(22.9) == "Normal" and bmi.bmi_category(23.0) == "Overweight (at risk)" and bmi.bmi_category(25) == "Obese"


def test_why_this_prediction_names_up_and_down_factors():
    shap = {"bmi": 0.3, "systolic_bp": 0.2, "age": -0.1, "smoking_status_Current": 0.05}
    text = explain.why_this_prediction(shap, "Level 4 (30 to < 40%)")
    assert "Level 4 (30 to < 40%)" in text and "higher risk" in text and "lowered" in text


def test_why_this_prediction_with_no_positive_factors():
    assert "higher risk" not in explain.why_this_prediction({"age": -0.2}, "Level 1 (< 10%)")


def test_glossary_covers_the_confusing_terms():
    assert {t for t, _ in explain.GLOSSARY} >= {"Risk category", "Composite Risk Score", "Model Confidence Score", "SHAP chart"}


def test_diet_card_tailored_and_cap_respected():
    p = {"systolic_bp": 150, "diastolic_bp": 95, "fasting_glucose": 130, "total_cholesterol": 240, "bmi": 28}
    cards = shap_utils.get_dynamic_recommendations({"bmi": 0.3, "age": 0.1}, top_n=4, patient=p)
    assert len(cards) <= 4
    diet = next(c for c in cards if c["title"] == "Healthy Diet")["desc"].lower()
    assert "salt" in diet and "sweet" in diet
    assert any(c["title"] == "Glucose in Diabetic Range" for c in cards)


def test_no_patient_leaves_recommendations_unchanged():
    a = shap_utils.get_dynamic_recommendations({"bmi": 0.3}, top_n=4)
    b = shap_utils.get_dynamic_recommendations({"bmi": 0.3}, top_n=4, patient=None)
    assert a == b


def test_medication_review_card():
    p = {"systolic_bp": 150, "diastolic_bp": 95, "on_bp_medication": True}
    cards = shap_utils.get_dynamic_recommendations({"age": 0.1}, top_n=4, patient=p)
    assert any(c["title"] == "Review Medication" for c in cards)


def test_case_code_history_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    base = {"age": 50, "sex": "Male", "predicted_risk_category": "Level 1 (< 10%)", "confidence_score": 80.0}
    db.insert_prediction({**base, "case_code": "RHU-1"})
    db.insert_prediction({**base, "case_code": "rhu-1"})
    db.insert_prediction({**base, "case_code": "RHU-2"})
    db.insert_prediction(base)
    assert len(db.get_case_history(" RHU-1 ")) == 2
    assert db.get_case_history("") == [] and db.get_case_history("x") == []


def test_old_database_is_migrated(tmp_path, monkeypatch):
    import sqlite3
    path = str(tmp_path / "old.db")
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, age INTEGER, sex TEXT, systolic_bp REAL, diastolic_bp REAL, total_cholesterol REAL, fasting_glucose REAL, bmi REAL, smoking_status TEXT, alcohol_consumption TEXT, physical_activity TEXT, family_history_cvd TEXT, predicted_risk_category TEXT, confidence_score REAL)")
    c.commit(); c.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    db.insert_prediction({"age": 40, "case_code": "A1", "predicted_risk_category": "Level 1 (< 10%)"})
    assert len(db.get_case_history("a1")) == 1
