"""
tests/test_referral.py
-----------------------
Unit tests for the PhilPEN referral-criteria check (referral.py).
"""

from referral import check_referral_criteria


def _base_patient(**overrides):
    patient = {
        "age": 54,
        "sex": "Male",
        "systolic_bp": 132,
        "diastolic_bp": 84,
        "total_cholesterol": 210,
        "fasting_glucose": 110,
        "bmi": 27.3,
        "smoking_status": "Former",
        "alcohol_consumption": "Occasional",
        "physical_activity": "Yes",
        "family_history_cvd": "No",
    }
    patient.update(overrides)
    return patient


def test_no_flags_for_an_unremarkable_patient():
    reasons = check_referral_criteria(_base_patient())
    assert reasons == []


def test_flags_high_bp_under_40():
    reasons = check_referral_criteria(_base_patient(age=35, systolic_bp=146, diastolic_bp=92))
    assert len(reasons) == 1
    assert "under 40" in reasons[0]


def test_does_not_flag_high_bp_at_or_over_40():
    reasons = check_referral_criteria(_base_patient(age=40, systolic_bp=146, diastolic_bp=92))
    assert reasons == []


def test_does_not_flag_normal_bp_under_40():
    reasons = check_referral_criteria(_base_patient(age=28, systolic_bp=118, diastolic_bp=76))
    assert reasons == []


def test_flags_severely_elevated_fasting_glucose():
    # >14 mmol/L ~= >252 mg/dL
    reasons = check_referral_criteria(_base_patient(fasting_glucose=260))
    assert len(reasons) == 1
    assert "glucose" in reasons[0].lower()


def test_does_not_flag_moderately_elevated_fasting_glucose():
    reasons = check_referral_criteria(_base_patient(fasting_glucose=180))
    assert reasons == []


def test_can_flag_both_criteria_at_once():
    reasons = check_referral_criteria(
        _base_patient(age=30, systolic_bp=150, diastolic_bp=95, fasting_glucose=270)
    )
    assert len(reasons) == 2


def test_missing_fields_do_not_raise():
    # Defensive: a partially-filled dict should not crash, just not flag.
    reasons = check_referral_criteria({"age": None, "systolic_bp": None, "diastolic_bp": None, "fasting_glucose": None})
    assert reasons == []
