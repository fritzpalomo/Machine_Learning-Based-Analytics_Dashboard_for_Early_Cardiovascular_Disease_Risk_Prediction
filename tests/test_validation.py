"""
test_validation.py
--------------------
Unit tests for validation.py. test_systolic_equal_to_diastolic_is_rejected
directly covers the bug found during manual functional testing (TC-09):
the dashboard previously accepted systolic_bp == diastolic_bp without
any error.
"""

from validation import validate_patient_input


def test_normal_reading_has_no_errors_or_warnings():
    errors, warnings = validate_patient_input({"systolic_bp": 120, "diastolic_bp": 80})
    assert errors == []
    assert warnings == []


def test_systolic_equal_to_diastolic_is_rejected():
    # Regression test for the bug reported during TC-09 manual testing:
    # a 130/130 reading was previously accepted and produced a prediction.
    errors, warnings = validate_patient_input({"systolic_bp": 130, "diastolic_bp": 130})
    assert len(errors) == 1
    assert "must be higher than" in errors[0]


def test_systolic_lower_than_diastolic_is_rejected():
    errors, warnings = validate_patient_input({"systolic_bp": 110, "diastolic_bp": 130})
    assert len(errors) == 1


def test_narrow_but_valid_gap_produces_warning_not_error():
    errors, warnings = validate_patient_input({"systolic_bp": 100, "diastolic_bp": 95})
    assert errors == []
    assert len(warnings) == 1


def test_gap_of_exactly_ten_has_no_warning():
    # boundary case: 10 mmHg gap should be accepted cleanly
    errors, warnings = validate_patient_input({"systolic_bp": 100, "diastolic_bp": 90})
    assert errors == []
    assert warnings == []
