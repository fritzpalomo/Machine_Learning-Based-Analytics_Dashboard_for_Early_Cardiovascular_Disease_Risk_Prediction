"""
validation.py
--------------
Clinical sanity checks on patient input, run before a prediction is
made. Separated from app.py so the rules are unit-testable without
spinning up Streamlit.

Returns two lists:
  errors   — physiologically invalid; prediction is blocked until fixed
  warnings — unusual but possible; prediction proceeds, user is notified
"""


def validate_patient_input(patient: dict) -> tuple[list, list]:
    errors = []
    warnings = []

    systolic = patient.get("systolic_bp")
    diastolic = patient.get("diastolic_bp")

    if systolic is not None and diastolic is not None:
        if systolic <= diastolic:
            errors.append(
                f"Systolic blood pressure ({systolic} mmHg) must be higher than "
                f"diastolic blood pressure ({diastolic} mmHg). Please re-check these values."
            )
        elif (systolic - diastolic) < 10:
            warnings.append(
                f"The gap between systolic and diastolic blood pressure "
                f"({systolic - diastolic} mmHg) is unusually narrow — please double-check these readings."
            )

    return errors, warnings
