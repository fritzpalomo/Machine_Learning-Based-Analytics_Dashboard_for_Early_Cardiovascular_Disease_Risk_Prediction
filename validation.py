"""
validation.py
--------------
Clinical sanity checks on patient input, run before an assessment is made.
Separated from app.py so the rules are unit-testable without Streamlit.

Returns two lists:
  errors   -- physiologically invalid; the assessment is blocked until fixed
  warnings -- unusual but possible, or worth a clinician's attention; the
              assessment proceeds and the user is notified
"""

DIABETIC_FASTING_GLUCOSE = 126  # mg/dL (7.0 mmol/L)


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

    age = patient.get("age")
    if age is not None:
        if age < 40:
            warnings.append(
                f"The patient is {age} years old. The WHO/ISH chart covers ages 40 to 79, so the level shown is "
                "for reference only. Blood pressure of 140/90 mmHg or higher under age 40 should be referred."
            )
        elif age >= 80:
            warnings.append(
                f"The patient is {age} years old. The WHO/ISH chart covers ages 40 to 79, so the level shown is for reference only."
            )

    glucose = patient.get("fasting_glucose")
    if glucose is not None and glucose >= DIABETIC_FASTING_GLUCOSE and not patient.get("known_diabetes"):
        warnings.append(
            f"Fasting glucose of {glucose} mg/dL is in the diabetic range (126 mg/dL or higher). The chart for people with "
            "diabetes is used. Confirm the diagnosis with a repeat test."
        )

    bmi = patient.get("bmi")
    if bmi is not None and not (12 <= bmi <= 60):
        warnings.append(f"BMI of {bmi} kg/m² is outside the usual range. Please re-check the height and weight.")

    return errors, warnings
