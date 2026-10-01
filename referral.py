"""
referral.py
-----------
Checks a patient's inputs against a conservative subset of the
PhilPEN primary-care referral criteria: findings that mean the patient
should be referred to a higher-level facility, independent of what the
model's own WHO/PhilPEN risk-category prediction says.

LIMITATION (important, read before extending this module): this
dashboard currently collects only age, blood pressure, total
cholesterol, fasting glucose, BMI, and lifestyle fields. Several of
the official PhilPEN referral criteria need data this app does not
collect -- a personal history of heart disease, stroke, TIA, diabetes,
or kidney disease; proteinuria; urine ketones; current medications and
treatment response; or foot ulcers/infection. Those criteria are
intentionally NOT implemented here, because guessing at them from
fields that don't actually capture them (e.g. inferring "known heart
disease" from family history, which records a relative's history, not
the patient's own) would be clinically misleading. An empty result
from check_referral_criteria() means "no flag was raised from the
checkable criteria," not "this patient does not need referral" -- the
remaining criteria must still be assessed clinically.
"""

MMOL_TO_MGDL = 18.0182  # conversion factor for glucose, 1 mmol/L ~= 18.0182 mg/dL
SEVERE_FASTING_GLUCOSE_THRESHOLD_MGDL = 14 * MMOL_TO_MGDL  # ~252 mg/dL


def check_referral_criteria(patient: dict) -> list:
    """
    Returns a list of human-readable reasons this patient should be
    referred to a higher-level facility, based on the subset of
    PhilPEN referral criteria this app can evaluate from its existing
    input fields. See the module docstring for which criteria are
    deliberately NOT covered and why.
    """
    reasons = []

    age = patient.get("age")
    systolic = patient.get("systolic_bp")
    diastolic = patient.get("diastolic_bp")
    fasting_glucose = patient.get("fasting_glucose")

    # Criterion: BP >=140/90 mmHg in a patient under 40 -- to exclude
    # secondary hypertension.
    if age is not None and age < 40 and systolic is not None and diastolic is not None:
        if systolic >= 140 or diastolic >= 90:
            reasons.append(
                "Blood pressure ≥140/90 mmHg in a patient under 40 years old "
                "— refer to rule out secondary hypertension."
            )

    # Criterion (partial): markedly elevated fasting glucose. The
    # official criterion is "DM with fasting blood glucose >14 mmol/L
    # despite maximal metformin with or without sulphonylurea," but
    # this app does not collect medication data, so it flags on the
    # glucose reading alone -- a reading this high warrants urgent
    # attention whether or not treatment status is known.
    if fasting_glucose is not None and fasting_glucose > SEVERE_FASTING_GLUCOSE_THRESHOLD_MGDL:
        reasons.append(
            f"Fasting blood glucose ({fasting_glucose} mg/dL) is markedly elevated "
            "(>14 mmol/L) — refer for urgent diabetes evaluation and management."
        )

    return reasons
