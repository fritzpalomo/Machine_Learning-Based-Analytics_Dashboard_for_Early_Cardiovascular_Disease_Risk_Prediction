# Updated deployed dashboard (26 Oct 2026)
A copy of `cvd_dashboard` (the version users evaluated: three categories, Composite Risk Score, Model Confidence Score, local SHAP) with the changes suggested in the usability survey (n = 30). The model, its features and its categories are unchanged, so **no retraining is needed**; your existing `models/` files keep working.

## Changes
- Height and weight replace the BMI box; BMI is worked out and shown with its Asia-Pacific category (`bmi.py`).
- "Why this prediction?" sentence above the SHAP chart, naming the factors that raised and lowered risk, so Section 3 connects to Section 2 (`explain.py`).
- "How to read these results" box explaining risk category, Composite Risk Score, Model Confidence Score and the SHAP chart in plain language.
- Warnings: age under 40 or 80 and over (chart range 40 to 79), fasting glucose of 126 mg/dL or higher (diabetic range), implausible BMI (`validation.py`).
- Current medication checkboxes (BP, diabetes, statin); raised BP on treatment adds a "Review Medication" card.
- Diet recommendations tailored to BP, glucose, cholesterol and BMI; "Glucose in Diabetic Range" card.
- Lighter background, stacked layout and larger buttons on phones.
- Optional made-up case code (no names or IDs) saved with each prediction; Reports page has "Track one patient". Existing databases are migrated automatically (`db.py`).
- Tests: `tests/test_survey_changes.py` (9 checks) and 5 new checks in `tests/test_validation.py`.

## Not changed
The login, the three categories and the model files. The Google Form label fix (Maintainability) is in the form, not here.
