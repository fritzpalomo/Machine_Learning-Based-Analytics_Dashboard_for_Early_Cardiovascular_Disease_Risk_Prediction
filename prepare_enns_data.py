"""
prepare_enns_data.py
--------------------
Builds data/patients.csv from the cleaned ENNS dataset (2018, 2019, 2021 and 2023
surveys; adults aged 40 to 79) so train_model.py learns the five PhilPEN levels.

    python prepare_enns_data.py ENNS_cleaned_dataset.xlsx

The label is the WHO/ISH chart level (chart_risk_level), the same five levels
the PhilPEN manual uses. The model reproduces the chart from the patient's
results; it does not predict future heart attacks or strokes. Family history is
not in ENNS, so it is not a model input. household_group is kept so the
train/test split never puts one household in both sets.
"""

import sys

import pandas as pd

from levels import LEVEL_NAMES

LABELS = {"<10%": 0, "10 to <20%": 1, "20 to <30%": 2, "30 to <40%": 3, ">=40%": 4}


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "age": df["age_years"].round(0).astype(int),
        "sex": df["sex"],
        "systolic_bp": df["systolic_bp_mmHg"],
        "diastolic_bp": df["diastolic_bp_mmHg"],
        "total_cholesterol": df["total_cholesterol_mgdl"],
        "fasting_glucose": df["fasting_glucose_mgdl"],
        "bmi": df["bmi"],
        "smoking_status": df["smoking_status"],
        "alcohol_consumption": df["alcohol_status"],
        "physical_activity": df["physical_activity"],
        "household_group": df["household_group"],
        "risk_category": df["chart_risk_level"].map(LABELS).map(lambda i: LEVEL_NAMES[int(i)]),
    })
    return out.dropna(subset=["risk_category"])


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "ENNS_cleaned_dataset.xlsx"
    data = build(pd.read_excel(src, sheet_name="Cleaned_Data"))
    import os
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/patients.csv", index=False)
    print(f"Wrote data/patients.csv: {len(data)} rows")
    print(data["risk_category"].value_counts().sort_index().to_string())
