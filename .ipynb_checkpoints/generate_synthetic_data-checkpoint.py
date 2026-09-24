"""
generate_synthetic_data.py
---------------------------
Generates a SYNTHETIC patient dataset that mimics the structure of a
WHO/PhilPEN cardiovascular risk dataset, purely so the rest of the
pipeline (preprocessing -> training -> SHAP -> dashboard) can run
end-to-end before you plug in the real, anonymized DOH dataset.

Replace this script's output (data/patients.csv) with your real
dataset, keeping the SAME COLUMN NAMES, and everything downstream
(train_model.py, app.py) will work unchanged.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 2000

def generate_synthetic_dataset(n=N_SAMPLES, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(25, 80, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    systolic_bp = rng.normal(130, 20, size=n).clip(90, 220).round(0)
    diastolic_bp = rng.normal(82, 12, size=n).clip(55, 130).round(0)
    total_cholesterol = rng.normal(200, 40, size=n).clip(120, 350).round(0)
    fasting_glucose = rng.normal(105, 30, size=n).clip(70, 300).round(0)
    bmi = rng.normal(26, 5, size=n).clip(15, 45).round(1)

    smoking_status = rng.choice(
        ["Never", "Former", "Current"], size=n, p=[0.55, 0.25, 0.20]
    )
    alcohol_consumption = rng.choice(
        ["None", "Occasional", "Regular"], size=n, p=[0.4, 0.4, 0.2]
    )
    physical_activity = rng.choice(["Yes", "No"], size=n, p=[0.5, 0.5])
    family_history_cvd = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])

    # --- Build a simple latent risk score to make labels realistic ---
    risk_score = (
        0.03 * (age - 25)
        + 0.02 * (systolic_bp - 90)
        + 0.015 * (total_cholesterol - 120)
        + 0.02 * (fasting_glucose - 70)
        + 0.5 * (bmi - 18)
        + np.where(smoking_status == "Current", 8, np.where(smoking_status == "Former", 3, 0))
        + np.where(family_history_cvd == "Yes", 6, 0)
        + np.where(physical_activity == "No", 4, 0)
        + rng.normal(0, 5, size=n)  # noise
    )

    # Convert continuous risk score into 3 WHO/PhilPEN-style categories
    low_cut, high_cut = np.percentile(risk_score, [40, 80])
    risk_category = np.where(
        risk_score < low_cut, "Low Risk",
        np.where(risk_score < high_cut, "Moderate Risk", "High Risk")
    )

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "total_cholesterol": total_cholesterol,
        "fasting_glucose": fasting_glucose,
        "bmi": bmi,
        "smoking_status": smoking_status,
        "alcohol_consumption": alcohol_consumption,
        "physical_activity": physical_activity,
        "family_history_cvd": family_history_cvd,
        "risk_category": risk_category,
    })
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    out_path = "data/patients.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to {out_path}")
    print(df["risk_category"].value_counts())
    print(df.head())
