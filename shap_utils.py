"""
shap_utils.py
-------------
Computes local SHAP values for a single patient and builds the
red/blue "factors increasing/decreasing risk" bar chart used in
Module 3 of the dashboard.
"""

import numpy as np
import plotly.graph_objects as go

from preprocessing import CATEGORICAL_FEATURES


def compute_shap_values(explainer, preprocessor, feature_names, patient_df, predicted_class_index, direction_weights=None):
    """
    Returns a dict of {feature_name: shap_value} for the predicted class,
    for a single-row patient_df (already in raw/original column form).

    One-hot encoding turns each categorical field into multiple columns
    (e.g. smoking_status_Never, smoking_status_Current). SHAP computes a
    value for every column regardless of whether it's the patient's actual
    answer, which would otherwise show a category the patient did NOT
    select (e.g. "_Yes") as a contributing factor for a patient who
    answered "No". This function keeps only the one-hot column matching
    what the patient actually selected for each categorical field.
    """
    X_transformed = preprocessor.transform(patient_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer.shap_values(X_transformed)

    # TreeExplainer on multi-class returns a list of arrays (one per class)
    # or a 3D array depending on SHAP version; normalise to (classes, features).
    if isinstance(shap_values, list):
        per_class = np.array([sv[0] for sv in shap_values])
    elif shap_values.ndim == 3:
        per_class = shap_values[0].T
    else:
        per_class = shap_values[0][None, :]

    if direction_weights is not None and per_class.shape[0] == len(direction_weights):
        # Direction of risk: with five ordered levels, a bar for the predicted
        # class alone is misleading (for a Level 1 patient a positive value
        # means "towards Level 1", i.e. lower risk). Combine the class values
        # with centred severity weights so positive always means "pushes
        # towards a higher level" and negative "towards a lower level".
        w = np.asarray(direction_weights, dtype=float)
        w = w - w.mean()
        class_shap = (w[:, None] * per_class).sum(axis=0)
    else:
        class_shap = per_class[min(predicted_class_index, per_class.shape[0] - 1)]

    raw_dict = dict(zip(feature_names, class_shap))
    return _keep_only_active_categories(raw_dict, patient_df)


def _keep_only_active_categories(shap_dict: dict, patient_df) -> dict:
    """Drops one-hot columns for categorical values the patient did NOT select."""
    patient_row = patient_df.iloc[0]
    filtered = {}
    for name, value in shap_dict.items():
        base = base_feature_name(name)
        if base in CATEGORICAL_FEATURES:
            actual_value = str(patient_row[base])
            expected_name = f"{base}_{actual_value}"
            if name != expected_name:
                continue  # this one-hot column is inactive for this patient — skip it
        filtered[name] = value
    return filtered


def build_shap_bar_chart(shap_dict: dict, top_n: int = 8):
    """Builds a horizontal bar chart: red = increases risk, blue = decreases risk."""
    items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    items = sorted(items, key=lambda kv: kv[1])  # ascending for horizontal bar display

    labels = [readable_feature_name(k) for k, _ in items]
    values = [v for _, v in items]
    colors = ["#2563eb" if v < 0 else "#dc2626" for v in values]  # blue decreases, red increases

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Local SHAP: effect on the predicted level",
        xaxis_title="Towards a lower level  ←  SHAP value  →  towards a higher level",
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
        showlegend=False,
    )
    return fig


def readable_feature_name(raw_name: str) -> str:
    """Converts encoded feature names like 'smoking_status_Current' into readable labels."""
    mapping = {
        "age": "Age",
        "systolic_bp": "Systolic Blood Pressure",
        "diastolic_bp": "Diastolic Blood Pressure",
        "total_cholesterol": "Total Cholesterol",
        "fasting_glucose": "Fasting Blood Glucose",
        "bmi": "BMI",
    }
    if raw_name in mapping:
        return mapping[raw_name]
    # one-hot encoded categorical, e.g. "smoking_status_Current"
    return raw_name.replace("_", " ").title()


# ------------------------------------------------------------------
# Personalized recommendations
# ------------------------------------------------------------------
# Maps each underlying clinical/lifestyle feature to guidance shown
# when SHAP flags that feature as increasing this patient's risk.
BASE_FEATURES = [
    "age",
    "sex",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "fasting_glucose",
    "bmi",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity",
]

RECOMMENDATIONS = {
    "systolic_bp": {
        "icon": "🩺",
        "title": "Monitor Blood Pressure",
        "desc": "Your systolic blood pressure is contributing to your risk. Monitor it regularly and keep it within the normal range.",
    },
    "diastolic_bp": {
        "icon": "🩺",
        "title": "Monitor Blood Pressure",
        "desc": "Your diastolic blood pressure is contributing to your risk. Monitor it regularly and keep it within the normal range.",
    },
    "total_cholesterol": {
        "icon": "🥗",
        "title": "Manage Cholesterol",
        "desc": "Elevated cholesterol is contributing to your risk. A diet low in saturated fat and rich in fruits, vegetables, and whole grains can help.",
    },
    "fasting_glucose": {
        "icon": "🩸",
        "title": "Monitor Blood Sugar",
        "desc": "Your fasting glucose is contributing to your risk. Limit refined sugar, monitor levels regularly, and consult your doctor about screening for diabetes.",
    },
    "bmi": {
        "icon": "⚖️",
        "title": "Maintain a Healthy Weight",
        "desc": "Your BMI is contributing to your risk. A combination of balanced diet and regular activity can help bring it toward a healthier range.",
    },
    "smoking_status": {
        "icon": "🚭",
        "title": "Avoid Tobacco",
        "desc": "Smoking status is contributing to your risk. Quitting smoking and avoiding secondhand smoke significantly lowers cardiovascular risk.",
    },
    "alcohol_consumption": {
        "icon": "🍷",
        "title": "Limit Alcohol Consumption",
        "desc": "Alcohol consumption is contributing to your risk. Reducing intake can help lower your cardiovascular risk over time.",
    },
    "physical_activity": {
        "icon": "🚶",
        "title": "Increase Physical Activity",
        "desc": "Low physical activity is contributing to your risk. Aim for at least 150 minutes of moderate activity per week, e.g. brisk walking.",
    },
    "age": {
        "icon": "📋",
        "title": "Prioritize Regular Check-ups",
        "desc": "Age is contributing to your risk. Regular check-ups help catch and manage cardiovascular risk early.",
    },
}

# For categorical features, a recommendation's advice text is only valid
# for CERTAIN selected values (e.g. "quit smoking" only makes sense if the
# patient actually smokes). SHAP can occasionally flag even the "safe"
# category (e.g. smoking_status = "Never") as marginally risk-increasing
# for a given patient/model, especially on small or synthetic datasets --
# without this guard, that would incorrectly surface advice like telling
# a lifelong non-smoker to quit smoking. Categorical features not listed
# here (there are none currently) would always be considered applicable.
CATEGORICAL_RECOMMENDATION_APPLICABLE_VALUES = {
    "smoking_status": {"Current", "Former"},
    "alcohol_consumption": {"Current"},
    "physical_activity": {"Low"},
}

# Shown when no prediction has been made yet, or as filler if fewer
# than the requested number of risk-increasing factors are found.
DEFAULT_RECOMMENDATIONS = [
    {"icon": "🩺", "title": "Monitor Blood Pressure", "desc": "Keep your blood pressure within the normal range and monitor regularly."},
    {"icon": "🥗", "title": "Healthy Diet", "desc": "A balanced diet rich in fruits, vegetables, whole grains, and low in salt."},
    {"icon": "🚶", "title": "Increase Physical Activity", "desc": "Do physical activity for at least 150 minutes per week (e.g., brisk walking)."},
    {"icon": "🚭", "title": "Avoid Tobacco", "desc": "Quit smoking and avoid exposure to secondhand smoke."},
    {"icon": "📋", "title": "Regular Check-ups", "desc": "Monitor your risk regularly and follow your doctor's recommendations."},
]


def base_feature_name(raw_name: str) -> str:
    """Maps a (possibly one-hot encoded) feature name back to its base feature,
    e.g. 'smoking_status_Current' -> 'smoking_status'; 'age' -> 'age'."""
    if raw_name in BASE_FEATURES:
        return raw_name
    for base in BASE_FEATURES:
        if raw_name.startswith(base + "_"):
            return base
    return raw_name


def get_dynamic_recommendations(shap_dict: dict, top_n: int = 4, patient: dict = None):
    """
    Returns up to top_n recommendation cards for this patient, chosen from
    RECOMMENDATIONS based on which base features have the largest positive
    (risk-increasing) SHAP values. Falls back to DEFAULT_RECOMMENDATIONS
    entries to fill any remaining slots if fewer than top_n factors qualify.

    For categorical features, a recommendation only fires if the patient's
    ACTUAL selected value is one the advice applies to (see
    CATEGORICAL_RECOMMENDATION_APPLICABLE_VALUES) -- e.g. smoking-cessation
    advice is suppressed if the patient's smoking status is "Never", even
    if SHAP assigned that column a (small) positive value.
    """
    # Aggregate SHAP value per base feature (categorical one-hot columns collapse together),
    # and separately track which specific category value is active for each categorical base.
    base_scores = {}
    base_active_value = {}
    for raw_name, value in shap_dict.items():
        if value <= 0:
            continue  # only risk-increasing factors
        base = base_feature_name(raw_name)
        base_scores[base] = base_scores.get(base, 0) + value
        if base in CATEGORICAL_FEATURES and raw_name.startswith(base + "_"):
            base_active_value[base] = raw_name[len(base) + 1:]

    ranked_bases = sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)

    cards = []
    seen_titles = set()
    for base, _ in ranked_bases:
        rec = RECOMMENDATIONS.get(base)
        if not rec:
            continue
        applicable_values = CATEGORICAL_RECOMMENDATION_APPLICABLE_VALUES.get(base)
        if applicable_values is not None and base_active_value.get(base) not in applicable_values:
            continue  # e.g. don't tell a "Never" smoker to quit smoking
        if rec["title"] not in seen_titles:
            cards.append(rec)
            seen_titles.add(rec["title"])
        if len(cards) == top_n:
            break

    # Fill remaining slots with generic guidance if not enough risk factors found
    if len(cards) < top_n:
        for rec in DEFAULT_RECOMMENDATIONS:
            if rec["title"] not in seen_titles:
                cards.append(rec)
                seen_titles.add(rec["title"])
            if len(cards) == top_n:
                break

    return _add_patient_cards(cards, patient, top_n)


def _add_patient_cards(cards: list, patient: dict, top_n: int) -> list:
    """Adds diet, glucose and medication cards from the patient's own inputs
    (suggested by evaluators), replacing generic filler cards first. Pure and
    optional: with no patient dict the list is returned unchanged."""
    if not patient:
        return cards
    extra = []
    diet = ["Fill half the plate with vegetables, and add fruit and whole grains."]
    if patient.get("systolic_bp", 0) >= 130 or patient.get("diastolic_bp", 0) >= 80:
        diet.append("Cut salt: go easy on bagoong, patis, soy sauce, instant noodles and dried or canned fish.")
    if (patient.get("fasting_glucose") or 0) >= 100:
        diet.append("Limit sweet drinks and sweets, and watch white rice portions.")
    if (patient.get("total_cholesterol") or 0) >= 200:
        diet.append("Choose fish, skinless chicken and beans over fatty pork and fried food.")
    if (patient.get("bmi") or 0) >= 23:
        diet.append("Smaller portions of rice and fatty food, aiming for a BMI under 23.")
    extra.append({"icon": "🥗", "title": "Healthy Diet", "desc": " ".join(diet)})
    if (patient.get("fasting_glucose") or 0) >= 126:
        extra.append({"icon": "🩸", "title": "Glucose in Diabetic Range",
                      "desc": "Fasting glucose is 126 mg/dL or higher: repeat the test to confirm and review for diabetes."})
    if patient.get("on_bp_medication") and (patient.get("systolic_bp", 0) >= 140 or patient.get("diastolic_bp", 0) >= 90):
        extra.append({"icon": "💊", "title": "Review Medication",
                      "desc": "Blood pressure is still raised on treatment: check adherence and review the dose with the physician."})
    out = [c for c in cards if c["title"] != "Healthy Diet"]
    generic = {r["title"] for r in DEFAULT_RECOMMENDATIONS}
    merged = extra + out
    # drop generic filler first if over the limit
    while len(merged) > top_n and any(c["title"] in generic for c in merged):
        for i in range(len(merged) - 1, -1, -1):
            if merged[i]["title"] in generic and merged[i]["title"] != "Healthy Diet":
                merged.pop(i); break
        else:
            break
    return merged[:top_n]
