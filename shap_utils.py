"""
shap_utils.py
-------------
Computes local SHAP values for a single patient and builds the
red/blue "factors increasing/decreasing risk" bar chart used in
Module 3 of the dashboard.
"""

import numpy as np
import plotly.graph_objects as go


def compute_shap_values(explainer, preprocessor, feature_names, patient_df, predicted_class_index):
    """
    Returns a dict of {feature_name: shap_value} for the predicted class,
    for a single-row patient_df (already in raw/original column form).
    """
    X_transformed = preprocessor.transform(patient_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer.shap_values(X_transformed)

    # TreeExplainer on multi-class returns a list of arrays (one per class)
    # or a 3D array depending on SHAP version; handle both.
    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class_index][0]
    elif shap_values.ndim == 3:
        class_shap = shap_values[0, :, predicted_class_index]
    else:
        class_shap = shap_values[0]

    return dict(zip(feature_names, class_shap))


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
        title="SHAP Value (Impact on Model Output)",
        xaxis_title="SHAP Value",
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
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "fasting_glucose",
    "bmi",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity",
    "family_history_cvd",
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
    "family_history_cvd": {
        "icon": "🧬",
        "title": "Prioritize Regular Check-ups",
        "desc": "Family history is contributing to your risk. Since this factor can't be changed, more frequent screening is especially important for you.",
    },
    "age": {
        "icon": "📋",
        "title": "Prioritize Regular Check-ups",
        "desc": "Age is contributing to your risk. Regular check-ups help catch and manage cardiovascular risk early.",
    },
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


def get_dynamic_recommendations(shap_dict: dict, top_n: int = 4):
    """
    Returns up to top_n recommendation cards for this patient, chosen from
    RECOMMENDATIONS based on which base features have the largest positive
    (risk-increasing) SHAP values. Falls back to DEFAULT_RECOMMENDATIONS
    entries to fill any remaining slots if fewer than top_n factors qualify.
    """
    # Aggregate SHAP value per base feature (categorical one-hot columns collapse together)
    base_scores = {}
    for raw_name, value in shap_dict.items():
        if value <= 0:
            continue  # only risk-increasing factors
        base = base_feature_name(raw_name)
        base_scores[base] = base_scores.get(base, 0) + value

    ranked_bases = sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)

    cards = []
    seen_titles = set()
    for base, _ in ranked_bases:
        rec = RECOMMENDATIONS.get(base)
        if rec and rec["title"] not in seen_titles:
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

    return cards