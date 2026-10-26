"""
explain.py
----------
Plain-language wording for the prediction, added after the usability survey
(evaluators asked for a simpler explanation of the SHAP chart and of the
difference between the risk category, the Composite Risk Score and the
Model Confidence Score). Pure functions, no Streamlit.
"""

from shap_utils import base_feature_name, readable_feature_name

GLOSSARY = [
    ("Risk category", "One of the five PhilPEN levels, from Level 1 (below 10% ten-year risk) to Level 5 (40% or more). This is the main result."),
    ("Composite Risk Score", "A 0 to 100 position along the Level 1 to Level 5 spectrum, worked out from the model's probabilities for all five levels. "
                             "It shows how close the patient is to the next category. It is not the chance of a heart attack or stroke."),
    ("Model Confidence Score", "How sure the model is of the category it chose. It is not the patient's risk: a model can be very sure that a patient is Level 1."),
    ("SHAP chart", "Shows which of the patient's own results pushed the prediction up (red) or down (blue). It explains the model, not the cause of disease."),
]


def why_this_prediction(shap_dict: dict, category: str, top_n: int = 3) -> str:
    """One or two sentences naming the factors that most raised and lowered the level.
    Factors with a negligible effect (under 5% of the largest) are left out."""
    biggest = max((abs(v) for v in shap_dict.values()), default=0)
    cut = 0.05 * biggest
    up = sorted(((v, k) for k, v in shap_dict.items() if v > cut), reverse=True)[:top_n]
    down = sorted(((v, k) for k, v in shap_dict.items() if v < -cut))[:2]
    name = lambda k: readable_feature_name(k).lower()
    text = f"The model placed this patient in **{category}**."
    if up:
        text += " The factors that pushed the result towards higher risk most were " + ", ".join(name(k) for _, k in up) + "."
    else:
        text += " No factor pushed the result towards a higher level."
    if down:
        text += " The factors that lowered it most were " + " and ".join(name(k) for _, k in down) + "."
    return text
