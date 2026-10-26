"""
levels.py
---------
The five PhilPEN risk levels as the dashboard's risk categories. The model is
trained on the WHO/ISH chart level, so its classes are exactly these five.
Pure functions, no Streamlit.
"""

from philpen_chart import LEVELS

# "Level 1 (< 10%)" ... "Level 5 (≥ 40%)"; sorts correctly as text.
LEVEL_NAMES = [f"Level {l['number']} ({l['label']})" for l in LEVELS]
LEVEL_COLORS = {n: l["color"] for n, l in zip(LEVEL_NAMES, LEVELS)}

# Legacy three-category labels, still found in older saved predictions.
LEGACY_COLORS = {"Low Risk": "#16a34a", "Moderate Risk": "#f59e0b", "High Risk": "#dc2626"}
LEGACY_WEIGHTS = {"Low Risk": 0, "Moderate Risk": 50, "High Risk": 100}


def level_number(name: str):
    """'Level 3 (20 to < 30%)' -> 3; anything else -> None."""
    if isinstance(name, str) and name.startswith("Level "):
        try:
            return int(name.split()[1])
        except (IndexError, ValueError):
            return None
    return None


def level_color(name: str) -> str:
    return LEVEL_COLORS.get(name) or LEGACY_COLORS.get(name, "#6b7280")


def level_weight(name: str, n_levels: int = 5) -> float:
    """Severity weight 0..100: Level 1 = 0, Level 5 = 100 (equal steps)."""
    n = level_number(name)
    if n is not None:
        return (n - 1) * 100 / (n_levels - 1)
    return LEGACY_WEIGHTS.get(name, 50)


def composite_score(classes, probabilities) -> float:
    """Probability-weighted position along the Level 1 to Level 5 spectrum (0..100)."""
    return float(sum(p * level_weight(c, len(classes)) for c, p in zip(classes, probabilities)))


def chart_reference(patient: dict, extras: dict = None):
    """WHO/ISH chart level for the patient, or None if the age is outside 40 to 79
    (the chart does not cover it). Diabetic if fasting glucose is 126 mg/dL or
    higher, or the patient is on diabetes medicine."""
    from philpen_chart import chart_level
    age = patient.get("age")
    if age is None or not (40 <= age <= 79):
        return None
    diabetic = (patient.get("fasting_glucose") or 0) >= 126 or bool((extras or {}).get("on_diabetes_medication"))
    res = chart_level(age, patient.get("sex"), patient.get("smoking_status") == "Current",
                      diabetic, patient.get("systolic_bp"), patient.get("total_cholesterol"))
    res["name_full"] = LEVEL_NAMES[res["index"]]
    return res
