"""
philpen_chart.py
----------------
The WHO/ISH cardiovascular risk prediction chart for the Western Pacific B
subregion, as adopted in the DOH Operations Manual on NCD Prevention and
Control (PhilPEN), as a pure lookup, plus the five PhilPEN risk levels and the
management actions the manual attaches to them.

The chart values were digitised from the manual's Figures (laboratory-based
and non-laboratory-based charts) into who_ish_wprb_digitised.json:

    key   = "<diabetes 0/1>|<age band 40..70>|<sex M/F>|<smoker 0/1>"
    value = rows of SBP bands, top to bottom: >=170, 150-169, 130-149, <130
            (the WHO chart rows 180, 160, 140, 120)
            lab chart: 5 columns for total cholesterol 4, 5, 6, 7, 8 mmol/L
            non-lab chart: one column
    cell  = risk level 0..4 (0 = <10%, 1 = 10-<20%, 2 = 20-<30%,
            3 = 30-<40%, 4 = >=40%)

This is a deterministic chart lookup. It reproduces the chart; it is not a
prediction of observed cardiovascular events.
"""

import json
import os

_CHART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "who_ish_wprb_digitised.json")
with open(_CHART_PATH, encoding="utf-8") as _f:
    _CHART = json.load(_f)

MMOL_PER_MGDL = 1 / 38.67  # the manual divides mg/dL by 38

LEVELS = [
    {"number": 1, "label": "< 10%", "name": "Low", "color": "#16a34a"},
    {"number": 2, "label": "10 to < 20%", "name": "Moderate", "color": "#84cc16"},
    {"number": 3, "label": "20 to < 30%", "name": "High", "color": "#f59e0b"},
    {"number": 4, "label": "30 to < 40%", "name": "Very high", "color": "#ea580c"},
    {"number": 5, "label": "≥ 40%", "name": "Very high", "color": "#dc2626"},
]

SBP_ROWS = ["≥ 170 mmHg", "150 to 169 mmHg", "130 to 149 mmHg", "< 130 mmHg"]

# Management actions per level: DOH Operations Manual on NCD Prevention and
# Control, Action 5 and second-visit guidance. Wording must be checked by a
# clinician before clinical use.
ACTIONS = {
    1: {"counsel": True, "follow_up": "Follow up in 12 months.", "treatment": None},
    2: {"counsel": True, "follow_up": "Follow up every 3 months until targets are met, then every 6 to 9 months.", "treatment": None},
    3: {"counsel": True, "follow_up": "Follow up every 3 to 6 months. Prescribe attendance at health education classes.",
        "treatment": "If blood pressure stays at or above 140/90 mmHg (130/80 in diabetes), consider a low dose of one drug (hydrochlorothiazide, enalapril, atenolol or amlodipine)."},
    4: {"counsel": True, "follow_up": "Follow up every 3 months. Prescribe health education classes. If risk is still above 30% after 3 to 6 months of interventions, refer to the next-level facility.",
        "treatment": "For persistent raised blood pressure give one of a thiazide, ACE inhibitor, beta-blocker or calcium channel blocker. Give a statin."},
    5: {"counsel": True, "follow_up": "Follow up every 3 months. Prescribe health education classes. If risk is still above 30% after 3 to 6 months of interventions, refer to the next-level facility.",
        "treatment": "Same actions as level 4 (the manual groups all risk above 30% together): antihypertensive for persistent raised blood pressure, and a statin."},
}

CHART_MIN_AGE, CHART_MAX_AGE = 40, 79


def age_band(age: float) -> int:
    """Chart age box: 40-49 -> 40, ..., 70 and over -> 70 (clipped at both ends)."""
    return int(min(max((age // 10) * 10, 40), 70))


def sbp_row(sbp: float) -> int:
    """Row index into the chart: 0 = >=170, 1 = 150-169, 2 = 130-149, 3 = <130."""
    if sbp < 130:
        return 3
    if sbp < 150:
        return 2
    if sbp < 170:
        return 1
    return 0


def cholesterol_column(chol_mgdl: float) -> int:
    """Column index (0..4) for total cholesterol: 4, 5, 6, 7, 8 mmol/L (nearest)."""
    mmol = chol_mgdl / 38.0
    return int(min(max(round(mmol), 4), 8)) - 4


def _cell(chart: str, diabetic: bool, band: int, sex_code: str, smoker: bool, row: int, col: int = 0) -> int:
    key = f"{int(diabetic)}|{band}|{sex_code}|{int(smoker)}"
    return _CHART[chart][key][row][col]


def chart_level(age, sex, smoker, diabetic, sbp, total_cholesterol_mgdl=None) -> dict:
    """
    Looks up the PhilPEN risk level.

    sex: "Male" / "Female". smoker: True for a current smoker. diabetic: True
    if the patient has diabetes. total_cholesterol_mgdl: optional; when given,
    the laboratory chart is used, otherwise the chart without cholesterol (the
    manual says to use it when cholesterol cannot be measured).
    """
    sex_code = "M" if str(sex).lower().startswith("m") else "F"
    band = age_band(age)
    row = sbp_row(sbp)
    use_lab = total_cholesterol_mgdl is not None
    col = cholesterol_column(total_cholesterol_mgdl) if use_lab else 0
    chart = "lab" if use_lab else "nonlab"
    idx = _cell(chart, bool(diabetic), band, sex_code, bool(smoker), row, col)

    # What the level would be one SBP band lower / higher (for the explanation panel).
    lower = _cell(chart, bool(diabetic), band, sex_code, bool(smoker), min(row + 1, 3), col)
    higher = _cell(chart, bool(diabetic), band, sex_code, bool(smoker), max(row - 1, 0), col)

    return {
        "index": idx,
        "number": idx + 1,
        "label": LEVELS[idx]["label"],
        "name": LEVELS[idx]["name"],
        "color": LEVELS[idx]["color"],
        "chart": chart,
        "chart_name": "laboratory-based chart (with total cholesterol)" if use_lab else "non-laboratory chart (without cholesterol)",
        "age_band": f"{band}" + ("-" + str(band + 9) if band < 70 else " and over"),
        "sex_code": sex_code,
        "smoker": bool(smoker),
        "diabetic": bool(diabetic),
        "sbp_row": SBP_ROWS[row],
        "cholesterol_column": (f"{cholesterol_column(total_cholesterol_mgdl) + 4} mmol/L" if use_lab else None),
        "level_if_sbp_lower_band": LEVELS[lower]["number"] if row < 3 else None,
        "level_if_sbp_higher_band": LEVELS[higher]["number"] if row > 0 else None,
        "age_in_chart_range": CHART_MIN_AGE <= age <= CHART_MAX_AGE,
    }
