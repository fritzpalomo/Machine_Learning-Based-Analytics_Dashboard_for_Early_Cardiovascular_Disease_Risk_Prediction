"""Five PhilPEN levels as the dashboard's categories."""
import numpy as np
import pandas as pd
from levels import LEVEL_NAMES, level_color, level_number, level_weight, composite_score, chart_reference
import prepare_enns_data


def test_five_levels_in_order():
    assert LEVEL_NAMES == ["Level 1 (< 10%)", "Level 2 (10 to < 20%)", "Level 3 (20 to < 30%)",
                           "Level 4 (30 to < 40%)", "Level 5 (≥ 40%)"]
    assert sorted(LEVEL_NAMES) == LEVEL_NAMES  # LabelEncoder keeps Level 1..5 order


def test_weights_and_colours():
    assert [level_weight(n) for n in LEVEL_NAMES] == [0, 25, 50, 75, 100]
    assert level_color(LEVEL_NAMES[0]) == "#16a34a" and level_color(LEVEL_NAMES[4]) == "#dc2626"
    assert level_color("High Risk") == "#dc2626"  # older saved rows still display
    assert level_number("Level 3 (20 to < 30%)") == 3 and level_number("High Risk") is None


def test_composite_score_extremes_and_middle():
    assert composite_score(LEVEL_NAMES, [1, 0, 0, 0, 0]) == 0
    assert composite_score(LEVEL_NAMES, [0, 0, 0, 0, 1]) == 100
    assert abs(composite_score(LEVEL_NAMES, [0.5, 0, 0, 0, 0.5]) - 50) < 1e-9


def test_chart_reference_matches_chart_and_age_range():
    p = {"age": 54, "sex": "Male", "systolic_bp": 132, "total_cholesterol": 210,
         "fasting_glucose": 110, "smoking_status": "Current"}
    ref = chart_reference(p, {})
    assert ref["name_full"] in LEVEL_NAMES and 1 <= ref["number"] <= 5
    assert chart_reference({**p, "age": 35}) is None
    # a diabetic reading (>= 126) cannot lower the level
    assert chart_reference({**p, "fasting_glucose": 140})["number"] >= ref["number"]


def test_prepare_maps_labels_and_drops_family_history():
    df = pd.DataFrame({
        "age_years": [45.2, 66.0], "sex": ["Male", "Female"], "systolic_bp_mmHg": [120, 170],
        "diastolic_bp_mmHg": [80, 95], "total_cholesterol_mgdl": [200, 260], "fasting_glucose_mgdl": [90, 150],
        "bmi": [22, 31], "smoking_status": ["Never", "Current"], "alcohol_status": ["Never", "Current"],
        "physical_activity": ["High", "Low"], "household_group": [1, 2],
        "chart_risk_level": ["<10%", ">=40%"],
    })
    out = prepare_enns_data.build(df)
    assert list(out["risk_category"]) == [LEVEL_NAMES[0], LEVEL_NAMES[4]]
    assert "family_history_cvd" not in out.columns and "household_group" in out.columns


def test_pipeline_trains_five_classes_with_stand_in_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from preprocessing import build_preprocessor, ALL_FEATURES
    assert "family_history_cvd" not in ALL_FEATURES
    from philpen_chart import chart_level
    rng = np.random.default_rng(1)
    n = 1500
    d = pd.DataFrame({
        "age": rng.integers(40, 80, n), "sex": rng.choice(["Male", "Female"], n),
        "systolic_bp": rng.integers(100, 200, n), "diastolic_bp": rng.integers(60, 110, n),
        "total_cholesterol": rng.integers(150, 320, n), "fasting_glucose": rng.integers(70, 200, n),
        "bmi": rng.uniform(17, 35, n).round(1), "smoking_status": rng.choice(["Never", "Former", "Current"], n),
        "alcohol_consumption": rng.choice(["Never", "Former", "Current"], n),
        "physical_activity": rng.choice(["High", "Low"], n),
    })
    y = [LEVEL_NAMES[chart_level(r.age, r.sex, r.smoking_status == "Current", r.fasting_glucose >= 126,
                                 r.systolic_bp, r.total_cholesterol)["index"]] for r in d.itertuples()]
    le = LabelEncoder(); ye = le.fit_transform(y)
    assert len(le.classes_) >= 4
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("clf", GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=0))])
    pipe.fit(d[ALL_FEATURES][:1200], ye[:1200])
    assert (pipe.predict(d[ALL_FEATURES][1200:]) == ye[1200:]).mean() > 0.6  # small random sample; the real model is trained on ENNS
    proba = pipe.predict_proba(d[ALL_FEATURES][1200:1201])[0]
    assert 0 <= composite_score(le.classes_, proba) <= 100
