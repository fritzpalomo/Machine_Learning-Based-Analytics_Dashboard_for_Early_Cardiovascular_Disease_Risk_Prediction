"""
test_recommendations.py
------------------------
Unit tests for the personalized recommendations logic in shap_utils.py.
"""

import pytest
from shap_utils import (
    base_feature_name,
    get_dynamic_recommendations,
    RECOMMENDATIONS,
    DEFAULT_RECOMMENDATIONS,
)


def test_base_feature_name_passthrough_for_numeric():
    assert base_feature_name("age") == "age"
    assert base_feature_name("bmi") == "bmi"


def test_base_feature_name_collapses_one_hot_categorical():
    assert base_feature_name("smoking_status_Current") == "smoking_status"
    assert base_feature_name("family_history_cvd_Yes") == "family_history_cvd"
    assert base_feature_name("physical_activity_No") == "physical_activity"


def test_recommendations_ranked_by_shap_magnitude():
    shap_dict = {
        "systolic_bp": 0.42,       # strongest risk-increasing factor
        "smoking_status_Current": 0.35,
        "family_history_cvd_Yes": 0.20,
        "physical_activity_No": 0.15,
        "bmi": -0.10,              # risk-decreasing, should be excluded
        "total_cholesterol": -0.05,
    }
    cards = get_dynamic_recommendations(shap_dict, top_n=4)
    titles = [c["title"] for c in cards]
    assert titles[0] == RECOMMENDATIONS["systolic_bp"]["title"]
    assert len(cards) == 4


def test_recommendations_fallback_when_few_risk_factors():
    shap_dict = {"age": -0.1, "bmi": -0.2, "systolic_bp": 0.01}
    cards = get_dynamic_recommendations(shap_dict, top_n=4)
    assert len(cards) == 4
    # should be filled from the generic defaults
    default_titles = {r["title"] for r in DEFAULT_RECOMMENDATIONS}
    assert any(c["title"] in default_titles for c in cards)


def test_recommendations_no_duplicate_titles():
    # both diastolic and systolic map to the same "Monitor Blood Pressure" card
    shap_dict = {"systolic_bp": 0.3, "diastolic_bp": 0.25, "bmi": 0.1, "age": 0.05}
    cards = get_dynamic_recommendations(shap_dict, top_n=4)
    titles = [c["title"] for c in cards]
    assert len(titles) == len(set(titles)), "Recommendation titles should not repeat"


def test_recommendations_all_have_required_fields():
    for rec in list(RECOMMENDATIONS.values()) + DEFAULT_RECOMMENDATIONS:
        assert set(rec.keys()) >= {"icon", "title", "desc"}
        assert rec["desc"].strip() != ""
