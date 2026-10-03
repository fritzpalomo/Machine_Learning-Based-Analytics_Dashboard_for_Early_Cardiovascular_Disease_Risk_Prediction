"""
app.py
------
Explainable ML Analytics Dashboard - WHO/PhilPEN Cardiovascular Risk
Classification. Implements the 4-module layout from the wireframe:

  1. Patient Information (input form)
  2. Prediction Result (risk category + confidence gauge)
  3. Local SHAP Explanation (feature contribution chart)
  4. Educational Information (WHO/DOH guideline cards)

Run: streamlit run app.py
"""

import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db
from auth import require_login, logout_control
from branding import GRADIENT
from preprocessing import ALL_FEATURES
from referral import check_referral_criteria
from shap_utils import compute_shap_values, build_shap_bar_chart, get_dynamic_recommendations
from validation import validate_patient_input

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable ML Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

db.init_db()
require_login()  # halts here until the user signs in successfully


# ------------------------------------------------------------------
# Cached loaders — model artifacts only need to load once per session
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_pipeline = joblib.load("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    return model_pipeline, preprocessor, label_encoder, explainer, feature_names


try:
    model_pipeline, preprocessor, label_encoder, explainer, feature_names = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


# ------------------------------------------------------------------
# Header — same gradient hero treatment as the login screen (branding.py).
# Title, subtitle, date, and the logout control all live inside ONE real
# Streamlit container (styled via its `key` -> CSS class, Streamlit >= 1.38)
# rather than a separate raw-HTML banner with widgets floating below it —
# that split is what caused the awkward gap/seam. Everything here is a
# single gradient box. (If an older Streamlit doesn't support key->CSS
# targeting, this degrades to a plain white box — still functional, just
# not colored — never a broken layout.)
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .st-key-dashboard_header {{
        background: {GRADIENT};
        border-radius: 14px;
        padding: 16px 24px 10px 24px;
        margin-bottom: 14px;
    }}
    .st-key-dashboard_header h3 {{ color: #ffffff !important; margin-bottom: 0; }}
    .st-key-dashboard_header p, .st-key-dashboard_header small {{ color: #ffffff !important; opacity: 0.88; }}
    .st-key-dashboard_header button {{
        background-color: rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.5) !important;
    }}
    .st-key-dashboard_header button:hover {{
        background-color: rgba(255,255,255,0.3) !important;
    }}
    /* Same rhythmic double-beat pulse as the login screen's heart icon. */
    @keyframes heartbeat {{
        0%   {{ transform: scale(1); }}
        14%  {{ transform: scale(1.3); }}
        28%  {{ transform: scale(1); }}
        42%  {{ transform: scale(1.3); }}
        70%  {{ transform: scale(1); }}
        100% {{ transform: scale(1); }}
    }}
    .heartbeat-icon {{
        display: inline-block;
        animation: heartbeat 1.4s ease-in-out infinite;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="dashboard_header"):
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            '### <span class="heartbeat-icon">🫀</span> Explainable ML Analytics Dashboard',
            unsafe_allow_html=True,
        )
        st.caption("WHO/PhilPEN Cardiovascular Risk Classification")
    with header_col2:
        st.markdown(
            f"<div style='text-align:right; font-size:13px; opacity:0.9;'>"
            f"📅 {datetime.date.today().strftime('%B %d, %Y')}</div>",
            unsafe_allow_html=True,
        )
        logout_control()

# ------------------------------------------------------------------
# Global visual polish — soft shadows on card panels, a light page
# background, and consistent numbered module badges. This keeps the
# existing 4-module layout (no structural redesign) but gives it the
# same "designed" card feel evaluators asked for, using branding.py's
# palette so it matches the login screen.
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
    }}
    .module-badge {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 8px;
        background: {GRADIENT}; color: #ffffff; font-weight: 700; font-size: 13px;
        margin-right: 8px; vertical-align: middle;
    }}
    .module-heading {{ font-size: 18px; font-weight: 700; color: inherit; vertical-align: middle; }}
    .rec-tile {{
        border: 1px solid rgba(128,128,128,0.35); border-radius: 12px; padding: 14px;
        min-height: 170px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        background-color: rgba(128,128,128,0.08);
    }}
    .rec-icon-badge {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 10px;
        background-color: rgba(99,102,241,0.2); font-size: 18px; margin-bottom: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def module_badge(number: int, title: str) -> str:
    """HTML for a consistent numbered module heading — a small colored
    badge matching the dashboard's branding, replacing the plain emoji
    numerals with a deliberate, designed look."""
    return (
        f'<div style="margin-bottom:2px;">'
        f'<span class="module-badge">{number}</span>'
        f'<span class="module-heading">{title}</span>'
        f"</div>"
    )


if not MODEL_LOADED:
    st.warning(
        "No trained model found. Run `python generate_synthetic_data.py` "
        "then `python train_model.py` first, then restart the app."
    )
    st.stop()

st.divider()

# ------------------------------------------------------------------
# Layout: 3 main columns (Modules 1-3), then a full-width row (Module 4)
# ------------------------------------------------------------------
col1, col2, col3 = st.columns([1.2, 1, 1.2])

# ==================== MODULE 1: PATIENT INFORMATION ====================
with col1:
    with st.container(border=True):
        st.markdown(module_badge(1, "Patient Information"), unsafe_allow_html=True)
        st.caption("Enter patient demographic, clinical, and lifestyle information.")

        with st.form("patient_form"):
            st.markdown("**Demographic Information**")
            d1, d2 = st.columns(2)
            age = d1.number_input("Age (years)", min_value=1, max_value=120, value=54)
            sex = d2.selectbox("Sex", ["Male", "Female"])

            st.markdown("**Clinical Information**")
            c1, c2 = st.columns(2)
            systolic_bp = c1.number_input("Systolic BP (mmHg)", min_value=70, max_value=260, value=132)
            diastolic_bp = c2.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=84)
            c3, c4 = st.columns(2)
            total_cholesterol = c3.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=210)
            fasting_glucose = c4.number_input("Fasting Blood Glucose (mg/dL)", min_value=50, max_value=400, value=110)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.3, step=0.1)

            st.markdown("**Lifestyle Information**")
            l1, l2 = st.columns(2)
            smoking_status = l1.selectbox("Smoking Status", ["Never", "Former", "Current"], index=1)
            alcohol_consumption = l2.selectbox("Alcohol Consumption", ["None", "Occasional", "Regular"], index=1)
            l3, l4 = st.columns(2)
            physical_activity = l3.selectbox("Physical Activity", ["Yes", "No"], index=0)
            family_history_cvd = l4.selectbox("Family History of CVD", ["Yes", "No"], index=0)

            submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# ==================== PREDICTION LOGIC ====================
if submitted:
    patient_dict = {
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
    }
    patient_df = pd.DataFrame([patient_dict])[ALL_FEATURES]

    errors, warnings = validate_patient_input(patient_dict)

    if errors:
        for msg in errors:
            st.error(f"⚠️ {msg}")
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_shap_dict", None)
        st.session_state.pop("last_referral_reasons", None)
    else:
        for msg in warnings:
            st.warning(f"⚠️ {msg}")

        st.session_state["last_referral_reasons"] = check_referral_criteria(patient_dict)

        pred_encoded = model_pipeline.predict(patient_df)[0]
        pred_proba = model_pipeline.predict_proba(patient_df)[0]
        predicted_class = label_encoder.inverse_transform([pred_encoded])[0]
        # float(...) matters here: some models (e.g. XGBoost) return
        # numpy.float32 probabilities, which sqlite3 silently stores as a
        # corrupt BLOB instead of a number unless cast to a native float
        # first (see DF-05 / db._to_native for the storage-side guard).
        confidence = float(pred_proba[pred_encoded]) * 100

        # A numeric "risk score" (0-100) derived from the model's full
        # probability distribution across the three categories, using
        # severity weights (Low=0, Moderate=50, High=100). This is a
        # DIFFERENT number from "confidence": confidence says how sure the
        # model is in whichever category it picked; risk_score says how far
        # along the low-to-high risk spectrum the model's overall belief
        # sits. It is also NOT the same as the WHO/PhilPEN 10-year CVD
        # event probability -- the model classifies into 3 categories, it
        # does not estimate a continuous clinical event probability.
        RISK_SEVERITY_WEIGHTS = {"Low Risk": 0, "Moderate Risk": 50, "High Risk": 100}
        risk_score = float(sum(
            prob * RISK_SEVERITY_WEIGHTS.get(cls, 50)
            for cls, prob in zip(label_encoder.classes_, pred_proba)
        ))

        st.session_state["last_prediction"] = {
            "patient_dict": patient_dict,
            "patient_df": patient_df,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "risk_score": risk_score,
            "pred_encoded": pred_encoded,
        }

        db.insert_prediction({
            **patient_dict,
            "predicted_risk_category": predicted_class,
            "confidence_score": round(confidence, 2),
        })

# ==================== MODULE 2: PREDICTION RESULT ====================
with col2:
    with st.container(border=True):
        st.markdown(module_badge(2, "Prediction Result"), unsafe_allow_html=True)
        st.caption("WHO/PhilPEN Cardiovascular Risk Classification.")

        if "last_prediction" in st.session_state:
            result = st.session_state["last_prediction"]
            risk_category = result["predicted_class"]
            confidence = result["confidence"]
            risk_score = result["risk_score"]

            # Color for the "Predicted Risk Category" box is keyed to the risk
            # category itself (High/Moderate/Low), since that box communicates
            # the classification, not the confidence.
            risk_colors = {
                "Low Risk": "#16a34a",
                "Moderate Risk": "#f59e0b",
                "High Risk": "#dc2626",
            }
            color = risk_colors.get(risk_category, "#6b7280")

            st.markdown(
                f"""
                <div style='background-color:{color}20; border:2px solid {color};
                            border-radius:10px; padding:16px; text-align:center; margin-bottom:16px;
                            box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
                    <div style='font-size:14px; opacity:0.75;'>Predicted Risk Category</div>
                    <div style='font-size:28px; font-weight:700; color:{color};'>{risk_category.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # The GAUGE is reserved for the Composite Risk Score — its color-banded
            # arc (green/yellow/red) genuinely represents a position on a risk
            # spectrum, so a gauge visual is appropriate here. Bands use equal
            # thirds of the 0-100 scale since risk_score is a model-derived
            # weighted index, not the WHO/PhilPEN clinical event-probability
            # scale, so WHO's exact cut-points (10%, 30%, 40%) don't apply.
            if risk_score < 33:
                risk_gauge_color = "#16a34a"
            elif risk_score < 67:
                risk_gauge_color = "#f59e0b"
            else:
                risk_gauge_color = "#dc2626"

            risk_gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Composite Risk Score"},
                number={"suffix": " / 100"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_gauge_color},
                    "steps": [
                        {"range": [0, 33], "color": "rgba(22,163,74,0.28)"},
                        {"range": [33, 67], "color": "rgba(245,158,11,0.28)"},
                        {"range": [67, 100], "color": "rgba(220,38,38,0.28)"},
                    ],
                },
            ))
            risk_gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(risk_gauge_fig, use_container_width=True)
            st.caption(
                "A numeric risk-level indicator (0 = fully Low Risk, 100 = fully High Risk), "
                "computed from the model's probability across all three categories. This is a "
                "model-derived index, not the WHO/PhilPEN 10-year cardiovascular event probability."
            )

            # Confidence is deliberately shown as a PLAIN NUMBER, not a gauge —
            # giving it a colored, banded arc (the same visual language as the
            # risk gauge above) would make it look like a second risk indicator,
            # which is exactly the confusion raised by evaluators. A neutral
            # stat box keeps it visually distinct from "risk level."
            st.markdown(
                f"""
                <div style='background-color:rgba(128,128,128,0.12); border:2px solid rgba(128,128,128,0.5);
                            border-radius:10px; padding:12px; text-align:center; margin-bottom:8px;'>
                    <div style='font-size:13px; opacity:0.75;'>Model Confidence Score</div>
                    <div style='font-size:28px; font-weight:700;'>{confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("ℹ️ What does the Model Confidence Score mean?"):
                st.write(
                    "This number shows how certain the model is in the risk category it just "
                    "predicted — it does **not** represent the patient's probability of a "
                    "cardiovascular event, which is why it's shown as a plain number rather than "
                    "a colored gauge like the Composite Risk Score above. A model can be highly "
                    "confident in any outcome, including a Low Risk classification; a low score "
                    "here means the model's probabilities were split closely between categories, "
                    "not that the patient is necessarily at lower risk."
                )

            st.info("This classification is generated by the machine learning model trained using WHO/PhilPEN cardiovascular risk categories.", icon="ℹ️")
        else:
            st.info("Fill in patient information and click **Predict Risk** to see results here.")

# ==================== MODULE 3: LOCAL SHAP EXPLANATION ====================
with col3:
    with st.container(border=True):
        st.markdown(module_badge(3, "Local SHAP Explanation"), unsafe_allow_html=True)
        st.caption("Top factors that contributed to the predicted result.")

        if "last_prediction" in st.session_state:
            result = st.session_state["last_prediction"]
            shap_dict = compute_shap_values(
                explainer, preprocessor, feature_names,
                result["patient_df"], result["pred_encoded"]
            )
            st.session_state["last_shap_dict"] = shap_dict
            fig = build_shap_bar_chart(shap_dict)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "SHAP (SHapley Additive exPlanations) values show how each factor "
                "influenced this patient's classification. Positive values increase "
                "risk, negative values decrease risk."
            )
        else:
            st.info("SHAP explanation will appear here after a prediction is made.")

st.divider()

# ==================== MODULE 4: RECOMMENDATIONS ====================
with st.container(border=True):
    st.markdown(module_badge(4, "Recommendations"), unsafe_allow_html=True)

    referral_reasons = st.session_state.get("last_referral_reasons")
    if referral_reasons:
        reasons_html = "".join(f"<li>{reason}</li>" for reason in referral_reasons)
        st.markdown(
            f"""
            <div style='background-color:#fee2e2; border:2px solid #dc2626; border-radius:10px;
                        padding:14px 18px; margin-bottom:16px; box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
                <div style='font-weight:700; color:#991b1b; font-size:16px;'>
                    ⚠️ Refer to a Higher-Level Facility
                </div>
                <ul style='margin:8px 0 2px 18px; color:#7f1d1d; font-size:13px;'>
                    {reasons_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Based on a subset of PhilPEN primary-care referral criteria checkable from this "
            "app's current inputs. This is not exhaustive — criteria requiring data this app "
            "does not collect (e.g. personal history of CVD/stroke/kidney disease, proteinuria, "
            "medication response, foot ulcers) must still be assessed clinically."
        )

    if "last_shap_dict" in st.session_state:
        st.caption("Personalized guidance based on the factors increasing this patient's risk.")
        edu_items = get_dynamic_recommendations(st.session_state["last_shap_dict"], top_n=4)
    else:
        st.caption("Based on WHO and DOH cardiovascular disease prevention guidelines. Run a prediction for personalized recommendations.")
        from shap_utils import DEFAULT_RECOMMENDATIONS
        edu_items = DEFAULT_RECOMMENDATIONS

    edu_cols = st.columns(len(edu_items))
    for edu_col, rec in zip(edu_cols, edu_items):
        with edu_col:
            st.markdown(
                f"""
                <div class="rec-tile">
                    <div class="rec-icon-badge">{rec['icon']}</div>
                    <div style='font-weight:600; margin-top:4px;'>{rec['title']}</div>
                    <div style='font-size:12px; opacity:0.75; margin-top:4px;'>{rec['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("ℹ️ This information is for educational purposes only and does not replace professional medical advice, diagnosis, or treatment.")

st.divider()

# ==================== PREDICTION HISTORY (DATABASE VIEW) ====================
with st.expander("📊 View Prediction History (every interaction saved to the database)"):
    history_rows = db.get_recent_predictions(limit=200)
    if history_rows:
        history_df = pd.DataFrame(history_rows)
        st.caption(f"Showing the {len(history_df)} most recent saved prediction(s), newest first.")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    else:
        st.caption("No predictions have been saved yet. Run a prediction above to see it appear here.")
