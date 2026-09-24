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
from preprocessing import ALL_FEATURES
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
# Header
# ------------------------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown("## Explainable ML Analytics Dashboard")
    st.caption("WHO/PhilPEN Cardiovascular Risk Classification")
with header_col2:
    st.markdown(
        f"<div style='text-align:right'>📅 {datetime.date.today().strftime('%B %d, %Y')}</div>",
        unsafe_allow_html=True,
    )
    logout_control()

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
    st.markdown("### 1️⃣ Patient Information")
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
    else:
        for msg in warnings:
            st.warning(f"⚠️ {msg}")

        pred_encoded = model_pipeline.predict(patient_df)[0]
        pred_proba = model_pipeline.predict_proba(patient_df)[0]
        predicted_class = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = pred_proba[pred_encoded] * 100

        st.session_state["last_prediction"] = {
            "patient_dict": patient_dict,
            "patient_df": patient_df,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "pred_encoded": pred_encoded,
        }

        db.insert_prediction({
            **patient_dict,
            "predicted_risk_category": predicted_class,
            "confidence_score": round(confidence, 2),
        })

# ==================== MODULE 2: PREDICTION RESULT ====================
with col2:
    st.markdown("### 2️⃣ Prediction Result")
    st.caption("WHO/PhilPEN Cardiovascular Risk Classification.")

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        risk_category = result["predicted_class"]
        confidence = result["confidence"]

        risk_colors = {
            "Low Risk": "#16a34a",
            "Moderate Risk": "#f59e0b",
            "High Risk": "#dc2626",
        }
        color = risk_colors.get(risk_category, "#6b7280")

        st.markdown(
            f"""
            <div style='background-color:{color}20; border:2px solid {color};
                        border-radius:10px; padding:16px; text-align:center; margin-bottom:16px;'>
                <div style='font-size:14px; color:#555;'>Predicted Risk Category</div>
                <div style='font-size:28px; font-weight:700; color:{color};'>{risk_category.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Model Confidence Score"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
            },
        ))
        gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.info("This classification is generated by the machine learning model trained using WHO/PhilPEN cardiovascular risk categories.", icon="ℹ️")
    else:
        st.info("Fill in patient information and click **Predict Risk** to see results here.")

# ==================== MODULE 3: LOCAL SHAP EXPLANATION ====================
with col3:
    st.markdown("### 3️⃣ Local SHAP Explanation")
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
st.markdown("### 4️⃣ Recommendations")

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
            <div style='border:1px solid #e5e7eb; border-radius:10px; padding:12px; height:160px;'>
                <div style='font-size:24px;'>{rec['icon']}</div>
                <div style='font-weight:600; margin-top:4px;'>{rec['title']}</div>
                <div style='font-size:12px; color:#666; margin-top:4px;'>{rec['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("ℹ️ This information is for educational purposes only and does not replace professional medical advice, diagnosis, or treatment.")
