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
import settings as ui_settings
from auth import require_login, _do_logout
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

# Per-user display settings (text size, density, ...). Loaded once per login
# session from the database; changes made in the settings popover are saved
# back to it, so they persist across logins.
if "ui_settings" not in st.session_state:
    st.session_state["ui_settings"] = ui_settings.sanitize(
        db.get_user_settings(st.session_state.get("username"))
    )


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
# Status line ABOVE the banner: date + who is signed in, read together.
st.markdown(
    f"<div style='text-align:right; font-size:0.8125rem; opacity:0.85; margin-bottom:6px;'>"
    f"📅 {datetime.date.today().strftime('%B %d, %Y')} &nbsp;·&nbsp; "
    f"👤 Logged in as <b>{st.session_state.get('username')}</b> ({st.session_state.get('role')})"
    f"</div>",
    unsafe_allow_html=True,
)
with st.container(key="dashboard_header"):
    st.markdown("### Explainable ML Analytics Dashboard")
    st.caption("WHO/PhilPEN Cardiovascular Risk Classification")

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
        background: {GRADIENT}; color: #ffffff; font-weight: 700; font-size:0.8125rem;
        margin-right: 8px; vertical-align: middle;
    }}
    .module-heading {{ font-size:1.125rem; font-weight: 700; color: inherit; vertical-align: middle; }}
    .rec-tile {{
        border: 1px solid rgba(128,128,128,0.35); border-radius: 12px; padding: 14px;
        min-height: 170px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        background-color: rgba(128,128,128,0.08);
    }}
    .rec-icon-badge {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 10px;
        background-color: rgba(99,102,241,0.2); font-size:1.125rem; margin-bottom: 6px;
    }}
    /* Animated recommendation icons (pure CSS, so no image files are needed).
       Each emoji gets a motion that fits its meaning. */
    .rec-icon-badge {{ width: 48px; height: 48px; font-size:1.625rem; border-radius: 14px; }}
    .rec-anim {{ display: inline-block; }}
    @keyframes recPulse  {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.25); }} }}
    @keyframes recBounce {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-5px); }} }}
    @keyframes recWalk   {{ 0%,100% {{ transform: translateX(-4px) rotate(-6deg); }} 50% {{ transform: translateX(4px) rotate(6deg); }} }}
    @keyframes recSway   {{ 0%,100% {{ transform: rotate(-12deg); }} 50% {{ transform: rotate(12deg); }} }}
    @keyframes recDrip   {{ 0% {{ transform: translateY(-4px); opacity: 0.4; }} 60% {{ transform: translateY(3px); opacity: 1; }} 100% {{ transform: translateY(-4px); opacity: 0.4; }} }}
    @keyframes recSpin   {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    @keyframes recFlip   {{ 0%,100% {{ transform: rotateY(0deg); }} 50% {{ transform: rotateY(180deg); }} }}
    .anim-pulse  {{ animation: recPulse 1.3s ease-in-out infinite; }}
    .anim-bounce {{ animation: recBounce 1.1s ease-in-out infinite; }}
    .anim-walk   {{ animation: recWalk 0.9s ease-in-out infinite; }}
    .anim-sway   {{ animation: recSway 1.8s ease-in-out infinite; transform-origin: 50% 20%; }}
    .anim-drip   {{ animation: recDrip 1.4s ease-in-out infinite; }}
    .anim-spin   {{ animation: recSpin 4s linear infinite; }}
    .anim-flip   {{ animation: recFlip 2.6s ease-in-out infinite; }}
    @media (prefers-reduced-motion: reduce) {{ .rec-anim {{ animation: none !important; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Which CSS animation each recommendation icon gets (see .anim-* styles above).
REC_ICON_ANIMATION = {
    "🩺": "anim-pulse",   # heartbeat-style pulse
    "🥗": "anim-bounce",  # tossed salad
    "🩸": "anim-drip",
    "⚖️": "anim-sway",    # scale tipping
    "🚭": "anim-pulse",
    "🍷": "anim-sway",
    "🚶": "anim-walk",
    "🧬": "anim-spin",
    "📋": "anim-flip",
}


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


# ------------------------------------------------------------------
# Sidebar navigation — two pages: "Dashboard" (the 4 modules) and
# "Reports" (prediction history from the database). Same navy gradient
# as the login screen and header, with a pill-style active item.
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* Slim floating icon rail (dark navy, rounded), like a modern health-app
       dashboard: only two pages, so icons alone are enough and the main area
       gets the space back. Hover tooltips name each page. */
    section[data-testid="stSidebar"] {{
        width: 92px !important; min-width: 92px !important; max-width: 92px !important;
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{ width: 92px !important; }}
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(180deg, #21295C 0%, #0b1b3a 100%);
        border-radius: 26px;
        margin: 10px 0 10px 10px;
        height: calc(100vh - 20px);
        box-shadow: 0 6px 24px rgba(11, 27, 58, 0.35);
    }}
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarResizeHandle"] {{ display: none !important; }}
    [data-testid="stSidebarContent"] [data-testid="stSidebarUserContent"] {{ padding: 18px 0 0 0; }}
    .nav-logo {{ text-align: center; font-size:1.875rem; line-height: 1; margin: 6px 0 0 0; }}
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
    section[data-testid="stSidebar"] .stButton button[aria-label="Log out"]:hover,
    .st-key-rail_logout button:hover {{ background: rgba(239, 68, 68, 0.3) !important; }}
    section[data-testid="stSidebar"] .stButton {{ display: flex; justify-content: center; margin-bottom: 10px; }}
    section[data-testid="stSidebar"] .stButton button {{
        width: 50px; height: 50px; min-height: 50px; padding: 0;
        border-radius: 16px; border: none;
        background: transparent; color: rgba(255,255,255,0.75);
        transition: background 0.15s ease, color 0.15s ease;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background: rgba(255,255,255,0.12); color: #ffffff;
    }}
    section[data-testid="stSidebar"] .stButton button p {{ display: none; }}
    section[data-testid="stSidebar"] .stButton button span {{ font-size:1.625rem; }}
    /* Active page = cyan gradient tile (the "primary" button). */
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, #22d3ee, #38bdf8);
        color: #0b1b3a; box-shadow: 0 0 16px rgba(34, 211, 238, 0.45);
    }}
    section[data-testid="stSidebar"] [data-testid="stPopover"] {{ display: flex; justify-content: center; margin-bottom: 10px; }}
    section[data-testid="stSidebar"] [data-testid="stPopover"] button {{
        width: 50px; height: 50px; min-height: 50px; padding: 0;
        border-radius: 16px; border: none; background: transparent; color: rgba(255,255,255,0.75);
    }}
    section[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {{ background: rgba(255,255,255,0.12); color: #ffffff; }}
    section[data-testid="stSidebar"] [data-testid="stPopover"] button p {{ display: none; }}
    section[data-testid="stSidebar"] [data-testid="stPopover"] button span {{ font-size: 26px; }}
    section[data-testid="stSidebar"] [data-testid="stPopover"] button svg:last-child {{ display: none; }}
    .stat-card {{
        border: 1px solid rgba(128,128,128,0.35); border-radius: 14px; padding: 14px 16px;
        background-color: rgba(128,128,128,0.08); box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    }}
    .stat-label {{ font-size:0.75rem; opacity: 0.75; }}
    .stat-value {{ font-size:1.625rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def _apply_settings():
    """Reads the settings widgets, validates, applies and saves them."""
    new = ui_settings.sanitize({
        "text_size": st.session_state.get("set_text_size"),
        "density": st.session_state.get("set_density"),
        "reduce_motion": st.session_state.get("set_reduce_motion"),
        "high_contrast": st.session_state.get("set_high_contrast"),
    })
    st.session_state["ui_settings"] = new
    db.save_user_settings(st.session_state.get("username"), new)


if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Dashboard"

st.markdown(ui_settings.build_css(st.session_state["ui_settings"]), unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="nav-logo"><span class="heartbeat-icon">🫀</span></div>', unsafe_allow_html=True)
    # Spacer pushes the two page buttons to the vertical middle of the rail.
    st.markdown("<div style='height: max(0px, calc(50vh - 168px));'></div>", unsafe_allow_html=True)
    if st.button(
        " ", key="nav_btn_dashboard", icon=":material/dashboard:", help="Dashboard",
        type="primary" if st.session_state["nav_page"] == "Dashboard" else "secondary",
    ):
        st.session_state["nav_page"] = "Dashboard"
        st.rerun()
    if st.button(
        " ", key="nav_btn_reports", icon=":material/description:", help="Reports — prediction history",
        type="primary" if st.session_state["nav_page"] == "Reports" else "secondary",
    ):
        st.session_state["nav_page"] = "Reports"
        st.rerun()
    # Log out sits at the bottom of the rail.
    st.markdown("<div style='height: max(0px, calc(50vh - 228px));'></div>", unsafe_allow_html=True)
    # Settings (text size, density, motion, contrast) — saved per user.
    with st.popover("", icon=":material/settings:", help="Display settings"):
        st.markdown("**Display settings**")
        _cur = st.session_state["ui_settings"]
        st.radio(
            "Text size", list(ui_settings.TEXT_SIZES), key="set_text_size",
            index=list(ui_settings.TEXT_SIZES).index(_cur["text_size"]),
            horizontal=True, on_change=_apply_settings,
        )
        st.radio(
            "Density", list(ui_settings.DENSITIES), key="set_density",
            index=list(ui_settings.DENSITIES).index(_cur["density"]),
            horizontal=True, on_change=_apply_settings,
        )
        st.toggle("Reduce motion", value=_cur["reduce_motion"], key="set_reduce_motion",
                  help="Turns off the beating heart and animated icons.", on_change=_apply_settings)
        st.toggle("High contrast", value=_cur["high_contrast"], key="set_high_contrast",
                  help="Stronger borders and fully opaque text.", on_change=_apply_settings)
    if st.button(" ", key="rail_logout", icon=":material/logout:", help="Log out"):
        _do_logout()

page = st.session_state["nav_page"]

if page == "Reports":
    st.markdown(module_badge("📑", "Reports — Prediction History"), unsafe_allow_html=True)
    st.caption("Every prediction saved to the database, newest first. No personally identifying information is stored.")
    history_rows = db.get_recent_predictions(limit=1000)
    if not history_rows:
        st.info("No predictions have been saved yet. Run a prediction on the Dashboard page and it will appear here.")
        st.stop()

    history_df = pd.DataFrame(history_rows)
    total = len(history_df)
    counts = history_df["predicted_risk_category"].value_counts()
    avg_conf = pd.to_numeric(history_df["confidence_score"], errors="coerce").mean()

    stat_cols = st.columns(5)
    stats = [
        ("Total predictions", f"{total}"),
        ("Low Risk", f"{int(counts.get('Low Risk', 0))}"),
        ("Moderate Risk", f"{int(counts.get('Moderate Risk', 0))}"),
        ("High Risk", f"{int(counts.get('High Risk', 0))}"),
        ("Avg. confidence", f"{avg_conf:.1f}%" if pd.notna(avg_conf) else "—"),
    ]
    for col, (label, value) in zip(stat_cols, stats):
        col.markdown(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    with st.container(border=True):
        categories = sorted(history_df["predicted_risk_category"].dropna().unique())
        selected = st.multiselect("Filter by risk category", categories, default=categories)
        filtered = history_df[history_df["predicted_risk_category"].isin(selected)]
        st.caption(f"Showing {len(filtered)} of {total} record(s).")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    st.stop()  # Reports page ends here; the Dashboard modules below don't render.


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
                    <div style='font-size:0.875rem; opacity:0.75;'>Predicted Risk Category</div>
                    <div style='font-size:1.75rem; font-weight:700; color:{color};'>{risk_category.upper()}</div>
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
                    <div style='font-size:0.8125rem; opacity:0.75;'>Model Confidence Score</div>
                    <div style='font-size:1.75rem; font-weight:700;'>{confidence:.1f}%</div>
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
                <div style='font-weight:700; color:#991b1b; font-size:1rem;'>
                    ⚠️ Refer to a Higher-Level Facility
                </div>
                <ul style='margin:8px 0 2px 18px; color:#7f1d1d; font-size:0.8125rem;'>
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
                    <div class="rec-icon-badge"><span class="rec-anim {REC_ICON_ANIMATION.get(rec['icon'], 'anim-pulse')}">{rec['icon']}</span></div>
                    <div style='font-weight:600; margin-top:4px;'>{rec['title']}</div>
                    <div style='font-size:0.75rem; opacity:0.75; margin-top:4px;'>{rec['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("ℹ️ This information is for educational purposes only and does not replace professional medical advice, diagnosis, or treatment.")
