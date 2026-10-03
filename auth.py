"""
auth.py
-------
Login gate for the dashboard. Uses the existing `users` table in db.py
for authentication (PBKDF2-hashed passwords — see db.py; nothing is
ever stored in plain text).

Admin bootstrap credentials are read from Streamlit secrets
(.streamlit/secrets.toml locally, or the "Secrets" section in Streamlit
Community Cloud's app settings) so a real deployment never ships a
hardcoded password in the source code. If no secrets are configured
(e.g. a fresh local clone with no secrets.toml), a clearly-labeled
development-only default is used instead, with an on-screen warning
so nobody mistakes it for a secure production credential.
"""

import streamlit as st

import db
from branding import NAVY as _NAVY, DEEPBLUE as _DEEPBLUE, TEAL as _TEAL, GRADIENT as _GRADIENT

DEV_DEFAULT_USERNAME = "admin"
DEV_DEFAULT_PASSWORD = "changeme123"  # DEV ONLY — overridden by st.secrets in real deployments

_LOGIN_CSS = f"""
<style>
/* Login-screen look: a frosted-glass card (glassmorphism) over a full-page
   navy-to-teal gradient, pill-shaped inputs, and a soft pulsing teal glow.
   This CSS is only injected on the login screen (require_login stops before
   the dashboard renders), so none of it leaks into the dashboard itself. */
.stApp {{
    background:
        radial-gradient(circle at 12% 18%, rgba(34, 211, 238, 0.28) 0%, transparent 38%),
        radial-gradient(circle at 88% 82%, rgba(56, 189, 248, 0.22) 0%, transparent 42%),
        {_GRADIENT};
    background-attachment: fixed;
}}
header[data-testid="stHeader"],
[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
.block-container {{ padding-top: 7vh !important; }}

.login-hero {{ text-align: center; margin-bottom: 22px; color: #ffffff; }}
.login-hero-icon {{ font-size: 46px; margin-bottom: 8px; line-height: 1; }}
.login-hero-title {{
    font-size: 26px; font-weight: 700; letter-spacing: 0.3px;
    text-shadow: 0 2px 14px rgba(0, 0, 0, 0.25);
}}
.login-hero-subtitle {{ font-size: 14px; opacity: 0.85; margin-top: 6px; }}

/* Rhythmic double-beat pulse ("lub-dub"), like a real heartbeat, applied
   to the heart icon instead of a static glyph. */
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

/* Soft breathing glow around the card. */
@keyframes glowPulse {{
    0%, 100% {{
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25),
                    0 0 18px rgba(34, 211, 238, 0.25),
                    inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }}
    50% {{
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25),
                    0 0 38px rgba(34, 211, 238, 0.55),
                    inset 0 0 0 1px rgba(255, 255, 255, 0.14);
    }}
}}

/* The card is a real Streamlit container (st.container(key=...)) styled via
   its `key` -> CSS class, NOT a raw HTML <div> split across several
   st.markdown() calls (that renders as an empty box, since each Streamlit
   call is its own sibling element in the DOM). */
.st-key-login_card {{
    background: rgba(255, 255, 255, 0.12);
    -webkit-backdrop-filter: blur(18px);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 24px;
    padding: 28px 30px 14px 30px;
    animation: glowPulse 3.2s ease-in-out infinite;
}}
.st-key-login_card h3 {{ color: #ffffff !important; text-align: center; margin-bottom: 0; }}
.st-key-login_card [data-testid="stCaptionContainer"] {{ text-align: center; }}
.st-key-login_card [data-testid="stCaptionContainer"] p,
.st-key-login_card small {{ color: rgba(255, 255, 255, 0.8) !important; }}
.st-key-login_card [data-testid="stWidgetLabel"] p {{ color: #ffffff !important; font-weight: 500; }}

/* Drop Streamlit's own form border so it's one card, not a box in a box. */
.st-key-login_card [data-testid="stForm"] {{
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}}

/* Pill-shaped inputs. */
.st-key-login_card div[data-baseweb="input"] {{
    border-radius: 999px !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    overflow: hidden;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.st-key-login_card div[data-baseweb="input"]:focus-within {{
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.25);
}}
.st-key-login_card div[data-baseweb="base-input"] {{ background: transparent !important; }}
.st-key-login_card input {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    padding-left: 18px !important;
}}
.st-key-login_card input::placeholder {{
    color: rgba(255, 255, 255, 0.6) !important;
    -webkit-text-fill-color: rgba(255, 255, 255, 0.6) !important;
}}
/* Browser autofill would otherwise paint a pale background over the glass. */
.st-key-login_card input:-webkit-autofill {{
    -webkit-box-shadow: 0 0 0 1000px rgba(12, 60, 100, 0.95) inset !important;
    -webkit-text-fill-color: #ffffff !important;
}}
/* Show/hide-password eye icon. */
.st-key-login_card div[data-baseweb="input"] button {{ background: transparent !important; }}
.st-key-login_card div[data-baseweb="input"] svg {{
    fill: rgba(255, 255, 255, 0.85) !important;
    color: rgba(255, 255, 255, 0.85) !important;
}}

/* Sign-in button: cyan pill, dark text, glows on hover. */
.st-key-login_card [data-testid="stFormSubmitButton"] button,
.st-key-login_card button[kind="secondaryFormSubmit"],
.st-key-login_card button[kind="primaryFormSubmit"] {{
    background: linear-gradient(90deg, #22d3ee, #38bdf8) !important;
    border: none !important;
    border-radius: 999px !important;
    height: 46px;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.st-key-login_card [data-testid="stFormSubmitButton"] button p {{
    color: #0b1b3a !important;
    font-weight: 700;
}}
.st-key-login_card [data-testid="stFormSubmitButton"] button:hover,
.st-key-login_card button[kind="secondaryFormSubmit"]:hover,
.st-key-login_card button[kind="primaryFormSubmit"]:hover {{
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.6);
    transform: translateY(-1px);
}}

.login-footer {{
    font-size: 11px;
    color: rgba(255, 255, 255, 0.7);
    text-align: center;
    margin-top: 6px;
}}
</style>
"""


def _get_secret(key, default=None):
    """Safely reads a Streamlit secret, falling back to `default` if no
    secrets.toml exists at all (as on a fresh local clone)."""
    try:
        return st.secrets[key]
    except Exception:
        return default


def require_login():
    """Renders a login form and halts the rest of the app until the
    user authenticates. Call this as the first thing in app.py, right
    after db.init_db()."""

    admin_username = _get_secret("ADMIN_USERNAME", DEV_DEFAULT_USERNAME)
    admin_password = _get_secret("ADMIN_PASSWORD", None)
    using_dev_default = admin_password is None
    if using_dev_default:
        admin_password = DEV_DEFAULT_PASSWORD

    db.ensure_default_admin(admin_username, admin_password)

    if st.session_state.get("authenticated"):
        return  # already logged in this session

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="login-hero">
            <div class="login-hero-icon heartbeat-icon">🫀</div>
            <div class="login-hero-title">Explainable ML Analytics Dashboard</div>
            <div class="login-hero-subtitle">WHO/PhilPEN Cardiovascular Risk Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_gutter, center, right_gutter = st.columns([1, 1.3, 1])
    with center:
        with st.container(key="login_card"):
            st.markdown("### 🔒 Sign In")
            st.caption("Please sign in to continue.")

            if using_dev_default:
                st.warning(
                    f"No `ADMIN_PASSWORD` secret configured — using a development-only "
                    f"default login (username: `{admin_username}`, password: `{DEV_DEFAULT_PASSWORD}`). "
                    f"Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` in Streamlit secrets before "
                    f"any real deployment.",
                    icon="⚠️",
                )

            with st.form("login_form"):
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Log In", use_container_width=True)

            if submitted:
                role = db.verify_user(username, password)
                if role:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = role
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.markdown(
                '<div class="login-footer">Patient data is stored without any personally '
                'identifying information.</div>',
                unsafe_allow_html=True,
            )

    st.stop()  # nothing below this point in app.py renders until logged in


def _do_logout():
    """The actual logout action: wipes the ENTIRE session state, not just
    the auth flags. This matters clinically -- without this, a previous
    patient's prediction, SHAP explanation, and recommendations would
    still be visible to the next person who logs into the same browser
    session, which is a real data leakage risk between users."""
    st.session_state.clear()
    st.rerun()


def logout_control():
    """Renders a small 'logged in as ... / Log out' control. Call this
    somewhere visible in the header once the user is authenticated."""
    if st.session_state.get("authenticated"):
        st.caption(f"Logged in as **{st.session_state.get('username')}** ({st.session_state.get('role')})")
        if st.button("Log out", key="logout_button"):
            _do_logout()
