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
.login-hero {{
    text-align: center;
    padding: 40px 20px 28px 20px;
    background: {_GRADIENT};
    border-radius: 16px;
    margin-bottom: 26px;
    color: #ffffff;
}}
.login-hero-icon {{ font-size: 42px; margin-bottom: 8px; line-height: 1; }}
.login-hero-title {{ font-size: 24px; font-weight: 700; letter-spacing: 0.2px; }}
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

/* The login card uses a real Streamlit container (st.container(key=...))
   styled via its `key` -> CSS class, NOT a raw HTML <div> split across
   multiple st.markdown() calls. Each st.markdown/st.form/st.warning call
   renders as its own independent sibling in the DOM -- an opening <div>
   in one call and a closing </div> in a later call do NOT actually wrap
   the Streamlit widgets rendered in between; the browser just shows an
   empty box with the real content left sitting outside it. Targeting a
   real container's key avoids that bug entirely. */
.st-key-login_card {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 24px 28px 10px 28px;
    box-shadow: 0 6px 20px rgba(33, 41, 92, 0.10);
}}
.st-key-login_card h3 {{ color: {_NAVY}; margin-bottom: 0; }}
.login-footer {{ font-size: 11px; color: #9ca3af; text-align: center; margin-top: 4px; }}

div[data-testid="stForm"] button {{
    background-color: {_NAVY};
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: background-color 0.15s ease-in-out;
}}
div[data-testid="stForm"] button:hover {{
    background-color: {_DEEPBLUE};
    color: #ffffff !important;
}}
div[data-testid="stForm"] input {{
    border-radius: 8px;
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
