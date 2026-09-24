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

DEV_DEFAULT_USERNAME = "admin"
DEV_DEFAULT_PASSWORD = "changeme123"  # DEV ONLY — overridden by st.secrets in real deployments


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

    st.markdown("## 🔒 Explainable ML Analytics Dashboard")
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
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        role = db.verify_user(username, password)
        if role:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()  # nothing below this point in app.py renders until logged in


def logout_control():
    """Renders a small 'logged in as ... / Log out' control. Call this
    somewhere visible in the header once the user is authenticated."""
    if st.session_state.get("authenticated"):
        st.caption(f"Logged in as **{st.session_state.get('username')}** ({st.session_state.get('role')})")
        if st.button("Log out", key="logout_button"):
            for key in ("authenticated", "username", "role"):
                st.session_state.pop(key, None)
            st.rerun()
