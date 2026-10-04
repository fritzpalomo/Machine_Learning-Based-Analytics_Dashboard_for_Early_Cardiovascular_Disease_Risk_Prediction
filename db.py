"""
db.py
-----
Lightweight SQLite data-access layer. Stores prediction records and
user accounts (for access control) only. NO personally identifiable
information from the DOH dataset is ever stored here -- only what a
clinician enters at prediction time for their own record-keeping (per
RA 10173 compliance, even that should be handled per your institution's
data policy).

Passwords are never stored in plain text: each password is hashed with
PBKDF2-HMAC-SHA256 and a unique random salt per user (both from Python's
standard library, so no extra dependency is needed).
"""

import json
import sqlite3
import hashlib
import hmac
import os
from datetime import datetime
from contextlib import contextmanager

# Resolved relative to THIS file's own location, not the process's current
# working directory. A bare relative path ("cvd_dashboard.db") would instead
# resolve against wherever `streamlit run app.py` happened to be launched
# from -- if that ever differs from this project folder (a different
# terminal cwd, an IDE's default run folder, a second project copy), Python
# silently creates/writes to a SEPARATE database file there, with no error.
# Predictions still "work" on screen; they just stop appearing in the
# database you're looking at. Anchoring to __file__ makes this path stable
# no matter where the app is launched from.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvd_dashboard.db")

PBKDF2_ITERATIONS = 200_000


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL DEFAULT 'clinician',
                password_hash TEXT,
                salt TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                systolic_bp REAL,
                diastolic_bp REAL,
                total_cholesterol REAL,
                fasting_glucose REAL,
                bmi REAL,
                smoking_status TEXT,
                alcohol_consumption TEXT,
                physical_activity TEXT,
                family_history_cvd TEXT,
                predicted_risk_category TEXT,
                confidence_score REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                username TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        _migrate_add_auth_columns(conn)


def _migrate_add_auth_columns(conn):
    """Adds password_hash/salt columns to a users table created before
    authentication existed, so existing databases don't break."""
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "password_hash" not in existing_cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "salt" not in existing_cols:
        conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")


def _hash_password(password: str, salt: str = None) -> tuple:
    """Returns (hash_hex, salt_hex). Generates a new random salt if none given."""
    if salt is None:
        salt_bytes = os.urandom(16)
    else:
        salt_bytes = bytes.fromhex(salt)
    hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS)
    return hash_bytes.hex(), salt_bytes.hex()


def create_user(username: str, password: str, role: str = "clinician") -> bool:
    """Creates a new user with a securely hashed password. Returns False if
    the username already exists."""
    password_hash, salt = _hash_password(password)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, role, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, role, password_hash, salt, datetime.now().isoformat()),
            )
        return True
    except sqlite3.IntegrityError:
        return False  # username already exists


def verify_user(username: str, password: str):
    """Returns the user's role (str) if the username/password are correct,
    otherwise None. Uses a timing-safe comparison to avoid leaking timing
    information about how much of the hash matched."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT role, password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None or not row["password_hash"] or not row["salt"]:
        return None
    computed_hash, _ = _hash_password(password, salt=row["salt"])
    if hmac.compare_digest(computed_hash, row["password_hash"]):
        return row["role"]
    return None


def any_users_exist() -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
    return row["n"] > 0


def ensure_default_admin(username: str, password: str):
    """Bootstraps a single admin account only if the users table is
    completely empty (first run). Never overwrites existing accounts."""
    if not any_users_exist():
        create_user(username, password, role="admin")


def _to_native(value):
    """Coerces numpy scalar types (e.g. numpy.float32 from XGBoost's
    predict_proba, as opposed to numpy.float64 which happens to subclass
    Python's own float) to plain Python int/float/str/None before binding
    to sqlite3.

    sqlite3 only recognizes int, float, str, bytes, and None natively.
    numpy.float64 is a subclass of float, so it has always worked by
    accident -- but numpy.float32 is not, and sqlite3 silently falls back
    to treating it as a buffer/BLOB instead of raising an error. That
    corrupted confidence_score as soon as the deployed model (XGBoost)
    started returning float32 probabilities: values were saved as raw
    4-byte blobs (e.g. b'\\n\\xd7^B') instead of numbers. See DF-05.
    """
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar (float32, float64, int64, ...)
        return value.item()
    return value


def insert_prediction(record: dict):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, age, sex, systolic_bp, diastolic_bp,
                total_cholesterol, fasting_glucose, bmi,
                smoking_status, alcohol_consumption, physical_activity,
                family_history_cvd, predicted_risk_category, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            _to_native(record.get("age")),
            record.get("sex"),
            _to_native(record.get("systolic_bp")),
            _to_native(record.get("diastolic_bp")),
            _to_native(record.get("total_cholesterol")),
            _to_native(record.get("fasting_glucose")),
            _to_native(record.get("bmi")),
            record.get("smoking_status"),
            record.get("alcohol_consumption"),
            record.get("physical_activity"),
            record.get("family_history_cvd"),
            record.get("predicted_risk_category"),
            _to_native(record.get("confidence_score")),
        ))


def get_recent_predictions(limit: int = 20):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_user_settings(username: str) -> dict:
    """Returns the saved display settings (text size, density, ...) for a
    user, or {} if none are saved yet / the stored JSON is unreadable. The
    caller is expected to validate the values (see settings.sanitize)."""
    if not username:
        return {}
    with get_connection() as conn:
        row = conn.execute(
            "SELECT settings_json FROM user_settings WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return {}
    try:
        data = json.loads(row["settings_json"])
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_user_settings(username: str, settings: dict):
    """Saves (inserts or replaces) a user's display settings."""
    if not username:
        return
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_settings (username, settings_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                settings_json = excluded.settings_json,
                updated_at = excluded.updated_at
            """,
            (username, json.dumps(settings), datetime.now().isoformat()),
        )
