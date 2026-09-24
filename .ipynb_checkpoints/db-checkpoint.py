"""
db.py
-----
Lightweight SQLite data-access layer. Stores prediction records and
basic user accounts only. NO personally identifiable information from
the DOH dataset is ever stored here -- only what a clinician enters at
prediction time for their own record-keeping (per RA 10173 compliance,
even that should be handled per your institution's data policy).
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "cvd_dashboard.db"


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
            record.get("age"),
            record.get("sex"),
            record.get("systolic_bp"),
            record.get("diastolic_bp"),
            record.get("total_cholesterol"),
            record.get("fasting_glucose"),
            record.get("bmi"),
            record.get("smoking_status"),
            record.get("alcohol_consumption"),
            record.get("physical_activity"),
            record.get("family_history_cvd"),
            record.get("predicted_risk_category"),
            record.get("confidence_score"),
        ))


def get_recent_predictions(limit: int = 20):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]
