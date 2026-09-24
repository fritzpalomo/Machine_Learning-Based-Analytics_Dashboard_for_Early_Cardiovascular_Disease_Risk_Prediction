"""
test_auth.py
------------
Unit tests for the authentication layer added to db.py: password
hashing, user creation, login verification, and admin bootstrapping.
Addresses the "appropriate access control for authorized users"
requirement — see DF-0x in Testing_Documentation.docx for the gap this
closed (the dashboard previously had no login at all).
"""

import os
import tempfile

import pytest


@pytest.fixture
def db_module(tmp_path, monkeypatch):
    """Reload db.py pointed at a throwaway temp database file."""
    import db as db_mod
    test_db_path = str(tmp_path / "test_auth.db")
    monkeypatch.setattr(db_mod, "DB_PATH", test_db_path)
    db_mod.init_db()
    yield db_mod
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_no_users_exist_on_fresh_database(db_module):
    assert db_module.any_users_exist() is False


def test_ensure_default_admin_bootstraps_once(db_module):
    db_module.ensure_default_admin("admin", "changeme123")
    assert db_module.any_users_exist() is True
    assert db_module.verify_user("admin", "changeme123") == "admin"


def test_ensure_default_admin_does_not_overwrite_existing_user(db_module):
    db_module.ensure_default_admin("admin", "firstpassword")
    db_module.ensure_default_admin("admin", "secondpassword")  # should be a no-op
    assert db_module.verify_user("admin", "firstpassword") == "admin"
    assert db_module.verify_user("admin", "secondpassword") is None


def test_correct_password_verifies(db_module):
    db_module.create_user("nurse1", "nursepass456", role="clinician")
    assert db_module.verify_user("nurse1", "nursepass456") == "clinician"


def test_wrong_password_is_rejected(db_module):
    db_module.create_user("nurse1", "nursepass456", role="clinician")
    assert db_module.verify_user("nurse1", "wrongpassword") is None


def test_nonexistent_user_is_rejected(db_module):
    assert db_module.verify_user("ghost", "anything") is None


def test_duplicate_username_is_rejected(db_module):
    first = db_module.create_user("nurse1", "pass12345", role="clinician")
    second = db_module.create_user("nurse1", "differentpass", role="clinician")
    assert first is True
    assert second is False


def test_password_is_never_stored_in_plaintext(db_module):
    db_module.create_user("nurse1", "supersecretpassword", role="clinician")
    with db_module.get_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", ("nurse1",)
        ).fetchone()
    assert row["password_hash"] != "supersecretpassword"
    assert len(row["password_hash"]) == 64  # sha256 hex digest length


def test_username_matching_is_case_sensitive(db_module):
    db_module.create_user("Nurse1", "pass12345", role="clinician")
    assert db_module.verify_user("nurse1", "pass12345") is None


def test_each_user_gets_a_unique_salt(db_module):
    db_module.create_user("nurse1", "samepassword123", role="clinician")
    db_module.create_user("nurse2", "samepassword123", role="clinician")
    with db_module.get_connection() as conn:
        rows = conn.execute("SELECT username, salt, password_hash FROM users").fetchall()
    salts = {r["salt"] for r in rows}
    hashes = {r["password_hash"] for r in rows}
    assert len(salts) == 2, "Each user should get an independently random salt"
    assert len(hashes) == 2, "Same password should still produce different hashes due to unique salts"
