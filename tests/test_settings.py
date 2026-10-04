"""
tests/test_settings.py
----------------------
Unit tests for per-user display settings: validation and CSS generation
(settings.py) and persistence (db.get_user_settings / save_user_settings).
"""

import os

import pytest

import settings as ui_settings


@pytest.fixture
def db_module(tmp_path, monkeypatch):
    import db as db_mod
    path = str(tmp_path / "test_settings.db")
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()
    yield db_mod
    if os.path.exists(path):
        os.remove(path)


def test_sanitize_defaults_for_garbage():
    assert ui_settings.sanitize(None) == ui_settings.DEFAULTS
    assert ui_settings.sanitize("nope") == ui_settings.DEFAULTS
    assert ui_settings.sanitize({"text_size": "Huge", "density": 3, "reduce_motion": "yes"}) == ui_settings.DEFAULTS


def test_sanitize_keeps_valid_values_and_drops_unknown_keys():
    out = ui_settings.sanitize({"text_size": "Large", "density": "Compact",
                                "reduce_motion": True, "high_contrast": True, "evil": "<script>"})
    assert out == {"text_size": "Large", "density": "Compact", "reduce_motion": True, "high_contrast": True}


def test_default_settings_produce_no_css():
    assert ui_settings.build_css(ui_settings.DEFAULTS) == ""


def test_text_size_css():
    css = ui_settings.build_css({"text_size": "Extra large"})
    assert "font-size: 130%" in css
    assert "font-size: 90%" in ui_settings.build_css({"text_size": "Small"})


def test_density_motion_contrast_css():
    assert "gap: 0.55rem" in ui_settings.build_css({"density": "Compact"})
    assert "animation: none" in ui_settings.build_css({"reduce_motion": True})
    assert "2px solid" in ui_settings.build_css({"high_contrast": True})


def test_settings_roundtrip_in_db(db_module):
    assert db_module.get_user_settings("alice") == {}
    db_module.save_user_settings("alice", {"text_size": "Large"})
    assert db_module.get_user_settings("alice") == {"text_size": "Large"}
    db_module.save_user_settings("alice", {"text_size": "Small", "density": "Compact"})
    assert db_module.get_user_settings("alice") == {"text_size": "Small", "density": "Compact"}
    assert db_module.get_user_settings("bob") == {}


def test_settings_for_blank_username_are_ignored(db_module):
    db_module.save_user_settings("", {"text_size": "Large"})
    assert db_module.get_user_settings("") == {}


def test_corrupt_stored_json_falls_back_to_empty(db_module):
    with db_module.get_connection() as conn:
        conn.execute("INSERT INTO user_settings VALUES ('carol', 'not json', 'x')")
    assert db_module.get_user_settings("carol") == {}
