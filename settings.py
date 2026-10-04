"""
settings.py
-----------
Per-user display settings (text size, density, reduced motion, high
contrast). Pure functions only -- no Streamlit calls -- so the validation
and CSS generation can be unit-tested. app.py stores the values in the
database (db.get_user_settings / save_user_settings) and injects the CSS
returned by build_css().
"""

TEXT_SIZES = {"Small": 90, "Default": 100, "Large": 115, "Extra large": 130}  # % of base font size
DENSITIES = ("Comfortable", "Compact")

DEFAULTS = {
    "text_size": "Default",
    "density": "Comfortable",
    "reduce_motion": False,
    "high_contrast": False,
}


def sanitize(raw) -> dict:
    """Returns a complete, valid settings dict. Unknown keys are dropped and
    any missing or invalid value falls back to its default, so stale or
    tampered stored data can never break the page."""
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULTS)
    if raw.get("text_size") in TEXT_SIZES:
        out["text_size"] = raw["text_size"]
    if raw.get("density") in DENSITIES:
        out["density"] = raw["density"]
    if isinstance(raw.get("reduce_motion"), bool):
        out["reduce_motion"] = raw["reduce_motion"]
    if isinstance(raw.get("high_contrast"), bool):
        out["high_contrast"] = raw["high_contrast"]
    return out


def build_css(settings: dict) -> str:
    """CSS <style> block implementing the given settings. Colors are left
    theme-neutral so this works in both Streamlit light and dark themes."""
    s = sanitize(settings)
    parts = []

    pct = TEXT_SIZES[s["text_size"]]
    if pct != 100:
        # Streamlit sizes text in rem, so scaling the root scales everything.
        parts.append(f"html, :root {{ font-size: {pct}% !important; }}")

    if s["density"] == "Compact":
        parts.append(
            """
            section.main .block-container, [data-testid="stMain"] .block-container {
                padding-top: 1rem !important; padding-bottom: 1rem !important;
            }
            section.main [data-testid="stVerticalBlock"], [data-testid="stMain"] [data-testid="stVerticalBlock"] {
                gap: 0.55rem !important;
            }
            .rec-tile { padding: 10px !important; min-height: 130px !important; }
            """
        )

    if s["reduce_motion"]:
        parts.append(
            """
            .heartbeat-icon, .rec-anim, .rec-anim * { animation: none !important; }
            """
        )

    if s["high_contrast"]:
        parts.append(
            """
            div[data-testid="stVerticalBlockBorderWrapper"] { border: 2px solid rgba(128,128,128,0.95) !important; }
            .rec-tile, .stat-card { border: 2px solid rgba(128,128,128,0.95) !important; }
            [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { opacity: 1 !important; }
            [style*="opacity:0.75"], [style*="opacity: 0.75"], [style*="opacity:0.85"], [style*="opacity: 0.85"] { opacity: 1 !important; }
            .stat-label { opacity: 1 !important; }
            """
        )

    if not parts:
        return ""
    return "<style>" + "\n".join(parts) + "</style>"
