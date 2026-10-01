"""
branding.py
-----------
Single source of truth for the color palette used across the login
screen and the main dashboard (and reused in the thesis documents and
slide deck as the "Ocean Gradient" palette), so a color change only
has to happen in one place instead of drifting between files.
"""

NAVY = "#21295C"
DEEPBLUE = "#065A82"
TEAL = "#1C7293"

GRADIENT = f"linear-gradient(135deg, {NAVY} 0%, {DEEPBLUE} 60%, {TEAL} 100%)"
