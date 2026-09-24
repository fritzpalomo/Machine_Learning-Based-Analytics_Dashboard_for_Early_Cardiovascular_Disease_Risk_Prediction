"""
conftest.py
-----------
Allows the test suite to run in environments where plotly (a UI-only
dependency) isn't installed, by providing a minimal stand-in. If the
real plotly package is present, this has no effect at all.
"""

import sys
import types

try:
    import plotly.graph_objects  # noqa: F401
except ImportError:
    fake_go = types.ModuleType("plotly.graph_objects")

    class _FakeFigure:
        def __init__(self, *args, **kwargs):
            pass

        def update_layout(self, *args, **kwargs):
            return self

    fake_go.Figure = _FakeFigure
    fake_go.Bar = lambda *a, **k: None

    fake_plotly = types.ModuleType("plotly")
    fake_plotly.graph_objects = fake_go
    sys.modules["plotly"] = fake_plotly
    sys.modules["plotly.graph_objects"] = fake_go
