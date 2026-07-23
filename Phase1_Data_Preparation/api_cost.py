# Compatibility wrapper — moved to src/lexior/observability/costs.py
import os as _os, sys as _sys, warnings as _w
_src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
_w.warn("api_cost est déplacé : utiliser lexior.observability.costs",
        DeprecationWarning, stacklevel=2)
from lexior.observability.costs import *  # noqa: F401,F403
