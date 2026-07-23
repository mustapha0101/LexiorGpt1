# Compatibility wrapper — moved to lexior/observability/costs.py
import warnings as _w
_w.warn("api_cost est déplacé : utiliser lexior.observability.costs",
        DeprecationWarning, stacklevel=2)
from lexior.observability.costs import *  # noqa: F401,F403
