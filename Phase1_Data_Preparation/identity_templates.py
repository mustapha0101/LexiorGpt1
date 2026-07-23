# Compatibility wrapper — moved to lexior/data/identity/templates.py
import warnings as _w
_w.warn("identity_templates est déplacé : utiliser lexior.data.identity.templates",
        DeprecationWarning, stacklevel=2)
from lexior.data.identity.templates import *  # noqa: F401,F403
