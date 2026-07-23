# Compatibility wrapper — moved to lexior/data/identity/policy.py
import warnings as _w
_w.warn("identity_policy est déplacé : utiliser lexior.data.identity.policy",
        DeprecationWarning, stacklevel=2)
from lexior.data.identity.policy import *  # noqa: F401,F403
