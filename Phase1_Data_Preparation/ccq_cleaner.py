# Compatibility wrapper — moved to lexior/data/cleaning/ccq.py
import warnings as _w
_w.warn("ccq_cleaner est déplacé : utiliser lexior.data.cleaning.ccq",
        DeprecationWarning, stacklevel=2)
from lexior.data.cleaning.ccq import *  # noqa: F401,F403
