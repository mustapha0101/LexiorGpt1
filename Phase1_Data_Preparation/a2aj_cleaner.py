# Compatibility wrapper — moved to lexior/data/cleaning/a2aj.py
import warnings as _w
_w.warn("a2aj_cleaner est déplacé : utiliser lexior.data.cleaning.a2aj",
        DeprecationWarning, stacklevel=2)
from lexior.data.cleaning.a2aj import *  # noqa: F401,F403
