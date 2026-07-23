# Compatibility wrapper — canonical source: src/lexior/agentic/prompts.py
import os as _os, sys as _sys
_src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
from lexior.agentic.prompts import *  # noqa: F401,F403
