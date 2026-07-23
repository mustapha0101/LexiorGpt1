# Compatibility wrapper — canonical source: src/lexior/agentic/__init__.py
import os as _os, sys as _sys
_src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
from lexior.agentic import SCHEMA_VERSION, DATASET_TYPE  # noqa: F401
