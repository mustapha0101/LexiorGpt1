# Compatibility wrapper — canonical source: src/lexior/agentic/error_codes.py
import os as _os, sys as _sys
_src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
import lexior.agentic.error_codes as _mod  # noqa: E402
from lexior.agentic.error_codes import *  # noqa: F401,F403,E402

BLOCKING_CODES = _mod.BLOCKING_CODES
ErrorCode = _mod.ErrorCode
extract_code = _mod.extract_code
strip_code = _mod.strip_code
tag = _mod.tag
