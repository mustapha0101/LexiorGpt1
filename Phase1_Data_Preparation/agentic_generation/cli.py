# Compatibility wrapper — canonical source: src/lexior/agentic/cli.py
import os as _os, sys as _sys
_src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
import lexior.agentic.cli as _mod  # noqa: E402
from lexior.agentic.cli import *  # noqa: F401,F403,E402
_print_progress = _mod._print_progress
_usage_snapshot = _mod._usage_snapshot
_next_request_type = _mod._next_request_type
