# Compatibility wrapper — canonical source: src/lexior/agentic/acceptance.py
import os as _os, sys as _sys
_src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
import lexior.agentic.acceptance as _mod  # noqa: E402
from lexior.agentic.acceptance import *  # noqa: F401,F403,E402
_check_article_grounding = _mod._check_article_grounding
_check_clarification_consistency = _mod._check_clarification_consistency
_check_jurisdiction_consistency = _mod._check_jurisdiction_consistency
_infer_jurisdiction_from_query = _mod._infer_jurisdiction_from_query
_critic_failure_reasons = _mod._critic_failure_reasons
