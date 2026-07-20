import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PHASE1 = REPO / "Phase1_Data_Preparation"
PHASE2 = REPO / "Phase2_FineTuning"
for path in (str(PHASE1), str(PHASE2)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentic_generation.tool_catalog import load_catalog


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(str(REPO / "docs" / "mcp_tools_catalog.json"))
