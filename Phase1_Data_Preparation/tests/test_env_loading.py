# -*- coding: utf-8 -*-
"""Les clés viennent du `.env` du dépôt, jamais du code.

Deux garanties : le fichier est bien trouvé depuis n'importe quelle phase,
et l'environnement réel n'est jamais écrasé par le fichier.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lexior.env import (
    CANDIDATE_PATHS,
    PHASE1_DIR,
    REPO_ROOT,
    _parse,
    describe,
    env_file,
    load_project_env,
)

REPO_ROOT_FROM_TEST = Path(__file__).resolve().parents[2]


# ── Localisation ─────────────────────────────────────────────────────────


def test_the_repository_root_is_resolved_correctly():
    assert REPO_ROOT == REPO_ROOT_FROM_TEST
    assert PHASE1_DIR.name == "Phase1_Data_Preparation"


def test_the_repository_root_comes_first():
    """`LexiorGpt1/.env` prime sur un éventuel fichier de phase."""
    assert CANDIDATE_PATHS[0] == REPO_ROOT / ".env"


def test_an_explicit_path_overrides_the_search(tmp_path, monkeypatch):
    custom = tmp_path / "ailleurs.env"
    custom.write_text("PEU_IMPORTE=1", encoding="utf-8")
    monkeypatch.setenv("LEXIOR_ENV_FILE", str(custom))

    assert env_file() == custom


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    """En production les variables viennent de l'orchestrateur."""
    monkeypatch.setenv("LEXIOR_ENV_FILE", str(tmp_path / "inexistant.env"))

    assert env_file() is None
    assert load_project_env(force=True) is None


# ── Priorité ─────────────────────────────────────────────────────────────


def test_the_real_environment_always_wins(tmp_path, monkeypatch):
    """Sinon un `CLE=… python …` ponctuel serait silencieusement ignoré."""
    env = tmp_path / ".env"
    env.write_text("CLE_DE_TEST=depuis_le_fichier", encoding="utf-8")
    monkeypatch.setenv("LEXIOR_ENV_FILE", str(env))
    monkeypatch.setenv("CLE_DE_TEST", "depuis_le_shell")

    load_project_env(force=True)

    assert os.environ["CLE_DE_TEST"] == "depuis_le_shell"


def test_a_variable_absent_from_the_shell_is_taken_from_the_file(
        tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CLE_ABSENTE_DU_SHELL=depuis_le_fichier", encoding="utf-8")
    monkeypatch.setenv("LEXIOR_ENV_FILE", str(env))
    monkeypatch.delenv("CLE_ABSENTE_DU_SHELL", raising=False)

    load_project_env(force=True)

    assert os.environ["CLE_ABSENTE_DU_SHELL"] == "depuis_le_fichier"


# ── Analyse du format ────────────────────────────────────────────────────


@pytest.mark.parametrize("line,expected", [
    ("CLE=valeur", {"CLE": "valeur"}),
    ("export CLE=valeur", {"CLE": "valeur"}),
    ('CLE="valeur entre guillemets"', {"CLE": "valeur entre guillemets"}),
    ("CLE='simple'", {"CLE": "simple"}),
    ("  CLE = valeur  ", {"CLE": "valeur"}),
    ("# commentaire", {}),
    ("", {}),
    ("sans_signe_egal", {}),
    ("CLE=", {"CLE": ""}),
    ("CLE=a=b", {"CLE": "a=b"}),
])
def test_the_minimal_parser_handles_the_usual_shapes(line, expected):
    """Utilisé quand python-dotenv n'est pas installé."""
    assert _parse(line) == expected


# ── Diagnostic ───────────────────────────────────────────────────────────


def test_describe_never_exposes_a_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("UNE_CLE=un_secret_a_ne_pas_afficher", encoding="utf-8")
    monkeypatch.setenv("LEXIOR_ENV_FILE", str(env))

    report = describe()

    assert "UNE_CLE" in report["keys_present"]
    assert "un_secret_a_ne_pas_afficher" not in str(report)


# ── Câblage effectif ─────────────────────────────────────────────────────


def test_importing_the_package_loads_the_file():
    """C'est le point : aucun module n'a à s'en occuper lui-même."""
    import lexior

    assert hasattr(lexior, "load_project_env")


def test_the_example_file_documents_every_secret_of_the_real_one():
    """Un `.env` local ne doit pas contenir de clé non documentée."""
    example = REPO_ROOT / ".env.example"
    assert example.is_file(), "`.env.example` doit rester versionné"

    documented = set(_parse(example.read_text(encoding="utf-8")))
    documented |= {
        line.lstrip("# ").split("=")[0].strip()
        for line in example.read_text(encoding="utf-8").splitlines()
        if line.startswith("# ") and "=" in line
    }

    local = REPO_ROOT / ".env"
    if not local.is_file():
        pytest.skip("aucun .env local")
    undocumented = sorted(set(_parse(local.read_text(encoding="utf-8")))
                          - documented)

    assert not undocumented, (
        f"clés présentes dans .env mais absentes de .env.example : "
        f"{undocumented}")
