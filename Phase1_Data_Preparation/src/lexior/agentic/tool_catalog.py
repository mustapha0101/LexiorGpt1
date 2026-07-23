# -*- coding: utf-8 -*-

"""
Catalogue d'outils MCP — source de vérité : docs/mcp_tools_catalog.json.

Rôles :
  1. charger le catalogue et calculer son haché (traçabilité du dataset) ;
  2. valider chaque appel d'outil AVANT exécution : nom canonique connu,
     arguments conformes à l'inputSchema (types, required, enums, inconnus) ;
  3. comparer le catalogue aux outils réellement exposés par les serveurs MCP
     et échouer proprement si un schéma a changé.

Toujours les canonicalName — jamais les noms préfixés VS Code
(mcp_lexior-legisq_..., mcp_canadian_lega_...).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

# Préfixes des noms observés par VS Code, interdits dans le dataset.
_VSCODE_PREFIX_RE = re.compile(r"^mcp[_-]", re.IGNORECASE)


class CatalogError(Exception):
    """Catalogue illisible, incohérent, ou divergent des serveurs réels."""


class ToolSpec:
    def __init__(self, entry: dict[str, Any]):
        self.canonical_name: str = entry["canonicalName"]
        self.server: str = entry.get("server", "")
        self.vscode_name: str = entry.get("vscodeObservedName", "")
        self.description: str = entry.get("description", "")
        self.input_schema: dict[str, Any] = entry.get("inputSchema", {}) or {}
        self.response_type: str = (entry.get("responseStructure") or {}).get("type", "text")
        self.raw_entry = entry

    @property
    def properties(self) -> dict[str, Any]:
        return self.input_schema.get("properties", {}) or {}

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []) or [])


def _type_ok(value: Any, expected: Any) -> bool:
    """Vérifie une valeur contre un type JSON Schema (ou une union)."""
    if isinstance(expected, list):
        return any(_type_ok(value, t) for t in expected)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True  # type non spécifié : on ne peut pas trancher


def _schema_types(schema: dict[str, Any]) -> Any:
    """Retourne la représentation canonique du type JSON Schema.

    MCP/Pydantic encode souvent `string | null` via `anyOf`, tandis que des
    catalogues plus anciens utilisaient une liste dans `type`.
    """
    if "type" in schema:
        return schema.get("type")
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        types = [part.get("type") for part in any_of
                 if isinstance(part, dict) and part.get("type") is not None]
        return types or None
    return None


class ToolCatalog:
    def __init__(self, data: dict[str, Any], path: str = ""):
        self.path = path
        self.raw = data
        self.servers = {s.get("name", ""): s for s in data.get("servers", [])}
        self.tools: dict[str, ToolSpec] = {}
        self._vscode_names: dict[str, str] = {}
        for entry in data.get("tools", []):
            spec = ToolSpec(entry)
            self.tools[spec.canonical_name] = spec
            if spec.vscode_name:
                self._vscode_names[spec.vscode_name] = spec.canonical_name
        if not self.tools:
            raise CatalogError(f"Catalogue vide : {path}")
        self.catalog_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        # Haché des seuls éléments qui définissent le contrat (nom + schéma),
        # stable face aux champs cosmétiques (generatedAt, notes...).
        contract = {
            name: {"server": t.server, "inputSchema": t.input_schema}
            for name, t in sorted(self.tools.items())
        }
        blob = json.dumps(contract, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Validation d'un appel
    # ------------------------------------------------------------------

    def resolve_name(self, name: str) -> Optional[str]:
        """Renvoie le canonicalName, ou None si le nom est inconnu."""
        if name in self.tools:
            return name
        return None

    def validate_call(self, name: str, arguments: dict[str, Any]) -> list[str]:
        """Renvoie la liste des erreurs (vide = appel valide).

        Ne modifie jamais l'appel : un appel invalide est rejeté, pas corrigé.
        """
        errors: list[str] = []
        if name not in self.tools:
            if name in self._vscode_names:
                errors.append(
                    f"nom VS Code « {name} » : utiliser le canonicalName "
                    f"« {self._vscode_names[name]} »")
            elif _VSCODE_PREFIX_RE.match(name or ""):
                errors.append(f"nom préfixé VS Code interdit : « {name} »")
            else:
                errors.append(f"outil inconnu : « {name} »")
            return errors

        spec = self.tools[name]
        if not isinstance(arguments, dict):
            return [f"{name} : arguments doit être un objet JSON"]

        for req in spec.required:
            if req not in arguments:
                errors.append(f"{name} : argument obligatoire absent : « {req} »")

        for key, value in arguments.items():
            if key not in spec.properties:
                errors.append(f"{name} : argument inconnu : « {key} »")
                continue
            prop = spec.properties[key]
            expected = _schema_types(prop)
            if expected and not _type_ok(value, expected):
                errors.append(
                    f"{name} : type incorrect pour « {key} » "
                    f"(attendu {expected}, reçu {type(value).__name__})")
                continue
            enum = prop.get("enum")
            if enum is not None and value is not None and value not in enum:
                errors.append(
                    f"{name} : valeur hors enum pour « {key} » : {value!r} "
                    f"(valides : {enum})")
        return errors

    def server_of(self, name: str) -> str:
        spec = self.tools.get(name)
        return spec.server if spec else ""

    def is_local(self, name: str) -> bool:
        spec = self.tools.get(name)
        server = self.servers.get(spec.server, {}) if spec else {}
        return str(server.get("type", "")).casefold() == "local"

    # ------------------------------------------------------------------
    # Comparaison avec les serveurs réels
    # ------------------------------------------------------------------

    def compare_with_live(self, live_tools: dict[str, dict[str, Any]]) -> list[str]:
        """Compare le catalogue aux outils réellement listés par les serveurs.

        `live_tools` : {nom_outil: inputSchema} observés via MCP list_tools.
        Renvoie la liste des divergences ; vide = catalogue à jour.
        On échoue proprement sur divergence : jamais de génération avec un
        schéma périmé.
        """
        problems: list[str] = []
        for name, spec in self.tools.items():
            if self.is_local(name):
                continue
            if name not in live_tools:
                problems.append(f"outil du catalogue absent du serveur : {name}")
                continue
            live_schema = live_tools[name] or {}
            live_props = set((live_schema.get("properties") or {}).keys())
            cat_props = set(spec.properties.keys())
            missing = cat_props - live_props
            if missing:
                problems.append(
                    f"{name} : propriétés du catalogue absentes du serveur : "
                    f"{sorted(missing)}")
            added = live_props - cat_props
            if added:
                problems.append(
                    f"{name} : nouvelles propriétés serveur absentes du catalogue : "
                    f"{sorted(added)}")
            live_req = set(live_schema.get("required") or [])
            cat_req = set(spec.required)
            if live_req != cat_req:
                problems.append(
                    f"{name} : required divergent (catalogue {sorted(cat_req)}, "
                    f"serveur {sorted(live_req)})")
            for key in cat_props & live_props:
                cat_prop = spec.properties[key]
                live_prop = (live_schema.get("properties") or {}).get(key, {})
                cat_type = _schema_types(cat_prop)
                live_type = _schema_types(live_prop)
                if cat_type != live_type:
                    problems.append(
                        f"{name}.{key} : type divergent (catalogue {cat_type}, "
                        f"serveur {live_type})")
                cat_enum = cat_prop.get("enum")
                live_enum = live_prop.get("enum")
                if (cat_enum is not None or live_enum is not None) and cat_enum != live_enum:
                    problems.append(
                        f"{name}.{key} : enum divergent (catalogue {cat_enum}, "
                        f"serveur {live_enum})")
        return problems


def load_catalog(path: str) -> ToolCatalog:
    if not os.path.exists(path):
        raise CatalogError(f"Catalogue introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except ValueError as e:
            raise CatalogError(f"Catalogue illisible ({path}) : {e}") from e
    return ToolCatalog(data, path=path)
