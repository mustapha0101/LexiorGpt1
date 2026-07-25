# -*- coding: utf-8 -*-
"""Préparation auditée et publication explicite du dataset agentique.

Trois garanties que cette étape doit tenir :

1. deux trajectoires partageant une source appartiennent au MÊME groupe —
   composantes connexes, pas cascade de clés (``leakage_groups``) ;
2. le chevauchement entre splits est MESURÉ, jamais affirmé
   (``measure_overlap``) ;
3. les proportions obtenues sont reportées telles qu'obtenues, pas telles
   que visées (``assign_splits``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import TrainingTrajectory
from .storage import iter_jsonl
from .validators import compute_metrics, validate_trajectory

# Dimensions de regroupement acceptées dans ``split.group_by``. Une clé
# inconnue est une erreur : le manifeste doit décrire ce qui s'applique
# vraiment.
GROUP_DIMENSIONS = ("citations", "primary_article", "source_id",
                    "scenario_family_id")
DEFAULT_GROUP_BY = GROUP_DIMENSIONS


class PublicationError(ValueError):
    """Audit de publication échoué."""


# ── Signatures et composantes connexes ───────────────────────────────────


def row_signatures(row: TrainingTrajectory,
                   group_by: Iterable[str] = DEFAULT_GROUP_BY) -> set[str]:
    """Toutes les empreintes partageables d'une trajectoire.

    Contrairement à l'ancienne cascade « première clé non vide », une
    trajectoire porte ici TOUTES ses empreintes : deux trajectoires citant
    1726 se retrouvent dans le même groupe même si l'une cite aussi 1728.
    """
    dimensions = list(group_by)
    unknown = [name for name in dimensions if name not in GROUP_DIMENSIONS]
    if unknown:
        raise PublicationError(
            f"dimensions de regroupement inconnues : {sorted(unknown)}; "
            f"attendu parmi {sorted(GROUP_DIMENSIONS)}")

    signatures: set[str] = set()
    if "citations" in dimensions:
        signatures.update(
            f"citation:{citation}"
            for grounding in row.grounding for citation in grounding.citations)
    if "primary_article" in dimensions:
        signatures.update(
            f"article:{observation.arguments.get('start_article')}"
            for observation in row.tool_trace
            if observation.arguments.get("start_article") is not None)
    if "source_id" in dimensions:
        signatures.update(
            f"url:{url}"
            for grounding in row.grounding for url in grounding.source_urls)
    if "scenario_family_id" in dimensions and row.scenario_family_id:
        signatures.add(f"family:{row.scenario_family_id}")
    return signatures


def leakage_groups(rows: list[TrainingTrajectory],
                   group_by: Iterable[str] = DEFAULT_GROUP_BY,
                   ) -> list[list[int]]:
    """Composantes connexes : indices des trajectoires, groupe par groupe.

    Union-find sur les empreintes. Deux trajectoires partageant au moins
    une citation, un article ou une URL finissent dans la même composante,
    par transitivité.
    """
    parent = list(range(len(rows)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_signature: dict[str, int] = {}
    for index, row in enumerate(rows):
        for signature in row_signatures(row, group_by):
            first = by_signature.setdefault(signature, index)
            union(first, index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    # Ordre stable : par racine croissante.
    return [members for _root, members in sorted(components.items())]


def leakage_group(row: TrainingTrajectory) -> str:
    """Clé mono-trajectoire — conservée pour les appelants historiques.

    Ne suffit PAS à empêcher une fuite : deux trajectoires partageant une
    seule citation obtiennent des clés différentes. Utiliser
    ``leakage_groups``.
    """
    signatures = sorted(row_signatures(row))
    return "|".join(signatures) if signatures else f"family:{row.scenario_family_id}"


# ── Répartition ──────────────────────────────────────────────────────────


def assign_splits(groups: list[list[int]], ratios: dict[str, float],
                  seed: int, group_keys: list[str] | None = None,
                  ) -> dict[str, list[int]]:
    """Remplissage glouton sensible à la taille, déterministe.

    Un hash indépendant par groupe dérivait fortement des proportions
    visées dès que les groupes étaient de tailles inégales. Ici les groupes
    sont traités du plus gros au plus petit et chacun va au split le plus
    en retard sur son quota. Le déterminisme vient de la graine, qui ne
    départage que les ex æquo.
    """
    total = sum(len(members) for members in groups)
    targets = {name: ratio * total for name, ratio in ratios.items()}
    buckets: dict[str, list[int]] = {name: [] for name in ratios}
    if not total:
        return buckets

    keys = group_keys or [str(index) for index in range(len(groups))]
    ordered = sorted(
        range(len(groups)),
        key=lambda index: (
            -len(groups[index]),
            hashlib.sha256(f"{seed}:{keys[index]}".encode()).hexdigest(),
        ),
    )
    filled = {name: 0 for name in ratios}
    for index in ordered:
        members = groups[index]
        # Le split dont il manque le plus, en valeur absolue de trajectoires.
        name = max(ratios, key=lambda split: (targets[split] - filled[split],
                                              ratios[split], split))
        buckets[name].extend(members)
        filled[name] += len(members)
    return buckets


# ── Mesure du chevauchement ──────────────────────────────────────────────


def _shareable(row: TrainingTrajectory) -> dict[str, set[str]]:
    return {
        "citations": {c for g in row.grounding for c in g.citations},
        "articles": {str(o.arguments.get("start_article"))
                     for o in row.tool_trace
                     if o.arguments.get("start_article") is not None},
        "urls": {u for g in row.grounding for u in g.source_urls},
    }


def measure_overlap(buckets: dict[str, list[TrainingTrajectory]],
                    ) -> dict[str, Any]:
    """Chevauchement RÉEL entre splits, par intersection des sources.

    Remplace un ``"group_overlap": 0`` écrit en dur, qui affirmait
    l'absence de fuite sans jamais la vérifier.
    """
    per_split: dict[str, dict[str, set[str]]] = {}
    for name, members in buckets.items():
        merged: dict[str, set[str]] = {
            "citations": set(), "articles": set(), "urls": set()}
        for row in members:
            for kind, values in _shareable(row).items():
                merged[kind] |= values
        per_split[name] = merged

    pairs: dict[str, Any] = {}
    total = 0
    names = sorted(per_split)
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            shared = {
                kind: sorted(per_split[left][kind] & per_split[right][kind])
                for kind in ("citations", "articles", "urls")
            }
            count = sum(len(values) for values in shared.values())
            total += count
            pairs[f"{left}|{right}"] = {
                "count": count,
                **{kind: values[:20] for kind, values in shared.items()},
            }
    return {"total": total, "pairs": pairs}


# ── Publication ──────────────────────────────────────────────────────────


def _clean(row: TrainingTrajectory) -> dict[str, Any]:
    payload = row.model_dump(mode="json")
    for observation in payload.get("tool_trace", []):
        observation.pop("raw_response", None)
        observation.pop("latency_ms", None)
    return payload


def _audit_rows(rows: list[TrainingTrajectory], catalog,
                legal_min_score: float, agentic_min_score: float) -> None:
    errors: list[str] = []
    for row in rows:
        validation = validate_trajectory(row, catalog, allow_mock=False)
        if not validation.valid:
            errors.extend(f"{row.scenario_id}: {error}"
                          for error in validation.errors)
        if row.quality.deterministic_validation is not True:
            errors.append(
                f"{row.scenario_id}: validation déterministe non confirmée")
        if (row.quality.legal_critic_score is None
                or row.quality.legal_critic_score < legal_min_score):
            errors.append(f"{row.scenario_id}: seuil Legal Critic non atteint")
        if (row.quality.agentic_critic_score is None
                or row.quality.agentic_critic_score < agentic_min_score):
            errors.append(
                f"{row.scenario_id}: seuil Agentic Critic non atteint")
    if errors:
        raise PublicationError(
            "audit de publication échoué: " + "; ".join(errors[:10]))


def _hold_out_families(rows: list[TrainingTrajectory], ratio: float,
                       seed: int) -> tuple[list[int], list[int]]:
    """Réserve des FAMILLES entières de scénarios pour l'évaluation.

    Un held-out tiré au hasard parmi les trajectoires ne mesure que la
    conformité de route, puisque la taxonomie définit les routes attendues
    ET sert à valider. Réserver des familles jamais vues à l'entraînement
    est la seule façon de mesurer une généralisation.

    Retourne (indices retenus pour l'évaluation, indices restants).
    """
    if ratio <= 0:
        return [], list(range(len(rows)))
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        families[row.scenario_family_id or f"__sans_famille_{index}"].append(
            index)
    ordered = sorted(
        families.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}:eval:{item[0]}".encode()).hexdigest())
    target = ratio * len(rows)
    held: list[int] = []
    for _family, members in ordered:
        if len(held) >= target:
            break
        held.extend(members)
    held_set = set(held)
    return sorted(held), [i for i in range(len(rows)) if i not in held_set]


def prepare_release(accepted_jsonl: str | Path, output_dir: str | Path,
                    catalog, manifest_path: str | Path, seed: int = 3407,
                    ratios: tuple[float, float, float] = (0.90, 0.05, 0.05),
                    legal_min_score: float = 0.7,
                    agentic_min_score: float = 0.7,
                    group_by: Iterable[str] = DEFAULT_GROUP_BY,
                    agentic_eval_ratio: float = 0.05,
                    separate_agentic_evaluation: bool = True,
                    ) -> dict[str, Any]:
    rows = [TrainingTrajectory.model_validate(item)
            for item in iter_jsonl(accepted_jsonl)]
    if not rows:
        raise PublicationError("aucune trajectoire acceptée à publier")
    _audit_rows(rows, catalog, legal_min_score, agentic_min_score)

    held_indices, remaining = ([], list(range(len(rows))))
    if separate_agentic_evaluation:
        held_indices, remaining = _hold_out_families(
            rows, agentic_eval_ratio, seed)

    trainable = [rows[index] for index in remaining]
    groups = leakage_groups(trainable, group_by)
    group_keys = [
        "|".join(sorted(row_signatures(trainable[members[0]], group_by)))
        or f"groupe_{position}"
        for position, members in enumerate(groups)
    ]
    ratio_map = {"train": ratios[0], "validation": ratios[1], "test": ratios[2]}
    index_buckets = assign_splits(groups, ratio_map, seed, group_keys)
    buckets = {name: [trainable[index] for index in indices]
               for name, indices in index_buckets.items()}

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, members in buckets.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in members:
                handle.write(json.dumps(_clean(row), ensure_ascii=False) + "\n")

    # Held-out réellement disjoint : des familles entières, jamais vues à
    # l'entraînement — et non une copie du split test.
    evaluation = [rows[index] for index in held_indices]
    with (output / "agentic_eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in evaluation:
            handle.write(json.dumps(_clean(row), ensure_ascii=False) + "\n")

    overlap = measure_overlap({**buckets, "agentic_eval": evaluation})
    total = sum(len(members) for members in buckets.values())
    achieved = {name: (len(members) / total if total else 0.0)
                for name, members in buckets.items()}
    audit = {
        "rows": len(rows),
        "splits": {name: len(members) for name, members in buckets.items()},
        "agentic_eval": len(evaluation),
        "groups": len(groups),
        "group_by": list(group_by),
        "target_ratios": ratio_map,
        "achieved_ratios": {name: round(value, 4)
                            for name, value in achieved.items()},
        "group_overlap": overlap["total"],
        "overlap_detail": overlap["pairs"],
        "held_out_families": sorted({
            row.scenario_family_id for row in evaluation}),
        "metrics": compute_metrics(rows, catalog),
    }
    audit["passed"] = overlap["total"] == 0
    (output / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    (output / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {"dataset_type": "agentic_legal", "schema_version": "agentic-1.0",
            "features": list(_clean(rows[0]).keys()),
            "splits": audit["splits"]}
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Lexior agentic legal dataset\n\n"
        "Trajectoires ChatML fondées sur des réponses MCP réelles. Les "
        "splits sont formés par composantes connexes de sources partagées; "
        "le chevauchement reporté dans `audit_report.json` est mesuré, pas "
        "supposé. `agentic_eval.jsonl` contient des familles de scénarios "
        "entières, absentes des trois autres splits.\n", encoding="utf-8")
    return audit


def push_release(output_dir: str | Path, repo_id: str,
                 allow_remote_calls: bool) -> None:
    if not allow_remote_calls:
        raise PermissionError("publication refusée sans --allow-remote-calls")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN requis pour publier")
    if not repo_id:
        raise ValueError("HF_DATASET_REPO_ID requis pour publier")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True,
                    exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset",
                      folder_path=str(output_dir),
                      commit_message="Publish validated agentic Lexior dataset")
