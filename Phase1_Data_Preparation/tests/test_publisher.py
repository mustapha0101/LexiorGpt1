# -*- coding: utf-8 -*-
"""Publication : groupes, splits, fuite mesurée, held-out disjoint.

``publisher.py`` n'avait aucun site d'appel ni aucun test : l'étape qui
produit ``train/validation/test.jsonl`` n'avait jamais tourné.
"""

from __future__ import annotations

import json

import pytest

from agentic_generation.schemas import (
    GroundingEntry, Message, QualityReport, Role, ToolObservation,
    TrainingTrajectory,
)
from lexior.agentic.publisher import (
    DEFAULT_GROUP_BY,
    PublicationError,
    assign_splits,
    leakage_groups,
    measure_overlap,
    prepare_release,
    row_signatures,
)


def _row(scenario_id: str, *, citations=(), articles=(), urls=(),
         family="f1", valid=True) -> TrainingTrajectory:
    trace = [
        ToolObservation(
            tool_name="get_ccq_articles",
            arguments={"start_article": article},
            normalized_response=f"Article {article}\nTexte officiel intégral.",
            raw_response=f"Article {article}",
            mock=False,
        ).finalize_hash()
        for article in articles
    ] or [ToolObservation(
        tool_name="get_ccq_articles", arguments={"start_article": 1457},
        normalized_response="Article 1457\nTexte officiel intégral.",
        raw_response="Article 1457", mock=False).finalize_hash()]
    calls = "\n".join(
        f'<tool_call>\n{{"name":"get_ccq_articles","arguments":'
        f'{{"start_article":{o.arguments["start_article"]}}}}}\n</tool_call>'
        for o in trace)
    messages = [Message(role=Role.user, content="Quelle règle s'applique?")]
    for observation in trace:
        messages.append(Message(role=Role.assistant, content=(
            f'<tool_call>\n{{"name":"get_ccq_articles","arguments":'
            f'{{"start_article":'
            f'{observation.arguments["start_article"]}}}}}\n</tool_call>')))
        messages.append(Message(role=Role.tool, name="get_ccq_articles",
                                content=observation.normalized_response))
    messages.append(Message(role=Role.assistant,
                            content="Texte officiel intégral."))
    del calls
    return TrainingTrajectory(
        scenario_id=scenario_id, scenario_family_id=family,
        request_type="case_analysis",
        expected_jurisdiction="Québec", resolved_jurisdiction="Québec",
        messages=messages, tool_trace=trace,
        grounding=[GroundingEntry(
            tool_name="get_ccq_articles", content_hash="h",
            source_urls=list(urls), citations=list(citations))],
        quality=QualityReport(
            deterministic_validation=valid,
            legal_critic_score=0.9 if valid else 0.1,
            agentic_critic_score=0.9 if valid else 0.1),
    )


# ── 3.2 : composantes connexes ───────────────────────────────────────────


def test_two_rows_sharing_one_citation_land_in_the_same_group():
    """Le défaut d'origine : {1726,1728} et {1726} obtenaient deux clés."""
    rows = [
        _row("a", citations=("1726", "1728"), family="fa"),
        _row("b", citations=("1726",), family="fb"),
    ]

    groups = leakage_groups(rows, ("citations",))

    assert len(groups) == 1, "1726 ne doit pas pouvoir tomber des deux côtés"


def test_grouping_is_transitive():
    """A partage avec B, B partage avec C : les trois sont solidaires."""
    rows = [
        _row("a", citations=("1726",), family="fa"),
        _row("b", citations=("1726", "1728"), family="fb"),
        _row("c", citations=("1728",), family="fc"),
    ]

    groups = leakage_groups(rows, ("citations",))

    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_rows_without_any_shared_source_stay_separate():
    rows = [
        _row("a", citations=("1726",), family="fa"),
        _row("b", citations=("2925",), family="fb"),
    ]

    assert len(leakage_groups(rows, ("citations",))) == 2


def test_a_row_carries_every_signature_not_just_the_first():
    row = _row("a", citations=("1726",), articles=(1726, 1728),
               urls=("https://example/1726",), family="fa")

    signatures = row_signatures(row, DEFAULT_GROUP_BY)

    assert "citation:1726" in signatures
    assert "article:1726" in signatures and "article:1728" in signatures
    assert "url:https://example/1726" in signatures
    assert "family:fa" in signatures


def test_an_unknown_grouping_dimension_is_refused():
    """Une clé de config qui ne s'applique pas ne doit pas passer en silence."""
    with pytest.raises(PublicationError, match="inconnues"):
        row_signatures(_row("a"), ("question_family",))


# ── 3.4 : proportions ────────────────────────────────────────────────────


def test_greedy_assignment_respects_the_target_proportions():
    # 20 groupes de tailles très inégales : le hash indépendant dérivait.
    groups = [list(range(size)) for size in
              [30, 25, 20, 15, 10, 8, 6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1]]
    total = sum(len(g) for g in groups)

    buckets = assign_splits(
        groups, {"train": 0.80, "validation": 0.10, "test": 0.10}, seed=3407)

    achieved = {name: len(items) / total for name, items in buckets.items()}
    assert abs(achieved["train"] - 0.80) < 0.12, achieved
    assert achieved["validation"] > 0 and achieved["test"] > 0, achieved


def test_assignment_is_deterministic_for_a_given_seed():
    groups = [list(range(size)) for size in [5, 4, 3, 2, 1]]
    ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

    first = assign_splits(groups, ratios, seed=7)
    second = assign_splits(groups, ratios, seed=7)

    assert first == second


def test_a_group_is_never_split_across_two_splits():
    groups = [list(range(size)) for size in [7, 5, 3, 2]]

    buckets = assign_splits(
        groups, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)

    placement = {}
    for name, indices in buckets.items():
        for index in indices:
            placement.setdefault(index, name)
    for group in groups:
        assert len({placement[i] for i in group}) == 1


# ── 3.3 : chevauchement mesuré ───────────────────────────────────────────


def test_overlap_is_detected_when_a_citation_crosses_splits():
    buckets = {
        "train": [_row("a", citations=("1726",), articles=(1726,))],
        "test": [_row("b", citations=("1726",), articles=(1726,))],
    }

    overlap = measure_overlap(buckets)

    assert overlap["total"] == 2, "une citation ET un article partagés"
    assert "1726" in overlap["pairs"]["test|train"]["citations"]
    assert "1726" in overlap["pairs"]["test|train"]["articles"]


def test_no_overlap_is_reported_when_there_is_none():
    buckets = {
        "train": [_row("a", citations=("1726",), articles=(1726,))],
        "test": [_row("b", citations=("2925",), articles=(2925,))],
    }

    assert measure_overlap(buckets)["total"] == 0


# ── Publication de bout en bout ──────────────────────────────────────────


def _write_accepted(tmp_path, rows):
    path = tmp_path / "accepted.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"),
                                    ensure_ascii=False) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "test"}), encoding="utf-8")
    return path, manifest


def test_prepare_release_writes_every_expected_file(tmp_path, catalog):
    rows = [_row(f"s{index}", citations=(str(1700 + index),),
                 articles=(1700 + index,), family=f"famille{index % 5}")
            for index in range(20)]
    accepted, manifest = _write_accepted(tmp_path, rows)
    output = tmp_path / "release"

    audit = prepare_release(accepted, output, catalog, manifest, seed=3407)

    for name in ("train.jsonl", "validation.jsonl", "test.jsonl",
                 "agentic_eval.jsonl", "audit_report.json",
                 "dataset_info.json", "generation_manifest.json", "README.md"):
        assert (output / name).exists(), name
    assert audit["rows"] == 20


def test_the_audit_reports_measured_overlap_not_a_constant(tmp_path, catalog):
    rows = [_row(f"s{index}", citations=(str(1700 + index),),
                 articles=(1700 + index,), family=f"famille{index % 5}")
            for index in range(20)]
    accepted, manifest = _write_accepted(tmp_path, rows)
    output = tmp_path / "release"

    audit = prepare_release(accepted, output, catalog, manifest, seed=3407)
    report = json.loads((output / "audit_report.json").read_text(
        encoding="utf-8"))

    assert "overlap_detail" in report, "le détail par paire doit être écrit"
    assert report["passed"] == (report["group_overlap"] == 0)
    assert report["achieved_ratios"], "les proportions obtenues sont reportées"


def test_the_evaluation_set_is_not_a_copy_of_the_test_split(tmp_path, catalog):
    rows = [_row(f"s{index}", citations=(str(1700 + index),),
                 articles=(1700 + index,), family=f"famille{index % 6}")
            for index in range(24)]
    accepted, manifest = _write_accepted(tmp_path, rows)
    output = tmp_path / "release"

    prepare_release(accepted, output, catalog, manifest, seed=3407,
                    agentic_eval_ratio=0.20)

    def ids(name):
        return {json.loads(line)["scenario_id"]
                for line in (output / name).read_text(
                    encoding="utf-8").splitlines() if line.strip()}

    evaluation = ids("agentic_eval.jsonl")
    assert evaluation, "le held-out ne doit pas être vide"
    assert not evaluation & ids("test.jsonl")
    assert not evaluation & ids("train.jsonl")
    assert not evaluation & ids("validation.jsonl")


def test_held_out_families_never_appear_in_training(tmp_path, catalog):
    rows = [_row(f"s{index}", citations=(str(1700 + index),),
                 articles=(1700 + index,), family=f"famille{index % 6}")
            for index in range(24)]
    accepted, manifest = _write_accepted(tmp_path, rows)
    output = tmp_path / "release"

    audit = prepare_release(accepted, output, catalog, manifest, seed=3407,
                            agentic_eval_ratio=0.20)

    train_families = {
        json.loads(line)["scenario_family_id"]
        for line in (output / "train.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()}
    assert not train_families & set(audit["held_out_families"])


def test_a_trajectory_below_the_critic_threshold_blocks_publication(
        tmp_path, catalog):
    rows = [_row("bon", citations=("1726",), articles=(1726,)),
            _row("mauvais", citations=("2925",), articles=(2925,),
                 valid=False)]
    accepted, manifest = _write_accepted(tmp_path, rows)

    with pytest.raises(PublicationError, match="audit de publication"):
        prepare_release(accepted, tmp_path / "release", catalog, manifest)


def test_publishing_nothing_is_an_error(tmp_path, catalog):
    accepted, manifest = _write_accepted(tmp_path, [])

    with pytest.raises(PublicationError, match="aucune trajectoire"):
        prepare_release(accepted, tmp_path / "release", catalog, manifest)
