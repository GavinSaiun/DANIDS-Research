from __future__ import annotations

import json
from pathlib import Path

import pytest

from danids.attacks import load_study5_contract
from danids.evaluation.threat_estimands import (
    DetectionCounts,
    FamilyRecallCell,
    FamilyTrajectoryCell,
    LearnedStatus,
    NoveltyStatus,
    ThreatEstimandError,
    aggregate_semantic_cells,
    detection_counts_from_tp,
    domain_entry_novelty,
    family_forgetting,
    hidden_family_failures,
    paired_supported_identities,
    physical_slice_digest,
    reconstruct_detection_counts,
    summarize_supported_cells,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "study5" / "attack_ontology_v1.yaml"


def _native_row(
    identity: str,
    family: str | None,
    support: int,
    tp: int,
    *,
    experiment_id: str = "run-1",
    physical: str = "slice-1",
    domain: str = "C",
) -> dict[str, object]:
    counts = detection_counts_from_tp(support, tp)
    return {
        "experiment_id": experiment_id,
        "physical_slice_digest": physical,
        "evaluated_domain": domain,
        "native_identity": identity,
        "mapping_status": "MAPPED" if family is not None else "UNMAPPED",
        "semantic_family": family,
        "support": support,
        "tp": tp,
        "fn": counts.fn,
        "recall": counts.recall,
    }


def test_reconstruct_detection_counts_requires_an_exact_integer_ratio() -> None:
    assert reconstruct_detection_counts(1_000, 0.943) == (943, 57)
    assert reconstruct_detection_counts(100, 1.0) == (100, 0)
    with pytest.raises(ThreatEstimandError, match="integer TP"):
        reconstruct_detection_counts(100, 0.955)
    with pytest.raises(ThreatEstimandError, match=r"\[0, 1\]"):
        reconstruct_detection_counts(100, 1.01)
    with pytest.raises(ThreatEstimandError, match="support"):
        reconstruct_detection_counts(0, 0.0)


def test_wilson_interval_uses_the_frozen_score_and_boundaries() -> None:
    low_95, high_95 = wilson_interval(95, 100)
    low_96, high_96 = wilson_interval(96, 100)
    assert low_95 == pytest.approx(0.8882495307680808)
    assert high_95 == pytest.approx(0.9784567282806333)
    assert low_96 == pytest.approx(0.9016292856411229)
    assert high_96 == pytest.approx(0.9843370267470979)
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0


def test_physical_slice_digest_is_canonical_and_excludes_implicit_state() -> None:
    left = {
        "dataset_fingerprint": "a" * 64,
        "dataset_id": "T",
        "stratum": "ONLINE_STREAM",
        "row_start": 0,
        "row_stop": 50_000,
    }
    right = dict(reversed(tuple(left.items())))
    assert physical_slice_digest(left) == physical_slice_digest(right)
    assert physical_slice_digest(left) != physical_slice_digest({**left, "row_stop": 100_000})
    with pytest.raises(ThreatEstimandError, match="canonical-JSON"):
        physical_slice_digest({"not_serializable": object()})


def test_semantic_aggregation_sums_mapped_counts_without_pooling_unmapped() -> None:
    rows = [
        _native_row("C::DoS_attacks-Hulk", "Availability / Impact", 100, 80),
        _native_row("C::DoS_attacks-Slowloris", "Availability / Impact", 50, 30),
        _native_row("C::Infilteration", None, 75, 15),
    ]
    result = aggregate_semantic_cells(
        rows,
        context_fields=("experiment_id", "physical_slice_digest", "evaluated_domain"),
    )
    assert len(result) == 1
    cell = result[0]
    assert cell["semantic_family"] == "Availability / Impact"
    assert cell["support"] == 150
    assert cell["tp"] == 110
    assert cell["fn"] == 40
    assert cell["recall"] == pytest.approx(110 / 150)
    assert json.loads(str(cell["native_members"])) == [
        "C::DoS_attacks-Hulk",
        "C::DoS_attacks-Slowloris",
    ]
    assert "Infilteration" not in str(cell["native_members"])


def test_semantic_aggregation_never_crosses_physical_or_domain_context() -> None:
    rows = [
        _native_row("U::DoS", "Availability / Impact", 60, 30, physical="u", domain="U"),
        _native_row("B::DDoS", "Availability / Impact", 90, 45, physical="b", domain="B"),
    ]
    result = aggregate_semantic_cells(
        rows,
        context_fields=("experiment_id", "physical_slice_digest", "evaluated_domain"),
    )
    assert [(row["evaluated_domain"], row["support"]) for row in result] == [
        ("B", 90),
        ("U", 60),
    ]


def test_supported_summary_is_unweighted_and_persists_all_exact_ties() -> None:
    rows = [
        _native_row("C::Bot", "Botnet / Command-and-Control", 100, 50),
        _native_row("C::SQL_Injection", "Application / Web Injection", 50, 25),
        _native_row("C::Brute_Force_-XSS", "Application / Web Injection", 49, 0),
    ]
    summary = summarize_supported_cells(
        rows,
        context_fields=("experiment_id", "physical_slice_digest", "evaluated_domain"),
        identity_field="native_identity",
    )[0]
    assert summary["supported_family_count"] == 2
    assert summary["macro_supported_recall"] == 0.5
    assert summary["worst_supported_recall"] == 0.5
    assert json.loads(str(summary["worst_family_identities"])) == [
        "C::Bot",
        "C::SQL_Injection",
    ]


def test_paired_support_is_a_physical_contract_not_a_pool() -> None:
    left = (
        FamilyRecallCell("U::DoS", detection_counts_from_tp(50, 25)),
        FamilyRecallCell("U::Worms", detection_counts_from_tp(48, 24)),
    )
    right = (
        FamilyRecallCell("U::DoS", detection_counts_from_tp(50, 40)),
        FamilyRecallCell("U::Worms", detection_counts_from_tp(48, 40)),
    )
    assert paired_supported_identities(left, right) == ("U::DoS",)
    corrupted = (
        FamilyRecallCell("U::DoS", detection_counts_from_tp(49, 40)),
        right[1],
    )
    with pytest.raises(ThreatEstimandError, match="physical family support"):
        paired_supported_identities(left, corrupted)


def test_novelty_uses_only_source_exposure_and_completed_prior_online_domains() -> None:
    contract = load_study5_contract(CONTRACT_PATH)
    lookup = {
        (item.dataset_id, item.semantic_family): item.status
        for item in domain_entry_novelty(contract, ("U", "T", "C", "B"))
    }
    assert lookup[("U", "Availability / Impact")] is NoveltyStatus.PREVIOUSLY_SEEN
    assert lookup[("T", "Application / Web Injection")] is NoveltyStatus.PREVIOUSLY_UNSEEN
    assert lookup[("C", "Application / Web Injection")] is NoveltyStatus.PREVIOUSLY_SEEN
    # T ransomware occurs only in its permanent holdout, which cannot enter history.
    assert lookup[("C", "Ransomware Impact")] is NoveltyStatus.PREVIOUSLY_UNSEEN
    assert lookup[("B", "Ransomware Impact")] is NoveltyStatus.PREVIOUSLY_UNSEEN
    # C Bot likewise occurs only in its permanent holdout.
    assert lookup[("B", "Botnet / Command-and-Control")] is NoveltyStatus.PREVIOUSLY_UNSEEN


def test_hidden_failure_uses_inclusive_aggregate_and_strict_family_boundaries() -> None:
    reference_aggregate = detection_counts_from_tp(100, 95)
    current_aggregate = detection_counts_from_tp(100, 85)
    reference = (FamilyRecallCell("family-a", detection_counts_from_tp(100, 90)),)
    current = (FamilyRecallCell("family-a", detection_counts_from_tp(100, 79)),)
    failures = hidden_family_failures(
        reference_aggregate,
        current_aggregate,
        reference,
        current,
    )
    assert len(failures) == 1
    assert failures[0].aggregate_recall_loss == pytest.approx(0.10)
    assert failures[0].family_recall_loss == pytest.approx(0.11)

    strict_boundary = (FamilyRecallCell("family-a", detection_counts_from_tp(100, 80)),)
    assert not hidden_family_failures(
        reference_aggregate,
        current_aggregate,
        reference,
        strict_boundary,
    )
    aggregate_too_large = detection_counts_from_tp(100, 84)
    assert not hidden_family_failures(
        reference_aggregate,
        aggregate_too_large,
        reference,
        current,
    )


def test_family_forgetting_begins_at_learned_event_and_excludes_pre_adapt() -> None:
    cells = (
        FamilyTrajectoryCell("family-a", "pre_adapt", 2, detection_counts_from_tp(100, 90)),
        FamilyTrajectoryCell("family-a", "post_adapt", 3, detection_counts_from_tp(100, 60)),
        FamilyTrajectoryCell("family-a", "domain_end", 4, detection_counts_from_tp(100, 55)),
        FamilyTrajectoryCell("family-a", "final", 5, detection_counts_from_tp(100, 50)),
    )
    result = family_forgetting(cells, learned_event_index=3, final_event_index=5)
    assert len(result) == 1
    assert result[0].learned_status is LearnedStatus.LEARNED
    assert result[0].maximum_event == "post_adapt"
    assert result[0].forgetting == pytest.approx(0.10)


def test_no_learned_update_is_explicit_and_does_not_manufacture_forgetting() -> None:
    cells = (
        FamilyTrajectoryCell("family-a", "domain_end", 4, detection_counts_from_tp(100, 70)),
        FamilyTrajectoryCell("family-a", "final", 5, detection_counts_from_tp(100, 50)),
    )
    result = family_forgetting(cells, learned_event_index=None, final_event_index=5)
    assert result[0].learned_status is LearnedStatus.NOT_LEARNED_NO_UPDATE
    assert result[0].learned_recall is None
    assert result[0].maximum_recall is None
    assert result[0].forgetting is None
    assert result[0].aggregate_eligible is False


def test_low_support_forgetting_cell_is_descriptive_but_unavailable() -> None:
    cells = (
        FamilyTrajectoryCell("family-a", "post_adapt", 3, detection_counts_from_tp(49, 40)),
        FamilyTrajectoryCell("family-a", "final", 5, detection_counts_from_tp(49, 30)),
    )
    result = family_forgetting(cells, learned_event_index=3, final_event_index=5)
    assert result[0].learned_status is LearnedStatus.UNAVAILABLE_SUPPORT
    assert result[0].forgetting is None


def test_detection_counts_type_is_preserved_for_adapter_use() -> None:
    counts = detection_counts_from_tp(75, 25)
    assert counts == DetectionCounts(75, 25, 50, 1 / 3)
