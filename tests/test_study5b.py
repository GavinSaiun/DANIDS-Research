from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from danids.attacks import MappingStatus, load_study5_contract
from danids.config.study5b import canonical_contract_sha256
from danids.continual.supervision import (
    SupervisionSchedule,
    generate_supervision_schedule_from_identity,
)
from danids.evaluation import study5_sources
from danids.evaluation.study5b import (
    ALL_ORDER_LAYER,
    ALL_OUTPUT_FILES,
    COMPETENCE_DIMENSION,
    CONTRACT_VERSION,
    EVALUATOR_VERSION,
    EXISTING_EVIDENCE,
    EXTENSION_ROTATIONS,
    FORGETTING_DIMENSION,
    METHODS,
    NOT_ASSESSED_INCOMPLETE,
    NOT_SUPPORTED,
    PARTIALLY_SUPPORTED,
    PROSPECTIVE_EVIDENCE,
    PROSPECTIVE_LAYER,
    SEEDS,
    SUPPORTED,
    _canonical_extension_schedule_validation,
    _cell_verdict,
    _derive_all,
    _derive_method_dimension_summaries,
    _derive_order_summaries,
    _derive_sequence_seed_units,
    _digest,
    _evidence_layer,
    _expected_pipeline_contract,
    _experiment_id,
    _file_sha256,
    _h7_verdict,
    _layer_verdict,
    _load_task009_contract,
    _normalise_trajectory,
    _ontology_semantic_supports,
    _source_artifacts_record,
    _task009_contract_record,
    _tree_digests,
    _validate_source_artifacts_record,
    _validated_pipeline_contract,
    _write_evaluation_bundle,
    _write_json,
    validate_study5b_evaluation,
)
from danids.evaluation.threat_estimands import physical_slice_digest, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
TASK009_CONTRACT = ROOT / "configs" / "study5" / "task009_h7_all_order_v1.yaml"
ONTOLOGY_PATH = ROOT / "configs" / "study5" / "attack_ontology_v1.yaml"
ONTOLOGY = load_study5_contract(ONTOLOGY_PATH)
ROTATIONS = (
    ("U", "T", "C", "B"),
    *EXTENSION_ROTATIONS,
)
LABELS = {
    domain: tuple(
        (item.key.exact_native_label, str(item.semantic_family))
        for item in ONTOLOGY.native_attacks
        if item.key.dataset_id == domain
        and item.mapping_status is MappingStatus.MAPPED
        and item.support.permanent_holdout > 0
    )
    for domain in ("U", "T", "C", "B")
}


def _trajectory_cell(
    *,
    sequence: tuple[str, ...],
    seed: int,
    method: str,
    domain: str,
    native_label: str,
    semantic_family: str,
    lifecycle: str,
    stage: int,
    event_index: int,
    tp: int,
    level: str,
) -> dict[str, object]:
    sequence_text = "-".join(sequence)
    experiment_id = _experiment_id(method, sequence, seed)
    position = sequence.index(domain) + 1
    support = ONTOLOGY.mapping_for(domain, native_label).support.permanent_holdout
    fingerprint = ONTOLOGY.dataset_fingerprints[domain]
    physical_start, physical_stop = study5_sources._partition_bounds_from_contract(
        ONTOLOGY, domain, "PERMANENT_HOLDOUT"
    )
    physical_identity = {
        "dataset_fingerprint": fingerprint,
        "dataset_id": domain,
        "row_start": physical_start,
        "row_stop": physical_stop,
        "stratum": "PERMANENT_HOLDOUT",
    }
    evaluation_digest = _digest(
        {
            "experiment_id": experiment_id,
            "domain": domain,
            "event_index": event_index,
            "lifecycle": lifecycle,
        }
    )
    low, high = wilson_interval(tp, support)
    identity = f"{domain}::{native_label}" if level == "NATIVE" else semantic_family
    return {
        "evidence_layer": _evidence_layer(sequence),
        "experiment_id": experiment_id,
        "method": method,
        "sequence": sequence_text,
        "seed": seed,
        "family_level": level,
        "stratum": "PERMANENT_HOLDOUT",
        "lifecycle_event": lifecycle,
        "stage": stage,
        "event_index": event_index,
        "evaluated_domain": domain,
        "domain_position": position,
        "family_identity": identity,
        "native_attack_label": native_label if level == "NATIVE" else "",
        "mapping_status": MappingStatus.MAPPED.value,
        "semantic_family": semantic_family,
        "physical_slice_start": physical_start,
        "physical_slice_stop": physical_stop,
        "dataset_fingerprint": fingerprint,
        "physical_slice_identity": json.dumps(
            physical_identity, sort_keys=True, separators=(",", ":")
        ),
        "support": support,
        "tp": tp,
        "fn": support - tp,
        "recall": tp / support,
        "wilson95_low": low,
        "wilson95_high": high,
        "supported_n50": support >= 50,
        "physical_slice_digest": physical_slice_digest(physical_identity),
        "evaluation_slice_digest": evaluation_digest,
        "model_digest": _digest({"experiment_id": experiment_id, "event": event_index}),
        "learned_state_status": (
            "LEARNED_REFERENCE"
            if (position == 1 and lifecycle == "source_initial")
            or (position > 1 and lifecycle == "post_adapt" and stage == position)
            else ""
        ),
    }


def _complete_trajectory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence in ROTATIONS:
        for seed in SEEDS:
            for method in METHODS:
                for position, domain in enumerate(sequence, start=1):
                    for native_label, semantic_family in LABELS[domain]:
                        support = ONTOLOGY.mapping_for(
                            domain, native_label
                        ).support.permanent_holdout
                        learned_tp = (4 * support) // 5
                        final_tp = {
                            "naive_ft": learned_tp - ((3 * support + 9) // 10),
                            "er": learned_tp - (support // 10),
                            "ft_mem": learned_tp - ((2 * support + 9) // 10),
                        }[method]
                        events: list[tuple[str, int, int, int]] = []
                        if position == 1:
                            events.append(("source_initial", 1, 1, learned_tp))
                        event_indices = {
                            2: (2, 3, 4),
                            3: (5, 6, 7),
                            4: (8, 9, 10),
                        }
                        for event_stage in range(max(2, position), 5):
                            pre, post, domain_end = event_indices[event_stage]
                            # High pre-adapt values must never enter M.  Post-adapt
                            # and domain-end ties test the earliest exact-max rule.
                            events.extend(
                                [
                                    ("pre_adapt", event_stage, pre, support),
                                    ("post_adapt", event_stage, post, learned_tp),
                                    ("domain_end", event_stage, domain_end, learned_tp),
                                ]
                            )
                        events.append(("final", 4, 11, final_tp))
                        for lifecycle, stage, event_index, tp in events:
                            for level in ("NATIVE", "SEMANTIC"):
                                rows.append(
                                    _trajectory_cell(
                                        sequence=sequence,
                                        seed=seed,
                                        method=method,
                                        domain=domain,
                                        native_label=native_label,
                                        semantic_family=semantic_family,
                                        lifecycle=lifecycle,
                                        stage=stage,
                                        event_index=event_index,
                                        tp=tp,
                                        level=level,
                                    )
                                )
    return _normalise_trajectory(rows, ONTOLOGY)


def _fake_source_identity(sequence: tuple[str, ...], seed: int) -> list[str]:
    return [
        _digest(
            {
                "sequence": sequence,
                "seed": seed,
                "source_identity_component": component,
            }
        )
        for component in (
            "checkpoint",
            "model",
            "preprocessor",
            "threshold",
        )
    ]


def _canonical_schedule_fixture(
    sequence: tuple[str, ...], seed: int
) -> tuple[SupervisionSchedule, dict[str, object]]:
    later = sequence[1:]
    fingerprints = {domain: ONTOLOGY.dataset_fingerprints[domain] for domain in later}
    ranges = {domain: (0, 100_000) for domain in later}
    schedule = generate_supervision_schedule_from_identity(
        sequence=sequence,
        seed=seed,
        dataset_fingerprints=fingerprints,
        online_stream_ranges=ranges,
    )
    actual_file_sha256 = _digest(
        {"sequence": sequence, "seed": seed, "canonical_schedule_file": True}
    )
    return schedule, {
        "status": "MATCHED_DETERMINISTIC_CANONICAL_SCHEDULE",
        "generator_version": schedule.version,
        "dataset_fingerprints": fingerprints,
        "online_stream_ranges": {
            domain: {"start": start, "stop": stop} for domain, (start, stop) in ranges.items()
        },
        "expected_schedule_digest": schedule.digest(),
        "actual_schedule_digest": schedule.digest(),
        "actual_schedule_file_sha256": actual_file_sha256,
        "canonical_schedule_equal": True,
    }


def _fake_source() -> dict[str, object]:
    metadata: list[dict[str, object]] = []
    for sequence in ROTATIONS:
        evidence = _evidence_layer(sequence)
        for seed in SEEDS:
            source_identity = _fake_source_identity(sequence, seed)
            static_files = {
                name: (
                    source_identity[0]
                    if name == "best_model.pt"
                    else _digest(
                        {
                            "kind": "static",
                            "sequence": sequence,
                            "seed": seed,
                            "file": name,
                        }
                    )
                )
                for name in study5_sources._STUDY1_CONSUMED_FILES
            }
            common: dict[str, object] = {
                "source_study": "S2",
                "analysis_role": "STUDY2_STATIC_REFERENCE",
                "sequence": list(sequence),
                "seed": seed,
                "path": f"runs/static-{'-'.join(sequence)}-{seed}",
                "dataset_fingerprints": dict(ONTOLOGY.dataset_fingerprints),
                "validation_kind": "validate_static_study1_run",
                "validated_file_digests": static_files,
                "validated_bundle_digest": _digest(static_files),
                "artifact_manifest_sha256": None,
                "artifact_manifest_bundle_digest": None,
                "evidence_layer": evidence,
                "task009_role": "STATIC_REFERENCE",
                "normalized_pipeline_signature_sha256": None,
                "source_identity": source_identity,
            }
            metadata.append(
                {
                    **common,
                    "experiment_id": f"E1_STATIC_MLP_{'-'.join(sequence)}_s{seed}",
                    "method": "static",
                }
            )
            schedule_file_sha256 = _digest(
                {"sequence": sequence, "seed": seed, "schedule_bytes": True}
            )
            if sequence in EXTENSION_ROTATIONS:
                schedule_file_sha256 = str(
                    _canonical_schedule_fixture(sequence, seed)[1]["actual_schedule_file_sha256"]
                )
            for method in METHODS:
                adaptive_files = {
                    name: (
                        schedule_file_sha256
                        if name == "supervision_schedule.json"
                        else _digest(
                            {
                                "kind": "adaptive",
                                "sequence": sequence,
                                "seed": seed,
                                "method": method,
                                "file": name,
                            }
                        )
                    )
                    for name in study5_sources._STUDY2_CONSUMED_FILES
                }
                metadata.append(
                    {
                        **common,
                        "analysis_role": "STUDY2_ADAPTIVE",
                        "experiment_id": _experiment_id(method, sequence, seed),
                        "method": method,
                        "path": f"runs/{_experiment_id(method, sequence, seed)}",
                        "validation_kind": "validate_continual_run",
                        "task009_role": "ADAPTIVE_RUN",
                        "validated_file_digests": adaptive_files,
                        "validated_bundle_digest": _digest(adaptive_files),
                        "normalized_pipeline_signature_sha256": _digest(
                            _expected_pipeline_contract()
                        ),
                        "source_identity": list(source_identity),
                    }
                )
    blocks = []
    for sequence in EXTENSION_ROTATIONS:
        for seed in SEEDS:
            schedule, validation = _canonical_schedule_fixture(sequence, seed)
            blocks.append(
                {
                    "sequence": list(sequence),
                    "seed": seed,
                    "methods": list(METHODS),
                    "schedule_digest": schedule.digest(),
                    "schedule_file_sha256": validation["actual_schedule_file_sha256"],
                    "canonical_schedule_validation": validation,
                    "source_identity": _fake_source_identity(sequence, seed),
                    "scientific_signature_sha256": "8" * 64,
                }
            )
    upstream = {
        "path": "study5/threat-audit-v1",
        "evaluator_version": "task008-study5a-threat-audit-v1",
        "bundle_digest": "9e339bc3dce92a72d666a5b750e49e2bcb7c3247b184d17bbe0410153ab87fea",
        "artifact_manifest_sha256": "9" * 64,
        "source_artifacts_sha256": "a" * 64,
        "native_family_long_sha256": "b" * 64,
        "reuse_policy": "VALIDATE_REUSE_NO_RERUN",
        "normalized_pipeline_contract": _expected_pipeline_contract(),
        "normalized_pipeline_signature_sha256": _digest(_expected_pipeline_contract()),
    }
    return _source_artifacts_record(
        ontology=ONTOLOGY,
        task009_contract_sha256=(
            "99d613455334f815a2472fe4f3c182bd402f29552bc4014013ed9e8e252e02db"
        ),
        upstream=upstream,
        metadata=metadata,
        pairing_blocks=blocks,
    )


def _write_schedule_validation_fixture(
    root: Path, sequence: tuple[str, ...], schedule: SupervisionSchedule
) -> None:
    later = sequence[1:]
    root.mkdir(parents=True)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_fingerprints": dict(ONTOLOGY.dataset_fingerprints),
                "manifest_partition_ranges": {
                    domain: {"online_stream": {"start": 0, "stop": 100_000}} for domain in later
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "supervision_schedule.json").write_text(schedule.to_json(), encoding="utf-8")


def _noncanonical_schedule(schedule: SupervisionSchedule) -> SupervisionSchedule:
    first = schedule.entries[0]
    positions = set(first.chronological_positions)
    replacement = next(position for position in range(50_000) if position not in positions)
    changed = tuple(sorted((positions - {min(positions)}) | {replacement}))
    result = replace(
        schedule,
        entries=(replace(first, chronological_positions=changed), *schedule.entries[1:]),
    )
    result.validate()
    return result


def test_prospective_schedule_validation_rejects_coherent_noncanonical_trio(
    tmp_path: Path,
) -> None:
    sequence = EXTENSION_ROTATIONS[0]
    seed = SEEDS[0]
    canonical, _record = _canonical_schedule_fixture(sequence, seed)
    noncanonical = _noncanonical_schedule(canonical)
    assert noncanonical.digest() != canonical.digest()
    assert (
        noncanonical.entries[0].chronological_positions
        != canonical.entries[0].chronological_positions
    )

    for method in METHODS:
        root = tmp_path / method
        _write_schedule_validation_fixture(root, sequence, noncanonical)
        with pytest.raises(ValueError, match="deterministic canonical TASK-009 schedule"):
            _canonical_extension_schedule_validation(root=root, sequence=sequence, seed=seed)
    assert not (tmp_path / "h7_verdict.json").exists()


def test_prospective_schedule_validation_accepts_canonical_trio_and_one_mismatch_fails(
    tmp_path: Path,
) -> None:
    sequence = EXTENSION_ROTATIONS[1]
    seed = SEEDS[1]
    canonical, expected_record = _canonical_schedule_fixture(sequence, seed)
    observed = []
    for method in METHODS:
        root = tmp_path / method
        _write_schedule_validation_fixture(root, sequence, canonical)
        observed.append(
            _canonical_extension_schedule_validation(root=root, sequence=sequence, seed=seed)
        )
    for record in observed:
        assert {
            key: value for key, value in record.items() if key != "actual_schedule_file_sha256"
        } == {
            key: value
            for key, value in expected_record.items()
            if key != "actual_schedule_file_sha256"
        }
        assert len(str(record["actual_schedule_file_sha256"])) == 64
    assert len({record["actual_schedule_file_sha256"] for record in observed}) == 1

    mismatched_root = tmp_path / METHODS[-1]
    (mismatched_root / "supervision_schedule.json").write_text(
        _noncanonical_schedule(canonical).to_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="deterministic canonical TASK-009 schedule"):
        _canonical_extension_schedule_validation(
            root=mismatched_root,
            sequence=sequence,
            seed=seed,
        )


def test_prior_u_order_schedule_remains_under_original_validated_contract() -> None:
    source = _fake_source()
    replacement_digest = "f" * 64
    for run in source["runs"]:
        if (
            run["task009_role"] == "ADAPTIVE_RUN"
            and run["sequence"] == ["U", "T", "C", "B"]
            and run["seed"] == 42
        ):
            run["validated_file_digests"]["supervision_schedule.json"] = replacement_digest
            run["validated_bundle_digest"] = _digest(run["validated_file_digests"])
    path, raw, digest = _load_task009_contract(TASK009_CONTRACT)
    contract_record = _task009_contract_record(
        path=path,
        raw=raw,
        contract_sha256=digest,
        ontology_path=ONTOLOGY_PATH,
        ontology=ONTOLOGY,
    )
    _validate_source_artifacts_record(source, contract_record, ONTOLOGY.dataset_fingerprints)


def test_contract_identity_and_evidence_vocab_are_exact(tmp_path: Path) -> None:
    _path, raw, digest = _load_task009_contract(TASK009_CONTRACT)
    assert digest == raw["contract_sha256"]
    assert digest == "99d613455334f815a2472fe4f3c182bd402f29552bc4014013ed9e8e252e02db"
    assert EXISTING_EVIDENCE == "PRIOR_EXISTING_EVIDENCE"
    assert PROSPECTIVE_EVIDENCE == "PROSPECTIVE_EXTENSION_EVIDENCE"
    assert PROSPECTIVE_LAYER == "PROSPECTIVE_MISSING_ROTATION_EXTENSION"
    with pytest.raises(ValueError, match="unknown TASK-009 method"):
        _experiment_id("ewc", ("T", "C", "B", "U"), 42)

    changed = copy.deepcopy(raw)
    changed["analysis"]["primary_minimum_physical_support"] = 51
    changed["contract_sha256"] = canonical_contract_sha256(changed)
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="digest differs from the code freeze"):
        _load_task009_contract(path)


def test_exact_outcomes_pairing_and_hierarchical_units() -> None:
    tables, verdicts = _derive_all(_complete_trajectory())
    outcomes = tables["family_outcomes.csv"]
    semantic_pairs = tables["semantic_paired_effects.csv"]

    assert len({row["experiment_id"] for row in outcomes}) == 36
    eligible_family_count = sum(
        support >= 50
        for families in _ontology_semantic_supports(ONTOLOGY).values()
        for support in families.values()
    )
    # Every domain is a previous domain in three of the four frozen rotations.
    assert len(semantic_pairs) == 3 * len(SEEDS) * (len(METHODS) - 1) * eligible_family_count
    assert len(tables["semantic_domain_units.csv"]) == 4 * 3 * 3 * 2
    assert len(tables["sequence_seed_units.csv"]) == 4 * 3 * 2
    assert all(row["domain_position"] <= 3 for row in semantic_pairs)
    source_outcome = next(
        row
        for row in outcomes
        if row["family_level"] == "SEMANTIC"
        and row["method"] == "naive_ft"
        and row["sequence"] == "U-T-C-B"
        and row["seed"] == 42
        and row["domain_position"] == 1
    )
    assert source_outcome["maximum_lifecycle_event"] == "source_initial"
    assert source_outcome["maximum_recall"] == source_outcome["learned_recall"]
    later_outcome = next(
        row
        for row in outcomes
        if row["family_level"] == "SEMANTIC"
        and row["method"] == "naive_ft"
        and row["sequence"] == "U-T-C-B"
        and row["seed"] == 42
        and row["domain_position"] == 2
    )
    assert later_outcome["maximum_lifecycle_event"] == "post_adapt"
    er_pair = next(row for row in semantic_pairs if row["replay_method"] == "er")
    assert Fraction(er_pair["forgetting_reduction_exact"]) > 0
    assert er_pair["final_competence_gain_exact"] == er_pair["forgetting_reduction_exact"]
    assert er_pair["bwt_gain_exact"] == er_pair["forgetting_reduction_exact"]
    assert er_pair["naive_forgetting_breach"] == 1
    assert er_pair["replay_forgetting_breach"] == 0  # strict > 0.10
    assert er_pair["forgetting_breach_reduction"] == 1
    secondary = [
        row
        for row in tables["method_dimension_summary.csv"]
        if row["analysis_layer"] == ALL_ORDER_LAYER
        and row["replay_method"] == "er"
        and row["dimension"]
        in {
            "BACKWARD_TRANSFER_GAIN_SECONDARY",
            "FORGETTING_OVER_0_10_REDUCTION_SECONDARY",
        }
    ]
    assert len(secondary) == 2
    assert {row["verdict"] for row in secondary} == {"NOT_APPLICABLE_DESCRIPTIVE_ONLY"}
    assert {row["n_sequence_seed_units"] for row in secondary} == {12}
    assert verdicts["prospective_extension_verdict.json"]["overall_verdict"] == SUPPORTED
    assert verdicts["all_order_synthesis_verdict.json"]["overall_verdict"] == SUPPORTED
    assert verdicts["h7_verdict.json"]["overall_h7_verdict"] == SUPPORTED


def _domain_row(position: int, effect: Fraction, family_count: int) -> dict[str, object]:
    exact = f"{effect.numerator}/{effect.denominator}"
    zero = "0/1"
    return {
        "evidence_layer": EXISTING_EVIDENCE,
        "sequence": "U-T-C-B",
        "seed": 42,
        "previous_domain": ("U", "T", "C")[position - 1],
        "domain_position": position,
        "replay_method": "er",
        "eligible_family_count": family_count,
        "naive_mean_forgetting": float(effect),
        "naive_mean_forgetting_exact": exact,
        "replay_mean_forgetting": 0.0,
        "replay_mean_forgetting_exact": zero,
        "forgetting_reduction": float(effect),
        "forgetting_reduction_exact": exact,
        "naive_mean_final_competence": 0.0,
        "naive_mean_final_competence_exact": zero,
        "replay_mean_final_competence": float(effect),
        "replay_mean_final_competence_exact": exact,
        "final_competence_gain": float(effect),
        "final_competence_gain_exact": exact,
        "naive_mean_bwt": 0.0,
        "naive_mean_bwt_exact": zero,
        "replay_mean_bwt": float(effect),
        "replay_mean_bwt_exact": exact,
        "bwt_gain": float(effect),
        "bwt_gain_exact": exact,
        "naive_forgetting_breach_rate": 0.0,
        "replay_forgetting_breach_rate": 0.0,
        "forgetting_breach_rate_reduction": 0.0,
        "forgetting_breach_rate_reduction_exact": zero,
    }


def test_domain_then_sequence_aggregation_does_not_weight_family_rows() -> None:
    units = _derive_sequence_seed_units(
        [
            _domain_row(1, Fraction(1), 100),
            _domain_row(2, Fraction(0), 1),
            _domain_row(3, Fraction(0), 1),
        ]
    )
    assert len(units) == 1
    assert units[0]["forgetting_reduction_exact"] == "1/3"
    assert units[0]["final_competence_gain_exact"] == "1/3"


def test_prefrozen_cell_and_layer_verdict_logic() -> None:
    tables, _verdicts = _derive_all(_complete_trajectory())
    sequence_units = copy.deepcopy(tables["sequence_seed_units.csv"])
    for row in sequence_units:
        if row["sequence"] == "T-C-B-U":
            row["forgetting_reduction_exact"] = "-1/10"
            row["forgetting_reduction"] = -0.1
        else:
            row["forgetting_reduction_exact"] = "1/10"
            row["forgetting_reduction"] = 0.1
        row["final_competence_gain_exact"] = "0/1"
        row["final_competence_gain"] = 0.0
    order = _derive_order_summaries(sequence_units)
    method = _derive_method_dimension_summaries(
        sequence_units, order, tables["semantic_domain_units.csv"]
    )
    all_order = _layer_verdict(
        layer=ALL_ORDER_LAYER,
        domain_units=tables["semantic_domain_units.csv"],
        sequence_units=sequence_units,
        method_summaries=method,
    )
    forgetting_cells = [
        row
        for row in all_order["method_outcome_subclaims"]
        if row["dimension"] == FORGETTING_DIMENSION
    ]
    competence_cells = [
        row
        for row in all_order["method_outcome_subclaims"]
        if row["dimension"] == COMPETENCE_DIMENSION
    ]
    assert {row["verdict"] for row in forgetting_cells} == {PARTIALLY_SUPPORTED}
    assert {row["verdict"] for row in competence_cells} == {NOT_SUPPORTED}
    assert {row["minimum_effect_exact"] for row in forgetting_cells} == {"-1/10"}
    assert {row["maximum_effect_exact"] for row in forgetting_cells} == {"1/10"}
    assert all_order["overall_verdict"] == PARTIALLY_SUPPORTED
    assert all_order["order_robustness"]["verdict"] == NOT_SUPPORTED
    assert _cell_verdict(Fraction(0), [Fraction(1)]) == NOT_SUPPORTED


def test_incomplete_domain_makes_both_required_layers_and_h7_not_assessed() -> None:
    tables, _verdicts = _derive_all(_complete_trajectory())
    domains = [
        row
        for row in tables["semantic_domain_units.csv"]
        if not (
            row["sequence"] == "T-C-B-U"
            and row["seed"] == 42
            and row["replay_method"] == "er"
            and row["domain_position"] == 1
        )
    ]
    sequence = _derive_sequence_seed_units(domains)
    order = _derive_order_summaries(sequence)
    method = _derive_method_dimension_summaries(sequence, order, domains)
    prospective = _layer_verdict(
        layer=PROSPECTIVE_LAYER,
        domain_units=domains,
        sequence_units=sequence,
        method_summaries=method,
    )
    all_order = _layer_verdict(
        layer=ALL_ORDER_LAYER,
        domain_units=domains,
        sequence_units=sequence,
        method_summaries=method,
    )
    h7 = _h7_verdict(prospective, all_order)
    assert prospective["overall_verdict"] == NOT_ASSESSED_INCOMPLETE
    assert all_order["overall_verdict"] == NOT_ASSESSED_INCOMPLETE
    assert h7["overall_h7_verdict"] == NOT_ASSESSED_INCOMPLETE


def test_physical_support_and_full_pair_provenance_mismatches_fail_closed() -> None:
    support_changed = copy.deepcopy(_complete_trajectory())
    target = next(
        row
        for row in support_changed
        if row["method"] == "er"
        and row["sequence"] == "U-T-C-B"
        and row["seed"] == 42
        and row["evaluated_domain"] == "U"
        and row["family_level"] == "NATIVE"
        and row["lifecycle_event"] == "source_initial"
    )
    changed_support = int(target["support"]) + 1
    target["support"] = changed_support
    target["fn"] = changed_support - int(target["tp"])
    target["recall"] = int(target["tp"]) / changed_support
    target["wilson95_low"], target["wilson95_high"] = wilson_interval(
        int(target["tp"]), changed_support
    )
    with pytest.raises(ValueError, match="TASK-007 permanent holdout"):
        _normalise_trajectory(support_changed, ONTOLOGY)

    range_changed = copy.deepcopy(_complete_trajectory())
    for row in range_changed:
        if (
            row["method"] == "er"
            and row["sequence"] == "U-T-C-B"
            and row["seed"] == 42
            and row["evaluated_domain"] == "U"
        ):
            changed_start = int(row["physical_slice_start"]) + 1
            changed_stop = int(row["physical_slice_stop"]) + 1
            identity = {
                "dataset_fingerprint": row["dataset_fingerprint"],
                "dataset_id": "U",
                "row_start": changed_start,
                "row_stop": changed_stop,
                "stratum": "PERMANENT_HOLDOUT",
            }
            row["physical_slice_start"] = changed_start
            row["physical_slice_stop"] = changed_stop
            row["physical_slice_identity"] = json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            )
            row["physical_slice_digest"] = physical_slice_digest(identity)
    with pytest.raises(ValueError, match="range differs from TASK-007"):
        _normalise_trajectory(range_changed, ONTOLOGY)


def test_lifecycle_event_matrix_and_anchor_indices_fail_closed() -> None:
    wrong_anchor = copy.deepcopy(_complete_trajectory())
    row = next(
        item
        for item in wrong_anchor
        if item["sequence"] == "U-T-C-B"
        and item["seed"] == 42
        and item["method"] == "naive_ft"
        and item["family_level"] == "SEMANTIC"
        and item["lifecycle_event"] == "final"
        and item["evaluated_domain"] == "U"
    )
    row["stage"] = 2
    with pytest.raises(ValueError, match="lifecycle/event-index schedule"):
        _normalise_trajectory(wrong_anchor, ONTOLOGY)

    missing_event = copy.deepcopy(_complete_trajectory())
    del missing_event[
        next(
            index
            for index, item in enumerate(missing_event)
            if item["sequence"] == "U-T-C-B"
            and item["seed"] == 42
            and item["method"] == "naive_ft"
            and item["family_level"] == "NATIVE"
            and item["lifecycle_event"] == "domain_end"
            and item["stage"] == 3
            and item["evaluated_domain"] == "U"
        )
    ]
    with pytest.raises(ValueError, match="lifecycle matrix is incomplete"):
        _normalise_trajectory(missing_event, ONTOLOGY)


def test_missing_complete_eligible_semantic_family_population_fails_closed() -> None:
    rows = [
        row
        for row in _complete_trajectory()
        if not (
            row["evaluated_domain"] == "U" and row["semantic_family"] == "Availability / Impact"
        )
    ]
    # Reconnaissance remains eligible for U, so a weak "at least one family"
    # check would incorrectly allow a verdict from this incomplete population.
    assert any(
        row["evaluated_domain"] == "U"
        and row["family_level"] == "SEMANTIC"
        and row["semantic_family"] == "Reconnaissance / Discovery"
        for row in rows
    )
    with pytest.raises(ValueError, match="eligible semantic family population"):
        _normalise_trajectory(rows, ONTOLOGY)


def test_source_provenance_recomputes_bundles_and_schedule_byte_identity() -> None:
    source = _fake_source()
    path, raw, digest = _load_task009_contract(TASK009_CONTRACT)
    contract_record = _task009_contract_record(
        path=path,
        raw=raw,
        contract_sha256=digest,
        ontology_path=ONTOLOGY_PATH,
        ontology=ONTOLOGY,
    )
    _validate_source_artifacts_record(source, contract_record, ONTOLOGY.dataset_fingerprints)

    bad_bundle = copy.deepcopy(source)
    bad_bundle["runs"][0]["validated_bundle_digest"] = "0" * 64
    with pytest.raises(ValueError, match="bundle digest differs"):
        _validate_source_artifacts_record(
            bad_bundle, contract_record, ONTOLOGY.dataset_fingerprints
        )

    bad_schedule = copy.deepcopy(source)
    run = next(
        item
        for item in bad_schedule["runs"]
        if item["task009_role"] == "ADAPTIVE_RUN"
        and item["sequence"] == ["T", "C", "B", "U"]
        and item["seed"] == 42
        and item["method"] == "er"
    )
    run["validated_file_digests"]["supervision_schedule.json"] = "f" * 64
    run["validated_bundle_digest"] = _digest(run["validated_file_digests"])
    with pytest.raises(ValueError, match="schedule byte identity"):
        _validate_source_artifacts_record(
            bad_schedule, contract_record, ONTOLOGY.dataset_fingerprints
        )

    bad_run_identity = copy.deepcopy(source)
    run = next(
        item
        for item in bad_run_identity["runs"]
        if item["task009_role"] == "ADAPTIVE_RUN"
        and item["sequence"] == ["T", "C", "B", "U"]
        and item["seed"] == 42
        and item["method"] == "er"
    )
    run["source_identity"][1] = "f" * 64
    with pytest.raises(ValueError, match="paired source-model identity"):
        _validate_source_artifacts_record(
            bad_run_identity, contract_record, ONTOLOGY.dataset_fingerprints
        )

    bad_block_identity = copy.deepcopy(source)
    bad_block_identity["extension_pairing_blocks"][0]["source_identity"][1] = "f" * 64
    with pytest.raises(ValueError, match="pairing block source-model identity"):
        _validate_source_artifacts_record(
            bad_block_identity, contract_record, ONTOLOGY.dataset_fingerprints
        )


def test_adaptive_resolved_config_drift_is_rejected(tmp_path: Path) -> None:
    sequence = ("T", "C", "B", "U")
    seed = 42
    method = "er"
    config = {
        "experiment_id": _experiment_id(method, sequence, seed),
        "study": "E2",
        "seed": seed,
        "datasets": {"sequence": list(sequence), "split_version": "task001-v1"},
        "stream": {"window_size": 50_000, "boundary_mode": "boundary_aware_control"},
        "splits": {
            "initial": {"train": 0.6, "validation": 0.2, "holdout": 0.2},
            "later": {"online": 0.8, "holdout": 0.2},
        },
        "supervision": {
            "label_budget_per_later_domain": 100,
            "label_delay_windows": 1,
            "schedule": "first_window_uniform",
        },
        "operating_envelope": {"target_fpr": 0.001},
        "adaptation": {
            "method": method,
            "optimizer": "AdamW",
            "learning_rate": 0.0002,  # resealed drift from the frozen 1e-4
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 20,
            "ewc_lambda": 100.0,
        },
        "memory": {"replay_per_domain": 400, "audit_per_domain": 0},
        "materialization": {"cache_root": "data/materialized", "csv_chunk_rows": 100_000},
        "initial_run": f"runs/E1_STATIC_MLP_{'-'.join(sequence)}_s{seed}",
        "supervision_schedule_path": "schedule.json",
        "smoke": None,
        "device": "cpu",
    }
    config_path = tmp_path / "config.resolved.yaml"
    provenance_path = tmp_path / "provenance.json"
    schedule_path = tmp_path / "supervision_schedule.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    provenance_path.write_text("{}\n", encoding="utf-8")
    schedule_path.write_text("{}\n", encoding="utf-8")
    digests = {
        path.name: _file_sha256(path) for path in (config_path, provenance_path, schedule_path)
    }
    with pytest.raises(ValueError, match="resolved config differs from frozen TASK-009"):
        _validated_pipeline_contract(
            root=tmp_path,
            metadata={"validated_file_digests": digests},
            sequence=sequence,
            seed=seed,
            method=method,
        )


def test_exact_write_once_bundle_and_deterministic_self_validation(tmp_path: Path) -> None:
    task009_path, raw, digest = _load_task009_contract(TASK009_CONTRACT)
    source = _fake_source()
    output = tmp_path / "study5b"
    written = _write_evaluation_bundle(
        output=output,
        task009_path=task009_path,
        task009_raw=raw,
        task009_sha=digest,
        ontology_path=ONTOLOGY_PATH,
        ontology=ONTOLOGY,
        source=source,
        trajectory=_complete_trajectory(),
    )
    assert written == output
    assert {path.name for path in output.iterdir() if path.is_file()} == ALL_OUTPUT_FILES
    validate_study5b_evaluation(output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_evaluation_bundle(
            output=output,
            task009_path=task009_path,
            task009_raw=raw,
            task009_sha=digest,
            ontology_path=ONTOLOGY_PATH,
            ontology=ONTOLOGY,
            source=source,
            trajectory=_complete_trajectory(),
        )

    original_contract = (output / "task009_contract.json").read_text(encoding="utf-8")
    corrupt_contract = json.loads(original_contract)
    corrupt_contract["analysis"]["family_to_domain"] = "FLOW_WEIGHTED"
    _write_json(output / "task009_contract.json", corrupt_contract)
    files = _tree_digests(output)
    _write_json(
        output / "artifact_manifest.json",
        {"version": EVALUATOR_VERSION, "files": files, "bundle_digest": _digest(files)},
    )
    with pytest.raises(ValueError, match="estimand/aggregation/verdict semantics"):
        validate_study5b_evaluation(output)
    (output / "task009_contract.json").write_text(original_contract, encoding="utf-8")

    # Resealing the manifest cannot conceal a changed derived table.
    outcomes = output / "family_outcomes.csv"
    outcomes.write_text(outcomes.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    files = _tree_digests(output)
    _write_json(
        output / "artifact_manifest.json",
        {"version": EVALUATOR_VERSION, "files": files, "bundle_digest": _digest(files)},
    )
    with pytest.raises(ValueError, match="deterministic recomputation"):
        validate_study5b_evaluation(output)


def test_contract_record_declares_versions_and_exact_output_set() -> None:
    path, raw, digest = _load_task009_contract(TASK009_CONTRACT)
    record = _task009_contract_record(
        path=path,
        raw=raw,
        contract_sha256=digest,
        ontology_path=ONTOLOGY_PATH,
        ontology=ONTOLOGY,
    )
    assert record["contract_version"] == CONTRACT_VERSION
    assert record["evaluator_version"] == EVALUATOR_VERSION
    assert record["output_files"] == sorted(ALL_OUTPUT_FILES)
    assert len(ALL_OUTPUT_FILES) == 15
