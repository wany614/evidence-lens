"""Independent algebraic and role-boundary checks for the separately registered multi-source study."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "prototype/many-source/src"))

from evidence_lens_many.evaluation import ap_and_base, labels, quota, ranking_metrics
from many_data_access import EvaluatorTruth
from many_data_prepare import confirmation_terminals
from many_data_visibility import Intersector, controls
from many_evaluate import scope_metrics
from many_head import balanced_weights, preprocess


def test_equal_scores_threshold_and_missing_error_denominators():
    rgb = np.zeros((252, 378, 3), np.float32)
    pred = rgb.copy()
    tolerance = np.float32(0.05)
    pred[0, 0] = tolerance
    one_ulp_above = np.nextafter(tolerance, np.float32(1))
    pred[0, 1] = np.nextafter(one_ulp_above, np.float32(1))
    pred[0, 4] = one_ulp_above
    pred[0, 2] = np.nan
    rgb[0, 3] = np.nan
    ref, valid, high, mae = labels(pred, rgb)
    assert high[0, :4].tolist() == [False, True, False, False]
    # Summing three equal FP32 channels one ULP above .05 and dividing by three rounds back to .05.
    assert mae[0, 4] == tolerance and not high[0, 4]
    assert ref[:1, :4].tolist() == [[True, True, True, False]]
    assert valid[:1, :4].tolist() == [[True, True, False, False]]
    assert np.isnan(mae[0, 2:4]).all()

    # Whole ties at ranks .8 and .1 give (.5 + .5) / 2 irrespective of the within-group order.
    scores = np.array([0.8, 0.8, 0.1, 0.1])
    for events in ([1, 0, 0, 1], [0, 1, 1, 0]):
        ap, base, reason = ap_and_base(scores, np.array(events, dtype=bool))
        assert ap == base == 0.5 and reason is None
    assert ap_and_base(scores, np.zeros(4, bool)) == (None, 0.0, "no_positive_labels")

    rank = np.array([[0.3, 0.3, 0.3, 0.4, 0.5, np.nan]], np.float32)
    all_valid = np.ones(rank.shape, bool)
    positive = np.array([[False, True, False, False, True, True]])
    scored = np.isfinite(rank)
    q = quota(rank, scored, positive, 20)
    assert q["eligible"] == 5 and q["selected"] == 1 and q["errors"] == 0
    assert q["pixel_indices_sha256"] == hashlib.sha256(np.array([0], dtype="<i4").tobytes()).hexdigest()
    row = ranking_metrics(rank, all_valid, all_valid, positive)
    assert (row["reference_errors"], row["positive"], row["denominator"]) == (3, 2, 5)
    assert all(q["selected"] == 0 and q["error_rate"] is None for q in row["quota"] if q["percent"] == 10)


def test_p1_uses_all_correct_even_query_unknown_or_no_library_witness():
    shape = (1, 5)

    def geometry(comparable):
        return {
            "comparable": np.array([comparable]),
            "incomplete": np.zeros(shape, bool),
            "invalid_geometry": np.zeros(shape, bool),
            "support": np.zeros(shape, np.uint32),
            "depth_conflict": np.zeros(shape, np.uint32),
            "colour_conflict": np.zeros(shape, np.uint32),
            "witness": np.array([[4, 7, -1, 6, 7]]),
        }

    small = geometry([0, 1, 0, 0, 0])
    full = geometry([1, 1, 0, 1, 1])
    valid = np.array([[True, True, True, True, False]])
    high = np.array([[False, False, False, True, False]])
    out = scope_metrics(small, full, valid, high)
    assert (out["all_reference_correct"], out["correct_with_gap"], out["correct_gap_rate"]) == (3, 1, 1 / 3)
    assert out["library_no_comparable"] == 1
    full["incomplete"][:] = True
    assert scope_metrics(small, full, valid, high)["correct_gap_rate"] is None


def test_training_mass_is_per_unit_and_physical_pixel():
    units = np.array([0, 0, 0, 0, 1, 1])
    px = np.array([4, 4, 4, 8, 4, 6])
    w = balanced_weights(units, px)
    assert np.allclose(w, [1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 4, 1 / 4])
    x = np.array([[1.0, np.nan], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0]], np.float32)
    normalized, mean, scale, mass = preprocess(x, w)
    np.testing.assert_allclose(mean, [1, 3])
    np.testing.assert_allclose(scale, [1, 1])
    assert np.max(abs(normalized)) < 1e-14
    np.testing.assert_allclose(mass, [1, 11 / 12])


def test_unified_opening_checks_other_scenes_and_real_matrix_digest():
    protocol = json.loads((ROOT / "protocol/many-source-data-v1.json").read_bytes())
    cases = [*protocol["originals"], *protocol["external"]["roster"]]
    cases = [r for r in cases if r["split"] == "confirmation"]
    allow = {r["id"]: [f"h{i:02d}" for i in range(r.get("H", 6))] for r in cases}
    terminals = {
        r["id"]: {
            k: {
                "status": "failed",
                "native_failure": "native-technical-run",
                "receipt": {"run_id": "terminal-" + k},
            }
            for k in (["16", "128"] if "seed" in r else [str(r["R"])])
        }
        for r in cases
    }
    digest = hashlib.sha256(json.dumps(terminals, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    gate = {
        "schema": "el-many-h-opening-1",
        "phase": "confirmation",
        "allowed_targets": allow,
        "terminals": terminals,
        "prediction_freeze_sha256": digest,
        "model_freeze": {"run_id": "registered-model"},
    }
    assert len(confirmation_terminals(gate, protocol)) == 7
    modified = copy.deepcopy(gate)
    modified["allowed_targets"]["bridge"].pop()
    with pytest.raises(ValueError, match="target roster"):
        confirmation_terminals(modified, protocol)
    modified = copy.deepcopy(gate)
    modified["terminals"]["living_room"]["55"]["native_failure"] = "different"
    with pytest.raises(ValueError, match="different frozen"):
        confirmation_terminals(modified, protocol)


def test_original_segment_controls_including_occlusion_within_one_percent(tmp_path):
    actual = controls(Intersector(ROOT / "scripts/many_data_visibility.cpp", tmp_path))
    assert actual["no_blocker"] and actual["coincident_endpoint"] and actual["near_wall_outside_segment"]
    assert not actual["near_wall"] and not actual["parallel_surface_5cm_in_front"]


def test_same_named_target_in_changed_frame_is_rejected(tmp_path, monkeypatch):
    import many_data_access

    n = 16
    r = [f"r{i:03d}" for i in range(0, 128, 8)]
    h = [f"h{i:02d}" for i in range(6)]
    source = {
        "run_id": "source",
        "manifest_sha256": "a" * 64,
        "member": "payload/source.json",
        "sha256": "1" * 64,
    }
    raw = {"run_id": "source", "manifest_sha256": "a" * 64, "member": "payload/R16.zip", "sha256": "2" * 64}
    k = [[12, 0, 7], [0, 12, 5], [0, 0, 1]]
    cameras = {i: {"K": k, "c2w": np.eye(4).tolist(), "width": 378, "height": 252} for i in r + h}
    native_frame = {"unit": "m", "world_to_deployment": np.eye(4).tolist(), "metric_anchor_ids": []}
    scene = {
        "run_id": "source",
        "scene_id": "test",
        "split": "train",
        "R": r,
        "A": r,
        "H": h,
        "unit": "m",
        "views": cameras,
        "grids": {},
        "frame_receipt": native_frame,
        "original_rgb": {v: {"sha256": "a" * 64} for v in r},
        "source_rgb": {v: {"sha256": "b" * 64} for v in r},
    }
    archive = tmp_path / "R16.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("scene.json", json.dumps(scene))
    index = tmp_path / "source.json"
    index.write_text(
        json.dumps(
            {
                "scene_id": "test",
                "packages": {
                    "R16": {"path": "R16.zip", "sha256": raw["sha256"], "bytes": archive.stat().st_size}
                },
            }
        )
    )
    monkeypatch.setattr(many_data_access, "resolve", lambda roots, p: archive if p == raw else index)
    truth = EvaluatorTruth.__new__(EvaluatorTruth)
    truth.meta = {
        "inputs": {"source": source},
        "scene_id": "test",
        "split": "train",
        "R": [f"r{i:03d}" for i in range(128)],
        "H": h,
    }
    bank = SimpleNamespace(
        reconstruction_ids=r,
        accessible_ids=r,
        target_ids=h,
        meta={
            "scene_id": "test",
            "provenance": {"native_evidence": {"data": raw}, "source_acquisition": "source", "unit": "m"},
            "frame_receipt": dict(native_frame, matrix=native_frame["world_to_deployment"]),
            "metric_anchor_ids": [],
            "grids": {},
            "cameras": copy.deepcopy(cameras),
            "sources": {v: {"original_sha256": "a" * 64, "photo": {"sha256": "b" * 64}} for v in r},
        },
    )
    assert truth.verify_bank(bank, store=tmp_path)["R"] == n
    bank.meta["cameras"]["h00"]["c2w"][0][3] = 0.01
    with pytest.raises(ValueError, match="cameras, grid, frame"):
        truth.verify_bank(bank, store=tmp_path)


def test_failed_fit_and_failed_ranks_keep_different_observable_denominators():
    from many_data_terminal import failed_view, unavailable_domains

    reference = np.zeros((252, 378, 3), np.float32)
    reference[0, 2] = np.nan
    count = 252 * 378 - 1
    missing, ref, valid, high = failed_view(reference, None, None, "f" * 64, "native fit failed")
    assert missing["reference"] == missing["prediction_unavailable"] == count
    assert missing["labelled"] == missing["errors"] == 0
    assert missing["p1"] is missing["p2"] is missing["error_rate"] is None
    prediction = reference.copy()
    prediction[0, 0] = 0.25
    prediction[0, 1] = np.nan
    ranked_missing, ref, valid, high = failed_view(
        reference, {"rgb": prediction}, "a" * 64, "f" * 64, "native score failed"
    )
    assert (
        ranked_missing["labelled"],
        ranked_missing["errors"],
        ranked_missing["prediction_unavailable"],
    ) == (count - 1, 1, 1)
    assert ranked_missing["p1"] is ranked_missing["p2"] is None
    domains = unavailable_domains("h00", 16, ref, valid, high, "native score failed")
    assert len(domains) == (5 * 4 + 2) * 3
    assert all(
        row["denominator"] == 0 and row["reference_errors"] == 1 and row["ap"] is None for row in domains
    )
    assert {r["method"] for r in domains if r["cell"] == "coverage-4"} >= {"MS-L1", "MS-ft"}


def test_failure_cutoff_requires_observed_unrecoverable_budget():
    from many_data_terminal import cutoff_condition

    limits = {
        "max_cpu_core_seconds": 36000,
        "max_gpu_seconds": 28800,
        "new_archive_bytes": 30 << 30,
        "cpu_scoring_rss_bytes": 2 << 30,
    }
    decision = {
        "schema": "el-many-terminal-cutoff-1",
        "family": "scene",
        "R": 16,
        "failed_native_run_id": "unsuccessful-run",
        "final_irrecoverable": True,
        "reason": "observed cutoff during frozen scoring",
        "evidence": {k: "fixture" for k in ("run_id", "member", "manifest_sha256", "sha256")},
        "budget": {
            "resource": "cpu",
            "unit": "core_seconds",
            "cap": 36000,
            "observed_used": 35997,
            "minimum_for_next_valid_attempt": 4,
        },
    }
    assert cutoff_condition(decision, "scene", 16, "unsuccessful-run", limits)["resource"] == "cpu"
    decision["budget"]["minimum_for_next_valid_attempt"] = 2
    with pytest.raises(ValueError, match="remaining budget"):
        cutoff_condition(decision, "scene", 16, "unsuccessful-run", limits)
    decision["budget"] = {
        "resource": "score_rss",
        "unit": "bytes",
        "cap": 2 << 30,
        "observed_used": (2 << 30) - 1,
        "minimum_for_next_valid_attempt": 2 << 30,
    }
    with pytest.raises(ValueError, match="remaining budget"):
        cutoff_condition(decision, "scene", 16, "unsuccessful-run", limits)


def test_independent_order_statistic_matches_row_major_across_dense_ties():
    from many_data_arithmetic import order_quota, hand_labels, mechanism

    x = np.array([[0.3, 0.2, np.nan, 0.1, 0.2], [0.1, 0.2, 0.9, 0.1, 0.2]], np.float32)
    domain = np.isfinite(x)
    for k in (0, 1, 3, 4, 6, 9):
        ids = sorted(np.flatnonzero(domain), key=lambda j: (x.ravel()[j], j))[:k]
        np.testing.assert_array_equal(order_quota(x, domain, k), sorted(ids))
    raw = np.zeros((252, 378, 3), np.uint8)
    p = np.zeros_like(raw, dtype=np.float32)
    p[0, 0] = [0, 0, 0.15]
    p[0, 1] = [0.2, 0, 0]
    p[0, 2] = np.nan
    ref, valid, errors = hand_labels(p, raw)
    assert (
        ref.all()
        and errors[0, :3].tolist() == [False, True, False]
        and valid[0, :3].tolist() == [True, True, False]
    )
    full = {
        "comparable": np.array([[2, 1, 0]]),
        "incomplete": np.zeros((1, 3), bool),
        "invalid_geometry": np.zeros((1, 3), bool),
        "witness": np.array([[1, 0, -1]]),
    }
    cut = dict(full, comparable=np.array([[0, 1, 0]]))
    stats = mechanism(
        cut, full, ["r0", "r1"], ["r0"], np.ones((1, 3), bool), np.zeros((1, 3), bool), ["r1", "r0"]
    )
    assert stats["correct_gap_rate"] == 1 / 3 and stats["witness_source_ids"] == ["r1"]
    with pytest.raises(ValueError, match="gap witness"):
        mechanism(
            cut,
            dict(full, witness=np.array([[0, 0, -1]])),
            ["r0", "r1"],
            ["r0"],
            np.ones((1, 3), bool),
            np.zeros((1, 3), bool),
            ["r1", "r0"],
        )


def test_independent_complete_arithmetic_matches_synthetic_rank_files(tmp_path):
    from many_data_arithmetic import compute_ready
    from evidence_lens_many.evaluation import mask_id

    ids = [f"r{i:03d}" for i in range(20)]
    witness = np.array([[6, 0, 17, 2, 5, 17, 7, 16, 1, 0]], np.int32)
    valid = np.ones(witness.shape, bool)
    high = np.array([[False, True, False, False, True, False, False, False, True, False]])
    rank = np.array([[0.8, 0.0, 0.4, 0.2, 0.0, 0.2, 0.1, 0.5, 0.8, 0.9]], np.float32)
    baseline = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], np.float32)
    queries = {"coverage-4": ids[:4], "fixed-4": ids[4:8], "coverage-16": ids[:16], "coverage-20": ids}
    manifest = {}
    cells = {}
    official = {}
    output = tmp_path / "payload"
    output.mkdir()
    for name, sources in queries.items():
        which = {ids.index(v) for v in sources}
        selected_witness = np.isin(witness, list(which))
        p = output / (name + ".npz")
        arrays = {
            "comparable": selected_witness.astype(np.uint32),
            "incomplete": np.zeros_like(valid),
            "invalid_geometry": np.zeros_like(valid),
            "witness": np.where(selected_witness, witness, -1),
        }
        if name == "coverage-16":
            arrays.update({"rank_B-CPU-LR": rank, "rank_coverage": baseline})
        np.savez_compressed(p, **arrays)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["payload/" + p.name] = {"bytes": p.stat().st_size, "sha256": digest}
        cells[name] = {
            "actual_k": len(sources),
            "queries": sources,
            "sha256": digest,
            "bytes": p.stat().st_size,
            "file": p.name,
        }
        correct = int((~high).sum())
        num = int((~high & ~selected_witness).sum())
        official[name] = {
            "all_reference_correct": correct,
            "correct_with_gap": num,
            "correct_gap_rate": num / correct,
            "audit_complete": True,
        }
    old = {
        "mechanism": official,
        "p1": {
            "coverage4_minus_fixed4": official["coverage-4"]["correct_gap_rate"]
            - official["fixed-4"]["correct_gap_rate"],
            "coverage16_minus_coverage4": official["coverage-16"]["correct_gap_rate"]
            - official["coverage-4"]["correct_gap_rate"],
        },
    }
    first = quota(rank, valid, high, 20)
    second = quota(baseline, valid, high, 20)
    old["p2"] = {
        "common": 10,
        "quota": 2,
        "rule": "coverage",
        "new_errors": first["errors"],
        "rule_errors": second["errors"],
        "new_error_rate": first["error_rate"],
        "rule_error_rate": second["error_rate"],
        "new_minus_rule": first["error_rate"] - second["error_rate"],
        "quota_over_reference": 0.2,
        "common_hash": mask_id(valid),
        "new_selection": first["pixel_indices_sha256"],
        "rule_selection": second["pixel_indices_sha256"],
    }
    owner = SimpleNamespace(files=manifest, open=lambda n: tmp_path / n)
    end = {"views": {"h00": {"cells": cells}}, "A": 20, "R": 20, "source_ids": ids}
    got = compute_ready(
        end,
        owner,
        {"accessible_ids": ids},
        "h00",
        valid,
        high,
        valid,
        {"strong_simple_rule": "coverage"},
        old,
    )
    assert (
        got["P2"]["new_errors"] == 2 and got["P2"]["rule_errors"] == 1 and got["P2"]["new_minus_rule"] == 0.5
    )
    old["p2"]["new_selection"] = "changed"
    with pytest.raises(ValueError, match="different independent P2 membership"):
        compute_ready(
            end,
            owner,
            {"accessible_ids": ids},
            "h00",
            valid,
            high,
            valid,
            {"strong_simple_rule": "coverage"},
            old,
        )
