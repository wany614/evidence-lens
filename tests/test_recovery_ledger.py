"""Check the arithmetic and failure boundaries of the resource-only reader."""

from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evidence_lens._bundle import BundleError, asset_path, sha
from recovery_fit_curves import fit_measurements, parse_events, read_steps
from recovery_ledger import receipt_pins


def mutation(op, before, selected, after):
    return {
        "operation": op,
        "before": before,
        "selected": selected,
        "affected_values": selected,
        "after": after,
        "adam_moments_checked": True,
    }


def test_topology_counts_are_row_transitions_and_adam_witness_is_required():
    events = [
        mutation("prune", 20, 2, 18),
        {"operation": "growth_budget", "candidates": 9, "available": 5, "admitted": 5, "blocked": 4},
        mutation("clone", 18, 3, 21),
        mutation("split", 21, 2, 23),
        mutation("reset", 23, 23, 23),
    ]
    row = {"n_before": 20, "n_after": 23, "topology": events}
    actual, checks = parse_events(row)
    assert checks == 4
    assert [e["event_index"] for e in actual] == list(range(5))
    events[2]["adam_moments_checked"] = False
    with pytest.raises(BundleError):
        parse_events(row)
    events[2]["adam_moments_checked"] = True
    events[3]["after"] = 25
    with pytest.raises(BundleError):
        parse_events(row)


def test_partial_optimization_keeps_complete_updates_and_counts_truncated_tail(tmp_path):
    raw = tmp_path / "steps.jsonl"
    raw.write_bytes(b'{"step":1}\n{"st')
    assert read_steps(raw, allow_partial=True) == ([{"step": 1}], 4)
    with pytest.raises(BundleError):
        read_steps(raw, allow_partial=False)
    raw.write_bytes(b'{"step":1}\ninvalid\n{"step":2}\n')
    with pytest.raises(BundleError):
        read_steps(raw, allow_partial=True)


def test_failed_optimizer_verifies_shared_recipe_and_keeps_its_source_counts(tmp_path):
    root = tmp_path / "run"
    (root / "payload").mkdir(parents=True)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"topology": {"start": 3, "stop": 8, "cap": 100}}))
    digest = sha(recipe)
    shared = asset_path(tmp_path, digest)
    shared.parent.mkdir(parents=True)
    shared.write_bytes(recipe.read_bytes())
    (root / "inputs.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "role": "recipe",
                        "storage": "shared",
                        "files": [{"sha256": digest, "bytes": recipe.stat().st_size}],
                    }
                ]
            }
        )
    )
    identity = {
        "recipe_sha256": digest,
        "object_id": "set1",
        "condition": "M2",
        "generator_source_ids": ["s00", "s03"],
    }
    meta = {
        "run_id": "fixture",
        "experiment_id": "recovery-fit",
        "trial_id": "set1",
        "execution_status": "failed",
    }
    update = {
        "step": 1,
        "phase": "warmup",
        "source": "s00",
        "source_counts": {"s00": 1, "s03": 0},
        "n_before": 12,
        "n_after": 12,
        "loss": 0.5,
        "display_l1": 0.4,
        "pixels": 256**2,
        "visible": 5,
        "max_radius_px": 8.0,
        "max_abs_visible_means2d_px": 150.0,
        "ssim": 0.3,
        "topology": [],
    }
    (root / "payload/steps.jsonl").write_text(json.dumps(update) + '\n{"step')
    detail = fit_measurements(root, meta, {"science": identity}, {}, tmp_path)
    assert detail["summary"]["updates_this_process"] is None
    assert detail["summary"]["complete_logged_updates"] == 1
    assert detail["steps"][0]["s03_updates"] == 0 and detail["steps"][0]["s01_updates"] is None
    assert detail["summary"]["trace_partial_tail_bytes"] == 6
    shared.write_bytes(shared.read_bytes().replace(b"100", b"101"))
    with pytest.raises(BundleError, match="recipe bytes differ"):
        fit_measurements(root, meta, {"science": identity}, {}, tmp_path)


def test_archive_receipt_links_the_original_manifest(tmp_path):
    parent = tmp_path / "receipts/archive"
    parent.mkdir(parents=True)
    rid, digest = "test-record", "a" * 64
    doc = {
        "kind": "archive",
        "receipt_id": "s",
        "run_id": rid,
        "manifest_sha256": digest,
        "created_at_utc": "2026-11-23T20:00:00Z",
    }
    (parent / "s.json").write_text(json.dumps(doc))
    chosen = {rid: {"manifest_sha256": digest}}
    assert len(receipt_pins(tmp_path, "archive", chosen, "2026-11-23T21:00:00Z")) == 1
    chosen[rid]["manifest_sha256"] = "b" * 64
    with pytest.raises(BundleError, match="another kind or seal"):
        receipt_pins(tmp_path, "archive", chosen, "2026-11-23T21:00:00Z")
