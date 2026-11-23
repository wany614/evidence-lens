"""The upstream optimizer must not start an unreviewed real-object job."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evidence_lens._bundle import BundleError
from recovery_dispatch import dispatch


@pytest.mark.parametrize(("object_id", "condition", "missing"), [
    ("set6", "M2", "freeze"),
    ("set1", "M4", "control_review"),
    ("scene_057_offset_bell_cage", "M4", "worker_source"),
])
def test_rejects_missing_scientific_gate_before_reading_any_inputs(tmp_path, object_id, condition, missing):
    roles = {"registration", "handoff", "initialization", "admission", "recipe", "worker_source"}
    roles.update({"freeze"} if object_id == "set6" else {"control_review"} if object_id == "set1" else set())
    paths = {k: tmp_path / k for k in roles - {missing}}
    with pytest.raises(BundleError, match="scientific evidence roster"):
        dispatch({"inputs": {}, "object_id": object_id, "condition": condition}, paths, tmp_path)


def test_rejects_target_reference_in_positive_control_ticket(tmp_path):
    roles = {"registration", "handoff", "initialization", "admission", "recipe", "worker_source", "reference"}
    with pytest.raises(BundleError, match="scientific evidence roster"):
        dispatch({"inputs": {}, "object_id": "scene_057_offset_bell_cage", "condition": "M4"},
                 {k: tmp_path / k for k in roles}, tmp_path)
