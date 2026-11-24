"""Reject an unregistered object or a foreign Windows return before scientific data is opened."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t0_reconstruct as jobs


def test_source_is_not_opened_for_object_outside_frozen_roster(tmp_path, monkeypatch):
    def pointer(run, member):
        return {
            "run_id": run,
            "manifest_sha256": "a" * 64,
            "member": "payload/" + member,
            "sha256": "b" * 64,
        }

    bridge = pointer("bridge", "bridge.json")
    freeze = pointer("freeze", "freeze.json")
    audit = pointer("audit", "reconstruction-readback.json")
    source = pointer("source", "source-provenance.json")
    docs = {
        "bridge": {"schema": "el-t0-recovery-bridge-1", "recipe_sha256": jobs.RECIPE},
        "freeze": {
            "schema": "el-t0-freeze-1",
            "object_ids": ["set2"],
            "conditions": ["M2", "M4"],
            "run_id": "freeze",
            "target_ids": [f"v{i:02d}" for i in range(1, 7)],
            "recipe_sha256": jobs.RECIPE,
            "stop_step": 5000,
            "target_reference_permission": "closed",
        },
        "audit": {
            "schema": "el-t0-reconstruction-readback-1",
            "forced_fallback_fixture": True,
            "verified": True,
            "completed_steps": 5000,
        },
    }
    for k, v in docs.items():
        (tmp_path / k).write_bytes(jobs.encoded(v))

    def exact_member(store, pin, name=None):
        assert pin["run_id"] in docs, "unauthorized source was opened"
        return tmp_path / pin["run_id"]

    monkeypatch.setattr(jobs, "member", exact_member)
    with pytest.raises(ValueError, match="Unregistered T0 object"):
        jobs.prepare(tmp_path, bridge, source, "set31", "M2", Path("python"), freeze, audit)


def test_return_summary_for_other_package_rejected_without_installing_an_archive(tmp_path, monkeypatch):
    trial = "t0-frozen-set2-M2"
    returns = tmp_path / "returns" / trial
    returns.mkdir(parents=True)
    ticket = {"package": {"sha256": "a" * 64}, "job_sha256": "c" * 64, "trial_id": trial}
    (tmp_path / "ticket").write_bytes(jobs.encoded(ticket))
    (tmp_path / "ready").write_bytes(jobs.encoded({"trial_id": trial}))
    (returns / "completed.json").write_bytes(
        jobs.encoded(
            {
                "schema": "el-t0-windows-return-1",
                "trial_id": trial,
                "package_sha256": "d" * 64,
                "job_sha256": "c" * 64,
            }
        )
    )
    monkeypatch.setattr(
        jobs, "member", lambda root, pin, name=None: tmp_path / ("ready" if name else "ticket")
    )
    store = tmp_path / "empty-store"
    with pytest.raises(ValueError, match="selected ticket"):
        jobs.accept(store, {}, returns, jobs.sha(returns / "completed.json"), Path("python"))
    assert not store.exists()
