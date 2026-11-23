"""Checks for the portable, numerical reconstruction-return boundary."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recovery_receive as intake


def trajectory(condition="M2"):
    ids = intake.SETS[condition] + intake.TARGETS
    return {
        "completed_steps": 5000,
        "schedule_steps": 5000,
        "status": "completed",
        "checkpoints": {
            str(s): {
                "step": s,
                "gaussians": 37,
                "phase": "frozen",
                "checkpoint_sha256": "a" * 64,
                "views": dict.fromkeys(ids),
            }
            for s in intake.STEPS
        },
    }


def test_successful_trajectory_requires_all_checkpoint_sources_and_targets():
    index = trajectory()
    intake.check_grid(index, "M2", 5000)
    index["checkpoints"]["500"]["views"].pop("v06")
    with pytest.raises(ValueError, match="roster"):
        intake.check_grid(index, "M2", 5000)
    index = trajectory()
    index["checkpoints"].pop("3000")
    with pytest.raises(ValueError, match="incomplete"):
        intake.check_grid(index, "M2", 5000)
    index = trajectory()
    with pytest.raises(ValueError, match="endpoint"):
        intake.check_grid(index, "M2", 1500)


def test_truncation_is_kept_with_actual_reason():
    index = trajectory()
    index.update(status="truncated", completed_steps=1754, failure_reason="registered wall time limit")
    for step in ("3000", "5000"):
        index["checkpoints"].pop(step)
    intake.check_grid(index, "M2", 5000)
    index.pop("failure_reason")
    with pytest.raises(ValueError, match="reason"):
        intake.check_grid(index, "M2", 5000)


def test_native_readback_preserves_unknown_and_rejects_quantized_rgb(tmp_path):
    rgb = np.full((256, 256, 3), 0.3, dtype=np.float32)
    alpha = np.ones((256, 256), dtype=np.float32)
    depth = np.full((256, 256), 2, dtype=np.float32)
    alpha[0, 0] = 0
    depth[0, 0] = 0
    depth[1, 1] = np.nan
    rgb[4, 2, 0] = np.nan
    p = tmp_path / "native.npz"
    np.savez_compressed(p, rgb=rgb, z=depth, alpha=alpha)
    stats = intake.authenticate_native(p)
    assert stats == {
        "pixels": 65536,
        "finite_display_rgb": 65535,
        "valid_opacity": 65536,
        "positive_finite_depth": 65534,
        "positive_opacity_pixels": 65535,
        "zero_opacity_pixels": 1,
        "zero_opacity_non_nan_depth": 1,
        "nonzero_opacity_invalid_depth": 1,
    }
    np.savez_compressed(p, rgb=np.zeros_like(rgb, dtype=np.uint8), z=depth, alpha=alpha)
    with pytest.raises(ValueError, match="precision"):
        intake.authenticate_native(p)


def test_checkpoint_geometry_is_bound_to_array_content_and_dimensions(tmp_path):
    path = tmp_path / "scene.npz"
    n = 7
    data = {
        field: np.ones((n, *tail), dtype=np.float32)
        for field, tail in zip(intake.PARAMS, [(3,), (3,), (4,), (), (3,)], strict=True)
    }
    np.savez(path, **data)
    first = intake.native_gaussians(path, n)
    data["means"][2, 0] += 0.5
    np.savez(path, **data)
    second = intake.native_gaussians(path, n)
    assert first != second
    with pytest.raises(ValueError, match="Gaussian parameter"):
        intake.native_gaussians(path, n + 1)


def test_unlisted_prediction_cannot_substitute_a_listed_file(tmp_path):
    folder = tmp_path / "payload"
    folder.mkdir()
    p = folder / "v01.npz"
    p.write_bytes(b"original")
    spec = {"path": p.name, **intake.signature(p)}
    assert intake.contained_asset(tmp_path, folder, spec, {"payload/v01.npz": dict(spec)}) == p
    with pytest.raises(ValueError):
        intake.contained_asset(tmp_path, folder, {**spec, "path": "../secrets"}, {})
    p.write_bytes(b"replaced")
    with pytest.raises(ValueError):
        intake.contained_asset(tmp_path, folder, spec, {"payload/v01.npz": dict(spec)})


def test_real_recorded_archive_and_return_receipt_are_bound_and_idempotent(tmp_path, capsys):
    import json
    from evidence_lens import recorder
    from evidence_lens.archive import pack

    # This small fixture uses two separate stores and the production recorder, packer and receiver.
    source = tmp_path / "code"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "control.py").write_text("print('synthetic unit')\n")
    config = tmp_path / "fixture.json"
    config.write_text("{}")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"local_free_floor_bytes": 0, "local_active_batch_target_bytes": 25_000_000})
    )
    before = tmp_path / "producer"
    after = tmp_path / "receiver"
    returns = tmp_path / "returns"
    returns.mkdir()
    (tmp_path / "cold-shelf/catalog").mkdir(parents=True)
    rc = recorder.main(
        [
            "--root",
            str(before),
            "--source-root",
            str(source),
            "--config",
            str(config),
            "--storage-policy",
            str(policy),
            "--kind",
            "archive_verify",
            "--experiment",
            "fixture-transport",
            "--trial",
            "synthetic-transport",
            "--",
            "{python}",
            "{source}/scripts/control.py",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    rid = result["run_id"]
    archive = returns / (rid + ".zip")
    pack(before / "runs" / rid, archive, before)
    receipt = returns / (rid + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "schema": "el-worker-return-1",
                "run": rid,
                "run_kind": "archive_verify",
                "trial_id": "synthetic-transport",
                "status": "succeeded",
                "package": archive.name,
                "manifest_sha256": result["manifest_sha256"],
                "sha256": intake.sha(archive),
                "bytes": archive.stat().st_size,
            }
        )
    )
    args = (after, returns, receipt, intake.sha(receipt), intake.sha(archive), policy)
    proof = intake.read_return(*args)
    assert (
        proof["status"] == "succeeded"
        and proof["local_transfer"]["status"] == "accepted"
        and intake.read_return(*args) == proof
    )
    with pytest.raises(ValueError, match="receipt"):
        intake.read_return(after, returns, receipt, "0" * 64, intake.sha(archive), policy)


def test_final_regression_can_return_only_the_frozen_render_endpoint():
    index = trajectory()
    index["completed_steps"] = 1500
    index["checkpoints"] = {"1500": index["checkpoints"]["1500"]}
    intake.check_grid(index, "M2", 1500, full_curve=False)
    with pytest.raises(ValueError, match="incomplete"):
        intake.check_grid(index, "M2", 1500, full_curve=True)


def test_regression_source_requires_actual_completed_freeze_before_originals(tmp_path, monkeypatch):
    import json
    import recovery_inputs
    from argparse import Namespace

    original = {"run_id": "20260101T000000000000Z-aaaaaaaaaaaaaaaa", "manifest_sha256": "a" * 64}
    p = tmp_path / "freeze.json"
    reg = {
        **recovery_inputs.REG,
        "member": "payload/registration.json",
        "sha256": "b8d8fa5b1e59e0b521b14a48cadf0728b5c9ef3f316cf617cac3fb0391f5be40",
    }
    record = {
        "schema": "el-recovery-final-freeze-1",
        "run_id": original["run_id"],
        "registration": reg,
        "protocol_sha256": recovery_inputs.STUDY,
        "authorized_new_conditions": {"set6": ["M2", "M4"]},
        "registered_selection": {"run_id": "fixture"},
        "risk_fits": 0,
        "risk_recalibrations": 0,
        "schedule_steps": 5000,
        "stop_step": 1500,
        "recipe_sha256": "e" * 64,
    }

    def supplied(content):
        p.write_text(json.dumps(content))
        return {**original, "member": "payload/freeze.json", "sha256": intake.sha(p)}

    monkeypatch.setattr(recovery_inputs, "member", lambda store, pin, member: p)
    assert recovery_inputs.validate_regression_freeze(tmp_path, supplied(record)) == original
    with pytest.raises(ValueError, match="complete recipe"):
        recovery_inputs.validate_regression_freeze(tmp_path, supplied({**record, "stop_step": 256}))
    with pytest.raises(ValueError, match="complete recipe"):
        recovery_inputs.validate_regression_freeze(
            tmp_path, supplied({**record, "registration": {**reg, "sha256": "c" * 64}})
        )
    monkeypatch.setattr(
        recovery_inputs,
        "source_role",
        lambda *args: pytest.fail("original input opened without final freeze"),
    )
    a = Namespace(project=tmp_path, store=tmp_path, object_id="set6", condition="M2", recipe_freeze=None)
    with pytest.raises(ValueError, match="separately sealed"):
        recovery_inputs.record(a)
