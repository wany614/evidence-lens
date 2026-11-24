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


def test_verified_external_history_replay_and_tampered_original(tmp_path, capsys):
    from evidence_lens import archive, recorder
    from evidence_lens._bundle import encoded, load, sha
    from t0_reconstruct_intake import historical_roots, receive_cross_store

    old = tmp_path / "old"
    store = tmp_path / "new"
    win = tmp_path / "worker"
    source = tmp_path / "source/scripts"
    source.mkdir(parents=True)
    (source / "fixture.py").write_text("from pathlib import Path\nimport sys\nPath(sys.argv[1], 'x.txt').write_text('fixture')\n")
    config = tmp_path / "fixture.json"
    config.write_text("{}")

    def record(root, trial, parent=None):
        refs = ["--parent", parent["run_id"] + "=" + parent["manifest_sha256"]] if parent else []
        rc = recorder.main(["--root", str(root), "--source-root", str(source.parent),
                            "--config", str(config), "--kind", "archive_verify", "--experiment", "fixture-t0",
                            "--trial", trial, *refs, "--", "{python}", "{source}/scripts/fixture.py", "{out}"])
        assert rc == 0
        entry = __import__("json").loads(capsys.readouterr().out)
        return {k: entry[k] for k in ("run_id", "manifest_sha256")}

    ancestor = record(old, "old-anchor")
    dispatch = record(store, "local-dispatch", ancestor)
    freeze = {"run_id": dispatch["run_id"], "manifest_sha256": dispatch["manifest_sha256"],
              "member":"payload/x.txt","sha256":sha(store / "runs" / dispatch["run_id"] / "payload/x.txt")}
    reg = store / "transport/Handoffs/t0-v1/frozen-12.json"
    reg.parent.mkdir(parents=True)
    reg.write_bytes(encoded({"freeze":freeze, "jobs":[{"ticket":dispatch}]}))
    proof = jobs.authorize_history(store, old, reg, sha(reg))
    assert load(store / "runs" / proof["run_id"] / proof["member"])["historical_parents"] == {
        ancestor["run_id"]:ancestor["manifest_sha256"]}
    returned = record(win, "remote-original", dispatch)
    z = tmp_path / "original.zip"
    archive.pack(win / "runs" / returned["run_id"], z, win)
    args = (z, store, sha(z), jobs.PROJECT / "configs/storage.json", old, proof)
    assert receive_cross_store(*args, freeze)["status"] == "accepted"
    assert receive_cross_store(*args, freeze)["status"] == "identical"
    with pytest.raises(ValueError, match="different freezes"):
        receive_cross_store(*args, dict(freeze, sha256="b" * 64))
    historical_roots.cache_clear()
    original = old / "runs" / ancestor["run_id"] / "payload/x.txt"
    original.chmod(0o600)
    original.write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        receive_cross_store(*args, freeze)


@pytest.mark.parametrize("index_fails", [False, True])
def test_deferred_capacity_keeps_its_receipt_and_preserves_index_error(tmp_path, monkeypatch, index_fails):
    import zipfile
    import t0_reconstruct_intake as intake
    from evidence_lens._bundle import load, sha

    store = tmp_path / "store"
    store.mkdir()
    package = tmp_path / "original.zip"
    with zipfile.ZipFile(package, "w") as writer:
        writer.writestr("fixture.txt", "original")
    digest = sha(package)
    freeze = {"run_id": "synthetic-freeze"}
    proof = {"run_id": "synthetic-proof", "manifest_sha256": "a" * 64, "sha256": "b" * 64,
             "member": "payload/cross-store-audit.json"}
    monkeypatch.setattr(intake, "historical_roots", lambda *args: (freeze, {}))
    monkeypatch.setattr(intake._capacity, "check", lambda *args: {"reason": "test capacity limit"})
    args = (package, store, digest, jobs.PROJECT / "configs/storage.json", tmp_path, proof, freeze)
    if index_fails:
        def unavailable_index(*args):
            raise OSError("test transfer index unavailable")
        monkeypatch.setattr(intake, "project_receipt", unavailable_index)
        with pytest.raises(OSError, match="test transfer index unavailable"):
            intake.receive_cross_store(*args)
    else:
        assert intake.receive_cross_store(*args)["status"] == "deferred_capacity"
    receipts = [load(f) for f in (store / "receipts/transfer").glob("*.json")]
    assert [(r["status"], r["reason"], r["container_sha256"]) for r in receipts] == [
        ("deferred_capacity", "test capacity limit", digest)]
    assert not list((store / "staging").iterdir()) and not (store / "quarantine").exists()
    assert sha(package) == digest
