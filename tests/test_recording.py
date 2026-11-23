import csv
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from evidence_lens import recorder
from evidence_lens import archive
from evidence_lens._bundle import entry, files, load, save, sha
from evidence_lens._verify import verify_bundle


SCRIPT = """import json, os, pathlib, subprocess, sys, time
out, config, source = (pathlib.Path(p) for p in sys.argv[1:4])
assert json.loads(config.read_text())['recording']['artifact_hash'] == 'sha256'
assert not (pathlib.Path.cwd() / 'sensitive.txt').exists()
assert len(os.environ['EL_RUN_ID']) > 10
(out/'value.txt').write_text(source.read_text())
print('fixture output')
print('fixture warning', file=sys.stderr)
if sys.argv[4] == 'partial-camera':
    (out/'cameras.json').write_bytes(b'{"views":')
    sys.exit(7)
if sys.argv[4] == 'illegal-name':
    (out/'bad?.txt').write_bytes(b'original forensic bytes')
    sys.exit(0)
if sys.argv[4] == 'symbolic-output':
    (out/'metrics.jsonl').symlink_to(source)
    sys.exit(0)
if sys.argv[4] == 'fail' or os.environ.get('EL_TEST_TRANSIENT_FAULT') == '1':
    sys.exit(7)
if sys.argv[4] == 'orphan':
    subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'])
    sys.exit(0)
worker = subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.22)'])
worker.wait()
(out/'metrics.jsonl').write_text(json.dumps({'metric_name':'fixture_mean',
    'metric_version':'1', 'unit':'fraction', 'value':0.125, 'numerator':0.375,
    'denominator':3, 'status':'defined'}))
"""


@pytest.fixture
def example(tmp_path):
    source = tmp_path / "project"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "fixture.py").write_text(SCRIPT)
    (source / "sensitive.txt").write_text("outside the source allowlist")
    cfg = tmp_path / "config.json"
    cfg.write_text('{"seed":17}')
    inp = tmp_path / "sample.txt"
    inp.write_text("fixture input")
    return source, cfg, inp


def command_args(example, store, mode="ok", extra=(), kind="scene_render", input_path=None):
    source, cfg, inp = example
    return ["--root", str(store), "--source-root", str(source), "--kind", kind, "--config", str(cfg),
            "--experiment", "fixture", "--trial", "single", "--interval", ".04",
            "--input", f"data={input_path or inp}", *extra, "--", "{python}", "{source}/scripts/fixture.py",
            "{out}", "{config}", "{input:data}", mode]


def launch(example, store, capsys, mode="ok", extra=(), kind="scene_render", input_path=None):
    code = recorder.main(command_args(example, store, mode, extra, kind, input_path))
    captured = capsys.readouterr()
    assert captured.out, captured.err
    result = json.loads(captured.out)
    assert code == result["exit_code"]
    return Path(result["path"])


def test_recorder_success_failure_and_linked_retry(tmp_path, example, capsys, monkeypatch):
    store = tmp_path / "store"
    good = launch(example, store, capsys)
    complete = verify_bundle(good)
    assert complete["run"]["execution_status"] == "succeeded"
    assert sha(good / "payload/value.txt") == sha(example[2])
    assert not (good / "source/sensitive.txt").exists()
    assert (good / "stdout.log").read_text().strip() == "fixture output"
    assert (good / "stderr.log").read_text().strip() == "fixture warning"
    rows = list(csv.DictReader((good / "telemetry.csv").open()))
    assert max(int(r["tree_process_count"]) for r in rows) >= 2
    assert int(complete["run"]["sampled_peak_tree_rss_bytes"]) > 0
    metric = pq.read_table(good / "metrics.parquet").to_pylist()[0]
    assert (metric["run_id"], metric["numerator"]) == (good.name, 0.375)
    seq = [json.loads(s)["seq"] for s in (good / "events.jsonl").read_text().splitlines()]
    assert seq == list(range(1, len(seq)+1))
    monkeypatch.setenv("EL_TEST_TRANSIENT_FAULT", "1")
    failed = launch(example, store, capsys)
    run = verify_bundle(failed)["run"]
    assert (run["execution_status"], run["exit_code"]) == ("failed", 7)
    assert (failed / "payload/value.txt").read_text() == "fixture input"
    monkeypatch.delenv("EL_TEST_TRANSIENT_FAULT")
    retry = launch(example, store, capsys, extra=["--retry-of", failed.name])
    assert retry != failed and verify_bundle(retry)["inputs"]["retry_of"]["run_id"] == failed.name
    excess = launch(example, store, capsys, extra=["--retry-of", failed.name])
    assert excess.parent.name == "quarantine"
    assert "already allocated" in load(excess / "run.json")["failure_reason"]
    assert archive.rebuild(store)["runs"] == 3


def rewrite(bundle, name, raw):
    path = bundle / name
    path.chmod(0o644)
    path.write_bytes(raw)


def reseal(bundle):
    """Make a self-consistent but different candidate for the existing run ID."""
    manifest = load(bundle / "manifest.json")
    manifest["files"] = [entry(p, n) for n, p in files(bundle).items()
                         if n not in {"manifest.json", "completion.json"}]
    (bundle / "manifest.json").chmod(0o644)
    save(bundle / "manifest.json", manifest)
    completion = load(bundle / "completion.json")
    completion["manifest_sha256"] = sha(bundle / "manifest.json")
    (bundle / "completion.json").chmod(0o644)
    save(bundle / "completion.json", completion)


def sql_snapshot(db):
    with sqlite3.connect(db) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: sorted(conn.execute(f"SELECT * FROM {table}").fetchall(), key=repr)
                for table in tables}


def test_archive_transfer_damage_conflict_replay_and_rebuild(tmp_path, example, capsys):
    producer, receiver = tmp_path / "producer", tmp_path / "receiver"
    good = launch(example, producer, capsys)
    first = archive.receive(good, receiver)
    assert first["status"] == "accepted"
    duplicate = archive.receive(good, receiver)
    assert duplicate["status"] == "identical"
    assert duplicate["receipt_id"] != first["receipt_id"]
    for action in ["missing", "changed", "extra", "conflict"]:
        candidate = tmp_path / action
        shutil.copytree(good, candidate)
        if action == "missing":
            loss = candidate / "payload/value.txt"
            loss.chmod(0o644)
            loss.unlink()
        elif action == "extra":
            (candidate / "unregistered.txt").write_text("surprise")
        else:
            rewrite(candidate, "stdout.log", b"changed output\n")
            if action == "conflict":
                reseal(candidate)
                verify_bundle(candidate)  # internally valid with distinct manifest
        rejected = archive.receive(candidate, receiver)
        assert rejected["status"] == "quarantined"
        assert (receiver / rejected["quarantine_path"]).is_dir()
        assert ("conflict" in rejected["reason"]) == (action == "conflict")
        assert verify_bundle(receiver / "runs" / good.name)["manifest_sha256"] == first["manifest_sha256"]

    container = tmp_path / "experiment.zip"
    packed = archive.pack(good, container, producer)
    twin = tmp_path / "copy.zip"
    assert archive.pack(good, twin, producer)["container_sha256"] == packed["container_sha256"]
    assert archive.receive(container, receiver, packed["container_sha256"])["status"] == "identical"
    assert archive.receive(container, receiver, "0" * 64)["status"] == "quarantined"
    db = receiver / "index/index.sqlite3"
    original = sql_snapshot(db)
    assert original["runs"][0][4:7] == ("succeeded", "verified", "not_archived")
    assert len(original["metrics"]) == 1 and len(original["transfer_receipts"]) == 8
    db.unlink()
    assert archive.rebuild(receiver)["runs"] == 1
    assert sql_snapshot(db) == original


def test_declared_parent_assets_and_shared_input(tmp_path, example, capsys):
    producer, receiver = tmp_path / "producer", tmp_path / "receiver"
    parent = launch(example, producer, capsys)
    shared = tmp_path / "versioned-weights.bin"
    shared.write_bytes(b"test weights")
    child = launch(example, producer, capsys, kind="render",
                   input_path=parent / "payload/value.txt",
                   extra=["--parent", parent.name, "--shared-input", f"weights={shared}"])
    verified = verify_bundle(child)
    assert verified["inputs"]["inputs"][0]["parent_run_id"] == parent.name
    assert len(verified["shared"]) == 1
    assert archive.receive(child, receiver)["status"] == "quarantined"  # independent shared input missing
    obj = verified["shared"][0]["path"]
    (receiver / obj).parent.mkdir(parents=True)
    shutil.copyfile(producer / obj, receiver / obj)
    assert "missing upstream" in archive.receive(child, receiver)["reason"]
    assert archive.receive(parent, receiver)["status"] == "accepted"
    assert archive.receive(child, receiver)["status"] == "accepted"
    (receiver / obj).write_bytes(b"wrong weights")
    assert "shared input" in archive.receive(child, receiver)["reason"]
    shutil.copyfile(producer / obj, receiver / obj)

    # Another complete but forged candidate has a digest that the declared parent
    # never emitted. File-level hashes alone must not authorize that dependency.
    staged = tmp_path / "forged"
    shutil.copytree(child, staged)
    inputs = load(staged / "inputs.json")
    item = inputs["inputs"][0]["files"][0]
    rewrite(staged, item["path"], b"not a parent asset")
    new = entry(staged / item["path"], item["path"])
    item.update(new, asset_id="sha256:" + new["sha256"])
    (staged / "inputs.json").chmod(0o644)
    save(staged / "inputs.json", inputs)
    reseal(staged)
    other = tmp_path / "other-store"
    archive.receive(parent, other)
    (other / obj).parent.mkdir(parents=True)
    shutil.copyfile(producer / obj, other / obj)
    refused = archive.receive(staged, other)
    assert "assets missing from declared parent" in refused["reason"]
    assert not (other / "runs" / child.name).exists()


def test_remote_parent_pack_and_mac_receipt(tmp_path, example, capsys):
    mac, worker, empty = tmp_path / "mac", tmp_path / "worker", tmp_path / "unprepared"
    truth = launch(example, mac, capsys)
    digest = verify_bundle(truth)["manifest_sha256"]
    handoff = tmp_path / "only-source.txt"
    shutil.copyfile(truth / "payload/value.txt", handoff)
    generated = launch(example, worker, capsys, kind="generation", input_path=handoff, extra=[
        "--parent", f"{truth.name}={digest}", "--input-parent", f"data={truth.name}"])
    assert not (worker / "runs" / truth.name).exists()
    packed = tmp_path / "worker.zip"
    receipt = archive.pack(generated, packed, worker)
    assert receipt["status"] == "packed_local"
    assert "missing upstream" in archive.receive(packed, empty, receipt["container_sha256"])["reason"]
    assert archive.receive(packed, mac, receipt["container_sha256"])["status"] == "accepted"
    assert archive.rebuild(mac)["runs"] == 2


def test_failed_setup_and_capacity_remain_indexable(tmp_path, example, capsys):
    store = tmp_path / "store"
    missing = launch(example, store, capsys, extra=["--input", f"absent={tmp_path / 'absent'}"])
    assert "missing" in verify_bundle(missing)["run"]["failure_reason"]
    assert recorder.main(command_args(example, store, extra=["--retry-of", missing.name])) == 2
    assert "omit --retry-of for the first scientific attempt" in capsys.readouterr().err
    assert not (missing / "inputs").exists()
    assert {r["role"] for r in load(missing / "inputs.json")["requested_inputs"]} == {"absent", "data"}
    policy = tmp_path / "quota.json"
    policy.write_text(json.dumps({"local_free_floor_bytes": 0, "local_active_batch_target_bytes": 1}))
    full = launch(example, store, capsys, extra=["--storage-policy", str(policy)])
    result = verify_bundle(full)
    assert "configured quota" in result["run"]["failure_reason"]
    assert result["command"]["argv"] is None and result["run"]["sampled_peak_tree_rss_bytes"] is None
    assert archive.rebuild(store)["runs"] == 2
    dest = tmp_path / "dest"
    deferred = archive.receive(missing, dest, storage_policy=policy)
    assert deferred["status"] == "deferred_capacity"
    assert deferred["storage_preflight"]["additional_estimated_bytes"] > 0
    assert not (dest / "runs").exists()
    assert archive.rebuild(dest)["receipts"] == 1
    # No source can be captured at all. A sealed failed run still has typed
    # snapshot metadata and can be restored in a fresh index.
    (example[0] / "scripts/fixture.py").unlink()
    empty = launch(example, store, capsys)
    assert verify_bundle(empty)["run"]["execution_status"] == "failed"
    assert archive.rebuild(store)["runs"] == 3


def test_snapshot_obeys_git_ignores_and_directory_inputs(tmp_path, example, capsys):
    source, _, inp = example
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    cfg = source / "configs"
    cfg.mkdir()
    (source / ".gitignore").write_text("configs/sensitive.json\n")
    (cfg / "sensitive.json").write_text("{}")
    (cfg / "public.json").write_text("{}")
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "one.txt").write_text("1")
    (selected / "two.txt").write_text("2")
    good = launch(example, tmp_path / "store", capsys, extra=["--input", f"set={selected}"])
    assert not (good / "source/configs/sensitive.json").exists()
    assert (good / "source/configs/public.json").exists()
    directory = [i for i in verify_bundle(good)["inputs"]["inputs"] if i["role"] == "set"][0]
    assert directory["kind"] == "directory" and len(directory["files"]) == 2


def test_zip_path_validation(tmp_path):
    candidate = tmp_path / "invalid.zip"
    with zipfile.ZipFile(candidate, "w") as z:
        z.writestr("../outside.txt", b"invalid path fixture")
    result = archive.receive(candidate, tmp_path / "store")
    assert result["status"] == "quarantined"
    assert "nonportable relative path" in result["reason"] and not (tmp_path / "outside.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifetime")
def test_lingering_descendant_cannot_outlive_seal(tmp_path, example, capsys):
    failed = launch(example, tmp_path / "store", capsys, mode="orphan")
    run = verify_bundle(failed)["run"]
    assert run["worker_exit_code"] == 0 and run["execution_status"] == "failed"
    assert "child processes" in run["failure_reason"]


@pytest.mark.parametrize(("config", "error"), [
    ({"identity": {"scene_id": 5}}, "scene_id must be a string or null"),
    ({"model_hash": "pending"}, "invalid model_hash SHA-256"),
])
def test_invalid_identity_rejected_before_allocation(tmp_path, example, capsys, config, error):
    store = tmp_path / "store"
    example[1].write_text(json.dumps(config))
    assert recorder.main(command_args(example, store)) == 2
    assert error in capsys.readouterr().err
    assert not store.exists()


def test_crashed_camera_catalog_is_preserved_as_partial(tmp_path, example, capsys):
    failed = launch(example, tmp_path / "producer", capsys, mode="partial-camera")
    checked = verify_bundle(failed)
    assert checked["run"]["exit_code"] == 7
    assert (failed / "payload/cameras.json").read_bytes() == b'{"views":'
    assert checked["artifacts"]["payload/cameras.json"]["status"] == "partial"
    receiver = tmp_path / "receiver"
    assert archive.receive(failed, receiver)["status"] == "accepted"
    assert archive.rebuild(receiver)["runs"] == 1
    with sqlite3.connect(receiver / "index/index.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM cameras").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM run_artifacts WHERE path='payload/cameras.json'").fetchone() == (1,)


@pytest.mark.skipif(os.name != "posix", reason="uses Unix filenames and unprivileged symlinks")
@pytest.mark.parametrize("mode", ["illegal-name", "symbolic-output"])
def test_invalid_payload_preserves_original_outside_sealed_failure(tmp_path, example, capsys, mode):
    store = tmp_path / "store"
    failed = launch(example, store, capsys, mode=mode)
    verified = verify_bundle(failed)
    run = verified["run"]
    assert (run["execution_status"], run["worker_exit_code"], run["exit_code"]) == ("failed", 0, 1)
    assert verified["artifacts"] == {}
    evidence = run["payload_quarantine"]
    quarantined = store / evidence["path"]
    assert quarantined.is_relative_to(store / "quarantine") and not quarantined.is_relative_to(failed)
    assert load(store / evidence["receipt"])["run_id"] == run["run_id"]
    assert (quarantined / "value.txt").read_bytes() == example[2].read_bytes()
    if mode == "illegal-name":
        assert (quarantined / "bad?.txt").read_bytes() == b"original forensic bytes"
    else:
        link = quarantined / "metrics.jsonl"
        assert link.is_symlink()
        assert link.readlink() == store / "work" / failed.name / "inputs/data/sample.txt"
    events = [json.loads(line)["event"] for line in (failed / "events.jsonl").read_text().splitlines()]
    assert events.index("payload_quarantined") < events.index("evidence_closed")
    assert archive.receive(failed, tmp_path / "receiver")["status"] == "accepted"
    assert archive.rebuild(store)["runs"] == 1


def test_preflight_and_process_creation_failure_leave_scientific_retry_available(tmp_path, example, capsys, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("EL_TEST_TRANSIENT_FAULT", "1")
    first = launch(example, store, capsys)
    monkeypatch.delenv("EL_TEST_TRANSIENT_FAULT")
    retry_args = ["--retry-of", first.name]
    with monkeypatch.context() as patch:
        patch.setattr(recorder._capacity, "check", lambda *args: {"reason": "temporary disk preflight failure"})
        no_launch = launch(example, store, capsys, extra=retry_args)
    before = verify_bundle(no_launch)["run"]
    assert no_launch.parent == store / "runs"
    assert before["retry_of_run_id"] is None and before["requested_retry_of_run_id"] == first.name
    assert recorder.main(command_args(example, store, extra=["--retry-of", no_launch.name])) == 2
    assert f"use --retry-of {first.name}" in capsys.readouterr().err

    with monkeypatch.context() as patch:
        launch_process = subprocess.Popen
        def temporarily_unavailable(args, **kwargs):
            if kwargs.get("shell") is False and str(args[1]).endswith("fixture.py"):
                raise FileNotFoundError("temporary scientific interpreter failure")
            return launch_process(args, **kwargs)
        patch.setattr(recorder.subprocess, "Popen", temporarily_unavailable)
        no_process = launch(example, store, capsys, extra=retry_args)
    assert verify_bundle(no_process)["run"]["worker_exit_code"] is None
    marker = store / "retry_claims" / first.name
    assert not marker.exists()
    # Restoring an older producer can leave a reservation for a provably unlaunched child.
    marker.write_text(no_process.name + "\n")
    scientific = launch(example, store, capsys, extra=retry_args)
    actual = verify_bundle(scientific)["run"]
    assert actual["retry_of_run_id"] == first.name and actual["execution_status"] == "succeeded"
    assert archive.rebuild(store)["runs"] == 4


def test_directory_retry_uses_child_paths_and_hashes(tmp_path, example, capsys, monkeypatch):
    store = tmp_path / "store"
    first_tree = tmp_path / "temporary-input-1"
    second_tree = tmp_path / "temporary-input-2"
    first_tree.mkdir()
    (first_tree / "camera.txt").write_text("same camera")
    shutil.copytree(first_tree, second_tree)
    monkeypatch.setenv("EL_TEST_TRANSIENT_FAULT", "1")
    first = launch(example, store, capsys, extra=["--input", f"views={first_tree}"])
    monkeypatch.delenv("EL_TEST_TRANSIENT_FAULT")
    args = ["--retry-of", first.name, "--input", f"views={second_tree}"]
    (second_tree / "camera.txt").rename(second_tree / "different.txt")
    changed = launch(example, store, capsys, extra=args)
    assert changed.parent == store / "quarantine"
    (second_tree / "different.txt").rename(second_tree / "camera.txt")
    (second_tree / "camera.txt").write_text("different bytes")
    assert launch(example, store, capsys, extra=args).parent == store / "quarantine"
    (second_tree / "camera.txt").write_text("same camera")
    good = launch(example, store, capsys, extra=args)
    checked = verify_bundle(good)
    assert checked["run"]["execution_status"] == "succeeded"
    assert checked["run"]["retry_of_run_id"] == first.name
    assert (good / "inputs/views/temporary-input-2/camera.txt").read_text() == "same camera"
    assert archive.rebuild(store)["runs"] == 2


def test_parked_parent_receive_and_rebuild(tmp_path, example, capsys):
    producer, receiver, shelf = (tmp_path / s for s in ("producer", "receiver", "shelf"))
    source = launch(example, producer, capsys)
    child = launch(example, producer, capsys, kind="render", input_path=source / "payload/value.txt",
                   extra=["--parent", source.name])
    assert archive.receive(source, receiver)["status"] == "accepted"
    digest = sha(source / "manifest.json")
    _, receipt = archive.make_receipt(receiver, "archive", run_id=source.name, manifest_sha256=digest,
                                      status="verified", container_sha256="1"*64, remote_object_ids=["test-object"])
    (shelf / "runs").mkdir(parents=True)
    (shelf / "catalog").mkdir()
    (receiver / "runs" / source.name).rename(shelf / "runs" / source.name)
    marker_path = shelf / "catalog" / (source.name + ".json")
    marker = {"run_id": source.name, "manifest_sha256": digest, "archive_receipt_id": receipt["receipt_id"]}
    save(marker_path, marker)
    assert archive.receive(child, receiver, shelf=shelf)["status"] == "accepted"
    assert archive.rebuild(receiver, shelf)["runs"] == 2
    assert not (receiver / "runs" / source.name).exists()
    save(marker_path, {**marker, "manifest_sha256": "0"*64})
    rejected = archive.receive(child, receiver, shelf=shelf)
    assert rejected["status"] == "quarantined" and "cold-shelf parent" in rejected["reason"]
