import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from evidence_lens import recorder
from evidence_lens import archive
from evidence_lens._bundle import BundleError, entry, files, load, save, sha
from evidence_lens._verify import verify_bundle


SCRIPT = """import json, os, pathlib, subprocess, sys, time
out, config, source = (pathlib.Path(p) for p in sys.argv[1:4])
assert json.loads(config.read_text())['recording']['artifact_hash'] == 'sha256'
assert not (pathlib.Path.cwd() / 'sensitive.txt').exists()
assert len(os.environ['EL_RUN_ID']) > 10
(out/'value.txt').write_text(source.read_text())
print('fixture output')
print('fixture warning', file=sys.stderr)
if sys.argv[4] == 'fail':
    sys.exit(7)
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


def launch(example, store, capsys, mode="ok", extra=(), kind="scene_render", input_path=None):
    source, cfg, inp = example
    args = ["--root", str(store), "--source-root", str(source), "--kind", kind, "--config", str(cfg),
            "--experiment", "fixture", "--trial", "single", "--interval", ".04",
            "--input", f"data={input_path or inp}", *extra, "--", "{python}", "{source}/scripts/fixture.py",
            "{out}", "{config}", "{input:data}", mode]
    code = recorder.main(args)
    captured = capsys.readouterr()
    assert captured.out, captured.err
    result = json.loads(captured.out)
    assert code == result["exit_code"]
    return Path(result["path"])


def test_recorder_success_failure_and_linked_retry(tmp_path, example, capsys):
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
    failed = launch(example, store, capsys, mode="fail")
    run = verify_bundle(failed)["run"]
    assert (run["execution_status"], run["exit_code"]) == ("failed", 7)
    assert (failed / "payload/value.txt").read_text() == "fixture input"
    retry = launch(example, store, capsys, extra=["--retry-of", failed.name])
    assert retry != failed and verify_bundle(retry)["inputs"]["retry_of"]["run_id"] == failed.name


def rewrite(bundle, name, raw):
    path = bundle / name
    path.chmod(0o644)
    path.write_bytes(raw)


def reseal(bundle):
    """Make a self-consistent but different candidate for the existing run ID."""
    manifest = load(bundle / "manifest.json")
    manifest["files"] = [entry(p, n) for n, p in files(bundle).items()
                         if n not in {"manifest.json", "completion.json"}]
    save(bundle / "manifest.json", manifest)
    completion = load(bundle / "completion.json")
    completion["manifest_sha256"] = sha(bundle / "manifest.json")
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
    save(staged / "inputs.json", inputs)
    reseal(staged)
    other = tmp_path / "other-store"
    archive.receive(parent, other)
    (other / obj).parent.mkdir(parents=True)
    shutil.copyfile(producer / obj, other / obj)
    refused = archive.receive(staged, other)
    assert "assets missing from declared parent" in refused["reason"]
    assert not (other / "runs" / child.name).exists()
