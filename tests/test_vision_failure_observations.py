"""Authentic unsuccessful run observed without declaring it a successful scientific parent."""

from hashlib import sha256
from pathlib import Path
from shutil import copytree, ignore_patterns
from types import SimpleNamespace

import pytest

from evidence_lens import recorder
from evidence_lens._bundle import load, sha
from evidence_lens.formal_inputs import PinnedRun, compact_pin
from vision_confirm import observed_failure_evidence, record, report_sources
from vision_record import publish


def test_failed_manifest_persists_with_recorder_parent_contract(tmp_path):
    pytest.importorskip("pyarrow")
    project = tmp_path / "project"
    store, shelf = tmp_path / "store", tmp_path / "shelf"
    copytree(Path(recorder.__file__).parent, project / "src/evidence_lens", ignore=ignore_patterns("__pycache__", "*.pyc"))
    script = project / "scripts/vision_confirmation.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "args=sys.argv\nroster=json.loads(Path(args[args.index('--roster')+1]).read_text())\n"
        "out=Path(args[args.index('--output')+1])\n"
        "(out/'fixture.json').write_text(json.dumps({'observed':roster['observed_failures']}))\n"
    )
    policy = project / "configs/storage.json"
    publish(policy, {"local_free_floor_bytes": 0, "local_active_batch_target_bytes": 1 << 30})
    for name in ("stage2-families-v1", "families-v1"):
        publish(project / "protocol" / (name + ".json"), {"fixture": name})
    cfg = tmp_path / "config.json"
    publish(cfg, {"fixture": True})
    args = SimpleNamespace(
        root=store,
        source_root=project,
        kind="archive_verify",
        config=cfg,
        storage_policy=policy,
        experiment="fixture-upstream",
        trial="fixture",
        interval=None,
        parent=[],
        retry_of=None,
        input=[],
        shared_input=[],
        input_parent=[],
        command=["--", "{python}", "-c", "raise SystemExit(7)"],
    )
    failed = compact_pin(recorder.record(args))
    args.command = ["--", "{python}", "-c", "pass"]
    success = compact_pin(recorder.record(args))
    with pytest.raises(ValueError, match="successful bundle"):
        recorder.resolve_parents(store, [failed["run_id"] + "=" + failed["manifest_sha256"]])
    parents, observed = report_sources((store, shelf), [failed, success])
    assert parents == [success] and observed == [failed]
    with pytest.raises(ValueError, match="unsuccessful"):
        observed_failure_evidence((store, shelf), parents, tmp_path / "wrong")
    # Moving the failed original to a cold shelf must retain its observational role.
    (shelf / "runs").mkdir(parents=True)
    (store / "runs" / failed["run_id"]).rename(shelf / "runs" / failed["run_id"])
    assert report_sources((store, shelf), [failed, success]) == (parents, observed)
    roster = {"action": "report", "catalog_sha256": "a" * 64, "observed_failures": observed}

    def record_it():
        return record(
            project,
            (store, shelf),
            tmp_path / "state",
            "fixture-observe",
            "report",
            "fixture-observe",
            roster,
            parents,
            "fixture.json",
            50000,
            None,
            {},
        )

    selected = record_it()
    assert selected == record_it()
    result = PinnedRun(store, compact_pin(selected))
    inputs = load(result.open("inputs.json"))
    assert inputs["parents"] == [success]
    source = next(r for r in inputs["inputs"] if r["role"] == "observed_failures")
    staged = result.open(source["path"])
    observed = load(staged)["observations"]
    assert (
        len(observed) == 1 and observed[0]["execution_status"] == "failed" and source["parent_run_id"] is None
    )
    assert (
        observed[0]["original_manifest_utf8"].encode()
        == (shelf / "runs" / failed["run_id"] / "manifest.json").read_bytes()
    )
    assert sha256(observed[0]["original_manifest_utf8"].encode()).hexdigest() == failed["manifest_sha256"]
    registered = load(result.open(next(r["path"] for r in inputs["inputs"] if r["role"] == "roster")))
    assert registered["observed_failure_evidence_sha256"] == sha(staged)
    assert load(result.open("payload/fixture.json"))["observed"] == [failed]
