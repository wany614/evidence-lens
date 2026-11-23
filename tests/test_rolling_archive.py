"""Exercise archive-verified parking, read-only report projection and exact restoration."""

import json
import os
import sys
from pathlib import Path

import pytest

from evidence_lens import recorder
from evidence_lens._bundle import load
from evidence_lens._verify import verify_bundle
from evidence_lens.archive import make_receipt
from evidence_lens.formal_inputs import PinnedRun

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from working_set import locations, relocate, report_view


def test_park_restores_original_and_report_consumes_only_pinned_files(tmp_path, capsys):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/unit.py").write_text("# test fixture\n")
    config = tmp_path / "config.json"
    config.write_text("{}")
    store = tmp_path / "store"
    rc = recorder.main(["--root", str(store), "--source-root", str(source), "--config", str(config),
                        "--kind", "archive_verify", "--experiment", "unit-archive", "--trial", "test", "--",
                        "{python}", "-c", "import pathlib,sys; pathlib.Path(sys.argv[1], 'sample.bin').write_bytes(b'unit')",
                        "{out}"])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    rid, digest = result["run_id"], result["manifest_sha256"]
    store, shelf = locations(store, tmp_path / "shelf")
    _, receipt = make_receipt(store, "archive", run_id=rid, manifest_sha256=digest, status="verified",
                              reason=None, remote_object_ids=["unit-fixture"], container_sha256="b" * 64)
    recid = receipt["receipt_id"]
    with pytest.raises(ValueError):
        relocate(store, shelf, rid, "c" * 64, recid, to_shelf=True)
    pin = relocate(store, shelf, rid, digest, recid, to_shelf=True)
    assert relocate(store, shelf, rid, digest, recid, to_shelf=True) == pin
    assert not (store / "runs" / rid).exists()
    view = store / "report-evidence"
    assert report_view(store, shelf, view) == {"evidence_runs": 1}
    selected = PinnedRun(view, pin)
    assert load(selected.open("run.json"))["run_id"] == rid
    assert not (selected.root / "payload/sample.bin").exists()
    relocate(store, shelf, rid, digest, recid, to_shelf=False)
    original = store / "runs" / rid
    assert verify_bundle(original)["manifest_sha256"] == digest
    assert os.path.samefile(selected.open("run.json"), original / "run.json")
