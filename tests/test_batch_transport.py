import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_windows_batch import digest, extract


def handoff(tmp_path, member):
    meta = b'{"trial_id":"fixture-B0"}'
    source = tmp_path / "fixture.zip"
    with zipfile.ZipFile(source, "x") as bundle:
        bundle.writestr("handoff.json", meta)
        bundle.writestr(member, b"# fixture\n")
    return {"name": source.name, "bytes": source.stat().st_size, "sha256": digest(source),
            "handoff_sha256": hashlib.sha256(meta).hexdigest(), "trial_id": json.loads(meta)["trial_id"]}


def test_module_names_and_repeated_extraction(tmp_path):
    member = "code/src/evidence_lens/__init__.py"
    job = handoff(tmp_path, member)
    stage = tmp_path / "staged"
    path = extract(job, tmp_path, stage)
    assert (path / member).read_bytes() == b"# fixture\n"
    assert extract(job, tmp_path, stage) == path


@pytest.mark.parametrize("name", ["../other.py", "code/../other.py", "/absolute.py", "code//module.py"])
def test_nonportable_paths_do_not_publish_stage(tmp_path, name):
    job = handoff(tmp_path, name)
    stage = tmp_path / "staged"
    with pytest.raises(ValueError, match="path"):
        extract(job, tmp_path, stage)
    assert not (stage / job["sha256"]).exists()
