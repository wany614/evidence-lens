"""The data exporter and external verifier agree on inventory and per-file bytes."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from stage3_training_transport import write_dataset
from verify_stage3_public import extract_checked
from vision_release import sig


def test_transport_excludes_unlisted_source_and_rejects_incorrect_member(tmp_path):
    data = tmp_path/"payload"
    data.mkdir()
    (data/"dataset.json").write_text('{"split":"train"}')
    (data/"rgb.npy").write_bytes(b"fixture of a prediction")
    (data/"other").write_text("not part of this transport")

    class Original:
        pin = {"run_id": "fixture", "manifest_sha256": "b"*64}
        files = {"payload/"+n: sig(data/n) for n in ("dataset.json","rgb.npy")}

        def open(self, rel):
            return tmp_path/rel

    table = {"split":"train", "manifest": sig(data/"dataset.json"), "files":{"rgb.npy":sig(data/"rgb.npy")}}
    transport = write_dataset(tmp_path, Original(), table, "train")
    with ZipFile(tmp_path/transport["name"]) as archive:
        assert set(archive.namelist()) == {"dataset.json", "rgb.npy"}
    table["files"]["rgb.npy"]["sha256"] = "a"*64
    other = tmp_path/"second"
    other.mkdir()
    with pytest.raises(ValueError, match="Original data member differs"):
        write_dataset(other, Original(), table, "train")


def test_public_zip_needs_the_exact_checked_member_bytes(tmp_path):
    body = b"native score"
    declared = {"bytes":len(body), "sha256":hashlib.sha256(body).hexdigest()}
    index = json.dumps({"files":{"data/map":declared}}).encode()
    descriptor = {"index":"index.json", "index_sha256":hashlib.sha256(index).hexdigest(), "members":2}
    bundle = tmp_path/"cand.zip"
    with ZipFile(bundle, "w") as f:
        f.writestr("index.json",index)
        f.writestr("data/map",body)
    assert extract_checked(bundle,tmp_path/"fresh",descriptor)["files"]["data/map"] == declared
    with ZipFile(tmp_path/"wrong.zip", "w") as f:
        f.writestr("index.json",index)
        f.writestr("data/map",b"native SCORE")
    with pytest.raises(ValueError, match="Public member differs"):
        extract_checked(tmp_path/"wrong.zip",tmp_path/"bad",descriptor)
