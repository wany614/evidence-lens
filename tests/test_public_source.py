"""A synthetic staged file cannot slip into the source candidate."""
import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/source_candidate.py"
spec = importlib.util.spec_from_file_location("source_candidate", MODULE)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def test_recovery_table_is_limited_to_pinned_columns_and_values():
    name = "docs/results/reconstruction-recovery/resources/telemetry.csv.gz"

    def verify(contents, path=name):
        raw = contents.encode("ascii")
        data = gzip.compress(raw, mtime=0)
        pin = {"format": "gzip-csv-1", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "csv_bytes": len(raw), "csv_sha256": hashlib.sha256(raw).hexdigest(), "rows": 1}
        candidate.audit_numeric_table(path, data, pin)

    header = candidate.RECOVERY_TABLES[name] + "\n"
    row = "20260101T000000000000Z-0000000000000000,1,4,256,\n"
    verify(header + row)
    with pytest.raises(ValueError):
        verify(header + row, "docs/results/reconstruction-recovery/resources/notes.csv.gz")
    with pytest.raises(ValueError):
        verify(header + row.replace(",4,", ",sample-host,"))
    with pytest.raises(ValueError):
        verify(header.replace("tree_rss_mib", "account") + row)


def test_exact_file_and_rights_inventory(tmp_path):
    def git(*argv):
        subprocess.run(["git", "-C", str(tmp_path), *argv], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path/"configs").mkdir()
    policy = tmp_path/candidate.POLICY
    contents = {"files": {candidate.POLICY: "original"}, "license_status": {"original": {"public_export": False}}}
    policy.write_text(json.dumps(contents))
    git("add", candidate.POLICY)
    assert len(candidate.audit(tmp_path)[1]) == 1
    (tmp_path/"private.txt").write_text("fixture only")
    git("add", "private.txt")
    with pytest.raises(ValueError, match="explicit file list"):
        candidate.audit(tmp_path)
    contents["files"]["private.txt"] = "unclassified"
    policy.write_text(json.dumps(contents))
    git("add", candidate.POLICY)
    with pytest.raises(ValueError, match="missing license record"):
        candidate.audit(tmp_path)

    # Only the staged bytes, execute bit and frozen tree identity enter the ZIP.
    contents["files"]["private.txt"] = "original"
    contents["license_status"]["original"]["public_export"] = True
    policy.write_text(json.dumps(contents))
    git("add", candidate.POLICY)
    git("update-index", "--chmod=+x", "private.txt")
    (tmp_path/"private.txt").write_text("unstaged bytes")
    export = tmp_path / "source.zip"
    subprocess.run([sys.executable, str(MODULE), "--root", str(tmp_path), "--export", str(export)], check=True)
    tree = subprocess.check_output(["git", "-C", str(tmp_path), "write-tree"]).strip()
    with zipfile.ZipFile(export) as bundle:
        assert bundle.comment == b"git-tree " + tree
        assert bundle.read("private.txt") == b"fixture only"
        assert bundle.getinfo("private.txt").external_attr >> 16 == 0o100755
