"""Regression checks for member-level network access and the two external data roles."""
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dtu_gate import file_hash, verify_members
from dtu_sample import RemoteZip


class FakeRange:
    def __init__(self, body):
        self.body = body
        self.reads = []

    def get(self, _url, *, span, limit, total=None, etag=None):
        a, b = span
        assert b - a + 1 == limit
        assert total in (None, len(self.body))
        assert etag in (None, '"example"')
        self.reads.append(span)
        return self.body[a:b + 1], {"content-range": f"bytes {a}-{b}/{len(self.body)}", "etag": '"example"'}


def test_exact_member_cache_does_not_read_neighbouring_photograph():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("source.png", b"s" * 500000)
        z.writestr("reference.png", b"r" * 500000)
    net = FakeRange(buf.getvalue())
    remote = RemoteZip(net, "unused")
    with zipfile.ZipFile(remote) as z:
        source, reference = z.getinfo("source.png"), z.getinfo("reference.png")
        net.reads.clear()
        remote.prime_member(source)
        assert z.read(source) == b"s" * 500000
        assert net.reads == [(0, 29), (30, reference.header_offset - 1)]


def test_external_sources_and_truth_require_distinct_pinned_gates(tmp_path):
    def save(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return p

    views = save("views.json", {"lighting_id":3,
        "source_view_ids":{"s00":24, "s01":23, "s02":17, "s03":22},
        "target_view_ids":{f"v{i+1:02d}":k for i,k in enumerate((16,18,25,37,35,36))}})
    vf = file_hash(views)
    freeze = save("freeze.json", {"protocol":{"external_objects":["set6"]},"view_set_sha256":vf})
    source = save("source.json", {"schema":"el-dtu-opening-1","object_ids":["set6"],"freeze_sha256":file_hash(freeze)})
    target = save("truth.json", {"schema":"el-dtu-reference-gate-1","object_ids":["set6"],"freeze_sha256":file_hash(freeze)})
    for role, gate, other, ids in (("source",source,target,(24,23,17,22)),("truth",target,source,(16,18,25,37,35,36))):
        names = [f"SampleSet/MVS Data/Rectified/scan6/rect_{k:03d}_3_r5000.png" for k in ids]
        spec = {"members":names, "object_id":"set6", "access_role":role, "view_set_sha256":vf, "authority_sha256":file_hash(gate)}
        verify_members(spec, views, gate, freeze)
        with pytest.raises(ValueError):
            verify_members(dict(spec, authority_sha256=file_hash(other)), views, other, freeze)
        with pytest.raises(ValueError):
            verify_members(dict(spec, members=names[:-1]), views, gate, freeze)
        with pytest.raises(ValueError):
            verify_members(spec, views, None, freeze)


def test_metadata_role_rejects_unregistered_movie():
    with pytest.raises(ValueError):
        verify_members({"members":["SampleSet/1895.mp4"]}, None, None, None)


def test_capacity_keeps_counting_after_ephemeral_entries_disappear(tmp_path, monkeypatch):
    from evidence_lens import _capacity

    (tmp_path / "still-here").write_bytes(b"x" * 41)
    def walk(_root, onerror):
        onerror(FileNotFoundError("ephemeral directory"))
        yield tmp_path, [], ["already-moved", "still-here"]
    monkeypatch.setattr(_capacity.os, "walk", walk)
    assert _capacity.bytes_on_tree(tmp_path) == 41

    def denied(_root, onerror):
        onerror(PermissionError("unreadable directory"))
        yield tmp_path, [], []
    monkeypatch.setattr(_capacity.os, "walk", denied)
    with pytest.raises(PermissionError):
        _capacity.bytes_on_tree(tmp_path)
