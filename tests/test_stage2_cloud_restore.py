"""Exercise segmented recovery, the 4 GiB ceiling and retries with disposable registered bytes."""

import io
from pathlib import Path
import sys
import pytest
from evidence_lens._bundle import sha, encoded, load, BundleError
from evidence_lens._verify import verify_bundle
from make_handoff import publish

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage2_cloud_restore as cloud
import stage2_confirmation_archive as archive
from stage2_confirmation_rehearsal import exercise


@pytest.fixture(autouse=True)
def private_fixture_location(tmp_path, monkeypatch):
    path = tmp_path / "private-location.json"
    publish(
        path,
        encoded(
            {
                "schema": "el-stage2-drive-location-1",
                "provider": "Google Drive",
                "sealed_runs_parent_id": "example-private-parent",
                "scientific_archive_parent_id": "example-research-folder",
                "engineering_fixture_parent_id": "example-fixture-folder",
            }
        ),
    )
    monkeypatch.setenv("EVIDENCE_LENS_DRIVE_ARCHIVE_CONFIG", str(path))
    monkeypatch.setenv("EVIDENCE_LENS_DRIVE_ARCHIVE_CONFIG_SHA256", sha(path))
    return path


def make(tmp):
    root = tmp / "fixture"
    proof = exercise(root)
    rid = proof["producer_run_id"]
    digest = proof["manifest_sha256"]
    q = tmp / "case.json"
    publish(q, encoded({"synthetic": True}))
    index, origins = archive.file_map(
        root / "receiver",
        root / "shelf",
        {rid: digest},
        {"protocol/case.json": q},
        None,
        {"purpose": "synthetic_engineering_fixture"},
    )
    container = tmp / "private.zip"
    archive.write_archive(index, origins, container, 0)
    outgoing = tmp / "source"
    op = cloud.offer(container, sha(container), outgoing, tmp / "offer", fixture=True, part_bytes=4096)
    offer = load(op)
    rows = [
        {
            "id": f"fixture-id-{n:04}",
            "name": r["name"],
            "bytes": r["bytes"],
            "parent_ids": [cloud.provider_location()["engineering_fixture_parent_id"]],
            "shared": False,
            "mime_type": "application/x-zip" if n == 0 else "application/octet-stream",
            "download_url": "https://example.googleusercontent.com/unit/" + str(n),
        }
        for n, r in enumerate(offer["parts"])
    ]
    return op, offer, index, rows, outgoing, rid, digest


def test_bounded_physical_restore_recovers_interruption_and_reuses_proof(tmp_path, monkeypatch):
    op, offer, index, rows, source, rid, digest = make(tmp_path)
    root = tmp_path / "scope"
    out = tmp_path / "proof"
    dest = tmp_path / "restored"
    calls = []

    def resolve(req, timeout):
        n = int(req.full_url.rsplit("/", 1)[1])
        calls.append(n)
        return io.BytesIO((source / rows[n]["name"]).read_bytes())

    monkeypatch.setattr(cloud.urllib.request, "urlopen", resolve)
    using = cloud.confirm(op, sha(op), rows, root, out, audit=True, restore=dest)
    assert len(calls) == len(rows) and sha(dest / "runs" / rid / "manifest.json") == digest
    assert (
        all(v["bytes"] <= 4096 for v in using["part_receipts"])
        and verify_bundle(dest / "runs" / rid)["manifest_sha256"] == digest
    )
    stale = root / ".stage2-private-recovery/in-flight"
    stale.mkdir()
    (stale / "uncommitted-range.tmp").write_bytes(b"unit")
    assert cloud.confirm(op, sha(op), rows, root, out, audit=True, restore=dest) == using and len(
        calls
    ) == len(rows)
    assert not stale.exists() and not (root / ".stage2-private-recovery/temporary").exists()
    with pytest.raises(BundleError, match="4 GiB"):
        cloud.bounded_restore(offer, rows, [rid], tmp_path / "too-small", index, root, 4096)
    with pytest.raises(BundleError, match="scientific"):
        cloud.certify(
            tmp_path / "empty",
            tmp_path / "no-shelf",
            op,
            sha(op),
            out / "cloud-proof.json",
            sha(out / "cloud-proof.json"),
        )
    rows[0]["shared"] = True
    with pytest.raises(BundleError, match="private folder"):
        cloud.confirm(op, sha(op), rows, root, out)


def test_lost_or_corrupt_segment_never_creates_cloud_proof(tmp_path, monkeypatch):
    op, offer, index, rows, source, rid, digest = make(tmp_path)
    out = tmp_path / "proof"

    def corrupt(req, timeout):
        n = int(req.full_url.rsplit("/", 1)[1])
        d = (source / rows[n]["name"]).read_bytes()
        return io.BytesIO((b"x" * len(d)) if n == 1 else d)

    monkeypatch.setattr(cloud.urllib.request, "urlopen", corrupt)
    with pytest.raises(BundleError, match="downloaded part differs"):
        cloud.confirm(op, sha(op), rows, tmp_path / "work", out, audit=True)
    assert (
        not (out / "cloud-proof.json").exists()
        and not (tmp_path / "work/.stage2-private-recovery/in-flight").exists()
    )

    # The large-archive path uses the same provider-checked segments with bounded random access.
    def good(req, timeout):
        n = int(req.full_url.rsplit("/", 1)[1])
        return io.BytesIO((source / rows[n]["name"]).read_bytes())

    monkeypatch.setattr(cloud.urllib.request, "urlopen", good)
    result = cloud.bounded_restore(offer, rows, [rid], tmp_path / "remote-range", index, tmp_path / "work")
    assert (
        result["runs"] == [rid]
        and verify_bundle(tmp_path / "remote-range/runs" / rid)["manifest_sha256"] == digest
    )


def test_private_parent_has_independent_hash_and_role_check(tmp_path, private_fixture_location):
    op, record, index, rows, source, rid, digest = make(tmp_path)
    with pytest.raises(BundleError, match="digest differs"):
        cloud.provider_location(path=private_fixture_location, digest="0" * 64)
    alternative = tmp_path / "other-location.json"
    old = load(private_fixture_location)
    old["engineering_fixture_parent_id"] = "example-different-fixture-folder"
    publish(alternative, encoded(old))
    loc = cloud.provider_location(path=alternative, digest=sha(alternative))
    with pytest.raises(BundleError, match="Unexpected restore plan"):
        cloud.read_offer(op, sha(op), loc)
    rows[0]["parent_ids"] = [old["scientific_archive_parent_id"]]
    with pytest.raises(BundleError, match="private folder"):
        cloud.metadata(record, rows)
