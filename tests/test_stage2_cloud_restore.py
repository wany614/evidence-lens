"""Exercise segmented recovery, the 4 GiB ceiling and retries with disposable registered bytes."""

import io
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
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


def test_snapshot_release_requires_restores_and_parked_seals_and_queue_can_resume(tmp_path, monkeypatch):
    from evidence_lens import recorder
    import stage2_confirmation as mac

    store, shelf, folder = (tmp_path / n for n in ("store", "shelf", "private-archives"))
    family = "synthetic_reclamation_case"
    catalog = "7" * 64
    base = store / "stage2" / catalog / "confirmation"
    source = tmp_path / "fixture-project"
    publish(source / "scripts/fixture.py", b"# Synthetic storage regression.\n")
    cfg = tmp_path / "cfg.json"
    publish(cfg, encoded({"scene_family_id": family, "synthetic": True}))
    limits = source / "configs/storage.json"
    publish(limits, encoded({"local_active_batch_target_bytes": 4 * 2**20, "local_free_floor_bytes": 0}))
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        status = recorder.main(["--root", str(store), "--source-root", str(source), "--config", str(cfg),
            "--storage-policy", str(limits), "--kind", "generation", "--experiment", "storage-fixture",
            "--trial", family + "-B0", "--", "{python}", "-c", "print('synthetic archive retention')"])
    assert status == 0
    rid = json.loads(stdout.getvalue())["run_id"]
    manifest = sha(store / "runs" / rid / "manifest.json")
    pin = {"run_id": rid, "manifest_sha256": manifest}
    group = base / "families" / (family + ".json")
    auth = base / "authorization.json"
    publish(auth, encoded({"purpose": "synthetic_storage_fixture"}))
    publish(group, encoded({"family": family, "trials": {"synthetic": {"run_pins": {"test": pin}, "returns": []}}}))
    identity = {"family_sha256": sha(group), "authorization_sha256": sha(auth), "catalog_sha256": catalog}
    doc, origins = archive.file_map(store, shelf, {rid: manifest}, {}, family, identity)
    packed = folder / (family + ".zip")
    original = archive.write_archive(doc, origins, packed, 0)
    local = {"schema": "el-stage2-family-local-archive-1", "family_sha256": sha(group),
             "authorization_sha256": sha(auth), "proof": original}
    publish(base / "archives" / (family + ".local.json"), encoded(local))
    transport = tmp_path / "fixture-provider"
    op = cloud.offer(packed, sha(packed), transport, base / "archives/offers")
    offer = load(op)
    publish(base / "archives" / (family + ".cloud-offer.json"), encoded({
        "family": family, "authorization_sha256": sha(auth), "archive_sha256": sha(packed),
        "offer_name": op.name, "offer_sha256": sha(op)}))
    part = offer["parts"][0]
    rows = [{"id": "synthetic-provider-object", "name": part["name"], "bytes": part["bytes"],
             "parent_ids": [offer["remote_parent_id"]], "shared": False, "mime_type": "application/zip",
             "download_url": "https://example.googleusercontent.com/unit/0"}]
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda req, timeout: io.BytesIO((transport / part["name"]).read_bytes()))
    evidence = base / "archives/provider" / family
    cloud.confirm(op, sha(op), rows, tmp_path / "bounded-work", evidence, audit=True)
    provider = evidence / "cloud-proof.json"
    cloud.certify(store, shelf, op, sha(op), provider, sha(provider))
    cert = base / "archives" / (sha(packed) + ".json")
    args = SimpleNamespace(store=store, shelf=shelf, proof=cert, proof_sha256=sha(cert), catalog_sha256=catalog,
                           private_archive=folder, cloud_stage_directory=transport)
    with pytest.raises(BundleError, match="parked"):
        archive.release_copy(args)
    assert packed.is_file()
    archive.park(args)
    audit = evidence / "restored-runs" / (rid + ".json")
    audit.rename(audit.with_suffix(".held"))
    with pytest.raises(FileNotFoundError):
        archive.release_copy(args)
    assert packed.is_file()
    audit.with_suffix(".held").rename(audit)
    good = packed.with_suffix(".saved")
    packed.rename(good)
    with pytest.raises(BundleError, match="without an authenticated release"):
        mac.optional_archive(args, {"family": family}, base, sha(auth))
    packed.write_bytes(b"damaged")
    with pytest.raises(BundleError, match="snapshot bytes differ"):
        archive.release_copy(args)
    packed.unlink()
    good.rename(packed)
    released = archive.release_copy(args)
    assert released["bytes"] == part["bytes"] and not packed.exists()
    assert archive.release_copy(args) == released
    assert mac.optional_archive(args, {"family": family}, base, sha(auth)) == local
    assert not packed.exists() and sha(transport / part["name"]) == part["sha256"]
    assert verify_bundle(shelf / "runs" / rid)["manifest_sha256"] == manifest
