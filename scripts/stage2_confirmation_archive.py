"""Privately preserve original B confirmation recorder packages and explicit external dependencies."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, encoded, freeze_tree, load, local_path, sha
from evidence_lens._verify import require, verify_bundle
from make_handoff import publish
from portable_family import entry
from working_set import locations, relocate
import run_stage2_confirmation as win

DOC = "archive.json"


def file_map(store, shelf, pins, extras, family, identity):
    files = {}
    origins = {}
    ancestors = {}
    shared = {}

    def insert(name, p, h=None, n=None):
        logical = local_path(name)
        size = p.stat().st_size
        digest = sha(p) if h is None else h
        require(
            not p.is_symlink() and p.is_file() and (n is None or size == n),
            "Changed source in private family archive",
        )
        row = {"bytes": size, "sha256": digest}
        require(logical not in files or files[logical] == row, "Conflicting archive paths")
        files[logical] = row
        origins.setdefault(digest, p)

    for rid, m in sorted(pins.items()):
        require(RUN_ID.fullmatch(rid) and HASH.fullmatch(m), "Malformed selected science pin")
        dirs = [r / "runs" / rid for r in (store, shelf) if (r / "runs" / rid).is_dir()]
        require(len(dirs) == 1, "Missing or duplicated canonical scientific run")
        d = dirs[0]
        rec = verify_bundle(d)
        require(
            rec["manifest_sha256"] == m and rec["run"]["scene_family_id"] == family,
            "Wrong family or manifest in archive",
        )
        for parent in rec["inputs"]["parents"]:
            pid = parent["run_id"]
            pd = parent["manifest_sha256"]
            require(ancestors.get(pid, pd) == pd and pins.get(pid, pd) == pd, "Conflicting parent pins")
            if pid not in pins:
                ancestors[pid] = pd
        for obj in rec["shared"]:
            h = obj["sha256"]
            n = obj["bytes"]
            require(shared.get(h, n) == n, "Conflicting external asset sizes")
            shared[h] = n
        prefix = "runs/" + rid + "/"
        for member in rec["manifest"]["files"]:
            insert(prefix + member["path"], d / member["path"], member["sha256"], member["bytes"])
        insert(prefix + "manifest.json", d / "manifest.json", m)
        insert(prefix + "completion.json", d / "completion.json")
    for n, p in sorted(extras.items()):
        insert(n, p)
    doc = {
        "schema": "el-stage2-confirmation-archive-1",
        "scope": "selected family recorder bytes, attempts, transfer and registration evidence",
        "family": family,
        "identity": identity,
        "run_manifests": pins,
        "external_parent_manifests": ancestors,
        "external_shared_assets": {h: {"bytes": n, "sha256": h} for h, n in shared.items()},
        "files": files,
    }
    return doc, origins


def write_archive(doc, origins, target, min_free):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with zipfile.ZipFile(target) as z:
            require(z.read(DOC) == encoded(doc), "Existing private archive differs")
        proof = verify_archive(target, sha(target))
        return proof
    sizes = {r["sha256"]: r["bytes"] for r in doc["files"].values()}
    require(
        shutil.disk_usage(target.parent).free - sum(sizes.values()) - 2**20 >= min_free,
        "Family archive would breach free disk floor",
    )
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=".stage2-archive-")
    os.close(fd)
    tmp = Path(name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr(entry(DOC), encoded(doc))
            for h, p in sorted(origins.items()):
                digest = hashlib.sha256()
                seen = 0
                with p.open("rb") as src, z.open(entry("objects/" + h), "w", force_zip64=True) as dst:
                    while block := src.read(1 << 20):
                        digest.update(block)
                        seen += len(block)
                        dst.write(block)
                require(
                    digest.hexdigest() == h and seen == sizes[h], "Archive source changed during streaming"
                )
        verified = verify_archive(tmp, sha(tmp))
        os.link(tmp, target)
        return verified
    finally:
        tmp.unlink(missing_ok=True)


def contents(container, digest):
    require(
        HASH.fullmatch(digest)
        and container.is_file()
        and not container.is_symlink()
        and sha(container) == digest,
        "Wrong private container bytes",
    )
    with zipfile.ZipFile(container) as z:
        raw = z.read(DOC)
        doc = json.loads(raw)
        require(
            doc["schema"] == "el-stage2-confirmation-archive-1" and raw == encoded(doc),
            "Unexpected or ambiguous archive metadata",
        )
        sizes = {}
        for k, r in doc["files"].items():
            local_path(k)
            h = r["sha256"]
            n = r["bytes"]
            require(
                HASH.fullmatch(h) and type(n) is int and n >= 0 and sizes.get(h, n) == n,
                "Invalid archived object or length",
            )
            sizes[h] = n
        require(
            sorted(z.namelist()) == sorted([DOC, *("objects/" + h for h in sizes)]),
            "Missing/duplicate/unregistered archive content",
        )
        for rid, m in doc["run_manifests"].items():
            prefix = "runs/" + rid + "/"
            require(
                RUN_ID.fullmatch(rid)
                and HASH.fullmatch(m)
                and doc["files"][prefix + "manifest.json"]["sha256"] == m,
                "Wrong internal recorder manifest",
            )
            manifest = json.loads(z.read("objects/" + m))
            seal = json.loads(z.read("objects/" + doc["files"][prefix + "completion.json"]["sha256"]))
            require(
                seal["run_id"] == manifest["run_id"] == rid and seal["manifest_sha256"] == m,
                "Invalid run seal",
            )
            members = {
                prefix + local_path(i["path"]): {"sha256": i["sha256"], "bytes": i["bytes"]}
                for i in manifest["files"]
            }
            require(
                len(members) == len(manifest["files"])
                and {n for n in doc["files"] if n.startswith(prefix)}
                == set(members) | {prefix + "completion.json", prefix + "manifest.json"}
                and all(doc["files"][n] == value for n, value in members.items()),
                "Original recorder bytes are incomplete",
            )
        require(
            all(not k.startswith("runs/") or k.split("/")[1] in doc["run_manifests"] for k in doc["files"]),
            "Extra unsealed run",
        )
    return doc, sizes


def verify_archive(container, digest, destination=None):
    doc, sizes = contents(container, digest)
    if destination is not None:
        require(not destination.exists(), "Restore requires an exclusive destination")
        destination.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent if destination is not None else None
    ) as scratch:
        scratch = Path(scratch)
        objs = scratch / "objects"
        tree = scratch / "tree"
        objs.mkdir()
        tree.mkdir()
        with zipfile.ZipFile(container) as z:
            for h, n in sorted(sizes.items()):
                key = "objects/" + h
                require(z.getinfo(key).file_size == n, "Archive size differs")
                with z.open(key) as inp:
                    if destination is None:
                        actual = hashlib.file_digest(inp, "sha256").hexdigest()
                    else:
                        obj = objs / h
                        with obj.open("xb") as out:
                            shutil.copyfileobj(inp, out, 1 << 20)
                        actual = sha(obj)
                        obj.chmod(0o444)
                require(actual == h, "Archive object bytes differ")
        if destination is not None:
            for n, row in doc["files"].items():
                p = tree / n
                p.parent.mkdir(parents=True, exist_ok=True)
                os.link(objs / row["sha256"], p)
            for rid, m in doc["run_manifests"].items():
                run = tree / "runs" / rid
                require(verify_bundle(run)["manifest_sha256"] == m, "Restored recorder does not verify")
                freeze_tree(run)
            os.rename(tree, destination)
    return {
        "schema": "el-stage2-confirmation-archive-proof-1",
        "family": doc["family"],
        "container_sha256": digest,
        "bytes": container.stat().st_size,
        "logical_files": len(doc["files"]),
        "unique_objects": len(sizes),
        "run_manifests": doc["run_manifests"],
        "external_parent_manifests": doc["external_parent_manifests"],
        "external_shared_assets": doc["external_shared_assets"],
        "restored": destination is not None,
    }


def snapshot(a):
    from stage2_opening import verify_opening

    catalog = win.signed(a.catalog, a.catalog_sha256)
    opening = verify_opening(a.store, catalog, a.catalog_sha256)
    root = a.store / "stage2" / a.catalog_sha256 / "confirmation"
    auth = win.signed(root / "authorization.json", a.authorization_sha256)
    require(
        auth["method_sha256"] == opening["method_sha256"]
        and auth["freeze_run_id"] == opening["freeze_run_id"]
        and auth["freeze_manifest_sha256"] == opening["freeze_manifest_sha256"]
        and auth["opening_sha256"]
        == sha(a.store / "protocol/stage2" / a.catalog_sha256 / "test-opening.json")
        and a.family in opening["test_family_ids"],
        "Different verified confirmation opening",
    )
    group_path = root / "families" / (a.family + ".json")
    group = win.signed(group_path, a.family_sha256)
    require(
        group["family"] == a.family
        and group["authorization_sha256"] == a.authorization_sha256
        and set(group["trials"]) == {a.family + "-" + c for c in ("B0", "R1", "Q1")},
        "Wrong family closure",
    )
    extras = {}
    pins = {}

    def add(p):
        extras[p.relative_to(a.store).as_posix()] = p

    def pin(r):
        rid, m = r["run_id"], r["manifest_sha256"]
        require(pins.get(rid, m) == m, "Archive run pin conflicts")
        pins[rid] = m

    add(group_path)
    add(root / "authorization.json")
    add(a.store / "protocol/stage2" / a.catalog_sha256 / "test-opening.json")
    add(a.store / "formal-references" / a.catalog_sha256 / (a.family + ".json"))
    for t, entry_ in group["trials"].items():
        choice = a.store / "trials" / t / "trial_selection.json"
        q = win.signed(choice, entry_["selection_sha256"])
        add(choice)
        add(root / "accepted" / (t + ".json"))
        for record in (*entry_["run_pins"].values(), *entry_["returns"]):
            pin(record)
        for attempt in entry_["returns"]:
            receipt = (
                root / "returns" / entry_["batch"] / "envelopes" / (attempt["run_id"] + ".receipt.json")
            )
            require(sha(receipt) == attempt["receipt_sha256"], "Preserved transport envelope changed")
            add(receipt)
        identity_dir = a.store / "formal" / a.catalog_sha256
        dispatch_path = (
            identity_dir / "dispatch"
            if entry_["condition_id"] == "B0"
            else identity_dir / "conditions" / "dispatch"
        ) / (t + ".json")
        add(dispatch_path)
        for note in entry_["controller_notes"]:
            path = root / "returns" / entry_["batch"] / "controllers" / note["file"]
            require(sha(path) == note["sha256"], "Transport note changed")
            add(path)
        ticket = root / "tickets" / (entry_["batch"] + ".json")
        require(sha(ticket) == entry_["ticket_sha256"], "Archive ticket mismatch")
        add(ticket)
        dispatch = (
            a.store
            / "stage2"
            / a.catalog_sha256
            / ("dispatches" if entry_["condition_id"] == "B0" else "condition-batches")
            / entry_["batch"]
        )
        for f in dispatch.iterdir():
            if f.is_file() and not f.name.startswith("."):
                add(f)
        # Preserve recorded failed attempts alongside scientifically selected successes.
        for tree in (a.store, a.shelf):
            for run in (tree / "runs").glob("*/run.json"):
                if RUN_ID.fullmatch(run.parent.name):
                    meta = load(run)
                    if (
                        meta.get("trial_id") == t
                        and meta.get("dataset_hash") == a.catalog_sha256
                        and meta.get("experiment_id")
                        in ("stage2-reference", "stage2-single-image", "stage2-heldout-conditions")
                    ):
                        pin(
                            {
                                "run_id": run.parent.name,
                                "manifest_sha256": sha(run.parent / "manifest.json"),
                            }
                        )
        for rlist in q.get("evidence", {}).get("transfer_receipts", {}).values():
            for rec in rlist:
                p = a.store / "receipts/transfer" / (rec["receipt_id"] + ".json")
                require(sha(p) == rec["sha256"], "Transfer receipt changed")
                add(p)
    for p in (a.store / "receipts/transfer").glob("*.json"):
        if load(p).get("run_id") in pins:
            add(p)
    extras["protocol/stage2-families-v1.json"] = a.catalog
    for name in (
        "protocol/stage2-reference-v1.json",
        "protocol/stage2-training-v1.json",
        "configs/stage2-reference-freeze-v1.json",
    ):
        extras[name] = a.project / name
    identity = {
        "catalog_sha256": a.catalog_sha256,
        "family_sha256": a.family_sha256,
        "authorization_sha256": a.authorization_sha256,
        "opening_sha256": auth["opening_sha256"],
    }
    doc, origins = file_map(a.store, a.shelf, pins, extras, a.family, identity)
    return write_archive(
        doc, origins, a.output, load(a.project / "configs/storage.json")["local_free_floor_bytes"]
    )


def park(a):
    proof = win.signed(a.proof, a.proof_sha256)
    require(
        proof["schema"] == "el-stage2-family-cloud-archive-1",
        "Expected independently confirmed provider archive",
    )
    store, shelf = locations(a.store, a.shelf)
    receipts = proof["receipts"]
    result = {}
    for rid, r in sorted(receipts.items()):
        result[rid] = relocate(
            store, shelf, rid, r["manifest_sha256"], r["archive_receipt_id"], to_shelf=True
        )
    return {"family": proof["family"], "parked": len(result), "run_ids": sorted(result)}


def retained_copy(store, shelf, base, family, certificate_sha256, *, require_parked=False, location=None):
    """Bind the original snapshot, its cloud readback and every retained recorder identity."""
    from stage2_cloud_restore import certify

    marker = base / "archives" / (family + ".local.json")
    local = load(marker)
    snapshot = local["proof"]
    digest = snapshot["container_sha256"]
    offer_pin = base / "archives" / (family + ".cloud-offer.json")
    pin = load(offer_pin)
    require(
        HASH.fullmatch(digest) and snapshot["family"] == family
        and pin["family"] == family and pin["archive_sha256"] == digest
        and sha(base / "families" / (family + ".json")) == local["family_sha256"]
        and pin["authorization_sha256"] == local["authorization_sha256"] == sha(base / "authorization.json"),
        "Local snapshot does not bind this family and opening",
    )
    compound = base / "archives" / (digest + ".json")
    cert = win.signed(compound, certificate_sha256)
    offer_name = local_path(pin["offer_name"])
    require("/" not in offer_name, "Invalid local offer filename")
    offer_path = base / "archives/offers" / offer_name
    offer = win.signed(offer_path, pin["offer_sha256"])
    provider_dir = base / "archives/provider" / family
    cloud_proof = provider_dir / "cloud-proof.json"
    require(
        cert["schema"] == "el-stage2-family-cloud-archive-1" and cert["family"] == family
        and cert["offer_sha256"] == pin["offer_sha256"] and cert["container_sha256"] == digest
        and offer["archive_sha256"] == digest and offer["archive_bytes"] == snapshot["bytes"]
        and offer["family"] == family and offer["fixture"] is False
        and offer["run_manifests"] == snapshot["run_manifests"]
        and sha(provider_dir / "independent-archive-index.json") == offer["archive_index_sha256"]
        and snapshot["run_manifests"] == {k: v["manifest_sha256"] for k, v in cert["receipts"].items()}
        and cert["receipts"],
        "Cloud certificate does not cover the selected family snapshot",
    )
    checked = certify(store, shelf, offer_path, pin["offer_sha256"], cloud_proof, cert["provider_proof_sha256"], location)
    require(checked == cert, "Independent restore certificate changed")
    for rid, item in cert["receipts"].items():
        m, receipt_id = item["manifest_sha256"], item["archive_receipt_id"]
        receipt = load(store / "receipts/archive" / (receipt_id + ".json"))
        shelf_record = shelf / "catalog" / (rid + ".json")
        require(shelf_record.is_file(), "Local original must be verified and parked before snapshot release")
        shelf_marker = load(shelf_record)
        require(
            receipt.get("provider") == "Google Drive" and receipt["container_sha256"] == digest
            and receipt["provider_proof_sha256"] == cert["provider_proof_sha256"]
            and receipt["offer_sha256"] == pin["offer_sha256"] and receipt["remote_object_ids"] == cert["object_ids"]
            and shelf_marker == {"run_id": rid, "manifest_sha256": m, "archive_receipt_id": receipt_id}
            and (not require_parked or (shelf / "runs" / rid).is_dir()),
            "Retained original or independently verified provider receipt differs",
        )
    return {
        "schema": "el-stage2-snapshot-release-1", "family": family, "container_sha256": digest,
        "bytes": snapshot["bytes"], "family_sha256": local["family_sha256"],
        "authorization_sha256": local["authorization_sha256"], "local_marker_sha256": sha(marker),
        "offer_sha256": pin["offer_sha256"], "provider_proof_sha256": cert["provider_proof_sha256"],
        "certificate_sha256": certificate_sha256, "sealed_runs": len(cert["receipts"]),
    }


def release_copy(a):
    store, shelf = locations(a.store, a.shelf)
    cert = win.signed(a.proof, a.proof_sha256)
    family = local_path(cert["family"])
    require("/" not in family and HASH.fullmatch(a.catalog_sha256), "Invalid family or catalog")
    base = store / "stage2" / a.catalog_sha256 / "confirmation"
    folder = a.private_archive.resolve()
    require(
        folder.is_dir() and folder.parent == store.parent and folder not in (store, shelf)
        and a.proof.resolve() == base / "archives" / (cert["container_sha256"] + ".json"),
        "Releasing a snapshot requires its canonical certificate and a dedicated sibling directory",
    )
    marker = base / "archives" / (family + ".released.json")
    packed = folder / (family + ".zip")
    require(not packed.is_symlink(), "Snapshot is a symbolic link")
    record = retained_copy(store, shelf, base, family, a.proof_sha256, require_parked=packed.exists())
    if packed.exists():
        require(packed.is_file() and packed.stat().st_size == record["bytes"]
                and sha(packed) == record["container_sha256"], "Local snapshot bytes differ")
        publish(marker, encoded(record))
        packed.unlink()
    else:
        require(marker.is_file() and marker.read_bytes() == encoded(record), "Missing local snapshot without prior release")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    snap = subs.add_parser("snapshot")
    for k in ("catalog", "store", "shelf", "project", "output"):
        snap.add_argument("--" + k, required=True, type=Path)
    for k in ("catalog-sha256", "family", "family-sha256", "authorization-sha256"):
        snap.add_argument("--" + k, required=True)
    for mode in ("verify", "restore"):
        sub = subs.add_parser(mode)
        sub.add_argument("--container", type=Path, required=True)
        sub.add_argument("--sha256", required=True)
        if mode == "restore":
            sub.add_argument("--destination", required=True, type=Path)
    for command in ("park", "release-copy"):
        sub = subs.add_parser(command)
        for k in ("store", "shelf", "proof"):
            sub.add_argument("--" + k, required=True, type=Path)
        sub.add_argument("--proof-sha256", required=True)
        if command == "release-copy":
            sub.add_argument("--private-archive", type=Path, required=True)
            sub.add_argument("--catalog-sha256", required=True)
    a = p.parse_args(argv)
    if a.mode == "snapshot":
        result = snapshot(a)
    elif a.mode == "park":
        result = park(a)
    elif a.mode == "release-copy":
        result = release_copy(a)
    else:
        result = verify_archive(a.container, a.sha256, getattr(a, "destination", None))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
