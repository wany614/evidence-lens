"""Privately preserve original B confirmation recorder packages and explicit external dependencies."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, encoded, freeze_tree, load, local_path, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import make_receipt, project_receipt
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


def provider(a):
    proof = verify_archive(a.container, a.sha256)
    doc, _ = contents(a.container, a.sha256)
    identity = doc["identity"]
    base = a.store / "stage2" / identity["catalog_sha256"] / "confirmation"
    f = doc["family"]
    require(
        isinstance(f, str) and re.fullmatch(r"scene_0[4][1-8]_[a-z0-9_]+", f), "Not a confirmation family"
    )
    group = win.signed(base / "families" / (f + ".json"), identity["family_sha256"])
    auth = win.signed(base / "authorization.json", identity["authorization_sha256"])
    require(
        group["family"] == f
        and f in auth["test_family_ids"]
        and group["authorization_sha256"] == identity["authorization_sha256"]
        and auth["opening_sha256"] == identity["opening_sha256"]
        and auth["catalog_sha256"] == identity["catalog_sha256"],
        "Private archive must match the selected family and opening",
    )
    direct = {
        v["run_id"]: v["manifest_sha256"]
        for cell in group["trials"].values()
        for v in (*cell["run_pins"].values(), *cell["returns"])
    }
    require(
        all(proof["run_manifests"].get(r) == d for r, d in direct.items()),
        "Missing family run in the remote container",
    )
    for rid, d in proof["run_manifests"].items():
        available = [t / "runs" / rid for t in (a.store, a.shelf) if (t / "runs" / rid).is_dir()]
        require(
            len(available) == 1
            and sha(available[0] / "manifest.json") == d
            and load(available[0] / "run.json")["scene_family_id"] == f,
            "Remote candidate includes an unavailable or unrelated recorder bundle",
        )
    require(re.fullmatch(r"[a-zA-Z0-9_-]{8,200}", a.object_id), "Invalid provider object identity")
    url = sys.stdin.readline().strip()
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    require(
        u.scheme == "https"
        and (
            host == "googleapis.com"
            or host.endswith(".googleapis.com")
            or host == "googleusercontent.com"
            or host.endswith(".googleusercontent.com")
            or host in ("drive.google.com", "drive.usercontent.google.com")
        ),
        "Expected a Google Drive provider download",
    )
    start = time.monotonic()
    val = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Evidence-Lens-Research-Integrity"}), timeout=120
    ) as inp:
        while block := inp.read(1 << 20):
            n += len(block)
            require(n <= proof["bytes"], "Unexpected provider size")
            val.update(block)
    require(n == proof["bytes"] and val.hexdigest() == a.sha256, "Independent provider bytes differ")
    root = a.store / "stage2" / doc["identity"]["catalog_sha256"] / "confirmation" / "archives"
    marker = root / (a.sha256 + ".json")
    with writer_gate(a.store):
        if marker.is_file():
            prior = load(marker)
            require(
                prior["object_id"] == a.object_id and prior["container_sha256"] == a.sha256,
                "Existing provider proof differs",
            )
            return prior
        receipts = {}
        for rid, m in proof["run_manifests"].items():
            rp, rec = make_receipt(
                a.store,
                "archive",
                run_id=rid,
                manifest_sha256=m,
                status="verified",
                reason=None,
                container_sha256=a.sha256,
                bytes=n,
                provider="Google Drive",
                remote_object_ids=[a.object_id],
                remote_relative_path=a.remote_name,
                verification_method="Independent provider byte stream and verified private family object map matched the original recorder manifest.",
            )
            project_receipt(a.store, rp)
            receipts[rid] = {"manifest_sha256": m, "archive_receipt_id": rec["receipt_id"]}
        out = {
            "schema": "el-stage2-family-cloud-archive-1",
            "family": doc["family"],
            "container_sha256": a.sha256,
            "object_id": a.object_id,
            "archive_bytes": n,
            "elapsed_seconds": time.monotonic() - start,
            "receipts": receipts,
        }
        publish(marker, encoded(out))
        return out


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    snap = subs.add_parser("snapshot")
    for k in ("catalog", "store", "shelf", "project", "output"):
        snap.add_argument("--" + k, required=True, type=Path)
    for k in ("catalog-sha256", "family", "family-sha256", "authorization-sha256"):
        snap.add_argument("--" + k, required=True)
    for mode in ("verify", "restore", "provider"):
        sub = subs.add_parser(mode)
        sub.add_argument("--container", type=Path, required=True)
        sub.add_argument("--sha256", required=True)
        if mode == "restore":
            sub.add_argument("--destination", required=True, type=Path)
        if mode == "provider":
            sub.add_argument("--store", type=Path, required=True)
            sub.add_argument("--shelf", type=Path, required=True)
            sub.add_argument("--object-id", required=True)
            sub.add_argument("--remote-name", required=True)
    sub = subs.add_parser("park")
    for k in ("store", "shelf", "proof"):
        sub.add_argument("--" + k, required=True, type=Path)
    sub.add_argument("--proof-sha256", required=True)
    a = p.parse_args(argv)
    if a.mode == "snapshot":
        result = snapshot(a)
    elif a.mode == "provider":
        result = provider(a)
    elif a.mode == "park":
        result = park(a)
    else:
        result = verify_archive(a.container, a.sha256, getattr(a, "destination", None))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
