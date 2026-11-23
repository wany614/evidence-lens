"""Copy one selected trial to a separate filesystem object archive, then verify retrieved bytes."""

import argparse
import hashlib
import json
import os
import shutil
import errno
import tempfile
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, asset_path, encoded, load, local_path, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require
from formal_lineage import VerifiedRuns, family_record
from make_handoff import publish

ROOT = Path(__file__).resolve().parents[1]


def inventory(store, catalog, lock, family_id):
    family = family_record(catalog, lock, family_id)
    trial = family["trial_id"]
    selected_path = store / "trials" / trial / "trial_selection.json"
    selected = load(selected_path)
    require(selected["selection_status"] == "selected" and selected["condition_id"] == "B0" and
            selected["trial_id"] == trial and selected["scene_family_id"] == family_id, "missing selected B0 trial")
    verified = VerifiedRuns(store)
    for row in selected["attempts"]:
        actual = verified.get(row["run_id"])
        require(row["manifest_sha256"] == actual["manifest_sha256"], "selected attempt manifest differs")
    for row in selected["selected"].values():
        require(row["run_id"] in verified.records and verified.records[row["run_id"]]["manifest_sha256"] == row["manifest_sha256"],
                "selected method stage missing from the attempt inventory")
    directory = store / "formal" / family["dataset_hash"]
    extra = [selected_path]
    extract = directory / "extracts" / (family_id + ".json")
    if extract.exists():
        rec = load(extract)
        require(rec["family_pin"]["selection_sha256"] == sha(selected_path) and
                verified.get(rec["run_id"], "dataset_build")["manifest_sha256"] == rec["manifest_sha256"], "family extract provenance differs")
        extra.append(extract)
    for locator in (directory / "dispatch" / (trial + ".json"), store / "formal-references" / family["dataset_hash"] / (family_id + ".json"),
                    directory / "worker-source.zip", directory / "worker-source.json"):
        if locator.is_file():
            extra.append(locator)
    if "preregistration" in selected:
        pin = selected["preregistration"]
        path = store / local_path(pin["path"])
        require(sha(path) == pin["sha256"], "selected preregistration differs")
        extra.append(path)
    for members in selected["evidence"]["transfer_receipts"].values():
        for receipt in members:
            name = local_path(receipt["receipt_id"] + ".json")
            path = store / "receipts/transfer" / name
            require(sha(path) == receipt["sha256"], "selected transfer receipt differs")
            extra.append(path)
    for path in sorted((store / "receipts").glob("*/*.json")):
        if load(path).get("run_id") in verified.records:
            extra.append(path)
    paths, origins = {}, {}

    def append(name, source, digest=None, size=None):
        name = local_path(name)
        digest = digest or sha(source)
        row = {"sha256": digest, "bytes": source.stat().st_size if size is None else size}
        require(name not in paths or paths[name] == row, "conflicting archive path")
        paths[name] = row
        origins.setdefault(digest, source)

    for rid, record in verified.records.items():
        bundle = store / "runs" / rid
        for row in record["manifest"]["files"]:
            append("runs/" + rid + "/" + row["path"], bundle / row["path"], row["sha256"], row["bytes"])
        append("runs/" + rid + "/manifest.json", bundle / "manifest.json", record["manifest_sha256"])
        append("runs/" + rid + "/completion.json", bundle / "completion.json")
        for obj in record["shared"]:
            path = asset_path(store, obj["sha256"])
            append(path.relative_to(store).as_posix(), path, obj["sha256"], obj["bytes"])
    for path in extra:
        append(path.relative_to(store).as_posix(), path)
    append("protocol/families-v1.json", catalog)
    append("protocol/formal-freeze-v1.json", lock)
    study = catalog.parent / "study-v1.json"
    if study.is_file():
        append("protocol/study-v1.json", study)
    result = {"schema": "el-family-archive-1", "dataset_hash": family["dataset_hash"], "scene_family_id": family_id,
              "trial_id": trial, "split": family["split"], "selection_sha256": sha(selected_path),
              "run_manifests": {rid: r["manifest_sha256"] for rid, r in verified.records.items()}, "files": paths}
    return result, origins


def copy_object(path, src, digest, size):
    if path.exists():
        require(not path.is_symlink() and path.stat().st_size == size and sha(path) == digest, "existing archive object differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".object-", delete=False) as out:
        temp = Path(out.name)
        try:
            with src.open("rb") as inp:
                shutil.copyfileobj(inp, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        require(temp.stat().st_size == size and sha(temp) == digest, "archived bytes differ from source SHA-256")
        temp.chmod(0o444)
        try:
            os.link(temp, path)
        except FileExistsError:
            require(path.stat().st_size == size and sha(path) == digest, "conflicting existing archive object")
        except OSError as error:
            if error.errno not in {errno.ENOTSUP, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}:
                raise
            # Mounted storage without hard links still gets exclusive publication.
            with path.open("xb") as out, temp.open("rb") as inp:
                shutil.copyfileobj(inp, out, 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            require(sha(path) == digest, "archive object publication readback failed")
    finally:
        temp.unlink(missing_ok=True)


def archive(store, vault, catalog, lock, family_id):
    store, vault = store.resolve(), vault.resolve()
    require(not vault.is_relative_to(store) and not store.is_relative_to(vault), "archive must be a separate tree from the active store")
    manifest, origins = inventory(store, catalog, lock, family_id)
    raw = encoded(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    folder = vault / "families" / manifest["dataset_hash"] / family_id
    folder.mkdir(parents=True, exist_ok=True)
    sizes = {r["sha256"]: r["bytes"] for r in manifest["files"].values()}
    with writer_gate(vault):
        needed = sum(n for obj, n in sizes.items() if not asset_path(vault, obj).exists()) + len(raw)
        floor = load(ROOT / "configs/storage.json")["local_free_floor_bytes"]
        require(shutil.disk_usage(vault).free - needed >= floor, "archive destination lacks capacity under the project's free-space floor")
        for obj, src in origins.items():
            copy_object(asset_path(vault, obj), src, obj, sizes[obj])
        dest = folder / (digest + ".json")
        publish(dest, raw)
    return {"manifest": str(dest), "manifest_sha256":digest, "stored_unique_bytes":sum(sizes.values()), "objects":len(origins)}


def verify_vault(vault, manifest_path, expected):
    require(HASH.fullmatch(expected) and sha(manifest_path) == expected, "retrieved family archive manifest SHA-256 differs")
    doc = load(manifest_path)
    require(doc.get("schema") == "el-family-archive-1", "invalid family archive schema")
    sizes = {}
    for path, row in doc["files"].items():
        local_path(path)
        digest, size = row["sha256"], row["bytes"]
        require(HASH.fullmatch(digest) and type(size) is int and size >= 0 and sizes.get(digest, size) == size,
                "archive object identity/size differs")
        sizes[digest] = size
    require(doc["files"]["trials/" + local_path(doc["trial_id"]) + "/trial_selection.json"]["sha256"] == doc["selection_sha256"] and
            doc["files"]["protocol/families-v1.json"]["sha256"] == doc["dataset_hash"], "retrieved selection or catalog pin differs")
    for rid, digest in doc["run_manifests"].items():
        require(RUN_ID.fullmatch(rid) and doc["files"]["runs/" + rid + "/manifest.json"]["sha256"] == digest,
                "archive run is not present under its recorded digest")
    for digest, size in sizes.items():
        path = asset_path(vault, digest)
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == size and sha(path) == digest,
                f"missing or changed retrieved archive object: {digest}")
    return {"schema":"el-family-archive-readback-1", "status":"verified_retrieved_bytes", "manifest_sha256":expected,
            "family":doc["scene_family_id"], "selection_sha256":doc["selection_sha256"], "objects":len(sizes), "bytes":sum(sizes.values())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    copy = stages.add_parser("copy")
    copy.add_argument("--store", required=True, type=Path)
    copy.add_argument("--family", required=True)
    copy.add_argument("--families", type=Path, default=ROOT / "protocol/families-v1.json")
    copy.add_argument("--freeze", type=Path, default=ROOT / "configs/formal-freeze-v1.json")
    verify = stages.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--sha256", required=True)
    verify.add_argument("--receipt", type=Path, help="optional immutable local verification receipt")
    for cmd in (copy, verify):
        cmd.add_argument("--vault", required=True, type=Path)
    args = p.parse_args(argv)
    if args.stage == "copy":
        result = archive(args.store, args.vault, args.families, args.freeze, args.family)
    else:
        result = verify_vault(args.vault, args.manifest, args.sha256)
        if args.receipt:
            publish(args.receipt, encoded(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
