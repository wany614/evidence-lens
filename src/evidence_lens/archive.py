"""Verify, receive, pack and re-index immutable local research runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from . import _capacity
from ._bundle import (BundleError, HASH, RUN_ID, SCHEMA, files, freeze_tree, load,
                      local_path, save, sha, utc)
from ._index import connect, index_receipt, index_run, writer_gate
from ._snapshot import commit_shared
from ._verify import parent_directory, require, verify_bundle, verify_dependencies, verify_shared


def store_of(bundle: Path) -> Path:
    require(bundle.parent.name == "runs", "provide --root for bundles outside STORE/runs")
    return bundle.parent.parent


def make_receipt(root: Path, kind: str, **data) -> tuple[Path, dict]:
    receipt_id = uuid.uuid4().hex
    directory = root / "receipts" / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (receipt_id + ".json")
    record = {"schema_version": SCHEMA, "receipt_id": receipt_id, "kind": kind,
              "created_at_utc": utc(), **data}
    save(path, record)
    path.chmod(0o444)
    return path, record


def unpack(source: Path, target: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        members = []
        seen = set()
        for info in archive.infolist():
            name = local_path(info.filename.removesuffix("/"))
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(name.casefold() not in seen and mode in {0, stat.S_IFREG, stat.S_IFDIR}
                    and not (info.flag_bits & 1), f"unsafe or duplicate ZIP member: {name}")
            seen.add(name.casefold())
            members.append((info, name))
        needed = sum(info.file_size for info, _ in members)
        require(needed <= shutil.disk_usage(target.parent).free, "insufficient free space for archive")
        for info, name in members:
            path = target / name
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)


def remove_stage(path: Path) -> None:
    # Windows needs writable mode before removing copies of sealed read-only files.
    def writable_then_retry(operation, name, exc):
        if not isinstance(exc[1], PermissionError):
            raise exc[1]
        os.chmod(name, stat.S_IREAD | stat.S_IWRITE)
        operation(name)
    shutil.rmtree(path, onerror=writable_then_retry)


def project_receipt(root: Path, receipt: Path, out: Path | None = None, result: dict | None = None,
                    upstreams: dict | None = None, shelf: Path | None = None) -> None:
    db = connect(root / "index" / "index.sqlite3")
    try:
        with db:
            for rid, entry in (upstreams or {}).items():
                bundle = parent_directory(root, {"run_id": rid, "manifest_sha256": entry["manifest_sha256"]}, shelf)
                index_run(db, bundle.parent.parent, bundle, entry)
            if out is not None:
                index_run(db, root, out, result)
            index_receipt(db, root, receipt)
    finally:
        db.close()


def configured_shelf(root: Path, shelf: Path | None) -> Path | None:
    supplied = shelf or os.environ.get("EVIDENCE_LENS_ARCHIVE_SHELF")
    if not supplied:
        return None
    path = Path(supplied).resolve()
    require(not path.is_relative_to(root) and not root.is_relative_to(path) and (path / "catalog").is_dir(),
            "supply an existing cold shelf outside the active store")
    return path


def receive(source: Path, root: Path, expected: str | None = None, storage_policy: Path | None = None,
            *, shelf: Path | None = None) -> dict:
    source, root = source.absolute(), root.resolve()
    shelf = configured_shelf(root, shelf)
    require(not source.is_symlink() and (source.is_dir() or source.is_file()),
            "source must be a run directory or ZIP")
    require(not root.is_relative_to(source.resolve()), "destination store is inside the source")
    require(expected is None or HASH.fullmatch(expected), "invalid expected container SHA-256")
    limits = _capacity.policy(storage_policy, Path.cwd())
    start = time.monotonic()
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=staging, prefix="receive-"))
    copy = stage / "bundle"
    container_hash, rid, digest = None, None, None
    size = None
    storage_reuse = {}
    source_kind = "directory" if source.is_dir() else "zip"
    with writer_gate(root):
        try:
            estimate = _capacity.bytes_on_tree(source, count_links=True)
            if source.is_file() and zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as remote:
                    estimate += sum(i.file_size for i in remote.infolist())
            space = _capacity.check(root, estimate, limits)
            if space["reason"]:
                remove_stage(stage)
                receipt, record = make_receipt(
                    root, "transfer", run_id=None, manifest_sha256=None, status="deferred_capacity",
                    reason=space["reason"], storage_preflight=space, source_kind=source_kind,
                    source_name=source.name, container_sha256=None, bytes=None,
                    elapsed_seconds=time.monotonic() - start)
                project_receipt(root, receipt)
                return record
            if source.is_dir():
                require(expected is None, "--sha256 applies to an archive container")
                shutil.copytree(source, copy, symlinks=True)
            else:
                incoming = stage / "incoming.zip"
                shutil.copyfile(source, incoming)
                size, container_hash = incoming.stat().st_size, sha(incoming)
                require(expected is None or container_hash == expected, "container SHA-256 mismatch")
                copy.mkdir()
                unpack(incoming, copy)
            if (copy / "run.json").is_file() and not (copy / "run.json").is_symlink():
                candidate = load(copy / "run.json").get("run_id")
                if isinstance(candidate, str) and RUN_ID.fullmatch(candidate):
                    rid = candidate
            if (copy / "manifest.json").is_file() and not (copy / "manifest.json").is_symlink():
                digest = sha(copy / "manifest.json")
            result = verify_bundle(copy)
            rid, digest = result["run"]["run_id"], result["manifest_sha256"]
            if size is None:
                size = sum(p.stat().st_size for p in files(copy).values())
            out = root / "runs" / rid
            if out.exists():
                existing = verify_bundle(out)
                require(existing["manifest_sha256"] == digest and existing["run"]["run_id"] == rid,
                        "conflict: this run ID already has a different manifest")
            upstreams = {}
            verify_dependencies(result, root, upstreams, shelf=shelf)
            if out.exists():
                status = "identical"
            else:
                out.parent.mkdir(exist_ok=True)
                freeze_tree(copy)
                if sys.platform == "darwin":
                    from ._dedup import compact_inputs
                    storage_reuse = compact_inputs(root, copy, result, upstreams)
                copy.rename(out)
                status = "accepted"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
            quarantine = root / "quarantine" / stage.name
            quarantine.parent.mkdir(exist_ok=True)
            stage.rename(quarantine)
            receipt, record = make_receipt(root, "transfer", run_id=rid, manifest_sha256=digest,
                                      status="quarantined", reason=str(error),
                                      quarantine_path=quarantine.relative_to(root).as_posix(),
                                      source_kind=source_kind, container_sha256=container_hash,
                                      bytes=size, elapsed_seconds=time.monotonic() - start)
            project_receipt(root, receipt)
            return record
        receipt, record = make_receipt(root, "transfer", run_id=rid, manifest_sha256=digest, status=status, local_input_storage=storage_reuse,
                                      reason=None, source_kind=source_kind, container_sha256=container_hash,
                                      bytes=size, elapsed_seconds=time.monotonic() - start)
        project_receipt(root, receipt, out, result, upstreams, shelf)
        remove_stage(stage)
        return record


def pack(bundle: Path, output: Path, root: Path) -> dict:
    start = time.monotonic()
    bundle, output = bundle.resolve(), output.resolve()
    require(not output.is_relative_to(bundle), "container must be outside the sealed bundle")
    require(not output.exists(), "container destination already exists")
    result = verify_bundle(bundle)
    verify_shared(result, root)
    members = {row["path"]: row["sha256"] for row in result["manifest"]["files"]}
    members.update({"manifest.json": result["manifest_sha256"], "completion.json": sha(bundle / "completion.json")})
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + "." + uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            actual = files(bundle)
            require(actual.keys() == members.keys(), "bundle changed during packaging")
            for name, path in actual.items():
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                info.file_size = path.stat().st_size
                h = hashlib.sha256()
                with path.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                        h.update(chunk)
                require(h.hexdigest() == members[name], f"file changed during packaging: {name}")
        digest = sha(temp)
        # An exclusive hard link makes concurrent packs no-overwrite as well.
        try:
            os.link(temp, output)
        except FileExistsError:
            raise
        except OSError:
            # FAT/exFAT and some mounted providers cannot create hard links.
            with output.open("xb") as dst:
                try:
                    with temp.open("rb") as src:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                except OSError:
                    dst.close()
                    output.unlink()
                    raise
    finally:
        temp.unlink(missing_ok=True)
    with writer_gate(root):
        receipt, record = make_receipt(root, "archive", run_id=result["run"]["run_id"],
                                       manifest_sha256=result["manifest_sha256"], status="packed_local",
                                       reason=None, container_sha256=digest, bytes=output.stat().st_size,
                                       container_name=output.name, elapsed_seconds=time.monotonic() - start)
        project_receipt(root, receipt)
    return record


def rebuild(root: Path, shelf: Path | None = None) -> dict:
    root = root.resolve()
    shelf = configured_shelf(root, shelf)
    with writer_gate(root):
        idx = root / "index"
        destination = idx / "index.sqlite3"
        temp = idx / (".rebuild-" + uuid.uuid4().hex + ".sqlite3")
        db = connect(temp)
        packages, count = {}, 0
        try:
            with db:
                directories = sorted((root / "runs").glob("*"))
                if shelf is not None:
                    directories += sorted((shelf / "runs").glob("*"))
                for path in directories:
                    result = verify_bundle(path)
                    require(path.name == result["run"]["run_id"] and path.name not in packages,
                            "duplicate bundle or directory differs from run identity")
                    if path.parent.parent == shelf:
                        pin = {"run_id": path.name, "manifest_sha256": result["manifest_sha256"]}
                        require(parent_directory(root, pin, shelf) == path, "cold shelf contains a second copy")
                    packages[path.name] = result
                    index_run(db, path.parent.parent, path, result)
                # Each node's own edges are checked below; cached parents already
                # have authenticated bytes. Detect cycles separately in this pass.
                checked, shared = set(), set()
                def visit(rid: str, active: frozenset) -> None:
                    if rid in checked:
                        return
                    require(rid not in active, "cyclic run provenance")
                    deps = packages[rid]["inputs"]
                    for ref in [*deps["parents"], *([deps["retry_of"]] if deps["retry_of"] else [])]:
                        require(ref["run_id"] in packages, f"missing upstream run: {ref['run_id']}")
                        visit(ref["run_id"], active | {rid})
                    verify_dependencies(packages[rid], root, packages, shelf=shelf, verified_assets=shared)
                    checked.add(rid)
                for rid in packages:
                    visit(rid, frozenset())
                for path in sorted((root / "receipts").glob("*/*.json")):
                    index_receipt(db, root, path)
                    count += 1
            db.close()
            os.replace(temp, destination)
        finally:
            db.close()
            temp.unlink(missing_ok=True)
    return {"runs": len(packages), "receipts": count, "path": str(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    v = commands.add_parser("verify")
    v.add_argument("bundle", type=Path)
    v.add_argument("--root", type=Path, help="store containing shared assets and upstreams")
    r = commands.add_parser("receive")
    r.add_argument("source", type=Path)
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--sha256", help="expected ZIP SHA-256 from the producer")
    r.add_argument("--storage-policy", type=Path, help="override configs/storage.json")
    for sub in (r, v):
        sub.add_argument("--shelf", type=Path, help="also verify parents in this authenticated cold shelf")
    p = commands.add_parser("pack")
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path)
    b = commands.add_parser("rebuild")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--shelf", type=Path, help="include every verified run in this cold shelf")
    a = commands.add_parser("add-asset", help="import one independently obtained shared file with a pinned SHA-256")
    a.add_argument("file", type=Path)
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--sha256", required=True)
    a.add_argument("--storage-policy", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "verify":
            bundle = args.bundle.resolve()
            result = verify_bundle(bundle)
            root = args.root or (store_of(bundle) if bundle.parent.name == "runs" else None)
            require(root is not None or not (result["inputs"]["parents"] or result["inputs"]["retry_of"]
                                             or result["shared"]), "supply --root to verify external references")
            if root is not None:
                verify_dependencies(result, root.resolve(), shelf=configured_shelf(root.resolve(), args.shelf))
            out = {"run_id": result["run"]["run_id"], "execution_status": result["run"]["execution_status"],
                   "manifest_sha256": result["manifest_sha256"], "integrity": "verified",
                   "manifest_bytes": result["bytes"]}
        elif args.action == "receive":
            out = receive(args.source, args.root, args.sha256, args.storage_policy, shelf=args.shelf)
        elif args.action == "pack":
            bundle = args.bundle.resolve()
            out = pack(bundle, args.output, (args.root or store_of(bundle)).resolve())
        elif args.action == "add-asset":
            require(HASH.fullmatch(args.sha256) is not None, "invalid asset SHA-256")
            root = args.root.resolve()
            root.mkdir(parents=True, exist_ok=True)
            with writer_gate(root):
                limits = _capacity.policy(args.storage_policy, Path.cwd())
                cap = _capacity.check(root, args.file.stat().st_size, limits)
                require(not cap["reason"], cap["reason"] or "")
                installed = commit_shared(args.file, root, args.sha256)
            out = {"asset_id": "sha256:" + args.sha256, "bytes": installed.stat().st_size,
                   "path": installed.relative_to(root).as_posix()}
        else:
            out = rebuild(args.root, args.shelf)
        print(json.dumps(out, ensure_ascii=False))
        return 1 if out.get("status") in {"quarantined", "deferred_capacity"} else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"archive error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
