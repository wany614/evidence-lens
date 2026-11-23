"""Verify, receive, pack and re-index immutable local research runs."""

from __future__ import annotations

import argparse
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

from ._bundle import (BundleError, HASH, RUN_ID, SCHEMA, files, freeze_tree, load,
                      local_path, save, sha, utc)
from ._index import connect, index_receipt, index_run, writer_gate
from ._verify import require, verify_bundle, verify_dependencies, verify_shared


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


def receive(source: Path, root: Path, expected: str | None = None) -> dict:
    source, root = source.absolute(), root.resolve()
    require(not source.is_symlink() and (source.is_dir() or source.is_file()),
            "source must be a run directory or ZIP")
    require(not root.is_relative_to(source.resolve()), "destination store is inside the source")
    require(expected is None or HASH.fullmatch(expected), "invalid expected container SHA-256")
    start = time.monotonic()
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=staging, prefix="receive-"))
    copy = stage / "bundle"
    container_hash, rid, digest = None, None, None
    size = None
    source_kind = "directory" if source.is_dir() else "zip"
    with writer_gate(root):
        try:
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
            verify_dependencies(result, root)
            if out.exists():
                status = "identical"
            else:
                out.parent.mkdir(exist_ok=True)
                freeze_tree(copy)
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
            db = connect(root / "index" / "index.sqlite3")
            try:
                with db:
                    index_receipt(db, root, receipt)
            finally:
                db.close()
            return record
        receipt, record = make_receipt(root, "transfer", run_id=rid, manifest_sha256=digest, status=status,
                                      reason=None, source_kind=source_kind, container_sha256=container_hash,
                                      bytes=size, elapsed_seconds=time.monotonic() - start)
        db = connect(root / "index" / "index.sqlite3")
        try:
            with db:
                index_run(db, root, out, result)
                index_receipt(db, root, receipt)
        finally:
            db.close()
        remove_stage(stage)
        return record


def pack(bundle: Path, output: Path, root: Path) -> dict:
    start = time.monotonic()
    bundle, output = bundle.resolve(), output.resolve()
    require(not output.is_relative_to(bundle), "container must be outside the sealed bundle")
    require(not output.exists(), "container destination already exists")
    result = verify_bundle(bundle)
    verify_shared(result, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + "." + uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, path in files(bundle).items():
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o444) << 16
                info.file_size = path.stat().st_size
                with path.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
        digest = sha(temp)
        # An exclusive hard link makes concurrent packs no-overwrite as well.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    with writer_gate(root):
        receipt, record = make_receipt(root, "archive", run_id=result["run"]["run_id"],
                                       manifest_sha256=result["manifest_sha256"], status="packed_local",
                                       reason=None, container_sha256=digest, bytes=output.stat().st_size,
                                       container_name=output.name, elapsed_seconds=time.monotonic() - start)
        db = connect(root / "index" / "index.sqlite3")
        try:
            with db:
                index_receipt(db, root, receipt)
        finally:
            db.close()
    return record


def rebuild(root: Path) -> dict:
    root = root.resolve()
    with writer_gate(root):
        idx = root / "index"
        destination = idx / "index.sqlite3"
        temp = idx / (".rebuild-" + uuid.uuid4().hex + ".sqlite3")
        db = connect(temp)
        packages, count = {}, 0
        try:
            with db:
                for path in sorted((root / "runs").glob("*")):
                    result = verify_bundle(path)
                    require(path.name == result["run"]["run_id"], "bundle directory differs from run identity")
                    packages[path.name] = result
                    index_run(db, root, path, result)
                # Each node's own edges are checked below; cached parents already
                # have authenticated bytes. Detect cycles separately in this pass.
                checked = set()
                def visit(rid: str, active: frozenset) -> None:
                    if rid in checked:
                        return
                    require(rid not in active, "cyclic run provenance")
                    deps = packages[rid]["inputs"]
                    for ref in [*deps["parents"], *([deps["retry_of"]] if deps["retry_of"] else [])]:
                        require(ref["run_id"] in packages, f"missing upstream run: {ref['run_id']}")
                        visit(ref["run_id"], active | {rid})
                    verify_dependencies(packages[rid], root, packages)
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
    p = commands.add_parser("pack")
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path)
    b = commands.add_parser("rebuild")
    b.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "verify":
            bundle = args.bundle.resolve()
            result = verify_bundle(bundle)
            root = args.root or (store_of(bundle) if bundle.parent.name == "runs" else None)
            require(root is not None or not (result["inputs"]["parents"] or result["inputs"]["retry_of"]
                                             or result["shared"]), "supply --root to verify external references")
            if root is not None:
                verify_dependencies(result, root.resolve())
            out = {"run_id": result["run"]["run_id"], "execution_status": result["run"]["execution_status"],
                   "manifest_sha256": result["manifest_sha256"], "integrity": "verified",
                   "manifest_bytes": result["bytes"]}
        elif args.action == "receive":
            out = receive(args.source, args.root, args.sha256)
        elif args.action == "pack":
            bundle = args.bundle.resolve()
            out = pack(bundle, args.output, (args.root or store_of(bundle)).resolve())
        else:
            out = rebuild(args.root)
        print(json.dumps(out, ensure_ascii=False))
        return 1 if out.get("status") == "quarantined" else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"archive error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
