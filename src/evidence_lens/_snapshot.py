"""Freeze the project's distributable executable sources and declared data inputs."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from ._bundle import (BundleError, LABEL, SCHEMA, asset, asset_path, encoded, entry,
                      files, freeze_tree, local_path, save, sha)

SOURCE_ROOTS = {
    "src": {".py", ".pyi"}, "scripts": {".py", ".ps1", ".sh"},
    "tests": {".py"}, "protocol": {".json", ".yaml", ".yml", ".toml"},
    "configs": {".json", ".yaml", ".yml", ".toml"},
    "reproduce": {".py", ".ps1", ".sh"}, "requirements": {".txt"},
}
SOURCE_FILES = {"pyproject.toml", "requirements.txt", "uv.lock"}


def source_name(name: str) -> bool:
    path = Path(name)
    if any(part.startswith(".") or part.lower() in {"private", "local", "__pycache__"}
           for part in path.parts):
        return False
    return (name in SOURCE_FILES or (len(path.parts) > 1
            and path.suffix.lower() in SOURCE_ROOTS.get(path.parts[0], ())))


def git_command(root: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def snapshot(source: Path, bundle: Path) -> dict:
    target = bundle / "source"
    target.mkdir()
    top = git_command(source, "rev-parse", "--show-toplevel")
    is_repo = top is not None and Path(os.fsdecode(top).strip()).resolve() == source
    if is_repo:
        listing = git_command(source, "ls-files", "-co", "--exclude-standard", "-z",
                              "--", *SOURCE_ROOTS, *SOURCE_FILES)
        if listing is None:
            raise BundleError("could not enumerate Git source files")
        names = {os.fsdecode(name) for name in listing.split(b"\0") if name}
    else:
        names = {name for name in SOURCE_FILES if (source / name).is_file()}
        for directory, extensions in SOURCE_ROOTS.items():
            base = source / directory
            if base.is_dir() and not base.is_symlink():
                for p in base.rglob("*"):
                    if p.suffix.lower() in extensions:
                        names.add(p.relative_to(source).as_posix())
    for name in sorted(names):
        if not source_name(name):
            continue
        local_path(name)
        p = source / name
        if not p.exists() and not p.is_symlink():   # tracked deletion in working tree
            continue
        if p.resolve() != p or not p.is_file():
            raise BundleError(f"source must be a regular file: {name}")
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    rows = [entry(p, n) for n, p in files(target).items()]
    if not rows:
        raise BundleError("no distributable source files in --source-root")
    commit = git_command(source, "rev-parse", "--verify", "HEAD") if is_repo else None
    status = git_command(source, "status", "--porcelain", "--", *SOURCE_ROOTS, *SOURCE_FILES) if is_repo else None
    record = {"schema_version": SCHEMA, "git_commit": commit.decode().strip() if commit else None,
              "git_dirty": bool(status) if status is not None else None,
              "snapshot_sha256": hashlib.sha256(encoded(rows)).hexdigest(), "files": rows}
    freeze_tree(target)
    save(bundle / "code_snapshot.json", record)
    return record


def bindings(specifications: list[str]) -> dict[str, str]:
    result = {}
    for specification in specifications:
        role, sep, value = specification.partition("=")
        if not sep or not LABEL.fullmatch(role) or not value or role.casefold() in {r.casefold() for r in result}:
            raise BundleError(f"expected a unique role=value binding: {specification!r}")
        result[role] = value
    return result


def commit_shared(source: Path, store: Path) -> Path:
    pool = store / "assets" / "sha256"
    pool.mkdir(parents=True, exist_ok=True)
    incoming = pool / ("." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, incoming)
        digest = sha(incoming)
        final = asset_path(store, digest)
        final.parent.mkdir(exist_ok=True)
        incoming.chmod(0o444)
        if final.exists():
            if sha(final) != digest:
                raise BundleError(f"shared object damaged: {digest}")
        else:
            try:
                os.link(incoming, final)
            except FileExistsError:
                if sha(final) != digest:
                    raise BundleError(f"shared object conflicted: {digest}")
        return final
    finally:
        if incoming.exists():
            # Windows denies unlink of read-only files. Change temporary link mode
            # only after removing it on POSIX; the CAS link retains read-only mode.
            try:
                incoming.unlink()
            except PermissionError:
                incoming.chmod(0o600)
                incoming.unlink()
                if 'final' in locals() and final.exists():
                    final.chmod(0o444)


def input_snapshot(root: Path, bundle: Path, embedded: dict[str, str], shared: dict[str, str],
                   claimed: dict[str, str], parents: dict[str, str]) -> tuple[list, dict]:
    if set(embedded) & set(shared) or not set(claimed) <= (embedded.keys() | shared.keys()):
        raise BundleError("duplicate input role or parent binding for an unknown role")
    records, substitutions = [], {}
    for role, value in {**embedded, **shared}.items():
        source = Path(value).expanduser().absolute()
        if source.is_symlink() or not source.exists():
            raise BundleError(f"missing or symbolic input: {role}")
        source = source.resolve()
        local_path(source.name)
        parent = claimed.get(role)
        try:
            source_parts = source.relative_to(root / "runs").parts
        except ValueError:
            source_parts = ()
        if source_parts and (parent is None or parent != source_parts[0]):
            if parent is not None:
                raise BundleError(f"incorrect input parent for {role}")
            parent = source_parts[0]
        if parent is not None and parent not in parents:
            raise BundleError(f"input {role} references an undeclared parent")
        is_dir = source.is_dir()
        storage = "shared" if role in shared else "embedded"
        if storage == "shared":
            if is_dir or not source.is_file():
                raise BundleError("--shared-input accepts regular files")
            location = commit_shared(source, root)
            rel = location.relative_to(root).as_posix()
            rows = [asset(location, rel, role, "valid")]
        else:
            if bundle.is_relative_to(source):
                raise BundleError(f"input {role} contains the new run")
            rel = f"inputs/{role}/{source.name}"
            location = bundle / rel
            location.parent.mkdir(parents=True, exist_ok=True)
            if is_dir:
                contents = files(source)
                if not contents:
                    raise BundleError(f"empty input directory: {role}")
                for name, p in contents.items():
                    dst = location / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dst)
                rows = [asset(p, f"{rel}/{n}", role, "valid") for n, p in files(location).items()]
            else:
                if not source.is_file():
                    raise BundleError(f"expected file or directory: {role}")
                shutil.copyfile(source, location)
                rows = [asset(location, rel, role, "valid")]
            freeze_tree(location.parent)
        records.append({"role": role, "kind": "directory" if is_dir else "file",
                        "storage": storage, "name": source.name, "path": rel, "files": rows,
                        "parent_run_id": parent})
        substitutions[f"input:{role}"] = str(location)
    return records, substitutions
