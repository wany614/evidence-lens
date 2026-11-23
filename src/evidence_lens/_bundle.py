"""Portable, shared primitives for immutable Evidence Lens run bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "1.0"
PROJECT = "evidence_lens"
KINDS = frozenset(
    "scene_render generation derivation render dataset_build evaluation features "
    "model_fit threshold_select model_score transfer_verify report archive_verify".split()
)
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
RUN_ID = re.compile(r"[0-9]{8}T[0-9]{12}Z-[a-f0-9]{16}")
HASH = re.compile(r"[0-9a-f]{64}")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
REQUIRED = {
    "run.json", "config.resolved.yaml", "environment.json", "command.json",
    "code_snapshot.json", "inputs.json", "events.jsonl", "telemetry.csv",
    "telemetry_schema.json", "stdout.log", "stderr.log", "metrics.csv",
    "metrics.parquet", "artifacts.json",
}


class BundleError(ValueError):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex[:16]


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def save(path: Path, value: object) -> None:
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temp.open("xb") as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load(path: Path) -> dict:
    def unique(pairs: list) -> dict:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise BundleError(f"duplicate JSON key in {path.name}")
        return result

    def invalid(value: str) -> None:
        raise BundleError(f"invalid JSON constant {value}")

    with path.open(encoding="utf-8") as handle:
        result = json.load(handle, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(result, dict):
        raise BundleError(f"{path.name} must contain a JSON object")
    return result


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_path(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise BundleError("invalid empty relative path")
    parts = PurePosixPath(name).parts
    if (not parts or "/".join(parts) != name or name.startswith("/")
            or any(part in {".", ".."} or part.endswith((".", " "))
                   or part.split(".")[0].upper() in
                   {"CON", "PRN", "AUX", "NUL", *(f"LPT{i}" for i in range(1, 10)),
                    *(f"COM{i}" for i in range(1, 10))}
                   for part in parts)
            or any(ord(c) < 32 or c in '<>:"\\|?*' for c in name)):
        raise BundleError(f"nonportable relative path: {name!r}")
    return name


def files(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise BundleError(f"expected regular directory: {root}")
    result = {}
    seen = set()
    def fail(error: OSError) -> None:
        raise error
    for curr, dirs, names in os.walk(root, onerror=fail):
        for name in sorted([*dirs, *names]):
            path = Path(curr) / name
            rel = local_path(path.relative_to(root).as_posix())
            if rel.casefold() in seen:
                raise BundleError(f"case collision: {rel}")
            seen.add(rel.casefold())
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                result[rel] = path
            elif not stat.S_ISDIR(mode):
                raise BundleError(f"links and special files are not permitted: {rel}")
    return dict(sorted(result.items()))


def entry(path: Path, name: str) -> dict:
    return {"path": local_path(name), "bytes": path.stat().st_size, "sha256": sha(path)}


def asset(path: Path, name: str, role: str, status: str) -> dict:
    row = entry(path, name)
    return dict(row, asset_id="sha256:" + row["sha256"], format=path.suffix[1:].lower()
                or "binary", role=role, status=status)


def asset_path(root: Path, digest: str) -> Path:
    if not isinstance(digest, str) or not HASH.fullmatch(digest):
        raise BundleError("malformed shared asset digest")
    return root / "assets" / "sha256" / digest[:2] / digest


def items(record: dict, key: str) -> list:
    rows = record.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise BundleError(f"expected list of objects in {key}")
    return rows


def freeze_tree(root: Path) -> None:
    # Files stay readable on both platforms; sealing is enforced by hashes.
    for path in files(root).values():
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
