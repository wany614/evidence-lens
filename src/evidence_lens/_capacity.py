"""Measure the non-synchronized project store before writing a new batch."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from ._bundle import BundleError, load

FALLBACK = {"local_free_floor_bytes": 30 * 2**30, "local_active_batch_target_bytes": 20 * 2**30}


def policy(explicit: Path | None, project: Path) -> dict:
    path = explicit or (project / "configs/storage.json")
    config = load(path) if explicit or path.is_file() else FALLBACK
    result = {k: config.get(k) for k in FALLBACK}
    if any(type(n) is not int or n < 0 for n in result.values()):
        raise BundleError("storage bounds must be non-negative integer byte counts")
    return result


def bytes_on_tree(path: Path, *, count_links: bool = False) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        raise BundleError(f"unexpected symbolic input or store: {path.name}")
    if path.is_file():
        return path.stat().st_size
    count = 0
    seen = set()
    def fail(error: OSError) -> None:
        # Another recorder may seal (rename) its work directory during this sample.
        if not isinstance(error, FileNotFoundError):
            raise error
    for parent, _, names in os.walk(path, onerror=fail):
        for name in names:
            try:
                item = (Path(parent) / name).lstat()
            except FileNotFoundError:
                # A disappearing entry consumed no bytes at its old path at stat time.
                continue
            inode = (item.st_dev, item.st_ino)
            if stat.S_ISREG(item.st_mode) and (count_links or not item.st_ino or inode not in seen):
                count += item.st_size
                if not count_links and item.st_nlink > 1:
                    seen.add(inode)
    return count


def check(root: Path, needed: int, limits: dict) -> dict:
    active = bytes_on_tree(root)
    free = shutil.disk_usage(root).free
    reason = None
    if free - needed < limits["local_free_floor_bytes"]:
        reason = "projected free disk below configured floor"
    elif active + needed > limits["local_active_batch_target_bytes"]:
        reason = "projected active batch exceeds configured quota"
    return {"active_bytes": active, "free_bytes": free, "additional_estimated_bytes": needed,
            **limits, "reason": reason}
