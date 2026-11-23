"""Share sealed Mac input bytes after the worker exits, before the new bundle is published."""

import hashlib
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path

from ._bundle import local_path, sha

MIN_BYTES = 64 * 1024


def compact_inputs(root: Path, bundle: Path, checked: dict, upstreams: dict | None = None) -> dict:
    """A failed storage optimization always leaves the independently verified input copy available."""
    started = time.monotonic()
    linked, saved, skipped = 0, 0, 0
    if sys.platform != "darwin":
        return {"linked_files": 0, "linked_input_bytes": 0, "skipped_files": 0, "elapsed_seconds": 0.0}
    pinned = {r["run_id"]: r["manifest_sha256"] for r in checked["inputs"]["parents"]}
    by_parent, confirmed = {}, set()
    for item in checked["inputs"]["inputs"]:
        parent = item["parent_run_id"]
        if item["storage"] != "embedded" or parent not in pinned:
            continue
        large = [row for row in item["files"] if row["bytes"] >= MIN_BYTES]
        if not large:
            continue
        ancestor = root / "runs" / parent
        if parent not in by_parent:
            candidates = {}
            try:
                if ancestor.resolve() == ancestor and not (ancestor / "manifest.json").is_symlink():
                    raw = (ancestor / "manifest.json").read_bytes()
                    if hashlib.sha256(raw).hexdigest() == pinned[parent]:
                        known = (upstreams or {}).get(parent)
                        manifest = known["manifest"] if known and known["manifest_sha256"] == pinned[parent] else json.loads(raw)
                        # Prefer the parent's scientific artifact over an embedded
                        # copy of the same bytes inherited by that parent.
                        for row in sorted(manifest["files"], key=lambda r: not r["path"].startswith("payload/")):
                            name = local_path(row["path"])
                            if name.startswith(("payload/", "inputs/")) and row["bytes"] >= MIN_BYTES:
                                candidates.setdefault((row["sha256"], row["bytes"]), []).append(ancestor / name)
            except (OSError, ValueError, KeyError, TypeError):
                # A missing externally pinned ancestor or unavailable volume is a
                # normal reason to retain an independent, already verified copy.
                pass
            by_parent[parent] = candidates
        for row in large:
            destination = bundle / row["path"]
            key = (row["sha256"], row["bytes"])
            shared = False
            for source in by_parent[parent].get(key, ()):
                temp = destination.parent / (".parent-link-" + uuid.uuid4().hex)
                try:
                    left, right = source.lstat(), destination.lstat()
                    if (not stat.S_ISREG(left.st_mode) or not stat.S_ISREG(right.st_mode) or
                            left.st_mode & 0o222 or right.st_mode & 0o222 or left.st_size != key[1] or right.st_size != key[1] or
                            source.resolve() != source or destination.resolve() != destination or left.st_dev != right.st_dev):
                        continue
                    if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
                        shared = True
                        break
                    # Creating the temporary link first pins the exact inode that
                    # would be published. Verify both sides and atomically replace
                    # only the new package's private copy, with the worker gone.
                    os.link(source, temp, follow_symlinks=False)
                    actual = temp.lstat()
                    signature = (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, key[0])
                    if signature not in confirmed:
                        if not stat.S_ISREG(actual.st_mode) or actual.st_mode & 0o222 or sha(temp) != key[0]:
                            continue
                        confirmed.add(signature)
                    if sha(destination) != key[0]:
                        continue
                    os.replace(temp, destination)
                    linked += 1
                    saved += key[1]
                    shared = True
                    break
                except OSError:
                    # Cross-device, unsupported and transient link failures retain
                    # the full preflight-charged, independent destination file.
                    continue
                finally:
                    temp.unlink(missing_ok=True)
            if not shared:
                skipped += 1
    return {"linked_files":linked, "linked_input_bytes":saved, "skipped_files":skipped,
            "elapsed_seconds":time.monotonic() - started}
