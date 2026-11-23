"""Audit the staged public-source tree and export its explicitly licensed Git blobs."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

POLICY = "configs/public-source.json"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def staged_tree(root):
    tree = git(root, "write-tree").decode("ascii").strip()
    entries = {}
    for item in git(root, "ls-tree", "-rz", tree).split(b"\0"):
        if item:
            header, path = item.split(b"\t", 1)
            mode, kind, oid = header.split()
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise ValueError(f"expected regular file in staged tree: {path!r}")
            entries[path.decode("utf-8")] = (int(mode, 8), oid.decode("ascii"))
    return tree, entries


def unique_keys(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate key in the source allowlist")
    return obj


def audit(root, entries=None):
    if entries is None:
        _, entries = staged_tree(root)
    if POLICY not in entries:
        raise ValueError("staged tree has no source allowlist")
    policy = json.loads(git(root, "cat-file", "blob", entries[POLICY][1]), object_pairs_hook=unique_keys)
    declared = policy["files"]
    binaries = policy.get("binary_examples", {})
    if not set(binaries) <= set(declared):
        raise ValueError("example manifest includes a file outside the source inventory")
    if set(entries) != set(declared):
        raise ValueError(f"index differs from explicit file list: {sorted(set(entries) ^ set(declared))}")
    blobs = {}
    for name, license_id in declared.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != name or "\\" in name or ":" in name:
            raise ValueError(f"nonportable path {name}")
        if not policy["license_status"].get(license_id):
            raise ValueError(f"missing license record for {name}")
        data = git(root, "cat-file", "blob", entries[name][1])
        if name in binaries:
            pin = binaries[name]
            if (not name.startswith("examples/") or not name.endswith(".png") or pin.get("format") != "rgb8-png"
                    or pin.get("bytes") != len(data) or pin.get("sha256") != hashlib.sha256(data).hexdigest()
                    or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR" or data[24:26] != b"\x08\x02"):
                raise ValueError(f"original PNG example does not match its public inventory: {name}")
        else:
            data.decode("utf-8")  # All other binary assets belong to the separate Release inventory.
        blobs[name] = data
    return policy, blobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--export", type=Path, help="write a new source ZIP when all licenses permit public export")
    args = p.parse_args()
    tree, entries = staged_tree(args.root)
    policy, blobs = audit(args.root, entries)
    if args.export:
        if any(policy["license_status"][v]["public_export"] is not True for v in set(policy["files"].values())):
            raise SystemExit("Source licensing decision is pending; the local audit is available.")
        with zipfile.ZipFile(args.export, mode="x") as bundle:
            bundle.comment = f"git-tree {tree}".encode("ascii")
            for name, data in sorted(blobs.items()):
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = entries[name][0] << 16
                bundle.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
    print(f"Audited {len(blobs)} exact staged source paths in Git tree {tree}.")


if __name__ == "__main__":
    main()
