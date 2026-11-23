"""Audit staged public-source candidates and export exactly the named Git blobs."""

import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

POLICY = "configs/public-source.json"


def index_bytes(root, name):
    return subprocess.check_output(["git", "-C", str(root), "show", f":{name}"])


def audit(root):
    # Read both the policy and content from the index to avoid exporting a different revision.
    policy = json.loads(index_bytes(root, POLICY))
    names = {bytes_name.decode() for bytes_name in subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"]).split(b"\0") if bytes_name}
    declared = policy["files"]
    if len(declared) != len(set(declared)) or names != set(declared):
        raise ValueError(f"index differs from explicit file list: {sorted(names ^ set(declared))}")
    blobs = {}
    for name, license_id in declared.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != name or "\\" in name or ":" in name:
            raise ValueError(f"nonportable path {name}")
        if license_id not in policy["license_status"] or not policy["license_status"][license_id]:
            raise ValueError(f"missing license record for {name}")
        mode = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "--", name]).split()[0]
        if mode not in {b"100644", b"100755"}:
            raise ValueError(f"expected regular tracked file: {name}")
        content = index_bytes(root, name)
        content.decode("utf-8")  # P0 source-only; binary examples have their own future release manifest.
        blobs[name] = content
    return policy, blobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--export", type=Path, help="write a new source ZIP when all licenses permit public export")
    args = p.parse_args()
    policy, blobs = audit(args.root)
    if args.export:
        if any(policy["license_status"][v]["public_export"] is not True for v in set(policy["files"].values())):
            raise SystemExit("Source licensing decision is pending; the local audit is available.")
        with zipfile.ZipFile(args.export, mode="x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in sorted(blobs.items()):
                bundle.writestr(name, data)
    print(f"Audited {len(blobs)} exact staged source paths.")


if __name__ == "__main__":
    main()
