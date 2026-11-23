"""Materialize and verify the original Windows recovery worker from the public Git checkout."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from tempfile import TemporaryDirectory
import zipfile


RECORDER_COMMIT = "d70547673ae871e390e797c1a99ab29bf6bb7248"


def source_bytes(project, materials, name, digest):
    parts = PurePosixPath(name).parts
    if not parts or ".." in parts or PurePosixPath(name).is_absolute():
        raise ValueError("invalid worker manifest path")
    if name == "configs/candidate-recipe.json":
        path = project / "protocol/recovery-gsplat-v1.json"
    elif name == "scripts/recovery_windows.py" or parts[:2] == ("src", "recovery"):
        path = project / name
    elif name in ("LICENSE", "configs/storage.json") or parts[:2] == ("src", "evidence_lens"):
        path = project / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            result = subprocess.run(["git", "show", RECORDER_COMMIT + ":" + name], cwd=project,
                                    capture_output=True, check=True)
            return result.stdout
    else:
        path = materials / name
    return path.read_bytes()


def build(project, output):
    materials = project / "reproduce/recovery-worker-v1"
    source_manifest = materials / "SOURCE-MANIFEST.json"
    manifest = json.loads(source_manifest.read_bytes())
    if output.exists():
        raise ValueError("choose a new worker output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        stage = Path(scratch) / "worker"
        stage.mkdir()
        for name, spec in manifest["files"].items():
            data = source_bytes(project, materials, name, spec["sha256"])
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError("worker source differs from the pinned file: " + name)
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (stage / source_manifest.name).write_bytes(source_manifest.read_bytes())
        stage.rename(output)
    return {"files": len(manifest["files"]), "source_tree_sha256": manifest["tree_sha256"],
            "recipe_sha256": manifest["candidate_recipe_sha256"]}


def receive(manifest_file, archive, expected, out):
    manifest = json.loads(manifest_file.read_bytes())
    with archive.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError("different original Windows source archive")
    with zipfile.ZipFile(archive) as z:
        names = [e.filename for e in z.infolist()]
        if sorted(names) != sorted([*manifest["files"], "SOURCE-MANIFEST.json"]):
            raise ValueError("the original source member roster differs")
        if z.read("SOURCE-MANIFEST.json") != manifest_file.read_bytes():
            raise ValueError("the Windows source manifest differs")
        for name, spec in manifest["files"].items():
            data = z.read(name)
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError("changed source archive member " + name)
    shutil.copyfile(archive, out / "source.zip")
    result = {"schema": "el-recovery-worker-source-receipt-1", "zip_sha256": expected,
              "bytes": archive.stat().st_size, "members": len(names),
              "tree_sha256": manifest["tree_sha256"], "candidate_recipe_sha256": manifest["candidate_recipe_sha256"]}
    (out / "source-receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expect")
    args = parser.parse_args()
    if args.archive:
        if not args.manifest or not args.expect:
            parser.error("archive intake requires its separately pinned manifest and SHA-256")
        print(json.dumps(receive(args.manifest, args.archive, args.expect, args.out)))
    else:
        print(json.dumps(build(args.project, args.out.absolute())))
