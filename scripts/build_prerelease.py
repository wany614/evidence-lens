"""Build a private CPU prerelease from committed source and explicitly pinned evidence."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from evidence_lens._bundle import encoded, load, local_path
from evidence_lens.scoring_io import new_output
from release_model import ROOT, build, package, require
from release_scope import python_version


def build_candidate(args):
    require("-rc." in args.version and re.fullmatch(r"[a-f0-9]{40}", args.commit),
            "supply a separate release candidate tag and its full Git commit")
    python_version(args.version)
    if args.rebuild_of:
        previous = load(args.rebuild_of)
        require(previous["schema"] == "el-private-cpu-candidate-1" and previous["version"] == args.version
                and previous["code_commit"] == args.commit, "rebuild pin identifies another source or release")
    with tempfile.TemporaryDirectory() as scratch, new_output(args.output) as out:
        temp = Path(scratch)
        code = temp / "source"
        code.mkdir()
        git = ["git", "-C", str(ROOT)]
        tools = {}
        for name in ("build_prerelease.py", "release_model.py", "release_identity.py", "release_media.py",
                     "release_scope.py", "release_wheel.py", "release_smoke.py"):
            raw = subprocess.check_output([*git, "show", args.commit + ":scripts/" + name])
            require(raw == Path(__file__).with_name(name).read_bytes(), "running release tool differs from commit: " + name)
            tools[name] = hashlib.sha256(raw).hexdigest()
        tar = temp / "source.tar"
        subprocess.run([*git, "archive", "--format=tar", "--output=" + str(tar), args.commit, "--", "src/evidence_lens",
                        "README.md", "LICENSE", "pyproject.toml"], check=True)
        with tarfile.open(tar) as tree:
            for entry in tree:
                target = code / local_path(entry.name)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                require(entry.isfile(), "wheel build source must be a regular Git blob")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tree.extractfile(entry) as inp, target.open("xb") as dst:
                    shutil.copyfileobj(inp, dst)
        epoch = subprocess.check_output([*git, "show", "-s", "--format=%ct", args.commit], text=True).strip()
        env = dict(os.environ, SOURCE_DATE_EPOCH=epoch, PYTHONHASHSEED="0")
        subprocess.run(["uv", "build", "--wheel", *(["--offline"] if args.offline else []), "--out-dir",
                        str(temp / "dist"), str(code)], check=True, env=env)
        wheel, = (temp / "dist").glob("*.whl")
        options = argparse.Namespace(**vars(args))
        options.output, options.wheel = temp / "payload", wheel
        product = build(options)
        name = f"evidence-lens-{args.version}-cpu.zip"
        archive = out / name
        packed = package(options.output, archive, product["release_sha256"], product["checksums_sha256"])
        record = {"schema": "el-private-cpu-candidate-1", "version": args.version, "code_commit": args.commit,
                  "method_freeze_sha256": args.method_sha256, "scope_sha256": args.scope_sha256,
                  "model_id": product["model_id"], "wheel_sha256": product["wheel_sha256"],
                  "release_sha256": product["release_sha256"], "checksums_sha256": product["checksums_sha256"],
                  "demo_sha256": product["demo_sha256"], "name": name, "sha256": packed["sha256"],
                  "bytes": packed["bytes"], "zip_members": packed["members"], "source_date_epoch": int(epoch),
                  "build_tools_sha256": tools}
        if args.rebuild_of:
            require(record == previous, "rebuilt ZIP, wheel, documentation or pinned inputs differ from the selected build")
        (out / "candidate.json").write_bytes(encoded(record))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "method", "scope", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("version", "commit", "method-sha256", "scope-sha256", "example", "example-source", "scores", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rebuild-of", type=Path, help="candidate.json from the first build")
    parser.add_argument("--offline", action="store_true", help="build the wheel with uv's existing backend cache")
    parser.add_argument("--fixture", action="store_true", help="accept only a synthetic fixture report")
    args = parser.parse_args()
    try:
        result = build_candidate(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
