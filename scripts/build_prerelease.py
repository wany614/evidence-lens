"""Build a private CPU prerelease from committed source and explicitly pinned evidence."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("build_prerelease.py", "release_model.py", "release_identity.py", "release_media.py",
         "release_scope.py", "release_wheel.py", "release_smoke.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_candidate(args):
    require("-rc." in args.version and re.fullmatch(r"[a-f0-9]{40}", args.commit),
            "supply a separate release candidate tag and its full Git commit")
    git = ["git", "-C", str(ROOT)]
    this_entry = subprocess.check_output([*git, "show", args.commit + ":scripts/build_prerelease.py"])
    require(this_entry == Path(__file__).read_bytes(), "running release entry point differs from the specified commit")
    paths = ["src/evidence_lens", "README.md", "LICENSE", "pyproject.toml", *("scripts/" + n for n in TOOLS)]
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch).resolve()
        tree = work / "source"
        tree.mkdir()
        tar = work / "source.tar"
        subprocess.run([*git, "archive", "--format=tar", "--output=" + str(tar), args.commit, "--", *paths], check=True)
        with tarfile.open(tar) as archive:
            for entry in archive:
                target = tree / entry.name
                require(target.resolve().is_relative_to(tree) and (entry.isdir() or entry.isfile()), "unsafe committed source member")
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    require(target.suffix not in (".pyc", ".pyo"), "unexpected compiled Python in archived source")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(entry) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
        fields = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
        # -I ignores inherited PYTHONPATH. The fresh process imports project modules only from Git-extracted files.
        launch = ("import sys; from pathlib import Path; "
                  "root=Path(sys.argv[1]); sys.path[:0]=[str(root/'scripts'),str(root/'src')]; "
                  "from build_prerelease import staged_main; staged_main(root,Path(sys.argv[2]))")
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", launch, str(tree), str(ROOT)],
                              cwd=work, input=json.dumps(fields), stdout=subprocess.PIPE, text=True, check=True)
        return json.loads(proc.stdout)


def staged_main(code, repository):
    import release_model
    from evidence_lens._bundle import encoded, load
    from evidence_lens.scoring_io import new_output
    from release_scope import archive_name, python_version

    release_model.ROOT = repository
    args = argparse.Namespace(**json.load(sys.stdin))
    for key in ("store", "scope", "method", "output", "rebuild_of"):
        if getattr(args,key,None):
            setattr(args,key,Path(getattr(args,key)))
    python_version(args.version)
    kind = "synthetic_fixture" if args.fixture else "heldout_study"
    schema = "el-synthetic-cpu-fixture-1" if args.fixture else "el-private-cpu-candidate-1"
    if args.rebuild_of:
        previous = load(args.rebuild_of)
        require(previous["schema"] == schema and previous["kind"] == kind and previous["version"] == args.version
                and previous["code_commit"] == args.commit, "rebuild pin identifies another source, release or study kind")
    for name, module in list(sys.modules.items()):
        if name == "evidence_lens" or name.startswith("evidence_lens.") or name + ".py" in TOOLS:
            require(Path(module.__file__).resolve().is_relative_to(code), "unbound project module: " + name)
    tools = {n: hashlib.sha256((code/"scripts"/n).read_bytes()).hexdigest() for n in TOOLS}
    git = ["git", "-C", str(repository)]
    with tempfile.TemporaryDirectory() as scratch, new_output(args.output) as out:
        temp = Path(scratch)
        epoch = subprocess.check_output([*git, "show", "-s", "--format=%ct", args.commit], text=True).strip()
        env = dict(os.environ, SOURCE_DATE_EPOCH=epoch, PYTHONHASHSEED="0")
        subprocess.run(["uv", "build", "--wheel", *(["--offline"] if args.offline else []), "--out-dir",
                        str(temp / "dist"), str(code)], check=True, env=env)
        wheel, = (temp / "dist").glob("*.whl")
        options = argparse.Namespace(**vars(args))
        options.output, options.wheel = temp / "payload", wheel
        product = release_model.build(options)
        require(product["kind"] == kind, "fixture and held-out result types differ")
        name = archive_name(args.version, kind)
        archive = out / name
        packed = release_model.package(options.output, archive, product["release_sha256"], product["checksums_sha256"])
        record = {"schema": schema, "kind": kind, "version": args.version, "code_commit": args.commit,
                  "method_freeze_sha256": args.method_sha256, "scope_sha256": args.scope_sha256,
                  "model_id": product["model_id"], "wheel_sha256": product["wheel_sha256"],
                  "release_sha256": product["release_sha256"], "checksums_sha256": product["checksums_sha256"],
                  "demo_sha256": product["demo_sha256"], "name": name, "sha256": packed["sha256"],
                  "bytes": packed["bytes"], "zip_members": packed["members"], "source_date_epoch": int(epoch),
                  "build_tools_sha256": tools}
        if args.rebuild_of:
            require(record == previous, "rebuilt ZIP, wheel, documentation or pinned inputs differ from the selected build")
        (out / "candidate.json").write_bytes(encoded(record))
    print(json.dumps(record, sort_keys=True))


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
