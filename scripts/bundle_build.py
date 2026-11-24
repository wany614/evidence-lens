"""Build the universal wheel from recorder-snapshotted sources and pinned released assets."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from bundle_original import check, put, sha


def run(args, **kwargs):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", UV_OFFLINE="1", SOURCE_DATE_EPOCH="946684800")
    cmd = [str(a) for a in args]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, **kwargs)
    if result.returncode:
        raise RuntimeError(
            f"exit {result.returncode}: {cmd[0]}: {result.stderr[-3600:]} {result.stdout[-600:]}"
        )
    return result.stdout


def environment(temp, wheel):
    uv = shutil.which("uv")
    check(uv, "uv executable required for the recorded clean-room installer")
    base = temp / "clean"
    run([uv, "venv", "--offline", "--no-project", "--python", sys.executable, base])
    exe = base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([uv, "pip", "install", "--offline", "--python", exe, wheel, "numpy==2.5.5", "Pillow==12.3.0"])
    inventory = json.loads(
        run(
            [
                exe,
                "-I",
                "-B",
                "-c",
                "import importlib.metadata as m,json,sys;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))",
            ]
        )
    )
    check(
        {s.lower().replace("_", "-") for s in inventory} == {"numpy", "pillow", "evidence-lens"},
        "clean installation acquired unexpected packages",
    )
    return base / ("Scripts/evidence-lens.exe" if os.name == "nt" else "bin/evidence-lens"), inventory


def build(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        scratch = Path(tmp)
        project = scratch / "project"
        project.mkdir()
        shutil.copyfile(a.source / "pyproject.toml", project / "pyproject.toml")
        shutil.copytree(
            a.source / "src",
            project / "src",
            ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "*.egg-info"),
        )
        shutil.copytree(a.model_assets, project / "src/evidence_lens_bundle/assets", dirs_exist_ok=True)
        for name, src in (
            ("README.md", a.readme),
            ("LICENSE", a.license),
            ("NOTICE", a.notice),
            ("THIRD_PARTY_NOTICES.md", a.third_party),
            ("licenses/CC-BY-4.0.txt", a.ccby),
            ("licenses/DEPENDENCY-SOURCES.json", a.deps),
        ):
            dest = project / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        wheels = a.out / "dist"
        wheels.mkdir()
        run([shutil.which("uv"), "build", "--wheel", "--offline", "--out-dir", wheels, project])
        files = list(wheels.glob("*.whl"))
        check(
            len(files) == 1 and files[0].name.endswith("py3-none-any.whl"),
            "universal build did not emit one pure-Python wheel",
        )
        wheel = files[0]
        with ZipFile(wheel) as z:
            nn = z.namelist()
            check(len(nn) == len(set(nn)), "duplicate wheel member")
            record = next(n for n in nn if n.endswith(".dist-info/RECORD"))
            rows = list(csv.reader(io.StringIO(z.read(record).decode())))
            check({n for n, *_ in rows} == set(nn), "incomplete wheel RECORD")
            for n, h, size in rows:
                if n == record:
                    continue
                raw = z.read(n)
                computed = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
                check(h == "sha256=" + computed and int(size) == len(raw), "wrong wheel member: " + n)
                if not any(part.endswith(".dist-info") for part in Path(n).parts):
                    check(
                        raw == (project / "src" / n).read_bytes(),
                        "wheel differs from bound source or model: " + n,
                    )
        cli, inventory = environment(scratch, wheel)
        bundles = {}
        for condition in ("M2", "M4"):
            src = a.cases / condition
            dest = a.out / "bundles" / condition
            reply = json.loads(
                run(
                    [
                        cli,
                        "adapt-legacy",
                        "--case",
                        src,
                        "--case-sha256",
                        sha((src / "case.json").read_bytes()),
                        "--input-domain",
                        "Original Blender offset bell cage, development control",
                        "--out",
                        dest,
                    ]
                )
            )
            validated = json.loads(
                run([cli, "validate", "--bundle", dest, "--bundle-sha256", reply["bundle_sha256"]])
            )
            check(reply == validated, "independent clean validate differs")
            bundles[condition] = validated
        put(
            a.out / "build.json",
            {
                "schema": "el-bundle-clean-build-1",
                "run_id": os.getenv("EL_RUN_ID"),
                "wheel": {
                    "name": wheel.name,
                    "sha256": sha(wheel.read_bytes()),
                    "bytes": wheel.stat().st_size,
                },
                "installed_distributions": inventory,
                "universal_wheel_record_verified": True,
                "bundles": bundles,
                "model_assets": {
                    p.relative_to(a.model_assets).as_posix(): sha(p.read_bytes())
                    for p in sorted(a.model_assets.rglob("*.json"))
                },
                "source_snapshot_sha256": json.loads((a.source.parent / "code_snapshot.json").read_bytes())[
                    "snapshot_sha256"
                ],
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "source",
        "cases",
        "model-assets",
        "readme",
        "license",
        "notice",
        "third-party",
        "ccby",
        "deps",
        "out",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    build(p.parse_args())
