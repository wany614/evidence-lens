"""Exercise explicitly malformed copies in the installed wheel without altering any original."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from bundle_build import environment
from bundle_original import check, put, sha


def read(path):
    return json.loads(path.read_bytes())


def bind(path, role):
    raw = path.read_bytes()
    return {"path": role, "bytes": len(raw), "sha256": sha(raw)}


def sample(cli, argv, destination):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([cli, *map(str, argv)], capture_output=True, text=True, env=env)
    try:
        error = json.loads(result.stderr)
    except ValueError:
        error = {}
    lines = result.stderr.strip().splitlines()
    return {
        "exit": result.returncode,
        "status": error.get("status"),
        "message": error.get("message", lines[-1] if lines else ""),
        "traceback": "Traceback (most recent call last)" in result.stderr,
        "output_created": bool(destination and destination.exists()),
    }


def audit(wheel, case, bundle, out, expect):
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        root = Path(tmp)
        cli, packages = environment(root, wheel)
        rows = {}
        for fault in (
            "generator_absent",
            "generator_blank",
            "prediction_views_scalar",
            "camera_views_scalar",
            "broken_npz",
        ):
            dest = root / (fault + "-output")
            legacy = fault in ("generator_absent", "generator_blank", "prediction_views_scalar")
            source = shutil.copytree(case if legacy else bundle, root / fault, copy_function=shutil.copyfile)
            if legacy:
                cm, pm = source / "case.json", source / "prediction.json"
                c, p = read(cm), read(pm)
                if fault == "generator_absent":
                    for obj in (c, p):
                        obj.pop("reconstruction_run_id")
                elif fault == "generator_blank":
                    c["reconstruction_run_id"] = p["reconstruction_run_id"] = "   "
                else:
                    p["views"][next(iter(p["views"]))] = []
                put(pm, p)
                c["prediction"] = bind(pm, "prediction.json")
                put(cm, c)
                argv = [
                    "adapt-legacy",
                    "--case",
                    source,
                    "--case-sha256",
                    sha(cm.read_bytes()),
                    "--out",
                    dest,
                ]
            else:
                bm, pm, cm = source / "bundle.json", source / "producer.json", source / "cameras.json"
                b, p = read(bm), read(pm)
                if fault == "camera_views_scalar":
                    c = read(cm)
                    c["views"][b["target_order"][0]] = None
                    put(cm, c)
                    b["camera"] = bind(cm, "cameras.json")
                    p["camera_sha256"] = b["camera"]["sha256"]
                else:
                    frame = p["frames"][b["target_order"][0]]
                    path = frame["file"]["path"]
                    (source / path).write_bytes(b"PK\x03\x04truncated")
                    frame["file"] = bind(source / path, path)
                put(pm, p)
                b["producer"] = bind(pm, "producer.json")
                put(bm, b)
                argv = ["validate", "--bundle", source, "--bundle-sha256", sha(bm.read_bytes())]
            rows[fault] = sample(cli, argv, dest)
        if expect == "fixed":
            for name, result in rows.items():
                check(
                    result["exit"] == 2
                    and result["status"] == "invalid"
                    and not result["traceback"]
                    and not result["output_created"],
                    name,
                )
                field = (
                    "reconstruction_run_id"
                    if name.startswith("generator")
                    else "NPZ"
                    if name == "broken_npz"
                    else "views"
                )
                check(field in result["message"], name + " error should identify " + field)
        put(
            out / "boundary.json",
            {
                "schema": "el-bundle-boundary-readback-1",
                "run_id": os.environ.get("EL_RUN_ID"),
                "wheel_sha256": sha(wheel.read_bytes()),
                "installed": packages,
                "expect": expect,
                "results": rows,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("wheel", "case", "bundle", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expect", choices=("baseline", "fixed"), required=True)
    audit(**vars(p.parse_args()))
