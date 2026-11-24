"""Exercise the actual installed bundle wheel in a disposable offline Linux CPU container."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import time

import numpy as np
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def inside(args):
    argv = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,size=600m",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
    ]
    for name, path in [
        ("dist", args.dist),
        ("bundle", args.bundle),
        ("wheels", args.wheels),
        ("output", args.out),
    ]:
        argv += ["-v", str(path.resolve()) + ":/" + name + ("" if name == "output" else ":ro")]
    image = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    (whl,) = args.dist.glob("*.whl")
    commands = [
        ["python", "-m", "venv", "/tmp/cpu"],
        ["/tmp/cpu/bin/pip", "install", "--no-index", "--find-links=/wheels", "/dist/" + whl.name],
        ["/tmp/cpu/bin/python", "-c", "import platform; print(platform.platform())"],
        ["/tmp/cpu/bin/pip", "list", "--format=json"],
        ["/tmp/cpu/bin/evidence-lens", "validate", "--bundle", "/bundle"],
        ["/tmp/cpu/bin/evidence-lens", "score", "--bundle", "/bundle", "--out", "/output/scores"],
        [
            "/tmp/cpu/bin/evidence-lens",
            "report",
            "--scores",
            "/output/scores",
            "--out",
            "/output/report.html",
        ],
    ]
    start = time.monotonic()
    p = subprocess.run(
        [*argv, image, "/bin/sh", "-ec", "\n".join(shlex.join(c) for c in commands)],
        capture_output=True,
        text=True,
    )
    (args.out / "linux.log").write_text(p.stdout + p.stderr)
    require(p.returncode == 0, "Linux acceptance failed: " + p.stderr[-480:])
    wall = time.monotonic() - start
    counts = []
    book = load(args.out / "scores/scores.json")
    for row in book["views"]:
        v = row["id"]
        new = args.out / "scores" / row["arrays"]["path"]
        old = args.gold / v / "MS-L1.npz"
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            mask = a["valid"] & b["valid"]
            delta = np.abs(a["risk"][mask] - b["risk"][mask])
            counts.append(
                {
                    "view": v,
                    "pixels": int(mask.size),
                    "compared_finite": int(mask.sum()),
                    "mismatched_valid": int(np.count_nonzero(a["valid"] != b["valid"])),
                    "mismatched_state": int(np.count_nonzero(a["state"] != b["state"])),
                    "max_abs": float(delta.max(initial=0)),
                    "gold_sha256": sha(old),
                    "scores_sha256": sha(new),
                }
            )
    require(
        len(counts) == 6
        and all(r["mismatched_valid"] == r["mismatched_state"] == 0 and r["max_abs"] <= 1e-4 for r in counts),
        "Linux numerical regression changed",
    )
    save(
        args.out / "linux.json",
        {
            "schema": "el-t0-linux-wheel-check-1",
            "run_id": os.environ["EL_RUN_ID"],
            "image": image,
            "wheel_sha256": sha(whl),
            "bundle_sha256": sha(args.bundle / "bundle.json"),
            "cost_install_validate_score_report_seconds": wall,
            "process_environment": book["cost"]["process_environment"],
            "html_sha256": sha(args.out / "report.html"),
            "comparisons": counts,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("dist", "bundle", "wheels", "gold", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--image", default="python:3.12-slim")
    inside(p.parse_args())
