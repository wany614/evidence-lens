"""Run one explicit frozen batch through the original Windows launcher and publish immutable cumulative receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def spec(path):
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def publish(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == raw, "Different immutable cumulative receipt")
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def completed(job, location):
    path = location / "completed.json"
    summary = load(path)
    require(
        summary["schema"] == "el-t0-windows-return-1"
        and summary["trial_id"] == job["trial_id"]
        and summary["job_sha256"] == job["job_sha256"]
        and summary["package_sha256"] == job["package"]["sha256"],
        "Completion is for another dispatch",
    )
    for name, p in summary["files"].items():
        key = PurePosixPath(name)
        require(
            not key.is_absolute()
            and ".." not in key.parts
            and str(key) == name
            and "\\" not in name
            and ":" not in name,
            "Invalid portable member",
        )
        f = location / key
        require(
            not f.is_symlink() and f.resolve().is_relative_to(location.resolve()) and spec(f) == p,
            "Changed or missing returned member " + name,
        )
    require(
        set(summary["files"])
        == {p.relative_to(location).as_posix() for p in location.rglob("*") if p.is_file() and p != path},
        "Unexpected Windows return member",
    )
    total = sum(p["bytes"] for p in summary["files"].values()) + path.stat().st_size
    return {
        "trial_id": job["trial_id"],
        "job_sha256": job["job_sha256"],
        "package_sha256": job["package"]["sha256"],
        "status": summary["status"],
        "runs": summary["runs"],
        "completed_sha256": digest(path),
        "completed_bytes": path.stat().st_size,
        "files": len(summary["files"]),
        "return_bytes": total,
    }


def execute(registry, expected, store, returns, slot):
    require(os.name == "nt" and digest(registry) == expected, "Use the frozen registry on Windows")
    batch = load(registry)
    jobs = batch["jobs"]
    require(
        batch["schema"] == "el-b2-windows-batch-1"
        and jobs
        and len(jobs) == len({j["trial_id"] for j in jobs}),
        "Duplicate or missing authorized jobs",
    )
    returns.mkdir(parents=True, exist_ok=True)
    rows = []
    for j in jobs:
        name = j["trial_id"]
        require(name.isascii() and all(c.isalnum() or c in "-_" for c in name), "Invalid trial ID")
        folder = registry.parent / name
        launcher = folder / "t0_worker_run.py"
        package = folder / "dispatch.zip"
        require(spec(launcher) == j["launcher"] and spec(package) == j["package"], "Changed job or launcher")
        target = returns / name
        if not (target / "completed.json").exists():
            args = [
                sys.executable,
                "-B",
                str(launcher),
                "--package",
                str(package),
                "--sha256",
                j["package"]["sha256"],
                "--store",
                str(store),
                "--returns",
                str(returns),
                "--gpu-slot",
                str(slot),
            ]
            subprocess.run(args, check=False)
            require(
                (target / "completed.json").exists(),
                "Windows launcher did not seal a return; inspect the recorder before retry",
            )
        row = completed(j, target)
        rows.append(row)
        ok = sum(r["status"] == "succeeded" for r in rows)
        publish(
            returns / (registry.stem + f"-completed-{len(rows):02d}.json"),
            {
                "schema": "el-b2-windows-cumulative-1",
                "batch": registry.stem,
                "batch_sha256": expected,
                "registered_jobs": len(jobs),
                "completed": len(rows),
                "succeeded": ok,
                "failed": len(rows) - ok,
                "jobs": rows,
            },
        )
    return all(r["status"] == "succeeded" for r in rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("registry", "store", "returns", "gpu-slot"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    a = p.parse_args()
    raise SystemExit(0 if execute(a.registry, a.sha256, a.store, a.returns, a.gpu_slot) else 1)
