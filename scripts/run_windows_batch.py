"""Process an explicit list of B0 handoffs sequentially on a project-local Windows worker."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MODULE_OR_DIRECTORY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
DIGEST = re.compile(r"[a-f0-9]{64}")
RUN_ID = re.compile(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checked(path, size, expected_hash):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != size
            or digest(path) != expected_hash):
        raise ValueError(f"source differs from manifest: {path.name}")


def deliver(source, destination):
    """Finish a same-directory temp file before making its final name visible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    size, expected = source.stat().st_size, digest(source)
    if destination.exists():
        checked(destination, size, expected)
        return
    temp = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with source.open("rb") as i, temp.open("xb") as o:
            shutil.copyfileobj(i, o, 1024 * 1024)
        checked(temp, size, expected)
        # On Windows, rename never overwrites an existing target.
        try:
            temp.rename(destination)
        except FileExistsError:
            checked(destination, size, expected)
    finally:
        temp.unlink(missing_ok=True)


def extract(job, directory, staged):
    name = job["name"]
    if (not NAME.fullmatch(name) or not name.endswith(".zip") or not DIGEST.fullmatch(job["sha256"])
            or type(job["bytes"]) is not int or job["bytes"] <= 0):
        raise ValueError("invalid job ZIP metadata")
    dest = staged / job["sha256"]
    if dest.exists():
        marker = json.loads((dest / "handoff.json").read_text(encoding="utf-8"))
        if marker.get("trial_id") != job["trial_id"] or digest(dest / "handoff.json") != job["handoff_sha256"]:
            raise ValueError("existing extracted handoff differs")
        return dest
    source = directory / name
    checked(source, job["bytes"], job["sha256"])
    staged.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staged) as scratch:
        temp = Path(scratch) / "handoff"
        temp.mkdir()
        with zipfile.ZipFile(source) as z:
            seen = set()
            for member in z.infolist():
                key = member.filename
                path = PurePosixPath(key)
                if (path.is_absolute() or not path.parts or any(p.endswith(".") or not MODULE_OR_DIRECTORY.fullmatch(p) for p in path.parts)
                        or key != path.as_posix() or key.casefold() in seen or member.is_dir()):
                    raise ValueError("invalid handoff member path")
                seen.add(key.casefold())
                out = temp / key
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as i, out.open("xb") as o:
                    shutil.copyfileobj(i, o)
        marker = json.loads((temp / "handoff.json").read_text(encoding="utf-8"))
        if marker.get("trial_id") != job["trial_id"] or digest(temp / "handoff.json") != job["handoff_sha256"]:
            raise ValueError("extracted handoff identity differs")
        temp.rename(dest)
    return dest


def run_stage(stage, folders, handoff, weights, job, generation=None):
    worker = handoff / "code/scripts/windows_stage.py"
    command = [sys.executable, str(worker), stage, "--handoff", str(handoff), "--store", str(folders.store),
               "--return-dir", str(folders.outgoing)]
    if stage == "generate":
        command.extend(["--weights", str(weights)])
    else:
        command.extend(["--generation-run", generation])
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONUTF8="1"))
    try:
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        expected_kind = {"generate": "generation", "render": "render"}[stage]
        if (receipt["schema"] != "el-worker-return-1" or not RUN_ID.fullmatch(receipt["run"])
                or receipt["package"] != receipt["run"] + ".zip" or receipt["exit_code"] != result.returncode
                or receipt["trial_id"] != job["trial_id"] or receipt["handoff_sha256"] != job["handoff_sha256"]
                or receipt["run_kind"] != expected_kind):
            raise ValueError("stage receipt does not match the dispatched job")
        if generation:
            runtime = json.loads((folders.store / "runs" / receipt["run"] / "run.json").read_text(encoding="utf-8"))
            if generation not in runtime["parent_run_ids"]:
                raise ValueError("render identifies a different generation parent")
    except (IndexError, KeyError, TypeError, ValueError, OSError) as exc:
        folders.outgoing.mkdir(parents=True, exist_ok=True)
        record = folders.outgoing / ("controller-" + uuid.uuid4().hex + ".json")
        with record.open("x", encoding="utf-8") as stream:
            json.dump({"schema": "el-worker-controller-failure-1", "trial_id": job["trial_id"], "stage": stage,
                       "handoff_sha256": job["handoff_sha256"], "exit_code": result.returncode,
                       "error": str(exc), "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-3000:]}, stream)
        deliver(record, folders.publish / record.name)
        raise RuntimeError(f"worker stage lacks a valid receipt; retained {record.name}") from exc
    filename = folders.outgoing / receipt["package"]
    checked(filename, receipt["bytes"], receipt["sha256"])
    deliver(filename, folders.publish / filename.name)
    marker = folders.outgoing / (receipt["run"] + ".receipt.json")
    if json.loads(marker.read_text(encoding="utf-8")) != receipt:
        raise ValueError("durable outgoing receipt differs from stage output")
    deliver(marker, folders.publish / marker.name)
    print(json.dumps({"trial": receipt["trial_id"], "stage": stage, "status": receipt["status"],
                      "run": receipt["run"], "package": receipt["package"], "sha256": receipt["sha256"],
                      "bytes": receipt["bytes"]}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "staged", "store", "outgoing", "weights", "publish"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="digest of the owner-dispatched exact batch JSON")
    args = parser.parse_args()
    if os.name != "nt" or digest(args.batch) != args.sha256:
        parser.error("run on the designated Windows worker with the expected batch SHA-256")
    local = (args.staged, args.store, args.outgoing, args.weights, Path(sys.executable))
    if len({p.resolve().drive.casefold() for p in local}) != 1:
        parser.error("all worker files, model and interpreter must be on the project volume")
    manifest = json.loads(args.batch.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if (manifest["schema"] != "el-windows-b0-batch-1" or digest(Path(__file__)) != manifest["runner_sha256"]
            or len({job["trial_id"] for job in jobs}) != len(jobs)):
        raise ValueError("wrong batch schema or duplicate trials")
    failures = []
    for job in jobs:
        handoff = extract(job, args.batch.parent, args.staged)
        generation = run_stage("generate", args, handoff, args.weights, job)
        if generation["exit_code"]:
            failures.append(generation["run"])
            continue
        render = run_stage("render", args, handoff, args.weights, job, generation["run"])
        if render["exit_code"]:
            failures.append(render["run"])
    if failures:
        raise SystemExit(f"sealed failed stages: {failures}")


if __name__ == "__main__":
    main()
