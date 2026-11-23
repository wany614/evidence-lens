"""Run a stage under frozen inputs and configuration, then seal its evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from . import _metrics
from ._bundle import (BundleError, HASH, KINDS, LABEL, PROJECT, RUN_ID, SCHEMA, asset, encoded,
                      entry, files, freeze_tree, load, new_id, save, sha, utc)
from ._snapshot import bindings, input_snapshot, snapshot
from ._telemetry import Sampler
from ._verify import PARENT_REQUIRED, require, retry_matches, verify_bundle

DEFAULTS = {"telemetry_interval_seconds": 1.0, "scientific_stage_max_retries": 1,
            "artifact_hash": "sha256"}
IDENTITY_FIELDS = ("scene_family_id", "scene_id", "scene_version", "scene_hash", "camera_set_id",
                   "camera_set_hash", "dataset_id", "dataset_hash", "model_id", "model_hash",
                   "condition_id", "split")


def environment() -> dict:
    packages = {}
    for name in ("evidence-lens", "psutil", "numpy", "Pillow", "pyarrow", "bpy", "torch",
                 "scikit-learn", "gsplat", "sharp"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"schema_version": SCHEMA, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": sys.version, "packages_in_recorder": packages,
            "memory_total_bytes": psutil.virtual_memory().total, "cpu_logical": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False), "hostname": socket.gethostname(),
            "worker_versions_source": "payload and stdout from executing worker"}


class Events:
    def __init__(self, path: Path, start: float):
        self.handle = path.open("x", encoding="utf-8", buffering=1)
        self.start = start
        self.seq = 0

    def write(self, event: str, **details) -> None:
        self.seq += 1
        record = {"seq": self.seq, "utc": utc(), "elapsed_seconds": time.monotonic() - self.start,
                  "event": event, **details}
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()


def resolve_config(path: Path, interval: float | None) -> dict:
    config = load(path)
    recording = {**DEFAULTS, **config.get("recording", {})}
    if interval is not None:
        recording["telemetry_interval_seconds"] = interval
    seconds = recording["telemetry_interval_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            "telemetry interval must be finite and positive")
    require(recording["artifact_hash"] == "sha256" and recording["scientific_stage_max_retries"] == 1,
            "unsupported artifact hash or scientific retry policy")
    config["recording"] = recording
    return config


def resolve_parents(root: Path, raw: list[str]) -> dict[str, str]:
    parents = {}
    for ref in raw:
        rid, _, pinned = ref.partition("=")
        require(bool(RUN_ID.fullmatch(rid)) and rid not in parents and
                (not pinned or bool(HASH.fullmatch(pinned))), "invalid or duplicate parent reference")
        local = root / "runs" / rid
        if local.exists() or not pinned:
            result = verify_bundle(local)
            require(result["run"]["run_id"] == rid and result["run"]["execution_status"] == "succeeded",
                    "scientific parent must identify a successful bundle")
            require(not pinned or pinned == result["manifest_sha256"], "pinned parent hash differs")
            pinned = result["manifest_sha256"]
        parents[rid] = pinned
    return parents


def expand(command: list[str], values: dict[str, str]) -> list[str]:
    pattern = re.compile(r"\{(python|source|config|out|run_id|input:[A-Za-z0-9_.-]+)\}")
    def substitute(match) -> str:
        key = match.group(1)
        if key not in values:
            raise BundleError(f"command references undeclared input {key}")
        return values[key]
    return [pattern.sub(substitute, arg) for arg in command]


def terminate(process: subprocess.Popen) -> None:
    try:
        handle = psutil.Process(process.pid)
        children = handle.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for item in children:
        try:
            item.terminate()
        except psutil.Error:
            pass
    if process.poll() is None:
        process.terminate()
    _, alive = psutil.wait_procs(children, timeout=3)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class RunInterrupted(Exception):
    pass


def record(args: argparse.Namespace) -> dict:
    root = args.root.expanduser().resolve()
    source = args.source_root.expanduser().resolve()
    config = resolve_config(args.config, args.interval)
    identity = config.get("identity", {})
    require(isinstance(identity, dict), "config.identity must be an object")
    experiment = args.experiment or identity.get("experiment_id") or config.get("experiment_id")
    trial = args.trial or identity.get("trial_id") or config.get("trial_id")
    require(isinstance(experiment, str) and LABEL.fullmatch(experiment)
            and isinstance(trial, str) and LABEL.fullmatch(trial),
            "supply valid --experiment and --trial IDs or include them in config.identity")
    require(config.get("project_id", PROJECT) == PROJECT, "wrong project identity")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(bool(command), "a stage command after -- is required")
    embedded, shared, claimed = (bindings(val) for val in (args.input, args.shared_input, args.input_parent))
    parents = resolve_parents(root, args.parent)
    require(args.kind not in PARENT_REQUIRED or parents, f"{args.kind} requires a pinned upstream run")
    old = None
    if args.retry_of:
        require(bool(RUN_ID.fullmatch(args.retry_of)), "invalid retry ID")
        old = verify_bundle(root / "runs" / args.retry_of)
        require(old["run"]["run_id"] == args.retry_of, "retry identity differs from path")
    rid = new_id()
    bundle = root / "work" / rid
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "payload").mkdir()
    (root / "runs").mkdir(exist_ok=True)
    start = time.monotonic()
    events = Events(bundle / "events.jsonl", start)
    events.write("run_started", run_id=rid, run_kind=args.kind)
    save(bundle / "config.resolved.yaml", config)
    run = {"schema_version": SCHEMA, "project_id": PROJECT, "experiment_id": experiment,
           "trial_id": trial, "run_id": rid, "run_kind": args.kind,
           "parent_run_ids": list(parents), "retry_of_run_id": args.retry_of,
           **{k: identity.get(k, config.get(k)) for k in IDENTITY_FIELDS},
           "hostname": socket.gethostname(), "started_at_utc": utc(), "execution_status": "running",
           "producer_integrity_snapshot": "unsealed", "config_sha256": sha(bundle / "config.resolved.yaml")}
    save(bundle / "run.json", run)
    save(bundle / "environment.json", environment())
    inputs = {"schema_version": SCHEMA, "inputs": [],
              "parents": [{"run_id": k, "manifest_sha256": v} for k, v in parents.items()],
              "retry_of": {"run_id": args.retry_of, "manifest_sha256": old["manifest_sha256"]} if old else None}
    save(bundle / "inputs.json", inputs)
    snap = {"schema_version": SCHEMA, "git_commit": None, "git_dirty": None,
            "snapshot_sha256": hashlib.sha256(encoded([])).hexdigest(), "files": []}
    save(bundle / "code_snapshot.json", snap)
    cmd = {"schema_version": SCHEMA, "argv_template": command, "argv": None,
           "cwd": str(bundle / "source"), "shell": False}
    save(bundle / "command.json", cmd)
    sampler = Sampler(bundle, start, config["recording"]["telemetry_interval_seconds"])
    status, code, reason = "failed", 1, None
    process, worker_code = None, None
    def interrupted(signum, frame):
        raise RunInterrupted(f"received signal {signum}")
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (bundle / "stdout.log").open("xb") as stdout, (bundle / "stderr.log").open("xb") as stderr:
            try:
                snap = snapshot(source, bundle)
                run.update(code_snapshot_sha256=snap["snapshot_sha256"], code_commit=snap["git_commit"])
                inputs["inputs"], substitutions = input_snapshot(root, bundle, embedded, shared, claimed, parents)
                save(bundle / "inputs.json", inputs)
                if old:
                    retry_matches(old, run, inputs)
                for inp in inputs["inputs"]:
                    parent_id = inp["parent_run_id"]
                    if parent_id and (root / "runs" / parent_id).exists():
                        parent = root / "runs" / parent_id
                        assets = {a["sha256"] for a in load(parent / "artifacts.json")["files"]}
                        assets.update(r["sha256"] for i in load(parent / "inputs.json")["inputs"] for r in i["files"])
                        require(all(r["sha256"] in assets for r in inp["files"]),
                                f"input assets missing from declared parent: {inp['role']}")
                values = {"source": str(bundle / "source"), "out": str(bundle / "payload"),
                          "config": str(bundle / "config.resolved.yaml"), "run_id": rid,
                          "python": sys.executable, **substitutions}
                cmd["argv"] = expand(command, values)
                save(bundle / "command.json", cmd)
                save(bundle / "run.json", run)
                (bundle / "config.resolved.yaml").chmod(0o444)
                events.write("context_frozen", config_sha256=run["config_sha256"],
                             code_snapshot_sha256=run["code_snapshot_sha256"], input_roles=list(embedded) + list(shared))
                env = {**os.environ, "EL_RUN_ID": rid, "EL_RUN_KIND": args.kind,
                       "PYTHONPATH": str(bundle / "source" / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
                process = subprocess.Popen(cmd["argv"], cwd=cmd["cwd"], env=env, stdout=stdout,
                                           stderr=stderr, stdin=subprocess.DEVNULL, shell=False)
                events.write("process_started", pid=process.pid)
                sampler.launch(process.pid)
                worker_code = process.wait()
                code = worker_code
                status = "succeeded" if code == 0 else "failed"
                reason = None if code == 0 else f"stage exited with code {code}"
            except (KeyboardInterrupt, RunInterrupted) as error:
                if process is not None:
                    terminate(process)
                    worker_code = process.returncode
                status, code = ("cancelled", 130) if isinstance(error, KeyboardInterrupt) else ("interrupted", 143)
                reason = str(error) or "keyboard interrupt"
                events.write("interrupted", message=reason)
            except Exception as error:
                if process is not None and process.poll() is None:
                    terminate(process)
                reason = f"{type(error).__name__}: {error}"
                stderr.write(traceback.format_exc().encode("utf-8"))
                events.write("exception", message=reason)
            finally:
                sampler.close()
        if sampler.failure:
            raise BundleError(f"resource sampler failed: {sampler.failure}")
        try:
            _metrics.export(bundle, run)
        except Exception as error:
            status, code, reason = "failed", 1, f"metric export failed: {error}"
            events.write("exception", message=reason)
            _metrics.export(bundle, run, empty=True)
        events.write("process_finished", exit_code=worker_code, execution_status=status)
        run.update(execution_status=status, exit_code=code, worker_exit_code=worker_code,
                   failure_reason=reason, ended_at_utc=utc(), elapsed_seconds_monotonic=time.monotonic() - start,
                   sampled_peak_tree_rss_bytes=sampler.peak_rss, code_snapshot_sha256=snap["snapshot_sha256"],
                   code_commit=snap["git_commit"], producer_integrity_snapshot="sealed")
        save(bundle / "run.json", run)
        events.write("evidence_closed", execution_status=status)
        events.handle.close()
        members = files(bundle / "payload")
        artifacts = [asset(p, "payload/" + n, n, "valid" if code == 0 else "partial") for n, p in members.items()]
        save(bundle / "artifacts.json", {"schema_version": SCHEMA, "files": artifacts})
        manifest = {"schema_version": SCHEMA, "run_id": rid, "parents": inputs["parents"], "retry_of": inputs["retry_of"],
                    "shared_assets": [r for i in inputs["inputs"] if i["storage"] == "shared" for r in i["files"]],
                    "files": [entry(p, n) for n, p in files(bundle).items()]}
        save(bundle / "manifest.json", manifest)
        save(bundle / "completion.json", {"schema_version": SCHEMA, "run_id": rid, "sealed_at_utc": utc(),
                                        "execution_status": status, "exit_code": code,
                                        "manifest_sha256": sha(bundle / "manifest.json")})
        result = verify_bundle(bundle)
        freeze_tree(bundle)
        final = root / "runs" / rid
        bundle.rename(final)
        return {"run_id": rid, "path": str(final), "execution_status": status, "exit_code": code,
                "failure_reason": reason, "manifest_sha256": result["manifest_sha256"],
                "manifest_files": len(manifest["files"]), "manifest_bytes": result["bytes"]}
    finally:
        signal.signal(signal.SIGTERM, previous)
        events.handle.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(".research"))
    p.add_argument("--source-root", type=Path, default=Path.cwd())
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--experiment")
    p.add_argument("--trial")
    p.add_argument("--interval", type=float)
    p.add_argument("--parent", action="append", default=[], metavar="RUN_ID[=MANIFEST_SHA256]")
    p.add_argument("--retry-of")
    p.add_argument("--input", action="append", default=[], metavar="ROLE=PATH")
    p.add_argument("--shared-input", action="append", default=[], metavar="ROLE=FILE")
    p.add_argument("--input-parent", action="append", default=[], metavar="ROLE=RUN_ID")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    try:
        result = record(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"recording error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result["exit_code"] if 0 <= result["exit_code"] <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
