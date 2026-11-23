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
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from . import _capacity, _metrics
from ._bundle import (BundleError, HASH, KINDS, LABEL, PROJECT, RUN_ID, SCHEMA, asset, encoded,
                      entry, files, freeze_tree, load, new_id, save, sha, utc)
from ._snapshot import bindings, input_snapshot, snapshot
from ._telemetry import Sampler
from ._verify import (IDENTITY_FIELDS, PARENT_REQUIRED, require, retry_matches,
                      validate_identity_fields, verify_bundle)

DEFAULTS = {"telemetry_interval_seconds": 1.0, "scientific_stage_max_retries": 1,
            "artifact_hash": "sha256"}


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


def resolve_config(path: Path, interval: float | None, limits: dict) -> dict:
    config = load(path)
    recording = {**DEFAULTS, **config.get("recording", {})}
    if interval is not None:
        recording["telemetry_interval_seconds"] = interval
    seconds = recording["telemetry_interval_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            "telemetry interval must be finite and positive")
    require(recording["artifact_hash"] == "sha256" and recording["scientific_stage_max_retries"] == 1,
            "unsupported artifact hash or scientific retry policy")
    estimate = recording.setdefault("expected_output_bytes", 0)
    require(type(estimate) is int and estimate >= 0, "expected_output_bytes must be a non-negative integer")
    recording["storage_limits"] = limits
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


def expand(command: list[str], values: dict[str, str], project: Path) -> list[str]:
    pattern = re.compile(r"\{(python|source|config|out|run_id|input:[A-Za-z0-9_.-]+)\}")
    def substitute(match) -> str:
        key = match.group(1)
        if key not in values:
            raise BundleError(f"command references undeclared input {key}")
        return values[key]
    expanded = []
    for original in command:
        arg = pattern.sub(substitute, original)
        if arg == original and arg.lower().endswith((".py", ".ps1", ".sh")):
            path = Path(arg)
            location = (path if path.is_absolute() else project / path).resolve()
            if location.is_file():
                if not location.is_relative_to(project):
                    raise BundleError("declare external stage scripts as explicit inputs")
                frozen = Path(values["source"]) / location.relative_to(project)
                require(frozen.is_file(), "stage script was excluded from the public source snapshot")
                arg = str(frozen)
        expanded.append(arg)
    return expanded


class RejectedLineage(BundleError):
    pass


def worker_started(bundle: Path, run: dict) -> bool:
    if run.get("worker_exit_code") is not None:
        return True
    with (bundle / "events.jsonl").open(encoding="utf-8") as stream:
        return any(json.loads(line).get("event") == "process_started" for line in stream)


def claim_retry(root: Path, previous: str, rid: str) -> Path:
    from ._index import writer_gate
    with writer_gate(root):
        claims = root / "retry_claims"
        claims.mkdir(exist_ok=True)
        claim = claims / previous
        if claim.exists():
            # Older recorders reserved the slot before Popen. A sealed record
            # can prove that an obsolete reservation never launched a worker.
            owner = claim.read_text(encoding="utf-8").strip()
            if RUN_ID.fullmatch(owner):
                old_path = root / "runs" / owner
                if (old_path / "completion.json").is_file():
                    old = verify_bundle(old_path)["run"]
                    original = old.get("requested_retry_of_run_id") or old["retry_of_run_id"]
                    if old["run_id"] == owner and original == previous and not worker_started(old_path, old):
                        claim.unlink()
        exhausted = "one scientific retry was already allocated for this execution"
        if claim.exists():
            raise RejectedLineage(exhausted)
        for path in (root / "runs").glob("*/run.json"):
            recorded = load(path)
            if recorded.get("retry_of_run_id") == previous and worker_started(path.parent, recorded):
                raise RejectedLineage(exhausted)
        with claim.open("x", encoding="utf-8") as handle:
            handle.write(rid + "\n")
        return claim


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


def close_lingering_group(pid: int) -> bool:
    if os.name != "posix":
        return False
    def alive() -> bool:
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False
    if not alive():
        return False
    for _ in range(10):
        time.sleep(0.05)
        if not alive():
            return False
    try:
        os.killpg(pid, signal.SIGTERM)
        time.sleep(0.15)
        if alive():
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


class RunInterrupted(Exception):
    pass


def collect_payload(root: Path, bundle: Path) -> tuple[list[dict], dict | None]:
    payload = bundle / "payload"
    try:
        rows = [asset(path, "payload/" + name, name, "valid") for name, path in files(payload).items()]
        return rows, None
    except (BundleError, OSError) as error:
        # Rename the original tree on the same volume. Never follow an offending
        # link, sanitize a filename or discard valid siblings from the evidence.
        quarantine = root / "quarantine" / "payloads" / bundle.name
        quarantine.mkdir(parents=True, exist_ok=False)
        retained = None
        if os.path.lexists(payload):
            payload.rename(quarantine / "payload")
            retained = (quarantine / "payload").relative_to(root).as_posix()
        payload.mkdir()
        detail = {"reason": str(error), "path": retained,
                  "receipt": (quarantine / "receipt.json").relative_to(root).as_posix()}
        save(quarantine / "receipt.json", {"schema_version": SCHEMA, "run_id": bundle.name,
                                            "created_at_utc": utc(), **detail})
        return [], detail


def record(args: argparse.Namespace) -> dict:
    root = args.root.expanduser().resolve()
    source = args.source_root.expanduser().resolve()
    limits = _capacity.policy(args.storage_policy, source)
    config = resolve_config(args.config, args.interval, limits)
    identity = config.get("identity", {})
    require(isinstance(identity, dict), "config.identity must be an object")
    identity_fields = {k: identity.get(k, config.get(k)) for k in IDENTITY_FIELDS}
    validate_identity_fields(identity_fields)
    experiment = args.experiment or identity.get("experiment_id") or config.get("experiment_id")
    trial = args.trial or identity.get("trial_id") or config.get("trial_id")
    require(isinstance(experiment, str) and LABEL.fullmatch(experiment)
            and isinstance(trial, str) and LABEL.fullmatch(trial),
            "supply valid --experiment and --trial IDs or include them in config.identity")
    require(config.get("project_id", PROJECT) == PROJECT, "wrong project identity")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(bool(command), "a stage command after -- is required")
    sensitive = re.compile(r"^(?:--?(?:token|password|secret|api[-_]key)(?:=|$)|authorization:\s*bearer\s)", re.I)
    require(not any(sensitive.search(arg) for arg in command), "keep credentials out of recorded command arguments")
    embedded, shared, claimed = (bindings(val) for val in (args.input, args.shared_input, args.input_parent))
    parents = resolve_parents(root, args.parent)
    require(args.kind not in PARENT_REQUIRED or parents, f"{args.kind} requires a pinned upstream run")
    old = None
    if args.retry_of:
        require(bool(RUN_ID.fullmatch(args.retry_of)), "invalid retry ID")
        old_dir = root / "runs" / args.retry_of
        old = verify_bundle(old_dir)
        require(old["run"]["run_id"] == args.retry_of, "retry identity differs from path")
        if not worker_started(old_dir, old["run"]):
            requested = old["run"].get("requested_retry_of_run_id") or old["run"]["retry_of_run_id"]
            action = f"use --retry-of {requested}" if requested else "omit --retry-of for the first scientific attempt"
            raise BundleError(f"selected run stopped before the scientific worker started: {action}")
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
           **identity_fields,
           "hostname": socket.gethostname(), "started_at_utc": utc(), "execution_status": "running",
           "producer_integrity_snapshot": "unsealed", "config_sha256": sha(bundle / "config.resolved.yaml")}
    save(bundle / "run.json", run)
    save(bundle / "environment.json", environment())
    inputs = {"schema_version": SCHEMA, "inputs": [],
              "requested_inputs": [{"role": role, "name": Path(path).name,
                                    "storage": "shared" if role in shared else "embedded",
                                    "requested_parent_run_id": claimed.get(role)}
                                   for role, path in {**embedded, **shared}.items()],
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
    rejected = False
    def interrupted(signum, frame):
        raise RunInterrupted(f"received signal {signum}")
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (bundle / "stdout.log").open("xb") as stdout, (bundle / "stderr.log").open("xb") as stderr:
            try:
                snap = snapshot(source, bundle)
                run.update(code_snapshot_sha256=snap["snapshot_sha256"], code_commit=snap["git_commit"])
                if args.kind in {"scene_render", "generation", "render"} or embedded or shared:
                    staged_shared = [path for path in shared.values() if not Path(path).expanduser().resolve().is_relative_to(root / "assets/sha256")]
                    expected = config["recording"]["expected_output_bytes"] + sum(
                        _capacity.bytes_on_tree(Path(path).expanduser()) for path in [*embedded.values(), *staged_shared])
                    space = _capacity.check(root, expected, limits)
                    events.write("storage_preflight", **space)
                    require(not space["reason"], space["reason"] or "")
                inputs["inputs"], substitutions = input_snapshot(root, bundle, embedded, shared, claimed, parents)
                save(bundle / "inputs.json", inputs)
                if old:
                    try:
                        retry_matches(old, run, inputs, command)
                    except BundleError as error:
                        raise RejectedLineage(str(error)) from error
                for inp in inputs["inputs"]:
                    parent_id = inp["parent_run_id"]
                    if parent_id and (root / "runs" / parent_id).exists():
                        parent = root / "runs" / parent_id
                        assets = {a["sha256"] for a in load(parent / "artifacts.json")["files"]}
                        assets.update(r["sha256"] for i in load(parent / "inputs.json")["inputs"] for r in i["files"])
                        if any(r["sha256"] not in assets for r in inp["files"]):
                            raise RejectedLineage(f"input assets missing from declared parent: {inp['role']}")
                values = {"source": str(bundle / "source"), "out": str(bundle / "payload"),
                          "config": str(bundle / "config.resolved.yaml"), "run_id": rid,
                          "python": sys.executable, **substitutions}
                cmd["argv"] = expand(command, values, source)
                binary = shutil.which(cmd["argv"][0])
                if binary:
                    cmd["executable_sha256"] = sha(Path(binary))
                save(bundle / "command.json", cmd)
                save(bundle / "run.json", run)
                (bundle / "config.resolved.yaml").chmod(0o444)
                events.write("context_frozen", config_sha256=run["config_sha256"],
                             code_snapshot_sha256=run["code_snapshot_sha256"], input_roles=list(embedded) + list(shared))
                env = {**os.environ, "EL_RUN_ID": rid, "EL_RUN_KIND": args.kind,
                       "PYTHONPATH": str(bundle / "source" / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
                claim = claim_retry(root, args.retry_of, rid) if old else None
                try:
                    process = subprocess.Popen(cmd["argv"], cwd=cmd["cwd"], env=env, stdout=stdout,
                                               stderr=stderr, stdin=subprocess.DEVNULL, shell=False,
                                               start_new_session=os.name == "posix")
                except (OSError, ValueError):
                    if claim is not None:
                        claim.unlink()
                    raise
                events.write("process_started", pid=process.pid)
                sampler.launch(process.pid)
                worker_code = process.wait()
                code = worker_code
                status = "succeeded" if code == 0 else "failed"
                reason = None if code == 0 else f"stage exited with code {code}"
                if code in {-signal.SIGINT, 0xC000013A}:
                    status = "cancelled"
                elif code in {-signal.SIGTERM, -9}:
                    status = "interrupted"
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
                status, code = "failed", 1
                rejected = isinstance(error, RejectedLineage)
                reason = f"{type(error).__name__}: {error}"
                stderr.write(traceback.format_exc().encode("utf-8"))
                events.write("exception", message=reason)
            finally:
                if process is not None and close_lingering_group(process.pid):
                    if status == "succeeded":
                        status, code, reason = "failed", 1, "worker exited while child processes were still running"
                    events.write("terminated_lingering_children")
                sampler.close()
        if sampler.failure:
            status, code, reason = "failed", 1, f"resource sampler failed: {sampler.failure}"
            events.write("exception", message=reason)
        artifacts, quarantined_payload = collect_payload(root, bundle)
        if quarantined_payload:
            run["payload_quarantine"] = quarantined_payload
            events.write("payload_quarantined", **quarantined_payload)
            message = f"payload failed collection: {quarantined_payload['reason']}"
            reason = f"{reason}. {message}" if reason else message
            status, code = "failed", code or 1
        try:
            _metrics.export(bundle, run)
        except Exception as error:
            status, code, reason = "failed", 1, f"metric export failed: {error}"
            events.write("exception", message=reason)
            _metrics.export(bundle, run, empty=True)
        events.write("process_finished", exit_code=worker_code, execution_status=status)
        if old and process is None and not rejected:
            # Preserve the request in the failed administrative record. Its
            # inputs or sources may never have frozen, so it cannot be a
            # scientific retry or an executed dependency in the formal graph.
            run["requested_retry_of_run_id"] = args.retry_of
            run["retry_of_run_id"] = None
            events.write("retry_not_started", **inputs["retry_of"])
            inputs["retry_of"] = None
            save(bundle / "inputs.json", inputs)
        run.update(execution_status=status, exit_code=code, worker_exit_code=worker_code,
                   failure_reason=reason, ended_at_utc=utc(), elapsed_seconds_monotonic=time.monotonic() - start,
                   sampled_peak_tree_rss_bytes=sampler.peak_rss, code_snapshot_sha256=snap["snapshot_sha256"],
                   code_commit=snap["git_commit"], producer_integrity_snapshot="sealed")
        save(bundle / "run.json", run)
        events.write("evidence_closed", execution_status=status)
        events.handle.close()
        if status != "succeeded":
            for row in artifacts:
                row["status"] = "partial"
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
        if old and process is not None and not rejected:
            try:
                retry_matches(old, run, inputs, command)
            except BundleError:
                rejected = True
        final = root / ("quarantine" if rejected else "runs") / rid
        final.parent.mkdir(exist_ok=True)
        bundle.rename(final)
        return {"run_id": rid, "path": str(final), "execution_status": status, "exit_code": code,
                "producer_disposition": "quarantined" if rejected else "sealed",
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
    p.add_argument("--storage-policy", type=Path, help="override configs/storage.json")
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
    code = result["exit_code"]
    if -127 <= code < 0:
        return 128 - code
    return code if 0 <= code <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
