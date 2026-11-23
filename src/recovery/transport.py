"""Preserve the original recorder ZIP and describe its recoverable shared-input closure."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from evidence_lens._bundle import load, save, sha
from evidence_lens._snapshot import commit_shared
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import pack
from recovery.contract import HASH, RUN


def resources(bundle, run):
    with (bundle / "telemetry.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    device = [gpu["memory.used"] for row in rows for gpu in json.loads(row.get("gpu_devices") or "[]")
              if gpu.get("memory.used") is not None]
    proc = [int(float(row["gpu_process_memory_bytes"])) for row in rows if row.get("gpu_process_memory_bytes")]
    reasons = {json.loads(row.get("missing_reason") or "{}").get("gpu_process_memory_bytes") for row in rows}
    wall = bundle / "payload/process-wall.json"
    alloc = {}
    for filename in ("reconstruction.json", "outputs.json", "ed-probe.json", "forced-topology.json"):
        path = bundle / "payload" / filename
        if path.is_file():
            record = load(path)
            alloc = record.get("resources", record.get("runtime", {}))
            break
    return {"recorder_wall_seconds": run["elapsed_seconds_monotonic"], "numeric_process_wall": load(wall) if wall.is_file() else None,
            "torch_allocated_peak_bytes": alloc.get("torch_allocated_peak_bytes"),
            "torch_reserved_peak_bytes": alloc.get("torch_reserved_peak_bytes"),
            "sampled_device_memory_peak_MiB": max(device, default=None), "device_memory_samples": len(device),
            "sampled_process_memory_peak_bytes": max(proc, default=None), "process_memory_samples": len(proc),
            "process_counter_missing_reasons": sorted(x for x in reasons if x)}


def retain(src, dst, digest):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_name("." + uuid.uuid4().hex + ".part")
        try:
            shutil.copyfile(src, tmp)
            require(sha(tmp) == digest, "Copy differs from the sealed input")
            try:
                os.link(tmp, dst)
            except FileExistsError:
                pass
            except OSError:
                with tmp.open("rb") as source, dst.open("xb") as dest:
                    shutil.copyfileobj(source, dest)
        finally:
            tmp.unlink(missing_ok=True)
    require(sha(dst) == digest, "Existing shared asset differs")


def deliver(store, returns, rid):
    bundle = store / "runs" / rid
    verified = verify_bundle(bundle)
    run = verified["run"]
    cfg = load(bundle / "config.resolved.yaml")
    returns.mkdir(parents=True, exist_ok=True)
    dest = returns / (rid + ".zip")
    if dest.exists():
        expected = {r["path"]: r["sha256"] for r in verified["manifest"]["files"]}
        expected.update({"manifest.json": verified["manifest_sha256"], "completion.json": sha(bundle / "completion.json")})
        with zipfile.ZipFile(dest) as z:
            require(len(z.namelist()) == len(expected) and set(z.namelist()) == set(expected), "Return ZIP membership")
            for name, want in expected.items():
                with z.open(name) as f:
                    require(hashlib.file_digest(f, "sha256").hexdigest() == want, "Return ZIP content")
        digest = sha(dest)
    else:
        digest = pack(bundle, dest, store)["container_sha256"]
    assets = {}
    locations = cfg.get("artifact_origins", {})
    for member in verified["inputs"]["inputs"]:
        if member["storage"] != "shared":
            continue
        row = member["files"][0]
        hashval = row["sha256"]
        if hashval in assets:
            continue
        origin = locations.get(member["role"])
        side = returns / (origin["run_id"] + ".receipt.json") if origin else None
        if side and side.is_file() and (returns / (origin["run_id"] + ".zip")).is_file():
            producer = load(side)
            zpath = returns / (origin["run_id"] + ".zip")
            require(producer["manifest_sha256"] == origin["manifest_sha256"] and sha(zpath) == producer["sha256"],
                    "Checkpoint producer return differs")
            with zipfile.ZipFile(zpath) as z, z.open(origin["member"]) as f:
                require(hashlib.file_digest(f, "sha256").hexdigest() == hashval, "Producer payload hash differs")
            location = {"package": zpath.name, "package_sha256": producer["sha256"], "member": origin["member"]}
        else:
            rel = f"assets/sha256/{hashval[:2]}/{hashval}"
            retain(store / row["path"], returns / rel, hashval)
            location = {"file": rel}
        assets[hashval] = {"bytes": row["bytes"], **location}
    receipt = {"schema": "el-recovery-return-1", "run": rid, "trial_id": run["trial_id"], "run_kind": run["run_kind"],
               "manifest_sha256": verified["manifest_sha256"], "status": run["execution_status"], "exit_code": run["exit_code"],
               "package": dest.name, "bytes": dest.stat().st_size, "sha256": digest, "shared_inputs": assets,
               "cell_sha256": cfg.get("cell_sha256"), "request_sha256": cfg.get("request_sha256"),
               "resources": resources(bundle, run)}
    side = returns / (rid + ".receipt.json")
    if side.exists():
        previous = load(side)
        require(all(previous.get(k) == value for k, value in receipt.items() if k != "shared_inputs") and
                set(previous["shared_inputs"]) == set(assets), "Existing original return receipt differs")
        return previous
    save(side, receipt)
    return receipt


def restore(receipt_path: Path, store: Path):
    receipt = load(receipt_path)
    require(RUN.fullmatch(receipt["run"]) and receipt["package"] == receipt["run"] + ".zip" and
            sha(receipt_path.parent / receipt["package"]) == receipt["sha256"], "Wrong run container")
    root = receipt_path.parent
    store.mkdir(exist_ok=True, parents=True)
    for digest, desc in receipt["shared_inputs"].items():
        require(HASH.fullmatch(digest), "Malformed asset digest")
        if "file" in desc:
            require(desc["file"] == f"assets/sha256/{digest[:2]}/{digest}", "Noncanonical asset location")
            source = root / desc["file"]
            require(source.stat().st_size == desc["bytes"], "Shared asset size")
            commit_shared(source, store, digest)
        else:
            name = desc["package"]
            require(name.endswith(".zip") and RUN.fullmatch(name[:-4]) and sha(root / name) == desc["package_sha256"],
                    "Invalid source run ZIP")
            with tempfile.TemporaryDirectory(dir=store) as tmp, zipfile.ZipFile(root / name) as z:
                src = Path(tmp) / "checkpoint.pt"
                with z.open(desc["member"]) as reader, src.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                require(src.stat().st_size == desc["bytes"], "Referenced checkpoint size")
                commit_shared(src, store, digest)
    return {"run": receipt["run"], "assets_restored_or_verified": len(receipt["shared_inputs"])}
