"""Inventory, record and publish numerical costs of the registered reconstruction recovery study."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from evidence_lens._bundle import HASH, RUN_ID, encoded, load, save, sha
from evidence_lens._capacity import bytes_on_tree
from evidence_lens._verify import require, verify_bundle
from recovery_fit_curves import fit_measurements, parse_events, plot

REGISTRATION = "20261123T211839939983Z-b5f7663e8f694647"
EXPERIMENT = "recovery-resource-ledger-v1"
EVENT_FIELDS = (
    "event_index",
    "operation",
    "selected",
    "affected_values",
    "before",
    "after",
    "adam_moments_checked",
    "candidates",
    "available",
    "admitted",
    "blocked",
)
CELL = ("run_id", "population", "recipe_id", "recipe_sha256", "object_id", "condition")
COSTS = (
    "role",
    "platform",
    "runs",
    "succeeded",
    "failed",
    "workers_started",
    "worker_wall_samples",
    "individual_recorder_seconds_sum",
    "worker_wall_seconds_sum",
    "max_sampled_tree_rss_bytes",
    "max_device_used_MiB",
    "max_process_gpu_bytes",
    "max_torch_allocated_bytes",
    "max_torch_reserved_bytes",
    "sealed_payload_bytes_sum",
    "issued_checkpoint_bytes_sum",
    "native_frame_npz_bytes_sum",
    "native_frame_png_bytes_sum",
)


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_rows(path, fields, rows, compressed=False):
    fields = list(fields)
    text = io.StringIO(newline="")
    table = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    table.writeheader()
    table.writerows(rows)
    value = text.getvalue().encode("utf-8")
    with path.open("xb") as f:
        f.write(gzip.compress(value, mtime=0) if compressed else value)


def known(experiment):
    return experiment == "reconstruction-recovery-v1" or experiment.startswith(
        ("r0-", "recovery-", "r-recovery-", "r1-", "r2-")
    )


def storage_snapshot(store):
    return {
        "utc": now(),
        "volume_free_bytes": shutil.disk_usage(store).free,
        "active_logical_bytes": bytes_on_tree(store),
        "subtrees_in_active_store": {
            n: bytes_on_tree(store / n)
            for n in ("runs", "quarantine", "assets", "work", "index", "receipts", "recovery", "r0")
        },
        "subtree_unit": "Logical file bytes with hard links coalesced within each measured subtree. Cross-subtree sums may repeat files.",
    }


def find_pin(root, section, entry):
    return root / section / entry["run_id"]


def locate(store, shelf, pin):
    for base in (store, shelf):
        p = find_pin(base, pin["section"], pin)
        if (p / "completion.json").is_file():
            return p
    raise FileNotFoundError(pin["run_id"])


def discover(store, shelf, bound=None):
    """Use the immutable originals in both stores, including runs excluded from the rebuilt index."""
    found = {}
    for base in (store, shelf):
        for section in ("runs", "quarantine"):
            for path in (base / section).glob("*/completion.json"):
                root = path.parent
                rid = root.name
                if not RUN_ID.fullmatch(rid):
                    continue
                meta = load(root / "run.json")
                if not known(meta["experiment_id"]) or (bound and meta["ended_at_utc"] > bound):
                    continue
                seal = load(path)
                digest = sha(root / "manifest.json")
                require(
                    meta["run_id"] == rid == seal["run_id"] and seal["manifest_sha256"] == digest,
                    "Changed study seal",
                )
                pin = {"run_id": rid, "manifest_sha256": digest, "section": section}
                require(found.setdefault(rid, pin) == pin, "Conflicting recovery run")
    require(REGISTRATION in found, "The original registered protocol is missing")
    return sorted(found.values(), key=lambda x: x["run_id"])


def receipt_pins(store, kind, by_id, cutoff):
    result = []
    for p in sorted((store / "receipts" / kind).glob("*.json")):
        doc = load(p)
        rid = doc.get("run_id")
        if rid in by_id and doc["created_at_utc"] <= cutoff:
            require(
                doc["kind"] == kind and doc.get("manifest_sha256") in (None, by_id[rid]["manifest_sha256"]),
                "Storage receipt identifies another kind or seal",
            )
            result.append({"receipt_id": doc["receipt_id"], "sha256": sha(p)})
    return result


def environment_observation(store):
    path = store / "work/r-control/mac-platform.json"
    if not path.is_file():
        return None
    doc = load(path)
    require(doc["schema"] == "el-recovery-host-observation-1", "Unrecognized control-platform observation")
    return {
        "source_sha256": sha(path),
        "utc": doc["utc"],
        "control": {k: doc["control"][k] for k in ("kernel", "release", "architecture")},
        "docker_context_exit_codes": {k: doc["docker_" + k]["exit"] for k in ("default", "orbstack")},
    }


def provider_reads(store):
    """Report observed original response bodies without publishing provider paths or object IDs."""
    base = store / "work/recovery-resource-readback"
    rows = []
    for name, label, prefix in (
        ("online-train-proof.json", "Stage C training transport cache", ""),
        ("online-dev-original-proof.json", "Stage C development producer archive", "archive_zip_"),
    ):
        path = base / name
        if path.is_file():
            proof = load(path)
            size = proof[prefix + "bytes"]
            digest = proof[prefix + "sha256"]
            require(size > 0 and HASH.fullmatch(digest), "Invalid historical cache provider proof")
            if prefix:
                require(
                    size == sum(x["size"] for x in proof["ordered_parts"]),
                    "Historical split archive size disagrees with its parts",
                )
            rows.append(
                {
                    "purpose": label,
                    "response_body_bytes": size,
                    "response_body_sha256": digest,
                    "objects": len(proof["ordered_parts"]) if prefix else 1,
                    "observed_utc": proof["as_of"] if prefix else proof["utc"],
                    "proof_sha256": sha(path),
                }
            )
    return rows


def source_provider_reads(store, originals):
    path = store / "work/r-control/source-exchange-cloud-readback.json"
    if not path.is_file():
        return []
    doc = load(path)
    result = []
    for cell, item in sorted(doc["files"].items()):
        pin = item["sealed_source"]
        require(
            originals.get(pin["run_id"], {}).get("manifest_sha256") == pin["manifest_sha256"]
            and HASH.fullmatch(item["sha256"])
            and item["bytes"] > 0,
            "Independent source transfer points to a different sealed handoff",
        )
        result.append(
            {
                "cell": cell,
                "run_id": pin["run_id"],
                "manifest_sha256": pin["manifest_sha256"],
                "response_body_sha256": item["sha256"],
                "response_body_bytes": item["bytes"],
                "observed_utc": doc["time_utc"],
                "proof_sha256": sha(path),
            }
        )
    regression = store / "work/r-control/set6-source-cloud-readback.json"
    if regression.is_file():
        receipt = load(regression)
        freeze = receipt["freeze"]
        require(
            receipt["schema"] == "el-recovery-frozen-input-provider-readback-1"
            and originals[freeze["run_id"]]["manifest_sha256"] == freeze["manifest_sha256"],
            "Regression source transfer lacks its frozen study decision",
        )
        for cell, record in sorted(receipt["files"].items()):
            source = record["source_only_handoff"]
            require(
                originals[source["run_id"]]["manifest_sha256"] == source["manifest_sha256"]
                and set(record["members"]) == {"READY.json", "handoff.json", cell + "-input.zip"},
                "Unexpected regression transport members",
            )
            for name, item in sorted(record["members"].items()):
                require(
                    item["bytes"] > 0 and HASH.fullmatch(item["sha256"]),
                    "Unmeasured regression source member",
                )
                result.append(
                    {
                        "cell": cell,
                        "member": name,
                        "run_id": source["run_id"],
                        "manifest_sha256": source["manifest_sha256"],
                        "response_body_sha256": item["sha256"],
                        "response_body_bytes": item["bytes"],
                        "observed_utc": None,
                        "proof_sha256": sha(regression),
                    }
                )
    return result


def returned_packages(store, originals):
    rows = []
    paths = sorted((store / "r0/intake").glob("*/2*.json")) + sorted(
        (store / "recovery/returns").glob("*/transport.json")
    )
    for path in paths:
        doc = load(path)
        if "worker_receipt" in doc:
            sender, receiver = doc["worker_receipt"], doc["local_transfer"]
            rid, digest = sender["run"], sender["manifest_sha256"]
            size, container = sender["bytes"], sender["sha256"]
            request = sender["request_sha256"]
            require(
                doc["pin"] == {"run_id": rid, "manifest_sha256": digest}
                and (doc["zip_bytes"], doc["zip_sha256"]) == (size, container),
                "Changed Windows return",
            )
        else:
            receiver = doc["archive"]
            rid, digest = doc["run_id"], doc["manifest_sha256"]
            size, container, request = doc["container_bytes"], doc["container_sha256"], doc["job_sha256"]
        if rid not in originals:
            continue
        require(
            digest == originals[rid]["manifest_sha256"] == receiver["manifest_sha256"]
            and (rid, size, container)
            == (receiver["run_id"], receiver["bytes"], receiver["container_sha256"])
            and receiver["status"] in {"accepted", "identical"},
            "Sending and verified receiving identities differ",
        )
        rows.append(
            {
                "run_id": rid,
                "manifest_sha256": digest,
                "zip_bytes": size,
                "zip_sha256": container,
                "request_sha256": request,
                "worker_receipt_sha256": doc["receipt_sha256"],
                "local_transfer_receipt_id": receiver["receipt_id"],
                "join_proof_sha256": sha(path),
            }
        )
    return rows


def return_provider_reads(store, originals):
    result = []
    for path in sorted(
        (store.parent / "recovery-provider-returns").glob("r2-*/independent-return-readback.json")
    ):
        doc = load(path)
        pins = doc["original_runs"]
        if any(p["run_id"] not in originals for p in pins.values()):
            continue
        items = doc["files"]
        require(
            all(originals[p["run_id"]]["manifest_sha256"] == p["manifest_sha256"] for p in pins.values())
            and all(r["bytes"] >= 0 and HASH.fullmatch(r["sha256"]) for r in items.values())
            and sum(r["bytes"] for r in items.values()) == doc["independent_bytes"],
            "Provider readback names another return or byte total",
        )
        result.append(
            {
                "trial_id": doc["scientific_trial_id"],
                "original_runs": pins,
                "response_body_bytes": doc["independent_bytes"],
                "objects": len(items),
                "observed_utc": doc["verified_at_utc"],
                "proof_sha256": sha(path),
            }
        )
    return result


def inventory(store, shelf, exchange=None):
    end = storage_snapshot(store)
    pins = discover(store, shelf, end["utc"])
    by_id = {p["run_id"]: p for p in pins}
    root = locate(store, shelf, by_id[REGISTRATION])
    cfg = load(root / "config.resolved.yaml")
    start = cfg["resource_baseline"]
    reg_preflight = next(
        e
        for e in map(json.loads, (root / "events.jsonl").read_text().splitlines())
        if e["event"] == "storage_preflight"
    )
    exchange_files = []
    exact_handoffs = {"recovery-r2-inputs-v1", "recovery-r2-jobs-v1"}
    exchange_started = now() if exchange else None
    if exchange:
        for group in ("Handoffs", "Returns"):
            for batch in sorted((exchange / group).iterdir()):
                exact = group == "Handoffs" and batch.name in exact_handoffs
                if not batch.is_dir() or not (exact or batch.name.startswith(("r-", "r0-", "r1-", "r2-"))):
                    continue
                remote = {}
                proof_sha = None
                proof_path = (
                    store.parent
                    / "recovery-provider-returns"
                    / batch.name
                    / "independent-return-readback.json"
                )
                if group == "Returns" and proof_path.is_file():
                    certification = load(proof_path)
                    require(
                        certification["schema"] == "el-recovery-independent-transfer-1"
                        and all(
                            by_id[p["run_id"]]["manifest_sha256"] == p["manifest_sha256"]
                            for p in certification["original_runs"].values()
                        ),
                        "Different provider readback for mounted return",
                    )
                    remote = certification["files"]
                    proof_sha = sha(proof_path)
                for p in sorted(batch.rglob("*")):
                    if (
                        p.is_symlink()
                        or not p.is_file()
                        or any(v.startswith(".") for v in p.relative_to(batch).parts)
                    ):
                        continue
                    if (
                        not exact
                        and p.suffix.lower() not in (".json", ".zip", ".sha256")
                        and not (p.parent.parent.name == "sha256" and HASH.fullmatch(p.name))
                    ):
                        continue
                    name = p.relative_to(batch).as_posix()
                    size = p.stat().st_size
                    certified = remote.get(name)
                    if certified:
                        require(
                            certified["bytes"] == size and HASH.fullmatch(certified["sha256"]),
                            "Mounted return size differs from independently read original",
                        )
                    exchange_files.append(
                        {
                            "direction": "Mac to Windows" if group == "Handoffs" else "Windows to Mac",
                            "batch": batch.name,
                            "member": name,
                            "bytes": size,
                            "sha256": certified["sha256"] if certified else sha(p),
                            "digest_source": "independent provider readback"
                            if certified
                            else "mounted file content",
                            "provider_proof_sha256": proof_sha if certified else None,
                        }
                    )
    return {
        "schema": "el-recovery-resource-inventory-2",
        "cutoff_utc": end["utc"],
        "start": start,
        "registration_storage_limits": {
            k: reg_preflight[k] for k in ("local_free_floor_bytes", "local_active_batch_target_bytes")
        },
        "end": end,
        "runs": pins,
        "transfer_receipts": receipt_pins(store, "transfer", by_id, end["utc"]),
        "archive_receipts": receipt_pins(store, "archive", by_id, end["utc"]),
        "platform_observation": environment_observation(store),
        "provider_reads": provider_reads(store),
        "source_provider_reads": source_provider_reads(store, by_id),
        "returned_packages": returned_packages(store, by_id),
        "return_provider_reads": return_provider_reads(store, by_id),
        "exchange_files": exchange_files,
        "exchange_inspection_utc": [exchange_started, now()] if exchange else None,
        "exchange_scope": "Named files visible on the project mount. Each digest comes from the indicated full provider readback or a direct mounted-file read."
        if exchange
        else "Mount not observed.",
    }


def role(meta, config):
    exp, trial, kind = meta["experiment_id"], meta["trial_id"], meta["run_kind"]
    attrs = config.get("science", {})
    if exp.startswith("r0-"):
        return {
            "r0-original-checkpoint-cuda": "R0 CUDA endpoint use and gradients",
            "r0-native-camera-projection": "R0 CUDA projection",
            "r0-sealed-cpu-checkpoint-readback": "R0 CPU checkpoints and image boards",
            "r0-native-projection-roundoff-readback": "R0 CPU projection roundoff",
            "r0-checkpoint-evidence-report": "R0 joined report",
        }.get(exp, "R0 diagnostic")
    if exp == EXPERIMENT:
        return "Resource accounting"
    if exp == "recovery-source-extent-readback-v1":
        return "CPU source-camera extent readback"
    if exp == "recovery-independent-schedule-audit":
        return "Independent checkpoint learning-rate readback"
    if trial.startswith("r-freeze-"):
        return "Frozen recipe and stopping rule"
    if trial.startswith("r-worker-source-"):
        return "Windows worker source verification"
    if trial.startswith("r-control-"):
        return "Authored control qualification"
    if trial.startswith("recovery-adapt-"):
        return "Native output adaptation for frozen scoring"
    if exp == "recovery-historical-transport-readback":
        return "Historical cache restoration and capacity audit"
    if exp.startswith("recovery-historical-cache-retirement-"):
        return "Historical cache retirement and capacity audit"
    if exp.startswith("recovery-first-return-cache-"):
        return "Verified Windows return-cache retirement"
    if exp == "recovery-windows-resource-readback-v1":
        return "Independent engineering qualification"
    if trial.startswith("r-dispatch-"):
        return "Scientific dispatch authorization"
    if kind in ("generation", "render"):
        prefix = (
            "Engineering"
            if attrs.get("object_id") == "generated-fixture" or "engineering" in exp
            else "Authored control"
            if attrs.get("object_id") == "scene_057_offset_bell_cage"
            else "DTU"
        )
        return prefix + (" Gaussian optimization" if kind == "generation" else " checkpoint rendering")
    if exp.startswith("r-recovery-") and "engineering" in exp:
        return "Windows engineering verification"
    if "registration" in trial or "recipe" in trial and "freeze" not in trial:
        return "Protocol or recipe registration"
    if "baseline" in trial:
        return "Zero-fit image baseline"
    if "initialization-decision" in trial:
        return "R1 disposition"
    if "public-057" in exp:
        return "Public 057 zero-fit replay"
    if "source-byte-replay" in exp:
        return "Source-only initialization replay"
    if "source-init" in exp:
        return "Source-only geometry initialization"
    if "input-handoff" in exp:
        return "Independent source-only Windows handoff"
    if "audit" in exp or "audit" in trial:
        return "Independent numerical or transport audit"
    if kind == "model_score":
        return "Frozen risk scoring"
    if kind == "evaluation":
        return "Independent evaluation"
    if kind == "report":
        return "Scientific report"
    return "Study decisions and verification"


def public_failure(root, run):
    if run["execution_status"] == "succeeded":
        return None
    reason = str(run.get("failure_reason") or "")
    err = (root / "stderr.log").read_text(encoding="utf8", errors="replace").splitlines()
    if reason.startswith("RejectedLineage: retry changes"):
        return "Frozen retry identity or budget rejected"
    if reason.startswith("RejectedLineage: input assets missing"):
        return "Declared input missing from its selected sealed parent"
    if run.get("exit_code") == 124:
        return "Wall-clock limit"
    if err:
        if match := re.search(r"ModuleNotFoundError: No module named '([a-zA-Z0-9_.]+)'$", err[-1]):
            return "Worker dependency unavailable (" + match[1] + ")"
        if re.match(r"^[A-Za-z][\w.]*(Error|Exception):", err[-1]):
            return err[-1].split(":", 1)[0]
    if "capacity" in reason.lower() or "storage" in reason.lower():
        return "Storage or capacity preflight"
    return "See sealed failure and stderr"


def checked(root, pin):
    result = verify_bundle(root)
    require(
        result["run"]["run_id"] == pin["run_id"] and result["manifest_sha256"] == pin["manifest_sha256"],
        "Changed selected original",
    )
    return result


def read_run(root, pin, tables, asset_store=None):
    result = checked(root, pin)
    meta, cfg = result["run"], load(root / "config.resolved.yaml")
    require(
        known(meta["experiment_id"]) and meta["run_kind"] not in {"model_fit", "threshold_select"},
        "Outside recovery scope",
    )
    files = {r["path"]: r for r in result["manifest"]["files"]}
    samples = []
    with (root / "telemetry.csv").open(encoding="utf-8", newline="") as reader:
        data = list(csv.DictReader(reader))
    missing = set()
    for obs in data:
        g = json.loads(obs.get("gpu_devices") or "[]")
        require(len(g) <= 1, "Multiple GPU devices need separate reporting")
        device = g[0].get("memory.used") if g else None
        process = obs.get("gpu_process_memory_bytes")
        tree = obs.get("tree_rss_bytes")
        reasons = json.loads(obs.get("missing_reason") or "{}")
        if not process and "gpu_process_memory_bytes" in reasons:
            missing.add(reasons["gpu_process_memory_bytes"])
        samples.append(
            {
                "run_id": pin["run_id"],
                "elapsed_seconds": float(obs["elapsed_seconds"]),
                "tree_rss_mib": int(tree) / 2**20 if tree else None,
                "device_used_mib": device,
                "process_gpu_bytes": int(process) if process else None,
            }
        )
    tables["samples"].extend(samples)
    events = [json.loads(l) for l in (root / "events.jsonl").read_text().splitlines() if l]
    checks = [e for e in events if e["event"] == "storage_preflight"]
    runtime = {}
    worker_wall = None
    doc = {}
    payload_kind = None
    for filename, key in (
        ("reconstruction.json", "resources"),
        ("outputs.json", "resources"),
        ("ed-probe.json", "resources"),
        ("forced-topology.json", "resources"),
        ("r0-gpu.json", "runtime"),
        ("centres.json", "runtime"),
    ):
        if "payload/" + filename in files:
            doc = load(root / "payload" / filename)
            runtime = doc.get(key) or doc.get("runtime", {})
            worker_wall = doc.get("worker_wall_seconds", doc.get("wall_seconds", doc.get("elapsed_s")))
            payload_kind = filename
            break

    def peak(values):
        return max((v for v in values if v is not None), default=None)

    def allocated(typ):
        return runtime.get("torch_" + typ + "_peak_bytes", runtime.get("torch_cuda_" + typ + "_peak_bytes"))

    wall = load(root / "payload/process-wall.json") if "payload/process-wall.json" in files else {}
    has_times = [
        e["elapsed_seconds"] for e in events if e["event"] in {"process_started", "process_finished"}
    ]
    if len(has_times) == 2:
        require(has_times[1] >= has_times[0], "Invalid child clock")
    row = {
        **pin,
        "role": role(meta, cfg),
        "experiment": meta["experiment_id"],
        "trial": meta["trial_id"],
        "kind": meta["run_kind"],
        "object_id": cfg.get("science", {}).get("object_id", meta.get("scene_family_id")),
        "condition": cfg.get("science", {}).get("condition", meta.get("condition_id")),
        "platform": load(root / "environment.json")["platform"].split("-")[0],
        "status": meta["execution_status"],
        "exit_code": meta.get("exit_code"),
        "worker_started": any(e["event"] == "process_started" for e in events),
        "failure_class": public_failure(root, meta),
        "stderr_sha256": files["stderr.log"]["sha256"] if meta["execution_status"] != "succeeded" else None,
        "retry_of": meta.get("retry_of_run_id"),
        "parents": " ".join(meta["parent_run_ids"]),
        "source_commit": meta.get("code_commit"),
        "source_snapshot_sha256": meta.get("code_snapshot_sha256"),
        "config_sha256": meta["config_sha256"],
        "started_utc": meta["started_at_utc"],
        "ended_utc": meta["ended_at_utc"],
        "wall_seconds": meta["elapsed_seconds_monotonic"],
        "recorder_child_interval_seconds": has_times[1] - has_times[0] if len(has_times) == 2 else None,
        "worker_wall_seconds": worker_wall,
        "guarded_process_seconds": wall.get("process_seconds"),
        "prior_guarded_seconds": wall.get("prior_seconds"),
        "cumulative_guarded_seconds": wall.get("cumulative_seconds"),
        "resource_cutoff": wall.get("resource_cutoff"),
        "sampled_tree_peak_rss_bytes": meta.get("sampled_peak_tree_rss_bytes"),
        "telemetry_observations": len(samples),
        "device_used_peak_mib": peak(s["device_used_mib"] for s in samples),
        "process_gpu_peak_bytes": peak(s["process_gpu_bytes"] for s in samples),
        "per_process_gpu_missing_reasons": ",".join(sorted(missing)) or None,
        "torch_allocated_peak_bytes": allocated("allocated"),
        "torch_reserved_peak_bytes": allocated("reserved"),
        "preflight_free_bytes": min((e["free_bytes"] for e in checks if "free_bytes" in e), default=None),
        "preflight_active_bytes": max(
            (e["active_bytes"] for e in checks if "active_bytes" in e), default=None
        ),
        "payload_bytes": sum(v["bytes"] for k, v in files.items() if k.startswith("payload/")),
        "manifest_file_bytes": sum(v["bytes"] for v in files.values()),
        "worker_torch": runtime.get("torch"),
        "worker_cuda_runtime": runtime.get("cuda_runtime"),
        "worker_gsplat": runtime.get("gsplat"),
        "worker_gpu_product": runtime.get("device"),
        "checkpoint_payload_bytes": sum(
            v["bytes"] for k, v in files.items() if k.startswith("payload/checkpoints/") and k.endswith(".pt")
        ),
        "native_render_npz_bytes": sum(
            v["bytes"] for k, v in files.items() if k.startswith("payload/step-") and k.endswith(".npz")
        ),
        "native_render_png_bytes": sum(
            v["bytes"] for k, v in files.items() if k.startswith("payload/step-") and k.endswith(".png")
        ),
        "native_render_frames": None,
    }
    if "payload/return-audit.json" in files:
        proof = load(root / "payload/return-audit.json")
        if proof.get("schema") == "el-recovery-mac-return-audit-1" and "original_generation" in proof:
            require(
                proof["run_id"] == meta["run_id"] and proof["passed"], "Wrong independent optimizer readback"
            )
            for key, receipt in proof["checkpoints"].items():
                state = receipt["cpu_restricted_checkpoint_readback"]
                require(
                    all(
                        state[k] is True
                        for k in (
                            "five_finite_fp32_tensors",
                            "five_adam_groups_and_moments_checked",
                            "sampler_phase_counts_rng_and_geometry_checked",
                        )
                    ),
                    "Incomplete independent checkpoint proof",
                )
                tables["audits"].append(
                    {
                        "audit_run_id": pin["run_id"],
                        "audit_manifest_sha256": pin["manifest_sha256"],
                        "audit_member_sha256": files["payload/return-audit.json"]["sha256"],
                        "generation_run_id": proof["original_generation"]["run_id"],
                        "generation_manifest_sha256": proof["original_generation"]["manifest_sha256"],
                        "render_run_id": proof["original_render"]["run_id"],
                        "render_manifest_sha256": proof["original_render"]["manifest_sha256"],
                        "step": int(key),
                        "gaussians": receipt["gaussians"],
                        "parameter_sha256": receipt["parameter_sha256"],
                        "native_frames_checked": len(receipt["views"]),
                        **state,
                    }
                )
    if meta["run_kind"] == "render" and "science" in cfg and payload_kind == "outputs.json":
        row["native_render_frames"] = sum(len(v["views"]) for v in doc["checkpoints"].values())
        for k, info in doc["checkpoints"].items():
            tables["renders"].append(
                {
                    "run_id": pin["run_id"],
                    "object_id": doc["object_id"],
                    "condition": doc["condition"],
                    "step": int(k),
                    "phase": info["phase"],
                    "gaussians": info["gaussians"],
                    "producer_run_id": info.get("producer_run_id"),
                    "checkpoint_sha256": info["checkpoint_sha256"],
                    "source_frames": sum(n.startswith("s") for n in info["views"]),
                    "target_frames": sum(n.startswith("v") for n in info["views"]),
                }
            )
    if payload_kind == "forced-topology.json":
        require(
            doc["passed"] is True and doc["fixture"] == "synthetic-20k-controlled-branches",
            "Unverified forced-branch engineering probe",
        )
        before = None
        for expected, record in enumerate(doc["steps"], 1):
            require(
                record["step"] == expected and (before is None or before == record["before"]),
                "Noncontiguous forced-branch control",
            )
            changes, checks = parse_events(
                {"topology": record["events"], "n_before": record["before"], "n_after": record["after"]}
            )
            counts = Counter()
            for mutation in changes:
                if "selected" in mutation:
                    counts[mutation["operation"]] += mutation["selected"]
            before = record["after"]
            tables["forced"].append(
                {
                    "run_id": pin["run_id"],
                    "manifest_sha256": pin["manifest_sha256"],
                    "proof_sha256": files["payload/forced-topology.json"]["sha256"],
                    "declared_candidate_recipe_sha256": cfg.get("recipe_sha256"),
                    "step": record["step"],
                    "phase": record["phase"],
                    "n_before": record["before"],
                    "n_after": record["after"],
                    "visible": record["visible"],
                    "max_radius_px": record["max_radius_px"],
                    "loss": record["loss"],
                    **{f"{op}_selected_rows": counts[op] for op in ("clone", "split", "prune", "reset")},
                    "adam_mutations_checked": checks,
                    **{
                        f"{g}_delta_l2": record["delta_l2"][g]
                        for g in ("means", "scales", "quats", "opacities", "colors")
                    },
                }
            )
    if "payload/resource-probe.json" in files:
        proof = load(root / "payload/resource-probe.json")
        require(
            proof["run_id"] == meta["run_id"] and proof["passed"] is True,
            "Wrong engineering qualification",
        )
        tables["qualification"].append(
            {
                "qualification_run_id": pin["run_id"],
                "qualification_manifest_sha256": pin["manifest_sha256"],
                "qualification_member_sha256": files["payload/resource-probe.json"]["sha256"],
                "scope": proof["measured_processor_scope"],
                "worker_tree_sha256": proof["worker_tree_sha256"],
                "worker_zip_sha256": proof["worker_zip_sha256"],
                "contributing_runs": proof["raw_runs"],
                "prior_engineering_failures": proof["earlier_engineering_failures"],
                "analytic_probe": {
                    "errors": proof["geometry"]["actual"]["errors"],
                    "passed": proof["geometry"]["actual"]["passed"],
                },
                "forced_density": {
                    k: proof["density"][k]
                    for k in (
                        "cap",
                        "forced_selected_totals",
                        "adam_moment_controls_passed",
                        "all_active_and_fixed_geometry_color_updated",
                    )
                },
                "short_resume": proof["continuation"],
                "watchdog_probe": proof["resource_cutoff"],
            }
        )
    if (
        meta["experiment_id"] == "recovery-source-extent-readback-v1"
        and meta["execution_status"] == "succeeded"
    ):
        geometry = load(root / "payload/projection.json")
        names = {"source-projection.csv", "checkpoint-scales.csv", "extreme-points.csv"}
        require(
            geometry["schema"] == "el-recovery-source-extent-1"
            and geometry["run_id"] == pin["run_id"]
            and geometry["optimizer_updates_executed"] == 0
            and {r["path"] for r in geometry["files"]} == names,
            "Unexpected source-camera checkpoint diagnostic",
        )
        for item in geometry["files"]:
            member = "payload/" + item["path"]
            require(
                files[member]["sha256"] == item["sha256"] and files[member]["bytes"] == item["bytes"],
                "Checkpoint diagnostic table differs from producer manifest",
            )
            with (root / member).open(newline="") as f:
                for fields in csv.DictReader(f):
                    cell = next(
                        c
                        for c in geometry["inputs"]
                        if (c["object_id"], c["condition"]) == (fields["object_id"], fields["condition"])
                    )
                    tables[item["path"]].append(
                        {
                            "diagnostic_run_id": pin["run_id"],
                            "generation_run_id": cell["generation"]["run_id"],
                            "recipe_sha256": cell["recipe_sha256"],
                            "source_cameras_sha256": cell["members"][cell["short"] + "-camera"]["sha256"],
                            **fields,
                        }
                    )
        tables["extent_proofs"].append(
            {
                **pin,
                "member": "payload/projection.json",
                "member_sha256": files["payload/projection.json"]["sha256"],
                **{
                    k: geometry[k]
                    for k in (
                        "inputs",
                        "projection",
                        "timing",
                        "interpretation",
                        "native_arrays_read",
                        "files",
                    )
                },
            }
        )
    if (
        meta["run_kind"] == "generation"
        and "science" in cfg
        and (row["worker_started"] or payload_kind == "reconstruction.json")
    ):
        detail = fit_measurements(root, meta, cfg, files, asset_store)
        for k in ("steps", "events", "checkpoints"):
            tables[k].extend(detail[k])
        tables["fits"].append(
            {
                **{
                    k: row[k]
                    for k in ("run_id", "experiment", "trial", "object_id", "condition", "status", "role")
                },
                **detail["summary"],
            }
        )
    for schedule_member in ("payload/schedule.json", "payload/schedule-and-state.json"):
        if schedule_member not in files:
            continue
        model = load(root / schedule_member)
        combined = model.get("schema") == "el-recovery-schedule-and-state-audit-1"
        if model.get("schema") == "el-recovery-schedule-audit-1" or combined:
            require(
                model["run_id"] == pin["run_id"] and model["passed"] is True,
                "Incomplete Mac learning-rate readback",
            )
            trace = model["inputs"]["trace"]
            if combined:
                require(
                    model["all_reached_checkpoint_adam_moments_and_cursor_verified"] is True
                    and model["step_zero_five_initial_fp32_parameter_arrays_bit_identical"] is True,
                    "Incomplete full checkpoint readback",
                )
                for key, point in model["checkpoints"].items():
                    issued = point["issued"]
                    tables["state_supplements"].append(
                        {
                            "audit_run_id": pin["run_id"],
                            "audit_manifest_sha256": pin["manifest_sha256"],
                            "audit_member_sha256": files[schedule_member]["sha256"],
                            "generation_run_id": issued["run_id"],
                            "generation_manifest_sha256": issued["manifest_sha256"],
                            "step": int(key),
                            "gaussians": point["gaussians"],
                            "phase": point["phase"],
                            "checkpoint_sha256": issued["sha256"],
                        }
                    )
            tables["schedules"].append(
                {
                    "run_id": pin["run_id"],
                    "manifest_sha256": pin["manifest_sha256"],
                    "member_sha256": files[schedule_member]["sha256"],
                    "audit_type": "learning_rates_and_full_checkpoint_state"
                    if combined
                    else "learning_rates",
                    "generation": trace["run_id"],
                    "generation_manifest_sha256": trace["manifest_sha256"],
                    "original_trace_sha256": trace["sha256"],
                    "recipe_sha256": model["inputs"]["recipe"]["sha256"],
                    **{
                        k: model[k]
                        for k in (
                            "object_id",
                            "condition",
                            "completed_steps",
                            "registered_horizon",
                            "max_relative_error",
                            "tolerance",
                        )
                    },
                    "issued_optimizer_states_checked": len(model["checkpoints"]),
                }
            )
    if "payload/cache-retirement.json" in files:
        proof = load(root / "payload/cache-retirement.json")
        require(
            proof["schema"] == "el-recovery-two-cache-retirement-1"
            and proof["run_id"] == pin["run_id"]
            and proof["removed_count"] == 2
            and set(proof["checked_items"]) == {"train", "development"},
            "Unexpected historical cache retirement",
        )
        tables["cache_retirement"].append(
            {
                "run_id": pin["run_id"],
                "manifest_sha256": pin["manifest_sha256"],
                "proof_sha256": files["payload/cache-retirement.json"]["sha256"],
                **{
                    k: proof[k]
                    for k in (
                        "removed_count",
                        "removed_logical_bytes",
                        "removed_exclusive_apfs_bytes",
                        "plan_member_sha256",
                        "observed_volume_available_change_bytes",
                        "operation_journal_sha256",
                    )
                },
                "available_before_bytes": proof["before"]["available_to_user_bytes"],
                "available_after_bytes": proof["after"]["available_to_user_bytes"],
            }
        )
    if "payload/provider-retirement.json" in files:
        proof = load(root / "payload/provider-retirement.json")
        require(
            proof["schema"] == "el-recovery-first-return-cache-retirement-1"
            and proof["run_id"] == pin["run_id"]
            and proof["removed_count"] == len(proof["rows"]),
            "Unexpected verified return-cache retirement",
        )
        tables["return_retirement"].append(
            {
                "run_id": pin["run_id"],
                "manifest_sha256": pin["manifest_sha256"],
                "proof_sha256": files["payload/provider-retirement.json"]["sha256"],
                **{
                    k: proof[k]
                    for k in (
                        "removed_count",
                        "bytes",
                        "apfs_private_bytes",
                        "observed_available_change_bytes",
                        "provider_response_body_bytes",
                        "provider_proof_sha256",
                        "operation_journal_sha256",
                    )
                },
                "available_before_bytes": proof["before"]["available_to_user_bytes"],
                "available_after_bytes": proof["after"]["available_to_user_bytes"],
            }
        )
    if "payload/return-index.json" in files:
        index = load(root / "payload/return-index.json")
        if index.get("schema") == "el-recovery-complete-scientific-return-index-1":
            tables["independent_return_index"].append(
                {
                    "run_id": pin["run_id"],
                    "manifest_sha256": pin["manifest_sha256"],
                    "member_sha256": files["payload/return-index.json"]["sha256"],
                    **{
                        k: index[k]
                        for k in (
                            "scientific_updates",
                            "native_source_and_target_frames",
                            "target_frames",
                            "source_pixel_visits",
                        )
                    },
                    "originals": [c["original_runs"] for c in index["cells"]],
                }
            )
    return row


def grouped(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["role"], r["platform"])].append(r)

    def biggest(g, field):
        return max((x[field] for x in g if x[field] is not None), default=None)

    return [
        {
            "role": role_name,
            "platform": plat,
            "runs": len(g),
            "succeeded": sum(r["status"] == "succeeded" for r in g),
            "failed": sum(r["status"] == "failed" for r in g),
            "workers_started": sum(r["worker_started"] for r in g),
            "worker_wall_samples": sum(r["worker_wall_seconds"] is not None for r in g),
            "individual_recorder_seconds_sum": sum(r["wall_seconds"] for r in g),
            "worker_wall_seconds_sum": sum(
                r["worker_wall_seconds"] for r in g if r["worker_wall_seconds"] is not None
            )
            if any(r["worker_wall_seconds"] is not None for r in g)
            else None,
            "max_sampled_tree_rss_bytes": biggest(g, "sampled_tree_peak_rss_bytes"),
            "max_device_used_MiB": biggest(g, "device_used_peak_mib"),
            "max_process_gpu_bytes": biggest(g, "process_gpu_peak_bytes"),
            "max_torch_allocated_bytes": biggest(g, "torch_allocated_peak_bytes"),
            "max_torch_reserved_bytes": biggest(g, "torch_reserved_peak_bytes"),
            "sealed_payload_bytes_sum": sum(r["payload_bytes"] for r in g),
            "issued_checkpoint_bytes_sum": sum(r["checkpoint_payload_bytes"] for r in g),
            "native_frame_npz_bytes_sum": sum(r["native_render_npz_bytes"] for r in g),
            "native_frame_png_bytes_sum": sum(r["native_render_png_bytes"] for r in g),
        }
        for (role_name, plat), g in sorted(groups.items())
    ]


def resource_readme(doc, cells, rows):
    def value(number):
        return "N/A" if number is None or number == "" else str(number)

    def gib(n):
        return "N/A" if n is None else f"{n / 2**30:.2f}"

    counts = doc["run_statuses_before_ledger"]
    exchange = doc["archive_exchange"]
    lines = [
        "# Reconstruction recovery resource ledger",
        "",
        f"At {doc['cutoff_utc']}, the inventory contained {len(rows)} sealed producer records. "
        f"{counts.get('succeeded', 0)} succeeded and {counts.get('failed', 0)} failed. "
        f"{doc['quarantined']} records were in quarantine. The [closure](closure.json) and "
        "[complete original list](runs-including-ledger.csv) add the resource report's own final seal, duration and memory.",
        "",
        "## Gaussian optimization",
        "",
        "| Object | Sources | Recipe SHA prefix | Processes | Updates across attempts | Largest step | Initial points | Peak points | Points at largest step |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        if cell["population"] == "engineering":
            continue
        lines.append(
            "| "
            + " | ".join(value(cell[k]) for k in ("object_id", "condition"))
            + " | "
            + cell["recipe_sha256"][:12]
            + " | "
            + " | ".join(
                value(cell[k])
                for k in (
                    "generation_processes",
                    "complete_logged_updates_all_attempts",
                    "max_observed_endpoint",
                    "observed_initial_gaussians",
                    "max_observed_gaussians",
                    "gaussians_at_largest_logged_endpoint",
                )
            )
            + " |"
        )
    if any(c["population"] != "engineering" and c["complete_logged_updates_all_attempts"] for c in cells):
        lines.extend(
            [
                "",
                "![Recorded Gaussian counts, host memory and device memory by object and source count](fit-and-resources.svg)",
            ]
        )
    lines.extend(
        [
            "",
            "[Fit cells](fit-cells.csv) identify each object, source condition, recipe and contributing process. "
            "[Per-process costs](fit-costs.csv) retain continuation origins, loop times and partial logs. "
            "[Per-update measurements](fit-steps.csv.gz) give the selected source, cumulative source counts, loss, phase and population. "
            "[Topology events](topology-events.csv) record each capacity decision and the Adam moment check for each mutation. "
            "[Issued checkpoints](fit-checkpoints.csv), [rendered checkpoints](render-checkpoints.csv) and [Mac checkpoint audits](checkpoint-audits.csv) identify the original bytes and sampler or optimizer readbacks.",
            "",
            "## Timing, memory and storage",
            "",
            "The [stage table](stages.csv) groups individual recorder durations by producer and operation. "
            "Work on two machines can overlap. Per-run worker, optimizer-loop and watchdog clocks retain their original boundaries. "
            "The [telemetry samples](telemetry.csv.gz) contain process-tree RSS, whole-device GPU memory and per-process GPU memory where the driver returned it. "
            "PyTorch peak allocation and reservation cover the worker process. An empty measurement is unavailable or inapplicable.",
            "",
            f"Mac physical free space was {gib(doc['start']['volume_free_bytes'])} GiB at registration and {gib(doc['end']['volume_free_bytes'])} GiB at the inventory cutoff. "
            f"The lowest recorded Mac write preflight was {gib(doc['min_observed_mac_preflight_free_bytes'])} GiB free. "
            f"The active store contained {gib(doc['start']['active_logical_bytes'])} GiB at registration and {gib(doc['end']['active_logical_bytes'])} GiB at cutoff. "
            f"The registered free-space floor was {gib((doc.get('registered_mac_storage_limits') or {}).get('local_free_floor_bytes'))} GiB "
            f"and the active-store target was {gib((doc.get('registered_mac_storage_limits') or {}).get('local_active_batch_target_bytes'))} GiB. "
            "Active logical bytes coalesce hard links within each measurement. Other processes may change available space on the same physical volume.",
            "",
            f"The Mac verified {exchange['unique_accepted_containers']} distinct incoming ZIPs totalling "
            f"{exchange['unique_received_zip_bytes']:,} bytes. [Receiving receipts](transfers.csv) preserve local verification time. "
            "[Joined Windows receipts](windows-package-receipts.csv) bind each original dispatch and archive digest. "
            "[Exchange files](exchange-files.csv) list each observed handoff and return. [Archive receipts](archives.csv) list recorder archives present at cutoff. "
            "The [summary](resources.json) separates provider response bodies, mounted file sizes and unique received ZIPs. Packet-level traffic was unmeasured.",
            "",
            "## Readback",
            "",
            "The [published member manifest](public-files.json) hashes each generated table and figure. With access to the registered study store, the original "
            "resource files can be reproduced from the original inventory using `EL_RUN_ID=RECORDED_LEDGER_ID python3 scripts/recovery_ledger.py "
            "export --inventory INVENTORY --store STORE --out NEW_DIRECTORY` in the recorded software environment. The numerical tables support redrawing the figure with "
            "`python3 scripts/recovery_fit_curves.py --steps fit-steps.csv.gz --resources telemetry.csv.gz --out output.svg`.",
        ]
    )
    if doc.get("source_extent_diagnostics"):
        lines += [
            "",
            "## Source-camera projection readback",
            "",
            "The [three-dimensional scale distributions](checkpoint-scales.csv), [source projection distributions](source-projection.csv) "
            "and [largest ellipses](extreme-points.csv) identify their original checkpoint tensors and registered recipes. "
            "The CPU statistic is the major semiaxis of the local three-standard-deviation ellipse in pixels. "
            "The source table separately gives the nearest gsplat forward-pass maximum and its signed update offset. "
            "An intersection is geometric and each row index refers to one checkpoint. The optimizer did not record individual point contributions. "
            "The [input and measurement proofs](source-extent-proofs.json) bind each zero-update CPU run.",
        ]
    if doc.get("verified_historical_cache_retirements"):
        lines += ["", "## Historical working-cache retirement", ""]
        for action in doc["verified_historical_cache_retirements"]:
            lines.append(
                f"The recorded capacity operation `{action['run_id']}` released {action['removed_count']} "
                f"independently recoverable Stage C transport files, containing {action['removed_logical_bytes']:,} logical bytes "
                f"and {action['removed_exclusive_apfs_bytes']:,} exclusive APFS bytes. The immediately paired volume readings "
                f"were {gib(action['available_before_bytes'])} and {gib(action['available_after_bytes'])} GiB available. "
                "The original producer runs, provider objects and transport receipt were retained."
            )
    return "\n".join(lines) + "\n"


def export(inv, store, shelf, out):
    require(inv["schema"] == "el-recovery-resource-inventory-2", "Unexpected resource inventory")
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Resource export requires an empty output directory")
    tables = defaultdict(list)
    seen, rows = {}, []
    for pin in inv["runs"]:
        rid = pin["run_id"]
        require(rid not in seen and pin["section"] in {"runs", "quarantine"}, "Duplicate or invalid original")
        seen[rid] = pin
        rows.append(read_run(locate(store, shelf, pin), pin, tables, store))
    require(REGISTRATION in seen, "Study registration missing")
    rows.sort(key=lambda r: r["started_utc"])
    write_rows(out / "runs.csv", rows[0], rows)
    write_rows(out / "stages.csv", COSTS, grouped(rows))
    for independent in tables["independent_return_index"]:
        gen_ids, render_ids = set(), set()
        for arm in independent["originals"]:
            for kind, group in (("generation", gen_ids), ("render", render_ids)):
                p = arm[kind]
                require(
                    seen[p["run_id"]]["manifest_sha256"] == p["manifest_sha256"],
                    "Changed indexed scientific original",
                )
                group.add(p["run_id"])
        logged = [r for r in tables["steps"] if r["run_id"] in gen_ids]
        measured = [r for r in tables["renders"] if r["run_id"] in render_ids]
        require(
            len(logged) == independent["scientific_updates"]
            and sum(r["pixels"] for r in logged) == independent["source_pixel_visits"]
            and sum(r["target_frames"] for r in measured) == independent["target_frames"]
            and sum(r["source_frames"] + r["target_frames"] for r in measured)
            == independent["native_source_and_target_frames"],
            "Independent all-cell totals disagree with original optimizer logs and renderer outputs",
        )
    frames = {(r["run_id"], r["step"]): r for r in tables["renders"]}
    checkpoints = {(r["run_id"], r["step"]): r for r in tables["checkpoints"]}
    seen_adam = set()
    for audit in tables["audits"]:
        frame = frames[(audit["render_run_id"], audit["step"])]
        source = checkpoints[(frame["producer_run_id"], audit["step"])]
        require(
            audit["generation_manifest_sha256"] == seen[audit["generation_run_id"]]["manifest_sha256"]
            and audit["render_manifest_sha256"] == seen[audit["render_run_id"]]["manifest_sha256"]
            and audit["gaussians"] == frame["gaussians"] == source["gaussians"]
            and frame["checkpoint_sha256"] == source["checkpoint_sha256"]
            and audit["native_frames_checked"] == frame["source_frames"] + frame["target_frames"],
            "Independent CPU audit differs from the original checkpoint or renderer",
        )
        audit["checkpoint_producer_run_id"] = frame["producer_run_id"]
        audit["checkpoint_sha256"] = source["checkpoint_sha256"]
        seen_adam.add((frame["producer_run_id"], audit["step"]))
    for state in tables["state_supplements"]:
        key = state["generation_run_id"], state["step"]
        orig = checkpoints[key]
        require(
            seen[key[0]]["manifest_sha256"] == state["generation_manifest_sha256"]
            and orig["checkpoint_sha256"] == state["checkpoint_sha256"]
            and orig["gaussians"] == state["gaussians"],
            "Full checkpoint supplement differs from original issued tensors",
        )
        seen_adam.add(key)
    if tables["state_supplements"]:
        write_rows(
            out / "full-state-supplements.csv",
            tables["state_supplements"][0].keys(),
            tables["state_supplements"],
        )
    write_rows(
        out / "checkpoint-audits.csv",
        (
            "audit_run_id",
            "audit_manifest_sha256",
            "audit_member_sha256",
            "generation_run_id",
            "generation_manifest_sha256",
            "render_run_id",
            "render_manifest_sha256",
            "checkpoint_producer_run_id",
            "checkpoint_sha256",
            "step",
            "gaussians",
            "parameter_sha256",
            "native_frames_checked",
            "five_finite_fp32_tensors",
            "five_adam_groups_and_moments_checked",
            "sampler_phase_counts_rng_and_geometry_checked",
        ),
        tables["audits"],
    )
    steps = tables["steps"]
    fit_groups = defaultdict(list)
    by_rid = {r["run_id"]: r for r in rows}
    for fit in tables["fits"]:
        fit_groups[(fit["population"], fit["recipe_sha256"], fit["object_id"], fit["condition"])].append(fit)
    cells = []
    for (population, recipe, obj, cond), group in sorted(fit_groups.items()):
        records = [by_rid[x["run_id"]] for x in group]
        loops = [x["loop_wall_seconds"] for x in group if x["loop_wall_seconds"] is not None]
        endpoint = max(
            (x["last_complete_logged_update"] for x in group if x["complete_logged_updates"]), default=None
        )
        initials = {x["start_N"] for x in group if x["first_prior_update"] == 0 and x["start_N"] is not None}
        ends = {
            x["end_N"]
            for x in group
            if x["last_complete_logged_update"] == endpoint and x["end_N"] is not None
        }
        cells.append(
            {
                "population": population,
                "recipe_sha256": recipe,
                "object_id": obj,
                "condition": cond,
                "generation_processes": len(group),
                "succeeded": sum(x["status"] == "succeeded" for x in group),
                "failed": sum(x["status"] == "failed" for x in group),
                "resumed_processes": sum(x["resume_run_id"] is not None for x in group),
                "resource_cutoffs": sum(by_rid[x["run_id"]]["resource_cutoff"] is True for x in group),
                "complete_logged_updates_all_attempts": sum(x["complete_logged_updates"] for x in group),
                "max_observed_endpoint": endpoint,
                "observed_initial_gaussians": " ".join(map(str, sorted(initials))) or None,
                "gaussians_at_largest_logged_endpoint": " ".join(map(str, sorted(ends))) or None,
                "max_observed_gaussians": max(
                    (x["max_observed_N"] for x in group if x["max_observed_N"] is not None), default=None
                ),
                "issued_checkpoint_records": sum(x["checkpoint_count"] for x in group),
                "recorder_seconds_all_attempts": sum(r["wall_seconds"] for r in records),
                "measured_optimizer_loop_seconds_sum": sum(loops) if loops else None,
                "measured_optimizer_loops": len(loops),
                "max_tree_rss_bytes": max(
                    (
                        r["sampled_tree_peak_rss_bytes"]
                        for r in records
                        if r["sampled_tree_peak_rss_bytes"] is not None
                    ),
                    default=None,
                ),
                "max_device_used_mib": max(
                    (r["device_used_peak_mib"] for r in records if r["device_used_peak_mib"] is not None),
                    default=None,
                ),
                "max_torch_reserved_bytes": max(
                    (
                        r["torch_reserved_peak_bytes"]
                        for r in records
                        if r["torch_reserved_peak_bytes"] is not None
                    ),
                    default=None,
                ),
                "runs": " ".join(sorted(x["run_id"] for x in group)),
            }
        )
    write_rows(
        out / "fit-cells.csv",
        (
            "population",
            "recipe_sha256",
            "object_id",
            "condition",
            "generation_processes",
            "succeeded",
            "failed",
            "resumed_processes",
            "resource_cutoffs",
            "complete_logged_updates_all_attempts",
            "max_observed_endpoint",
            "observed_initial_gaussians",
            "gaussians_at_largest_logged_endpoint",
            "max_observed_gaussians",
            "issued_checkpoint_records",
            "recorder_seconds_all_attempts",
            "measured_optimizer_loop_seconds_sum",
            "measured_optimizer_loops",
            "max_tree_rss_bytes",
            "max_device_used_mib",
            "max_torch_reserved_bytes",
            "runs",
        ),
        cells,
    )
    fallback_fields = {
        "samples": ("run_id", "elapsed_seconds", "tree_rss_mib", "device_used_mib", "process_gpu_bytes"),
        "steps": (
            *CELL,
            "step",
            "phase",
            "source",
            "s00_updates",
            "s01_updates",
            "s02_updates",
            "s03_updates",
            "n_before",
            "gaussians",
            "pixels",
            "visible",
            "loss",
            "display_l1",
            "ssim",
            "max_radius_px",
            "max_abs_visible_means2d_px",
            "topology_events",
            "adam_mutations_checked",
        ),
        "checkpoints": (*CELL, "step", "phase", "gaussians", "checkpoint_sha256", "bytes"),
        "fits": (
            "run_id",
            "experiment",
            "trial",
            "object_id",
            "condition",
            "status",
            "role",
            "updates_this_process",
            "loop_wall_seconds",
        ),
        "renders": (
            "run_id",
            "object_id",
            "condition",
            "step",
            "phase",
            "gaussians",
            "producer_run_id",
            "checkpoint_sha256",
            "source_frames",
            "target_frames",
        ),
    }
    for key, target in (
        ("samples", "telemetry.csv.gz"),
        ("steps", "fit-steps.csv.gz"),
        ("checkpoints", "fit-checkpoints.csv"),
        ("fits", "fit-costs.csv"),
        ("renders", "render-checkpoints.csv"),
    ):
        data = tables[key]
        write_rows(
            out / target, data[0].keys() if data else fallback_fields[key], data, target.endswith(".gz")
        )
    write_rows(out / "topology-events.csv", (*CELL, "step", *EVENT_FIELDS), tables["events"])
    if tables["forced"]:
        write_rows(out / "engineering-density-steps.csv", tables["forced"][0].keys(), tables["forced"])
    if tables["schedules"]:
        write_rows(out / "schedule-audits.csv", tables["schedules"][0].keys(), tables["schedules"])
    if tables["extent_proofs"]:
        save(
            out / "source-extent-proofs.json",
            {"schema": "el-recovery-extent-proofs-1", "diagnostics": tables["extent_proofs"]},
        )
        for name in ("source-projection.csv", "checkpoint-scales.csv", "extreme-points.csv"):
            data = tables[name]
            if data:
                write_rows(out / name, data[0].keys(), data)
        pairs = defaultdict(dict)
        for r in tables["source-projection.csv"]:
            trial = by_rid[r["generation_run_id"]]["trial"]
            recipe = next(
                (k for k in ("geometry-01", "scale-decay-02") if trial.startswith("r2-" + k + "-")), None
            )
            if recipe is None or int(r["step"]) not in (3000, 5000):
                continue
            key = tuple(r[k] for k in ("object_id", "condition", "source_id", "step"))
            require(recipe not in pairs[key], "Repeated source-camera checkpoint diagnosis")
            pairs[key][recipe] = r
        comparison = []
        for (obj, cond, source, step), arms in sorted(pairs.items()):
            if len(arms) != 2:
                continue
            require(
                len({r["source_cameras_sha256"] for r in arms.values()}) == 1, "Different source calibrations"
            )
            x = {"object_id": obj, "condition": cond, "source_id": source, "step": step}
            for recipe, r in sorted(arms.items()):
                for field in (
                    "recipe_sha256",
                    "checkpoint_sha256",
                    "N",
                    "ellipse_intersects_n",
                    "ellipse_intersects_major_gt64",
                    "ellipse_intersects_major_gt256",
                    "ellipse_intersects_major_px_max",
                    "native_gsplat_max_radius_px",
                    "native_state_step_minus_checkpoint",
                ):
                    x[recipe.replace("-", "_") + "_" + field] = r[field]
            comparison.append(x)
        if comparison:
            write_rows(out / "extent-recipe-comparison.csv", comparison[0].keys(), comparison)
    if any(r["population"] != "engineering" for r in steps):
        plot(out / "fit-steps.csv.gz", out / "telemetry.csv.gz", out / "fit-and-resources.svg")

    tf = (
        "receipt_id",
        "sha256",
        "run_id",
        "manifest_sha256",
        "container_sha256",
        "bytes",
        "elapsed_seconds",
        "created_at_utc",
        "status",
    )
    storage_receipts = {}
    for kind, fields in (("transfer", tf), ("archive", (*tf, "provider"))):
        receipts = []
        for pin in inv.get(kind + "_receipts", []):
            path = store / "receipts" / kind / (pin["receipt_id"] + ".json")
            require(sha(path) == pin["sha256"], "Changed storage receipt")
            doc = load(path)
            rid = doc["run_id"]
            require(
                rid in seen
                and doc["kind"] == kind
                and doc.get("manifest_sha256") in (None, seen[rid]["manifest_sha256"]),
                "Storage receipt outside the selected seals",
            )
            receipts.append({k: pin[k] if k == "sha256" else doc.get(k) for k in fields})
        write_rows(out / (kind + "s.csv"), fields, receipts)
        storage_receipts[kind] = receipts
    transfer = storage_receipts["transfer"]
    archives = storage_receipts["archive"]
    write_rows(
        out / "exchange-files.csv",
        ("direction", "batch", "member", "bytes", "sha256", "digest_source", "provider_proof_sha256"),
        inv["exchange_files"],
    )
    write_rows(
        out / "windows-package-receipts.csv",
        (
            "run_id",
            "manifest_sha256",
            "zip_bytes",
            "zip_sha256",
            "request_sha256",
            "worker_receipt_sha256",
            "local_transfer_receipt_id",
            "join_proof_sha256",
        ),
        inv.get("returned_packages", []),
    )
    unique = {}
    for e in transfer:
        if e["container_sha256"] and e["status"] in {"accepted", "identical"}:
            require(
                unique.setdefault(e["container_sha256"], e["bytes"]) == e["bytes"],
                "Conflicting received size",
            )
    exchange_totals = {}
    for e in inv["exchange_files"]:
        values = exchange_totals.setdefault(e["direction"], {"objects": 0, "bytes": 0, "by_sha": {}})
        values["objects"] += 1
        values["bytes"] += e["bytes"]
        require(
            values["by_sha"].setdefault(e["sha256"], e["bytes"]) == e["bytes"],
            "Conflicting exchange content size",
        )
    own = os.environ.get("EL_RUN_ID")
    doc = {
        "schema": "el-recovery-resource-ledger-2",
        "ledger_run_id": own,
        "registration": seen[REGISTRATION],
        "cutoff_utc": inv["cutoff_utc"],
        "run_statuses_before_ledger": dict(Counter(r["status"] for r in rows)),
        "failures_by_public_class": dict(
            Counter(r["failure_class"] for r in rows if r["status"] == "failed")
        ),
        "failed_with_worker_started": sum(r["status"] == "failed" and r["worker_started"] for r in rows),
        "failed_before_worker_start": sum(r["status"] == "failed" and not r["worker_started"] for r in rows),
        "control_utc_seconds_from_registration_baseline_to_cutoff": (
            datetime.fromisoformat(inv["cutoff_utc"]) - datetime.fromisoformat(inv["start"]["utc"])
        ).total_seconds(),
        "total_originals_before_ledger": len(rows),
        "quarantined": sum(r["section"] == "quarantine" for r in rows),
        "min_observed_mac_preflight_free_bytes": min(
            (
                r["preflight_free_bytes"]
                for r in rows
                if r["platform"] == "macOS" and r["preflight_free_bytes"] is not None
            ),
            default=None,
        ),
        "max_observed_mac_preflight_active_bytes": max(
            (
                r["preflight_active_bytes"]
                for r in rows
                if r["platform"] == "macOS" and r["preflight_active_bytes"] is not None
            ),
            default=None,
        ),
        "start": inv["start"],
        "registered_mac_storage_limits": inv.get("registration_storage_limits"),
        "end": inv["end"],
        "active_logical_change_bytes": inv["end"]["active_logical_bytes"]
        - inv["start"]["active_logical_bytes"],
        "physical_volume_free_change_bytes": inv["end"]["volume_free_bytes"]
        - inv["start"]["volume_free_bytes"],
        "archive_exchange": {
            "receive_attempts": len(transfer),
            "unique_accepted_containers": len(unique),
            "unique_received_zip_bytes": sum(unique.values()),
            "local_receipt_wall_seconds_sum": sum(
                t["elapsed_seconds"] for t in transfer if t["elapsed_seconds"] is not None
            ),
            "mounted_logical_bytes_by_direction": {k: v["bytes"] for k, v in sorted(exchange_totals.items())},
            "mounted_distinct_content_bytes_by_direction": {
                k: sum(v["by_sha"].values()) for k, v in sorted(exchange_totals.items())
            },
            "mounted_objects_by_direction": {k: v["objects"] for k, v in sorted(exchange_totals.items())},
            "mounted_inspection_utc": inv.get("exchange_inspection_utc"),
            "scope": inv["exchange_scope"],
            "archive_receipts_by_status": dict(Counter(r["status"] for r in archives)),
            "online_historical_cache_response_bodies": inv.get("provider_reads", []),
            "online_scientific_handoff_response_bodies": inv.get("source_provider_reads", []),
            "online_scientific_return_response_bodies": inv.get("return_provider_reads", []),
            "wire_bytes": None,
        },
        "platform_observation": inv.get("platform_observation"),
        "recorded_platforms": sorted({r["platform"] for r in rows}),
        "engineering_qualification": tables["qualification"],
        "independently_verified_scientific_return_totals": tables["independent_return_index"],
        "source_extent_diagnostics": len(tables["extent_proofs"]),
        "source_camera_checkpoint_projection_rows": len(tables["source-projection.csv"]),
        "source_extent_checkpoint_states": len(tables["checkpoint-scales.csv"]),
        "independent_learning_rate_audits": len(tables["schedules"]),
        "independent_learning_rate_trace_updates": sum(x["completed_steps"] for x in tables["schedules"]),
        "unique_learning_rate_optimized_processes": len({x["generation"] for x in tables["schedules"]}),
        "unique_learning_rate_trace_updates": sum(
            {x["generation"]: x["completed_steps"] for x in tables["schedules"]}.values()
        ),
        "independent_learning_rate_checkpoint_states": sum(
            x["issued_optimizer_states_checked"] for x in tables["schedules"]
        ),
        "unique_independently_checked_checkpoint_adam_states": len(seen_adam),
        "independent_render_joined_checkpoint_adam_audits": len(tables["audits"]),
        "supplementary_full_checkpoint_state_audits": len(tables["state_supplements"]),
        "independent_native_checkpoint_frames": sum(x["native_frames_checked"] for x in tables["audits"]),
        "forced_engineering_probe_runs": len({x["run_id"] for x in tables["forced"]}),
        "verified_historical_cache_retirements": tables["cache_retirement"],
        "verified_return_download_cache_retirements": tables["return_retirement"],
        "new_risk_model_fits": 0,
        "new_threshold_selections": 0,
        "complete_logged_optimizer_updates_by_role": {
            k: sum(f["complete_logged_updates"] for f in tables["fits"] if f["role"] == k)
            for k in sorted({f["role"] for f in tables["fits"]})
        },
        "scientific_object_condition_recipe_combinations_started": sum(
            c["population"] != "engineering" for c in cells
        ),
        "engineering_optimizer_processes": sum(
            c["generation_processes"] for c in cells if c["population"] == "engineering"
        ),
        "forced_engineering_probe_updates": len(tables["forced"]),
        "notes": [
            "Recorder intervals are producer-local elapsed times. Summed durations may overlap across processes and hosts.",
            "CSV blanks mean unmeasured or inapplicable. Torch allocator, process GPU, whole-device GPU and host RSS measure different scopes.",
            "Active logical bytes coalesce hard links. File-manifest totals, mounted exchange objects, local receiving time and physical volume free space have different boundaries.",
            "Mounted logical totals count each named file. Distinct-content totals coalesce identical SHA-256 values within a direction. Alternate delivery of original job bytes adds no scientific execution. Network wire bytes across transports were unmeasured.",
            "For an interrupted process, complete_logged_updates counts fully recorded steps. The process total is unavailable absent a signed terminal optimizer summary.",
            "The allocator peaks are reported for entire processes. The optimizer recorded no per-update clock or per-update allocator trace.",
            "The recorder's own manifest and final cost are attached after sealing in the published closure.",
        ],
    }
    save(out / "resources.json", doc)
    (out / "README.md").write_text(resource_readme(doc, cells, rows), encoding="utf-8")
    inventory_files = [
        {"path": f.name, "bytes": f.stat().st_size, "sha256": sha(f)}
        for f in sorted(out.iterdir())
        if f.is_file()
    ]
    save(
        out / "public-files.json",
        {"schema": "el-recovery-resource-files-1", "run_id": own, "files": inventory_files},
    )
    return doc


def record(a):
    require(
        importlib.util.find_spec("matplotlib") is not None,
        "Use the analysis environment with matplotlib installed",
    )
    store = a.store.resolve()
    shelf = a.shelf.resolve() if a.shelf else store.parent / "cold-shelf"
    inv = inventory(store, shelf, a.exchange)
    rid, digest = a.final_report.split("=", 1)
    final = next((r for r in inv["runs"] if r["run_id"] == rid and r["manifest_sha256"] == digest), None)
    require(
        final is not None and load(locate(store, shelf, final) / "run.json")["run_kind"] == "report",
        "No final scientific report",
    )
    prior = [
        p
        for p in inv["runs"]
        if load(locate(store, shelf, p) / "run.json")["experiment_id"] == EXPERIMENT
        and load(locate(store, shelf, p) / "run.json")["execution_status"] == "succeeded"
    ]
    require(not prior, "A sealed final resource report already exists")
    from tempfile import TemporaryDirectory

    src = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(dir=store / "work", prefix="r-resource-config-") as td:
        work = Path(td)
        source = work / "source"
        (source / "scripts").mkdir(parents=True)
        for name in ("recovery_ledger.py", "recovery_fit_curves.py"):
            shutil.copyfile(src / "scripts" / name, source / "scripts" / name)
        for filename in ("__init__.py", "_bundle.py", "_capacity.py", "_verify.py", "_metrics.py"):
            target = source / "src/evidence_lens" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / "src/evidence_lens" / filename, target)
        cfg = {
            "experiment_id": EXPERIMENT,
            "trial_id": "r-resources-final-1",
            "schema": "el-recovery-ledger-job-1",
            "final_scientific_report": final,
            "inventory_sha256": None,
            "recording": {"expected_output_bytes": 20 * 2**20},
        }
        manifest = work / "inventory.json"
        save(manifest, inv)
        cfg["inventory_sha256"] = sha(manifest)
        save(work / "config.json", cfg)
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(source),
            "--storage-policy",
            str(src / "configs/storage.json"),
            "--kind",
            "report",
            "--config",
            str(work / "config.json"),
            "--parent",
            rid + "=" + digest,
            "--input",
            "inventory=" + str(manifest),
            "--",
            "{python}",
            "-B",
            "{source}/scripts/recovery_ledger.py",
            "export",
            "--inventory",
            "{input:inventory}",
            "--store",
            str(store),
            "--shelf",
            str(shelf),
            "--out",
            "{out}",
        ]
        return subprocess.run(cmd, env={**os.environ, "MPLCONFIGDIR": str(work / "matplotlib")}).returncode


def close(store, shelf, pin, out):
    from make_handoff import publish

    root = locate(store, shelf, pin)
    result = checked(root, pin)
    require(
        result["run"]["experiment_id"] == EXPERIMENT and result["run"]["execution_status"] == "succeeded",
        "Wrong final ledger",
    )
    q = load(root / "payload/public-files.json")
    require(
        q["run_id"] == pin["run_id"] and all("/" not in e["path"] for e in q["files"]),
        "Wrong ledger publication",
    )
    namespace = {p["path"]: p for p in result["manifest"]["files"]}
    out.mkdir(parents=True, exist_ok=True)
    for e in [
        *q["files"],
        {
            "path": "public-files.json",
            **{k: namespace["payload/public-files.json"][k] for k in ("bytes", "sha256")},
        },
    ]:
        member = "payload/" + e["path"]
        require(
            all(namespace[member][k] == e[k] for k in ("bytes", "sha256")), "Changed original ledger member"
        )
        publish(out / e["path"], (root / member).read_bytes())
    with (root / "payload/runs.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        before = list(reader)
    self_row = read_run(root, pin, defaultdict(list))
    require(self_row["run_id"] not in {r["run_id"] for r in before}, "Repeated ledger origin")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows([*before, self_row])
    publish(out / "runs-including-ledger.csv", buf.getvalue().encode())
    doc = {
        "schema": "el-recovery-resource-closure-1",
        "ledger": pin,
        "sources": {
            n: namespace[n]["sha256"]
            for n in (
                "run.json",
                "events.jsonl",
                "telemetry.csv",
                "environment.json",
                "payload/resources.json",
                "payload/public-files.json",
            )
        },
        "recorded_input_runs": len(before),
        "all_originals_including_ledger": len(before) + 1,
        "all_statuses": dict(Counter(r["status"] for r in [*before, self_row])),
        "all_runs_csv_sha256": sha(out / "runs-including-ledger.csv"),
        "ledger_wall_seconds": self_row["wall_seconds"],
        "ledger_peak_sampled_tree_rss_bytes": self_row["sampled_tree_peak_rss_bytes"],
        "ledger_free_bytes_at_write_preflight": self_row["preflight_free_bytes"],
        "ledger_active_bytes_at_write_preflight": self_row["preflight_active_bytes"],
    }
    publish(out / "closure.json", encoded(doc))
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("inventory", "export", "record", "close"):
        q = sub.add_parser(action)
        q.add_argument("--store", type=Path, required=True)
        q.add_argument("--shelf", type=Path)
        if action in ("inventory", "record"):
            q.add_argument("--exchange", type=Path)
        if action in ("inventory", "export", "close"):
            q.add_argument("--out", type=Path, required=True)
        if action == "export":
            q.add_argument("--inventory", type=Path, required=True)
        if action == "record":
            q.add_argument("--final-report", required=True, metavar="RUN=SHA")
        if action == "close":
            q.add_argument("--ledger", required=True, metavar="RUN=SHA")
    a = p.parse_args()
    store = a.store.resolve()
    shelf = a.shelf.resolve() if a.shelf else store.parent / "cold-shelf"
    if a.action == "inventory":
        value = inventory(store, shelf, a.exchange)
        save(a.out, value)
        print(
            json.dumps(
                {k: value[k] for k in ("cutoff_utc", "end")}
                | {"runs": len(value["runs"]), "transfer_receipts": len(value["transfer_receipts"])}
            )
        )
    elif a.action == "export":
        result = export(load(a.inventory), store, shelf, a.out)
        print(json.dumps(result["run_statuses_before_ledger"]))
    elif a.action == "record":
        return record(a)
    else:
        rid, digest = a.ledger.split("=", 1)
        print(
            json.dumps(
                close(store, shelf, {"section": "runs", "run_id": rid, "manifest_sha256": digest}, a.out)
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
