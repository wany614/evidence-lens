"""Summarize an explicit sealed D0/D1 run inventory without opening photographic or label arrays."""

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidence_lens._bundle import RUN_ID, TERMINAL, load, sha
from evidence_lens._verify import require


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def write_csv(path, records, fields=None):
    with path.open("x", newline="", encoding="utf8") as dest:
        w = csv.DictWriter(dest, fieldnames=fields or list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def peak(data):
    vals = [x for x in data if x is not None]
    return max(vals) if vals else None


def fetch(root, byname, name):
    row = byname[name]
    path = root / name
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == row["bytes"]
        and sha(path) == row["sha256"],
        "Changed selected recorded member: " + name,
    )
    return path


def one(store, shelf, pin):
    rid = pin["run_id"]
    require(RUN_ID.fullmatch(rid), "Invalid recorder identity")
    disposition = pin["disposition"]
    require(disposition in {"sealed", "quarantined"}, "Unregistered recorder disposition")
    section = "quarantine" if disposition == "quarantined" else "runs"
    root = store / section / rid
    if not root.is_dir():
        root = shelf / section / rid
    seal = load(root / "completion.json")
    require(
        sha(root / "manifest.json") == pin["manifest_sha256"] == seal["manifest_sha256"]
        and seal["run_id"] == rid,
        "Incorrect selected manifest",
    )
    man = load(root / "manifest.json")
    spec = {r["path"]: r for r in man["files"]}
    require(len(spec) == len(man["files"]) and man["run_id"] == rid, "Repeated or unrelated stage manifest")

    def read(name):
        return load(fetch(root, spec, name))

    meta = read("run.json")
    fetch(root, spec, "config.resolved.yaml")
    environment = read("environment.json")
    require(
        meta["run_id"] == rid
        and meta["execution_status"] == seal["execution_status"]
        and meta["execution_status"] in TERMINAL
        and meta["config_sha256"] == spec["config.resolved.yaml"]["sha256"]
        and meta["experiment_id"].startswith(("d0-", "d1-")),
        "Run outside registered D0/D1 scope",
    )
    attempt = meta["execution_status"]
    kind = meta["run_kind"]
    trial = meta["trial_id"]
    experiment = meta["experiment_id"]
    if "d0-linux" in experiment:
        stage = "Linux public CPU acceptance"
    elif "learning-curves" in experiment:
        stage = "existing train/development curves"
    elif experiment == "d1-dtu-external":
        stage = "official catalogue, ZIP index and selected HTTP members"
    elif experiment == "d1-public-input-replay":
        stage = "public-entry metadata and camera replay"
    elif kind == "generation":
        stage = "Windows Gaussian optimization"
    elif kind == "render":
        stage = "Windows prediction and camera probes"
    elif kind == "model_score":
        stage = "frozen public risk scores"
    elif experiment == "d1-dtu-crop256":
        stage = "crop and convert permitted input"
    elif experiment == "d1-dtu-case-readback":
        stage = "verify reconstruction and assemble case"
    elif experiment == "d1-dtu-development-budget-diagnostic":
        stage = "development 256/1024 RGB diagnostic"
    elif experiment == "d1-development-budget-appendix":
        stage = "development budget appendix"
    elif experiment == "d1-dtu-dev-budget":
        stage = "register development-only budget"
    elif experiment == "d1-dtu-authority":
        stage = "external method freeze or data opening"
    elif kind == "evaluation":
        stage = "independent RGB evaluation"
    elif kind == "report":
        stage = "study or resource report"
    elif "replay" in experiment:
        stage = "independent numerical replay"
    elif "regression" in experiment or "published-assets" in experiment:
        stage = "published-case regression or asset restore"
    elif "freeze" in experiment or "opening" in experiment or "gate" in experiment:
        stage = "protocol freeze or opening"
    else:
        stage = "camera, dataset or interface verification"
    platform = environment["platform"].split("-")[0]
    if experiment == "d0-linux-public-acceptance" and attempt == "succeeded":
        platform = "Linux on macOS"
    require(
        kind not in ("model_fit", "threshold_select"),
        "An unregistered risk-model fit or threshold-selection operation needs explicit accounting",
    )
    files = [x for x in spec.values() if x["path"].startswith("payload/")]
    # These are logical sealed-file sizes from the complete immutable manifest. A size is not unique disk use.
    row = {
        "run_id": rid,
        "manifest_sha256": pin["manifest_sha256"],
        "work_package": experiment[:2].upper(),
        "stage": stage,
        "experiment": experiment,
        "trial": trial,
        "condition": meta.get("condition_id"),
        "platform": platform,
        "kind": kind,
        "scope": (
            "diagnostic_1024"
            if "1024" in trial
            else "primary_256"
            if kind in ("generation", "render")
            else "published_057"
            if "057" in trial
            else None
        ),
        "status": attempt,
        "producer_disposition": disposition,
        "exit_code": meta["exit_code"],
        "retry_of": meta["retry_of_run_id"],
        "worker_started": None,
        "failure_class": None,
        "opened_utc": meta["started_at_utc"],
        "closed_utc": meta["ended_at_utc"],
        "recorder_elapsed_s": meta.get("elapsed_seconds_monotonic"),
        "worker_stage_s": None,
        "optimization_s": None,
        "model_load_s": None,
        "sampled_process_tree_rss_peak_B": None,
        "telemetry_samples": 0,
        "sampled_process_gpu_peak_B": None,
        "sampled_device_wide_vram_peak_MiB": None,
        "torch_allocated_peak_B": None,
        "torch_reserved_peak_B": None,
        "linux_container_peak_B": None,
        "preflight_store_active_B": None,
        "preflight_volume_free_B": None,
        "preflight_projected_additional_B": None,
        "linux_net_rx_B": None,
        "linux_net_tx_B": None,
        "official_DTU_HTTP_body_B": None,
        "sealed_manifest_members": len(spec),
        "sealed_manifest_logical_B": sum(x["bytes"] for x in spec.values()),
        "payload_members": len(files),
        "payload_logical_B": sum(x["bytes"] for x in files),
        "gaussians": None,
        "optimization_steps": None,
        "target_renders": None,
        "source_self_renders": None,
        "risk_model_optimizer_steps": 0,
    }
    if "events.jsonl" in spec:
        with fetch(root, spec, "events.jsonl").open() as f:
            events = [json.loads(s) for s in f if s.strip()]
        row["worker_started"] = any(e.get("event") == "process_started" for e in events)
        capacity = [e for e in events if e.get("event") == "storage_preflight"]
        if capacity:
            require(len(capacity) == 1, "Ambiguous recorder capacity preflight")
            gate = capacity[0]
            row.update(
                preflight_store_active_B=gate["active_bytes"],
                preflight_volume_free_B=gate["free_bytes"],
                preflight_projected_additional_B=gate["additional_estimated_bytes"],
            )
    if attempt != "succeeded":
        reason = (meta.get("failure_reason") or "").lower()
        trace = fetch(root, spec, "stderr.log").read_text(errors="replace")
        # Publish a short symptom, without host paths, credentials or arbitrary worker exception text.
        if experiment == "d0-linux-public-acceptance" and "no such file" in reason:
            row["failure_class"] = "host_worker_launch"
        elif "no such file" in reason and "_capacity.check(" in trace and "bytes_on_tree" in trace:
            row["failure_class"] = "capacity_scan_transient_file"
        elif "undeclared parent" in reason or (
            "parent" in reason and ("failed" in reason or "unsuccessful" in reason)
        ):
            row["failure_class"] = "unsuitable_input_parent"
        elif experiment == "d1-joint-report" and "KeyError: 'schema'" in trace:
            row["failure_class"] = "report_input_schema_lookup"
        elif disposition == "quarantined" and "rejectedlineage" in reason:
            row["failure_class"] = "retry_changed_frozen_inputs"
        else:
            row["failure_class"] = "worker_exit" if row["worker_started"] else "recorder_setup"
        if row["worker_started"] is False and experiment in {"d1-dtu-external", "d1-catalogue-identity"}:
            row["official_DTU_HTTP_body_B"] = 0
    if "telemetry.csv" in spec:
        with fetch(root, spec, "telemetry.csv").open(newline="") as f:
            tel = list(csv.DictReader(f))
        row["telemetry_samples"] = len(tel)
        for field, key in [
            ("tree_rss_bytes", "sampled_process_tree_rss_peak_B"),
            ("gpu_process_memory_bytes", "sampled_process_gpu_peak_B"),
        ]:
            row[key] = peak(int(x[field]) for x in tel if x.get(field) not in (None, ""))
        row["sampled_device_wide_vram_peak_MiB"] = peak(
            float(g["memory.used"])
            for x in tel
            if x.get("gpu_devices")
            for g in json.loads(x["gpu_devices"])
            if isinstance(g.get("memory.used"), (float, int))
        )
    if "payload/linux-resources.json" in spec:
        linux = read("payload/linux-resources.json")
        net = linux["network_bytes"]
        row.update(
            worker_stage_s=linux["elapsed_seconds"],
            linux_container_peak_B=linux["container_peak_memory_bytes"],
            linux_net_rx_B=net["rx_bytes"],
            linux_net_tx_B=net["tx_bytes"],
        )
    if "payload/network.json" in spec:
        n = read("payload/network.json")
        row["official_DTU_HTTP_body_B"] = n["body_bytes"]
        require(0 <= n["body_bytes"] <= n["cap_bytes"], "Invalid recorded HTTP byte count")
    if kind in ("render", "generation"):
        name = "payload/" + ("reconstruction.json" if kind == "generation" else "prediction.json")
        if name in spec and attempt == "succeeded":
            j = read(name)
            t = j["runtime"]
            row.update(
                worker_stage_s=j["stage_seconds"],
                torch_allocated_peak_B=t["torch_cuda_allocated_peak_bytes"],
                torch_reserved_peak_B=t["torch_cuda_reserved_peak_bytes"],
            )
            if kind == "generation":
                row.update(
                    gaussians=j["final_gaussian_count"],
                    optimization_steps=j["gaussian_optimization_steps"],
                    optimization_s=j["optimization_seconds"],
                )
            else:
                row.update(
                    target_renders=len(j["target_ids"]), source_self_renders=len(j["generator_source_ids"])
                )
    if kind == "model_score" and "payload/scores.json" in spec:
        j = read("payload/scores.json")
        row.update(model_load_s=j["loaded_model_seconds"], worker_stage_s=j["worker_wall_seconds"])
    return row


def summarize(out, rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["work_package"], r["stage"], r["scope"], r["platform"], r["status"]].append(r)
    stats = []
    for (phase, stage, scope, platform, status), items in sorted(
        groups.items(), key=lambda x: tuple(v or "" for v in x[0])
    ):
        result = {
            "work_package": phase,
            "stage": stage,
            "scope": scope,
            "platform": platform,
            "status": status,
            "runs": len(items),
            "sum_of_recorded_stage_wall_s": sum(
                x["recorder_elapsed_s"] for x in items if x["recorder_elapsed_s"] is not None
            ),
            "sealed_payload_logical_B": sum(x["payload_logical_B"] for x in items),
        }
        for field in ("worker_stage_s", "optimization_s", "model_load_s"):
            times = [r[field] for r in items if r.get(field) is not None]
            result[field + "_observations"] = len(times)
            result[field + "_sum"] = sum(times) if times else None
        for field in (
            "official_DTU_HTTP_body_B",
            "linux_net_rx_B",
            "linux_net_tx_B",
            "sealed_manifest_logical_B",
        ):
            sizes = [r[field] for r in items if r.get(field) is not None]
            result[field + "_sum"] = sum(sizes) if sizes else None
        for key in (
            "sampled_process_tree_rss_peak_B",
            "sampled_process_gpu_peak_B",
            "sampled_device_wide_vram_peak_MiB",
            "torch_allocated_peak_B",
            "torch_reserved_peak_B",
            "linux_container_peak_B",
        ):
            result[key] = peak(x[key] for x in items)
        stats.append(result)
    write_csv(out / "stages.csv", stats)


def archive_readback(transfers, batch, name, remote):
    rid = name[:-4]
    require(RUN_ID.fullmatch(rid), "Invalid producer run name")
    entry = batch["archive_receipts"][rid]
    path = transfers / batch["batch"] / entry["condition"] / "transfers" / (rid + ".json")
    require(
        entry["condition"] in ("M2", "M4") and sha(path) == entry["sha256"], "Changed Mac receiving receipt"
    )
    receipt = load(path)
    local = receipt["archive_receive"]
    require(
        receipt["run_id"] == local["run_id"] == rid
        and receipt["manifest_sha256"] == local["manifest_sha256"]
        and receipt["container_sha256"] == local["container_sha256"] == remote["sha256"]
        and receipt["container_bytes"] == local["bytes"] == remote["bytes"]
        and local["status"] in ("accepted", "identical"),
        "Mac receiving receipt and returned bytes disagree",
    )
    return {
        "mac_archive_status": local["status"],
        "mac_archive_elapsed_s": local["elapsed_seconds"],
        "mac_linked_input_B": local.get("local_input_storage", {}).get("linked_input_bytes"),
        "mac_receipt_sha256": entry["sha256"],
    }


def main(a):
    index = load(a.inventory)
    require(
        index["schema"] == "el-dtu-resource-roster-1"
        and bool(index["runs"])
        and len(index["runs"]) == len({x["run_id"] for x in index["runs"]}),
        "Wrong or duplicate resource roster",
    )
    out = a.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    require(not any(out.iterdir()), "Use a fresh resource output directory")
    rows = [one(a.store.resolve(), a.shelf.resolve(), p) for p in index["runs"]]
    write_csv(out / "runs.csv", rows)
    summarize(out, rows)
    transfer = []
    for b in index["batches"]:
        meta = load(a.transfers / b["batch"] / "transport-manifest.json")
        require(
            sha(a.transfers / b["batch"] / "transport-manifest.json") == b["transport_manifest_sha256"]
            and meta["batch"] == b["batch"]
            and meta["dispatch_container_sha256"] == b["dispatch_container_sha256"],
            "Transport certificate differs",
        )
        require(
            set(b["archive_receipts"]) == {n[:-4] for n in meta["files"] if n.endswith(".zip")},
            "Missing Mac receiving receipt",
        )
        transfer.append(
            {
                "batch": b["batch"],
                "direction": "Mac to Windows",
                "item": "source-only job container",
                "items": 1,
                "logical_bytes": b["dispatch_container_bytes"],
                "sha256": b["dispatch_container_sha256"],
                "wire_bytes": None,
            }
        )
        for n, ref in sorted(meta["files"].items()):
            transfer.append(
                {
                    "batch": b["batch"],
                    "direction": "Windows to Mac",
                    "item": "sealed run ZIP"
                    if n.endswith(".zip")
                    else "worker summary"
                    if "-worker-summary" in n
                    else "worker receipt",
                    "items": 1,
                    "logical_bytes": ref["bytes"],
                    "sha256": ref["sha256"],
                    "wire_bytes": None,
                    **(archive_readback(a.transfers, b, n, ref) if n.endswith(".zip") else {}),
                }
            )
    write_csv(
        out / "transfers.csv",
        transfer,
        [
            "batch",
            "direction",
            "item",
            "items",
            "logical_bytes",
            "sha256",
            "wire_bytes",
            "mac_archive_status",
            "mac_archive_elapsed_s",
            "mac_linked_input_B",
            "mac_receipt_sha256",
        ],
    )
    scientific = [r for r in rows if r["kind"] == "generation" and r["status"] == "succeeded"]
    rendered = [r for r in rows if r["kind"] == "render" and r["status"] == "succeeded"]
    counts = Counter((r["work_package"], r["status"]) for r in rows)
    arms = {}
    for row in scientific + rendered:
        arm = (
            "set1_diagnostic_1024"
            if "1024" in row["trial"]
            else ("set6_external_256" if "set6" in row["trial"] else "set1_debug_256")
        )
        arm_totals = arms.setdefault(
            arm,
            {
                "successful_reconstructions": 0,
                "gaussian_optimizer_updates": 0,
                "targets": 0,
                "self_renders": 0,
            },
        )
        if row["kind"] == "generation":
            arm_totals["successful_reconstructions"] += 1
            arm_totals["gaussian_optimizer_updates"] += row["optimization_steps"]
        else:
            arm_totals["targets"] += row["target_renders"]
            arm_totals["self_renders"] += row["source_self_renders"]
    storage = {}
    for host in ("Mac", "Windows"):
        observations = [
            r
            for r in rows
            if r["preflight_store_active_B"] is not None
            and ("Windows" if r["platform"] == "Windows" else "Mac") == host
        ]
        if observations:
            storage[host] = {
                "preflights": len(observations),
                "min_volume_free_B": min(r["preflight_volume_free_B"] for r in observations),
                "max_observed_active_store_B": max(r["preflight_store_active_B"] for r in observations),
            }
    summary = {
        "schema": "el-dtu-resource-summary-1",
        "acquisition_by_arm": arms,
        "storage_at_cutoff": index.get("storage_at_cutoff"),
        "storage_preflights_by_host": storage,
        "inventory_sha256": sha(a.inventory),
        "cutoff_utc": index["cutoff_utc"],
        "run_counts": {
            phase: {status: counts[phase, status] for status in sorted(TERMINAL)}
            for phase in sorted({r["work_package"] for r in rows})
        },
        "producer_dispositions": dict(Counter(r["producer_disposition"] for r in rows)),
        "recorded_pipeline_wall_time": "Each run is timed on its producer's monotonic clock. Stage sums measure recorded work. Some producers ran concurrently on separate host clocks.",
        "model_training_runs": sum(r["kind"] == "model_fit" for r in rows),
        "fitted_risk_models": 0,
        "risk_threshold_recalibrations": 0,
        "successful_reconstructions": len(scientific),
        "true_gaussian_optimizer_updates": sum(r["optimization_steps"] for r in scientific),
        "target_prediction_views": sum(r["target_renders"] for r in rendered),
        "source_self_render_views": sum(r["source_self_renders"] for r in rendered),
        "mac_archive_receiving": {
            "verified_originals": sum("mac_archive_status" in r for r in transfer),
            "sum_of_local_archive_receipt_times_s": sum(r.get("mac_archive_elapsed_s", 0) for r in transfer),
            "scope": "Archive.receive on the Mac, including storage preflight, local copying and recursive verification of the producer's parents.",
        },
        "training_label": "Gaussian parameter optimization fits the scene. The released four risk scorers are frozen.",
        "network": {
            "linux_container_rx_B": sum(r["linux_net_rx_B"] or 0 for r in rows),
            "linux_container_tx_B": sum(r["linux_net_tx_B"] or 0 for r in rows),
            "official_DTU_HTTP_body_B": sum(r["official_DTU_HTTP_body_B"] or 0 for r in rows),
            "dispatch_logical_B": sum(
                t["logical_bytes"] for t in transfer if t["direction"] == "Mac to Windows"
            ),
            "returns_logical_B": sum(
                t["logical_bytes"] for t in transfer if t["direction"] == "Windows to Mac"
            ),
            "wire_bytes_Google_Drive": None,
            "scope": "Linux interface counters include release assets and dependencies. DTU counts are HTTP response body bytes. For the mounted exchange only logical container and receipt bytes were observed. These counters describe different boundaries.",
        },
        "missing": "Blank CSV cells indicate unmeasured or inapplicable counters. A process RSS sample excludes the OS and GPU. Device-wide GPU use includes other processes. CUDA allocator allocated/reserved peaks are separate.",
        "archive_scope": "Per-run manifest bytes sum the hash-bound members. The manifest and completion wrapper sit outside that inventory. Payload is a subset. Shared bytes can appear in several runs. Storage measurements were collected at discrete producer preflights. On Windows they cover the selected batch store.",
        "cutoff_scope": index["cutoff_scope"],
        "additional_events": index.get("additional_events", []),
        "public_files": {
            f.name: {"sha256": sha(f), "bytes": f.stat().st_size} for f in sorted(out.glob("*.csv"))
        },
    }
    write_json(out / "resource-summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("inventory", "store", "shelf", "transfers", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    main(p.parse_args())
