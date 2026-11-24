"""Summarise the explicitly closed T0 recorder runs without exporting images or private environment data."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from evidence_lens._capacity import bytes_on_tree
from t0_evaluate import table


def number(x):
    return int(float(x)) if x not in (None, "") else None


def row(store, pin):
    directory = store / "runs" / pin["run_id"]
    if not directory.is_dir():
        directory = store / "quarantine" / pin["run_id"]
    manifest = load(directory / "manifest.json")
    run = load(directory / "run.json")
    seal = load(directory / "completion.json")
    require(
        pin["manifest_sha256"] == sha(directory / "manifest.json") == seal["manifest_sha256"]
        and seal["execution_status"] == run["execution_status"]
        and run["run_id"] == pin["run_id"],
        "inconsistent ledger input",
    )
    files = {f["path"]: f for f in manifest["files"]}

    def checked(name):
        p = directory / name
        require(sha(p) == files[name]["sha256"], "changed source for ledger: " + name)
        return p

    with checked("telemetry.csv").open(newline="") as f:
        measured = list(csv.DictReader(f))
    rss = [number(a["tree_rss_bytes"]) for a in measured if a["tree_rss_bytes"]]
    gpu = [
        d["memory.used"]
        for a in measured
        if a["gpu_devices"]
        for d in json.loads(a["gpu_devices"])
        if isinstance(d.get("memory.used"), (int, float))
    ]
    available = [number(a["system_available_bytes"]) for a in measured if a["system_available_bytes"]]
    platform = load(checked("environment.json"))["platform"].split("-")[0]
    rec = {
        "run_id": run["run_id"],
        "manifest_sha256": pin["manifest_sha256"],
        "trial": run["trial_id"],
        "kind": run["run_kind"],
        "status": run["execution_status"],
        "recording_host_OS": platform,
        "start_utc": run["started_at_utc"],
        "end_utc": run["ended_at_utc"],
        "elapsed_s": run["elapsed_seconds_monotonic"],
        "payload_bytes": sum(f["bytes"] for n, f in files.items() if n.startswith("payload/")),
        "manifest_member_bytes": sum(f["bytes"] for f in files.values()),
        "sampled_tree_RSS_peak_bytes": max(rss, default=None),
        "sampled_system_available_min_bytes": min(available, default=None),
        "sampled_whole_GPU_used_peak_MiB": max(gpu, default=None),
    }
    fields = {
        "loop_wall_seconds": "fit_loop_s",
        "worker_wall_seconds": "algorithm_worker_s",
        "torch_allocated_peak_bytes": "torch_allocated_peak_bytes",
        "torch_reserved_peak_bytes": "torch_reserved_peak_bytes",
    }
    for name in ("payload/reconstruction.json", "payload/outputs.json", "payload/scores.json"):
        if name in files:
            d = load(checked(name))
            flat = {**d, **d.get("resources", {})}
            rec.update({dst: flat[k] for k, dst in fields.items() if k in flat})
    if "payload/process-wall.json" in files:
        doc = load(checked("payload/process-wall.json"))
        rec["numerical_process_s"] = next(
            (
                v
                for k, v in doc.items()
                if k
                in ("process_seconds", "process_wall_seconds", "wall_seconds", "elapsed_seconds_monotonic")
                and isinstance(v, (int, float))
            ),
            None,
        )
    httpnames = [
        n
        for n in files
        if n.startswith("payload/") and (n.endswith("http.jsonl") or n.endswith("requests.jsonl"))
    ]
    for name in httpnames:
        with checked(name).open() as f:
            for line in f:
                j = json.loads(line)
                rec["http_body_bytes"] = rec.get("http_body_bytes", 0) + j.get("body_bytes", 0)
                rec["http_requests"] = rec.get("http_requests", 0) + 1
    return rec


def run(store, pins, dest):
    rows = [row(store, x) for x in pins]
    require(len(rows) == len({r["run_id"] for r in rows}), "duplicate run in ledger")
    table(dest, "runs.csv", rows)
    actual = shutil.disk_usage(store)
    save(
        dest / "resources.json",
        {
            "schema": "el-t0-resources-1",
            "run_id": os.environ["EL_RUN_ID"],
            "scope": "All recorder originals listed in the sealed input, including failed Mac attempts and imported Windows originals, before the ledger run itself. Time sums across hosts are work, not end-to-end elapsed time.",
            "count": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "per_stage_wall_seconds": {
                kind: sum(r["elapsed_s"] for r in rows if r["kind"] == kind)
                for kind in sorted({r["kind"] for r in rows})
            },
            "deduplicated_store_file_bytes_during_ledger": bytes_on_tree(store),
            "volume_total_bytes": actual.total,
            "volume_free_bytes_during_ledger": actual.free,
            "run_rows_sha256": sha(dest / "runs.csv"),
            "GPU_memory": "Whole-device telemetry includes other workloads. Allocator counters measure PyTorch in the individual process.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "inventory", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    run(a.store, load(a.inventory)["runs"], a.out)
