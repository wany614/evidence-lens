"""Close a batch-two resource inventory using only sealed originals and measured process telemetry."""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from b2_records import stage
from evidence_lens._bundle import RUN_ID, load, save, sha, utc
from evidence_lens._capacity import bytes_on_tree
from evidence_lens._verify import require
from t0_evaluate import table
from t0_resource_ledger import row


def inventory(store):
    runs = {}
    for folder in ("runs", "quarantine"):
        for seal in (store / folder).glob("*/completion.json"):
            rid = seal.parent.name
            if not RUN_ID.fullmatch(rid):
                continue
            completion = load(seal)
            digest = completion["manifest_sha256"]
            require(rid == completion["run_id"] and runs.setdefault(rid, digest) == digest,
                    "conflicting recorder identities at inventory close")
    require(runs, "empty resource inventory")
    return {"closed_utc": utc(), "runs": [
        {"run_id": k, "manifest_sha256": v} for k, v in sorted(runs.items())
    ]}


def census(store, snapshot, out, parents):
    rows = [row(store, pin) for pin in snapshot["runs"]]
    require(len(rows) == len({r["run_id"] for r in rows}), "duplicate recorded run")
    table(out, "runs.csv", rows)
    by_stage = {}
    for r in rows:
        label = (r["recording_host_OS"], r["kind"])
        group = by_stage.setdefault(label, {"runs": 0, "elapsed_s": 0.0, "failed": 0})
        group["runs"] += 1
        group["elapsed_s"] += r["elapsed_s"]
        group["failed"] += r["status"] != "succeeded"
    vol = shutil.disk_usage(store)
    save(out / "resources.json", {
        "schema": "el-b2-resources-1", "run_id": os.environ["EL_RUN_ID"],
        "inventory_closed_utc": snapshot["closed_utc"], "parents": parents,
        "scope": "Originals present in this batch store at the inventory close, including failed attempts and received Windows runs. Subsequent sealing and publication are outside this census. Parallel elapsed times sum worker effort.",
        "count": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
        "by_platform_and_stage": [{"platform": p, "kind": k, **v} for (p, k), v in sorted(by_stage.items())],
        "sampled_tree_rss_max_bytes_across_runs": max((r["sampled_tree_RSS_peak_bytes"] for r in rows if r["sampled_tree_RSS_peak_bytes"] is not None), default=None),
        "sampled_device_used_max_MiB_across_runs": max((r["sampled_whole_GPU_used_peak_MiB"] for r in rows if r["sampled_whole_GPU_used_peak_MiB"] is not None), default=None),
        "http_log_requests": sum(r.get("http_requests", 0) for r in rows),
        "http_response_body_bytes": sum(r.get("http_body_bytes", 0) for r in rows),
        "store_hard_link_deduplicated_file_bytes_at_measurement": bytes_on_tree(store),
        "volume_free_bytes_at_measurement": vol.free, "volume_total_bytes": vol.total,
        "measurement_notes": "HTTP bytes sum recorded HTTP body logs. Return package and local store sizes measure different boundaries. RSS is the sampled process tree; device memory includes other GPU workloads. JSON null denotes unavailable telemetry.",
        "run_rows_sha256": sha(out / "runs.csv"),
    })


def main(a):
    if a.action == "worker":
        return census(a.store, load(a.inventory), a.out, load(a.config)["sources"])
    controller = a.store / "work/controller"
    first_close = controller / "closed-b2-originals.json"
    path = controller / "closed-b2-platform-originals.json" if a.with_platform_readback else first_close
    if not path.exists():
        save(path, inventory(a.store))
    parents = load(a.evidence)
    closed = {r["run_id"]:r["manifest_sha256"] for r in load(path)["runs"]}
    if a.with_platform_readback:
        require(all(closed.get(p["run_id"]) == p["manifest_sha256"] for p in load(first_close)["runs"]),
                "the platform supplement omits a previously closed original")
    pending = list(parents.values())
    seen = set()
    while pending:
        pin = pending.pop()
        rid, digest = pin["run_id"], pin["manifest_sha256"]
        require(closed.get(rid) == digest, "evidence is newer than the resource close: " + rid)
        if rid in seen:
            continue
        seen.add(rid)
        directory = a.store / "runs" / rid
        if not directory.is_dir():
            directory = a.store / "quarantine" / rid
        manifest = load(directory / "manifest.json")
        refs = manifest["parents"] + ([manifest["retry_of"]] if manifest.get("retry_of") else [])
        pending.extend(p for p in refs if p["run_id"] in closed
                       or any((a.store / k / p["run_id"]).is_dir() for k in ("runs", "quarantine")))
    name = "b2-resource-census-platform-readback" if a.with_platform_readback else "b2-resource-census"
    scope = ("Cumulative recorder originals through the final Windows CPU readback, including the first census"
             if a.with_platform_readback else "Measured original recorder runs through the batch B evidence close")
    return stage(a.store.resolve(), a.history.resolve(), name, "report", parents,
                 {"scope": scope},
                 ["{python}", "{source}/scripts/b2_resource_census.py", "worker", "--store", str(a.store.resolve()),
                  "--inventory", "{input:inventory}", "--config", "{config}", "--out", "{out}"],
                 "payload/resources.json", 700000, {"inventory": path}, a.retry_of)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seal", "worker"])
    for arg in ("store", "history", "inventory", "config", "out", "evidence"):
        parser.add_argument("--" + arg, type=Path)
    parser.add_argument("--retry-of")
    parser.add_argument("--with-platform-readback", action="store_true")
    print(json.dumps(main(parser.parse_args()), sort_keys=True))
