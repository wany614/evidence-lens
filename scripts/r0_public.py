"""Export the R0 numerical allowlist, original recorder costs and optional projection audit."""

import argparse
import csv
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens.formal_inputs import require
from make_handoff import publish
from r0_cpu import pinned

REPORT_FILES = (
    "first-step-group-gradient-parity.csv",
    "historical-log.csv",
    "image-quality.csv",
    "native-source-union.csv",
    "native-views.csv",
    "original-target-error-parity.csv",
    "parameter-rows.csv",
    "projected-views.csv",
    "projection-parity.csv",
    "source-union.csv",
    "state-quantiles.json",
    "summary.json",
)


def export(store, report_pin, out):
    import io, json

    shelf = store.parent / "cold-shelf"
    report = pinned(store, shelf, report_pin)
    cfg = load(report.open("config.resolved.yaml"))
    require(
        report.run["experiment_id"] == "r0-checkpoint-evidence-report" and report.run["run_kind"] == "report",
        "Wrong report scope",
    )
    summary = load(report.open("payload/public/summary.json"))
    require(
        summary["run_id"] == report.pin["run_id"]
        and set(summary["files"]) == set(REPORT_FILES) - {"summary.json"},
        "Incomplete numeric report",
    )
    cpu = pinned(store, shelf, cfg["cpu"])
    entries = []

    def member(owner, rel, target, license):
        src = owner.open(rel)
        digest = sha(src)
        if target in summary["files"]:
            require(digest == summary["files"][target]["sha256"], "Numerical index differs")
        publish(out / target, src.read_bytes())
        entries.append(
            {
                "path": target,
                "bytes": src.stat().st_size,
                "sha256": digest,
                "license": license,
                "source": {**owner.pin, "member": rel, "sha256": digest},
            }
        )

    for name in REPORT_FILES:
        member(report, "payload/public/" + name, name, "CC-BY-4.0")
    member(cpu, "config.resolved.yaml", "cpu-config.json", "Apache-2.0")
    member(report, "config.resolved.yaml", "report-config.json", "Apache-2.0")
    roles = [
        ("cpu_checkpoint_and_board_readback", cpu),
        ("gpu_camera_projection", pinned(store, shelf, cfg["centres"])),
        ("joined_readback", report),
    ]
    roles += [("gpu_checkpoint_" + c, pinned(store, shelf, p)) for c, p in cfg["gpu"].items()]
    rows = []
    cost_origins = []
    for role, r in roles:
        meta = r.run
        papers = ["run.json", "telemetry.csv"]
        runtime = {}
        inner = None
        with r.open("telemetry.csv").open(newline="") as f:
            t = list(csv.DictReader(f))
        if role.startswith("gpu_"):
            f = "payload/" + ("centres.json" if role == "gpu_camera_projection" else "r0-gpu.json")
            papers.append(f)
            proof = load(r.open(f))
            runtime = proof["runtime"]
            inner = proof.get("elapsed_s")
        devices = [
            float(g["memory.used"])
            for x in t
            if x.get("gpu_devices")
            for g in json.loads(x["gpu_devices"])
            if isinstance(g.get("memory.used"), (int, float))
        ]
        perproc = [int(x["gpu_process_memory_bytes"]) for x in t if x.get("gpu_process_memory_bytes")]
        rows.append(
            {
                "role": role,
                **r.pin,
                "platform": load(r.open("environment.json"))["platform"].split("-")[0],
                "status": meta["execution_status"],
                "wall_recorder_s": meta["elapsed_seconds_monotonic"],
                "worker_timer_s": inner,
                "rss_tree_sampled_B": meta.get("sampled_peak_tree_rss_bytes"),
                "n_telemetry_samples": len(t),
                "process_gpu_sampled_B": max(perproc, default=None),
                "devicewide_peak_MiB": max(devices, default=None),
                "torch_allocated_peak_B": runtime.get("torch_cuda_allocated_peak_bytes"),
                "torch_reserved_peak_B": runtime.get("torch_cuda_reserved_peak_bytes"),
                "manifest_logical_B": sum(x["bytes"] for x in r.files.values()),
                "payload_logical_B": sum(x["bytes"] for n, x in r.files.items() if n.startswith("payload/")),
            }
        )
        cost_origins.append(
            {**r.pin, "members": {n: r.files[n]["sha256"] for n in papers + ["environment.json"]}}
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = output.getvalue().encode()
    publish(out / "run-resources.csv", data)
    entries.append(
        {
            "path": "run-resources.csv",
            "bytes": len(data),
            "sha256": sha(out / "run-resources.csv"),
            "license": "CC-BY-4.0",
            "derivation": "scripts/r0_public.py reads only named sealed timing and environment members",
            "sources": cost_origins,
        }
    )
    inventory = {
        "schema": "el-r0-public-numerical-1",
        "report": report.pin,
        "cpu": cfg["cpu"],
        "files": entries,
        "author": "wany614",
        "in_scope": "Aggregated original measurements, protocol configurations, execution receipts and source identities.",
        "private": "DTU source/target photos, self/target render images, comparisons and per-row private checkpoint data remain in the research archive.",
    }
    publish(out / "public-members.json", encoded(inventory))
    return inventory



def include_roundoff(store, core, pin, out):
    import io

    shelf = store.parent / "cold-shelf"
    audit = pinned(store, shelf, pin)
    packet = load(audit.open("payload/roundoff.json"))
    require(audit.run["run_kind"] == "archive_verify"
            and audit.run["experiment_id"] == "r0-native-projection-roundoff-readback"
            and packet["schema"] == "el-r0-roundoff-readback-1" and packet["run_id"] == pin["run_id"]
            and packet["parents"]["report"] == core["report"] and packet["parents"]["cpu"] == core["cpu"]
            and packet["optimizer_steps"] == packet["photos_read"] == 0
            and packet["route_table_sha256"] == sha(audit.open("payload/roundoff-routes.csv")),
            "Projection audit disagrees with the selected report")
    members = list(core["files"])
    for rel, target, lic in (
        ("payload/roundoff.json", "roundoff.json", "CC-BY-4.0"),
        ("payload/roundoff-routes.csv", "roundoff-routes.csv", "CC-BY-4.0"),
        ("config.resolved.yaml", "roundoff-config.json", "Apache-2.0"),
    ):
        src = audit.open(rel)
        fingerprint = sha(src)
        publish(out / target, src.read_bytes())
        members.append({"path": target, "bytes": src.stat().st_size, "sha256": fingerprint,
                        "license": lic, "source": {**pin, "member": rel, "sha256": fingerprint}})
    stream = io.StringIO(newline="")
    with (out / "run-resources.csv").open(newline="") as original:
        baseline = csv.DictReader(original)
        writer = csv.DictWriter(stream, baseline.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(baseline)
    with audit.open("telemetry.csv").open(newline="") as log:
        observations = list(csv.DictReader(log))
    papers = ["run.json", "telemetry.csv", "environment.json"]
    writer.writerow({
        "role": "cpu_projection_roundoff_readback", **pin,
        "platform": load(audit.open("environment.json"))["platform"].split("-")[0],
        "status": audit.run["execution_status"], "wall_recorder_s": audit.run["elapsed_seconds_monotonic"],
        "rss_tree_sampled_B": audit.run.get("sampled_peak_tree_rss_bytes"), "n_telemetry_samples": len(observations),
        "manifest_logical_B": sum(x["bytes"] for x in audit.files.values()),
        "payload_logical_B": sum(x["bytes"] for n, x in audit.files.items() if n.startswith("payload/")),
    })
    res = stream.getvalue().encode()
    publish(out / "all-run-resources.csv", res)
    sources = next(x["sources"] for x in core["files"] if x["path"] == "run-resources.csv")
    members.append({"path": "all-run-resources.csv", "bytes": len(res), "sha256": sha(out / "all-run-resources.csv"),
                    "license": "CC-BY-4.0", "sources": [*sources, {**pin, "members": {n: audit.files[n]["sha256"] for n in papers}}],
                    "derivation": "Original seven R0 resource rows and the post hoc projection readback, each from sealed members."})
    result = {**core, "schema": "el-r0-public-numerical-2", "files": members, "projection_audit": pin,
              "original_inventory_sha256": sha(out / "public-members.json")}
    publish(out / "public-members-v2.json", encoded(result))
    return result


def pin(raw):
    rid, digest = raw.split("=")
    return {"run_id": rid, "manifest_sha256": digest}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--report", required=True, metavar="RUN_ID=MANIFEST_SHA256")
    p.add_argument("--roundoff", metavar="RUN_ID=MANIFEST_SHA256")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = export(a.store.resolve(), pin(a.report), out)
    selected = "public-members.json"
    if a.roundoff:
        result = include_roundoff(a.store.resolve(), result, pin(a.roundoff), out)
        selected = "public-members-v2.json"
    print(len(result["files"]), sha(out / selected))
