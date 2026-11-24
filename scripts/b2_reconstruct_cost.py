"""Read the measured costs and numerical endpoints of a sealed, source-only batch-two reconstruction."""

import csv
from functools import lru_cache
import json
from math import isfinite
from evidence_lens._bundle import load, sha
from evidence_lens._verify import require
from evidence_lens.formal_inputs import PinnedRun


def peak(values):
    finite = [float(v) for v in values if v not in ("", None) and isfinite(float(v))]
    return max(finite, default=None)


@lru_cache(maxsize=512)
def stage(store, rid, manifest):
    pin = {"run_id": rid, "manifest_sha256": manifest}
    original = PinnedRun(store, pin)
    with original.open("telemetry.csv").open(newline="") as f:
        telemetry = list(csv.DictReader(f))
    hardware = []
    for row in telemetry:
        for dev in json.loads(row["gpu_devices"] or "[]") or []:
            if isinstance(dev, dict):
                hardware.append(dev.get("memory.used"))
    return {
        "run_id": rid,
        "manifest_sha256": manifest,
        "run_kind": original.run["run_kind"],
        "status": original.run["execution_status"],
        "recorder_elapsed_s": original.run["elapsed_seconds_monotonic"],
        "sampled_tree_rss_peak_bytes": peak(row["tree_rss_bytes"] for row in telemetry),
        "sampled_all_device_memory_peak_MiB": peak(hardware),
        "sampled_process_gpu_memory_peak_bytes": peak(row["gpu_process_memory_bytes"] for row in telemetry),
    }


def stage_pin(store, p):
    return stage(store, p["run_id"], p["manifest_sha256"])


def observed(store, observation, readback, receipt):
    roles = {r: p for r, p in receipt["origins"].items() if r in ("init", "generation", "render", "dispatch")}
    roles.update(source=observation, readback=readback)
    stages = {r: stage_pin(store, p).copy() for r, p in roles.items()}
    raw = store / "transport/Returns/b2-v1" / receipt["trial_id"] / "completed.json"
    require(sha(raw) == receipt["return_summary_sha256"], "Return size inventory does not match the readback")
    transfer = load(raw)
    for role, member in (("generation", "payload/reconstruction.json"), ("render", "payload/outputs.json")):
        if role in roles and stages[role]["status"] == "succeeded":
            run = PinnedRun(store, roles[role])
            f = load(run.open(member))
            resources = f["resources"]
            for key in ("torch_allocated_peak_bytes", "torch_reserved_peak_bytes"):
                stages[role][key] = resources[key]
            stages[role]["worker_wall_seconds"] = f["worker_wall_seconds"]
    src = PinnedRun(store, observation)
    meta = load(src.open(observation["member"]))
    returned = sum(f["bytes"] for f in transfer["files"].values()) + raw.stat().st_size
    return {
        "physical_trial_id": receipt["trial_id"],
        "stages": stages,
        "windows_return_files_including_summary": len(transfer["files"]) + 1,
        "windows_return_object_bytes_including_summary": returned,
        "windows_return_summary_sha256": sha(raw),
        "source_candidate_http_body_bytes": meta.get("candidate_http_response_bytes"),
        "endpoint_pipeline_elapsed_s_if_charged_to_this_trajectory": sum(
            r["recorder_elapsed_s"] for r in stages.values()
        ),
        "scope": "Sum of observed source/init/dispatch/generation/render/readback stage elapsed seconds. Shared original M2, query renders, policy evaluation, interstage waits and network overhead are accounted separately.",
    }


def unique_cost(rows):
    stages = {}
    returns = {}
    http = {}
    for row in rows:
        cost = row["cost"]
        key = cost["physical_trial_id"]
        returns[key] = cost["windows_return_object_bytes_including_summary"]
        for role, data in cost["stages"].items():
            table = stages.setdefault(role, {})
            require(
                data["run_id"] not in table or table[data["run_id"]] == data,
                "Conflicting resource observation",
            )
            table[data["run_id"]] = data
        source = cost["stages"]["source"]["run_id"]
        http[source] = cost["source_candidate_http_body_bytes"]
    return {
        "physical_reconstructions": len(returns),
        "windows_return_object_bytes_including_summary": sum(returns.values()),
        "source_candidate_http_body_bytes": sum(x for x in http.values() if x is not None),
        "unique_source_runs_with_recorded_candidate_http": sum(x is not None for x in http.values()),
        "stages": {
            k: {
                "unique_runs": len(t),
                "sum_recorder_elapsed_s": sum(x["recorder_elapsed_s"] for x in t.values()),
                "max_sampled_tree_rss_bytes": peak(x["sampled_tree_rss_peak_bytes"] for x in t.values()),
                "max_torch_allocated_bytes": peak(x.get("torch_allocated_peak_bytes") for x in t.values()),
                "max_torch_reserved_bytes": peak(x.get("torch_reserved_peak_bytes") for x in t.values()),
            }
            for k, t in stages.items()
        },
    }


def access_proof(store, selection, source, meta, decision):
    """Read only the two authorized member ranges, source roles and immutable decision timing."""
    from datetime import datetime

    decision_run = PinnedRun(store, selection)
    source_run = PinnedRun(store, source)
    decision_seal = load(decision_run.root / "completion.json")["sealed_at_utc"]
    opened = source_run.run["started_at_utc"]
    require(
        datetime.fromisoformat(decision_seal) <= datetime.fromisoformat(opened),
        "Photo access predates the selection seal",
    )
    inputs = load(source_run.open("inputs.json"))["inputs"]
    files = {r["role"]: r["files"][0]["path"] for r in inputs}
    require(
        set(k for k in files if k in ("s00", "s01", "s02", "s03")) == {"s00", "s03"}
        and set(k for k in files if k.startswith("projection")) == {"projections01", "projections02"}
        and decision["legal_photo_ids_before_selection"] == [24, 22],
        "Incorrect preselected photo roles",
    )
    start = load(source_run.open(files["start"]))
    require(
        start["evaluation_poses_in_selector"]
        == start["candidate_RGB_opened"]
        == start["evaluation_photographs_opened"]
        == 0
        and start["checkpoint_sha256"] == decision["m2_checkpoint_sha256"],
        "Unexpected selector input",
    )
    directory = {r["name"]: r for r in json.loads(source_run.open(files["directory"]).read_bytes())}
    archive = load(source_run.open(files["archive"]))
    requests = [json.loads(x) for x in source_run.open("payload/http.jsonl").read_text().splitlines()]
    require(
        len(requests) == 10 and source_run.run["parent_run_ids"].count(selection["run_id"]) == 1,
        "Unexpected raw range requests or missing authorization parent",
    )
    photo_bytes = 0
    for group, sid, n in zip(
        (requests[:5], requests[5:]), ("s01", "s02"), decision["selected_official_ids"], strict=True
    ):
        name = f"SampleSet/MVS Data/Rectified/scan{int(meta['object_id'][3:])}/rect_{n:03d}_3_r5000.png"
        record = directory[name]
        header = record["offset"]
        name_bytes = group[3]["body_bytes"]
        start_data = header + 30 + name_bytes
        expected = [
            [0, 3],
            [header, header + 29],
            [header, header + 29],
            [header + 30, start_data - 1],
            [start_data, start_data + record["compressed"] - 1],
        ]
        require(
            len(name.encode()) <= name_bytes <= len(name.encode()) + 65535
            and [q["range"] for q in group] == expected
            and all(
                q["url"] == archive["url"]
                and q["etag"] == archive["etag"]
                and q["last-modified"] == archive["last_modified"]
                and q["status"] == 206
                and q["body_bytes"] == q["range"][1] - q["range"][0] + 1
                for q in group
            ),
            "Observed HTTP exceeds the two permitted named members",
        )
        actual = meta["sources"][sid]
        require(
            actual["official_view_id"] == n
            and actual["original_bytes"] == record["bytes"]
            and actual["original_crc32"] == record["crc32"]
            and source_run.files["payload/original/" + sid + ".png"]["sha256"] == actual["original_sha256"],
            "Original readback or official CRC differs",
        )
        photo_bytes += group[-1]["body_bytes"]
    total = sum(q["body_bytes"] for q in requests)
    require(total == meta["candidate_http_response_bytes"], "Source network accounting differs")
    return {
        "decision_sealed_at_utc": decision_seal,
        "source_recorder_started_at_utc": opened,
        "selector_rgb_ids": [24, 22],
        "selector_evaluation_poses": 0,
        "selected_official_ids": decision["selected_official_ids"],
        "observed_named_origin_ranges": len(requests),
        "source_pixel_body_bytes": photo_bytes,
        "metadata_body_bytes": total - photo_bytes,
        "total_http_body_bytes": total,
    }


def historical_start(store, old, packet, doc):
    carrier = PinnedRun(store, packet)
    source = load(carrier.open("config.resolved.yaml"))["sources"]
    require(
        set(source) == {"cameras", "checkpoint", "r00", "r03", "recipe", "s00", "s03"}
        and doc["evaluation_poses_in_selector"]
        == doc["candidate_RGB_opened"]
        == doc["evaluation_photographs_opened"]
        == 0
        and doc["checkpoint_sha256"] == source["checkpoint"]["sha256"],
        "Different historical starting packet",
    )
    g = stage_pin(old, source["checkpoint"])
    r = stage_pin(old, source["r00"])
    require(
        source["r00"]["run_id"] == source["r03"]["run_id"] and doc["condition"] == "M2",
        "Wrong native source renders",
    )
    return {
        "object_id": doc["object_id"],
        "selector_packet": packet,
        "checkpoint": source["checkpoint"],
        "historical_optimizer": g,
        "historical_renderer": r,
        "selector_packet_build": stage_pin(store, packet),
        "new_historical_fit_runs": 0,
        "renderer_scope": "The historical run renders its original checkpoint grid. Its wall time is not an isolated step-5000 endpoint render time.",
    }
