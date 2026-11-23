"""Freeze the acquisition budget from four selected B0 training families and their actual transports."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from evidence_lens._bundle import load, save, sha
from evidence_lens._capacity import bytes_on_tree
from evidence_lens._verify import require, verify_bundle


def measure_run(store, pin, family):
    root = store / "runs" / pin["run_id"]
    proof = verify_bundle(root)
    run = proof["run"]
    require(proof["manifest_sha256"] == pin["manifest_sha256"] and run["scene_family_id"] == family
            and run["split"] == "train" and run["execution_status"] == "succeeded", "wrong pilot run or split")
    result = {"run_id": run["run_id"], "manifest_sha256": proof["manifest_sha256"],
              "seconds": run["elapsed_seconds_monotonic"], "tree_peak_rss_bytes": run["sampled_peak_tree_rss_bytes"],
              "sealed_logical_bytes": proof["bytes"], "manifest_members": len(proof["manifest"]["files"])}
    with (root / "telemetry.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    ram = [int(row["system_available_bytes"]) for row in samples if row["system_available_bytes"]]
    boards = [g["memory.used"] for row in samples if row["gpu_devices"]
              for g in json.loads(row["gpu_devices"]) if g.get("memory.used") is not None]
    result.update(min_sampled_available_ram_bytes=min(ram) if ram else None,
                  max_sampled_device_wide_gpu_used_MiB=max(boards) if boards else None)
    data = {"generation": "generation.json", "render": "prediction.json"}
    if run["run_kind"] in data:
        phases = load(root / "payload" / data[run["run_kind"]])["phases"].values()
        for field in ("cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes"):
            values = [ph[field] for ph in phases if ph.get(field) is not None]
            result[field] = max(values) if values else None
    return result


def calculate(a):
    study, roster, dataset = load(a.study), load(a.catalog), load(a.dataset)
    expected = set(roster["pilot_families"])
    require(len(expected) == 4 and study["catalog_sha256"] == sha(a.catalog)
            and dataset["scope"] == "engineering_pilot" and dataset["split"] == "train"
            and {v["family"] for v in dataset["views"]} == expected, "wrong pilot population")
    require(len(dataset["views"]) == 24 and {v["condition"] for v in dataset["views"]} == {"B0"},
            "pilot roster must retain all targets")
    data = {}
    for f in sorted(expected):
        selection = load(a.selections / (f + ".json"))
        require(selection["scene_family_id"] == f and selection["selection_status"] == "selected",
                "four completed pilot trials are required before expanding this acquisition line")
        stages = {k: measure_run(a.store, pin, f) for k, pin in selection["selected"].items()}
        labels = [v for v in dataset["views"] if v["family"] == f]
        require(len(labels) == 6 and len({v["view"] for v in labels}) == 6, "duplicate or missing pilot target")
        data[f] = {"selection_sha256": sha(a.selections / (f + ".json")), "stages": stages,
                   "failed_attempts": [p["run_id"] for p in selection["attempts"] if p["execution_status"] != "succeeded"],
                   "V": sum(v["V"] for v in labels), "E": sum(v["E"] for v in labels),
                   "positive_E": sum(v["positive_E"] for v in labels),
                   "sampled_rows": sum(v["sample_count"] for v in labels)}
    logical = [sum(stage["sealed_logical_bytes"] for stage in x["stages"].values()) for x in data.values()]
    cpu = [sum(x["stages"][k]["seconds"] for k in ("generation", "render")) for x in data.values()]
    exchange = load(a.dispatch)
    require(exchange["batch_id"] == "v11-b-pilot01" and len(exchange["jobs"]) == 4, "wrong pilot dispatch")
    returns = [load(p) for p in a.returns.glob("*.json")]
    needed = {p["run_id"] for v in data.values() for k, p in v["stages"].items() if k in {"generation", "render"}}
    actual = {r["run"]: r for r in returns if r.get("schema") == "el-worker-return-1"}
    require(needed <= actual.keys() and all(actual[k]["status"] == "succeeded" for k in needed),
            "missing measured Windows transfers")
    policy = load(a.policy)
    free = shutil.disk_usage(a.store).free
    active = bytes_on_tree(a.store)
    # The renderer records logical allocator reservation on WDDM, in addition to whole-device telemetry.
    # The 40-condition estimate is an acquisition planning proxy; Q1 has its own separate derivation cost.
    block = 4
    margin = 1 + policy["margin_fraction"]
    estimate = int(max(logical) * block * margin)
    proceed = free >= policy["local_free_floor_bytes"] + estimate and active < policy["local_active_batch_target_bytes"]
    report = {"schema": "el-stage2-pilot-budget-1", "status": "approved" if proceed else "capacity_review",
              "decision": "proceed" if proceed else "pause_expansion", "catalog_sha256": sha(a.catalog),
              "training_protocol_sha256": sha(a.study), "pilot_dataset_sha256": sha(a.dataset),
              "pilot_families": sorted(expected), "measurements": {"families": data,
                  "returned_containers_bytes": sum(actual[k]["bytes"] for k in needed),
                  "B0_family_sealed_logical_bytes_min_max": [min(logical), max(logical)],
                  "B0_generation_and_render_seconds_min_max": [min(cpu), max(cpu)],
                  "Windows_preflight": load(a.windows)["resources"], "Mac_free_bytes": free, "Mac_active_bytes": active},
              "resource_policy": {"four_family_logical_upper_including_margin_bytes": estimate,
                  "conservative_40_B0_equivalent_logical_bytes_including_margin": int(40 * max(logical) * margin),
                  "scientific_acquisition_cells": 40, "scientific_network_calls": 32, "seven_view_outputs": 280,
                  "projection_limitations": "Includes logically repeated sealed imports. Q1 reuses B0 generation, R1 has its own source. Neural weights, optimizer, failure retries and provider cache are measured separately.",
                  "disk_limits": policy, "maximum_acquisition_block_families": block,
                  "rotation": "Check actual disk on every new receipt and producer. Archive and verify completed originals in the private store before moving them to the cold shelf. Release only local caches with verified remote originals."},
              "contracts": {"reference_study_sha256": study["reference_contract_sha256"],
                  "source": "One real B0 source image and calibrated cameras only",
                  "label": "el-labels-1, display-sRGB MAE strictly above 0.05", "sample_limit": 4096,
                  "background_grid": "Registered original lit opaque Blender studio, 512 by 512, fixed world and cameras",
                  "test": "Eight registered test families remain closed pending one complete model and threshold freeze."},
              "gpu_measurement_note": "PyTorch reservation under Windows WDDM can exceed installed dedicated memory. Device-wide sampled residency includes other processes and is a separate measure."}
    a.output.mkdir(parents=True, exist_ok=True)
    require(not any(a.output.iterdir()), "budget output must be empty")
    save(a.output / "budget.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "catalog", "dataset", "selections", "store", "windows", "dispatch", "returns", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    record = calculate(parser.parse_args())
    print(json.dumps({"decision": record["decision"], "stage": record["schema"]}))
