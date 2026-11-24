"""Report actual per-stage costs and project only numerical T1 products into the public result directory."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from b2_reconstruct_cost import stage_pin
from make_handoff import publish
from t0_evaluate import table
from t0_resource_ledger import row as original_cost
from t1_access import OBJECTS, POOL, S0, STRATEGIES
from t1_study import DEFAULT, bind, locate, selected, stage

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require


def inventory(store):
    found = []
    for folder in ("runs", "quarantine"):
        for run in sorted((store / folder).glob("*/run.json")):
            d = load(run)
            trial = d["trial_id"]
            if (
                d["experiment_id"] == "general-t1-confirmation"
                or d["experiment_id"] == "general-1"
                and trial.startswith("general-t1-")
            ):
                if trial == "t1-resources":
                    continue
                p = run.parent / "completion.json"
                if p.is_file():
                    found.append({"run_id": d["run_id"], "manifest_sha256": load(p)["manifest_sha256"]})
    require(len(found) == len({r["run_id"] for r in found}), "Duplicate resource-inventory entry")
    return found


def probe_receipts(roots, entry, freeze):
    """Reconstruct the exact native probe configs from their separately pinned Mac and Windows sources."""
    store = roots[0]
    frozen = load(locate(roots, freeze))
    registered = frozen["sources"]
    expected_code = frozen["selection_code"]["scripts/b2_probe_worker.py"]
    rows = []
    for obj in OBJECTS:
        readback = load(locate(roots, selected(store, "t1-readback-" + obj + "-M2-fixed")))
        ret = store / "transport/Returns/t1-v1/probes" / obj / "completed.json"
        incoming = load(ret)
        p = store / "runs" / incoming["run_id"]
        cp = readback["checkpoints"][-1]
        g = readback["origins"]["generation"]
        expected = {
            "experiment_id": "general-t1-confirmation",
            "trial_id": "t1-probe-" + obj,
            "object_id": obj,
            "recording": {"expected_output_bytes": 38000000},
            "pool": POOL,
            "checkpoint_sha256": cp["checkpoint_sha256"],
            "checkpoint_producer": g["run_id"],
            "cameras_sha256": registered["query_cameras"]["sha256"],
            "recipe_sha256": readback["recipe_sha256"],
            "entry": entry,
            "frozen_selector": freeze,
            "selection_freeze": registered["selector"],
            "registration_sha256": readback["job_sha256"],
            "input_roles": ["checkpoint", "cameras"],
        }
        cfg_hash = hashlib.sha256(encoded(expected)).hexdigest()
        require(
            cfg_hash == incoming["config_sha256"]
            and sha(p / "manifest.json") == incoming["manifest_sha256"]
            and sha(p / "source/scripts/b2_probe_worker.py") == expected_code
            and incoming["checkpoint_producer"] == g["run_id"]
            and incoming["checkpoint_sha256"] == cp["checkpoint_sha256"]
            and incoming["cameras"] == registered["query_cameras"]
            and sha(ret.parent / incoming["zip"]) == incoming["zip_sha256"],
            "Probe executable, configuration, checkpoint producer, camera authority or original carrier changed",
        )
        rows.append(
            {
                "object_id": obj,
                "final_gaussians": cp["N"],
                "generation": g["run_id"],
                "probe_run": incoming["run_id"],
                "status": incoming["status"],
                "config_sha256": cfg_hash,
                "checkpoint_sha256": cp["checkpoint_sha256"],
                "camera_sha256": registered["query_cameras"]["sha256"],
                "worker_sha256": expected_code,
                "zip_sha256": incoming["zip_sha256"],
            }
        )
    return rows


def costs(roots, cfg, out):
    store = roots[0]
    pins = cfg["sources"]
    t = load(locate(roots, pins["terminals"]))
    probes = probe_receipts(roots, t["entry"], t["freeze"])
    table(out, "probe-lineage.csv", probes)
    failed_probes = {row["probe_run"] for row in probes if row["status"] == "failed"}
    trajectories, receipts = [], []
    cells = {(c["object_id"], c["condition"]): c for c in t["cells"]}
    for obj in OBJECTS:
        pin = selected(store, "t1-decisions-" + obj)
        joint = load(locate(roots, pin))
        for strategy in STRATEGIES:
            c = cells[obj, "M4-" + strategy]
            decision_pin = bind(roots, pin, "payload/" + strategy + ".json")
            original = load(locate(roots, decision_pin))
            pair = joint["results"][strategy]
            obs = c["source"]
            raw = load(locate(roots, obs)) if obs else None
            observed = [raw["official_by_slot"][k] for k in ("s01", "s02")] if raw else None
            require(
                decision_pin["sha256"] == joint["selections"][strategy]
                and hashlib.sha256(encoded(original)).hexdigest() == decision_pin["sha256"]
                and pair == original["selected_official_ids"] == observed
                and (raw is not None or c.get("disposition") == pin),
                "Strategy-specific source observation differs from its sealed pair or failure",
            )
            receipts.append({"decision": decision_pin, "original_receipt": original, "observation": obs})
            trajectories.append(
                {
                    "object_id": obj,
                    "strategy": strategy,
                    "decision_status": "selected" if pair else "failed",
                    "decision_run": pin["run_id"],
                    "decision_document_sha256": joint["selections"][strategy],
                    "authorized_new_camera_ids": " ".join(map(str, pair or [])),
                    "observed_new_camera_ids": " ".join(map(str, observed or [])),
                    "observed_count": len(observed or []),
                    "source_run": obs["run_id"] if obs else None,
                    "reconstruction_status": c["status"],
                    "alias_of": c["alias_of"],
                }
            )
    table(out, "trajectory-observations.csv", trajectories)
    save(out / "strategy-receipts.json", {"schema": "el-t1-strategy-receipts-1", "trajectories": receipts})
    partial = []
    for pin in cfg["closed_runs"]:
        r = original_cost(store, pin)
        source = store / "runs" / pin["run_id"]
        if not source.is_dir():
            source = store / "quarantine" / pin["run_id"]
        manifest = {m["path"] for m in load(source / "manifest.json")["files"]}
        tags = load(source / "run.json")
        r.update(trial=tags["trial_id"], experiment=tags["experiment_id"], fail_reason=tags["failure_reason"])
        if r["status"] != "succeeded":
            if r["run_id"] in failed_probes:
                r["failure_class"] = "frozen scientific probe"
            elif str(r["fail_reason"]).startswith("BundleError: input diagnostic references an undeclared parent"):
                r["failure_class"] = "recorder input preparation"
            else:
                r["failure_class"] = "see original run diagnostic"
        if "payload/render.json" in manifest:
            native = load(source / "payload/render.json")
            r["algorithm_worker_s"] = native["worker_wall_seconds"]
            for k in ("torch_allocated_peak_bytes", "torch_reserved_peak_bytes"):
                r[k] = native["resources"][k]
        if "payload/choices.json" in manifest:
            decision = load(source / "payload/choices.json")
            if decision["schema"] == "el-b2-joint-nbv-choices-1":
                r["feature_score_s"] = decision["feature_and_score_s"]
                r["selector_s"] = decision["selection_stage_s"]
        partial.append(r)
    table(out, "resource-runs.csv", partial)
    known = {r["run_id"] for r in partial}
    fit_ids = set()
    rows = []
    for c in t["cells"]:
        if c["ticket"] is None:
            require(
                c["status"] == "failed" and c["source"] is None and c["disposition"],
                "Unproven unexecuted condition",
            )
            rows.append(
                {
                    "object_id": c["object_id"],
                    "condition": c["condition"],
                    "alias_of": None,
                    "physical_fit": 0,
                    "source_http_response_bytes": 0,
                    "source_access_s": None,
                    "ordered_photos": "",
                    "logical_new_photos": 0,
                    "physical_init_s": 0,
                    "physical_fit_s": 0,
                    "physical_endpoint_render_s": 0,
                    "upload_packet_bytes": 0,
                    "windows_return_bytes": 0,
                    "fit_run": None,
                    "endpoint_run": None,
                    "reason": c["reason"],
                }
            )
            continue
        options = c["alias_of"] is None
        for key in ("generation", "render"):
            if c[key]:
                require(c[key]["run_id"] in known, "Missing original CUDA run in closed resource inventory")
        rec = load(locate(roots, c["readback"]))
        ids = [c[k]["run_id"] for k in ("initialization", "generation", "render") if c[k]]
        source = load(locate(roots, bind(roots, c["source"], "payload/access.json")))
        tfile = store / "transport/Returns/t1-v1" / rec["trial_id"] / "completed.json"
        require(sha(tfile) == rec["return_summary_sha256"], "Changed Windows resource inventory")
        transport = load(tfile)
        downloaded = sum(m["bytes"] for m in transport["files"].values()) + tfile.stat().st_size
        ticket = load(locate(roots, c["ticket"]))
        gen = stage_pin(store, c["generation"])
        rnd = stage_pin(store, c["render"]) if c["render"] else None
        init = stage_pin(store, c["initialization"])
        if options:
            require(not (set(ids) & fit_ids), "Duplicate actual fit in cost totals")
            fit_ids.update(ids)
        rows.append(
            {
                "object_id": c["object_id"],
                "condition": c["condition"],
                "alias_of": c["alias_of"],
                "physical_fit": int(options),
                "source_http_response_bytes": source["HTTP_response_bytes"],
                "source_access_s": source["elapsed_seconds"],
                "ordered_photos": " ".join(map(str, c["official_camera_order"])),
                "logical_new_photos": 2,
                "physical_init_s": init["recorder_elapsed_s"] if options else 0,
                "physical_fit_s": gen["recorder_elapsed_s"] if options else 0,
                "physical_endpoint_render_s": (rnd["recorder_elapsed_s"] if rnd else 0) if options else 0,
                "upload_packet_bytes": ticket["package"]["bytes"] if options else 0,
                "windows_return_bytes": downloaded if options else 0,
                "fit_run": c["generation"]["run_id"],
                "endpoint_run": c["render"]["run_id"] if c["render"] else None,
            }
        )
    table(out, "resource-conditions.csv", rows)
    fits = sum(x["physical_fit"] for x in rows)
    require(
        fits == t["reconstruction_fit_count"], "Fit accounting differs from the scientific condition manifest"
    )
    photo_rows = []
    for c in t["cells"]:
        if c["source"] is None:
            continue
        doc = load(locate(roots, c["source"]))
        for slot, image in doc["sources"].items():
            photo_rows.append(
                {
                    "object_id": c["object_id"],
                    "condition": c["condition"],
                    "camera": image["official_view_id"],
                    "role": slot,
                    "original_bytes": image["original_bytes"],
                    "original_sha256": image["original_sha256"],
                    "crop_bytes": image["rgb"]["bytes"],
                    "crop_sha256": image["rgb"]["sha256"],
                    "acquisition": image["acquisition"]["kind"],
                }
            )
    opening = selected(store, "t1-opening")
    opening_seal = load(store / "runs" / opening["run_id"] / "completion.json")["sealed_at_utc"]
    access_begins = []
    for o in OBJECTS:
        refpin = pins["ref-" + o]
        d = load(locate(roots, refpin))
        started = load(store / "runs" / refpin["run_id"] / "run.json")["started_at_utc"]
        require(
            d["opening"] == opening and started >= opening_seal,
            "Reference acquisition preceded the joint opening",
        )
        access_begins.append(started)
        for k, v in d["views"].items():
            photo_rows.append(
                {
                    "object_id": o,
                    "condition": "joint-reference",
                    "camera": v["official_view_id"],
                    "role": k,
                    "original_bytes": v["original_bytes"],
                    "original_sha256": v["original_sha256"],
                    "crop_bytes": v["rgb"]["bytes"],
                    "crop_sha256": v["rgb"]["sha256"],
                    "acquisition": v["acquisition"]["kind"],
                }
            )
    table(out, "photo-exposure.csv", photo_rows)
    logical = [(r["object_id"], r["camera"]) for r in photo_rows if r["condition"] != "joint-reference"]
    by_object = []
    for obj in OBJECTS:
        sources = {cam for o, cam in logical if o == obj}
        candidates = sorted(sources.intersection(POOL))
        decision_count = sum(
            row["decision_status"] == "selected" for row in trajectories if row["object_id"] == obj
        )
        require(
            set(S0) <= sources and (decision_count == 5 or not candidates),
            "Candidate exposure without all five decisions",
        )
        refs = {r["camera"] for r in photo_rows if r["object_id"] == obj and r["condition"] == "joint-reference"}
        by_object.append(
            {
                "object_id": obj,
                "sealed_pair_decisions": decision_count,
                "candidate_access_authorized": decision_count == 5,
                "distinct_source_photos_before_opening": len(sources),
                "distinct_candidate_photos_before_opening": len(candidates),
                "observed_candidate_ids": " ".join(map(str, candidates)),
                "reference_photos_before_opening": 0,
                "reference_photos_after_opening": len(refs),
            }
        )
    table(out, "object-access.csv", by_object)
    source_fetches = sum(
        r["acquisition"] == "exact named HTTP ranges"
        for r in photo_rows
        if r["condition"] != "joint-reference"
    )
    statuses = Counter(r["status"] for r in partial)
    by_kind = defaultdict(float)
    for r in partial:
        by_kind[r["kind"]] += r["elapsed_s"]
    save(
        out / "resources.json",
        {
            "schema": "el-t1-resources-1",
            "run_id": os.environ["EL_RUN_ID"],
            "recorded_runs_before_this_resource_census": len(partial),
            "run_status": dict(statuses),
            "failed_run_classes": dict(Counter(r["failure_class"] for r in partial if r.get("failure_class"))),
            "stage_s": dict(by_kind),
            "actual_fits": fits,
            "aliases": t["aliases"],
            "undispatched_conditions": t["undispatched_conditions"],
            "success_conditions": 42 - t["failed_conditions"],
            "failed_conditions": t["failed_conditions"],
            "source_trajectory_observations": len(logical),
            "physical_source_photos": len(set(logical)),
            "uncached_source_fetches": source_fetches,
            "unique_reference_photos": len(
                {(r["object_id"], r["camera"]) for r in photo_rows if r["condition"] == "joint-reference"}
            ),
            "physical_fit_recorder_s": sum(r["physical_fit_s"] for r in rows),
            "physical_endpoint_recorder_s": sum(r["physical_endpoint_render_s"] for r in rows),
            "actual_dispatch_zip_bytes": sum(r["upload_packet_bytes"] for r in rows),
            "actual_windows_return_bytes": sum(r["windows_return_bytes"] for r in rows),
            "store_bytes_free_at_census": shutil.disk_usage(store).free,
            "new_risk_training_count": 0,
            "public_photographic_bytes": 0,
            "joint_opening_sealed_utc": opening_seal,
            "first_reference_acquisition_started_utc": min(access_begins),
        },
    )


def run(roots):
    s = roots[0]
    pins = {
        "terminals": selected(s, "t1-terminals"),
        "nbv": selected(s, "t1-nbv-evaluation"),
        "audit": selected(s, "t1-nbv-independent"),
    }
    pins.update({"ref-" + o: selected(s, "t1-reference-" + o) for o in OBJECTS})
    closed = s / "work/t1-resource-inventory.json"
    if not closed.is_file():
        publish(closed, encoded(inventory(s)))
    original_inventory = closed.read_bytes()
    runs = json.loads(original_inventory)
    require(
        isinstance(runs, list) and encoded(runs) == original_inventory,
        "Expected the original canonical run inventory",
    )
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/t1_resources.py",
        "worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--store",
        str(s),
    ]
    return stage(
        roots, "t1-resources", "report", pins, {"closed_runs": runs}, cmd, "resources.json", 900000
    )


def publish_public(roots, dest):
    mapping = {
        "t1-terminals": [
            "conditions.csv",
            "decisions.csv",
            "observations.csv",
            "source-access.csv",
            "query-cost.csv",
            "pair-utilities.csv",
            "terminals.json",
        ],
        "t1-nbv-evaluation": [
            "reconstruction-views.csv",
            "reconstruction-conditions.csv",
            "reconstruction-object-equal.csv",
            "paired.csv",
            "nbv.json",
        ],
        "t1-nbv-independent": ["independent-nbv.json"],
        "t1-resources": [
            "probe-lineage.csv",
            "strategy-receipts.json",
            "trajectory-observations.csv",
            "object-access.csv",
            "resource-runs.csv",
            "resource-conditions.csv",
            "photo-exposure.csv",
            "resources.json",
        ],
    }
    for stage_name, names in mapping.items():
        pin = selected(roots[0], stage_name)
        for name in names:
            asset = bind(roots, pin, "payload/" + name)
            publish(dest / name, locate(roots, asset).read_bytes())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("census", "worker", "publish"))
    p.add_argument("--store", type=Path, default=DEFAULT)
    p.add_argument("--dest", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    roots = a.store.resolve(), a.store.resolve() / "history"
    if a.action == "census":
        print(json.dumps(run(roots)))
    elif a.action == "worker":
        costs(roots, load(a.config), a.out)
    else:
        require(a.dest is not None, "Explicit result directory required")
        publish_public(roots, a.dest)
