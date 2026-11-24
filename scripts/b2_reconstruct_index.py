"""Seal the fourteen authorized batch-two development observations and their unique reconstruction origins."""

import argparse
import os

import numpy as np
from evidence_lens.formal_inputs import PinnedRun
from pathlib import Path

from b2_reconstruct_cost import access_proof, historical_start, observed, unique_cost
from b2_reconstruct import bind, encoded, load, publish, record, require, resolve, sha


def describe(store, cfg, inputs, out, historical_store):
    doc = load(inputs["observations"])
    sens = load(inputs["sensitivity"])
    rows = []
    require(
        doc["schema"] == "el-b2-development-sources-and-branches-1"
        and sens["batch"] == "D-fixed-M4-sensitivity",
        "Wrong development source or sensitivity register",
    )
    specs = doc["sources"] + [
        {
            "object_id": j["object_id"],
            "branch": "seed-" + str(j["optimizer_seed"]),
            "selection": None,
            "trial_id": j["trial_id"],
            "ticket": j["ticket"],
            "reuse_of": None,
            "pair": None,
            "source": cfg["sources"]["source" + str(i + len(doc["sources"]))],
        }
        for i, j in enumerate(sens["jobs"])
    ]
    strategies = (
        "random-137101",
        "random-137102",
        "random-137103",
        "uniform",
        "risk",
        "seed-43172",
        "seed-43173",
    )
    require(
        len(specs) == 14
        and {(x["object_id"], x["branch"]) for x in specs}
        == {(o, b) for o in ("set1", "set6") for b in strategies},
        "Wrong logical observation roster",
    )
    verified = {}
    common_endpoints = {}
    for index, s in enumerate(specs):
        key = str(index)
        obj = s["object_id"]
        branch = s["branch"]
        reuse = s["reuse_of"]
        receipt = load(inputs["read" + key])
        src = load(inputs["source" + key])
        require(
            s["source"] == cfg["sources"]["source" + key]
            and src["object_id"] == obj == receipt["object_id"]
            and src["condition"] == receipt["condition"] == "M4"
            and receipt["trial_id"] == s["trial_id"]
            and receipt["official_by_slot"] == src["official_by_slot"]
            and receipt["target_reference_files_opened"] == 0,
            "Source observation and returned reconstruction do not match",
        )
        if s["selection"]:
            decision = load(inputs["decision" + key])
            require(
                cfg["sources"]["decision" + key] == src["selection"] == s["selection"]
                and decision["object_id"] == obj
                and decision["strategy_id"] == branch
                and decision["selected_official_ids"] == s["pair"]
                and [receipt["official_by_slot"][k] for k in ("s01", "s02")] == s["pair"]
                and receipt["layout"] == "M4-selected"
                and receipt["optimizer_seed"] == 43171,
                "Unregistered policy inputs",
            )
        else:
            require(
                receipt["optimizer_seed"] == int(branch.split("-")[1])
                and receipt["layout"] == "M4-fixed"
                and src["selection"] is None,
                "Wrong sensitivity",
            )
        d = {
            "object_id": obj,
            "branch": branch,
            "selection": s["selection"],
            "source": s["source"],
            "source_acquisition": s["source"],
            "trial_id": s["trial_id"],
            "readback": cfg["sources"]["read" + key],
            "ticket": s["ticket"],
            "case": None,
            "status": receipt["status"],
            "reuse_of": reuse,
            "official_by_slot": receipt["official_by_slot"],
            "s0_official_by_slot": {s: src["sources"][s]["official_view_id"] for s in ("s00", "s03")},
            "optimizer_seed": receipt["optimizer_seed"],
            "initialization": receipt["initialization"],
            "recipe_sha256": receipt["recipe_sha256"],
        }
        if s["selection"]:
            d["source_access_readback"] = access_proof(store, s["selection"], s["source"], src, decision)
        d["cost"] = observed(store, s["source"], d["readback"], receipt)
        d["initialization_method"] = receipt["actual_method"]
        d["initial_points"] = receipt["initial_points"]
        d["triangulated_tracks"] = receipt["natural_eligibility"]["independent_tracks"]
        d["scene_scale_m"] = receipt.get("scene_scale_m")
        d["trace"] = receipt.get("trace")
        d["completed_steps"] = receipt.get("completed_steps")
        if receipt["verified"]:
            case = load(inputs["case" + key])
            require(
                sha(inputs["case" + key]) == receipt["case_sha256"]
                and receipt["source_camera_sha256"] == src["source_camera_sha256"]
                and receipt["render_camera_sha256"] == src["render_camera_sha256"]
                and receipt["completed_steps"] == 5000
                and case["reconstruction_run_id"] == receipt["origins"]["generation"]["run_id"]
                and case["source_images"].keys() == src["sources"].keys()
                and all(
                    spec["sha256"] == src["sources"][sid]["rgb"]["sha256"]
                    for sid, spec in case["source_images"].items()
                ),
                "Wrong case, source bytes or terminal schedule",
            )
            source_original = PinnedRun(store, s["source"])
            cameras = load(source_original.open("payload/source_cameras.json"))["views"]
            xyz = {sid: np.array(cameras[sid]["c2w"], np.float64)[:3, 3] for sid in ("s00", "s03")}
            distance = float(np.linalg.norm(xyz["s03"] - xyz["s00"]))
            generation = PinnedRun(store, receipt["origins"]["generation"])
            reconstruction = load(generation.open("payload/reconstruction.json"))
            endpoint = {
                sid: {"camera": cameras[sid], "rgb_sha256": src["sources"][sid]["rgb"]["sha256"]}
                for sid in ("s00", "s03")
            }
            require(
                common_endpoints.setdefault(obj, endpoint) == endpoint
                and distance == reconstruction["scene_scale_m"] == receipt["scene_scale_m"]
                and reconstruction["sampling_counts"]
                == receipt["trace"]["source_updates_in_this_process"]
                == {sid: 1250 for sid in ("s00", "s01", "s02", "s03")},
                "Changed shared endpoint, scene scale or source sampling budget",
            )
            d["endpoint_baseline_recomputed_m"] = distance
            d["case"] = cfg["sources"]["case" + key]
        else:
            d.update(disposition=d["readback"], reason=receipt["failure_reasons"])
        if reuse is not None:
            origin = verified.get((obj, reuse))
            require(
                origin is not None
                and all(
                    d[k] == origin[k]
                    for k in (
                        "ticket",
                        "readback",
                        "case",
                        "official_by_slot",
                        "optimizer_seed",
                        "initialization",
                        "recipe_sha256",
                    )
                ),
                "Alias does not share the exact immutable execution and calibrated inputs",
            )
        else:
            require(
                receipt["source"] == s["source"] and receipt["selection"] == s["selection"],
                "Incorrect execution owner",
            )
        verified[obj, branch] = d
        rows.append(d)
    (out / "branches.json").write_bytes(
        encoded(
            {
                "schema": "el-b2-development-branches-1",
                "run_id": os.environ["EL_RUN_ID"],
                "observations": len(rows),
                "unique_new_reconstructions": len({r["trial_id"] for r in rows}),
                "authorizations": doc["decisions"],
                "shared_M2": {
                    obj: historical_start(
                        store, historical_store, cfg["sources"]["start" + obj], load(inputs["start" + obj])
                    )
                    for obj in ("set1", "set6")
                },
                "target_reference_files_opened": 0,
                "physical_totals": unique_cost(rows),
                "source_metadata_note": "The original development-source S0 display field inherited all four historical fixed-slot IDs. The sealed selector package, role-level inputs and legal_photo_ids_before_selection identify only s00=24 and s03=22 before joint selection. Row s0_official_by_slot is derived from those two endpoint source entries. Current source tooling emits that same two-slot S0 metadata.",
                "rows": rows,
            }
        )
    )


def seal(store, history):
    obsfile = store / "work/controller/d-authorized-branches.json"
    obs = load(obsfile)
    sf = store / "transport/Handoffs/b2-v1/D-sensitivity-4.json"
    sens = load(sf)
    pairs = obs["sources"] + [
        dict(
            object_id=j["object_id"],
            trial_id=j["trial_id"],
            source=load(store / "runs" / j["ticket"]["run_id"] / "payload/READY.json")["source"],
            selection=None,
        )
        for j in sens["jobs"]
    ]
    pins = {"authority": obs["authority"]}
    for obj in ("set1", "set6"):
        pins["start" + obj] = load(store / "work/controller" / ("b2-nbv-original-" + obj) / "selected.json")
    for i, row in enumerate(pairs):
        suffix = str(i)
        r = load(store / "work/controller" / ("b2-d-readback-" + row["trial_id"]) / "selected.json")
        receipt = load(resolve((store, history), r))
        pins["read" + suffix] = r
        pins["source" + suffix] = row["source"]
        if receipt["verified"]:
            pins["case" + suffix] = bind(store, r["run_id"], "payload/case/case.json")
        if row["selection"]:
            pins["decision" + suffix] = row["selection"]
    raws = {"observations": obsfile, "sensitivity": sf}
    command = [
        "{python}",
        "{source}/scripts/b2_reconstruct_index.py",
        "worker",
        "--store",
        str(store),
        "--history",
        str(history),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for key in list(pins) + list(raws):
        command += ["--input", key, "{input:" + key + "}"]
    pin = record(
        store,
        history,
        "b2-d-reconstruction-branches",
        "archive_verify",
        pins,
        {"experiment_id": "b2-terminal-branch-index", "trial_id": "b2-terminal-branch-index"},
        command,
        "payload/branches.json",
        2**20,
        raws,
    )
    publish(store / "work/controller/d-branches.json", resolve((store, history), pin).read_bytes())
    return pin


def calibration(store, cfg, inputs, out):
    registry = load(inputs["registry"])
    cohort = load(inputs["cohort"])
    pairs = [(o, c) for o in cohort["splits"]["C6"] for c in ("M2", "M4")]
    require(
        registry["batch"] == "C6-fixed"
        and registry["cohort"] == cfg["sources"]["cohort"]
        and len(pairs) == 12
        and [(j["object_id"], j["condition"]) for j in registry["jobs"]] == pairs,
        "Different calibrated cohort or order",
    )
    rows = []
    for n, j in enumerate(registry["jobs"]):
        role = "read" + str(n)
        p = cfg["sources"][role]
        d = load(inputs[role])
        require(
            d["object_id"] == j["object_id"]
            and d["condition"] == j["condition"]
            and d["cohort"] == registry["cohort"]
            and d["trial_id"] == j["trial_id"]
            and d["job_sha256"] == j["job_sha256"]
            and d["target_reference_files_opened"] == 0,
            "Wrong calibration case or observation scope",
        )
        cp = cfg["sources"].get("case" + str(n))
        require(
            (d["verified"] and cp and d["case_sha256"] == cp["sha256"] and d["completed_steps"] == 5000)
            or (not d["verified"] and cp is None),
            "Case lacks terminal evidence",
        )
        rows.append(
            {
                "object_id": j["object_id"],
                "condition": j["condition"],
                "source": d["source"],
                "readback": p,
                "case": cp,
                "status": d["status"],
                "initialization_method": d["actual_method"],
                "initial_points": d["initial_points"],
                "triangulated_tracks": d["natural_eligibility"]["independent_tracks"],
                "scene_scale_m": d.get("scene_scale_m"),
                "trace": d.get("trace"),
                "ticket": j["ticket"],
                "completed_steps": d.get("completed_steps"),
                "failure_reasons": d.get("failure_reasons"),
                "cost": observed(store, d["source"], p, d),
            }
        )
    result = {
        "schema": "el-b2-c6-reconstruction-index-1",
        "run_id": os.environ["EL_RUN_ID"],
        "cohort": registry["cohort"],
        "target_reference_files_opened": 0,
        "rows": rows,
        "physical_totals": unique_cost(rows),
    }
    (out / "c6-reconstructions.json").write_bytes(encoded(result))


def seal_calibration(store, history):
    reg = store / "transport/Handoffs/b2-v1/C6-fixed-12.json"
    doc = load(reg)
    pins = {"cohort": doc["cohort"], "authority": doc["authority"]}
    for i, j in enumerate(doc["jobs"]):
        o, c = j["object_id"], j["condition"]
        r = load(store / "work/controller" / f"b2-c6-readback-{o}-{c}" / "selected.json")
        pins["read" + str(i)] = r
        if load(resolve((store, history), r))["verified"]:
            pins["case" + str(i)] = bind(store, r["run_id"], "payload/case/case.json")
    cmd = [
        "{python}",
        "{source}/scripts/b2_reconstruct_index.py",
        "worker-c6",
        "--store",
        str(store),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    for role in [*pins, "registry"]:
        cmd += ["--input", role, "{input:" + role + "}"]
    p = record(
        store,
        history,
        "b2-c6-reconstruction-index",
        "archive_verify",
        pins,
        {
            "experiment_id": "b2-calibration-reconstruction-index",
            "trial_id": "b2-calibration-reconstruction-index",
        },
        cmd,
        "payload/c6-reconstructions.json",
        2**20,
        {"registry": reg},
    )
    publish(store / "work/controller/c6-reconstructions.json", resolve((store, history), p).read_bytes())
    return p


def export_indices(store, history, destination):
    selected = (
        ("b2-c6-reconstruction-index", "el-b2-c6-reconstruction-index-1", "c6-reconstructions.json", 12),
        ("b2-d-reconstruction-branches", "el-b2-development-branches-1", "development-branches.json", 14),
    )
    for name, schema, filename, count in selected:
        pin = load(store / "work/controller" / name / "selected.json")
        path = resolve((store, history), pin)
        doc = load(path)
        require(
            doc["schema"] == schema
            and doc["run_id"] == pin["run_id"]
            and doc["target_reference_files_opened"] == 0
            and len(doc["rows"]) == count,
            "Wrong numerical reconstruction index",
        )
        publish(destination / filename, path.read_bytes())
        require(sha(destination / filename) == pin["sha256"], "Published index bytes changed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["seal", "worker", "seal-c6", "worker-c6", "export"])
    for k in ("store", "history", "config", "out"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--input", nargs=2, action="append")
    a = p.parse_args()
    if a.mode == "export":
        export_indices(a.store, a.history, a.out)
    elif a.mode.startswith("worker"):
        values = {k: Path(v) for k, v in a.input}
        if a.mode == "worker-c6":
            calibration(a.store, load(a.config), values, a.out)
        else:
            describe(a.store, load(a.config), values, a.out, a.history)
    else:
        controller = seal_calibration if a.mode == "seal-c6" else seal
        print(controller(a.store.resolve(), a.history.resolve()))
