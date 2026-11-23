"""Bind each scheduled C reconstruction and every consumed byte to explicit original recorder seals."""

import json
import os
import shutil
from pathlib import Path

from evidence_lens.formal_inputs import compact_pin, require, signature
from evidence_lens.multisource import CONDITIONS, SOURCES, TARGETS, schema_sha256
from evidence_lens.multisource_io import Case, cameras
from evidence_lens.scoring_io import file_hash, write_json
from vision_sources import find_run, json_member, member, signed

CATALOG = "f14ca80777cf72e6f99075fb860e24cd38a6b4d9db342011173e9b216d3c66f5"
STUDY = "dd227c25648af4ab5e4e42d3fdd07702a8e46b3def36810ac9d65c55b07a7b23"
LEARNING = "23ec905c06acb361b3749c50dd1fc6e3c13920fb4f185ac3375be5d019867e5e"
TRANSFER = {
    "run_id": "20261123T132250544881Z-e115a32a4501435b",
    "manifest_sha256": "137fde135925ad311179bf83c31b041c5dd5f6389ee51bd47a0a472aebfe7f14",
}


def authorities(catalog_path, study_path, learning_path, split, cohort, roots, opening=None):
    require(
        split in ("train", "development", "test")
        and cohort in ("pilot", "formal")
        and (cohort != "pilot" or split == "train"),
        "invalid learning split or pilot",
    )
    c = signed(catalog_path, CATALOG)
    s = signed(study_path, STUDY)
    l = signed(learning_path, LEARNING)
    require(
        c["schema"] == "el-stage3-family-roster-1"
        and len(c["families"]) == 16
        and s["catalog_sha256"] == l["catalog_sha256"] == CATALOG
        and l["study_sha256"] == STUDY
        and s["feature_schema_sha256"] == l["feature_schema_sha256"] == schema_sha256(),
        "C authority mismatch",
    )
    f = {r["scene_family_id"]: r for r in c["families"]}
    counts = {k: sum(r["split"] == k for r in f.values()) for k in ("train", "development", "test")}
    require(
        len(f) == 16
        and counts == {"train": 8, "development": 4, "test": 4}
        and set(c["test_families"]) == {k for k, r in f.items() if r["split"] == "test"},
        "invalid disjoint family roster",
    )
    rows = [r for r in c["families"] if r["split"] == split and (cohort != "pilot" or r["pilot"])]
    marker = None
    if split == "test":
        require(bool(opening), "test requires explicit official opening SHA")
        from evidence_lens.stage3_gate import require_opening

        marker = require_opening(roots[0], CATALOG, opening_sha256=opening, shelf=roots[1])
        require(
            set(marker["test_families"]) == {r["scene_family_id"] for r in rows},
            "wrong test opening families",
        )
    return c, s, rows, marker


def trial_location(store, row, condition, cohort):
    require(cohort in ("pilot", "formal"), "unregistered acquisition cohort")
    pilot = cohort == "pilot"
    require(
        not pilot or (row["split"] == "train" and row["pilot"]),
        "engineering selection outside registered pilot",
    )
    trial = row["trials"][condition] + ("-pilot" if pilot else "")
    path = store / "stage3" / CATALOG / ("engineering" if pilot else "trials") / trial / "selection.json"
    return trial, path, ("engineering-pilot" if pilot else "production")


def pin_selections(roots, rows, cohort):
    """Bind exactly the caller-authorized registered cells; never enumerate the run store."""
    records = []
    for r in rows:
        f = r["scene_family_id"]
        for cond in CONDITIONS:
            trial, p, phase = trial_location(roots[0], r, cond, cohort)
            require(p.is_file() and not p.is_symlink(), "selection unavailable: " + trial)
            q = json.loads(p.read_bytes())
            require(
                q.get("family", q.get("scene_family_id")) == f
                and q.get("condition", q.get("condition_id")) == cond
                and q.get("trial_id", q.get("trial")) == trial
                and q["phase"] == phase,
                "wrong trial selection",
            )
            records.append(
                {
                    "family": f,
                    "cohort": cohort,
                    "phase": phase,
                    "condition": cond,
                    "production_budget": q.get("production_budget"),
                    "trial_id": trial,
                    "sha256": file_hash(p),
                    "selected": {k: compact_pin(v) for k, v in q["selected"].items()},
                    "failure": compact_pin(q["failure"]) if "failure" in q else None,
                }
            )
    return records


def link_asset(path, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(path, dst)
    except OSError:
        shutil.copyfile(path, dst)


def case_sources(roots, record, row, destination):
    f, c, trial = record["family"], record["condition"], record["trial_id"]
    ids = CONDITIONS[c]
    wanted, p, phase = trial_location(roots[0], row, c, record["cohort"])
    q = signed(p, record["sha256"])
    require(
        q.get("family", q.get("scene_family_id")) == f
        and q.get("condition", q.get("condition_id")) == c
        and q.get("trial_id", q.get("trial")) == wanted == trial
        and q["phase"] == phase == record["phase"]
        and q["split"] == row["split"]
        and q["generator_source_ids"] == list(ids)
        and q["target_ids"] == list(TARGETS)
        and q.get("catalog_sha256", CATALOG) == CATALOG,
        "invalid trial provenance",
    )
    require(record["production_budget"] == q.get("production_budget"), "production budget pin changed")
    if phase == "production":
        head = json.loads((roots[0] / "stage3" / CATALOG / "production-budget.json").read_bytes())
        bp = compact_pin(record["production_budget"])
        require(
            compact_pin(head["budget"]) == bp and head["catalog_sha256"] == CATALOG, "wrong production freeze"
        )
        budget = find_run(roots, bp)
        receipt, _ = json_member(budget, "payload/production-budget.json")
        require(
            budget.run["execution_status"] == "succeeded"
            and receipt["catalog_sha256"] == CATALOG
            and receipt["optimization_steps"] == q["optimization_steps"]
            and receipt["optimizer_sha256"] == q["optimizer_sha256"]
            and receipt["worker_sha256"] == q["worker_sha256"],
            "scientific acquisition is not bound to production settings",
        )
    else:
        require(
            q.get("production_budget") is None and q["optimization_steps"] == 64, "engineering pilot impostor"
        )
    pins = {k: compact_pin(v) for k, v in q["selected"].items()}
    require(pins == record["selected"], "selected run changed")
    ref, back = (find_run(roots, pins[k]) for k in ("reference", "readback"))
    require(
        all(r.run["execution_status"] == "succeeded" for r in (ref, back)),
        "reference or readback unsuccessful",
    )
    meta, refspec = json_member(ref, "payload/reference.json")
    validation, _ = json_member(ref, "payload/validation/results.json", meta["validation"])
    check, readback_spec = json_member(back, "payload/readback.json")
    require(
        meta["schema"] == "el-stage3-reference-1"
        and meta["family"] == f
        and meta["catalog_sha256"] == CATALOG
        and meta["source_ids"] == list(SOURCES)
        and meta["target_ids"] == list(TARGETS)
        and meta["run_id"] == ref.pin["run_id"]
        and validation["passed"] is True
        and check["schema"] == "el-stage3-native-readback-1"
        and check["passed"] is True
        and check["reference_run_id"] == ref.pin["run_id"]
        and check["family"] == f
        and check["reference_sha256"] == refspec["sha256"]
        and check["camera_sha256"] == meta["camera_sha256"] == row["camera_sha256"],
        "reference/validation/readback differs",
    )

    def attached(key, path, expected):
        item = q[key]
        require(
            compact_pin(item) == pins["reference"] and item["path"] == path, "selection substituted " + key
        )
        file, entry = member(ref, path, item)
        require(
            entry["sha256"] == (expected if isinstance(expected, str) else expected["sha256"]),
            "selected " + key + " differs from reference",
        )
        return file, entry

    full, full_spec = attached("cameras", "payload/cameras.json", meta["camera_sha256"])
    complete = json.loads(full.read_bytes())
    cams = cameras(complete, (*SOURCES, *TARGETS))
    camera_files = {}
    for key, names in (("source_cameras", ids), ("render_cameras", (*ids, *TARGETS))):
        path, entry = attached(key, f"payload/condition/{c}/{key}.json", meta["condition_cameras"][c][key])
        raw = json.loads(path.read_bytes())
        cameras(raw, names)
        require(all(raw["views"][k] == complete["views"][k] for k in names), "changed camera in subset")
        camera_files[key] = (path, entry)
    source_files = {}
    require(set(q["source_images"]) == set(ids), "selected unauthorized sources")
    for s in ids:
        item = q["source_images"][s]
        name = f"payload/sources/{s}.png"
        require(
            compact_pin(item) == pins["reference"] and item["path"] == name, "wrong original source parent"
        )
        path, entry = member(ref, name, item)
        require(entry["sha256"] == meta["source_images"][s]["source"]["sha256"], "wrong original source RGB")
        source_files[s] = (path, entry)
    references = {v: member(ref, "payload/reference/" + v + ".npz", meta["views"][v]["npz"]) for v in TARGETS}
    common = {
        "family": f,
        "phase": phase,
        "production_budget": q.get("production_budget"),
        "trial_id": trial,
        "job_sha256": q["job_sha256"],
        "batch_sha256": q["batch_sha256"],
        "attempts": q["attempts"],
        "condition": c,
        "selection_sha256": record["sha256"],
        "selected": pins,
        "cameras_sha256": full_spec["sha256"],
        "source_camera_sha256": camera_files["source_cameras"][1]["sha256"],
        "render_camera_sha256": camera_files["render_cameras"][1]["sha256"],
        "source_rgb_sha256": {s: v[1]["sha256"] for s, v in source_files.items()},
        "reference_sha256": refspec["sha256"],
        "readback_sha256": readback_spec["sha256"],
    }
    if q["schema"] == "el-stage3-failed-trial-1":
        fail = q["failure"]
        run = find_run(roots, compact_pin(fail))
        require(
            q["selection_status"] == "terminal_failure"
            and compact_pin(fail) == record["failure"]
            and run.run["execution_status"] in ("failed", "interrupted", "cancelled")
            and isinstance(fail["reason"], str)
            and fail["reason"].strip(),
            "invalid scientific failure",
        )
        return None, references, cams, {**common, "failure": fail}
    require(
        q["schema"] == "el-stage3-trial-selection-1" and q["selection_status"] == "selected",
        "unknown successful selection",
    )
    recon, render = (find_run(roots, pins[k]) for k in ("reconstruction", "render"))
    require(
        all(
            r.run["execution_status"] == "succeeded"
            and r.run.get("condition_id") == c
            and r.run.get("scene_family_id") == f
            for r in (recon, render)
        ),
        "wrong scientific run identity",
    )
    require(
        recon.run["run_kind"] == "generation" and render.run["run_kind"] == "render",
        "wrong reconstruction/render kinds",
    )
    for run in (recon, render):
        cfg, _ = json_member(run, "config.resolved.yaml")
        require(
            run.run["experiment_id"] == "stage3-joint-" + phase
            and run.run["trial_id"] == trial
            and run.run["dataset_hash"] == CATALOG
            and cfg["phase"] == phase
            and cfg["production_budget"] == q.get("production_budget")
            and cfg["optimization_steps"] == q["optimization_steps"]
            and cfg["optimizer_sha256"] == q["optimizer_sha256"]
            and cfg["formal_handoff_sha256"] == q["job_sha256"]
            and cfg["input_worker_sha256"] == q["worker_sha256"],
            "optimization phase/source worker differs from trial selection",
        )
    rec, rs = json_member(recon, "payload/reconstruction.json")
    pr, ps = json_member(render, "payload/prediction.json")
    checkpoint = member(recon, "payload/scene.npz", rec["checkpoint_sha256"])[1]
    probe, probe_spec = json_member(render, "payload/depth-probe.json", pr["depth_probe_sha256"])
    require(
        probe["passed"] is pr["depth_verified"]
        and rec["generator_source_ids"] == list(ids)
        and rec["run_id"] == recon.pin["run_id"]
        and pr["run_id"] == render.pin["run_id"],
        "wrong renderer/depth probe identity",
    )
    cost = {
        "gaussian_optimization_steps": rec["gaussian_optimization_steps"],
        "sampled_source_pixels": rec["sampled_source_pixels"],
        "source_sampling": rec["source_sampling"],
        "initial_gaussian_count": rec["initial_gaussian_count"],
        "final_gaussian_count": rec["final_gaussian_count"],
        "topology_events": rec["topology_events"],
        "optimizer_sha256": rec["optimizer_sha256"],
        "optimization_gpu_synchronized_seconds": rec["optimization_seconds"],
        "reconstruction_worker_seconds": rec["stage_seconds"],
        "render_worker_seconds": pr["stage_seconds"],
        "reconstruction_runtime": rec["runtime"],
        "render_runtime": pr["runtime"],
        "reconstruction_recorder_wall_seconds": recon.run.get("elapsed_seconds_monotonic"),
        "render_recorder_wall_seconds": render.run.get("elapsed_seconds_monotonic"),
        "reconstruction_peak_tree_rss_bytes": recon.run.get("sampled_peak_tree_rss_bytes"),
        "render_peak_tree_rss_bytes": render.run.get("sampled_peak_tree_rss_bytes"),
        "per_view_render_write_seconds": {v: e["seconds"] for v, e in pr["views"].items()},
    }
    require(
        set(cost["source_sampling"]) == set(ids)
        and sum(v["pixels"] for v in cost["source_sampling"].values()) == cost["sampled_source_pixels"]
        and sum(v["samples"] for v in cost["source_sampling"].values())
        == cost["gaussian_optimization_steps"],
        "invalid original joint source optimization budget",
    )
    require(
        rec["schema"] == "el-stage3-reconstruction-1"
        and pr["reconstruction_sha256"] == rs["sha256"]
        and pr["checkpoint_sha256"] == checkpoint["sha256"],
        "renderer consumed another Gaussian checkpoint",
    )
    require(
        pr["reconstruction_run_id"] == pins["reconstruction"]["run_id"]
        and rec["source_rgb_sha256"] == pr["source_rgb_sha256"] == common["source_rgb_sha256"]
        and rec["source_camera_sha256"] == pr["source_camera_sha256"] == common["source_camera_sha256"]
        and common["render_camera_sha256"] == pr["render_camera_sha256"],
        "mismatched allowed input rows",
    )
    destination.mkdir(parents=True)

    def placed(original, entry, name):
        link_asset(original, destination / name)
        return {"path": name, "bytes": entry["bytes"], "sha256": entry["sha256"]}

    m = {
        "schema": "el-stage3-deploy-case-1",
        "condition": c,
        "generator_source_ids": list(ids),
        "target_ids": list(TARGETS),
        "reconstruction_run_id": pins["reconstruction"]["run_id"],
        "checkpoint_sha256": pr["checkpoint_sha256"],
        "prediction": placed(render.open("payload/prediction.json"), ps, "prediction.json"),
        "source_images": {s: placed(*a, f"sources/{s}.png") for s, a in source_files.items()},
        "renders": {},
    }
    for key, a in camera_files.items():
        m[key] = placed(*a, key + ".json")
    for v, s in pr["views"].items():
        require(v in (*ids, *TARGETS), "unauthorized predicted view")
        m["renders"][v] = placed(*member(render, f"payload/{v}.npz", s), f"predictions/{v}.npz")
    write_json(destination / "case.json", m)
    case = Case(destination, file_hash(destination / "case.json"))
    return (
        case,
        references,
        cams,
        {
            **common,
            "prediction_sha256": ps["sha256"],
            "reconstruction_sha256": rs["sha256"],
            "checkpoint_sha256": pr["checkpoint_sha256"],
            "initial_parameter_sha256": rec["initial_parameter_sha256"],
            "initial_checkpoint_sha256": rec["initial_checkpoint_sha256"],
            "final_parameter_sha256": rec["final_parameter_sha256"],
            "case_sha256": case.sha256,
            "depth_probe_sha256": probe_spec["sha256"],
            "cost": cost,
            "failure": None,
        },
    )


def asset(root, path):
    return {"path": path.relative_to(root).as_posix(), **signature(path)}
