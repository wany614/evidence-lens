"""Package one batch-two source-only optimizer job and its separate frozen endpoint renderer."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
from t0_worker_pack import check, digest, encoded, load, spec, unzip_worker, write_zip

SOURCES = {"M2": ("s00", "s03"), "M4": ("s00", "s01", "s02", "s03")}
TARGETS = tuple(f"v{i:02}" for i in range(1, 7))


def pack(cfg, role, out):
    obj = cfg["scene_family_id"]
    cond = cfg["condition_id"]
    ids = SOURCES[cond]
    trial = cfg["scientific_trial_id"]
    study = load(role["study"])
    bridge = load(role["authority"])
    init = load(role["initmeta"])
    recipe = load(role["recipe"])
    sc = load(role["cameras"])
    rc = load(role["rendercameras"])
    source = load(role["provenance"])
    seed = cfg["optimizer_seed"]
    check(
        cfg["layout"] in ("M2-fixed", "M4-fixed", "M4-selected")
        and cfg["split"] in ("calibration", "development")
        and bridge["schema"] == "el-b2-reconstruction-origin-1"
        and bridge["optimizer_seed"] == recipe["seed"] == seed
        and bridge["study_sha256"] == digest(role["study"])
        and digest(role["recipe"]) == bridge["recipe_sha256"]
        and recipe["stop_step"] == 5000
        and recipe["topology"]["cap"] == 20000,
        "Unregistered seed or layout",
    )
    check(
        set(sc["views"]) == set(ids)
        and set(rc["views"]) == set(ids + TARGETS)
        and all(sc["views"][s] == rc["views"][s] for s in ids),
        "Wrong source/render camera partition",
    )
    check(
        init["run_id"] == cfg["initialization"]["run_id"]
        and init["object_id"] == source["object_id"] == obj
        and init["condition"] == source["condition"] == cond
        and init["source_camera_sha256"] == digest(role["cameras"])
        and init["init_npz_sha256"] == digest(role["init"])
        and init["source_rgb_sha256"] == {s: digest(role[s]) for s in ids},
        "Wrong input or initialization",
    )
    actual = {s: source["sources"][s]["official_view_id"] for s in ids}
    check(
        actual == cfg["official_by_slot"] and actual["s00"] == 24 and actual["s03"] == 22,
        "Source ID mismatch",
    )
    if cfg["layout"] == "M4-selected":
        check(
            cfg["selection"]
            and list(actual[s] for s in ("s01", "s02")) == sorted(cfg["selected_official_ids"])
            and all(
                n in study["development_selection"]["candidate_ids"] for n in cfg["selected_official_ids"]
            ),
            "Unsealed candidate mapping",
        )
    else:
        check(actual == study["calibration"]["fixed_" + cond], "Wrong fixed slot mapping")
    check(
        init["actual_method"] == "known_pose_sift_triangulation"
        if init["eligibility"]["qualified"]
        else init["actual_method"] == "endpoint_stratified_inverse_depth" and init["N"] == 2048,
        "Initializer gate mismatch",
    )
    tree = unzip_worker(role["worker"], study["worker_zip_sha256"])
    member_roles = {
        "init.npz": "init",
        "init.json": "initmeta",
        "source_cameras.json": "cameras",
        "render_cameras.json": "rendercameras",
        **{"sources/" + s + ".png": s for s in ids},
    }
    origins = cfg["sources"]
    parents = {
        v["run_id"]: {"run_id": v["run_id"], "manifest_sha256": v["manifest_sha256"]}
        for v in origins.values()
    }
    job = {
        "schema": "el-recovery-cell-1",
        "purpose": "formal",
        "identity": {"scene_family_id": obj, "scene_id": obj, "condition_id": cond, "split": cfg["split"]},
        "object_id": obj,
        "condition": cond,
        "trial_id": trial,
        "experiment_id": "b2-source-reconstruction",
        "stop_step": 5000,
        "recipe_sha256": digest(role["recipe"]),
        "parents": [parents[k] for k in sorted(parents)],
        "input_parents": {
            **{k: origins[v]["run_id"] for k, v in member_roles.items()},
            "recipe": origins["recipe"]["run_id"],
        },
        "files": {k: spec(role[v]) for k, v in member_roles.items()},
    }
    raw = encoded(job)
    (out / "job.json").write_bytes(raw)
    files = {"cell/" + n: role[v].read_bytes() for n, v in member_roles.items()}
    files.update(
        {
            "cell/job.json": raw,
            "recipe.json": role["recipe"].read_bytes(),
            "worker.zip": role["worker"].read_bytes(),
        }
    )
    ready = {
        "schema": "el-b2-worker-ticket-1",
        "run_id": os.environ["EL_RUN_ID"],
        "trial_id": trial,
        "object_id": obj,
        "condition": cond,
        "job_sha256": hashlib.sha256(raw).hexdigest(),
        "worker_tree_sha256": tree,
        "forced_fallback_fixture": False,
        "initialization": cfg["initialization"],
        "source": cfg["source"],
        "freeze": cfg["authority"],
        "cohort": cfg.get("cohort"),
        "selection": cfg.get("selection"),
        "layout": cfg["layout"],
        "official_by_slot": actual,
        "optimizer_seed": seed,
        "actual_method": init["actual_method"],
        "initial_points": init["N"],
        "source_eligibility": init["eligibility"],
        "render_steps": [5000],
        "target_reference_members": 0,
        "members": {n: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for n, v in files.items()},
    }
    files["READY.json"] = encoded(ready)
    (out / "READY.json").write_bytes(files["READY.json"])
    write_zip(out / "dispatch.zip", files)
    shutil.copyfile(Path(__file__).with_name("t0_worker_run.py"), out / "t0_worker_run.py")
    (out / "ticket.json").write_bytes(
        encoded(
            {
                "schema": "el-b2-transport-ticket-1",
                "run_id": os.environ["EL_RUN_ID"],
                "trial_id": trial,
                "package": spec(out / "dispatch.zip"),
                "launcher": spec(out / "t0_worker_run.py"),
                "job_sha256": ready["job_sha256"],
                "worker_sha256": study["worker_zip_sha256"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--input", action="append", nargs=2, required=True)
    a = p.parse_args()
    pack(load(a.config), {k: Path(v) for k, v in a.input}, a.out)
