"""Establish the permitted T1 source opening from the original closed holdout and a new authorization."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from b2_t1_freeze import scopes
from evidence_lens._bundle import RUN_ID, load, save, sha, utc
from evidence_lens._verify import require
from general_records import bind, resolve, stage


BASE_COMMIT = "cdf2267a8775030cf26616aeaa472f38ec5b2f7f"
FINAL_RUN = "20261124T023243052988Z-5e3007d1b3ed48dc"
FINAL_HASH = "b59dddc26d22cb28908ba5b4d27d7795871998e878408243a178ad1c7a2bf946"
OBJECTS = ["set118","set76","set45","set103","set61","set32"]


def index(store, sources):
    """Index original directories by identity; verify every duplicate before reusing it."""
    root = store / "history" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, source in sources.items():
        n = 0
        for path in sorted((source / "runs").iterdir()):
            if not RUN_ID.fullmatch(path.name) or not (path / "completion.json").is_file():
                continue
            src = path.resolve()
            dest = root / path.name
            if dest.is_symlink():
                if dest.resolve() != src:
                    require(sha(dest / "manifest.json") == sha(src / "manifest.json"), "conflicting historical originals")
            else:
                require(not dest.exists(), "history entry is not owned by the read-only index")
                dest.symlink_to(src, target_is_directory=True)
            n += 1
        counts[name] = n
    return {"locations": counts, "unique_run_ids": sum(1 for _ in root.iterdir())}


def verify_entry(a):
    config = load(a.config)
    roots = (a.store, a.store / "history")
    pins = config["sources"]
    freeze_path = resolve(roots, pins["freeze"])
    f = load(freeze_path)
    require(sha(freeze_path) == FINAL_HASH and f["run_id"] == FINAL_RUN and
            f["T1_objects"] == OBJECTS and f["T1_photographic_opening_authorized"] is False,
            "incorrect previous holdout")
    for role, pin in f["sources"].items():
        resolve(roots, pin)
    root = Path(__file__).resolve().parents[1]
    files = {**f["scientific_code_sha256"], **f["selection_code"]}
    for group in (f["scientific_code_sha256"], f["selection_code"]):
        for name, digest in group.items():
            require(sha(root / name) == digest, "immutable scientific file differs: " + name)
    prior = load(resolve(roots, pins["prospective"]))
    cohort = load(resolve(roots, pins["cohort"]))
    require(prior["endpoints"] == f["endpoints"] and cohort["splits"]["T1"] == OBJECTS,
            "frozen roster or endpoints changed")
    prior_access = load(resolve(roots, pins["prior_access"]))
    require(prior_access["T1_photo_receipts"] == 0 and prior_access["rows"] == scopes(a.batch_two,cohort),
            "prior archive photographic access differs from frozen inventory")
    release = load(resolve(roots, pins["risk_release"]))
    require(f["released_models"] == release["models"] and len(release["models"]) == 4,
            "release parameters or thresholds changed")
    for cond, profile in f["layout_profiles"].items():
        doc = load(resolve(roots, pins["profile" + cond]))
        require(profile["pin"] == pins["profile" + cond] and doc["tau"] is None and
                doc["status"] == "no_feasible_operating_point", "calibration profile changed")
    recipe = root / "protocol/recovery-gsplat-scale-decay-v1.json"
    require(sha(recipe) == "57730c706aaaa2f88c7f39ff179ba04ff971f31e599bf3409a3d545c61d72c98"
            and load(recipe)["stop_step"] == 5000, "frozen reconstruction changed")
    auth = load(a.authorization)
    require(auth["study"] == "G0-G6 general closeout v1.4" and auth["ordered_access"]
            and auth["roadmap_sha256"] == config["roadmap_sha256"] and bool(auth["user_task_utc"]),
            "missing current authority")
    save(a.out / "entry.json", {
        "schema":"el-general-t1-entry-1", "run_id":os.environ["EL_RUN_ID"], "utc":utc(),
        "study":"general-1","scientific_base_commit":BASE_COMMIT,"T1_objects":OBJECTS,
        "freeze":pins["freeze"], "freeze_sha256":FINAL_HASH,"sources":pins,
        "previous_access": {"T1_photo_receipts":0,"inventoried_acquisitions":len(prior_access["rows"])},
        "user_authorization":auth,"source_authorization":"S0_only","S0":[24,22],
        "selection_before_candidate_access":5,"gate_includes_fixed_extra_cameras":[17,23],
        "target_reference_permission":"closed","joint_reference_openings_permitted_after_terminals":1,
        "required_condition_terminals":42,"required_fixed_score_terminals":288,
        "selected_descriptive_risk_scores":0,"new_risk_training":0,"threshold_refits":0,
        "physical_camera_contract":f["endpoints"]["camera_contract"],
        "frozen_model_identities":release["models"], "scientific_files":files,
        "feature_columns":{"B-L1-transfer":26,"MS-L1":39,"MS-frozen":39,"MS-ft":39},
        "recipe_sha256":sha(recipe),"layout_profiles":f["layout_profiles"],
        "release":"v0.3.0-rc.1","historical_index":config["historical_index"],
        "volume_free_bytes":shutil.disk_usage(a.store).free,
    })


def main(a):
    if a.action == "worker":
        return verify_entry(a)
    old = a.history.resolve()
    sources = {"batch_two": a.batch_two.resolve(), "batch_one":a.batch_one.resolve(),
               "history":old, "cold_shelf":old.parent / "cold-shelf"}
    counts = index(a.store, sources)
    h = a.store / "history"
    fp = bind(a.batch_two, FINAL_RUN, "payload/t1-freeze.json")
    freeze = load(resolve((h,),fp))
    pins = {"freeze":fp, "prior_access":bind(a.batch_two,FINAL_RUN,"payload/acquisition-audit.json"),
            **{k:freeze["sources"][k] for k in ("prospective","cohort","risk_release","reconstruction",
                                                 "selector","profileM2","profileM4")}}
    return stage(a.store,h,"g0-entry","archive_verify",pins,
                 {"role":"independent G0 authority and immutable identities","historical_index":counts,
                  "roadmap_sha256":load(a.authorization)["roadmap_sha256"]},
                 ["{python}","{source}/scripts/g0_entry.py","worker","--config","{config}",
                  "--store",str(a.store),"--batch-two",str(a.batch_two),"--authorization","{input:authorization}",
                  "--out","{out}"], "payload/entry.json", 38000, {"authorization":a.authorization})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("seal","worker"))
    for key in ("store","batch-one","batch-two","history","config","authorization","out"):
        p.add_argument("--"+key,type=Path)
    print(json.dumps(main(p.parse_args()), sort_keys=True))
