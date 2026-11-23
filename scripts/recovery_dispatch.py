"""Issue one reconstruction job from the registered source-only bundle and an admitted recipe."""

import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from recovery.contract import HORIZON, SOURCES, cell_manifest, validate_recipe


def run_ref(pin):
    return {k: pin[k] for k in ("run_id", "manifest_sha256")}


def dispatch(cfg, paths, out):
    pins = cfg["inputs"]
    role = cfg["object_id"]
    condition = cfg["condition"]
    required = {"registration", "handoff", "initialization", "admission", "recipe", "worker_source"}
    if role == "set1":
        required.add("control_review")
    elif role == "set6":
        required.add("freeze")
    else:
        require(role == "scene_057_offset_bell_cage" and condition == "M4", "unknown positive control")
    require(set(paths) == required and condition in SOURCES, "incorrect scientific evidence roster")
    hand, init, reg, admit = (load(paths[k]) for k in ("handoff", "initialization", "registration", "admission"))
    require(hand["registered_study_sha256"] == reg["protocol_sha256"]
            and hand["registration"] == run_ref(pins["registration"])
            and hand["init_pin"] == run_ref(pins["initialization"])
            and hand["run_id"] == pins["handoff"]["run_id"]
            and all(d["object_id"] == role and d["condition"] == condition for d in (hand, init))
            and hand["parameter_sha256"] == init["parameter_sha256"]
            and hand["generator_source_ids"] == list(SOURCES[condition]), "incorrect source handoff or frame")
    require(admit["schema"] == "el-recovery-recipe-admission-1" and admit["decision"] == "admit"
            and admit["registration"] == pins["registration"]
            and admit["recipe_sha256"] == sha(paths["recipe"])
            and run_ref(pins["recipe"]) == run_ref(pins["admission"])
            and admit["inputs"]["worker_source"] == pins["worker_source"], "recipe or executable differs from admission")
    validate_recipe(load(paths["recipe"]))
    stop = HORIZON
    if role == "set1":
        control = load(paths["control_review"])
        require(control["schema"] == "el-recovery-control-review-1"
                and control["decision"] == "proceed" and control["registration"] == pins["registration"]
                and control["recipe_sha256"] == admit["recipe_sha256"], "missing synthetic-control review")
    if role == "set6":
        freeze = load(paths["freeze"])
        require(hand["regression_recipe_freeze"] == pins["freeze"]
                and freeze["schema"] == "el-recovery-final-freeze-1" and freeze["registration"] == pins["registration"]
                and freeze["admission"] == pins["admission"] and freeze["worker_source"] == pins["worker_source"]
                and freeze["recipe_sha256"] == admit["recipe_sha256"]
                and freeze["recipe_id"] == admit["recipe_id"] and condition in freeze["authorized_new_conditions"][role]
                and freeze["schedule_steps"] == HORIZON and freeze["stop_step"] in (500, 1500, 3000, 5000),
                "set6 requires the registered common stopping rule and a post-freeze handoff")
        stop = freeze["stop_step"]
    else:
        require(hand["regression_recipe_freeze"] is None, "non-regression source handoff has a freeze")
    with zipfile.ZipFile(paths["worker_source"]) as z:
        source = json.loads(z.read("SOURCE-MANIFEST.json"))
    parent_refs = {p["run_id"]: run_ref(p) for p in pins.values()}
    parents = {n: pins["initialization" if n in ("init.json", "init.npz") else "handoff"]["run_id"]
               for n in hand["files"]}
    parents["recipe"] = pins["recipe"]["run_id"]
    job = {"schema": "el-recovery-cell-1", "purpose": "formal", "object_id": role, "condition": condition,
           "experiment_id": cfg["experiment_id"], "trial_id": cfg["scientific_trial_id"],
           "stop_step": stop, "parents": [parent_refs[r] for r in sorted(parent_refs)],
           "recipe_sha256": admit["recipe_sha256"], "files": hand["files"], "input_parents": parents}
    with TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / "job.json"
        save(path, job)
        require(cell_manifest(path.parent, sha(path)) == job, "producer and recipient job contracts differ")
        path.rename(out / "job.json")
    save(out / "dispatch.json", {"schema": "el-recovery-dispatch-1", "run_id": os.environ["EL_RUN_ID"],
         "job_sha256": sha(out / "job.json"), "scientific_trial_id": cfg["scientific_trial_id"],
         "object_id": role, "condition": condition, "schedule_steps": HORIZON, "stop_step": stop,
         "worker_tree_sha256": source["tree_sha256"], "inputs": pins, "risk_scores_read": 0})
