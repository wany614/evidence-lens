"""Seal a recipe admission, control interpretation or final stopping rule using the study recorder."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from evidence_lens._bundle import HASH, LABEL, encoded, load, save, sha, utc
from evidence_lens._verify import require, verify_bundle

from dtu_record_gate import verified_member
from make_handoff import publish
from recovery_freeze import freeze


def decide(source, cfg, paths, out):
    action = cfg["decision_type"]
    reg = load(paths["registration"])
    require(reg["schema"] == "el-recovery-registration-1" and
            reg["protocol_sha256"] == sha(source / "protocol/reconstruction-recovery-v1.json"),
            "wrong registration")
    for role, ref in cfg["inputs"].items():
        require(sha(paths[role]) == ref["sha256"], f"changed input: {role}")
    for name, digest in cfg["implementation_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and sha(source / name) == digest,
                f"implementation differs from its configuration: {name}")
    if action == "freeze":
        freeze(source, cfg, paths, out)
        return
    if action == "dispatch":
        from recovery_dispatch import dispatch
        dispatch(cfg, paths, out)
        return
    note = cfg["interpretation"]
    require(isinstance(note, str) and note.strip(), "record the evidence and interpretation")
    decision = {
        "run_id": os.environ["EL_RUN_ID"], "utc": utc(), "registration": cfg["inputs"]["registration"],
        "inputs": cfg["inputs"], "implementation_sha256": cfg["implementation_sha256"],
        "interpretation": note, "decision": cfg["decision"], "risk_results_used": False,
    }
    if action == "initialization":
        require(set(paths) == {"registration", "r0", "init_M2", "init_M4"}, "wrong R1 evidence scope")
        diagnostics = load(paths["r0"])
        require(diagnostics.get("run_id") == cfg["inputs"]["r0"]["run_id"] and cfg["decision"] == "skip_R1",
                "wrong checkpoint diagnostic or initialization decision")
        observed = {}
        for cond in ("M2", "M4"):
            item = load(paths["init_" + cond])
            require(item["object_id"] == "set1" and item["condition"] == cond and
                    item["actual_method"] == "known_pose_sift_triangulation" and item["eligibility"]["qualified"],
                    "random-box ablation cannot be skipped on these source-geometry grounds")
            observed[cond] = {"points": item["N"], "source_ids": item["generator_source_ids"],
                              "scene_scale_m": item["scene_scale_m"], "parameter_sha256": item["parameter_sha256"]}
        require(observed["M2"]["scene_scale_m"] == observed["M4"]["scene_scale_m"],
                "two conditions used different metric scale")
        decision.update(schema="el-recovery-r1-decision-1", actual_initializers=observed, new_R1_fits=0)
        save(out / "r1-decision.json", decision)
    elif action == "recipe":
        require({"registration", "r0", "resource_probe", "init_M2", "init_M4", "worker_source"} <= paths.keys(),
                "missing historical diagnostic, resource preflight or source qualification")
        require(cfg["decision"] == "admit" and LABEL.fullmatch(cfg["recipe_id"]), "invalid recipe admission")
        name = Path(cfg["recipe_member"])
        require(not name.is_absolute() and name.parts[0] == "protocol" and ".." not in name.parts,
                "invalid public recipe name")
        recipe_file = source / name
        require(sha(recipe_file) == cfg["recipe_sha256"], "recipe file differs from the request")
        probe = load(paths["resource_probe"])
        recipe = load(recipe_file)
        require(probe["schema"] == "el-recovery-resource-probe-1" and probe["passed"] is True
                and probe["worker_zip_sha256"] == cfg["inputs"]["worker_source"]["sha256"]
                and probe["density"]["cap"] >= recipe["topology"]["cap"],
                "the short resource and interface tests did not qualify this worker and budget")
        if probe["candidate_recipe_sha256"] != cfg["recipe_sha256"]:
            require({"prior_recipe", "prior_admission", "capacity_confirmation", "quality_M2", "quality_M4",
                     "extent_057", "extent_set1", "prior_control"} <= paths.keys(),
                    "a revised schedule needs an original candidate, measured bottleneck and fresh capacity confirmation")
            base = load(paths["prior_recipe"])
            prior = load(paths["prior_admission"])
            compared = copy.deepcopy(recipe)
            compared["learning_rates"]["log_scales"][1] = base["learning_rates"]["log_scales"][1]
            require(base == compared and 0 < recipe["learning_rates"]["log_scales"][1] < base["learning_rates"]["log_scales"][1]
                    and sha(paths["prior_recipe"]) == prior["recipe_sha256"] == probe["candidate_recipe_sha256"]
                    and prior["inputs"]["worker_source"] == cfg["inputs"]["worker_source"],
                    "only a lower terminal log-scale rate can inherit this renderer and capacity control")
            actual = load(paths["capacity_confirmation"])
            require(actual["passed"] is True and actual["nominal_resolution"] == [256, 256]
                    and max(r["after"] for r in actual["steps"]) == recipe["topology"]["cap"]
                    and actual["runtime"]["torch_reserved_peak_bytes"] > 0, "missing revised-recipe 20K measurement")
            decision["resource_basis"] = {"unchanged_renderer_and_resume": cfg["inputs"]["resource_probe"],
                                          "fresh_capacity": cfg["inputs"]["capacity_confirmation"],
                                          "original_log_scale_rate": base["learning_rates"]["log_scales"],
                                          "revised_log_scale_rate": recipe["learning_rates"]["log_scales"]}
        (out / "recipe.json").write_bytes(recipe_file.read_bytes())
        decision.update(schema="el-recovery-recipe-admission-1", recipe_id=cfg["recipe_id"],
                        recipe_sha256=sha(recipe_file), R1=cfg["R1"], reason_for_candidate=cfg["candidate_reason"])
        save(out / "admission.json", decision)
    elif action == "control":
        require({"registration", "recipe", "quality", "fit"} <= paths.keys(), "missing control evidence")
        quality = load(paths["quality"])
        fit = load(paths["fit"])
        require(quality["condition"] == "M4" and quality["object_id"] == "scene_057_offset_bell_cage" and
                quality["recipe_sha256"] == sha(paths["recipe"]) and quality["risk_inputs_read"] == 0 and
                quality["input_pins"]["protocol"] == cfg["inputs"]["registration"] and
                quality["input_pins"]["recipe"] == cfg["inputs"]["recipe"] and
                quality["execution_status"] == fit["status"] == "completed" and
                quality["completed_steps"] == fit["last_completed_step"] == 5000 and
                fit["identity"]["recipe_sha256"] == quality["recipe_sha256"] and
                fit["identity"]["object_id"] == quality["object_id"] and fit["identity"]["condition"] == "M4" and
                fit["checkpoints"]["5000"]["checkpoint_sha256"] == quality["steps"]["5000"]["checkpoint_sha256"],
                "control result is outside this recipe")
        decision.update(schema="el-recovery-control-review-1", recipe_sha256=sha(paths["recipe"]),
                        object_role="synthetic-positive-control", condition="M4",
                        quality_evidence=cfg["inputs"]["quality"])
        save(out / "control-review.json", decision)
    else:
        raise ValueError("unknown recovery decision")


def record_once(store, project, request):
    cfg = load(request)
    trial = cfg["trial_id"]
    require(trial.startswith("r-") and "/" not in trial and "\\" not in trial, "invalid decision identity")
    roles = {role: verified_member(store, pin) for role, pin in cfg["inputs"].items()}
    if cfg["decision_type"] == "recipe" and "capacity_confirmation" in roles:
        proof = cfg["inputs"]["capacity_confirmation"]
        root = store / "runs" / proof["run_id"]
        inputs = load(root / "inputs.json")["inputs"]
        probe = load(roles["resource_probe"])
        require(proof["member"] == "payload/forced-topology.json" and
                load(root / "code_snapshot.json")["snapshot_sha256"] == probe["actual_worker_snapshots"]["topology"]
                and any(r["role"] in {"recipe", "candidate"} and len(r["files"]) == 1 and
                        r["files"][0]["sha256"] == cfg["recipe_sha256"] for r in inputs),
                "capacity confirmation used another executable or optimizer recipe")
    parents = {p["run_id"]: p["manifest_sha256"] for p in cfg["inputs"].values()}
    args = [sys.executable, "-m", "evidence_lens.recorder", "--source-root", str(project),
            "--root", str(store), "--kind", "archive_verify", "--config", str(request.resolve())]
    for rid, digest in sorted(parents.items()):
        args.extend(["--parent", rid + "=" + digest])
    worker = ["{python}", "{source}/scripts/recovery_decide.py", "worker",
              "--source", "{source}", "--config", "{config}", "--out", "{out}"]
    for role, path in sorted(roles.items()):
        args.extend(["--input", role + "=" + str(path), "--input-parent", role + "=" + cfg["inputs"][role]["run_id"]])
        worker.extend(["--binding", role, "{input:" + role + "}"])
    proc = subprocess.run([*args, "--", *worker], cwd=project, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError((proc.stdout + proc.stderr)[-1800:])
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    stage = store / "runs" / result["run_id"]
    proof = verify_bundle(stage)
    output_name = {"initialization": "r1-decision.json", "recipe": "admission.json", "dispatch": "job.json",
                   "control": "control-review.json", "freeze": "freeze.json"}[cfg["decision_type"]]
    return {"run_id": result["run_id"], "manifest_sha256": proof["manifest_sha256"],
            "member": "payload/" + output_name, "sha256": sha(stage / "payload" / output_name)}


def record(store, project, request):
    config = load(request)
    action = config["decision_type"]
    require(action in {"initialization", "recipe", "control", "freeze", "dispatch"} and LABEL.fullmatch(config["trial_id"]) and
            HASH.fullmatch(config["inputs"]["registration"]["sha256"]),
            "unknown decision or invalid trial ID")
    directory = store / "recovery" / "decisions" / config["inputs"]["registration"]["sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    key = config["trial_id"]
    if action == "dispatch":
        obj, cond = config["object_id"], config["condition"]
        require(obj in {"set1", "set6", "scene_057_offset_bell_cage"} and cond in {"M2", "M4"}, "unknown job")
        key = "dispatch-" + obj + "-" + cond
        if obj != "set6":
            recipe_hash = config["inputs"]["recipe"]["sha256"]
            require(HASH.fullmatch(recipe_hash), "invalid recipe identity")
            key += "-" + recipe_hash
    if action == "recipe":
        require(LABEL.fullmatch(config["recipe_id"]), "malformed recipe ID")
        key = "admitted-" + config["recipe_id"]
    marker = directory / ("final-freeze.json" if action == "freeze" else key + ".json")
    with (directory / "lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            existing = load(marker)
            require(existing["request_sha256"] == sha(request), "a different study decision is already sealed")
            verified_member(store, existing["pin"])
            return existing["pin"]
        final = directory / "final-freeze.json"
        if action == "dispatch" and config["object_id"] == "set6":
            require(final.is_file() and load(final)["pin"] == config["inputs"]["freeze"],
                    "regression job needs the unique sealed final freeze")
        else:
            require(action == "freeze" or not final.exists(), "the reconstruction search already closed")
        if action == "recipe":
            previous = [load(verified_member(store, load(p)["pin"])) for p in directory.glob("admitted-*.json")]
            require(len(previous) < 2, "registered recipe budget is exhausted")
            require(all(p["recipe_sha256"] != config["recipe_sha256"] or
                        p["inputs"]["worker_source"] != config["inputs"]["worker_source"] for p in previous),
                    "this recipe and executable already have a scientific admission")
        if action == "freeze":
            recorded = {load(p)["pin"]["run_id"] for p in directory.glob("admitted-*.json")}
            supplied = {v["run_id"] for k, v in config["inputs"].items() if k in {"admission", "other_admission"}}
            require(recorded and recorded == supplied, "the freeze must account for every admitted candidate")
        pin = record_once(store, project, request)
        publish(marker, encoded({"request_sha256": sha(request), "pin": pin}))
        return pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    host = sub.add_parser("record")
    host.add_argument("--store", type=Path, required=True)
    host.add_argument("--request", type=Path, required=True)
    host.add_argument("--project", type=Path, default=Path.cwd())
    worker = sub.add_parser("worker")
    for value in ("source", "config", "out"):
        worker.add_argument("--" + value, required=True, type=Path)
    worker.add_argument("--binding", nargs=2, action="append", default=[])
    a = parser.parse_args()
    if a.action == "record":
        print(json.dumps(record(a.store.resolve(), a.project.resolve(), a.request)))
    else:
        decide(a.source, load(a.config), {k: Path(v) for k, v in a.binding}, a.out)
