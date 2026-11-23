"""Authenticate a four-method operating decision before admitting the held-out cohort."""

import hashlib
import math
from pathlib import Path
from statistics import mean

from ._bundle import HASH, load
from ._verify import parent_directory, require
from .formal_inputs import PinnedRun
from evidence_lens_vision.contract import json_bytes


# The sole interrupted optimizer admitted by the registered four-fit comparison.
INTERRUPTED = {
    "run_id": "20261123T163117676676Z-81c14259dee84abf",
    "manifest_sha256": "2f811f866d6581f846484f089ddbeb9c36239499e213af0fff338243c149e536",
}
TERMINAL_WEIGHTS = "00be5de5abd30bf3e7eb1686e458a96e3ca55dffd6b892af862c98913c85c0b6"
TERMINAL_STATE = "8df402a3d8170c39bc9b1decd531f7dd5e2ff7e3a13a5fcb90030505a808bae9"


def validate_fit_origin(run, store, shelf):
    """Attribute optimizer updates to a sealed training run, including the documented export recovery."""
    require(run.run["execution_status"] == "succeeded", "The deployable producer did not succeed")
    kind = (run.run["run_kind"], run.run["experiment_id"])
    require(kind in {("model_fit", "stage3-risk-neural-fit"), ("derivation", "stage3-terminal-export-recovery")},
            "Unrecognized multisource fit producer")
    fit = load(run.open("payload/training.json"))
    proof = load(run.open("payload/independent-proof/verified-updates.json"))
    steps = fit["optimizer_steps"]
    require(fit["schema"] == "el-multisource-risk-training-1" and fit["phase"] == proof["phase"] == "formal"
            and fit["run_id"] == proof["run_id"] == run.pin["run_id"] and type(steps) is int
            and steps == proof["optimizer_steps"] and steps > 0
            and fit["weights_sha256"] == proof["weights_sha256"] == run.files["payload/package/weights.safetensors"]["sha256"],
            "The training evidence and deployable parameters differ")
    if kind[0] == "model_fit":
        return {"kind": "native_training", "optimizer_run": run.pin, "export_run": run.pin,
                "optimizer_steps": steps, "optimizer_steps_in_this_record": steps}

    def pinned(pin):
        parent = parent_directory(Path(store).resolve(), pin, Path(shelf).resolve() if shelf else None)
        return PinnedRun(parent.parent.parent, pin)

    recovery = load(run.open("payload/recovery.json"))
    require(recovery["schema"] == "el-stage3-terminal-export-recovery-1" and recovery["run_id"] == fit["run_id"]
            and fit["recovery_sha256"] == run.files["payload/recovery.json"]["sha256"]
            and recovery["optimization_parent"] == fit["optimization_parent"] == INTERRUPTED
            and (fit["method"], fit["seed"]) == ("MS-frozen", 1729)
            and fit["optimizer_steps_in_this_record"] == recovery["optimizer_steps_in_this_record"] == 0
            and steps == recovery["optimizer_steps_on_original"] == 288
            and fit["weights_sha256"] == recovery["original_weights_sha256"] == TERMINAL_WEIGHTS
            and recovery["terminal_checkpoint_sha256"] == TERMINAL_STATE, "Incorrect terminal optimizer recovery")
    receipt = recovery["terminal_readback"]
    cpin = {k: receipt[k] for k in ("run_id", "manifest_sha256")}
    require(cpin in load(run.open("inputs.json"))["parents"] and receipt["member"] == "payload/terminal-optimization.json",
            "Missing direct readback of the interrupted optimizer")
    certificate = pinned(cpin)
    doc = load(certificate.open(receipt["member"]))
    require(certificate.files[receipt["member"]]["sha256"] == receipt["sha256"]
            and certificate.run["execution_status"] == "succeeded" and certificate.run["run_kind"] == "archive_verify"
            and certificate.run["experiment_id"] == "stage3-terminal-optimization-readback"
            and doc["schema"] == "el-stage3-terminal-optimization-readback-1" and doc["run_id"] == cpin["run_id"]
            and doc["optimization_parent"] == INTERRUPTED and doc["optimizer_steps"] == steps
            and doc["original_weights_sha256"] == TERMINAL_WEIGHTS and doc["terminal_checkpoint_sha256"] == TERMINAL_STATE
            and doc["validated_tensors"] == recovery["verified_terminal_tensors"] == 181,
            "The terminal readback does not certify the selected checkpoint")
    original = pinned(INTERRUPTED)
    require(original.run["execution_status"] == "failed" and original.run["run_kind"] == "model_fit"
            and original.run["experiment_id"] == "stage3-risk-neural-fit", "The interrupted optimization seal differs")
    setup = load(original.open("payload/fit/setup.json"))
    before = setup["identity"]
    after = load(run.open("payload/optimization-original/corrected-export-config.json"))
    scientific = {k: v for k, v in before.items() if k not in ("model_id", "config_sha256")}
    require(set(scientific) == set(after)
            and {k for k, v in scientific.items() if v != after[k]} == {"code_files", "worker_zip_sha256"}
            and set(before["code_files"]) == set(after["code_files"])
            and {n for n in before["code_files"] if before["code_files"][n] != after["code_files"][n]} ==
                {"src/evidence_lens_multirisk/deploy.py"}
            and doc["source_equivalence"]["original_optimizer_worker_sha256"] == before["worker_zip_sha256"]
            and doc["source_equivalence"]["corrected_export_worker_sha256"] == after["worker_zip_sha256"]
            and all(fit[k] == before[k] == after[k] for k in ("training_dataset_sha256", "development_dataset_sha256")),
            "Original optimization and corrected export used different scientific inputs")
    def design_id(identity):
        keys = ("model_id", "method", "seed", "config_sha256")
        return hashlib.sha256(json_bytes({k:v for k,v in identity.items() if k not in keys})).hexdigest()
    require(design_id(before) == setup["paired_design_sha256"] == recovery["original_paired_design_sha256"]
            and design_id(after) == fit["paired_design_sha256"] == recovery["corrected_worker_paired_design_sha256"],
            "Original and corrected worker design identities cannot be reproduced")
    selected_scores = load(run.open("payload/development-scores/scores.json"))
    m2 = [row for row in selected_scores["views"] if row["condition"] == "M2"]
    require(selected_scores["dataset_sha256"] == before["development_dataset_sha256"] and len(m2) == 24
            and all(row["modes"]["two"]["file"] == row["modes"]["all"]["file"]
                    and row["modes"]["all"]["alias_of"] == "two" for row in m2),
            "Recovered two-source alias differs from its original Windows map")
    copies = {n: n for n in ("curves.json", "initialization.json", "gradient-witnesses.json", "parameter-changes.json",
                              "initial-head.safetensors", "development.json", "training-state.pt", "package/weights.safetensors")}
    copies.update({"setup.json": "optimization-original/setup.json", "package/model.json": "optimization-original/broken-model.json"})
    prefix = "payload/fit/development-scores/"
    maps = [n.removeprefix("payload/fit/") for n in original.files if n.startswith(prefix) and "/all/" in n and n.endswith(".npz")]
    require(len(maps) == recovery["original_all_scope_maps"] == 48, "The original development maps are incomplete")
    copies.update({n: n for n in maps})
    for before, after in copies.items():
        a, b = "payload/fit/" + before, "payload/" + after
        original.open(a)
        run.open(b)
        require(original.files[a]["sha256"] == run.files[b]["sha256"], "An original optimizer product changed: " + before)
    require(original.files["payload/fit/training-state.pt"]["sha256"] == TERMINAL_STATE,
            "The archived checkpoint differs from the certified checkpoint")
    return {"kind": "terminal_export_recovery", "optimizer_run": INTERRUPTED.copy(), "export_run": run.pin,
            "optimizer_steps": steps, "optimizer_steps_in_this_record": 0, "terminal_readback": cpin}


def check_selection(method, study, selected, cutoffs, light):
    components = method["components"]
    dev = components["development_dataset"]["sha256"]
    models = method["models"]
    seeds = list(map(str, study["neural"]["candidate_seeds"]))
    release = str(study["neural"]["release_seed"])
    require(selected["schema"] == "el-stage3-neural-selection-1"
            and selected["run_id"] == components["neural_selection"]["run_id"]
            and selected["cohort"] == "development" and selected["study_sha256"] == method["study_sha256"]
            and selected["catalog_sha256"] == method["catalog_sha256"]
            and selected["development_dataset_sha256"] == dev
            and set(selected["methods"]) == set(selected["models"]) == set(models), "Incorrect development selection source")
    require(cutoffs["schema"] == "el-stage3-threshold-recalculation-1"
            and cutoffs["dataset_sha256"] == dev and cutoffs["catalog_sha256"] == method["catalog_sha256"]
            and set(cutoffs["methods"]) == set(models) and set(cutoffs["all_seed_models"]) == set(seeds)
            and selected["files"]["thresholds.json"]["sha256"] == components["thresholds"]["sha256"],
            "Operating points are detached from the selected development study")
    require(light["schema"] == "el-stage3-light-development-1" and light["development_dataset_sha256"] == dev
            and light["model_index_sha256"] == models["MS-L1"]["model_index_sha256"]
            and light["run_id"] == models["MS-L1"]["model_id"]
            and light["weights_sha256"] == models["MS-L1"]["weights_sha256"]
            and light["threshold"]["tau"] == method["thresholds"]["MS-L1"],
            "Lightweight comparison differs from its real development evaluation")
    for name, frozen in models.items():
        chosen = selected["models"][name]
        require(chosen == cutoffs["all_seed_models"][release][name]
                and all(chosen[k] == frozen[k] for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold"))
                and cutoffs["methods"][name]["tau"] == frozen["threshold"],
                "Primary model differs from the real score producer or threshold calculation")
    for seed in seeds:
        for name in ("B-L1-transfer", "MS-L1"):
            require(cutoffs["all_seed_models"][seed][name] == selected["models"][name],
                    "Lightweight and transfer models changed across seeds")
    rule = study["promotion"]
    rows = selected["summaries"]
    require(set(rows) == set(seeds) and all(set(rows[s]) == set(models) for s in seeds)
            and rule["paired_seeds"] == study["neural"]["candidate_seeds"], "Incomplete paired development roster")
    tests = {}
    for opponent in rule["competitors"]:
        gaps = []
        for seed in seeds:
            ft, base = rows[seed]["MS-ft"], rows[seed][opponent]
            if ft["complete"] and base["complete"] and all(r["ap"] is not None and math.isfinite(r["ap"]) for r in (ft, base)):
                gaps.append(ft["ap"] - base["ap"])
        tests[opponent + "_AP"] = (len(gaps) == len(seeds) and min(gaps) > 0
                                    and mean(gaps) >= rule["minimum_mean_ap_gain_against_each"])
    for seed in seeds:
        ft, base = rows[seed]["MS-ft"], rows[seed]["MS-L1"]
        complete = ft["complete"] and base["complete"] and all(
            r[k] is not None and math.isfinite(r[k]) for r in (ft, base) for k in ("low_coverage", "low_error"))
        tests[seed + "_coverage"] = bool(complete and ft["low_coverage"] >= base["low_coverage"] - rule["maximum_low_coverage_loss_vs_light"])
        tests[seed + "_low_error"] = bool(complete and ft["low_error"] <= base["low_error"] + rule["maximum_low_error_increase_vs_light"])
    decision = selected["decision"]
    expected = rule["candidate"] if all(tests.values()) else rule["otherwise"]
    require(decision["checks"] == tests and decision["release_seed"] == int(release)
            and expected == decision["default"] == method["default_method"], "Selected default violates the registered rule")


def check_neural_fits(method, study, cutoffs, original, store, shelf):
    """Recheck each fit against its recorded model and native gradient verification."""
    dev = method["components"]["development_dataset"]["sha256"]
    train = method["components"]["training_dataset"]["sha256"]
    identities = {}
    roster = {}
    for seed in study["neural"]["candidate_seeds"]:
        for name in study["neural"]["branches"]:
            selected = cutoffs["all_seed_models"][str(seed)][name]
            run = original(selected["model_id"])
            validate_fit_origin(run, store, shelf)
            fit = load(run.open("payload/training.json"))
            proof = load(run.open("payload/independent-proof/verified-updates.json"))
            require(fit["phase"] == proof["phase"] == "formal" and fit["run_id"] == proof["run_id"] == selected["model_id"]
                    and fit["method"] == proof["method"] == name and fit["seed"] == proof["seed"] == seed
                    and fit["schema"] == "el-multisource-risk-training-1" and proof["schema"] == "el-verified-multisource-update-1",
                    "Unrecognized formal paired neural fit")
            for key in ("model_index_sha256", "weights_sha256"):
                require(HASH.fullmatch(selected[key]) and fit[key] == proof[key] == selected[key], "Neural fit identity changed")
            require(run.files["payload/package/model.json"]["sha256"] == fit["model_index_sha256"]
                    and run.files["payload/package/weights.safetensors"]["sha256"] == fit["weights_sha256"]
                    and fit["threshold"] == selected["threshold"] and fit["training_dataset_sha256"] == train
                    and fit["development_dataset_sha256"] == dev and fit["feature_schema_sha256"] == method["feature_schema_sha256"],
                    "A formal fit uses another dataset, checkpoint or operating threshold")
            gradients = proof["nonzero_task_gradient_minibatches"]
            require(0 < fit["optimizer_steps"] == proof["optimizer_steps"]
                    and proof["all_exported_tensor_hashes_match"] is True and proof["verified_tensors"] == 181
                    and proof["checked_original_backbone"] == 175 and gradients and min(gradients.values()) >= 3
                    and proof["source_B_weights_sha256"] == study["stage_b"]["encoder_parent_weights_sha256"]
                    and (proof["backbone_changed_scalars"] > 0 if name == "MS-ft" else proof["backbone_changed_scalars"] == 0),
                    "Task-gradient or pretrained tensor evidence differs")
            pair = (fit["initial_head_sha256"], fit["paired_design_sha256"])
            require(identities.setdefault(seed, pair) == pair and all(HASH.fullmatch(value) for value in pair),
                    "Neural branches do not share their registered initialization and sample design")
            roster[(seed, name)] = (run.pin, selected)
    require(len({v[0]["run_id"] for v in roster.values()}) == 4
            and len({a for a, _ in identities.values()}) == len(identities), "Distinct registered fits or seeds are missing")
    return roster


def check_independent_readbacks(method, study, roster, tensors, cpu, original):
    components = method["components"]

    def artifact(parent, relative):
        local = "payload/" + relative["path"]
        require(parent.files.get(local, {}).get("sha256") == relative["sha256"]
                and parent.files[local]["bytes"] == relative["bytes"], "Independent readback inventory differs")
        return parent.open(local)

    t_run = original(components["tensor_comparison"]["run_id"])
    require(tensors["schema"] == "el-stage3-neural-comparison-1" and tensors["run_id"] == t_run.pin["run_id"]
            and tensors["catalog_sha256"] == method["catalog_sha256"] and tensors["study_sha256"] == method["study_sha256"]
            and tensors["feature_schema_sha256"] == method["feature_schema_sha256"]
            and tensors["B_weights_sha256"] == study["stage_b"]["encoder_parent_weights_sha256"]
            and tensors["primary_seed"] == method["primary_seed"]
            and len(tensors["fits"]) == len(roster), "Invalid four-fit tensor comparison")
    included = set()
    for row in tensors["fits"]:
        key = (row["seed"], row["method"])
        require(key in roster and key not in included, "Unregistered or duplicate independent fit")
        included.add(key)
        pin, model = roster[key]
        parent = original(pin["run_id"])
        proof = row["original_proof"]
        require(all(row[k] == v == proof[k] for k, v in pin.items())
                and all(row[k] == model[k] for k in ("weights_sha256", "model_index_sha256"))
                and proof["member"] == "payload/independent-proof/verified-updates.json"
                and parent.files[proof["member"]]["sha256"] == proof["sha256"],
                "Wrong original for the independent tensor comparison")
        require(load(artifact(t_run, row["independent_readback"])) == load(parent.open(proof["member"])),
                "Independent and original tensor proofs differ")
    c_run = original(components["cpu_validation"]["run_id"])
    require(cpu["schema"] == "el-stage3-neural-cpu-validation-1" and cpu["run_id"] == c_run.pin["run_id"],
            "Invalid native CPU validation")
    pairs = {(n, c) for n in study["neural"]["branches"] for c in method["conditions"]}
    require(len(cpu["cases"]) == len(pairs) and {(r["method"], r["condition"]) for r in cpu["cases"]} == pairs,
            "Native CPU cost evidence has incomplete conditions or methods")
    for row in cpu["cases"]:
        cost = load(artifact(c_run, row["original_cost"]))
        require(cost["schema"] == "el-stage3-native-cost-1" and cost["grid"] == study["resolution"]
                and cost["model_sha256"] == row["model_sha256"] == method["models"][row["method"]]["model_index_sha256"]
                and cost["case_sha256"] == row["case_sha256"] and cost["cold_warm_v01_equal"] is True
                and cost["warm_process"]["sampled_peak_process_tree_rss_bytes"] > 0,
                "Native CPU check uses a different model, example or grid")
        for scope in ("one", "all"):
            require(cost["cold_v01_by_scope"][scope]["wall_started_before_python_seconds"] > 0
                    and cost["warm_views"][scope]["timed_native_views"] >= len(study["targets"])
                    and cost["warm_views"][scope]["median_seconds"] > 0,
                    "Native CPU cost missing recorded cold or warm measurements")
