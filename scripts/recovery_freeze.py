"""Bind the independently selected development endpoint to a target-free stopping rule."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from evidence_lens._bundle import load, save, sha, utc
from evidence_lens._verify import require


def freeze(source, config, files, out):
    expected = {"registration", "selection", "control_review", "recipe", "admission", "worker_source"}
    require(set(files) in (expected, expected | {"other_admission"}), "incomplete reconstruction freeze roster")
    registration, selection, control, recipe, admission = (
        load(files[k]) for k in ("registration", "selection", "control_review", "recipe", "admission"))
    require(all(sha(path) == config["inputs"][role]["sha256"] for role, path in files.items()),
            "a decision input differs from its recorded digest")
    require(registration["schema"] == "el-recovery-registration-1" and
            selection["schema"] == "el-recovery-quality-selection-1" and selection["status"] == "selected",
            "a recorded registration and completed independent development selection are required")
    require(selection["run_id"] == config["inputs"]["selection"]["run_id"] and
            registration["run_id"] == config["inputs"]["registration"]["run_id"], "decision run IDs differ")
    require(selection["protocol"] == config["inputs"]["registration"] and
            selection["risk_read"] is False and selection["set6_used_for_ranking"] is False and
            selection["synthetic_control_used_for_ranking"] is False,
            "selection used another registration or an ineligible source of evidence")
    study = load(source / "protocol/reconstruction-recovery-v1.json")
    for name, digest in registration["registered_files"].items():
        require(sha(source / name) == digest, f"registered rule changed before freeze: {name}")
    for name, digest in registration["historical_source_sha256"].items():
        require(sha(source / name) == digest, f"historical implementation changed: {name}")
    pins = config["implementation_sha256"]
    require(bool(pins), "missing selected reconstruction implementation")
    for name, digest in pins.items():
        p = Path(name)
        require(not p.is_absolute() and ".." not in p.parts and sha(source / p) == digest,
                f"selected implementation changed: {name}")
    chosen = {field: selection[field] for field in ("recipe_id", "recipe_sha256", "stop_step", "schedule_steps")}
    require(chosen["recipe_sha256"] == sha(files["recipe"]) and chosen["schedule_steps"] == 5000 and
            chosen["stop_step"] in study["R2"]["selection"]["eligible_checkpoints"],
            "selection is outside registered recipe, schedule or stopping points")
    require(admission["schema"] == "el-recovery-recipe-admission-1" and admission["decision"] == "admit"
            and admission["registration"] == config["inputs"]["registration"] and admission["recipe_id"] == chosen["recipe_id"]
            and admission["recipe_sha256"] == chosen["recipe_sha256"]
            and admission["inputs"]["worker_source"] == config["inputs"]["worker_source"]
            and all(config["inputs"]["admission"][k] == config["inputs"]["recipe"][k]
                    for k in ("run_id", "manifest_sha256")), "selected recipe or worker differs from its admission")
    with zipfile.ZipFile(files["worker_source"]) as z:
        src = json.loads(z.read("SOURCE-MANIFEST.json"))
    for name, row in src["files"].items():
        if name == "scripts/recovery_windows.py" or name.startswith("src/recovery/"):
            require(sha(source / name) == row["sha256"], "selected Windows executable differs: " + name)
    admitted = [load(files[k]) for k in ("admission", "other_admission") if k in files]
    eligible_steps = study["R2"]["selection"]["eligible_checkpoints"]
    expected_rows = sorted((a["recipe_id"], a["recipe_sha256"], step) for a in admitted for step in eligible_steps)
    measured_rows = sorted((r["recipe_id"], r["recipe_sha256"], r["step"]) for r in selection["candidates"])
    require(measured_rows == expected_rows, "ranking omits or duplicates a registered candidate or checkpoint")
    winner = min(selection["eligible_candidates"], key=lambda r: (r["macro_target_MAE"], r["step"], r["recipe_id"]))
    require(winner == selection["winner"] and winner["recipe_id"] == chosen["recipe_id"] and
            winner["step"] == chosen["stop_step"] and set(winner["checkpoint_sha256"]) == {"M2", "M4"},
            "selected row or condition checkpoint identities differ from the recorded ranking")
    require(control["schema"] == "el-recovery-control-review-1" and control["recipe_sha256"] == chosen["recipe_sha256"]
            and control["object_role"] == "synthetic-positive-control" and control["condition"] == "M4"
            and control["decision"] == "proceed" and control["quality_evidence"]
            and control["interpretation"], "selected recipe lacks a reviewed positive control")
    save(out / "freeze.json", {
        "schema": "el-recovery-final-freeze-1", "run_id": os.environ["EL_RUN_ID"], "utc": utc(),
        **chosen, "registered_selection": config["inputs"]["selection"], "selection": selection,
        "registration": config["inputs"]["registration"], "protocol_sha256": registration["protocol_sha256"],
        "control_review": config["inputs"]["control_review"], "selected_recipe": recipe,
        "admission": config["inputs"]["admission"], "worker_source": config["inputs"]["worker_source"],
        "worker_tree_sha256": src["tree_sha256"],
        "admitted_candidates": {a["recipe_id"]: a["recipe_sha256"] for a in admitted},
        "implementation_sha256": pins, "release": registration["release"], "models": registration["models"],
        "scopes": registration["scopes"], "risk_fits": 0, "risk_recalibrations": 0,
        "authorized_new_conditions": {"set6": ["M2", "M4"]},
        "stop_rule": "Run the selected recipe on its complete original schedule until exactly stop_step. Apply the registered resource abort first. An aborted fit has no replacement endpoint.",
        "risk_order": "The reconstruction search is closed. Frozen scoring can read renders produced from the selected parameters, independently of all target photographs.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "config", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--binding", nargs=2, action="append", required=True)
    args = parser.parse_args()
    freeze(args.source, load(args.config), {k: Path(v) for k, v in args.binding}, args.out)
