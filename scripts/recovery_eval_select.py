"""Select a shared stopping step from complete blind-to-risk paired set1 quality trajectories."""

import argparse
import json
import math
import os
from pathlib import Path
from dtu_score import VIEWS
from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import new_output, write_json
from recovery_eval_io import authority, original, STEPS
from vision_sources import signed

CANDIDATE_STEPS = (500, 1500, 3000, 5000)


def target_mean(point):
    targets = point.get("targets", {})
    if point.get("status") != "measured" or set(targets) != set(VIEWS):
        return None, "missing_comparable_target"
    values = [targets[v].get("full_crop", {}).get("MAE") for v in VIEWS]
    if any(value is None for value in values):
        return None, "missing_comparable_target"
    if not all(type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        return None, "invalid_native_rgb_mae"
    return sum(values) / len(values), None


def selection(trajectories):
    table = []
    seen = set()
    by_recipe = {}
    for row in trajectories:
        require(
            row["schema"] == "el-recovery-quality-1"
            and row["object_id"] == "set1"
            and row["role"] == "development_selection"
            and row["condition"] in ("M2", "M4")
            and row["risk_inputs_read"] == 0,
            "selection accepts only the registered set1 quality curves",
        )
        key = (row["recipe_id"], row["condition"])
        require(key not in seen, "duplicate development trajectory")
        seen.add(key)
        by_recipe.setdefault(row["recipe_id"], {})[row["condition"]] = row
    require(1 <= len(by_recipe) <= 2, "registered upper limit of two candidate recipes")
    for name, conditions in sorted(by_recipe.items()):
        require(set(conditions) == {"M2", "M4"}, "a candidate needs both registered condition dispositions")
        require(
            len({r["recipe_sha256"] for r in conditions.values()}) == 1,
            "paired conditions used different recipes",
        )
        complete = all(
            r["execution_status"] == "completed" and r["completed_steps"] == r["schedule_steps"] == 5000
            for r in conditions.values()
        )
        for step in CANDIDATE_STEPS:
            means = {}
            reasons = {}
            if not complete:
                why = "incomplete_full_development_schedule"
            else:
                for c in ("M2", "M4"):
                    point = conditions[c]["steps"].get(str(step), {})
                    means[c], problem = target_mean(point)
                    if problem:
                        reasons[c] = problem
                why = next(iter(reasons.values()), None)
            table.append(
                {
                    "recipe_id": name,
                    "recipe_sha256": conditions["M2"]["recipe_sha256"],
                    "step": step,
                    "condition_means": means if why is None else None,
                    "macro_target_MAE": (means["M2"] + means["M4"]) / 2 if why is None else None,
                    "reason": why,
                    "condition_exclusions": reasons,
                    "checkpoint_sha256": {
                        c: conditions[c]["steps"].get(str(step), {}).get("checkpoint_sha256")
                        for c in ("M2", "M4")
                    },
                    "phase": {
                        c: conditions[c]["steps"].get(str(step), {}).get("phase") for c in ("M2", "M4")
                    },
                }
            )
    require(
        {(r["recipe_id"], r["step"]) for r in table}
        == {(name, step) for name in by_recipe for step in CANDIDATE_STEPS}
        and len(table) == len(by_recipe) * len(CANDIDATE_STEPS)
        and all((r["macro_target_MAE"] is None) == (r["reason"] is not None) for r in table),
        "all four registered checkpoints need one eligible or excluded disposition per recipe",
    )
    ready = [r for r in table if r["reason"] is None]
    winner = min(ready, key=lambda r: (r["macro_target_MAE"], r["step"], r["recipe_id"])) if ready else None
    return table, winner


def work(roots, req, out, rid):
    require(req["schema"] == "el-recovery-selection-request-1", "wrong development selection")
    reg = authority(roots, req)
    require(
        tuple(reg["selection"]["eligible_checkpoints"]) == CANDIDATE_STEPS == tuple(STEPS[1:]),
        "candidate stopping points differ from the archived registration",
    )
    curves = []
    for spec in req["quality"]:
        _, _, q = original(roots, spec)
        require(
            q["protocol_sha256"] == reg["protocol_sha256"] and q["input_pins"]["protocol"] == req["protocol"],
            "curve belongs to different registration",
        )
        curves.append(q)
    rows, winner = selection(curves)
    d = {
        "schema": "el-recovery-quality-selection-1",
        "run_id": rid,
        "status": "selected" if winner else "no_complete_recipe",
        "registered_selection": reg["selection"],
        "protocol": req["protocol"],
        "quality": req["quality"],
        "selection_metric": "equal mean of the six set1 target full-crop native FP32 RGB MAEs per condition, then equal mean of M2 and M4",
        "eligible_candidates": [r for r in rows if r["reason"] is None],
        "recipe_id": winner["recipe_id"] if winner else None,
        "recipe_sha256": winner["recipe_sha256"] if winner else None,
        "stop_step": winner["step"] if winner else None,
        "schedule_steps": 5000,
        "candidates": rows,
        "winner": winner,
        "risk_read": False,
        "synthetic_control_used_for_ranking": False,
        "set6_used_for_ranking": False,
    }
    with new_output(out) as dest:
        write_json(dest / "selection.json", d)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("store", "shelf", "request", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            work(
                (a.store, a.shelf), signed(a.request, a.request_sha256), a.output, os.environ.get("EL_RUN_ID")
            )["winner"]
        )
    )
