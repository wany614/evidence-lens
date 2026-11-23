"""Independently compare public CPU deployment against six frozen B0 development views."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from evidence_lens.l1 import KEYS
from evidence_lens.scoring_io import TARGETS, file_hash, new_output, write_json


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_npz(path, digest):
    require(file_hash(path) == digest, f"changed numeric file {path.name}")
    return np.load(path, allow_pickle=False)


def view_parity(view, family, dev, saved, observed, labels, rule, thresholds):
    v, e, y = (labels[k] for k in ("reference_valid", "rgb_evaluable", "rgb_high_error"))
    indices = np.flatnonzero(e)
    require(v.dtype == e.dtype == np.bool_ and y.dtype == np.uint8 and v.shape == e.shape == y.shape
            and not (e & ~v).any() and np.array_equal(saved["index"], indices)
            and np.array_equal(saved["Y"], y.ravel()[indices]), "development coordinates or labels differ from independent P2 evaluation")
    outside = v & ~e
    true_error = y.ravel()[indices] == 1
    methods = {}
    for method, key in KEYS.items():
        if method == "L0":
            reference = np.full(len(indices), np.nan, np.float32) if rule is None else rule["l0"].ravel()[indices]
        else:
            reference = saved[key]
        raw, raw_mask, raw_states = (observed[k] for k in (key, key + "_valid", key + "_state"))
        require(reference.dtype == raw.dtype == np.float32 and raw.shape == raw_mask.shape == raw_states.shape == v.shape
                and raw_mask.dtype == np.bool_ and raw_states.dtype == np.uint8 and reference.shape == (len(indices),), f"invalid {method} maps")
        observed_e, mask_e, state_e = (a.ravel()[indices] for a in (raw, raw_mask, raw_states))
        ref_ok, actual_ok = np.isfinite(reference), np.isfinite(observed_e)
        same_masks = np.array_equal(ref_ok, actual_ok) and np.array_equal(mask_e, ref_ok) and np.array_equal(np.isnan(reference), np.isnan(observed_e))
        states = np.full(len(indices), 255, np.uint8)
        if method == "L0":
            require(np.isin(reference[ref_ok], [0, 1]).all(), "original L0 contains a nonbinary finite score")
            states[ref_ok] = reference[ref_ok].astype(np.uint8)
        else:
            states[ref_ok] = np.where(reference[ref_ok].astype(np.float64) < thresholds[method], 0, 1)
        common = ref_ok & actual_ok
        gap = float(np.abs(reference[common].astype(np.float64) - observed_e[common]).max()) if common.any() else None
        in_l, in_h, in_u = (state_e == value for value in (0, 1, 255))
        counts = {"V":int(v.sum()), "E":len(indices), "H":int(in_h.sum()), "L":int(in_l.sum()), "U":int(in_u.sum()),
                  "Y":int(true_error.sum()), "Y_in_L":int((in_l & true_error).sum()), "Y_in_H":int((in_h & true_error).sum()),
                  "Y_in_U":int((in_u & true_error).sum())}
        if method == "L0":
            # Raw rule decisions may legally warn on failed RGB. Verify them throughout the image without adding them to E.
            orig = np.full(v.shape, np.nan, np.float32) if rule is None else rule["l0"]
            expected_raw_states = np.where(np.isfinite(orig), np.where(orig == 0, 0, 1), 255).astype(np.uint8)
            raw_agrees = (np.array_equal(raw, orig, equal_nan=True) and np.array_equal(raw_mask, np.isfinite(orig))
                          and np.array_equal(raw_states, expected_raw_states))
            if rule is not None:
                raw_agrees = raw_agrees and np.array_equal(observed["l0_reason"], rule["l0_reason"])
            match_counts = True
        else:
            frozen = dev["methods"][method]["views"][family][view]
            match_counts = all(frozen[k] == n for k, n in counts.items())
            raw_agrees = True
        match_states = bool(np.array_equal(states, state_e))
        methods[method] = {**counts, "finite_E":int(ref_ok.sum()), "max_abs_score_difference":gap,
                           "validity_equal":bool(same_masks), "strict_states_equal":match_states, "frozen_counts_equal":bool(match_counts),
                           "raw_rule_equal":bool(raw_agrees), "V_minus_E_raw_low":int((outside & (raw_states == 0)).sum()),
                           "V_minus_E_raw_high":int((outside & (raw_states == 1)).sum()), "V_minus_E_raw_unknown":int((outside & (raw_states == 255)).sum()),
                           "passed": bool(same_masks and match_states and match_counts and raw_agrees and (gap is None or gap <= 1e-6))}
    return {"V":int(v.sum()), "E":len(indices), "V_minus_E":int(outside.sum()), "methods":methods,
            "passed":all(r["passed"] for r in methods.values())}


def compare_files(frozen, deployed, rules, evaluation, family, output, run_id):
    dev = load(frozen / "development.json")
    model = load(frozen / "model/model.json")
    cut = load(frozen / "model/thresholds.json")
    score, baseline, labels = (load(p) for p in (deployed / "l1_score.json", rules / "score.json", evaluation / "evaluation.json"))
    rec = dev["family_sources"][family]
    pin = rec["pin"]
    require(dev["schema"] == "el-development-freeze-1" and dev["status"] == "frozen" and dev["run_id"] == model["model_id"] == score["model_id"]
            and score["schema"] == "el-l1-score-1" and score["methods"] == KEYS and set(score["views"]) == set(TARGETS)
            and dev["model_index_sha256"] == file_hash(frozen / "model/model.json") == score["model_files"]["model_index"]["sha256"]
            and model["files"]["thresholds.json"] == file_hash(frozen / "model/thresholds.json")
            and cut["thresholds"] == score["thresholds"] == {m:d["tau"] for m, d in dev["methods"].items()}, "frozen model/threshold identity differs")
    require(pin["split"] == "dev" and pin["condition_id"] == labels["condition_id"] == "B0" and pin["scene_family_id"] == labels["scene_family_id"] == family
            and labels["run_id"] == pin["evaluation_run_id"] and labels["scoring_run_id"] == baseline["run_id"] == pin["score_run_id"]
            and file_hash(rules / "score.json") == pin["score_receipt_sha256"] and file_hash(evaluation / "evaluation.json") == pin["evaluation_receipt_sha256"]
            and score["producer_run_id"] == labels["producer_run_id"] == baseline["producer_run_id"]
            and score["inputs"] == baseline["inputs"] and rec["source_input_sha256"] == score["inputs"]["input_png"]["sha256"],
            "deployed source or P2 label/score provenance differs from development")
    views = {}
    for view in TARGETS:
        original = baseline["views"][view]
        rule = checked_npz(rules / (view + ".npz"), original["score_sha256"]) if original["status"] == "scored" else None
        try:
            with (checked_npz(frozen / "development_scores" / family / (view + ".npz"), dev["development_scores"][family][view]["sha256"]) as saved,
                  checked_npz(deployed / (view + ".npz"), score["views"][view]["sha256"]) as observed,
                  checked_npz(evaluation / "labels" / (view + ".npz"), labels["views"][view]["label_sha256"]) as lab):
                views[view] = view_parity(view, family, dev, saved, observed, lab, rule, cut["thresholds"])
        finally:
            if rule is not None:
                rule.close()
    passed = all(v["passed"] for v in views.values())
    result = {"schema":"el-development-deployment-parity-1", "run_id":run_id, "status":"passed" if passed else "failed",
              "scene_family_id":family, "split":"development", "condition_id":"B0", "threshold_select_run_id":dev["run_id"],
              "model_index_sha256":dev["model_index_sha256"], "deployment_run_id":score["run_id"], "original_score_run_id":baseline["run_id"],
              "evaluation_run_id":labels["run_id"], "selection_sha256":pin["selection_sha256"], "absolute_tolerance":1e-6, "views":views,
              "family_V":sum(v["V"] for v in views.values()), "family_E":sum(v["E"] for v in views.values()),
              "family_V_minus_E":sum(v["V_minus_E"] for v in views.values())}
    with new_output(output) as folder:
        write_json(folder / "development_parity.json", result)
    return result


def record_audit(a):
    from formal_learning import ROOT, record_once, record_score, run_pin
    from evidence_lens.formal_inputs import PinnedRun
    from formal_lineage import family_record
    from mac_trial import select_files
    store, catalog, lock = a.store.resolve(), a.families.resolve(), a.freeze.resolve()
    method_file = a.method_freeze or store / "protocol/method-freeze-v1.json"
    method = load(method_file)
    require(method["threshold_select_run_id"] == a.threshold_run, "audit requires the explicitly selected threshold run")
    family = min(r["scene_family_id"] for r in load(catalog)["families"] if r["split"] == "development")
    f = family_record(catalog, lock, family)
    observed = record_score(store, catalog, lock, a.study, method_file, family, "B0", a.cpu_python,
                            retry=a.retry_score_of, development_parity=True)
    frozen_pin = {"run_id":a.threshold_run, "manifest_sha256":method["threshold_select_manifest_sha256"]}
    release = PinnedRun(store, frozen_pin)
    dev = load(release.open("payload/development.json"))
    pin = dev["family_sources"][family]["pin"]
    selection_file = store / "trials" / f["trial_id"] / "trial_selection.json"
    selected = load(selection_file)["selected"]
    require(file_hash(selection_file) == pin["selection_sha256"] and all(
            run_pin(selected[role]) == {"run_id":pin[role + "_run_id"], "manifest_sha256":pin[role + "_manifest_sha256"]}
            for role in ("score","evaluation")), "source trial selection differs from threshold dataset")
    rules_ref, eval_ref = (run_pin(selected[k]) for k in ("score", "evaluation"))
    source = PinnedRun(store, rules_ref)
    baseline = load(source.open("payload/score.json"))
    parents = [frozen_pin, run_pin(observed), rules_ref, eval_ref]
    with tempfile.TemporaryDirectory(dir=store, prefix="dev-parity-") as tmp:
        folder = Path(tmp)
        specs = {"release": (frozen_pin, ["development.json","model/model.json","model/thresholds.json",
                                          *(f"development_scores/{family}/{v}.npz" for v in TARGETS)]),
                 "deployed": (run_pin(observed), ["l1_score.json", *(v + ".npz" for v in TARGETS)]),
                 "rules": (rules_ref, ["score.json", *(v + ".npz" for v in TARGETS if baseline["views"][v]["status"] == "scored")]),
                 "evaluation": (eval_ref, ["evaluation.json", *(f"labels/{v}.npz" for v in TARGETS)])}
        inputs = {}
        for role, (parent, filenames) in specs.items():
            rid = parent["run_id"]
            inputs[role] = (select_files(store / "runs" / rid / "payload", folder / role, filenames), rid)
        config = {"schema":"el-development-deployment-audit-1", "run_kind":"archive_verify", "trial_id":f["trial_id"],
                  "experiment_id":"p5-deployment-parity-check", "identity":dict(f, model_id=method["model_id"], model_hash=method["model_index_sha256"]),
                  "family_rule":"lexicographically first development family", "selected":parents, "method_freeze_sha256":file_hash(method_file),
                  "audit_code_sha256":file_hash(ROOT / "scripts/verify_l1_development.py")}
        cmd = [str(a.cpu_python.absolute()), "{source}/scripts/verify_l1_development.py", "_compare", "--family", family,
               "--output", "{out}", "--run-id", "{run_id}"]
        for role in specs:
            cmd.extend(["--" + role, "{input:" + role + "}"])
        return record_once(store, store / "formal" / f["dataset_hash"] / "learning/development-parity.json", config, parents, inputs, cmd, retry=a.retry_audit_of)


def pilot_masks(a):
    from evidence_lens.formal_inputs import PinnedRun
    from formal_lineage import family_record
    family = family_record(a.families, a.freeze, a.family)
    require(family["split"] in ("train", "development"), "pre-freeze pilot mask inspection only reads training/development B0")
    selection = a.store / "trials" / family["trial_id"] / "trial_selection.json"
    s = load(selection)
    require(s["schema"] == "el-p2-trial-selection-1" and s["selection_status"] == "selected"
            and all(s[k] == family[k] for k in ("scene_family_id","scene_id","condition_id","trial_id")), "invalid pilot selection")
    score, evaluation = (PinnedRun(a.store, s["selected"][k]) for k in ("score","evaluation"))
    rules = load(score.open("payload/score.json"))
    lab = load(evaluation.open("payload/evaluation.json"))
    require(rules["run_id"] == lab["scoring_run_id"] == score.run["run_id"] and lab["run_id"] == evaluation.run["run_id"]
            and lab["inputs"]["score_metadata"]["sha256"] == file_hash(score.open("payload/score.json")), "pilot score/label mismatch")
    views = {}
    for v in TARGETS:
        with checked_npz(evaluation.open(f"payload/labels/{v}.npz"), lab["views"][v]["label_sha256"]) as labels:
            valid, evaluable = labels["reference_valid"], labels["rgb_evaluable"]
        outside = valid & ~evaluable
        r = rules["views"][v]
        if r["status"] == "scored":
            with checked_npz(score.open(f"payload/{v}.npz"), r["score_sha256"]) as source:
                values, mask = source["l0"], source["l0_valid"]
            require(np.array_equal(mask, np.isfinite(values)) and np.isin(values[mask], [0,1]).all(), "invalid original L0 score")
        else:
            require(r["status"] == "render_failed" and not evaluable.any(), "inconsistent pilot render failure")
            values = np.full(valid.shape, np.nan, np.float32)
        views[v] = {"V":int(valid.sum()), "E":int(evaluable.sum()), "V_minus_E":int(outside.sum()),
                    "V_minus_E_raw_L0_low":int((outside & (values == 0)).sum()),
                    "V_minus_E_raw_L0_high":int((outside & (values == 1)).sum()),
                    "V_minus_E_raw_L0_unknown":int((outside & ~np.isfinite(values)).sum())}
    totals = {k:sum(view[k] for view in views.values()) for k in views[TARGETS[0]]}
    result = {"schema":"el-pilot-mask-check-1", "scene_family_id":a.family, "trial_id":family["trial_id"],
              "selection_sha256":file_hash(selection), "score":score.pin, "evaluation":evaluation.pin,
              "L0_operating_rule":"finite 0 is low and 1 high on E", "views":views, "totals":totals}
    with new_output(a.output) as out:
        write_json(out / "pilot_masks.json", result)
    return totals


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="op", required=True)
    run = commands.add_parser("record")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--families", type=Path, default=root / "protocol/families-v1.json")
    run.add_argument("--study", type=Path, default=root / "protocol/study-v1.json")
    run.add_argument("--freeze", type=Path, default=root / "configs/formal-freeze-v1.json")
    run.add_argument("--method-freeze", type=Path)
    run.add_argument("--threshold-run", required=True)
    run.add_argument("--cpu-python", required=True, type=Path)
    run.add_argument("--retry-score-of")
    run.add_argument("--retry-audit-of")
    pilot = commands.add_parser("pilot", help="inspect failed predicted RGB and raw L0 in a selected B0 pilot")
    pilot.add_argument("--store", required=True, type=Path)
    pilot.add_argument("--family", required=True)
    pilot.add_argument("--families", type=Path, default=root / "protocol/families-v1.json")
    pilot.add_argument("--freeze", type=Path, default=root / "configs/formal-freeze-v1.json")
    pilot.add_argument("--output", required=True, type=Path)
    check = commands.add_parser("_compare")
    for key in ("release","deployed","rules","evaluation","output"):
        check.add_argument("--" + key, type=Path, required=True)
    check.add_argument("--family", required=True)
    check.add_argument("--run-id", default=os.environ.get("EL_RUN_ID"))
    a = p.parse_args(argv)
    if a.op == "pilot":
        print(json.dumps(pilot_masks(a)))
        return 0
    result = record_audit(a) if a.op == "record" else compare_files(a.release, a.deployed, a.rules, a.evaluation, a.family, a.output, a.run_id)
    print(json.dumps(result if a.op == "record" else {"status":result["status"], "run_id":result["run_id"]}))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
