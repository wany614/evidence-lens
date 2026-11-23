"""Freeze the first complete, independently checked P2 B0 scientific chain."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens._bundle import LABEL, RUN_ID, BundleError, encoded, load, sha, utc
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, timestamp, verify_bundle, verify_dependencies
from evidence_lens.recorder import worker_started
from evidence_lens.scoring_io import TARGETS, VIEWS, PredictionBundle, ReadAudit, load_cameras, load_source_png

KINDS = {"reference": "scene_render", "readback": "archive_verify", "generation": "generation",
         "render": "render", "score": "model_score", "evaluation": "evaluation", "recompute": "archive_verify"}
RULE = "el-p2-first-complete-1"
CHECK_FILE = "independent-view-check.json"
SCIENCE_ORDER = ("generation", "render", "score", "evaluation", "recompute")


def verified_trial(store, trial, frozen_ids=None):
    verified, shared = {}, {}

    def visit(rid, ancestors=frozenset()):
        if rid in verified:
            return
        require(rid not in ancestors and RUN_ID.fullmatch(rid), "invalid/cyclic run dependency")
        path = store / "runs" / rid
        result = verify_bundle(path)
        require(result["run"]["run_id"] == rid, "run ID differs from its directory")
        deps = result["inputs"]
        for ref in [*deps["parents"], *([deps["retry_of"]] if deps["retry_of"] else [])]:
            visit(ref["run_id"], ancestors | {rid})
        # A shared checkpoint can occur in multiple attempts. Verify its bytes
        # once, then check subsequent length claims against that verified object.
        unseen = []
        for row in result["shared"]:
            digest = row["sha256"]
            if digest in shared:
                require(shared[digest] == row["bytes"], "inconsistent shared asset size")
            else:
                shared[digest] = row["bytes"]
                unseen.append(row)
        verify_dependencies(dict(result, shared=unseen), store, verified)
        verified[rid] = result

    if frozen_ids is None:
        ids = [p.parent.name for p in sorted((store / "runs").glob("*/run.json")) if load(p)["trial_id"] == trial]
    else:
        ids = list(frozen_ids)
    for rid in ids:
        visit(rid)
    return {rid: verified[rid] for rid in ids}


def transfer_evidence(store, records):
    receipts = {}
    for path in sorted((store / "receipts" / "transfer").glob("*.json")):
        rec = load(path)
        rid = rec.get("run_id")
        if rid in records and rec.get("status") in {"accepted", "identical"}:
            require(rec.get("kind") == "transfer" and rec.get("receipt_id") == path.stem
                    and rec.get("manifest_sha256") == records[rid]["manifest_sha256"], "transfer receipt identity mismatch")
            receipts.setdefault(rid, []).append({"receipt_id": path.stem, "sha256": sha(path),
                                                 "status": rec["status"], "source_kind": rec["source_kind"],
                                                 "container_sha256": rec.get("container_sha256")})
    return receipts


def chain_ids(records, check_id):
    def parent(rid, kind):
        matches = [p for p in records[rid]["run"]["parent_run_ids"] if p in records and records[p]["run"]["run_kind"] == kind]
        require(len(matches) == 1, f"{rid} does not pin one {kind} parent from this trial")
        return matches[0]

    evaluation = parent(check_id, "evaluation")
    render = parent(evaluation, "render")
    ids = {"reference": parent(evaluation, "scene_render"), "readback": parent(evaluation, "archive_verify"),
           "generation": parent(render, "generation"), "render": render, "score": parent(evaluation, "model_score"),
           "evaluation": evaluation, "recompute": check_id}
    edges = {"reference": (), "readback": ("reference",), "generation": ("reference",),
             "render": ("reference", "generation"), "score": ("reference", "readback", "render"),
             "evaluation": ("reference", "readback", "render", "score"),
             "recompute": ("reference", "readback", "render", "score", "evaluation")}
    for role, rid in ids.items():
        rec = records[rid]["run"]
        require(rec["execution_status"] == "succeeded" and rec["run_kind"] == KINDS[role] and
                set(rec["parent_run_ids"]) == {ids[p] for p in edges[role]}, f"wrong {role} status or parent graph")
    return ids


def check_bound_to_evaluation(records, rid, lineage):
    evaluation = lineage["evaluation"]
    parents = records[evaluation]["run"]["parent_run_ids"]
    reference = [p for p in parents if p in records and records[p]["run"]["run_kind"] == "scene_render"]
    if len(reference) != 1 or set(records[rid]["run"]["parent_run_ids"]) != set(parents) | {evaluation}:
        return False
    sources = {reference[0]: ("cameras.json", "reference/v01.npz"),
               lineage["render"]: ("prediction.json", "source.npz", "v01.npz"), lineage["score"]: ("v01.npz",),
               evaluation: ("labels/v01.npz", "metrics.csv")}
    for parent, names in sources.items():
        bound = {r["sha256"] for inp in records[rid]["inputs"]["inputs"] if inp["parent_run_id"] == parent
                 for r in inp["files"]}
        if any(records[parent]["artifacts"].get("payload/" + name, {}).get("sha256") not in bound for name in names):
            return False
    return True


def scientific_gates(store, records, ids, receipts, experiment):
    def path(role, name):
        return store / "runs" / ids[role] / "payload" / name

    def artifact(role, name):
        entry = records[ids[role]]["artifacts"].get("payload/" + name)
        require(entry is not None and entry["bytes"] > 0 and entry["status"] == "valid", f"unavailable {role}/{name}")
        return entry["sha256"]

    def document(role, name):
        artifact(role, name)
        return load(path(role, name))

    def inputs_from(role, parent):
        return {row["sha256"] for inp in records[ids[role]]["inputs"]["inputs"] if inp["parent_run_id"] == ids[parent]
                for row in inp["files"]}

    def bound(role, parent, names):
        require({artifact(parent, n) for n in names} <= inputs_from(role, parent),
                f"{role} did not bind the specified {parent} source assets")

    meta = {role: records[rid]["run"] for role, rid in ids.items()}
    ref = document("reference", "reference.json")
    source, cameras_hash = artifact("reference", "input.png"), artifact("reference", "cameras.json")
    require(ref["schema"] == "el-reference-1" and ref["validation_passed"] is True and
            ref["input_sha256"] == source and ref["camera_sha256"] == cameras_hash and
            set(ref["views"]) == set(VIEWS), "reference interface or validation failed")
    for role, run in meta.items():
        for key in ("scene_id", "scene_family_id"):
            require(isinstance(ref[key], str) and ref[key] and
                    (run.get(key) == ref[key] or role in {"readback", "recompute"} and run.get(key) is None),
                    f"{role} {key} does not match the reference")
        require(run["trial_id"] == meta["reference"]["trial_id"], f"{role} belongs to another trial")
        if role not in {"reference", "readback"}:
            require(run["experiment_id"] == experiment, f"{role} belongs to another experiment")
        require(run.get("condition_id") == "B0" or
                role in {"reference", "readback", "recompute"} and run.get("condition_id") is None,
                f"{role} belongs to another condition")
        require(run.get("scene_hash") in (None, ref["blend_sha256"]) and
                run.get("camera_set_hash") in (None, cameras_hash), f"{role} has a different scene or camera hash")
    rb = document("readback", "reference_readback.json")
    require(rb.get("schema") == "el-reference-readback-1" and rb.get("integrity_checked") is True and
            rb.get("scientific_acceptance") is True and rb.get("failed_scientific_checks") == [] and
            rb["reference_json_sha256"] == artifact("reference", "reference.json") and set(rb["views"]) == set(VIEWS),
            "independent reference readback did not accept the selected reference")
    bound("readback", "reference", ["reference.json"])
    gen = document("generation", "generation.json")
    require(gen.get("schema") == "el-generation-1" and gen["producer_run_id"] == ids["generation"] and
            gen["input_sha256"] == source and gen["camera_sha256"] == cameras_hash and
            gen["ply_sha256"] == artifact("generation", "scene.ply") and
            type(gen.get("gaussian_count")) is int and gen["gaussian_count"] > 0 and
            gen["checkpoint_sha256"] in {r["sha256"] for r in records[ids["generation"]]["shared"]},
            "generator identities, checkpoint, input or PLY do not match")
    for role in ("generation", "render"):
        require(bool(receipts.get(ids[role])), f"missing Mac transfer receipt for {role}")
        bound(role, "reference", ["input.png", "cameras.json"])
    bound("render", "generation", ["generation.json", "scene.ply"])

    audit = ReadAudit()
    cameras = load_cameras(path("reference", "cameras.json"), audit)
    input_rgb = load_source_png(path("reference", "input.png"), cameras["source"], audit)
    predictions = PredictionBundle(path("render", "."), audit)
    pred, probe = predictions.meta, document("render", "depth-probe.json")
    probe_hash = artifact("render", "depth-probe.json")
    require(pred["producer_run_id"] == ids["render"] and pred["generation_run_id"] == ids["generation"] and
            pred["ply_sha256"] == gen["ply_sha256"] and not predictions.failed and set(pred["views"]) == set(VIEWS)
            and predictions.depth_verified and predictions.depth_receipt == probe_hash and pred["depth_validation"] == probe
            and all(probe.get(k) is True for k in ("off_axis_plane_passed", "alpha_normalization_passed",
                                                   "translation_passed", "linear_blend_passed"))
            and type(probe.get("empty_pixel_alpha")) in (int, float) and probe["empty_pixel_alpha"] == 0,
            "render lineage, seven views or camera/depth/colour/empty calibration failed")
    source_check = None
    for view in VIEWS:
        data = predictions.read(view, cameras[view])
        require(pred["views"][view]["sha256"] == artifact("render", view + ".npz") and data is not None,
                f"invalid {view} prediction arrays")
        artifact("render", view + ".png")
        with Image.open(path("render", view + ".png")) as image:
            require(image.format == "PNG" and image.mode == "RGB" and image.size == cameras[view].shape[::-1],
                    f"invalid {view} prediction PNG")
            require(np.array_equal(np.asarray(image), np.rint(255 * np.nan_to_num(data["rgb"])).astype("uint8")),
                    f"{view} PNG does not represent its prediction array")
        if view == "source":
            error = np.abs(data["rgb"] - input_rgb).mean(axis=-1)
            finite = np.isfinite(error)
            source_check = {"valid_pixels": int(finite.sum()), "total_pixels": int(finite.size),
                            "mae": float(error[finite].mean(dtype=np.float64)) if finite.any() else None}

    score = document("score", "score.json")
    evaluation = document("evaluation", "evaluation.json")
    require(score["schema"] == "el-score-1" and score["run_id"] == ids["score"] and score["producer_run_id"] == ids["render"] and
            score["depth_verified"] is True and score["depth_validation_receipt_sha256"] == probe_hash and
            set(score["views"]) == set(TARGETS) and set(score["methods"]) == {"L0", "1-alpha"}, "invalid L0 score receipt")
    for diagnostic, valid, total in ((pred["source_rgb_check"], "finite_pixels", "pixels"),
                                      (score["source_comparison"], "valid_pixels", "total_pixels")):
        require(diagnostic[valid] == source_check["valid_pixels"] and diagnostic[total] == source_check["total_pixels"]
                and ((source_check["mae"] is None and diagnostic["mae"] is None) or
                     (source_check["mae"] is not None and diagnostic["mae"] is not None and
                      np.isclose(diagnostic["mae"], source_check["mae"], rtol=1e-6, atol=1e-8))),
                "source self-render diagnostic does not reproduce its finite-pixel measurement")
    require(evaluation["schema"] == "el-evaluation-1" and evaluation["run_id"] == ids["evaluation"] and evaluation["producer_run_id"] == ids["render"] and
            evaluation["scoring_run_id"] == ids["score"] and evaluation["depth_verified"] is True and
            evaluation["depth_validation_receipt_sha256"] == probe_hash and
            all(evaluation[k] == meta["evaluation"][k] for k in ("trial_id","scene_id","scene_family_id","condition_id")) and
            evaluation["completed_target_views"] == evaluation["planned_target_views"] == len(TARGETS) and
            set(evaluation["views"]) == set(TARGETS), "evaluation does not cover the selected trial")
    bound("score", "render", ["prediction.json", *(v + ".npz" for v in VIEWS)])
    bound("evaluation", "score", ["score.json", *(v + ".npz" for v in TARGETS)])
    denominators = {}
    for view in TARGETS:
        s, e = score["views"][view], evaluation["views"][view]
        require(s["status"] == "scored" and s["score_sha256"] == artifact("score", view + ".npz") and
                e["render_status"] == "completed" and e["label_sha256"] == artifact("evaluation", f"labels/{view}.npz")
                and 0 <= e["rgb_evaluable"] <= e["reference_valid"] <= cameras[view].height * cameras[view].width
                and 0 <= e["depth_evaluable"] <= cameras[view].height * cameras[view].width, f"invalid {view} counts")
        denominators[view] = {}
        with np.load(path("evaluation", f"labels/{view}.npz"), allow_pickle=False) as labels:
            for key in ("reference_valid", "rgb_evaluable", "depth_evaluable"):
                mask = labels[key]
                require(type(e[key]) is int and mask.dtype == np.bool_ and mask.shape == cameras[view].shape and
                        np.count_nonzero(mask) == e[key], f"{view} {key} denominator disagrees with its mask")
                denominators[view][key] = e[key]
    board = document("evaluation", "board_v01.json")
    require(board["run_id"] == ids["evaluation"] and board["camera_id"] == "v01" and
            board["board_sha256"] == artifact("evaluation", "board_v01.png"), "fixed v01 board is missing or mismatched")
    with Image.open(path("evaluation", "board_v01.png")) as image:
        require(image.format == "PNG", "invalid comparison board format")
        image.verify()

    recomputed = document("recompute", CHECK_FILE)
    require(check_bound_to_evaluation(records, ids["recompute"], ids),
            "independent check did not use this evaluation's declared evidence")
    v = evaluation["views"]["v01"]
    cam = cameras["v01"]
    comparisons = recomputed["ratios_numerator_denominator"]
    keys = {"output/prediction_valid_fraction", "output/invalid_output_fraction", "output/rgb_error_fraction_gt_0.05",
            "output/depth_error_fraction", "L0/high_risk_recall", "L0/warning_precision", "L0/low_risk_error_rate",
            "L0/overall_low_risk_coverage", "L0/unknown_fraction"}
    require(keys <= comparisons.keys() and all(isinstance(pair, list) and len(pair) == 2 and
            all(type(n) is int for n in pair) and 0 <= pair[0] <= pair[1] for pair in comparisons.values()),
            "independent v01 count comparisons are incomplete")
    require(recomputed["view"] == "v01" and recomputed["passed"] is True and
            recomputed["pixel_labels_checked"] == cam.height * cam.width and
            recomputed["scalar_l0_witnesses"] == min(17,cam.height) * min(17,cam.width) and
            recomputed["signed_camera_z_pixels"] == v["depth_evaluable"] and
            comparisons["output/depth_error_fraction"][1] == v["depth_evaluable"] and
            comparisons["output/prediction_valid_fraction"] == [v["rgb_evaluable"], v["reference_valid"]],
            "independent v01 recomputation is incomplete or inconsistent")
    return {"reference_readback_sha256": artifact("readback", "reference_readback.json"),
            "camera_set_sha256": cameras_hash, "input_sha256": source, "scene_sha256": ref["blend_sha256"],
            "checkpoint_sha256": gen["checkpoint_sha256"], "ply_sha256": gen["ply_sha256"],
            "prediction_metadata_sha256": artifact("render", "prediction.json"),
            "score_metadata_sha256": artifact("score", "score.json"),
            "evaluation_metadata_sha256": artifact("evaluation", "evaluation.json"),
            "depth_probe_sha256": probe_hash, "views": list(VIEWS),
            "target_denominators": denominators, "source_self_check": source_check,
            "board_sha256": board["board_sha256"], "independent_view_check_sha256": artifact("recompute", CHECK_FILE),
            "transfer_receipts": {role: receipts[ids[role]] for role in ("generation", "render")}}


def attempt(record):
    run = record["run"]
    keys = ("run_id", "run_kind", "experiment_id", "trial_id", "scene_id", "scene_family_id", "condition_id",
            "execution_status", "retry_of_run_id", "failure_reason", "exit_code", "ended_at_utc", "parent_run_ids")
    return {**{k: run[k] for k in keys}, "requested_retry_of_run_id": run.get("requested_retry_of_run_id"),
            "manifest_sha256": record["manifest_sha256"]}


class SelectionPending(BundleError):
    """An earlier successful scientific execution still lacks a decision."""


def first_scientific_chain(store, records, receipts, experiment, registered_reference=None):
    """Traverse scientific attempts first. Readback scheduling cannot reorder them."""
    excluded, terminal = [], []
    ordered = sorted(records, key=lambda r: (timestamp(records[r]["run"]["started_at_utc"]),
                                              timestamp(records[r]["run"]["ended_at_utc"]), r))

    def stage_ready(role, rid, lineage):
        result = records[rid]
        run = result["run"]
        require(run.get("condition_id") == "B0", f"{role} does not use B0")
        refs = [p for p in run["parent_run_ids"] if p in records and records[p]["run"]["run_kind"] == "scene_render"]
        require(len(refs) == 1 and all(run.get(k) == records[refs[0]]["run"].get(k) for k in ("scene_id", "scene_family_id"))
                and (registered_reference is None or registered_reference == refs[0]),
                f"{role} has a different scene or reference")
        filename = {"generation": "generation.json", "render": "prediction.json",
                    "score": "score.json", "evaluation": "evaluation.json"}[role]
        require("payload/" + filename in result["artifacts"], f"completed {role} has no {filename}")
        doc = load(store / "runs" / rid / "payload" / filename)
        if role == "generation":
            require(run["parent_run_ids"] == refs, "generation has an extra upstream")
            ply = result["artifacts"].get("payload/scene.ply", {})
            require(doc.get("producer_run_id") == rid and type(doc.get("gaussian_count")) is int and
                    doc["gaussian_count"] > 0 and ply.get("bytes", 0) > 0 and ply["sha256"] == doc.get("ply_sha256"),
                    "generation did not produce its declared Gaussian asset")
        elif role == "render":
            require(doc.get("producer_run_id") == rid and doc.get("generation_run_id") == lineage["generation"]
                    and set(doc.get("views", {})) == set(VIEWS) and not doc.get("failed_views")
                    and "payload/depth-probe.json" in result["artifacts"], "render is incomplete or misbound")
            probe = load(store / "runs" / rid / "payload/depth-probe.json")
            require(probe == doc.get("depth_validation") and all(probe.get(k) is True for k in (
                    "off_axis_plane_passed", "alpha_normalization_passed", "translation_passed", "linear_blend_passed"))
                    and type(probe.get("empty_pixel_alpha")) in (int,float) and probe["empty_pixel_alpha"] == 0,
                    "documented render calibration did not pass")
        elif role == "score":
            require(doc.get("run_id") == rid and doc.get("producer_run_id") == lineage["render"] and
                    set(doc.get("views", {})) == set(TARGETS) and
                    all(v.get("status") == "scored" for v in doc["views"].values()), "incomplete or misbound score")
        else:
            require(doc.get("run_id") == rid and doc.get("scoring_run_id") == lineage["score"] and
                    doc.get("producer_run_id") == lineage["render"] and set(doc.get("views", {})) == set(TARGETS)
                    and doc.get("completed_target_views") == len(TARGETS) and
                    "payload/board_v01.png" in result["artifacts"], "incomplete or misbound evaluation")
        if role in ("generation", "render") and not receipts.get(rid):
            raise SelectionPending(f"earlier unresolved {role} {rid}: missing verified transfer receipt")

    def descend(level, lineage):
        role = SCIENCE_ORDER[level]
        previous = SCIENCE_ORDER[level - 1] if level else None
        candidates = []
        for rid in ordered:
            result = records[rid]
            run = result["run"]
            if run["run_kind"] != KINDS[role] or run["experiment_id"] != experiment:
                continue
            if previous and lineage[previous] not in run["parent_run_ids"]:
                continue
            if role == "recompute" and "payload/" + CHECK_FILE not in result["artifacts"]:
                intended = any(a.replace("\\", "/").endswith("/recompute_view.py")
                               for a in result["command"]["argv_template"])
                if not intended or run["execution_status"] == "succeeded":
                    continue
            candidates.append(rid)
        if not candidates and previous:
            raise SelectionPending(f"earlier unresolved {previous} {lineage[previous]}: no recorded {role} outcome")
        scientific_outcome = False
        for rid in candidates:
            run = records[rid]["run"]
            if role == "recompute" and not check_bound_to_evaluation(records, rid, lineage):
                terminal.append({"role": role, "run_id": rid, "reason": "check did not bind the evaluation's evidence",
                                 "execution_status": run["execution_status"]})
                continue
            if run["execution_status"] != "succeeded":
                terminal.append({"role": role, "run_id": rid, "reason": run["failure_reason"],
                                 "execution_status": run["execution_status"]})
                if worker_started(store / "runs" / rid, run):
                    scientific_outcome = True
                continue
            scientific_outcome = True
            try:
                if role == "recompute":
                    ids = chain_ids(records, rid)
                    require(all(ids[k] == v for k, v in lineage.items()), "check points to another scientific path")
                    return ids, scientific_gates(store, records, ids, receipts, experiment)
                stage_ready(role, rid, lineage)
            except SelectionPending:
                raise
            except (ValueError, KeyError, TypeError) as error:
                terminal.append({"role": role, "run_id": rid, "reason": str(error), "execution_status": "succeeded"})
                if role == "recompute":
                    excluded.append({"recompute_run_id": rid, "reason": str(error)})
                continue
            completed = descend(level + 1, dict(lineage, **{role: rid}))
            if completed is not None:
                return completed
        if previous and not scientific_outcome:
            raise SelectionPending(f"earlier unresolved {previous} {lineage[previous]}: no scientifically bound {role} outcome")
        return None

    winner = descend(0, {})
    require(winner is not None, f"no full chain passed the P2 gates: {terminal}")
    ids, proof = winner
    return ids, proof, excluded, terminal


def freeze(store, trial, requested):
    require(LABEL.fullmatch(trial) and set(requested) == set(KINDS) and
            all(RUN_ID.fullmatch(rid) for rid in requested.values()) and len(set(requested.values())) == len(KINDS),
            "supply one explicit, distinct run ID for each P2 role")
    store = store.resolve()
    dest = store / "trials" / trial / "trial_selection.json"
    with writer_gate(store):
        require(not dest.is_symlink(), "canonical trial selection is a symbolic link")
        existing = load(dest) if dest.exists() else None
        if existing is not None:
            require(existing.get("schema") == "el-p2-trial-selection-1" and existing.get("rule_version") == RULE and
                    existing.get("trial_id") == trial and existing.get("selection_status") == "selected" and
                    {k:v["run_id"] for k,v in existing["selected"].items()} == requested,
                    "immutable selection conflicts with requested chain")
        records = verified_trial(store, trial, [a["run_id"] for a in existing["attempts"]] if existing else None)
        require(set(requested.values()) <= records.keys(), "a nominated run is absent from the sealed trial inventory")
        prereg = store / "notes" / "p2-prereg.json"
        use_prereg = prereg.is_file() if existing is None else "preregistration" in existing
        registration = load(prereg) if use_prereg else {}
        registered = registration.get("trial_id") == trial
        experiment = registration["experiment_id"] if registered else records[requested["generation"]]["run"]["experiment_id"]
        if registered:
            require(registration["upstream_reference"] == requested["reference"], "preregistration names a different reference")
        inventory = [attempt(records[rid]) for rid in sorted(records)]
        receipts = transfer_evidence(store, records)
        ids, evidence, excluded, terminal = first_scientific_chain(
            store, records, receipts, experiment, registration["upstream_reference"] if registered else None)
        require(ids == requested, f"earlier verified chain or mismatched nomination: {ids}")
        selected = {role: attempt(records[rid]) for role,rid in ids.items()}
        if existing is not None:
            require(existing["attempts"] == inventory and existing["selected"] == selected,
                    "frozen inventory or selected manifest differs")
            require(existing["scene_id"] == selected["reference"]["scene_id"] and
                    existing["scene_family_id"] == selected["reference"]["scene_family_id"] and
                    existing["condition_id"] == "B0" and existing["experiment_id"] == experiment,
                    "frozen trial identity differs")
            saved_proof = existing["evidence"]
            original_receipts = saved_proof["transfer_receipts"]
            for role in ("generation", "render"):
                require(original_receipts[role] and all(r in evidence["transfer_receipts"][role]
                        for r in original_receipts[role]), f"frozen {role} transfer receipt differs")
            require(dict(evidence, transfer_receipts=original_receipts) == saved_proof, "frozen evidence differs")
            if "preregistration" in existing:
                pin = existing["preregistration"]
                require(pin["path"] == "notes/p2-prereg.json" and sha(store / pin["path"]) == pin["sha256"],
                        "frozen preregistration differs")
            return dest, "identical"
        result = {"schema": "el-p2-trial-selection-1", "rule_version": RULE, "selection_status": "selected",
                  "selector": {"path": "scripts/select_trial.py", "sha256": sha(Path(__file__))},
                  "trial_id": trial, "experiment_id": experiment, "scene_id": selected["reference"]["scene_id"],
                  "scene_family_id": selected["reference"]["scene_family_id"], "condition_id": "B0",
                  "frozen_at_utc": utc(), "selected": selected, "attempts": inventory,
                  "selection_reason": "First scientifically eligible generation, then render, score and evaluation. "
                                      "At each stage order by recorded UTC start, completion and run ID. Independent "
                                      "checks confirm that evaluation. Earlier unresolved executions prevent freezing. "
                                      "Selection does not use error magnitude or method ranking.",
                  "inventory_scope": "All sealed runs for this trial in the canonical Mac store at freeze time.",
                  "earlier_ineligible_chains": excluded, "earlier_terminal_dispositions": terminal, "evidence": evidence}
        if registered:
            result["preregistration"] = {"path": prereg.relative_to(store).as_posix(), "sha256": sha(prereg)}
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
            temp = Path(out.name)
            out.write(encoded(result))
            out.flush()
            os.fsync(out.fileno())
        try:
            temp.chmod(0o444)
            os.link(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
    return dest, "frozen"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--trial", required=True)
    for role in KINDS:
        p.add_argument(f"--{role}-run", required=True)
    args = p.parse_args()
    try:
        path, status = freeze(args.store, args.trial, {role:getattr(args, role + "_run") for role in KINDS})
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.exit(1, f"selection failed: {error}\n")
    print(json.dumps({"status": status, "path": str(path), "sha256": sha(path)}))


if __name__ == "__main__":
    main()
