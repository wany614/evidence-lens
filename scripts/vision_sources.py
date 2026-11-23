"""Freeze explicit B learning sources. Each selection, recorder seal and consumed member is checked."""

import argparse
import json
from pathlib import Path

from evidence_lens._bundle import HASH, load
from evidence_lens.formal_inputs import PinnedRun, compact_pin, quality_parents, require, signature
from evidence_lens.l1 import FEATURE_SCHEMA
from evidence_lens.scoring_io import TARGETS, file_hash, new_output, write_json


def signed(path, digest):
    path = Path(path)
    require(
        isinstance(digest, str) and HASH.fullmatch(digest) and file_hash(path) == digest,
        f"wrong or absent SHA-256 for {path.name}",
    )
    return load(path)


def find_run(roots, pin):
    for root in roots:
        if (root / "runs" / pin["run_id"]).is_dir():
            return PinnedRun(root, pin)
    raise ValueError(f"archived run is unavailable: {pin['run_id']}")


def member(run, name, expected=None):
    path = run.open(name)
    row = run.files[name]
    spec = {"path": name, "bytes": row["bytes"], "sha256": row["sha256"]}
    if expected is not None:
        wanted = expected if isinstance(expected, dict) else {"sha256": expected}
        require(
            isinstance(wanted.get("sha256"), str) and HASH.fullmatch(wanted["sha256"]),
            "missing source member hash",
        )
        require(
            all(spec[k] == v for k, v in wanted.items() if k in ("bytes", "sha256")),
            f"producer/consumer identity differs for {name}",
        )
    return path, spec


def json_member(run, name, expected=None):
    path, spec = member(run, name, expected)
    return load(path), spec


def failed_cell(roots, q, selection_hash, row, condition):
    import numpy as np
    from evidence_lens.scoring_io import rgb_valid

    family, scene = row["scene_family_id"], row["scene_id"]
    trial = row["trials"][condition]
    require(
        q["selection_status"] == "failed"
        and q["planned_targets"] == list(TARGETS)
        and q["trial_id"] == trial
        and q["scene_family_id"] == family
        and q["condition_id"] == condition
        and isinstance(q["reason"], str)
        and q["reason"].strip(),
        "incomplete terminal trial disposition",
    )
    pins = {key: compact_pin(q[key]) for key in ("reference", "failure")}
    ref, fail = (find_run(roots, pins[k]) for k in ("reference", "failure"))
    require(
        ref.run["execution_status"] == "succeeded"
        and fail.run["execution_status"] in ("failed", "cancelled", "interrupted")
        and fail.run.get("trial_id") == trial
        and fail.run.get("scene_family_id") == family
        and fail.run.get("condition_id") == condition
        and ref.run.get("scene_family_id") == family,
        "unsuccessful trial is not bound to its actual failed run and reference",
    )
    meta, rm = json_member(ref, "payload/reference.json")
    require(
        meta["schema"] == "el-reference-1"
        and meta["validation_passed"] is True
        and meta["scene_family_id"] == family
        and meta["scene_id"] == scene,
        "wrong failure reference identity",
    )
    validation = member(ref, "payload/validation/results.json", meta["validation_sha256"])[1]
    views = {}
    for view in TARGETS:
        header = meta["views"][view]
        path, spec = member(ref, f"payload/reference/{view}.npz", header["reference_sha256"])
        with np.load(path, allow_pickle=False) as arr:
            valid = rgb_valid(arr["rgb"])
        require(
            valid.shape == (header["height"], header["width"]) == (512, 512),
            "terminal reference grid differs",
        )
        views[view] = {
            "view": view,
            "height": 512,
            "width": 512,
            "status": "render_failed",
            "reason": q["reason"],
            "reference": spec,
            "labels": None,
            "prediction": None,
            "features": None,
            "V": int(valid.sum()),
            "E": 0,
        }
    return {
        "family": family,
        "scene": scene,
        "split": row["split"],
        "condition": condition,
        "trial_id": trial,
        "cv_fold": row.get("cv_fold"),
        "group": row.get("test_group"),
        "selection_sha256": selection_hash,
        "parents": pins,
        "receipts": {"reference_metadata": rm, "reference_validation": validation},
        "views": views,
        "scientific_disposition": "terminal_upstream_failure",
        "failure_kind": fail.run["run_kind"],
        "input_sha256": None,
        "camera_sha256": None,
    }


def effective_reference(roots, condition, q, pins, original_run, evidence, prediction):
    original, original_spec = json_member(original_run, "payload/reference.json")
    if condition != "R1":
        require(original_spec["sha256"] == evidence["sha256"], "wrong B0/Q1 target reference metadata")
        return original, original_spec, original, original_spec
    asset = q["sources"]["reference_metadata"]
    require(
        compact_pin(asset) == pins["preparation"], "R1 must use the selected half-resolution metadata owner"
    )
    bound, spec = json_member(find_run(roots, pins["preparation"]), asset["path"], evidence)
    require(
        asset["sha256"] == spec["sha256"]
        and bound["parent_reference_sha256"] == original_spec["sha256"]
        and bound["source_condition"] == "R1"
        and bound["input_sha256"] == prediction["input_sha256"]
        and bound["camera_sha256"] == prediction["camera_sha256"]
        and bound["validation_passed"] is True
        and bound["validation_sha256"] == original["validation_sha256"]
        and all(bound["views"][v] == original["views"][v] for v in TARGETS),
        "R1 reference must bind its reduced source while preserving six native targets and validation",
    )
    return bound, spec, original, original_spec


def paired_render_sources(cells):
    by = {c["condition"]: c for c in cells}
    base = by["B0"]
    if base.get("scientific_disposition") == "terminal_upstream_failure":
        return
    for condition in ("R1", "Q1"):
        stress = by[condition]
        if stress.get("scientific_disposition") == "terminal_upstream_failure":
            continue
        require(
            compact_pin(stress["parents"]["prediction"]) != compact_pin(base["parents"]["prediction"]),
            "a stress condition cannot use the original B0 render",
        )
        same = (
            stress["input_sha256"] == base["input_sha256"],
            stress["camera_sha256"] == base["camera_sha256"],
        )
        require(
            same == ((False, False) if condition == "R1" else (True, True)),
            "wrong input image or calibration for " + condition,
        )
        require(
            compact_pin(stress["parents"]["reference"]) == compact_pin(base["parents"]["reference"]),
            "condition reference is not its paired B0 original",
        )


def pinned_cell(roots, selection_path, selection_hash, row, condition, historical=None):
    q = signed(selection_path, selection_hash)
    if q.get("schema") == "el-vision-failed-trial-1":
        return failed_cell(roots, q, selection_hash, row, condition)
    family, scene = row["scene_family_id"], row["scene_id"]
    trial = row["trials"][condition]
    require(
        all(
            q.get(k) == v
            for k, v in (
                ("scene_family_id", family),
                ("scene_id", scene),
                ("trial_id", trial),
                ("condition_id", condition),
            )
        ),
        "trial identity mismatch",
    )
    pins = quality_parents(q)
    runs = {name: find_run(roots, pins[name]) for name in ("prediction", "rules", "labels", "reference")}
    require(all(r.run["execution_status"] == "succeeded" for r in runs.values()), "unusable selected run")
    sr, er, pr, rr = (runs[k] for k in ("rules", "labels", "prediction", "reference"))
    score, sm = json_member(sr, "payload/score.json")
    evaluation, em = json_member(er, "payload/evaluation.json")
    prediction, pm = json_member(pr, "payload/prediction.json", score["inputs"]["prediction_metadata"])
    ref, rm, native_ref, native_spec = effective_reference(
        roots, condition, q, pins, rr, evaluation["inputs"]["reference_metadata"], prediction
    )
    schema, feature_spec = json_member(sr, "payload/feature_schema.json", score["feature_schema_sha256"])
    require(
        score["schema"] == "el-score-1"
        and evaluation["schema"] == "el-evaluation-1"
        and evaluation["label_version"] == "el-labels-1"
        and schema == FEATURE_SCHEMA
        and ref["schema"] == "el-reference-1"
        and ref["validation_passed"] is True,
        "unrecognized source semantics",
    )
    require(
        set(score["views"]) == set(evaluation["views"]) == set(TARGETS)
        and score["run_id"] == sr.run["run_id"] == evaluation["scoring_run_id"]
        and prediction["producer_run_id"]
        == pr.run["run_id"]
        == score["producer_run_id"]
        == evaluation["producer_run_id"]
        and evaluation["run_id"] == er.run["run_id"]
        and all(evaluation[k] == q[k] for k in ("condition_id", "trial_id", "scene_id", "scene_family_id")),
        "score, render, evaluation or trial identity differs",
    )
    require(
        score["depth_verified"] is True
        and evaluation["depth_verified"] is True
        and score["depth_validation_receipt_sha256"]
        == evaluation["depth_validation_receipt_sha256"]
        == prediction["depth_validation_receipt_sha256"],
        "unverified or inconsistent depth evidence",
    )
    require(
        all(v == evaluation["inputs"].get(k) for k, v in score["inputs"].items())
        and sm["sha256"] == evaluation["inputs"]["score_metadata"]["sha256"]
        and score["inputs"]["input_png"]["sha256"] == prediction["input_sha256"]
        and score["inputs"]["cameras"]["sha256"] == prediction["camera_sha256"],
        "source fingerprints differ",
    )
    if historical:
        require(
            historical["selection_sha256"] == selection_hash
            and historical["score_receipt_sha256"] == sm["sha256"]
            and historical["evaluation_receipt_sha256"] == em["sha256"],
            "historical extract pins differ",
        )
    conditioned = condition != "B0"
    if conditioned:
        asset = q["sources"]["cameras"]
        owner = pins["preparation" if condition == "R1" else "reference"]
        require(
            compact_pin(asset) == owner and compact_pin(q["sources"]["input_png"]) == owner,
            "condition camera and input must come from their selected source owner",
        )
        cr = find_run(roots, owner)
        cameras, cm = json_member(cr, asset["path"], prediction["camera_sha256"])
        input_pin = q["sources"]["input_png"]
        condition_render = find_run(roots, compact_pin(input_pin))
        member(condition_render, input_pin["path"], prediction["input_sha256"])
        require(
            input_pin["sha256"] == prediction["input_sha256"]
            and asset["sha256"] == prediction["camera_sha256"],
            "condition-specific source or camera mismatch",
        )
    else:
        cameras, cm = json_member(rr, "payload/cameras.json", prediction["camera_sha256"])
        require(ref["input_sha256"] == prediction["input_sha256"], "B0 input differs from reference source")
    require(cameras["schema"] == "el-camera-1", "unsupported camera")
    reference_cameras, _ = json_member(rr, "payload/cameras.json", native_ref["camera_sha256"])
    for v in TARGETS:
        require(
            all(
                cameras["views"][v][k] == reference_cameras["views"][v][k]
                for k in ("width", "height", "K", "c2w")
            ),
            "condition target camera differs from fixed reference",
        )
    if condition == "R1":
        import numpy as np

        source, original = cameras["views"]["source"], reference_cameras["views"]["source"]
        scaling = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])
        require(
            source["c2w"] == original["c2w"]
            and np.allclose(source["K"], scaling @ np.asarray(original["K"]), rtol=0, atol=1e-10),
            "R1 source pose or half-resolution pixel-centre intrinsics differ",
        )
        require(
            2 * source["width"] == original["width"] and 2 * source["height"] == original["height"],
            "R1 requires its own native half-resolution source",
        )
    if row["split"] == "test":
        expected = {"scene_family_id": family, "trial_id": trial, "condition_id": condition}
        require(
            all(pr.run.get(k) == value for k, value in expected.items()),
            "original render is from a different condition",
        )
    source_hash = score["inputs"].get("prediction/source")
    if source_hash:
        source_member = member(pr, "payload/source.npz", source_hash)[1]
        require(
            source_member["sha256"] == prediction["views"]["source"]["sha256"],
            "source prediction is not from the selected condition render",
        )
    else:
        require(
            isinstance(prediction.get("failed_views", {}).get("source"), str)
            and prediction["failed_views"]["source"].strip()
            and "payload/source.npz" not in pr.files,
            "missing source prediction has no sealed failure disposition",
        )
        source_member = None
    validation = member(rr, "payload/validation/results.json", ref["validation_sha256"])[1]
    refs = {
        "score_metadata": sm,
        "evaluation_metadata": em,
        "prediction_metadata": pm,
        "reference_metadata": rm,
        "native_reference_metadata": native_spec,
        "reference_validation": validation,
        "feature_schema": feature_spec,
        "cameras": cm,
        "source_prediction": source_member,
    }
    views = {}
    for view in TARGETS:
        s, e = score["views"][view], evaluation["views"][view]
        camera, reference_info = cameras["views"][view], ref["views"][view]
        require(
            (camera["height"], camera["width"])
            == (reference_info["height"], reference_info["width"])
            == (512, 512),
            "target reference and prediction must use native 512 grid",
        )
        r = member(rr, f"payload/reference/{view}.npz", evaluation["inputs"][f"reference/{view}"])[1]
        require(r["sha256"] == reference_info["reference_sha256"], "reference member mismatch")
        labels = member(er, f"payload/labels/{view}.npz", e["label_sha256"])[1]
        item = {
            "view": view,
            "height": camera["height"],
            "width": camera["width"],
            "reference": r,
            "labels": labels,
            "V": e["reference_valid"],
            "E": e["rgb_evaluable"],
            "prediction": None,
            "features": None,
            "reason": e["render_failure_reason"],
        }
        if s["status"] == "scored" and e["render_status"] == "completed":
            raw = member(pr, f"payload/{view}.npz", score["inputs"][f"prediction/{view}"])[1]
            feats = member(sr, f"payload/{view}.npz", evaluation["inputs"][f"score/{view}"])[1]
            require(
                raw["sha256"] == prediction["views"][view]["sha256"] and feats["sha256"] == s["score_sha256"],
                f"source member contradiction in {view}",
            )
            item.update(status="completed", prediction=raw, features=feats)
        else:
            require(
                s["status"] == "render_failed"
                and e["render_status"] == "failed"
                and e["rgb_evaluable"] == 0
                and item["reason"],
                f"undeclared missing view {view}",
            )
            item.update(status="render_failed")
        views[view] = item
    return {
        "family": family,
        "scene": scene,
        "split": row["split"],
        "condition": condition,
        "trial_id": trial,
        "cv_fold": row.get("cv_fold"),
        "group": row.get("test_group"),
        "selection_sha256": selection_hash,
        "parents": pins,
        "receipts": refs,
        "views": views,
        "input_sha256": prediction["input_sha256"],
        "camera_sha256": prediction["camera_sha256"],
    }


METHODS = ("constant", "1-alpha", "L0", "L1-full", "L1-geometry", "L2-frozen", "L2-vision-ft")


def check_gate(roots, catalog_sha, families, supplied=None, expected=None):
    """Authenticate the one canonical opening marker and the sealed report it identifies."""
    from evidence_lens.stage2_authority import verify_frozen_authority

    path = roots[0] / "protocol" / "stage2" / catalog_sha / "test-opening.json"
    require(supplied is None or Path(supplied).resolve() == path.resolve(), "unexpected test-opening path")
    require(path.is_file() and not path.is_symlink(), "test remains sealed")
    digest = file_hash(path)
    require(expected is None or expected == digest, "opening marker digest changed")
    opening = load(path)
    require(
        opening["schema"] == "el-stage2-test-opening-1"
        and opening["catalog_sha256"] == catalog_sha
        and len(families) == 8
        and sorted(opening["test_family_ids"]) == sorted(families)
        and opening["method_member"] == "payload/method-freeze.json",
        "incorrect test roster or freeze member",
    )
    freeze = find_run(
        roots, {"run_id": opening["freeze_run_id"], "manifest_sha256": opening["freeze_manifest_sha256"]}
    )
    member(freeze, opening["method_member"], opening["method_sha256"])
    verify_frozen_authority(roots, freeze, catalog_sha, families)
    return digest


def test_authority(roots, catalog_pin):
    require(
        catalog_pin is not None and len(catalog_pin) == 2,
        "pin the registered new family catalog before reading a test asset",
    )
    path, sha = catalog_pin
    catalog = signed(path, sha)
    require(catalog["schema"] == "el-vision-family-roster-1", "wrong confirmation cohort")
    names = {r["scene_family_id"] for r in catalog["families"] if r["split"] == "test"}
    return check_gate(roots, sha, names)


def make_index(
    store,
    shelf,
    catalog_path,
    catalog_sha,
    split,
    output,
    selections=None,
    selection_sha=None,
    test_gate=None,
    test_gate_sha=None,
    only_families=None,
):
    catalog = signed(catalog_path, catalog_sha)
    old = catalog["schema"] == "el-formal-families-1"
    require(old or catalog["schema"] == "el-vision-family-roster-1", "unrecognized split roster")
    require(
        split in ("train", "development", "test") and (not old or split != "test"),
        "historical test material is outside the B learning and confirmation cohorts",
    )
    roots = [Path(store).resolve(), Path(shelf).resolve()]
    rows = [r for r in catalog["families"] if r["split"] == split]
    require(rows and len({r["scene_family_id"] for r in rows}) == len(rows), "empty or duplicate split")
    if split == "test":
        test_gate_sha = check_gate(
            roots, catalog_sha, {r["scene_family_id"] for r in rows}, test_gate, test_gate_sha
        )
    if only_families:
        require(
            split != "test"
            and len(set(only_families)) == len(only_families)
            and set(only_families) <= {r["scene_family_id"] for r in rows},
            "invalid partial learning cohort",
        )
        rows = [r for r in rows if r["scene_family_id"] in only_families]
    selection_list = None
    if not old:
        selection_list = signed(selections, selection_sha)
        require(
            selection_list["schema"] == "el-vision-selected-trials-1"
            and selection_list["catalog_sha256"] == catalog_sha
            and selection_list["split"] == split,
            "trial registry belongs to a different cohort",
        )
    sources = []
    for r in sorted(rows, key=lambda row: row["scene_family_id"]):
        f = r["scene_family_id"]
        for condition in ("B0", "R1", "Q1") if split == "test" else ("B0",):
            trial = r["trials"][condition]
            path = Path(store) / "trials" / trial / "trial_selection.json"
            historical = None
            if old:
                ext = load(Path(store) / "formal" / catalog_sha / "extracts" / (f + ".json"))
                record = find_run(roots, compact_pin(ext))
                history, _ = json_member(record, "payload/family.json", ext["extract_manifest_sha256"])
                historical = history["pin"]
                require(
                    ext["family_pin"] == historical
                    and historical["scene_family_id"] == f
                    and historical["split_manifest_sha256"] == catalog_sha
                    and historical["condition_id"] == "B0"
                    and historical["split"] == ("dev" if split == "development" else "train")
                    and historical["cv_fold"] == r.get("cv_fold")
                    and historical["trial_id"] == trial,
                    "historical learning and catalog assignment differ",
                )
                expected_selection = historical["selection_sha256"]
            else:
                expected_selection = selection_list["trials"][trial]["selection_sha256"]
            if condition != "B0":
                q = signed(path, expected_selection)
                if q["selection_status"] == "selected":
                    base_path = Path(store) / "trials" / r["trials"]["B0"] / "trial_selection.json"
                    base_sha = selection_list["trials"][r["trials"]["B0"]]["selection_sha256"]
                    base = signed(base_path, base_sha)
                    require(
                        q["b0_selection_sha256"] == base_sha
                        and q["scene_family_id"] == base["scene_family_id"]
                        and compact_pin(q["selected"]["reference"])
                        == compact_pin(base["selected"]["reference"]),
                        "stress condition is not paired to the selected B0 reference",
                    )
                    if condition == "Q1":
                        require(
                            compact_pin(q["selected"]["generation"])
                            == compact_pin(base["selected"]["generation"])
                            and compact_pin(q["selected"]["preparation"])
                            == compact_pin(q["selected"]["producer"]),
                            "Q1 is not the precision derivative of the paired original generation",
                        )
            cell = pinned_cell(roots, path, expected_selection, r, condition, historical)
            if old:
                cell["historical_extract"] = compact_pin(ext) | {
                    "extract_manifest_sha256": ext["extract_manifest_sha256"]
                }
            sources.append(cell)
    if split == "test":
        for row in rows:
            paired_render_sources([s for s in sources if s["family"] == row["scene_family_id"]])
    with new_output(output) as out:
        result = {
            "schema": "el-vision-material-index-1",
            "split": split,
            "catalog_sha256": catalog_sha,
            "selection_registry_sha256": selection_sha,
            "test_unlock_sha256": test_gate_sha,
            "cohort": "original_learning" if old else "v11_new",
            "cells": sources,
        }
        write_json(out / "materials.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "shelf", "catalog", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--catalog-sha256", required=True)
    p.add_argument("--split", required=True, choices=("train", "development", "test"))
    p.add_argument("--selections", type=Path)
    p.add_argument("--selection-sha256")
    p.add_argument("--test-gate", type=Path)
    p.add_argument("--test-gate-sha256")
    p.add_argument("--family", action="append", help="Optional explicitly named train/development subset")
    a = p.parse_args()
    result = make_index(
        a.store,
        a.shelf,
        a.catalog,
        a.catalog_sha256,
        a.split,
        a.output,
        a.selections,
        a.selection_sha256,
        a.test_gate,
        a.test_gate_sha256,
        a.family,
    )
    print(
        json.dumps(
            {
                "split": result["split"],
                "cells": len(result["cells"]),
                **signature(a.output / "materials.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
