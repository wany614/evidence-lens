"""Project the six released native landscape bell views and their independent measurements."""

import csv
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image

from evidence_lens._verify import require
from evidence_lens_bundle.bindings import array_sha
from foundation_case_io import (ATTRIBUTION, BELL, REASON_COLORS, UNKNOWN, arrays, categorical_counts,
                                checked_zip, decode_png, digest, encode_png, evaluation, legend, maps, rgb8)

VIEWS = [f"v{i:02}" for i in range(1, 7)]
SOURCES = [f"s{i:02}" for i in range(4)]
METHODS = ("MS-L1", "MS-ft", "MS-frozen", "B-L1-transfer")
BASE = "bundles/landscape/"


def bell_outputs(inputs, pub):
    bundles = inputs["bell_bundles"]
    spec = bundles.metadata["routes"]["landscape"]
    meta = json.loads(bundles.read(spec))
    require(spec["analysis_size"] == [448, 252] and meta["source_order"] == SOURCES and meta["target_order"] == VIEWS
            and bundles.metadata["license"]["original_bell_photographs_and_derived_renders"] == "CC-BY-4.0",
            "incorrect authored A route or rights")
    producer = json.loads(bundles.read(meta["producer"], BASE))
    observations = json.loads(bundles.path(BASE + "source-observations.json").read_bytes())
    photos = inputs["bell_sources"]
    require(observations["observation_run"] == photos.pin and photos.metadata["role"] == "sources",
            "builder used different source observations")
    sources = []
    for sid in SOURCES:
        source = photos.metadata["products"]["landscape"][sid]
        proof = observations["original_observations"][sid]
        original = photos.read(source["rgb8"])
        analysis = bundles.read(meta["sources"][sid], BASE)
        require(proof["analysis_png_sha256"] == digest(analysis) and proof["original_rgb8_sha256"] == digest(original)
                and source["license"] == "CC-BY-4.0"
                and np.array_equal(decode_png(analysis), np.asarray(Image.fromarray(decode_png(original)).resize(
                    (448, 252), Image.Resampling.LANCZOS))), "original observation does not reproduce deployed pixels")
        row = pub.source(BELL, sid, analysis, original, 224)
        row["original_fit_source_sha256"] = proof["original_fit_source_sha256"]
        sources.append(row)

    scores = inputs["bell_scores"]
    archive = scores.metadata
    files = {f["path"]: f for f in archive["score_members"]}
    comparison = inputs["bell_evaluation"]
    population = next(f for f in comparison.metadata["files"] if f["member"] == "per-view-populations.csv")
    with comparison.path(population["member"], population).open() as fp:
        pops = {(r["view"], r["method"]): r for r in csv.DictReader(fp)
                if r["plane"] == "own_content" and r["route"] == "landscape"}
    registered = {item["view"]: item for item in comparison.metadata["views"]}
    gate = inputs["bell_opening"]
    refs = inputs["bell_truth"]
    require(gate.metadata["scores"] == scores.pin and refs.metadata["opening"] == gate.pin
            and refs.metadata["role"] == "references" and refs.metadata["source_renders"] == 0
            and gate.metadata["targets"] == VIEWS and set(registered) == set(VIEWS), "incorrect independent A opening")
    output = []
    with ZipFile(scores.path(archive["archive"], archive)) as z:
        def archived(path):
            return checked_zip(z, path, files[path])

        cells = {}
        for method in METHODS:
            base = "landscape/" + method + "/"
            score = json.loads(archived(base + "scores/scores.json"))
            grade = json.loads(archived(base + "grade.json"))
            require(grade["scores_sha256"] == files[base + "scores/scores.json"]["sha256"]
                    and score["bundle_sha256"] == spec["sha256"] and score["model"] == grade["model"]
                    and score["target_order"] == VIEWS and score["source_order"] == SOURCES,
                    "changed installed score release")
            cells[method] = (score, grade, base)
        for i, view in enumerate(VIEWS):
            fixed = producer["frames"][view]
            rec = registered[view]["deployment"]["landscape"]
            require(fixed["status"] == "ready" and fixed == rec["terminal"], "missing A prediction")
            raw = bundles.read(fixed["file"], BASE)
            pred = arrays(raw)["rgb"]
            require(pred.shape == (252, 448, 3) and array_sha(pred) == rec["original_prediction_array_sha256"]
                    == rec["content_rgb_sha256"], "native FP32 prediction identity changed")
            pred_record = pub.asset(f"{BELL}/predictions/{view}.npz", raw, "sealed native RGB, alpha and camera-Z")
            prediction_id = digest(raw)
            truth, _, _ = bell_reference(inputs, pub, view, pred, registered[view])
            conditions = []
            display = None
            for method, (score, grade, base) in cells.items():
                row = score["views"][i]
                require(row["id"] == view and row["render"] == fixed and row["arrays"] == rec["methods"][method]["arrays"]
                        and score["checkpoint_sha256"] == rec["checkpoint_sha256"], "score attached to other prediction")
                picture = archived(base + "scores/" + row["prediction"]["path"])
                require(digest(picture) == row["prediction"]["sha256"] and np.array_equal(decode_png(picture), rgb8(pred)),
                        "published score preview differs from the native prediction")
                if display is None:
                    display = pub.png(f"{BELL}/targets/{view}.png", picture)
                require(display["sha256"] == digest(picture), "prediction PNG changed across methods")
                nraw = archived(base + "scores/" + row["arrays"]["path"])
                conditions.append(bell_condition(pub, view, method, score, grade, row, nraw, pops[(view, method)],
                                                 truth["counts"], prediction_id, inputs))
            output.append(dict(id=view, target=display, target_prediction_sha256=prediction_id,
                               target_rgb_sha256=digest(pred.tobytes()), target_rgb_typed_sha256=array_sha(pred),
                               native_prediction=pred_record, evaluation=truth, conditions=conditions,
                               identical_prediction=True))
    return dict(id=BELL, title="Offset Bell Cage", rights=ATTRIBUTION, size=[448, 252],
                analysis_profile="analysis-448x252-v1", available_count=4, reconstruction_count=4,
                source_processing="Four independent 896 × 504 authored source observations at the registered cameras, reduced to 448 × 252. The historical M4 fit source hashes are listed separately.",
                provenance={k: inputs[k].pin for k in ("bell_bundles", "bell_sources", "bell_scores", "bell_opening",
                                                        "bell_truth", "bell_evaluation")},
                checkpoint_sha256=producer["checkpoint_sha256"], renderer=producer["renderer"],
                source_ids=SOURCES, sources=sources, views=output)


def bell_reference(inputs, pub, view, pred, registered):
    refs = inputs["bell_truth"]
    ref = refs.metadata["products"]["landscape"][view]
    require(ref == registered["reference"] and ref["license"] == "CC-BY-4.0", "unregistered A reference")
    raw = refs.read(ref["display_fp32"])
    measurement = arrays(raw)["rgb"]
    require(measurement.shape == (504, 896, 3) and measurement.dtype == np.float32
            and np.isfinite(measurement).all() and array_sha(measurement) == registered["reference_array_sha256"],
            "independent A photograph or camera grid differs")
    aligned = measurement.reshape(252, 2, 448, 2, 3).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    p = f"{BELL}/evaluation/{view}"
    original = pub.asset(p + "-original.png", refs.read(ref["rgb8"]), "independent evaluation only, original RGB8")
    fp32 = pub.asset(p + "-measurement.npz", raw, "independent evaluation only, original FP32 display-sRGB")
    return evaluation(pub, BELL, view, pred, aligned, encode_png(rgb8(aligned)), original=original,
                      measurement=fp32, origin="Independent Blender Cycles measurement at 896 × 504. Each 2 × 2 FP32 footprint is averaged in float64 and returned to FP32 before measurement.")


def bell_condition(pub, view, method, score, grade, row, raw, population, measured, prediction, inputs):
    a = arrays(raw)
    risk, finite, state, reason = (a[k] for k in ("risk", "valid", "state", "unknown_reason"))
    hashes = {k: array_sha(v) for k, v in a.items()}
    require(digest(raw) == row["arrays"]["sha256"] and risk.shape == (252, 448)
            and all(grade["views"][view][k]["array_digests"] == hashes for k in ("0", str(64 << 20))),
            "four stored score channels disagree with independent resident replay")
    tau = score["model"]["threshold"]
    require(np.array_equal(np.isfinite(risk), finite) and np.array_equal(reason == 0, finite)
            and np.array_equal(state, np.where(finite, risk.astype(np.float64) >= tau, 255)), "A state mismatch")
    why = legend([score["reason_codes"][str(i)] for i in range(len(score["reason_codes"]))], REASON_COLORS)
    states = legend(["low", "high", "unknown"], ["#307392", "#df7737", UNKNOWN], [0, 1, 255])
    reason_counts = categorical_counts(reason, why)
    state_counts = categorical_counts(state, states)
    require({k: v for k, v in reason_counts.items() if k != "scoreable"} == row["unknown_reasons"]
            and all(state_counts[k] == row["counts"][k] for k in state_counts)
            and int(finite.sum()) == int(population["finite_risk"]) == row["counts"]["scoreable"]
            and int(population["high_error_pixels"]) == measured["errors"]
            and int(population["labelled_pixels"]) == measured["labelled"]
            and int(population["reference_pixels"]) == measured["reference"]
            and abs(float(population["mean_pixel_error"]) - measured["mean_rgb_mae"]) < 1e-13,
            "independent evaluation or model state counts do not reproduce the released rows")
    p = f"{BELL}/models/{view}-{method}"
    q = score["scorer_source_ids"]
    require(q == (["s00"] if method == "B-L1-transfer" else SOURCES), "incorrect actual source use")
    counts = dict(canvas=risk.size, finite=int(finite.sum()), reference=measured["reference"], **state_counts,
                  state_counts=state_counts, reason_counts=reason_counts, query_has_comparable=None,
                  selected_gap=None, library_no_comparable=None, library_incomplete=None, invalid_geometry=None,
                  new_comparable_since_q4=None, new_comparable_since_previous=None)
    return dict(id=method, method=method, requested_k=len(q), k=len(q), query_ids=q, actual_q=q,
                reconstruction_count=4, available_count=4, full_bank_audited=False, counts=counts,
                grain="one native analysis pixel", threshold=tau, threshold_status=score["threshold_applicability"],
                target_prediction_sha256=prediction, model=score["model"],
                provenance=dict(score_run=score["run_id"], score_seal=inputs["bell_scores"].pin,
                                checkpoint_sha256=score["checkpoint_sha256"], renderer=score["renderer"],
                                risk_arrays_sha256=row["arrays"]["sha256"], array_digests=hashes),
                native_arrays=pub.asset(p + ".npz", raw, "sealed risk, valid, threshold state and unknown reason"),
                **maps(pub, p, risk, state, reason, states, why))
