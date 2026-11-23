"""Private formal evaluation of one explicitly selected held-out family and condition."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluation import _cross_counts, _cross_rows
from .features import extract_features
from .formal_board import board
from .formal_inputs import Staged, protocol, require, verify_q1_metadata
from .formal_metrics import (
    METHODS,
    curve_arrays,
    distribution,
    pool_family,
    pooled_groups,
    view_statistics,
)
from .l1 import columns, risk_state
from .labels import LABEL_VERSION, label_view
from .risk_model import score_features
from .scoring_io import (
    TARGETS,
    Camera,
    alpha_valid,
    file_hash,
    new_output,
    rgb_valid,
    write_json,
)

VERSION = "el-formal-comparison-1"


def arrays(stage, rel, keys=None):
    with np.load(stage.open(rel), allow_pickle=False) as z:
        return {k: z[k] for k in (keys or z.files)}


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def matched(stage, path, digest):
    require(stage.files[path]["sha256"] == digest, f"{path} differs from its producer receipt")


def check_inputs(s):
    m = s.meta
    require(
        m["schema"] == "el-formal-comparison-input-1" and m["kind"] in ("heldout_study", "synthetic_fixture"),
        "unsupported comparison kind",
    )
    method = s.json("protocol/method_freeze.json")
    catalog = s.json("protocol/catalog.json")
    study = s.json("protocol/study.json")
    for rel, key in (
        ("catalog", "dataset_hash"),
        ("study", "study_sha256"),
        ("method_freeze", "method_freeze_sha256"),
    ):
        matched(s, "protocol/" + rel + ".json", m[key])
    require(
        method["comparison_protocol_sha256"]
        == m["comparison_protocol_sha256"]
        == protocol(Path(__file__).resolve().parents[2])["comparison_protocol_sha256"]
        and method["catalog_sha256"] == m["dataset_hash"]
        and method["study_sha256"] == m["study_sha256"]
        and method["model_id"] == m["model_id"]
        and method["threshold_select_run_id"] == m["parents"]["model"]["run_id"]
        and method["threshold_select_manifest_sha256"] == m["parents"]["model"]["manifest_sha256"],
        "P5 or comparison version mismatch",
    )
    (row,) = [r for r in catalog["families"] if r["scene_family_id"] == m["scene_family_id"]]
    require(
        row["trials"][m["condition_id"]] == m["trial_id"]
        and row["scene_id"] == m["scene_id"]
        and row["split"] == "test"
        and row["example_view"] == "v01"
        and study["test_methods"] == list(METHODS),
        "identity or scientific comparison differs from freeze",
    )
    model, tau = s.json("model/model.json"), s.json("model/thresholds.json")
    matched(s, "model/thresholds.json", model["files"]["thresholds.json"])
    if m["kind"] == "heldout_study":
        matched(s, "model/model.json", method["model_index_sha256"])
        matched(s, "model/thresholds.json", method["thresholds_sha256"])

    require(
        model["schema"] == "el-l1-package-1"
        and model["status"] == "frozen"
        and model["model_id"] == model["threshold_run_id"] == tau["threshold_run_id"] == m["model_id"]
        and model["split_manifest_sha256"] == tau["split_manifest_sha256"] == m["dataset_hash"]
        and set(tau["thresholds"]) == set(METHODS) - {"L0"},
        "unmatched frozen risk package",
    )
    thresholds = tau["thresholds"] | {"L0": 0.5}
    prior = model["constant_risk"]
    require(
        0 <= prior <= 1
        and all(isinstance(t, (int, float)) and 0 <= t <= 1.000001 for t in thresholds.values()),
        "invalid frozen parameter",
    )
    labels, ref = s.json("labels/evaluation.json"), s.json("reference/reference.json")
    require(
        labels["schema"] == "el-evaluation-1"
        and labels["label_version"] == LABEL_VERSION
        and labels["run_id"] == m["parents"]["labels"]["run_id"]
        and set(labels["views"]) == set(TARGETS)
        and all(labels[k] == m[k] for k in ("scene_id", "scene_family_id", "trial_id", "condition_id")),
        "incorrect label receipt",
    )
    require(
        ref["schema"] == "el-reference-1"
        and ref["validation_passed"] is True
        and all(ref[k] == m[k] for k in ("scene_id", "scene_family_id"))
        and ref["camera_sha256"] == row["camera_sha256"],
        "incorrect original B0 reference",
    )
    if row.get("recipe_sha256") and m["kind"] == "heldout_study":
        require(
            ref.get("formal", {}).get("recipe_sha256") == row["recipe_sha256"], "unfrozen reference recipe"
        )
    if "reference/condition_metadata.json" in s.files:
        matched(s, "reference/condition_metadata.json", labels["inputs"]["reference_metadata"]["sha256"])
    else:
        matched(s, "reference/reference.json", labels["inputs"]["reference_metadata"]["sha256"])
    if m["kind"] == "heldout_study" and m["condition_id"] in ("R1", "Q1"):
        require("intervention.json" in s.files, "formal stress condition requires its actual intervention")
    if "intervention.json" in s.files:
        intervention = s.json("intervention.json")
        if m["condition_id"] == "R1":
            require(
                intervention["schema"] == "el-r1-source-1"
                and intervention["parent_camera_sha256"] == ref["camera_sha256"]
                and intervention["parent_input_sha256"] == ref["input_sha256"],
                "wrong R1 source derivation",
            )
            matched(s, "source/input.png", intervention["input_sha256"])
            matched(s, "source/cameras.json", intervention["camera_sha256"])
        elif m["condition_id"] == "Q1":
            require(
                intervention["schema"] == "el-gaussian-derivation-1"
                and intervention["forward_calls_in_this_stage"] == 0
                and set(intervention["colour_fields"]) == {"f_dc_0", "f_dc_1", "f_dc_2"},
                "wrong Q1 intervention",
            )
            if m["kind"] == "heldout_study":
                binding = m["q1_verified_ply_sha256"]
                base = s.json("q1/b0_generation.json")
                verify_q1_metadata(binding, m["parents"], base, intervention)
                for prefix, c, role in (("", "Q1", "preparation"), ("paired_", "B0", "generation")):
                    if not m.get(prefix + "render_failure_reason"):
                        pred = s.json(prefix + "prediction/prediction.json")
                        require(
                            pred["producer_run_id"] == m["parents"][prefix + "prediction"]["run_id"]
                            and pred["generation_run_id"] == m["parents"][role]["run_id"]
                            and pred["ply_sha256"] == binding[c],
                            "render did not use its indexed PLY",
                        )
    for role, rel in (("input_png", "source/input.png"), ("cameras", "source/cameras.json")):
        matched(s, rel, labels["inputs"][role]["sha256"])
    if m["condition_id"] in ("B0", "Q1"):
        matched(s, "source/input.png", ref["input_sha256"])
        matched(s, "source/cameras.json", ref["camera_sha256"])
    if m.get("render_failure_reason"):
        require(
            labels["producer_run_id"] is None and labels["scoring_run_id"] is None,
            "whole-render failure has a successful producer or rule score",
        )
    cameras = s.json("source/cameras.json")
    require(
        cameras["schema"] == "el-camera-1" and set(cameras["views"]) == {"source", *TARGETS},
        "incomplete cameras",
    )
    with Image.open(s.open("source/input.png")) as source:
        require(
            source.format == "PNG"
            and source.mode == "RGB"
            and source.size == tuple(cameras["views"]["source"][a] for a in ("width", "height")),
            "condition PNG disagrees with source camera",
        )
        if m["kind"] == "heldout_study":
            expected = 256 if m["condition_id"] == "R1" else 512
            require(
                source.size == (expected, expected), "source dimensions disagree with declared intervention"
            )
        source_image = source.copy()
    if m["condition_id"] == "R1" and m["kind"] == "heldout_study":
        matched(s, "reference/cameras.json", ref["camera_sha256"])
        matched(s, "reference/input.png", ref["input_sha256"])
        original = s.json("reference/cameras.json")["views"]
        for v in TARGETS:
            require(cameras["views"][v] == original[v], "R1 changed a target camera")
        original_source = original["source"]
        half = cameras["views"]["source"]
        require(
            (original_source["width"], original_source["height"]) == (512, 512),
            "registered original R1 source camera must be 512 pixels square",
        )
        require(half["c2w"] == original_source["c2w"], "R1 source pose differs from B0")
        K = np.asarray(original_source["K"], dtype=np.float64)
        conversion = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])
        require(
            np.allclose(conversion @ K, half["K"], atol=1e-12, rtol=0),
            "R1 integer-centre source K differs from half-size transform",
        )
        with Image.open(s.open("reference/input.png")) as orig_img:
            require(
                orig_img.format == "PNG" and orig_img.mode == "RGB" and orig_img.size == (512, 512),
                "registered original R1 source must be a 512-pixel RGB PNG",
            )
            a = np.asarray(orig_img, dtype=np.uint16)
        total = a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]
        expected = np.rint(total.astype(np.float64) / 4).astype(np.uint8)
        require(
            np.array_equal(np.asarray(source_image), expected),
            "R1 actual source pixels differ from the fixed box mean",
        )

    return labels, ref, thresholds, prior, cameras, source_image


def prediction_chain(s, prefix, labels, thresholds):
    missing = s.meta.get(prefix + "render_failure_reason")
    if missing:
        return None
    ids = {k: s.meta["parents"][prefix + k]["run_id"] for k in ("prediction", "rules", "learned")}
    p = s.json(prefix + "prediction/prediction.json")
    r = s.json(prefix + "rules/score.json")
    l = s.json(prefix + "learned/l1_score.json")
    require(
        p["schema"] == "el-prediction-1"
        and p["producer_run_id"] == ids["prediction"]
        and p["rgb_encoding"] == "display_srgb_0_1"
        and p["alpha_semantics"] == "accumulated_opacity"
        and p["depth_semantics"] == "expected_camera_z_m",
        "incompatible prediction",
    )
    require(
        r["schema"] == "el-score-1"
        and r["run_id"] == ids["rules"]
        and r["producer_run_id"] == ids["prediction"],
        "L0/alpha scores use another render",
    )
    require(
        l["schema"] == "el-l1-score-1"
        and l["run_id"] == ids["learned"]
        and l["model_id"] == s.meta["model_id"]
        and l["methods"] == METHODS
        and l["thresholds"] == {m: t for m, t in thresholds.items() if m != "L0"},
        "incorrect deployment",
    )
    matched(s, "model/model.json", l["model_files"]["model_index"]["sha256"])
    require(
        l.get("producer_run_id") == ids["prediction"] or l.get("status") == "unsupported",
        "wrong L1 render parent",
    )
    for name, key in (("source/input.png", "input_sha256"), ("source/cameras.json", "camera_sha256")):
        matched(s, name, p[key])
    for role, receipt in (("rules", r), ("learned", l)):
        bound = receipt["inputs"]
        for role_name, path in (
            ("input_png", "source/input.png"),
            ("cameras", "source/cameras.json"),
            ("prediction_metadata", prefix + "prediction/prediction.json"),
        ):
            if role_name in bound:
                matched(s, path, bound[role_name]["sha256"])
            else:
                require(role == "learned" and l.get("status") == "unsupported", "missing score input pin")
    if not prefix:
        require(
            labels["producer_run_id"] == ids["prediction"]
            and labels["scoring_run_id"] == ids["rules"]
            and labels["depth_verified"] == r["depth_verified"],
            "wrong P2 label chain",
        )
        matched(s, "rules/score.json", labels["inputs"]["score_metadata"]["sha256"])
        matched(s, "prediction/prediction.json", labels["inputs"]["prediction_metadata"]["sha256"])
    return p, r, l


def target(s, prefix, chain, view, shape, prior, thresholds, geom_checks=False):
    missing = bool(chain is None or view in chain[0].get("failed_views", {}))
    failed_reason = (
        s.meta.get(prefix + "render_failure_reason")
        if chain is None
        else chain[0].get("failed_views", {}).get(view)
    )
    blank = np.full(shape, np.nan, np.float32)
    if missing:
        if chain:
            require(
                chain[1]["views"][view]["status"] == "render_failed"
                and chain[2]["views"][view]["status"] in ("render_failed", "unknown"),
                "declared missing render disagrees with scoring receipts",
            )
            for name in ("prediction", "rules"):
                require(
                    f"{prefix}{name}/{view}.npz" not in s.files, "declared failed render has output arrays"
                )
            rel = f"{prefix}learned/{view}.npz"
            if rel in s.files:
                learned = arrays(s, rel)
                require(
                    all(
                        np.isnan(learned[k]).all()
                        and not learned[k + "_valid"].any()
                        and np.all(learned[k + "_state"] == 255)
                        for k in METHODS.values()
                    ),
                    "failed target has finite risk",
                )
        return None, {method: blank for method in METHODS}, None, failed_reason, {}, True
    p, r, l = chain
    raw_pred = f"{prefix}prediction/{view}.npz"
    raw_rules = f"{prefix}rules/{view}.npz"
    matched(s, raw_pred, p["views"][view]["sha256"])
    matched(s, raw_rules, r["views"][view]["score_sha256"])
    matched(s, raw_pred, r["inputs"]["prediction/" + view]["sha256"])
    require(r["views"][view]["status"] == "scored", "missing successful rule status")
    pred = arrays(s, raw_pred, ("rgb", "z", "alpha"))
    require(
        pred["rgb"].shape == (*shape, 3)
        and all(a.dtype == np.float32 for a in pred.values())
        and pred["z"].shape == pred["alpha"].shape == shape,
        "incompatible prediction array",
    )
    names = ("l0", "alpha_score", "l0_valid", "alpha_score_valid", "l0_reason")
    geom_names = ("scoreable", "frame_relation", *columns("L1-geometry")[1]) if geom_checks else ()
    rules = arrays(s, raw_rules, names + geom_names)
    scores = {
        "constant": np.where(rgb_valid(pred["rgb"]), np.float32(prior), np.float32(np.nan)),
        "1-alpha": rules["alpha_score"],
        "L0": rules["l0"],
    }
    for key in ("alpha_score", "l0"):
        require(same(rules[key + "_valid"], np.isfinite(rules[key])), f"{key} availability mismatch")
    method_status = l["views"][view]["status"]
    require(
        method_status in ("scored", "unsupported_l1_depth", "unknown"), "undeclared learned-score failure"
    )
    unsupported = method_status != "scored"
    rel = f"{prefix}learned/{view}.npz"
    if rel in s.files:
        matched(s, rel, l["views"][view]["sha256"])
        if method_status != "unknown":
            matched(s, raw_pred, l["inputs"]["prediction/" + view]["sha256"])
        learned = arrays(s, rel)
        for method, key in METHODS.items():
            arr = learned[key]
            require(
                same(learned[key + "_valid"], np.isfinite(arr))
                and same(learned[key + "_state"], risk_state(arr, thresholds[method])),
                f"{method} frozen state or validity mismatch",
            )
            if method in scores and method_status != "unknown":
                require(same(scores[method], arr), f"{method} deployment and rule/constant scores differ")
            if method not in scores:
                scores[method] = arr
        require(
            same(np.isfinite(scores["L1-full"]), np.isfinite(scores["L1-geometry"])),
            "ablations changed availability",
        )
    else:
        require(method_status == "unknown", "missing L1 numerical maps")
        scores["L1-full"] = scores["L1-geometry"] = blank
    if unsupported:
        require(
            np.isnan(scores["L1-full"]).all() and np.isnan(scores["L1-geometry"]).all(),
            "unsupported L1 produced a score",
        )
    return pred, scores, rules["l0_reason"], None, {k: rules[k] for k in geom_names}, unsupported


def field_change(a, b):
    require(a.dtype == b.dtype and a.shape == b.shape, "paired intervention field dtype/shape changed")
    # Byte comparison includes signed zero and NaN payload. Also give numeric differences on jointly finite values.
    ap, bp = np.ascontiguousarray(a), np.ascontiguousarray(b)
    unequal = (ap.view(np.uint8).reshape(-1, a.itemsize) != bp.view(np.uint8).reshape(-1, b.itemsize)).any(
        axis=1
    )
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    return {
        "values": int(a.size),
        "bitwise_different": int(unequal.sum()),
        "finite_mask_different": int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
        "joint_finite": int(d.size),
        "max_abs": float(d.max()) if d.size else None,
        "sum_abs": float(d.sum()),
        "numeric_over_1e_6": int(np.count_nonzero(d > 1e-6)),
    }


def contribution_changes(a, b):
    valid_a, valid_b = alpha_valid(a), alpha_valid(b)
    return {
        "valid_contribution": field_change(valid_a, valid_b),
        "low_contribution": field_change(valid_a & (a < 0.5), valid_b & (b < 0.5)),
        "sufficient_contribution": field_change(valid_a & (a >= 0.5), valid_b & (b >= 0.5)),
    }


def paired_response(current, previous):
    pred, scores, l0_reason, reason, geom, _ = current
    base, b_scores, b_l0_reason, b_reason, b_geom, _ = previous
    if pred is None or base is None:
        return {"status": "unavailable", "reason": reason or b_reason, "paired_valid_rgb": 0, "rgb_mae": None}
    include = rgb_valid(pred["rgb"]) & rgb_valid(base["rgb"])
    difference = pred["rgb"][include].astype(np.float64) - base["rgb"][include].astype(np.float64)
    d = np.mean(np.abs(difference), axis=-1)
    fields = {name: field_change(pred[name], base[name]) for name in ("z", "alpha")}
    fields.update({name: field_change(geom[name], b_geom[name]) for name in geom})
    for method in ("L0", "1-alpha", "L1-geometry", "L1-full"):
        fields[method] = field_change(scores[method], b_scores[method])
    fields["l0_reason"] = field_change(l0_reason, b_l0_reason)
    fields.update(contribution_changes(pred["alpha"], base["alpha"]))
    return {
        "status": "compared",
        "paired_valid_rgb": int(d.size),
        "rgb_abs_sum": float(d.sum()),
        "rgb_mae": float(d.mean()) if d.size else None,
        "rgb_max_pixel_mae": float(d.max()) if d.size else None,
        "rgb_changed_pixel_count": int(np.count_nonzero(d > 0)),
        "rgb_signed_channel_sum": difference.sum(axis=0).tolist(),
        "fields": fields,
        "geometry_bitwise_equal": all(
            v["bitwise_different"] == 0 for k, v in fields.items() if k != "L1-full"
        ),
    }


def source_response(s, chain, baseline):
    info, geometry = {}, {}
    cs = s.json("source/cameras.json")["views"]["source"]
    camera = Camera(cs["width"], cs["height"], np.array(cs["K"]), np.array(cs["c2w"]))
    with Image.open(s.open("source/input.png")) as image:
        input_rgb = np.asarray(image, dtype=np.float32) / np.float32(255)
    for prefix, part in (("", chain), ("paired_", baseline)):
        if part is None or "source" in part[0].get("failed_views", {}):
            return {"status": "unavailable", "reason": "one or both selected source renders failed"}
        rel = prefix + "prediction/source.npz"
        matched(s, rel, part[0]["views"]["source"]["sha256"])
        matched(s, rel, part[1]["inputs"]["prediction/source"]["sha256"])
        info[prefix] = arrays(s, rel, ("rgb", "z", "alpha"))
        native = info[prefix]
        features = extract_features(input_rgb, native, native, camera, camera, part[1]["depth_verified"])
        rules = score_features(features, part[1]["depth_verified"])
        geometry[prefix] = {
            k: features[k] for k in ("scoreable", "frame_relation", *columns("L1-geometry")[1])
        }
        geometry[prefix].update(
            {"L0": rules["l0"], "1-alpha": rules["alpha_score"], "l0_reason": rules["l0_reason"]}
        )
    q, b = info[""], info["paired_"]
    fields = {k: field_change(q[k], b[k]) for k in ("z", "alpha")}
    fields.update(contribution_changes(q["alpha"], b["alpha"]))
    fields.update({k: field_change(a, geometry["paired_"][k]) for k, a in geometry[""].items()})
    support = rgb_valid(q["rgb"]) & rgb_valid(b["rgb"])
    diff = np.abs(q["rgb"][support].astype(np.float64) - b["rgb"][support].astype(np.float64))
    return {
        "status": "compared",
        "fields": fields,
        "derived_fields_origin": "Public feature and L0 functions applied source-to-itself on each selected native render; excluded from six-target quality metrics.",
        "geometry_bitwise_equal": all(v["bitwise_different"] == 0 for v in fields.values()),
        "paired_valid_rgb": int(support.sum()),
        "rgb_mae": float(diff.mean()) if diff.size else None,
    }


def run(bundle, output, run_id):
    s = Staged(bundle)
    m = s.meta
    labels_doc, ref_doc, thresholds, prior, cameras, source = check_inputs(s)
    chain = prediction_chain(s, "", labels_doc, thresholds)
    pair = m["condition_id"] == "Q1"
    paired = prediction_chain(s, "paired_", labels_doc, thresholds) if pair else None
    with new_output(Path(output)) as out:
        (out / "curves").mkdir()
        groups = {m: [] for m in METHODS}
        views, hashes, by_view_reasons = {}, {}, []
        depth_samples = {k: [] for k in ("signed_depth_error", "signed_relative_depth_error")}
        joint_reasons = np.zeros((11, 4), dtype=np.int64)

        def save_curves(key, groups_by_method, V, region):
            entries = {}
            for method, group in groups_by_method.items():
                data = curve_arrays(group, V, thresholds[method])
                idx = data["frozen_index"]
                counts = region["methods"][method]["counts"]
                require(
                    data["accepted"][idx] == counts["L"] and data["errors"][idx] == counts["LY"],
                    "frozen curve position disagrees with system state",
                )
                entries.update({METHODS[method] + "__" + k: v for k, v in data.items()})
            rel = "curves/" + key + ".npz"
            np.savez_compressed(out / rel, **entries)
            hashes[rel] = file_hash(out / rel)

        for view in TARGETS:
            labels_file, ref_file = f"labels/{view}.npz", f"reference/{view}.npz"
            matched(s, labels_file, labels_doc["views"][view]["label_sha256"])
            matched(s, ref_file, ref_doc["views"][view]["reference_sha256"])
            matched(s, ref_file, labels_doc["inputs"]["reference/" + view]["sha256"])
            ref, labels = (
                arrays(s, ref_file, ("rgb", "z", "position", "visibility", "edge")),
                arrays(s, labels_file),
            )
            edge = ref["edge"]
            require(
                edge.dtype in (np.bool_, np.uint8) and np.all((edge == 0) | (edge == 1)),
                "invalid reference edge mask",
            )
            ref["edge"] = edge.astype(bool, copy=False)
            shape = tuple(cameras["views"][view][k] for k in ("height", "width"))
            if m["kind"] == "heldout_study":
                require(shape == (512, 512), "formal target must be 512 square")
            require(
                labels["reference_valid"].shape == shape and ref["rgb"].shape == (*shape, 3),
                "camera and labels differ",
            )
            current = target(s, "", chain, view, shape, prior, thresholds, pair)
            pred, scores, reasons, failure, _, unsupported = current
            label_row = labels_doc["views"][view]
            require(
                (pred is None) == (label_row["render_status"] == "failed")
                and failure == label_row["render_failure_reason"],
                "conflicting planned-view render status",
            )
            if pred is not None:
                matched(
                    s, "prediction/" + view + ".npz", labels_doc["inputs"]["prediction/" + view]["sha256"]
                )
                matched(s, "rules/" + view + ".npz", labels_doc["inputs"]["score/" + view]["sha256"])
            expected = label_view(ref, pred, labels_doc["depth_verified"])
            require(
                all(same(labels[k], a) for k, a in expected.items()),
                f"{view} labels differ from direct reference/prediction recomputation",
            )
            for k, views_for_depth in depth_samples.items():
                views_for_depth.append(labels[k][labels["depth_evaluable"]])
            regions, view_groups = view_statistics(
                labels,
                scores,
                reasons if reasons is not None else np.full(shape, 10, np.uint8),
                thresholds,
                unsupported,
            )
            rec = {
                "render_status": label_row["render_status"],
                "failure_reason": failure,
                "learned_status": chain[2]["views"][view]["status"] if chain else "render_failed",
                "learned_reason": chain[2]["views"][view].get("reason", chain[2].get("reason"))
                if chain
                else failure,
                "regions": regions,
                "depth_distribution": {k: distribution(v[-1]) for k, v in depth_samples.items()},
            }
            if pair:
                rec["paired_Q1_minus_B0"] = paired_response(
                    current, target(s, "paired_", paired, view, shape, prior, thresholds, True)
                )
            views[view] = rec
            counts = _cross_counts(labels, {"l0_reason": reasons} if reasons is not None else None)
            joint_reasons += counts
            by_view_reasons.extend(_cross_rows(counts, view))
            for method in METHODS:
                groups[method].append(view_groups[method])
            save_curves(view, view_groups, regions["all"]["counts"]["V"], regions["all"])
            if view == "v01":
                selected = {
                    k: v["sha256"]
                    for k, v in s.files.items()
                    if k.endswith("/v01.npz") or k == "source/input.png"
                }
                board(out / "board", m, source, ref, pred, labels, scores, run_id, failure, selected)
        family = pool_family(views)
        save_curves(
            "family",
            {k: pooled_groups(v) for k, v in groups.items()},
            family["all"]["counts"]["V"],
            family["all"],
        )
        write_json(
            out / "reason_visibility.json",
            {
                "views": by_view_reasons,
                "family": _cross_rows(joint_reasons, "all_six_target_views"),
                "mask": "full reference-valid RGB V with row and column totals",
            },
        )
        receipt = {
            k: m[k]
            for k in (
                "kind",
                "scene_family_id",
                "scene_id",
                "trial_id",
                "condition_id",
                "dataset_id",
                "dataset_hash",
                "study_sha256",
                "method_freeze_sha256",
                "comparison_protocol_sha256",
                "model_id",
                "parents",
            )
        }
        write_json(
            out / "curve_schema.json",
            {
                "schema": "el-attainable-curve-1",
                "prefixes": METHODS,
                "files": "curves/v01.npz through v06.npz and curves/family.npz",
                "columns": {
                    "score": "greatest admitted float32 risk in an entire tie group (row 0 = NaN for accept-none)",
                    "accepted": "cumulative accepted pixels from the method's finite-score subset of E",
                    "errors": "cumulative actual primary RGB errors among accepted pixels",
                    "V": "full reference-valid pixel count for this view or family",
                    "frozen_tau": "development-frozen float64 threshold (L0 uses 0.5)",
                    "frozen_index": "row admitting exactly the finite-score pixels with score < frozen_tau",
                },
                "coverage": "accepted / V (undefined if V=0)",
                "risk": "errors / accepted (undefined if accepted=0)",
                "family_pool": "All six targets use one common threshold. Equal scores across views are admitted together.",
            },
        )
        receipt.update(
            schema=VERSION,
            curve_schema_sha256=file_hash(out / "curve_schema.json"),
            run_id=run_id,
            comparison_inputs_sha256=file_hash(s.path / "comparison.json"),
            render_failure_reason=m.get("render_failure_reason"),
            generation_run_id=chain[0].get("generation_run_id") if chain else None,
            source_Q1_minus_B0=source_response(s, chain, paired) if pair else None,
            methods=list(METHODS),
            thresholds=thresholds,
            constant_risk=prior,
            planned_targets=len(TARGETS),
            completed_targets=sum(r["render_status"] == "completed" for r in views.values()),
            target_reference_sha256={v: ref_doc["views"][v]["reference_sha256"] for v in TARGETS},
            input_sha256=s.files["source/input.png"]["sha256"],
            camera_sha256=s.files["source/cameras.json"]["sha256"],
            reference_camera_sha256=ref_doc["camera_sha256"],
            views=views,
            family=family,
            intervention=s.json("intervention.json") if "intervention.json" in s.files else None,
            depth_distribution={k: distribution(np.concatenate(v)) for k, v in depth_samples.items()},
            curves_sha256=hashes,
            board_sha256=file_hash(out / "board/board_v01.png"),
            reason_visibility_sha256=file_hash(out / "reason_visibility.json"),
        )
        write_json(out / "formal_evaluation.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    args = p.parse_args(argv)
    run(args.bundle, args.output, args.run_id)


if __name__ == "__main__":
    main()
