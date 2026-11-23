"""Exercise the registered branches with generated unit fixtures in pytest's temporary store."""

import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image
from test_formal_evaluation import ROOT, rid, seal

from evidence_lens.formal_evaluation import check_inputs
from evidence_lens.formal_inputs import Staged, protocol, stage
from evidence_lens.formal_metrics import (
    METHODS,
    pool_family,
    ranking_ap,
    view_statistics,
)
from evidence_lens.labels import LABEL_VERSION, label_view
from evidence_lens.scoring_io import TARGETS, file_hash, write_json


def staged_case(tmp, condition, fault=None):
    """Only images and receipts needed by stage/check_inputs are materialized. Never opens a research store."""
    store = tmp / "unit-store"

    def doc(name, value):
        path = tmp / (name + ".json")
        write_json(path, value)
        return path

    def pin(path):
        return {"path": str(path), "sha256": file_hash(path)}

    f = "fabricated-unit-family"
    trials = {c: f + "-" + c for c in ("B0", "R1", "Q1")}
    scene = f + "/v001"
    K = [[400, 0, 255.5], [0, 400, 255.5], [0, 0, 1]]
    source = {"width": 512, "height": 512, "K": K, "c2w": np.eye(4).tolist()}
    original = {"schema": "el-camera-1", "views": {k: source for k in ["source", *TARGETS]}}
    cameras = doc("baseline-cameras", original)
    image = tmp / "baseline.png"
    yy, xx = np.indices((512, 512))
    rgb = np.stack((xx % 251, yy % 251, (xx + yy) % 251), axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(image)
    common = {"camera_sha256": file_hash(cameras), "input_sha256": file_hash(image)}
    ref_metadata = doc(
        "reference",
        {
            "schema": "el-reference-1",
            "validation_passed": True,
            "scene_id": scene,
            "scene_family_id": f,
            **common,
        },
    )
    row = {
        "scene_family_id": f,
        "scene_id": scene,
        "split": "test",
        "example_view": "v01",
        "trials": trials,
        "camera_sha256": file_hash(cameras),
    }
    catalog = doc("toy-catalog", {"dataset_id": "evidence-lens-families-v1", "families": [row]})
    study = doc("toy-study", {"test_methods": list(METHODS)})
    catalog_sha = file_hash(catalog)

    def identity(c):
        return {
            "scene_id": scene,
            "scene_family_id": f,
            "condition_id": c,
            "trial_id": trials[c],
            "dataset_hash": catalog_sha,
        }

    placeholder = tmp / "fixture-data.npz"
    np.savez(placeholder, fixture=np.array([1]))
    refpin = seal(
        store,
        1,
        "scene_render",
        identity("B0"),
        {
            "payload/input.png": image,
            "payload/cameras.json": cameras,
            "payload/reference.json": ref_metadata,
            **{f"payload/reference/{v}.npz": placeholder for v in TARGETS},
        },
    )
    tau = doc(
        "thresholds",
        {
            "threshold_run_id": rid(2),
            "split_manifest_sha256": catalog_sha,
            "thresholds": {k: 0.5 for k in METHODS if k != "L0"},
        },
    )
    model = doc(
        "model",
        {
            "schema": "el-l1-package-1",
            "status": "frozen",
            "model_id": rid(2),
            "threshold_run_id": rid(2),
            "split_manifest_sha256": catalog_sha,
            "constant_risk": 0.2,
            "files": {"thresholds.json": file_hash(tau)},
        },
    )
    mp = seal(
        store,
        2,
        "threshold_select",
        {},
        {"payload/model/model.json": model, "payload/model/thresholds.json": tau},
    )
    proto = protocol(ROOT)
    method = doc(
        "method",
        {
            "schema": "el-method-freeze-1",
            "catalog_sha256": catalog_sha,
            "study_sha256": file_hash(study),
            "model_id": rid(2),
            "threshold_select_run_id": rid(2),
            "threshold_select_manifest_sha256": mp["manifest_sha256"],
            "model_index_sha256": file_hash(model),
            "thresholds_sha256": file_hash(tau),
            "comparison_files": proto["files"],
            "comparison_protocol_sha256": proto["comparison_protocol_sha256"],
        },
    )

    def received(name, parent, files):
        return doc(
            name,
            {
                "inputs": [
                    {"parent_run_id": parent["run_id"], "files": [{"sha256": file_hash(p)} for p in files]}
                ]
            },
        )

    pins = {"reference": refpin, "model": mp}
    img, cam, condition_ref = image, cameras, ref_metadata
    base_ply, variant_ply = tmp / "original.ply", tmp / "variant.ply"
    base_ply.write_bytes(b"unit-only base geometry")
    variant_ply.write_bytes(b"unit-only color variant")
    g = doc(
        "base-geometry",
        {
            "schema": "el-generation-1",
            "producer_run_id": rid(3),
            "gaussian_count": 5,
            "ply_sha256": file_hash(base_ply),
            **common,
        },
    )
    pins["generation"] = seal(
        store, 3, "generation", identity("B0"), {"payload/scene.ply": base_ply, "payload/generation.json": g}
    )
    source_role = "reference"
    if condition == "R1":
        source_role = "preparation"
        half = deepcopy(original)
        half["views"]["source"] = {
            "width": 256,
            "height": 256,
            "K": [[200, 0, 127.5], [0, 200, 127.5], [0, 0, 1]],
            "c2w": source["c2w"],
        }
        if fault == "half pixel":
            half["views"]["source"]["K"][0][2] += 0.5
        cam = doc("half-camera", half)
        img = tmp / "half.png"
        pixels = np.rint(rgb.astype(float).reshape(256, 2, 256, 2, 3).mean(axis=(1, 3))).astype(np.uint8)
        if fault == "one LSB":
            pixels[7, 9, 2] += 1
        Image.fromarray(pixels).save(img)
        condition_ref = doc(
            "half-reference",
            dict(
                json.loads(ref_metadata.read_text()),
                input_sha256=file_hash(img),
                camera_sha256=file_hash(cam),
            ),
        )
        receipt = doc(
            "half-receipt",
            {
                "schema": "el-r1-source-1",
                "parent_camera_sha256": common["camera_sha256"],
                "parent_input_sha256": common["input_sha256"],
                "camera_sha256": file_hash(cam),
                "input_sha256": file_hash(img),
            },
        )
        pins["preparation"] = seal(
            store,
            4,
            "derivation",
            identity("R1"),
            {
                "payload/input.png": img,
                "payload/cameras.json": cam,
                "payload/reference.json": condition_ref,
                "payload/source_derivation.json": receipt,
            },
        )
    else:
        receipt = doc(
            "q1-derivation",
            {
                "schema": "el-gaussian-derivation-1",
                "condition_id": "Q1",
                "producer_run_id": rid(4),
                "gaussian_count": 5,
                "parent_generation_run_id": rid(70) if fault == "wrong generation" else rid(3),
                "parent_ply_sha256": file_hash(variant_ply if fault == "wrong original ply" else base_ply),
                "ply_sha256": file_hash(variant_ply),
                "forward_calls_in_this_stage": 0,
                "colour_fields": ["f_dc_0", "f_dc_1", "f_dc_2"],
                **common,
            },
        )
        pins["preparation"] = seal(
            store,
            4,
            "derivation",
            dict(identity("Q1"), parent_run_ids=[rid(3)]),
            {
                "payload/generation.json": receipt,
                "payload/scene.ply": variant_ply,
                "inputs.json": received("derivation-inputs", pins["generation"], [base_ply, g]),
            },
        )
    markers = {}
    for baseline in [True, False] if condition == "Q1" else [False]:
        c = "B0" if baseline else condition
        idx = 10 if baseline else 15
        prefix = "paired_" if baseline else ""
        owner = "generation" if baseline else "preparation"
        child_parent = pins[owner]
        native_ply = base_ply if baseline else variant_ply
        native_meta = g if baseline else receipt
        prediction = doc(
            c + "-prediction",
            {
                "producer_run_id": rid(idx),
                "generation_run_id": rid(3)
                if fault == "wrong render" and not baseline
                else child_parent["run_id"],
                "ply_sha256": file_hash(native_ply),
                "views": {},
            },
        )
        pins[prefix + "prediction"] = seal(
            store,
            idx,
            "render",
            dict(identity(c), parent_run_ids=[rid(1), child_parent["run_id"]]),
            {
                "payload/prediction.json": prediction,
                "inputs.json": received(c + "-input", child_parent, [native_meta, native_ply]),
            },
        )
        pins[prefix + "rules"] = seal(
            store,
            idx + 1,
            "model_score",
            identity(c),
            {"payload/score.json": doc(c + "-rules", {"views": {}})},
        )
        selected = {
            "reference": refpin,
            "render": pins[prefix + "prediction"],
            "score": pins[prefix + "rules"],
            "generation": pins["generation"],
        }
        marker = {
            "schema": "el-p2-trial-selection-1" if baseline else "el-condition-selection-1",
            "selection_status": "selected",
            "scene_family_id": f,
            "condition_id": c,
            "selected": selected,
        }
        if not baseline:
            selected["preparation"] = pins["preparation"]
            selected["producer"] = pins["preparation"]
            if c == "Q1":
                marker["b0_selection_sha256"] = markers["paired_quality_selection"]["sha256"]
        elif fault == "wrong selection":
            selected["generation"] = dict(pins["generation"], run_id=rid(70))
        q = doc(c + "-marker", marker)
        markers[prefix + "quality_selection"] = pin(q)
        cfg = doc(
            c + "-learned-config",
            {
                "schema": "el-heldout-l1-score-1",
                "quality_result": True,
                "timing_repeat": None,
                "render": pins[prefix + "prediction"],
                "model": mp,
                "source": refpin if baseline else pins[source_role],
                "method_freeze_sha256": file_hash(method),
                "selection_sha256": file_hash(q),
            },
        )
        pins[prefix + "learned"] = seal(
            store,
            idx + 2,
            "model_score",
            dict(identity(c), experiment_id="p6-frozen-model-score"),
            {"config.resolved.yaml": cfg, "payload/l1_score.json": doc(c + "-learned", {"views": {}})},
        )
    labels = doc(
        "evaluation",
        {
            "schema": "el-evaluation-1",
            "label_version": LABEL_VERSION,
            "views": {v: {} for v in TARGETS},
            "run_id": rid(20),
            **identity(condition),
            "inputs": {
                k: {"sha256": file_hash(p)}
                for k, p in (("reference_metadata", condition_ref), ("input_png", img), ("cameras", cam))
            },
        },
    )
    pins["labels"] = seal(
        store,
        20,
        "evaluation",
        identity(condition),
        {"payload/evaluation.json": labels, **{f"payload/labels/{v}.npz": placeholder for v in TARGETS}},
    )
    sel = doc(
        "select",
        {
            "schema": "el-formal-comparison-selection-1",
            "kind": "heldout_study",
            "scene_family_id": f,
            "condition_id": condition,
            "parents": pins,
            **markers,
            "source": {"parent_role": source_role, "path": "payload/input.png"},
            "cameras": {"parent_role": source_role, "path": "payload/cameras.json"},
            "protocol_files": {"method_freeze": pin(method), "study": pin(study), "catalog": pin(catalog)},
        },
    )
    stage(store, sel, tmp / "private-staged", ROOT)
    return Staged(tmp / "private-staged")


@pytest.mark.parametrize(
    "fault,match",
    [(None, None), ("one LSB", "actual source pixels"), ("half pixel", "integer-centre source K")],
)
def test_registered_r1_pixel_and_projection(tmp_path, fault, match):
    staged = staged_case(tmp_path, "R1", fault)
    if match:
        with pytest.raises(ValueError, match=match):
            check_inputs(staged)
    else:
        assert check_inputs(staged)[-1].size == (256, 256)


@pytest.mark.parametrize(
    "fault,match",
    [
        (None, None),
        ("wrong selection", "selected B0 generation"),
        ("wrong generation", "parent PLY"),
        ("wrong original ply", "parent PLY"),
        ("wrong render", "actual selected PLY"),
    ],
)
def test_registered_q1_original_derived_and_both_renderer_pins(tmp_path, fault, match):
    if match:
        with pytest.raises(ValueError, match=match):
            staged_case(tmp_path, "Q1", fault)
    else:
        stage = staged_case(tmp_path, "Q1")
        assert check_inputs(stage)[-1].size == (512, 512)
        assert stage.files["intervention.json"]["parent_role"] == "preparation"
        assert stage.files["q1/b0_generation.json"]["parent_role"] == "generation"


def test_low_support_surface_depth_guard_and_explicit_ap_eligibility():
    rgb = np.zeros((1, 3, 3), np.float32)
    ref = {
        "rgb": rgb,
        "z": np.array([[1, 2, np.nan]], np.float32),
        "position": np.array([[[0, 0, 1], [1, 0, 2], [np.nan] * 3]], np.float32),
        "edge": np.zeros((1, 3), bool),
        "visibility": np.array([[1, 1, 0]], np.uint8),
    }
    pred = {
        "rgb": rgb.copy(),
        "z": np.ones((1, 3), np.float32),
        "alpha": np.array([[0.1, 1, 0.1]], np.float32),
    }
    labels = label_view(ref, pred, True)
    labels["low_support"][0, 2] = True
    scores = {m: np.zeros((1, 3), np.float32) for m in METHODS}

    def evaluate():
        return view_statistics(labels, scores, np.zeros((1, 3), np.uint8), dict.fromkeys(METHODS, 0.5))[0]

    rows = evaluate()
    assert rows["all"]["outcomes"]["low_support_surface_fraction"]["value"] == 0.5
    rows["all"]["methods"]["L0"]["common_ranking"]["ap"] = 1
    assert (
        pool_family({"v01": {"regions": rows}})["all"]["methods"]["L0"]["common_ranking"][
            "eligible_view_count"
        ]
        == 0
    )
    for invalid in (None, np.nan, np.inf, 1.01):
        with pytest.raises(ValueError, match="finite AP"):
            ranking_ap({"status": "two_class", "ap": invalid})
    for key, value in (("depth_error", np.nan), ("depth_error", -0.1), ("signed_depth_error", np.inf)):
        orig = labels[key][0, 1]
        labels[key][0, 1] = value
        with pytest.raises(ValueError, match="evaluable depth"):
            evaluate()
        labels[key][0, 1] = orig
