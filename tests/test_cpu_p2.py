"""Private, deterministic unit fixtures. These tests produce no engineering-trial results."""

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evidence_lens.comparison_board import INVALID_COLOR, RISK_COLORS, render_board
from evidence_lens.evaluation import _ap, _groups, run_evaluation
from evidence_lens.features import extract_features
from evidence_lens.labels import ReferenceBundle, label_view
from evidence_lens.risk_model import run_score, score_features
from evidence_lens.scoring_io import (
    TARGETS, VIEWS, Camera, PredictionBundle, ReadAudit, file_hash, load_cameras, load_source_png, write_json,
)


def camera(w=3, h=2, f=4):
    return Camera(w, h, np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]]), np.eye(4))


def arrays(cam):
    return {"rgb": np.zeros((*cam.shape, 3), np.float32), "z": np.full(cam.shape, 2, np.float32),
            "alpha": np.ones(cam.shape, np.float32)}


def test_half_size_projection_ties_and_neighbourhood():
    src, tar = camera(4, 3, 3), camera(8, 6, 6)
    source, target = arrays(src), arrays(tar)
    input_rgb = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 35
    source["rgb"][:] = input_rgb
    target["rgb"][:] = input_rgb.repeat(2, axis=0).repeat(2, axis=1)
    feat = extract_features(input_rgb, source, target, src, tar, True)
    assert feat["frame_inside"].all()
    assert np.all(feat["source_rgb_difference"] == 0)
    assert np.all(feat["source_render_residual"] == 0)
    np.testing.assert_allclose(feat["border_distance"][0, 0], 0.25 / 3)
    np.testing.assert_allclose(feat["local_rgb_difference"][1, 1],
                               np.abs(target["rgb"][:3, :3] - target["rgb"][1, 1]).mean(), atol=1e-7)
    # +0.5 integer-centre pixel goes to the larger index, including the excluded right boundary.
    original = camera(3, 2, 4)
    shifted = camera(3, 2, 4)
    shifted.c2w[0, 3] = 0.25
    source, target = arrays(original), arrays(shifted)
    source["alpha"][0] = [0.6, 0.7, 0.8]
    feat = extract_features(source["rgb"], source, target, original, shifted, True)
    np.testing.assert_allclose(feat["source_alpha"][0, :2], [0.7, 0.8])
    assert feat["frame_outside"][0, 2] and score_features(feat, True)["l0_reason"][0, 2] == 5
    shifted.c2w[0, 3] = -0.25
    assert extract_features(source["rgb"], source, target, original, shifted, True)["frame_inside"][0, 0]
    shifted.c2w[:3, :3] = np.diag([-1, 1, -1])
    back = extract_features(source["rgb"], source, target, original, shifted, True)
    assert back["frame_behind_camera"].all() and (score_features(back, True)["l0_reason"] == 5).all()
    tiny = camera(2, 1)
    t = arrays(tiny)
    feat = extract_features(t["rgb"], t, t, tiny, tiny, True)
    assert feat["missing_local_rgb_difference"].all() and feat["missing_local_depth_range"].all()


def test_l0_priority_front_diagnostic_and_depth_gate():
    cam = camera(8, 2)
    src, tar = arrays(cam), arrays(cam)
    tar["alpha"][0, :2] = [np.nan, 0.1]
    tar["z"][0, 1:3] = np.nan
    src["alpha"][0, 3:5] = [np.nan, 0.1]
    src["z"][0, 5:8] = [np.nan, 1, 3]
    feat = extract_features(src["rgb"], src, tar, cam, cam, True)
    np.testing.assert_array_equal(score_features(feat, True)["l0_reason"][0], [1, 2, 4, 6, 7, 8, 9, 0])
    assert np.isnan(feat["source_rgb_difference"][0, 7]) and feat["c_front"][0, 7] > 0.1
    assert feat["source_render_residual"][0, 7] == 0
    feat = extract_features(src["rgb"], src, tar, cam, cam, False)
    np.testing.assert_array_equal(score_features(feat, False)["l0_reason"][0], [1, 2, 3, 3, 3, 3, 3, 3])
    assert not feat["scoreable"].any() and feat["frame_unprojectable"].all()
    assert np.isfinite(score_features(feat, False)["alpha_score"][0, 2])


def deployable(root):
    """Only the scorer inputs exist at this point. Ground truth is constructed separately."""
    p = root / "prediction"
    p.mkdir()
    c = camera()
    views = {v: {"width": c.width, "height": c.height, "K": c.K.tolist(), "c2w": c.c2w.tolist()} for v in VIEWS}
    write_json(root / "cameras.json", {"schema": "el-camera-1", "pixel_center": "integer", "unit": "m", "views": views})
    Image.fromarray(np.zeros((*c.shape, 3), np.uint8)).save(root / "input.png")
    source = arrays(c)
    source["z"][0, 0] = 1
    np.savez_compressed(p / "source.npz", **source)
    for v in TARGETS:
        target = arrays(c)
        target["rgb"][0, 0] = [0.3, 0, 0]
        target["rgb"][0, 2] = 0.3
        target["rgb"][1, 0] = np.nan
        target["rgb"][1, 1] = 0.05
        target["rgb"][1, 2] = [0.01, 0.02, 0.03]
        target["alpha"][0, 1:3] = [0.25, np.nan]
        target["z"][1, 1] = 1.8
        np.savez_compressed(p / f"{v}.npz", **target)
    meta = {"schema": "el-prediction-1", "producer_run_id": "synthetic-fixture-only",
            "rgb_encoding": "display_srgb_0_1", "depth_semantics": "expected_camera_z_m",
            "alpha_semantics": "accumulated_opacity", "camera_sha256": file_hash(root / "cameras.json"),
            "input_sha256": file_hash(root / "input.png"),
            "depth_validation": {"off_axis_plane_passed": True, "alpha_normalization_passed": True}}
    write_json(p / "prediction.json", meta)
    return p


def reference(root):
    directory = root / "private_reference"
    (directory / "reference").mkdir(parents=True)
    (directory / "validation").mkdir()
    write_json(directory / "validation" / "results.json", {"passed": True, "record_role": "synthetic_fixture_only"})
    c = camera()
    yy, xx = np.indices(c.shape)
    position = ((np.stack((xx, yy, np.ones(c.shape)), axis=-1) @ np.linalg.inv(c.K).T) * 2).astype(np.float32)
    position[1, 2] = np.nan
    z = position[..., 2].copy()
    views = {}
    for view in TARGETS:
        path = directory / "reference" / f"{view}.npz"
        np.savez_compressed(path, rgb=arrays(c)["rgb"], z=z, position=position,
                            object_id=np.array([[1, 1, 1], [2, 2, 0]], np.uint16),
                            visibility=np.array([[1, 2, 1], [1, 3, 0]], np.uint8),
                            edge=np.array([[False, True, True], [False, True, False]]))
        views[view] = {"reference_sha256": file_hash(path)}
    write_json(directory / "reference.json", {
        "schema": "el-reference-1", "input_sha256": file_hash(root / "input.png"),
        "scene_id": "synthetic-fixture-only", "scene_family_id": "synthetic-fixture-only",
        "rgb_encoding": "display_srgb_0_1", "depth_semantics": "positive_optical_axis_camera_z_m",
        "camera_sha256": file_hash(root / "cameras.json"), "views": views, "validation_passed": True,
        "validation_sha256": file_hash(directory / "validation" / "results.json")})
    return directory


IDENTITY = {key: "synthetic-fixture-only" for key in ("run_id", "trial_id", "scene_family_id", "scene_id", "condition_id")}


def all_metrics(out):
    return {(r["method_id"], r["metric_name"]): r for line in (out / "metrics.jsonl").read_text().splitlines()
            if (r := json.loads(line))["camera_id"] == "v01" and r["region_name"] == "all"}


def test_isolated_cli_and_independent_denominators(tmp_path):
    p = deployable(tmp_path)
    s, out = tmp_path / "scores", tmp_path / "evaluation"
    # Unrelated poison file must never be discovered/decoded. No reference exists in the subprocess inputs.
    (p / "unrelated.npz").write_bytes(b"deliberately invalid")
    guarded_main = """
import os, runpy, sys
from pathlib import Path
root = Path.cwd()
allowed = {root / "cameras.json", root / "input.png", root / "prediction" / "prediction.json"}
allowed.update(root / "prediction" / (v + ".npz") for v in ("source","v01","v02","v03","v04","v05","v06"))
def check(event, args):
    if event in ("os.listdir", "os.scandir") and isinstance(args[0], (str, bytes)):
        if Path(os.fsdecode(args[0])).resolve() == root / "prediction":
            raise AssertionError("must not discover files in prediction bundle")
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    if p.is_relative_to(root) and p not in allowed and not p.relative_to(root).parts[0].startswith(".scores-"):
        raise AssertionError("undeclared scoring file " + str(p))
sys.addaudithook(check)
runpy.run_module("evidence_lens.risk_model", run_name="__main__")
assert not any(k in sys.modules for k in ("evidence_lens.labels", "evidence_lens.evaluation", "torch", "bpy", "pyarrow"))
"""
    subprocess.run([sys.executable, "-c", guarded_main, "--input", str(tmp_path / "input.png"),
                    "--cameras", str(tmp_path / "cameras.json"), "--predictions", str(p), "--output", str(s)],
                   check=True, cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")))
    receipts = json.loads((s / "score.json").read_text())
    allowed = {"input_png", "cameras", "prediction_metadata"} | {f"prediction/{v}" for v in VIEWS}
    assert set(receipts["inputs"]) == allowed and not (tmp_path / "private_reference").exists()
    ref = reference(tmp_path)
    receipt = run_evaluation(tmp_path / "input.png", tmp_path / "cameras.json", ref, p, s, out, IDENTITY)
    assert receipt["completed_target_views"] == 6
    data = all_metrics(out)
    expected = {
        ("output", "rgb_error_fraction_gt_0.05"): (2, 5), ("output", "invalid_output_fraction"): (1, 6),
        ("L0", "high_risk_recall"): (1, 2), ("L0", "warning_precision"): (1, 2),
        ("L0", "low_risk_error_rate"): (0, 2), ("L0", "overall_low_risk_coverage"): (2, 6),
        ("L0", "unknown_fraction"): (1, 5), ("L0", "error_fraction_within_unknown"): (1, 1),
    }
    for name, counts in expected.items():
        assert (data[name]["numerator"], data[name]["denominator"]) == counts
        assert data[name]["value"] == pytest.approx(counts[0] / counts[1])
    with np.load(out / "labels" / "v01.npz") as labs:
        assert np.isclose(labs["rgb_error"][0, 0], 0.1)
        assert labs["rgb_high_error"][1, 0] == 255 and np.isnan(labs["rgb_error"][1, 0])
        assert labs["rgb_evaluable"][1, 2] and not labs["reference_surface"][1, 2]  # valid background
        assert labs["low_support"][0, 1] and not labs["depth_evaluable"][0, 1] and labs["rgb_evaluable"][0, 1]
        assert labs["depth_high_error"][1, 1] == 1 and labs["signed_depth_error"][1, 1] == pytest.approx(-0.2)
        assert labs["rgb_high_error"][1, 1] == 0  # strict > at 0.05
    with np.load(out / "curves" / "v01.npz") as curves:
        assert curves["l0_accepted"].tolist() == [0, 2, 4] and np.isnan(curves["l0_error_rate"][0])
    board = json.loads((out / "board_v01.json").read_text())
    assert board["board_sha256"] == file_hash(out / "board_v01.png")
    for name in ("v01_l0_state.png", "v01_rgb_error.png"):
        assert board["native_preview_sha256"][name] == file_hash(out / name)
    cross = json.loads((out / "reason_visibility_counts.json").read_text())["family"]
    assert sum(cell["count"] for cell in cross) == 36
    assert all(cell["reason_total"] >= cell["count"] <= cell["visibility_total"] for cell in cross)
    verify = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/recompute_view.py"),
              "--cameras", str(tmp_path / "cameras.json"), "--metadata", str(p / "prediction.json"),
              "--source-prediction", str(p / "source.npz"), "--prediction", str(p / "v01.npz"),
              "--reference", str(ref / "reference/v01.npz"), "--score", str(s / "v01.npz"),
              "--labels", str(out / "labels/v01.npz"), "--metrics", str(out / "metrics.csv")]
    subprocess.run([*verify, "--out", str(tmp_path / "verified")], check=True)
    proof = json.loads((tmp_path / "verified/independent-view-check.json").read_text())
    assert proof["scalar_l0_witnesses"] == proof["pixel_labels_checked"] == 6
    with np.load(s / "v01.npz") as original:
        corrupted_score = {key: original[key] for key in original.files}
    corrupted_score["l0"][0, 0] = 0
    np.savez(s / "v01.npz", **corrupted_score)
    rejected = subprocess.run([*verify, "--out", str(tmp_path / "corrupt")], capture_output=True, text=True)
    assert rejected.returncode and "stored L0 risk" in rejected.stderr
    with pytest.raises(FileExistsError):
        run_score(tmp_path / "input.png", tmp_path / "cameras.json", p, s)


def test_recorded_render_failure_has_real_denominators(tmp_path):
    deployable(tmp_path)
    out = tmp_path / "unavailable-evaluation"
    receipt = run_evaluation(tmp_path / "input.png", tmp_path / "cameras.json", reference(tmp_path),
                             None, None, out, IDENTITY, render_failed="synthetic_fixture_render_failure")
    assert receipt["planned_target_views"] == 6 and receipt["completed_target_views"] == 0
    data = all_metrics(out)
    assert data["output", "invalid_output_fraction"]["value"] == 1
    assert data["L0", "overall_low_risk_coverage"]["value"] == 0
    assert data["output", "rgb_mae"]["value"] is None and data["output", "rgb_mae"]["denominator"] == 0
    assert data["L0", "average_precision_common"]["missing_reason"] == "synthetic_fixture_render_failure"
    assert json.loads((out / "board_v01.json").read_text())["native_preview_sha256"] == {}


def test_risk_tile_and_native_previews_distinguish_unknown_from_invalid_rgb(tmp_path):
    cam = camera()
    source, target = arrays(cam), arrays(cam)
    yy, xx = np.indices(cam.shape)
    pos = ((np.stack((xx, yy, np.ones(cam.shape)), axis=-1) @ np.linalg.inv(cam.K).T) * 2).astype(np.float32)
    ref = dict(rgb=source["rgb"].copy(), z=source["z"], position=pos, edge=np.zeros(cam.shape, bool),
               visibility=np.ones(cam.shape, np.uint8))
    ref["rgb"][0, 1] = np.nan
    target["rgb"][1, 0] = np.nan
    target["rgb"][1, 2] = 0.3
    target["alpha"][:] = [[1, 0.25, np.nan], [0.25, 0.25, 1]]
    scores = score_features(extract_features(source["rgb"], source, target, cam, cam, True), True)
    labs = label_view(ref, target, True)
    render_board(tmp_path / "board.png", source["rgb"], ref, target, labs, scores, "synthetic-fixture-only")
    assert scores["l0"][0, 1] == scores["l0"][1, 0] == 1  # the raw diagnostic is preserved
    with Image.open(tmp_path / "v01_l0_state.png") as state:
        assert state.mode == "P"
        np.testing.assert_array_equal(np.asarray(state), [[0, 255, 2], [255, 1, 0]])
        assert state.convert("RGB").getpixel((2, 0)) == RISK_COLORS["unknown"]
        assert state.convert("RGB").getpixel((1, 0)) == INVALID_COLOR
    with Image.open(tmp_path / "v01_rgb_error.png") as err:
        np.testing.assert_array_equal(np.asarray(err), [[0, 255, 0], [255, 0, 254]])
    with Image.open(tmp_path / "board.png") as board:
        assert board.getpixel((800, 845)) == INVALID_COLOR
        assert board.getpixel((970, 845)) == RISK_COLORS["unknown"]


def test_partial_source_target_failure_and_typed_output(tmp_path):
    p, ref = deployable(tmp_path), reference(tmp_path)
    meta = json.loads((p / "prediction.json").read_text())
    meta["failed_views"] = {"source": "fixture_source_failure", "v06": "fixture_target_failure"}
    for v in meta["failed_views"]:
        (p / f"{v}.npz").unlink()
    meta["views"] = {v: {"sha256": file_hash(p / f"{v}.npz")} for v in VIEWS if v not in meta["failed_views"]}
    write_json(p / "depth-probe.json", {"role": "synthetic-fixture-only", "depth_checks": meta["depth_validation"]})
    meta["depth_validation_receipt_sha256"] = file_hash(p / "depth-probe.json")
    write_json(p / "prediction.json", meta)
    s, out = tmp_path / "precreated_score", tmp_path / "precreated_evaluation"
    s.mkdir()
    out.mkdir()
    run_score(tmp_path / "input.png", tmp_path / "cameras.json", p, s)
    parquet = find_spec("pyarrow") is not None
    receipt = run_evaluation(tmp_path / "input.png", tmp_path / "cameras.json", ref, p, s, out, IDENTITY, parquet=parquet)
    assert receipt["completed_target_views"] == 5
    assert receipt["depth_validation_receipt_sha256"] == meta["depth_validation_receipt_sha256"]
    assert receipt["views"]["v06"]["render_failure_reason"] == "fixture_target_failure"
    assert all_metrics(out)["L0", "unknown_fraction"]["value"] == pytest.approx(4 / 5)
    if parquet:
        import pyarrow.parquet as pq
        from evidence_lens._metrics import SCHEMA
        table = pq.read_table(out / "metrics.parquet")
        assert table.schema.equals(SCHEMA) and table.num_rows == len((out / "metrics.jsonl").read_text().splitlines())
    with (p / "v01.npz").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="digest differs from prediction.json"):
        run_score(tmp_path / "input.png", tmp_path / "cameras.json", p, tmp_path / "wrong-prediction")


def test_linkage_and_schema_rejects_wrong_bytes(tmp_path):
    p = deployable(tmp_path)
    ref = reference(tmp_path)
    cam_path, image_path = tmp_path / "cameras.json", tmp_path / "input.png"
    s = tmp_path / "scores"
    run_score(image_path, cam_path, p, s)
    with (s / "v02.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="score hash"):
        run_evaluation(image_path, cam_path, ref, p, s, tmp_path / "bad", IDENTITY)
    assert not (tmp_path / "bad").exists()
    audit = ReadAudit()
    cameras = load_cameras(cam_path, audit)
    load_source_png(image_path, cameras["source"], audit)
    bundle = PredictionBundle(p, audit)
    a = arrays(camera())
    np.savez(p / "v04.npz", **a, truth=np.array([1], np.uint8))
    with pytest.raises(ValueError, match="exactly"):
        bundle.read("v04", cameras["v04"])
    a["alpha"][0, 0] = 0
    np.savez(p / "v03.npz", **a)
    with pytest.raises(ValueError, match="zero-alpha"):
        bundle.read("v03", cameras["v03"])
    ref_meta = json.loads((ref / "reference.json").read_text())
    ref_meta["validation_passed"] = False
    write_json(ref / "reference.json", ref_meta)
    with pytest.raises(ValueError, match="validation"):
        ReferenceBundle(ref, audit)
    # A visually identical PNG with different bytes is a different source asset.
    with image_path.open("ab") as f:
        f.write(b"different-input-bytes")
    with pytest.raises(ValueError, match="input_sha256"):
        run_score(image_path, cam_path, p, tmp_path / "wrong-source")


def test_whole_ties_and_single_class_ranking():
    values = np.array([0.8, 0.8, 0.2, 0.2], np.float32)
    thresholds, n, pos = _groups(values, np.array([1, 0, 0, 1]))
    np.testing.assert_allclose(thresholds, [0.2, 0.8])
    np.testing.assert_array_equal(n, [2, 2])
    assert _ap(n, pos) == 0.5 and _ap(n, n) is None and _ap(n, np.zeros_like(n)) is None
