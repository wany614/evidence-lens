"""Original engineering fixtures for calibrated import and threshold applicability."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b2_asset_fixture import build
from b2_asset_windows import basis
from evidence_lens_asset.colmap import pose
from evidence_lens_asset.gaussian import load as read_gaussians
from evidence_lens_asset.prepare import inspect
from evidence_lens_bundle.common import Unsupported, profile, sha, write
from evidence_lens_bundle.calibration import declarations, select, states
from evidence_lens_bundle.bundle import Bundle
from test_bundle import fixture


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("beads")
    build(root)
    return root


@pytest.mark.parametrize("fmt", ("text", "binary"))
@pytest.mark.parametrize(
    "asset,p",
    (
        ("project.npz", "evidence-lens-npz-v1"),
        ("graphdeco-sh0.ply", "graphdeco-v1"),
        ("graphdeco-sh3.ply", "graphdeco-v1"),
    ),
)
def test_native_params_and_camera(demo, fmt, asset, p):
    stats, c, photos, a = inspect(
        demo / "colmap" / fmt, demo / "images", demo / asset, demo / "plan.json", fmt, p
    )
    assert (
        set(photos) == {"41", "5"}
        and stats["target_order"] == ["70", "109", "13"]
        and stats["gaussians"] == 145
    )
    gold = json.loads((demo / "native-cameras.json").read_bytes())
    for k, v in c["views"].items():
        np.testing.assert_allclose(v["c2w"], gold[k]["c2w"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(v["K"], gold[k]["K"], atol=1e-8, rtol=0)
    with np.load(demo / "native.npz", allow_pickle=False) as original:
        np.testing.assert_allclose(a["means"], original["centers_m"], rtol=0, atol=1e-7)
        if "sh" in a:
            np.testing.assert_array_equal(a["sh"], original["sh"][:, : a["sh"].shape[1]])


def test_sh_basis_axes():
    x = basis(np.eye(3), 3)
    np.testing.assert_allclose(x[:, 0], 1 / np.sqrt(4 * np.pi))
    np.testing.assert_allclose(
        x[:, :4][:, 1:], np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]) * np.sqrt(3 / (4 * np.pi)), atol=1e-14
    )
    for l in range(4):
        np.testing.assert_allclose(
            (x[:, l * l : (l + 1) ** 2] ** 2).sum(axis=1), (2 * l + 1) / (4 * np.pi), atol=1e-14
        )


def test_strict_threshold_float64_neighbour():
    a = np.array([[0.5, np.nan]], np.float32)
    t = float(np.nextafter(0.5, np.inf))
    assert states(a, 0.5).tolist() == [[1, 255]] and states(a, t).tolist() == [[0, 255]]


def test_unknown_asset_type_or_scale(demo, tmp_path):
    f = tmp_path / "pc.ply"
    f.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
    )
    with pytest.raises(Unsupported, match="point-cloud"):
        read_gaussians(f, "graphdeco-v1", 1.0)
    for invalid in (0, float("nan"), -1.0):
        with pytest.raises(Unsupported, match="scale"):
            read_gaussians(demo / "project.npz", "evidence-lens-npz-v1", invalid)


def test_camera_right_down_forward():
    pose_m = pose([1, 0, 0, 0], [100, 200, 0], 0.01)
    np.testing.assert_array_equal(np.asarray(pose_m)[:3, 3], [-1, -2, 0])


def make_profile():
    model = profile()["models"]["MS-L1"]
    h = "e" * 64
    scope = {
        "layout": "DTU-fixed",
        "primary_source_id": "s00",
        "lighting_id": 3,
        "resolution": [256, 256],
        "rgb_encoding": "display_srgb_0_1",
        "depth_semantics": "expected_camera_z_m",
        "alpha_semantics": "accumulated_opacity",
        "recipe_sha256": h,
        "stop_step": 5000,
        "renderer": {"name": "gsplat", "version": "1.5.3"},
        "reconstructor": {
            "name": "fixture",
            "optimizer_seed": 1,
            "gaussian_cap": 99,
            "worker_tree_sha256": h,
        },
        "release_index_sha256": profile()["release_index_sha256"]
        if "release_index_sha256" in profile()
        else "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d",
        "model_sha256": model["weights_sha256"],
        "model_index_sha256": model["model_index_sha256"],
        "feature_schema_sha256": model["feature_schema_sha256"],
        "preprocessing_sha256": model["preprocessing_sha256"],
        "feature_columns": 39,
        "source_ids": ["s00", "s03"],
        "source_camera_sha256": h,
        "render_camera_sha256": h,
    }
    return {
        "schema": "el-b2-calibration-profile-1",
        "profile_version": 1,
        "method": "MS-L1",
        "condition": "M2",
        "status": "empirical_calibration_only",
        "tau": 0.2,
        "comparison": "score < tau",
        "original_tau": model["threshold"],
        "epsilon": 0.05,
        "target": {
            "maximum_mean_low_error": 0.05,
            "minimum_mean_coverage": 0.2,
            "require_every_object_nonempty": True,
        },
        "scope": scope,
        "evaluated_on_holdout": False,
        "holdout_target_met": None,
    }


def test_scope_refuses_external_or_wrong_receipt(tmp_path):
    b = Bundle(fixture(tmp_path / "bundle"))
    f = tmp_path / "p.json"
    j = make_profile()
    write(f, j)
    h = sha(f.read_bytes())
    result = select(b, "MS-L1", f, h, None, None)
    assert not result["applied"] and result["status"] == "outside_calibration_scope"
    with pytest.raises(ValueError, match="SHA-256 differs"):
        select(b, "MS-L1", f, "0" * 64, None, None)


def test_scope_matching_and_no_feasible_explicit(tmp_path):
    from types import SimpleNamespace

    j = make_profile()
    h = "d" * 64
    b = SimpleNamespace(
        sha256=h,
        condition="M2",
        order=("s00", "s03"),
        primary="s00",
        meta={
            "creation": {"original_case_sha256": h},
            "camera": {"sha256": h},
            "sources": {v: {"sha256": h} for v in ("s00", "s03")},
        },
        producer={"checkpoint_sha256": h, "frames": {}},
    )
    pin = {"run_id": "fixture", "manifest_sha256": h, "sha256": h, "member": "payload/x.json"}
    proof = {
        "schema": "el-b2-deployment-scope-1",
        "verified": True,
        "condition": "M2",
        "scope": j["scope"],
        "bundle": declarations(b),
        "origins": dict.fromkeys(("source", "cohort", "freeze", "readback", "case"), pin),
    }
    f, s = tmp_path / "p.json", tmp_path / "s.json"
    write(s, proof)
    for status, tau, active in (
        ("empirical_calibration_only", 0.2, True),
        ("no_feasible_operating_point", None, False),
    ):
        j.update(status=status, tau=tau)
        write(f, j)
        o = select(b, "MS-L1", f, sha(f.read_bytes()), s, sha(s.read_bytes()))
        assert o["applied"] == active and o["status"] == status
    proof["bundle"]["checkpoint_sha256"] = "a" * 64
    write(s, proof)
    with pytest.raises(ValueError, match="different bundle"):
        select(b, "MS-L1", f, sha(f.read_bytes()), s, sha(s.read_bytes()))


def test_report_renders_and_rechecks_synthetic_calibrated_map(tmp_path):
    """An artificial display receipt checks the HTML-active branch before any empirical threshold is loaded."""
    from evidence_lens_bundle.calibration import state_counts
    from evidence_lens_bundle.common import row
    from evidence_lens_bundle.scoring import score
    from evidence_lens_bundle.report import render

    root = fixture(tmp_path / "bundle")
    output = tmp_path / "score"
    result = score(root, output)
    assert result["calibration_profile"] is None
    threshold = 0.5
    totals = {"canvas": 0, "low": 0, "high": 0, "unknown": 0}
    for v in result["views"]:
        fp = output / v["arrays"]["path"]
        with np.load(fp, allow_pickle=False) as data:
            chosen = states(data["risk"], threshold)
        f = fp.with_name("calibrated-state.npy")
        np.save(f, chosen, allow_pickle=False)
        quantities = state_counts(chosen)
        v["calibrated"] = {
            "counts": quantities,
            "state": row(f.relative_to(output).as_posix(), f.read_bytes()),
        }
        for k, n in quantities.items():
            totals[k] += n
    result["calibration_profile"] = {
        "status": "empirical_calibration_only",
        "profile_sha256": "4" * 64,
        "scope_sha256": "3" * 64,
        "applied": True,
        "threshold": threshold,
        "holdout_target_met": None,
        "counts": totals,
    }
    write(output / "scores.json", result)
    html = tmp_path / "view.html"
    render(output, html)
    assert html.read_text().count("Empirical threshold state</figcaption>") == 2
    result["calibration_profile"]["counts"]["low"] += 1
    write(output / "scores.json", result)
    with pytest.raises(ValueError, match="overall calibrated counts"):
        render(output, tmp_path / "bad.html")
