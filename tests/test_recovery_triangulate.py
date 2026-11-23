"""Geometry controls for the preregistered, source-only R2 initializer."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
import recovery_triangulate as geometry


def camera_document(ids, *, baseline=0.3, rotate=True):
    """Known metric world with a noncentral principal point and distinct yaw angles."""
    views = {}
    for source in ids:
        fraction = int(source[-1]) / 3
        yaw = np.deg2rad((0.5 - fraction) * 8) if rotate else 0.0
        c, s = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4)
        pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        pose[:3, 3] = [(fraction - 0.5) * baseline, fraction * 0.017 if rotate else 0, 0.0]
        views[source] = {
            "K": [[236, 0, 121.2], [0, 219, 135.8], [0, 0, 1]],
            "c2w": pose.tolist(),
            "clip_m": [0.01, 50],
            "width": 256,
            "height": 256,
        }
    return {
        "schema": "el-camera-1",
        "unit": "m",
        "pixel_center": "integer",
        "axis_convention": "OpenCV x right, y down, z forward",
        "views": views,
    }


def project(c, point):
    vector = c["K"] @ (c["w2c"] @ np.r_[point, 1])
    return vector[:2] / vector[2]


def test_source_camera_set_cannot_include_unused_pose():
    ids = geometry.SETS["M2"]
    with pytest.raises(ValueError, match="Unauthorized source"):
        geometry.cameras(camera_document(geometry.SETS["M4"]), ids)


def test_joint_dlt_offaxis_and_depth_parallax_gates():
    ids = geometry.SETS["M4"]
    cams = list(geometry.cameras(camera_document(ids), ids).values())
    point = np.array([0.175, -0.071, 2.12])
    obs = [(i, project(c, point)) for i, c in enumerate(cams)]
    got, reason = geometry.point_from_track(obs, cams)
    assert reason is None and np.allclose(got["point"], point, atol=1e-12)
    assert max(got["errors"]) < 1e-10 and min(got["depths"]) > 0 and got["parallax_deg"] > 2
    F = geometry.fundamental(cams[0], cams[3])
    assert abs(np.r_[obs[3][1], 1] @ F @ np.r_[obs[0][1], 1]) < 1e-12
    bad = [*obs[:-1], (3, obs[-1][1] + np.array([0, 10]))]
    assert geometry.point_from_track(bad, cams)[1] == "reprojection_error"
    for point, reason in [
        ([0.1, 0, -2], "outside_positive_source_clip"),
        ([0.1, 0, 15], "insufficient_parallax"),
    ]:
        assert (
            geometry.point_from_track([(i, project(c, point)) for i, c in enumerate(cams)], cams)[1]
            == reason
        )


def test_alias_and_conflicting_tracks_are_not_new_points():
    coords = [
        cv2.KeyPoint(10.0, 15.0, 2.0, 0.0, 0.3),
        cv2.KeyPoint(10.0, 15.0, 3.0, 90.0, 0.6),
        cv2.KeyPoint(22.0, 11.0, 4.0, 2.0, 0.2),
    ]

    class Detector:
        def detectAndCompute(self, image, mask):
            return coords, np.zeros((3, 128), np.float32)

    observed = geometry.measure_features(np.zeros((256, 256, 3), np.uint8), Detector())
    assert observed["unique"] == 2 and observed["alias"] == [1, 1, 2]
    A, B, C, X = (0, 1), (1, 2), (2, 4), (1, 9)
    edges = [(0.10, 0.0, 1.0, A, B), (0.20, 0.0, 2.0, A, C), (0.25, 0.0, 3.0, B, C), (0.30, 0.0, 4.0, C, X)]
    tracks, audit = geometry.build_tracks(edges)
    assert (
        tracks == [(A, B, C)]
        and audit["ambiguous_component_edges_rejected"] == 1
        and audit["redundant_compatible_edges"] == 1
    )
    first = {
        "point": np.array([0.0, 0.0, 1.0]),
        "observations": [1, 2, 3],
        "max_error": 0.1,
        "ids": (A, B, C),
    }
    second = {**first, "point": np.array([0.0, 0.0, 1.00002]), "observations": [1, 2], "max_error": 0.0}
    assert geometry.unique_tracks([second, first], 0.00005) == [first]


def test_reciprocal_ratio_ties_and_symmetric_two_line_error():
    ids = geometry.SETS["M2"]
    rig = geometry.cameras(camera_document(ids, rotate=False), ids)
    points = [[0.1, 0, 1.2], [-0.12, 0.08, 1.8], [0.0, -0.12, 1.1]]
    features = []
    for s in ids:
        desc = np.zeros((3, 128), np.float32)
        desc[range(3), range(3)] = 10
        features.append(
            {"xy": np.array([project(rig[s], p) for p in points]), "desc": desc, "alias": [0, 1, 2]}
        )
    features[1]["xy"][2, 1] += 4
    edges, stats = geometry.pair_edges(0, 1, *features, rig["s00"], rig["s03"])
    assert stats["reciprocal"] == 3 and len(edges) == 2
    duplicate = np.vstack([features[1]["desc"][0], features[1]["desc"][0]])
    accepted, _ = geometry.knn(features[0]["desc"][:1], duplicate, cv2.BFMatcher(cv2.NORM_L2))
    assert accepted == {}


def test_qualification_uses_independent_count_and_each_allowed_source():
    ids = geometry.SETS["M2"]
    obs = [
        {
            "observations": [
                (0, np.array([10 + (n % 4) * 64, 20.0])),
                (1, np.array([10.0, 15 + (n % 4) * 64])),
            ]
        }
        for n in range(24)
    ]
    assert geometry.eligibility(obs, ids)["qualified"]
    assert not geometry.eligibility(obs[:-1], ids)["qualified"]
    for item in obs[7:]:
        item["observations"].pop()
    result = geometry.eligibility(obs, ids)
    assert "s03_fewer_than_8_tracks" in result["failed_gates"]


def test_parallel_endpoint_fallback_is_metric_and_shared_between_conditions():
    m2, m4 = geometry.SETS["M2"], geometry.SETS["M4"]
    rig = geometry.cameras(camera_document(m4, rotate=False), m4)
    points2, report = geometry.random_prior(rig, m2, 0.3)
    points4, other = geometry.random_prior(rig, m4, 0.3)
    a, b = ([v["point"] for v in pts] for pts in [points2, points4])
    assert np.array_equal(a, b) and report == other and len(np.unique(a, axis=0)) == 2048
    assert report["method"] == "three_times_endpoint_baseline"
    for s in m2:
        assert report["sources"][s]["focus_m"] == pytest.approx(0.9)
        assert report["sources"][s]["positive_interval_m"] == pytest.approx([0.525, 1.275])
    images = {s: np.full((256, 256, 3), 128, np.uint8) for s in m2}
    arrays, stats = geometry.parameters(points2, images, rig, m2, 0.3)
    assert arrays["means"].dtype == np.dtype("<f4") and arrays["means"].shape == (2048, 3)
    assert np.allclose(1 / (1 + np.exp(-arrays["color_logits"])), 0.2158605, atol=1e-6)
    assert np.allclose(1 / (1 + np.exp(-arrays["opacity_logits"])), 0.1)
    assert stats["sigma_m"]["min"] > 0


def test_worker_rejects_target_role_before_opening_photographs(tmp_path, monkeypatch):
    import json
    import importlib.metadata

    for name, version in {
        "opencv-python-headless": "4.11.0.86",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "pillow": "11.3.0",
    }.items():
        if importlib.metadata.version(name) != version:
            pytest.skip("Pinned geometry environment required")
    rules = Path(__file__).resolve().parents[1] / "protocol/recovery-initialization-v1.json"
    cameras = tmp_path / "cameras.json"
    cameras.write_text(json.dumps(camera_document(geometry.SETS["M2"])))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "condition_id": "M2",
                "initialization_protocol_sha256": geometry.PROTOCOL_SHA,
                "generator_source_ids": list(geometry.SETS["M2"]),
                "source_camera_sha256": geometry.sha(cameras),
            }
        )
    )
    a = SimpleNamespace(
        protocol=rules,
        config=config,
        cameras=cameras,
        source=[("s00", "missing"), ("s03", "missing"), ("v01", "unread")],
        out=tmp_path,
    )
    with pytest.raises(ValueError, match="Extra source or target"):
        geometry.run(a)
