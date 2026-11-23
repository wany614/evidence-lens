"""Local numerical regressions for the independent pinhole Gaussian readback."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from r0_geometry import covariance, ellipse_rect, project


def test_rotated_covariance_and_exact_ellipse_rectangle_contact():
    # A diagonal correlation gives an AABB that reaches the image corner while the ellipse misses it.
    centres = np.array([[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
    covs = np.array([[[0.25, -0.24], [-0.24, 0.25]], [[0.25, 0.24], [0.24, 0.25]], [[1.0, 0.0], [0.0, 1.0]]])
    val = ellipse_rect(centres, covs, (-1.0, 1.0, -1.0, 1.0))
    assert val[0] > 9 and val[1] < 9 and val[2] == 0
    s = {"log_scales": np.log([[0.2, 0.06, 0.01]]), "quats": np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])}
    assert np.allclose(covariance(s), np.array([np.diag([0.06**2, 0.2**2, 0.01**2])]), atol=1e-15)


def test_integer_coordinates_positive_camera_depth_and_offscreen_source():
    scene = {
        "means": np.array([[0.0, 0.0, 1.0], [0.35, 0.0, 1.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.01]]),
        "log_scales": np.log(np.full((4, 3), 0.05)),
        "quats": np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
    }
    cam = {
        "K": [[40.0, 0.0, 8.0], [0.0, 40.0, 8.0], [0, 0, 1.0]],
        "c2w": np.eye(4),
        "height": 17,
        "width": 17,
        "clip_m": [0.05, 30.0],
    }
    d = project(scene, covariance(scene), cam, 0.0)
    assert np.array_equal(d["inside_clip"], [True, True, False, False])
    assert np.allclose(d["centre_uv"][:2], [[8.0, 8.0], [22.0, 8.0]])
    assert d["ellipse_intersects"].tolist() == [True, True, False, False]
    assert d["centre_inside"].tolist() == [True, False, False, False]
    assert np.isnan(d["centre_uv"][2:]).all()


def test_roundoff_routes_apply_principal_offset_once_and_isolate_observed_depth():
    from r0_roundoff import pixel_routes

    cam = {
        "K": [[64.0, 0.0, 17.25], [0.0, 32.0, 19.75], [0.0, 0.0, 1.0]],
        "c2w": np.eye(4),
    }
    points = np.array([[0.25, -0.125, 0.5], [0, 0, 1]], dtype="f4")
    measured_z = np.array([0.25, 1], dtype="f8")
    expected = [[49.25, 11.75], [17.25, 19.75]]
    routes = pixel_routes(points, cam, measured_z)
    for name, (xy, z) in routes.items():
        if name == "fp64_with_observed_cuda_z":
            np.testing.assert_array_equal(xy, [[81.25, 3.75], [17.25, 19.75]])
            np.testing.assert_array_equal(z, measured_z)
        else:
            np.testing.assert_array_equal(xy, expected)
            np.testing.assert_array_equal(z, [0.5, 1.0])
