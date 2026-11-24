"""Analytic controls for the post-opening comparison. These tests use generated numeric arrays."""

from types import SimpleNamespace

import numpy as np
import pytest

from evidence_lens.scoring_io import Camera
from evidence_lens_bundle.bindings import array_sha
from v05_bell_common import OLD, camera_on_grid
from v05_bell_evaluate import area_rgb, labels, nearest, quota_rows, verify_frustum


def test_area_reference_includes_fractional_edges_and_masks_unknown():
    values = np.broadcast_to(np.arange(7, dtype=np.float32)[None, :, None], (3, 7, 3)).copy()
    reduced = area_rgb(values, (2, 1))
    # Each output spans three complete pixels and half of the middle pixel.
    np.testing.assert_allclose(reduced[0, :, 0], [4.5 / 3.5, 16.5 / 3.5], rtol=1e-7)
    values[0, 1, 0] = np.nan
    masked = area_rgb(values, (2, 1))
    assert np.isnan(masked[0, 0]).all() and np.array_equal(masked[0, 1], reduced[0, 1])
    a = np.arange(35).reshape(5, 7)
    np.testing.assert_array_equal(nearest(a, (2, 3)), a[np.ix_([0, 2, 4], [1, 5])])


def test_distinct_pixel_labels_strict_boundary_and_quota_ties():
    measured = np.zeros((3, 4, 3), np.float32)
    p = measured.copy()
    p[0, 0] = 0.05
    p[2, 1] = 0.2
    q = measured.copy()
    q[1, 2] = 0.2
    err, valid, high = labels(p, measured)
    assert err.mean() < 0.05 and high.sum() == 1 and not high[0, 0] and high[2, 1]
    assert not np.array_equal(high, labels(q, measured)[2])
    valid[0, 0] = False
    scores = np.flatnonzero(valid)
    risk = np.ones_like(err)
    risk[2, 1] = 0.25
    rows, chosen = quota_rows(risk, valid, high, "test", example=True)
    assert rows[0]["eligible_pixels"] == 11 and rows[0]["selected_pixels"] == 1
    assert rows[1]["selected_pixels"] == 2 and rows[1]["selected_errors"] == 1
    assert np.array_equal(np.flatnonzero(chosen), [scores[0], 9])
    assert rows[1]["selected_row_major_indices_sha256"] == array_sha(np.array([1, 9], dtype="<i8"))
    none, _ = quota_rows(risk, valid & False, high, "empty")
    assert none[0]["selected_error_rate"] is None and none[0]["selected_pixels"] == 0


def test_same_frustum_detects_padding_or_pose_shift():
    original = {
        "sensor_fit": "HORIZONTAL",
        "lens_shift": [0, 0],
        "pixel_aspect": [1, 1],
        "width": 256,
        "height": 256,
        "lens_mm": 41.0,
        "sensor_width_mm": 32.0,
        "clip_m": [0.1, 200.0],
        "c2w": np.eye(4).tolist(),
    }
    reference = camera_on_grid(original, 896, 504)
    bundles = {}
    for tag, w, h in (("landscape", 448, 252), (OLD, 256, 144)):
        expected = camera_on_grid(original, w, h)
        k = np.array(expected["K"])
        if tag == OLD:
            k[1, 2] += 56
            h += 112
        cam = Camera(w, h, k, np.eye(4))
        grids = {
            "views": {
                "v01": {
                    "original_size": [896, 504],
                    "crop_xywh": [0, 0, 896, 504],
                    "original_K": reference["K"],
                }
            }
        }
        bundles[tag] = (SimpleNamespace(cameras={"v01": cam}, grids=grids), "")
    assert max(verify_frustum(reference, bundles, "v01")["max_K_residual_pixels"].values()) < 1e-12
    bundles[OLD][0].cameras["v01"].K[1, 2] += 1
    with pytest.raises(ValueError, match="same physical frustum"):
        verify_frustum(reference, bundles, "v01")
