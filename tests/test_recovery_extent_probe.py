"""Check the units, denominators and training-state alignment of the CPU diagnostic."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_extent_probe import inspect_state, native_log
from r0_geometry import covariance


def test_isotropic_radius_in_pixels_and_clip_domain():
    z = [1, 2, 0.01]
    scale = np.full((3, 3), 0.5)
    state = {
        "means": np.array([[0, 0, v] for v in z]),
        "log_scales": np.log(scale),
        "quats": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        "opacity_logits": np.zeros(3),
    }
    cam = {
        "K": [[100, 0, 127.5], [0, 100, 127.5], [0, 0, 1]],
        "c2w": np.eye(4),
        "clip_m": [0.05, 30],
        "width": 256,
        "height": 256,
    }
    row, extremes = inspect_state(state, cam, scale, covariance(state), 0)
    assert row["ellipse_intersects_n"] == 2
    assert row["ellipse_intersects_major_gt64"] == 2
    assert row["ellipse_intersects_major_gt256"] == 0
    assert row["ellipse_intersects_major_px_max"] == pytest.approx(150)
    assert row["positive_at_or_before_near"] == 1
    assert all(x["checkpoint_row"] == 0 and x["camera_z_m"] == 1 for x in extremes)


def test_native_draw_precedes_its_adam_update_and_prefers_earlier_tie():
    trace = [
        {"source": s, "step": i, "max_radius_px": i, "visible": 3}
        for i, s in enumerate(("s00", "s03", "s00", "s03", "s00", "s03"), 1)
    ]
    assert native_log(trace, "s00", 0)["native_state_step_minus_checkpoint"] == 0
    assert native_log(trace, "s03", 0)["native_forward_logged_update"] == 2
    assert native_log(trace, "s03", 4)["native_state_step_minus_checkpoint"] == -1
    assert native_log(trace, "s00", 6)["native_forward_state_step"] == 4
