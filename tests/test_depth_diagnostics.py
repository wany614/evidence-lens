"""Original errors and domains on tiny explicit arrays, including native-camera and invalid-depth cases."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from evidence_lens.labels import label_view
from evidence_lens.scoring_io import Camera

ROOT = Path(__file__).resolve().parents[1]


def synthetic():
    z = np.ones((2, 4), np.float32) * 2
    rgb = np.ones((2, 4, 3), np.float32) * 0.25
    ref = {
        "rgb": rgb,
        "z": z,
        "position": np.ones((2, 4, 3), np.float32),
        "object_id": np.array([[1, 2, 3, 4], [5, 1, 2, 6]], dtype=np.uint16),
        "visibility": np.array([[1, 2, 3, 1], [2, 3, 1, 2]], np.uint8),
        "edge": np.zeros_like(z, bool),
    }
    pred = {
        "z": np.array([[1, 2, 3, -1], [1, 1, 1, 1]], np.float32),
        "rgb": rgb.copy(),
        "alpha": np.array([[0.5, 1, 0.75, 1], [np.nan, 0.499, 0.7, 1]], np.float32),
    }
    pred["rgb"][1, 2] = np.nan
    return ref, pred


def test_regions_depth_failure_counts_and_signed_ratios(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = runpy.run_path(str(ROOT / "scripts/depth_diagnostics.py"))
    ref, pred = synthetic()
    lab = label_view(ref, pred, True)
    region = mod["masks"](ref)
    d = mod["regional_statistics"](ref, pred, lab, "all", region["all"])
    assert (d["E"], d["surface"], d["Z"], d["Z_outside_E"]) == (7, 8, 5, 1)
    assert [d[k] for k in ("failed_invalid_predicted_z", "failed_invalid_alpha", "failed_low_alpha")] == [
        1,
        1,
        1,
    ]
    assert (d["z_negative"], d["z_zero"], d["z_positive"], d["ZY"]) == (3, 1, 1, 4)
    assert (
        d["z_mae_m"] == pytest.approx(4 / 5)
        and d["z_signed_mean_m"] == pytest.approx(-2 / 5)
        and d["ratio_q50"] == 0.5
    )
    empty = mod["regional_statistics"](ref, pred, lab, "empty", np.zeros_like(region["all"]))
    assert empty["Z"] == 0 and empty["z_mae_m"] is None and empty["ratio_q05"] is None
    assert np.all(
        region["foreground"].astype(int) + region["floor"] + region["back_wall"] + region["no_mesh"] == 1
    )
    failed = mod["regional_statistics"](ref, None, label_view(ref, None, False), "all", region["all"], False)
    assert (
        failed["failed_no_prediction"] == 8
        and failed["E"] == failed["Z"] == 0
        and failed["ratio_q50"] is None
    )
    false = mod["regional_statistics"](ref, pred, label_view(ref, pred, False), "all", region["all"], False)
    assert false["failed_unverified_semantics"] == 8 and false["Z"] == 0
    a = Camera(4, 2, np.eye(3), np.eye(4))
    b = Camera(2, 1, np.eye(3), np.eye(4))
    assert not mod["same_camera"](a, b, 1e-8)
