"""Test the independent RGB regression reader on tied scores, invalid RGB and an unknown."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
per_view = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/rgb_headline_regression.py"))[
    "per_view"
]


def test_ties_and_whole_population_operating_states():
    ref = np.zeros((1, 6, 3), np.float32)
    pred = ref.copy()
    pred[0, [0, 1, 4]] = 0.5
    pred[0, 5] = np.nan
    score = np.array([[1.0, 0.1, 1.0, 0.1, np.nan, 0.1]], np.float32)
    tau = float(np.float32(0.1))
    rows = per_view(
        ref, pred, {"tied": score, "constant": np.zeros((1, 6), np.float32)}, {"tied": tau, "constant": 0.5}
    )
    tied, constant = rows
    assert (tied["V"], tied["E"], tied["Y"], tied["S"], tied["SY"]) == (6, 5, 3, 4, 2)
    assert tied["common_ap"] == constant["common_ap"] == 0.5
    assert (tied["H"], tied["L"], tied["U"], tied["HY"], tied["LY"], tied["UY"]) == (4, 0, 1, 2, 0, 1)
    assert tied["missed_errors"] == 1 and tied["low_risk_error"] is None
    assert constant["low_coverage"] == 5 / 6 and constant["low_risk_error"] == 3 / 5
    pred[:] = 0
    assert per_view(ref, pred, {"tied": score}, {"tied": tau})[0]["common_ap"] is None
