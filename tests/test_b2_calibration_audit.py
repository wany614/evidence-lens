from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b2_calibration_audit import direct_histograms, verify_weighted_quantiles


def test_histogram_preserves_strict_repeated_cuts_and_rational_object_weights():
    low = float(np.float32(0.1))
    high = float(np.float32(0.8))
    nlow = np.nextafter(low, np.inf)
    nhigh = np.nextafter(high, np.inf)
    a = np.array([low, low, high], np.float64)
    data = {"scored": a, "events": np.array([True, False, True]), "deploy": a}
    assert direct_histograms(data, [low, nlow, nlow, high, nhigh]) == (
        [0, 2, 2, 2, 3],
        [0, 1, 1, 1, 2],
    )
    solo = {"deploy": np.array([high])}
    correct = [low] + [nlow] * 66 + [nhigh] * 134
    verify_weighted_quantiles([data, solo], correct)
    faulty = correct.copy()
    faulty[67] = nlow
    with pytest.raises(ValueError):
        verify_weighted_quantiles([data, solo], faulty)
