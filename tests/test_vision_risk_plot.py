"""A discrete baseline can reach one nonempty coverage and an empty set has no error rate."""
from copy import deepcopy

import pytest

from plot_vision_risk_coverage import coordinates


def test_operating_points_preserve_ties_and_undefined_risk():
    points = []
    for i in range(0, 101, 5):
        low = 0 if i <= 50 else 20
        points.append({"requested_coverage": i / 100, "accepted": low,
                       "positive": 4 if low else 0, "V": 20, "coverage": low / 20,
                       "risk": .2 if low else None})
    curve = {"points": points, "frozen_L": 20, "frozen_LY": 4}
    assert coordinates(curve) == ([(1., .2)], (1., .2))
    assert coordinates({**curve, "frozen_L": 0, "frozen_LY": 0}) == ([(1., .2)], None)
    wrong = deepcopy(curve)
    wrong["points"][0]["risk"] = 0.
    with pytest.raises(ValueError, match="denominator"):
        coordinates(wrong)
