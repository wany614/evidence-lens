import numpy as np
import pytest
from stage3_risk_replay import independent_counts


def test_independent_tie_group_and_threshold_state():
    pytest.importorskip("sklearn")
    v = np.array([[1, 1, 1, 1, 1, 1, 1]], bool)
    e = v.copy()
    e[0, -1] = False
    y = np.array([[0, 1, 0, 1, 1, 0, 255]], np.uint8)
    risk = np.array([[0.0, 0.25, 0.25, 0.8, np.nan, 0.8, 0.25]], np.float32)
    state = np.array([[0, 1, 1, 1, 255, 1, 1]], np.uint8)
    finite = e & np.isfinite(risk)
    c, a = independent_counts(v, e, y, risk, state, finite, 0.25)
    assert c["V"] == 7 and c["E"] == 6 and c["S"] == 5
    assert (c["HY"], c["LY"], c["UY"], c["L"], c["H"], c["U"], c["missed"]) == (2, 0, 1, 1, 4, 1, 1)
    assert a == {"own": 0.5, "common": 0.5}
    c, a = independent_counts(
        v, np.zeros_like(v), np.full(y.shape, 255, np.uint8), risk, state, np.zeros_like(v), 0.25
    )
    assert c["E"] == 0 and c["invalid"] == 7 and a == {"own": None, "common": None}
    wrong = state.copy()
    wrong[0, 1] = 0
    with pytest.raises(ValueError, match="threshold state"):
        independent_counts(v, e, y, risk, wrong, finite, 0.25)


@pytest.mark.parametrize("bad", ("e", "common", "labels"))
def test_reject_broken_native_domain(bad):
    pytest.importorskip("sklearn")
    v = np.array([[True, False]])
    e = np.array([[True, False]])
    cm = e.copy()
    y = np.array([[1, 255]], np.uint8)
    risk = np.array([[0.1, np.nan]], np.float32)
    state = np.array([[0, 255]], np.uint8)
    if bad == "e":
        e[0, 1] = True
    elif bad == "common":
        cm[0, 1] = True
    else:
        y[0, 0] = 255
    with pytest.raises(ValueError, match="native V/E/common"):
        independent_counts(v, e, y, risk, state, cm, 0.5)
