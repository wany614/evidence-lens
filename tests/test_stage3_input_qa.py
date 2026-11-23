import numpy as np
import pytest
from stage3_risk_input_qa import bits, independent_y, indices


def test_independent_rgb_boundary_and_bitwise_input_identity():
    reference = np.zeros((1, 4, 3), np.float32)
    rendered = reference.copy()
    reference[0, 0, 0] = np.nan
    rendered[0, 1, :] = np.float32(.05)
    rendered[0, 2, :] = np.float32(.05000001)
    rendered[0, 3, 0] = np.nan
    v, e, y = independent_y(reference, rendered)
    assert v.tolist() == [[False, True, True, True]]
    assert e.tolist() == [[False, True, True, False]] and y.tolist() == [[255, 0, 1, 255]]
    bits(rendered, rendered.copy(), "source RGB")
    other = rendered.copy()
    other[0, 1, 0] = -0.
    with pytest.raises(ValueError, match="source RGB"):
        bits(rendered, other, "source RGB")
    _, missing, label = independent_y(reference, None)
    assert not missing.any() and (label == 255).all()


def test_uniform_supervised_coordinates_replay_conditions():
    mask = np.ones((100, 100), bool)
    mask[17, :] = False
    a = indices(mask, "fixture", "M2", "v01")
    assert a.dtype == np.int32 and len(a) == len(set(a)) == 4096 and np.all(np.diff(a) > 0)
    assert np.all(mask.ravel()[a]) and np.array_equal(a, indices(mask, "fixture", "M2", "v01"))
    assert not np.array_equal(a, indices(mask, "fixture", "M4", "v01"))
