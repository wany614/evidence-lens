import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evidence_lens_vision.model import RectangularFrame, rectangular_image_tensor, rectangular_sample


def test_partial_patches_count_only_observable_rgb():
    rgb = np.full((251, 375, 3), 0.25, dtype=np.float32)
    rgb[14:28, 14:28, 0] = np.nan
    image, support, frame, observed = rectangular_image_tensor(rgb, "cpu")
    assert frame.encoded_shape == (252, 378) and frame.patch_shape == (18, 27)
    assert torch.count_nonzero(image[:, :, :, 375:]) == 0
    assert torch.count_nonzero(image[:, :, 251:, :]) == 0
    assert torch.count_nonzero(image[:, :, 14:28, 14:28]) == 0
    assert observed.sum() == 251 * 375 - 196 and support[0, 0, 1, 1] == 0
    np.testing.assert_allclose(support[0, 0, -1, -1], 13 * 11 / 196)
    np.testing.assert_allclose(support[0, 0, 0, -1], 11 / 14)
    np.testing.assert_allclose(support[0, 0, -1, 0], 13 / 14)


def test_asymmetric_patch_values_sample_native_centres_with_border_clamp():
    frame = RectangularFrame(28, 42)
    patch_features = torch.tensor([[[[0.0, 1.0, 2.0], [100.0, 101.0, 102.0]]]])
    support = torch.ones(1, 1, 2, 3)
    indices = [0, 41, 27 * 42, 28 * 42 - 1, 13 * 42 + 13]
    sampled, fraction = rectangular_sample(patch_features, support, frame, indices)
    expected_inner = 101 * ((13 + 0.5) / 14 - 0.5)
    np.testing.assert_allclose(sampled[:, 0], [0, 2, 100, 102, expected_inner], rtol=2e-7)
    np.testing.assert_array_equal(fraction.numpy(), np.ones(5))
    support.zero_()
    sampled, fraction = rectangular_sample(patch_features, support, frame, indices)
    assert not sampled.any() and not fraction.any()
    with pytest.raises(ValueError, match="unpadded"):
        frame.patch_coordinates([42 * 28], "cpu")
