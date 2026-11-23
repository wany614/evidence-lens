from PIL import Image
import numpy as np
from stage3_risk_caption import mapped, state_colors
from stage3_risk_boards import draw_header, heat, state


def test_raster_legend_distinguishes_unknown_invalid_and_threshold():
    values = np.array([[0.0, 0.25, 0.5, 0.75, 1.0, 3.0, np.nan, np.nan]], np.float32)
    eval_mask = np.array([[1, 1, 1, 1, 1, 1, 1, 0]], bool)
    colors = mapped(values, eval_mask, 1.0)
    assert colors.tolist()[0] == [
        [23, 71, 96],
        [37, 138, 151],
        [137, 200, 162],
        [245, 205, 123],
        [207, 63, 68],
        [207, 63, 68],
        [113, 118, 127],
        [127, 55, 150],
    ]
    assert np.array_equal(colors, heat(values, 1.0, eval_mask))
    sc = state_colors(values, eval_mask, 0.5)
    assert sc.tolist()[0][0:3] == [[75, 168, 184], [75, 168, 184], [230, 99, 67]]
    assert np.array_equal(sc, state(values, 0.5, eval_mask))
    bmp = Image.new("RGB", (2114, 1278), (0, 1, 2))
    draw_header(bmp, "fixture", "M2", "v01", 256, 256, "ready")
    arr = np.asarray(bmp)
    assert (arr[90:] == [0, 1, 2]).all()
    left = bmp.width - 368
    assert list(arr[55, left]) == [75, 168, 184] and list(arr[55, left + 220]) == [127, 55, 150]
