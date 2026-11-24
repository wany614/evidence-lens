"""Camera geometry and closed role lists for the authored v0.5 engineering comparison."""

import numpy as np

from evidence_lens_bundle.common import require

SOURCES = [f"s{i:02d}" for i in range(4)]
TARGETS = [f"v{i:02d}" for i in range(1, 7)]
SIZES = {"square": (256, 256), "landscape": (448, 252), "portrait": (252, 448), "photo": (378, 252), "odd": (375, 251)}
OLD = "legacy-letterbox"
HISTORICAL_RUN = "20261123T162553037969Z-c6c429e1d8704905"
HISTORICAL_MANIFEST = "746df66da8aee2cfdcf2de1981e731f9e488cd128a76458b4b4ea7e2eaa218fa"
CHECKPOINT = "eedb19d170445f22273333b8c7f7e014c160fc606bc3f9b12f7f0ed0f84a7fac"


def camera_on_grid(old, width, height):
    """Change only the raster resolution of the original physical horizontal-fit Blender camera."""
    require(old["sensor_fit"] == "HORIZONTAL" and old["lens_shift"] == [0, 0]
            and old["pixel_aspect"] == [1, 1] and old["width"] == old["height"] == 256,
            "original bell physical camera, sensor convention or lens shift differs")
    focal = old["lens_mm"] * width / old["sensor_width_mm"]
    k = [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]
    return {"width": width, "height": height, "K": k, "c2w": old["c2w"], "clip_m": old["clip_m"],
            "projection": "pinhole", "distortion": "none"}


def projections(originals):
    require(set(originals) == set(SOURCES + TARGETS), "unexpected source or target roster")
    out = {}
    for tag, (w, h) in SIZES.items():
        out[tag] = {"source_shape": tag, "analysis_size": [w, h], "active_xywh": [0, 0, w, h],
                    "original_views": {n: camera_on_grid(c, 2*w, 2*h) for n,c in originals.items()},
                    "render_views": {n: camera_on_grid(c, w, h) for n,c in originals.items()}}
        out[tag]["analysis_views"] = out[tag]["render_views"]
    cameras = {n: camera_on_grid(c, 256, 144) for n,c in originals.items()}
    square = {}
    for n,c in cameras.items():
        k = np.asarray(c["K"], np.float64).copy()
        k[1, 2] += 56
        square[n] = {**c, "height": 256, "K": k.tolist()}
    out[OLD] = {"source_shape": "landscape", "analysis_size": [256, 256], "active_xywh": [0, 56, 256, 144],
                "original_views": out["landscape"]["original_views"], "render_views": cameras, "analysis_views": square}
    return out
