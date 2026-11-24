"""Finite v0.5 pinhole analysis grids with explicit pixel-centre and display metadata."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .common import BundleError, Unsupported, encoded, exact_keys, require


PROFILE_SIZES = {f"analysis-{w}x{h}-v1": (w, h) for w, h in
                 ((256, 256), (448, 252), (252, 448), (378, 252), (375, 251))}
PATCH = 14


def _size(width, height):
    require(all(type(x) is int and 0 < x <= 65536 for x in (width, height)),
            "positive integer image dimensions required")
    return width, height


def _intrinsics(value):
    try:
        k = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BundleError("expected a finite zero-skew pinhole intrinsic matrix") from exc
    require(k.shape == (3, 3) and np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0
            and k[0, 1] == k[1, 0] == 0 and np.array_equal(k[2], [0., 0., 1.]),
            "expected a finite zero-skew pinhole intrinsic matrix")
    return k


def center_crop(original_size, analysis_size):
    """Largest centred integer window, rounding its shorter dimension to nearest pixel."""
    ow, oh = _size(*original_size)
    w, h = _size(*analysis_size)
    if ow * h >= oh * w:
        cw, ch = max(1, (2 * oh * w + h) // (2 * h)), oh
    else:
        cw, ch = ow, max(1, (2 * ow * h + w) // (2 * w))
    return [(ow - cw) // 2, (oh - ch) // 2, cw, ch]


def crop_transform(crop_xywh, analysis_size):
    """Map integer-centre original x,y to analysis x,y. Camera Z remains unchanged."""
    x, y, cw, ch = crop_xywh
    w, h = analysis_size
    sx, sy = w / cw, h / ch
    return np.array([[sx, 0., sx * (0.5 - x) - 0.5],
                     [0., sy, sy * (0.5 - y) - 0.5], [0., 0., 1.]], dtype=np.float64)


def checked_display_size(analysis_size, requested):
    w, h = analysis_size
    require(isinstance(requested, (tuple, list)) and len(requested) == 2, "expected display W H")
    dw, dh = _size(*requested)
    require(max(dw, dh) <= 4096 and dw * dh <= 8_847_360 and dw * h == dh * w,
            "display size must preserve the analysis aspect and fit the display budget")
    return dw, dh


def build_grids(profile_id, original_views, *, display_size=None):
    require(isinstance(profile_id, str) and profile_id in PROFILE_SIZES,
            "unsupported analysis profile", Unsupported)
    w, h = PROFILE_SIZES[profile_id]
    dw, dh = checked_display_size((w, h), display_size if display_size is not None else (w, h))
    require(isinstance(original_views, dict) and original_views, "original cameras required")
    views = {}
    for name, original in original_views.items():
        require(isinstance(original, dict) and all(key in original for key in ("width", "height", "K", "distortion")),
                "missing original pinhole camera metadata")
        ow, oh = _size(original["width"], original["height"])
        require(original["distortion"] in ("none", "rectified"),
                "supply an undistorted or previously rectified source camera", Unsupported)
        k = _intrinsics(original["K"])
        crop = center_crop((ow, oh), (w, h))
        a = crop_transform(crop, (w, h))
        # Explicit binary64 operations keep the wire identity independent of BLAS/FMA dispatch.
        sx, sy, tx, ty = float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])
        adjusted = [[sx * float(k[0, 0]), 0.0, sx * float(k[0, 2]) + tx],
                    [0.0, sy * float(k[1, 1]), sy * float(k[1, 2]) + ty],
                    [0.0, 0.0, 1.0]]
        views[name] = {"original_size": [ow, oh], "original_K": k.tolist(),
                       "distortion": original["distortion"], "crop_xywh": crop,
                       "original_to_analysis": a.tolist(), "analysis_K": adjusted}
    pw, ph = (-w) % PATCH, (-h) % PATCH
    return {"schema": "el-image-grids-1", "profile_id": profile_id, "analysis_size": [w, h],
            "pixel_centres": "integer_xy_top_left", "photo_resampling": "display_srgb8_crop_lanczos_v1",
            "prediction_sampling": "native_analysis_rgb_expected_camera_z_alpha",
            "encoder": {"input_size": [w + pw, h + ph], "patch_size": PATCH,
                        "patch_grid": [(w + pw) // PATCH, (h + ph) // PATCH],
                        "pad_ltrb": [0, 0, pw, ph], "padding": "normalized_zero_excluded_from_rgb_support",
                        "position": "dinov2_vits14_bicubic_offset_0.1_no_antialias",
                        "sampling": "center_bilinear_with_patch_support"},
            "display": {"size": [dw, dh], "rgb_filter": "lanczos",
                        "risk_filter": "nearest", "state_reason_filter": "nearest"},
            "views": views}


def validate_grid_record(grids, view_ids):
    """Validate the entire self-contained score/report witness without reopening referenced images."""
    require(isinstance(grids, dict) and isinstance(grids.get("views"), dict) and
            isinstance(grids.get("display"), dict), "missing image-grid specification")
    exact_keys(grids["views"], view_ids, "per-view grid identity")
    originals = {}
    for name, view in grids["views"].items():
        require(isinstance(view, dict) and isinstance(view.get("original_size"), list)
                and len(view["original_size"]) == 2
                and "original_K" in view and "distortion" in view, "malformed original grid")
        originals[name] = dict(width=view["original_size"][0], height=view["original_size"][1],
                               K=view["original_K"], distortion=view["distortion"])
    size = grids["display"].get("size")
    require(isinstance(size, list) and len(size) == 2, "missing display size")
    expected = build_grids(grids.get("profile_id"), originals, display_size=size)
    require(encoded(expected) == encoded(grids), "grid receipt differs from the declared transformation")
    return expected["profile_id"]


def validate_grids(grids, camera_document, view_ids):
    profile_id = validate_grid_record(grids, view_ids)
    for name in view_ids:
        cam = camera_document["views"][name]
        require([cam["width"], cam["height"]] == grids["analysis_size"]
                and np.allclose(_intrinsics(cam["K"]), grids["views"][name]["analysis_K"],
                                rtol=0., atol=1e-9),
                "analysis camera dimensions or K differ from original crop: " + name)
    return profile_id


def resample_photo(image, view_grid, analysis_size):
    """Apply only to an undistorted 8-bit display-sRGB source photograph."""
    if not isinstance(image, Image.Image):
        a = np.asarray(image)
        require(a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] == 3, "expected uint8 RGB photo")
        image = Image.fromarray(a)
    require(image.mode == "RGB" and list(image.size) == view_grid["original_size"],
            "original photo differs from its grid")
    x, y, cw, ch = view_grid["crop_xywh"]
    return image.crop((x, y, x + cw, y + ch)).resize(tuple(analysis_size), Image.Resampling.LANCZOS)
