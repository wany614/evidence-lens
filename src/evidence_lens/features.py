"""Deployable features from one condition's exact input, cameras and predicted arrays."""

import numpy as np

from .scoring_io import Camera, alpha_valid, depth_valid, rgb_valid


ALPHA_MIN = np.float32(0.5)
CONFLICT_MAX = np.float32(0.10)
FRAME_STATES = ("unprojectable", "inside", "outside", "behind_camera")
FEATURES = {
    "target_alpha": ("fraction", "Target accumulated opacity"),
    "source_alpha": ("fraction", "Source-render opacity at the projected location"),
    "c_behind": ("fraction", "Positive excess projected Z divided by projected Z"),
    "c_front": ("fraction", "Positive excess source-render Z divided by source-render Z"),
    "border_distance": ("source_short_sides", "Signed distance from the projected point to the nearest input border"),
    "relative_translation": ("fraction", "Camera-centre separation divided by target expected Z"),
    "view_angle": ("radians", "Angle between source and target observation directions"),
    "local_depth_range": ("fraction", "3x3 valid target Z range divided by centre Z"),
    "local_rgb_difference": ("display_srgb", "3x3 mean of per-neighbour channel MAE to the centre"),
    "source_rgb_difference": ("display_srgb", "Target RGB versus exact input at a supported consistent correspondence"),
    "source_render_residual": ("display_srgb", "Source self-render versus exact input at the projected location"),
}


def feature_schema():
    return {
        "schema": "el-features-1",
        "columns": [*FEATURES, *("frame_" + name for name in FRAME_STATES),
                    *("missing_" + name for name in FEATURES)],
        "numeric_features": {name: {"dtype": "float32", "unit": unit, "definition": definition}
                             for name, (unit, definition) in FEATURES.items()},
        "flag_dtype": "bool",
        "frame_relation_uint8": dict(enumerate(FRAME_STATES)),
        "missing_numeric": "NaN and per-column missing flag",
        "array_order": "row, column; rows increase downward",
        "projection": "OpenCV integer-centre K; add 0.5 to obtain edge coordinates, then floor inside [0,W)x[0,H)",
        "neighbourhood": "3x3 truncated at boundaries including centre; at least 3 valid samples",
        "source_rgb_difference_requires": "inside, both alphas >=0.5, positive validated depths and both conflicts <=0.10",
        "source_render_residual_requires": "inside and finite source RGB values",
        "depth_semantics": "verified expected camera-Z, metres",
        "eligibility_mask": "scoreable = depth verified, valid target Z, alpha and RGB",
        "alpha_min": float(ALPHA_MIN),
        "relative_depth_conflict_threshold": 0.10,
    }


def _blank(shape):
    return np.full(shape, np.nan, dtype=np.float32)


def _local_variation(rgb, z, z_ok):
    h, w = z.shape
    rgb_ok = rgb_valid(rgb)
    rgb_count = np.zeros((h, w), np.uint8)
    z_count = np.zeros_like(rgb_count)
    rgb_sum = np.zeros((h, w), np.float32)
    z_min = np.full((h, w), np.inf, np.float32)
    z_max = np.full((h, w), -np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            centres = (slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx)))
            near = (slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx)))
            valid = rgb_ok[centres] & rgb_ok[near]
            with np.errstate(invalid="ignore"):
                diff = np.abs(rgb[near] - rgb[centres]).mean(axis=-1)
            rgb_sum[centres] += np.where(valid, diff, 0)
            rgb_count[centres] += valid
            valid = z_ok[near]
            z_min[centres] = np.minimum(z_min[centres], np.where(valid, z[near], np.inf))
            z_max[centres] = np.maximum(z_max[centres], np.where(valid, z[near], -np.inf))
            z_count[centres] += valid
    rgb_result = np.divide(rgb_sum, rgb_count, out=_blank((h, w)), where=rgb_count >= 3)
    z_result = np.divide(z_max - z_min, z, out=_blank((h, w)), where=z_ok & (z_count >= 3))
    return rgb_result, z_result


def extract_features(input_rgb, source, target, source_cam: Camera, target_cam: Camera, depth_verified: bool):
    """All file I/O is outside this function. Source render may have a declared failure."""
    z, rgb, alpha = target["z"], target["rgb"], target["alpha"]
    shape = target_cam.shape
    z_ok = depth_valid(z) & depth_verified
    alpha_ok = alpha_valid(alpha)
    rgb_ok = rgb_valid(rgb)
    result = {name: _blank(shape) for name in FEATURES}
    result["target_alpha"] = np.where(alpha_ok, alpha, np.float32(np.nan))
    result["scoreable"] = z_ok & alpha_ok & rgb_ok
    result["local_rgb_difference"], result["local_depth_range"] = _local_variation(rgb, z, z_ok)
    frame = np.zeros(shape, np.uint8)
    if z_ok.any():
        yy, xx = np.indices(shape, dtype=np.float64)
        pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        target_point = (pixels @ np.linalg.inv(target_cam.K).T) * np.where(z_ok, z, 0)[..., None]
        transform = np.linalg.solve(source_cam.c2w, target_cam.c2w)
        target_direction = target_point @ transform[:3, :3].T
        source_point = target_direction + transform[:3, 3]
        source_z = source_point[..., 2]
        forward = z_ok & (source_z > 0)
        frame[z_ok & ~forward] = 3
        homogeneous = source_point @ source_cam.K.T
        q = np.full((*shape, 2), np.nan)
        np.divide(homogeneous[..., :2], source_z[..., None], out=q, where=forward[..., None])
        q += 0.5
        qx, qy = q[..., 0], q[..., 1]
        W, H = source_cam.width, source_cam.height
        inside = forward & (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
        frame[forward & ~inside], frame[inside] = 2, 1
        border = np.minimum.reduce([qx, W - qx, qy, H - qy]) / min(W, H)
        result["border_distance"][forward] = border[forward]
        separation = np.linalg.norm(source_cam.c2w[:3, 3] - target_cam.c2w[:3, 3])
        result["relative_translation"][z_ok] = separation / z[z_ok]
        cross = np.linalg.norm(np.cross(target_direction, source_point), axis=-1)
        dot = np.sum(target_direction * source_point, axis=-1)
        angle_ok = z_ok & (np.linalg.norm(source_point, axis=-1) > 0)
        result["view_angle"][angle_ok] = np.arctan2(cross[angle_ok], dot[angle_ok])
        if source is not None and inside.any():
            u = np.where(inside, qx, 0).astype(np.intp)
            v = np.where(inside, qy, 0).astype(np.intp)
            a0, z0 = source["alpha"][v, u], source["z"][v, u]
            a0_ok = inside & alpha_valid(a0)
            result["source_alpha"][a0_ok] = a0[a0_ok]
            both_z = inside & depth_valid(z0)
            behind, front = result["c_behind"], result["c_front"]
            depth_delta = source_z[both_z] - z0[both_z]
            behind[both_z] = np.maximum(0, depth_delta) / np.maximum(source_z[both_z], 1e-6)
            front[both_z] = np.maximum(0, -depth_delta) / np.maximum(z0[both_z], 1e-6)
            source_input, self_render = input_rgb[v, u], source["rgb"][v, u]
            supported = (both_z & alpha_ok & (alpha >= ALPHA_MIN) & a0_ok & (a0 >= ALPHA_MIN)
                         & (behind <= CONFLICT_MAX) & (front <= CONFLICT_MAX) & rgb_ok)
            result["source_rgb_difference"][supported] = np.abs(rgb[supported] - source_input[supported]).mean(axis=-1)
            residual_ok = inside & rgb_valid(self_render)
            result["source_render_residual"][residual_ok] = np.abs(
                self_render[residual_ok] - source_input[residual_ok]
            ).mean(axis=-1)
    result["frame_relation"] = frame
    for code, name in enumerate(FRAME_STATES):
        result[f"frame_{name}"] = frame == code
    for name in FEATURES:
        result[f"missing_{name}"] = ~np.isfinite(result[name])
    return result
