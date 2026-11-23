"""Read native render passes and derive truth masks in the frozen camera convention."""
import hashlib
import json
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image
from mathutils import Vector

from scene import project


REASONS = {
    0: "target_no_surface", 1: "same_first_hit", 2: "nearer_first_hit", 3: "outside_input_image",
    4: "behind_input_camera", 5: "nonfinite_position", 6: "invalid_target_z",
    7: "no_source_ray_hit", 8: "first_hit_farther_than_target", 9: "degenerate_input_ray",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def read_passes(path):
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        if len(exr.parts) != 1:
            raise ValueError("Expected one multilayer EXR part")
        channels = exr.channels()
        def vector(pass_name, component_names):
            names = [f"Reference.{pass_name}.{component}" for component in component_names]
            if not all(k in channels for k in names):
                raise ValueError(f"Missing {names} in EXR channels {list(channels)}")
            return np.stack([channels[k].pixels for k in names], axis=-1)
        linear = vector("Combined", "RGB")
        position = vector("Position", "XYZ")
        ids = [v.pixels.copy() for k, v in channels.items() if k.startswith("Reference.IndexOB.")]
        if linear.dtype != np.float32 or position.dtype != np.float32 or len(ids) != 1:
            raise ValueError(f"Expected 32-bit raw passes and one object ID channel, got {list(channels)}")
        if not (np.isfinite(ids[0]).all() and np.array_equal(ids[0], np.rint(ids[0])) and
                ids[0].min() >= 0 and ids[0].max() <= 65535):
            raise ValueError("Invalid object ID pass")
        return linear, position, ids[0].astype(np.uint16), sorted(channels)


def display_srgb(linear):
    # Standard with sRGB display, no Look, exposure 0 and gamma 1.
    # The active Blender OCIO output is independently checked against its native PNG.
    x = np.maximum(np.asarray(linear, dtype=np.float32), 0)
    return np.clip(np.where(x <= 0.0031308, 12.92*x, 1.055*x**(1/2.4) - 0.055), 0, 1).astype(np.float32)


def png_codes(rgb):
    return np.floor(rgb * 255.0 + 0.5).astype(np.uint8)


def save_png(path, rgb):
    Image.fromarray(png_codes(rgb)).save(path, compress_level=6)


def png_pixels(path):
    with Image.open(path) as image:
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Unexpected image mode {image.mode}")
        return np.array(image.convert("RGB"))


def reduce_half(code, ties_to_even=False):
    """Exact integer 2x2 box average in display-sRGB, with one final rounding."""
    h, w, c = code.shape
    if h % 2 or w % 2 or code.dtype != np.uint8:
        raise ValueError("Half resolution requires an even-sized uint8 source")
    total = code.reshape(h//2, 2, w//2, 2, c).sum(axis=(1, 3), dtype=np.uint16)
    if ties_to_even:
        quotient, remainder = np.divmod(total, 4)
        return (quotient + ((remainder > 2) | ((remainder == 2) & (quotient % 2 == 1)))).astype(np.uint8)
    return ((total + 2) // 4).astype(np.uint8)


def half_camera(view):
    transform = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
    return {**view, "width": view["width"]//2, "height": view["height"]//2,
            "K": (transform @ view["K"]).tolist(),
            "K_edge": (np.diag([0.5, 0.5, 1.0]) @ view["K_edge"]).tolist()}


def visibility(position, object_id, camera_z, source, bvh, cfg):
    surface = (object_id != 0) & np.isfinite(position).all(axis=-1) & np.isfinite(camera_z) & (camera_z > 0)
    reasons = np.zeros(object_id.shape, dtype=np.uint8)
    reasons[(object_id > 0) & ~np.isfinite(position).all(axis=-1)] = 5
    reasons[(object_id > 0) & np.isfinite(position).all(axis=-1) & ~(camera_z > 0)] = 6
    hit_ids = np.zeros(object_id.shape, dtype=np.uint16)
    uv, src_z = project(position, source)
    in_front = surface & (src_z > 0)
    reasons[surface & ~in_front] = 4
    inside = (in_front & (uv[..., 0] >= -0.5) & (uv[..., 0] < source["width"] - 0.5) &
              (uv[..., 1] >= -0.5) & (uv[..., 1] < source["height"] - 0.5))
    reasons[in_front & ~inside] = 3
    center = np.asarray(source["c2w"])[:3, 3]
    origin = Vector(center)
    tree, owners = bvh
    flat = np.flatnonzero(inside.ravel())
    # Pass short blocks to the C BVH to bound Python list allocation.
    for start in range(0, len(flat), 16384):
        indices = flat[start:start+16384]
        vectors = position.reshape(-1, 3)[indices].astype(np.float64) - center
        distances = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(distances[:, None], 1e-30)
        epsilons = np.maximum(cfg["ray_epsilon_abs_m"], cfg["ray_epsilon_rel"] * distances)
        chunk_codes, chunk_ids = [], []
        for direction, distance, eps in zip(directions.tolist(), distances.tolist(), epsilons.tolist()):
            hit_index, hit_dist = tree.ray_cast(origin, direction)[2:] if distance > 0 else (None, None)
            if distance <= 0:
                code = 9
            elif hit_index is None:
                code = 7
            elif abs(hit_dist - distance) <= eps:
                code = 1
            elif hit_dist < distance - eps:
                code = 2
            else:
                code = 8
            chunk_codes.append(code)
            chunk_ids.append(owners[hit_index] if hit_index is not None else 0)
        reasons.ravel()[indices] = chunk_codes
        hit_ids.ravel()[indices] = chunk_ids
    classes = np.take(np.array([0, 1, 2, 3, 3, 0, 0, 0, 0, 0], dtype=np.uint8), reasons)
    return surface, classes, reasons, hit_ids


def edges(object_id, surface, classes, radius):
    edge = np.zeros_like(surface)
    for field in (object_id, surface, classes):
        different = field[1:, :] != field[:-1, :]
        edge[1:, :] |= different
        edge[:-1, :] |= different
        different = field[:, 1:] != field[:, :-1]
        edge[:, 1:] |= different
        edge[:, :-1] |= different
    if radius:
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius, radius)
            window = np.lib.stride_tricks.sliding_window_view(np.pad(edge, padding), 2*radius+1, axis=axis)
            edge = window.any(axis=-1)
    return edge


def longest_horizontal_run(mask):
    padded = np.pad(mask, ((0, 0), (1, 1)))
    start = np.nonzero(~padded[:, :-1] & padded[:, 1:])[1]
    end = np.nonzero(padded[:, :-1] & ~padded[:, 1:])[1]
    return int((end-start).max(initial=0))


def geometric_disocclusion(obj, classes, hit_ids, edge, cfg):
    """Find measurable bands formed by a single identified receiver/occluder pair."""
    valid = (classes == 2) & (obj != 0) & (hit_ids != 0) & (obj != hit_ids)
    keys = np.where(valid, (obj.astype(np.uint32) << 16) | hit_ids, 0)
    pair_ids, counts = np.unique(keys[valid], return_counts=True)
    if not len(pair_ids):
        return {"pair_count": 0, "measurable_pairs": [], "meets_width_and_interior": False}
    # The start and end arrays have identical row/run ordering, including runs at either image boundary.
    padded = np.pad(keys, ((0, 0), (1, 1)))
    rows, starts = np.nonzero(valid & (keys != padded[:, :-2]))
    _, ends = np.nonzero(valid & (keys != padded[:, 2:]))
    width = np.zeros(len(pair_ids), dtype=int)
    np.maximum.at(width, np.searchsorted(pair_ids, keys[rows, starts]), ends - starts + 1)
    interior = np.bincount(np.searchsorted(pair_ids, keys[valid & ~edge]), minlength=len(pair_ids))
    qualified = (width >= cfg["band_min_width_px"]) & (interior >= cfg["band_min_interior_pixels"])
    pairs = [{"receiver_id": int(pair_ids[i]) >> 16, "occluder_id": int(pair_ids[i]) & 65535,
              "total_pixels": int(counts[i]), "interior_pixels": int(interior[i]),
              "max_contiguous_horizontal_width_px": int(width[i])} for i in np.flatnonzero(qualified)]
    return {"pair_count": len(pair_ids), "measurable_pairs": pairs, "meets_width_and_interior": bool(pairs)}


def geometry_passes(pos, obj, view, source, bvh, cfg):
    uv, z = project(pos, view)
    surface, classes, reasons, hit_ids = visibility(pos, obj, z, source, bvh, cfg)
    y, x = np.nonzero(surface)
    err = np.linalg.norm(uv[surface] - np.column_stack((x, y)), axis=1)
    edge = edges(obj, surface, classes, cfg["edge_radius_px"])
    info = {"surface_pixels": int(surface.sum()), "edge_pixels": int(edge.sum()),
            "max_reprojection_px": float(err.max()) if len(err) else None,
            "p99_reprojection_px": float(np.percentile(err, 99)) if len(err) else None,
            "visibility_counts": np.bincount(classes.ravel(), minlength=4).tolist(),
            "visibility_reasons": {REASONS[i]: int(n) for i, n in enumerate(
                np.bincount(reasons.ravel(), minlength=len(REASONS)))}}
    if cfg.get("occlusion_profile") == "opaque_mesh_pairs":
        info["geometric_disocclusion"] = geometric_disocclusion(obj, classes, hit_ids, edge, cfg)
    else:
        revealed = (obj == cfg["revealed_object_id"]) & (classes == 2) & (hit_ids == cfg["occluder_id"])
        info["known_occlusion_pair"] = {"total_pixels": int(revealed.sum()), "interior_pixels": int((revealed & ~edge).sum()),
                                        "max_contiguous_horizontal_width_px": longest_horizontal_run(revealed)}
    numerical = {"position": np.where(surface[..., None], pos, np.nan).astype(np.float32),
                 "z": np.where(surface, z, np.nan).astype(np.float32), "object_id": obj,
                 "visibility": classes, "visibility_reason": reasons, "first_source_hit_id": hit_ids,
                 "reference_surface": surface, "edge": edge}
    return numerical, info
