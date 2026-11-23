"""Deployment-only multi-source risk features. All RGB is display sRGB; all Z is predicted camera Z."""

import hashlib
import itertools
import json

import numpy as np

from .features import ALPHA_MIN, CONFLICT_MAX, _local_variation
from .scoring_io import alpha_valid, check_rgb, depth_valid, rgb_valid

SOURCES = ("s00", "s01", "s02", "s03")
CONDITIONS = {"M2": ("s00", "s03"), "M4": SOURCES}
SCOPES = ("one", "two", "all")
TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
COLOR_LIMIT = np.float32(0.10)
# Column order is part of the binary model contract.
DEFINITIONS = {
    "target_alpha": ("fraction", "Finite target accumulated opacity in [0,1]."),
    "target_local_depth_range": (
        "fraction",
        "Truncated 3x3 positive Z range divided by centre Z; at least 3 valid samples.",
    ),
    "target_local_rgb_mae": (
        "display_srgb",
        "Truncated 3x3 mean channel MAE to centre; at least 3 valid samples.",
    ),
    "n_sources": ("sources", "Number of source RGB frames allowed for this score."),
    "n_in_frame": ("sources", "Positive projected Z and nearest integer-centre pixel in the source frame."),
    "n_comparable": (
        "sources",
        "In frame, valid source and target opacities, positive validated source and target Z.",
    ),
    "comparable_fraction": ("fraction", "n_comparable divided by n_sources."),
    "n_supported": ("sources", "Comparable with both opacities >=0.5 and relative depth residual <=0.10."),
    "support_fraction": ("fraction", "n_supported divided by n_sources."),
    "n_depth_conflict": (
        "sources",
        "Comparable with both opacities >=0.5 and relative depth residual >0.10.",
    ),
    "n_colour_conflict": ("sources", "Supported original source versus target-prediction RGB MAE >0.10."),
    "n_conflict": ("sources", "Number of sources with either a depth conflict or a colour conflict."),
    "source_alpha_mean": ("fraction", "Mean source opacity among comparable samples."),
    "depth_residual_mean": (
        "fraction",
        "Mean |Z_projected-Z_source|/max(Z_projected,Z_source,1e-6) among comparable samples.",
    ),
    "depth_residual_max": ("fraction", "Maximum of the comparable relative depth residuals."),
    "view_angle_mean": (
        "radians",
        "Mean angle between source and target observation rays among comparable sources.",
    ),
    "view_angle_max": ("radians", "Maximum comparable observation-ray angle."),
    "target_source_mae_mean": (
        "display_srgb",
        "Mean supported target-prediction versus original-source channel MAE.",
    ),
    "target_source_mae_min": (
        "display_srgb",
        "Minimum supported target-prediction versus original-source channel MAE.",
    ),
    "target_source_mae_range": (
        "display_srgb",
        "Max minus min among at least two supported finite target/source colour residuals.",
    ),
    "self_render_mae_mean": (
        "display_srgb",
        "Mean original versus self-render source colour MAE among comparable finite source renders.",
    ),
    "source_pair_mae_max": (
        "display_srgb",
        "Maximum channel MAE between any two supported original source colours.",
    ),
}
NUMERIC = tuple(DEFINITIONS)
OPTIONAL = (*NUMERIC[:3], *NUMERIC[12:])
FLAGS = ("no_in_frame", "no_comparable", "no_support", "fewer_than_two_support")
COLUMNS = (*NUMERIC, *("missing_" + k for k in OPTIONAL), *FLAGS)
REASONS = (
    "scoreable",
    "depth_unverified",
    "invalid_target_rgb",
    "invalid_target_alpha",
    "invalid_target_z",
    "no_in_frame_projection",
    "no_comparable_source",
)


def source_ids(condition, scope):
    if condition not in CONDITIONS or scope not in SCOPES:
        raise ValueError("expected M2/M4 and one/two/all")
    return ("s00",) if scope == "one" else CONDITIONS["M2"] if scope == "two" else CONDITIONS[condition]


def schema():
    return {
        "schema": "multisource-v1",
        "version": 1,
        "columns": list(COLUMNS),
        "numeric_columns": list(NUMERIC),
        "definitions": {k: {"unit": a, "meaning": b} for k, (a, b) in DEFINITIONS.items()},
        "missing_indicators_for": list(OPTIONAL),
        "empty_indicators": list(FLAGS),
        "dtype": "float32",
        "missing_value": "IEEE NaN in numeric column; 1 in its missing indicator",
        "scoreable": "verified ED, finite valid target RGB, alpha and positive Z, and at least one comparable source",
        "reason_codes": dict(enumerate(REASONS)),
        "reason_priority": list(REASONS[1:]),
        "coordinate_convention": "OpenCV +x right, +y down, +z forward; integer pixel centres; nearest floor(u+0.5), half-open bounds",
        "neighbourhood": "3x3 includes centre and truncates at image boundaries",
        "source_order": "sort by original source ID, reduce finite values in float64, emit float32",
        "source_sampling": "original source RGB8 converted to float32 /255; nearest integer pixel; self-render from same reconstruction",
        "conditions": {k: list(v) for k, v in CONDITIONS.items()},
        "scopes": list(SCOPES),
        "unavailable_source_render": "still counts as provided and can be in frame, cannot be comparable",
        "unsupported_domains": "one-source scores are extrapolation from all-source M2/M4 training; no evidence means unknown",
        "thresholds": {"alpha_min": 0.5, "relative_depth_conflict": 0.10, "colour_mae_conflict": 0.10},
    }


def schema_bytes():
    return (json.dumps(schema(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def schema_sha256():
    return hashlib.sha256(schema_bytes()).hexdigest()


def validate_render(frame, camera):
    if set(frame) != {"rgb", "z", "alpha"}:
        raise ValueError("a numeric render must contain exactly rgb,z,alpha")
    check_rgb(frame["rgb"], camera, "prediction")
    for k in ("z", "alpha"):
        if frame[k].dtype != np.float32 or frame[k].shape != camera.shape:
            raise ValueError(f"render {k} must be native float32 of the camera shape")
    if np.any((frame["alpha"] == 0) & depth_valid(frame["z"])):
        raise ValueError("positive expected depth at zero opacity")


def _project(points, target_z_ok, target_camera, source_camera):
    transform = np.linalg.solve(source_camera.c2w, target_camera.c2w)
    direction = points @ transform[:3, :3].T
    source_point = direction + transform[:3, 3]
    projected_z = source_point[..., 2]
    forward = target_z_ok & (projected_z > 0)
    q = source_point @ source_camera.K.T
    xy = np.full((*forward.shape, 2), np.nan)
    np.divide(q[..., :2], projected_z[..., None], out=xy, where=forward[..., None])
    x, y = xy[..., 0] + 0.5, xy[..., 1] + 0.5
    inside = forward & (x >= 0) & (x < source_camera.width) & (y >= 0) & (y < source_camera.height)
    u = np.where(inside, x, 0).astype(np.intp)
    v = np.where(inside, y, 0).astype(np.intp)
    angle = np.arctan2(
        np.linalg.norm(np.cross(direction, source_point), axis=-1), np.sum(direction * source_point, axis=-1)
    )
    return inside, v, u, projected_z, angle


def _reduce(values, kind):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    if kind == "mean":
        return np.divide(
            np.where(finite, values, 0).sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0
        )
    if kind == "max":
        return np.where(n > 0, np.where(finite, values, -np.inf).max(axis=0), np.nan)
    return np.where(n > 0, np.where(finite, values, np.inf).min(axis=0), np.nan)


def extract(target, target_camera, sources, *, depth_verified=True):
    """Accept only legal (original source RGB, same-reconstruction self render, source Camera) tuples."""
    if type(depth_verified) is not bool or not sources or not set(sources) <= set(SOURCES):
        raise ValueError("invalid predicted-depth validation or source IDs")
    validate_render(target, target_camera)
    rgb, z, alpha = (target[k] for k in ("rgb", "z", "alpha"))
    tz = depth_valid(z) & depth_verified
    ta, tr = alpha_valid(alpha), rgb_valid(rgb)
    rgb_local, z_local = _local_variation(rgb, z, tz)
    yy, xx = np.indices(z.shape, dtype=np.float64)
    ray = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(target_camera.K).T
    points = ray * np.where(tz, z, 0)[..., None]
    names = ("inside", "comparable", "support", "depth_conflict", "colour_conflict")
    counts = {k: np.zeros(z.shape, np.uint8) for k in names}
    quantities = {k: [] for k in ("alpha", "residual", "angle", "target_mae", "self_mae", "colour")}
    for name in sorted(sources):
        image, frame, camera = sources[name]
        check_rgb(image, camera, f"original {name}")
        if not rgb_valid(image).all():
            raise ValueError("an original source PNG must be finite")
        inside, v, u, projected_z, angle = _project(points, tz, target_camera, camera)
        counts["inside"] += inside
        if frame is None:
            continue
        validate_render(frame, camera)
        a, zs = frame["alpha"][v, u], frame["z"][v, u]
        compare = inside & ta & alpha_valid(a) & depth_valid(zs)
        residual = np.full(z.shape, np.nan, np.float32)
        residual[compare] = np.abs(projected_z[compare] - zs[compare]) / np.maximum(
            np.maximum(projected_z[compare], zs[compare]), 1e-6
        )
        opaque = compare & (alpha >= ALPHA_MIN) & (a >= ALPHA_MIN)
        support = opaque & (residual <= CONFLICT_MAX)
        d_conflict = opaque & ~support
        image_rgb, frame_rgb = image[v, u], frame["rgb"][v, u]
        with np.errstate(invalid="ignore"):
            target_mae = np.where(support & tr, np.abs(rgb - image_rgb).mean(axis=-1), np.nan)
            self_mae = np.where(
                compare & rgb_valid(frame_rgb), np.abs(frame_rgb - image_rgb).mean(axis=-1), np.nan
            )
        counts["comparable"] += compare
        counts["support"] += support
        counts["depth_conflict"] += d_conflict
        counts["colour_conflict"] += np.isfinite(target_mae) & (target_mae > COLOR_LIMIT)
        for key, value in (
            ("alpha", np.where(compare, a, np.nan)),
            ("residual", residual),
            ("angle", np.where(compare, angle, np.nan)),
            ("target_mae", target_mae),
            ("self_mae", self_mae),
            ("colour", np.where(support[..., None], image_rgb, np.nan)),
        ):
            quantities[key].append(value)
    empty = np.full(z.shape, np.nan, np.float32)

    def reduced(name, kind):
        return _reduce(quantities[name], kind) if quantities[name] else empty

    n = len(sources)
    sup, comp, inside = counts["support"], counts["comparable"], counts["inside"]
    conflict = counts["depth_conflict"] + counts["colour_conflict"]
    pair = [np.abs(a - b).mean(axis=-1) for a, b in itertools.combinations(quantities["colour"], 2)]
    colour_range = np.where(
        (sup >= 2) & tr, reduced("target_mae", "max") - reduced("target_mae", "min"), np.nan
    )
    values = [
        np.where(ta, alpha, np.nan),
        z_local,
        rgb_local,
        np.full(z.shape, n),
        inside,
        comp,
        comp / n,
        sup,
        sup / n,
        counts["depth_conflict"],
        counts["colour_conflict"],
        conflict,
        reduced("alpha", "mean"),
        reduced("residual", "mean"),
        reduced("residual", "max"),
        reduced("angle", "mean"),
        reduced("angle", "max"),
        reduced("target_mae", "mean"),
        reduced("target_mae", "min"),
        colour_range,
        reduced("self_mae", "mean"),
        _reduce(pair, "max") if pair else empty,
    ]
    missing = [~np.isfinite(values[NUMERIC.index(k)]) for k in OPTIONAL]
    X = np.stack([*values, *missing, inside == 0, comp == 0, sup == 0, sup < 2], axis=-1).astype(np.float32)
    why = np.select(
        [np.full(z.shape, not depth_verified), ~tr, ~ta, ~tz, inside == 0, comp == 0],
        [1, 2, 3, 4, 5, 6],
        default=0,
    ).astype(np.uint8)
    return {
        "X": X,
        "scoreable": why == 0,
        "unknown_reason": why,
        "support_count": sup,
        "conflict_count": conflict,
        "depth_conflict_count": counts["depth_conflict"],
        "colour_conflict_count": counts["colour_conflict"],
        "in_frame_count": inside,
        "comparable_count": comp,
    }
