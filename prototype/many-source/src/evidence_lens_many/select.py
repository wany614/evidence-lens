"""Deterministic retrieval on fixed camera geometry and optional coarse target depth."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from evidence_lens.multisource import _project
from evidence_lens_bundle.common import require

from .schema import MIN_ALPHA


@dataclass(frozen=True)
class Plan:
    bank_sha256: str
    target_id: str
    strategy: str
    ids: tuple
    reasons: tuple
    cost: dict


def rig_geometry(cams):
    centres = np.array([c.c2w[:3, 3] for c in cams])
    axes = np.array([c.c2w[:3, 2] for c in cams])
    separation = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    values = separation[separation > 1e-12]
    scale = float(np.median(values)) if values.size else 1.0
    angles = np.arccos(np.clip(axes @ axes.T, -1, 1))
    pair = 0.5 * np.minimum(separation / scale, 1) + 0.5 * angles / np.pi
    close = (separation <= 0.02 * scale) & (angles <= np.deg2rad(2.0))
    return centres, axes, scale, pair, close


def _tiles(camera, geom):
    if geom is None:
        return np.empty((0, 3), np.float64)
    z, alpha = geom["z"], geom["alpha"]
    ok = np.isfinite(z) & (z > 0) & np.isfinite(alpha) & (alpha >= MIN_ALPHA) & (alpha <= 1)
    coordinates = []
    for ya, yb in zip(
        np.linspace(0, camera.height, 9, dtype=int)[:-1], np.linspace(0, camera.height, 9, dtype=int)[1:]
    ):
        for xa, xb in zip(
            np.linspace(0, camera.width, 13, dtype=int)[:-1], np.linspace(0, camera.width, 13, dtype=int)[1:]
        ):
            y, x = np.nonzero(ok[ya:yb, xa:xb])
            if len(y):
                idx = np.argmin((2 * y - (yb - ya - 1)) ** 2 + (2 * x - (xb - xa - 1)) ** 2)
                coordinates.append((int(x[idx]) + xa, int(y[idx]) + ya))
    if not coordinates:
        return np.empty((0, 3), np.float64)
    x, y = np.array(coordinates).T
    return (np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(camera.K).T) * z[y, x, None]


def select_sources(bank, target_id, strategy="coverage"):
    start = perf_counter()
    require(
        target_id in bank.target_ids and strategy in ("fixed", "nearest", "coverage"),
        "unregistered target or strategy",
    )
    names = tuple(sorted(n for n in bank.ids if n in bank.granted))
    require(bool(names), "no authorized source cameras")
    cams = [bank.cameras[n] for n in names]
    target = bank.cameras[target_id]
    # One scale is fixed by the whole allowed rig. It never depends on the current Q.
    centres, axes, scale, pair, close = rig_geometry(cams)
    target_distance = np.linalg.norm(centres - target.c2w[:3, 3], axis=1) / scale
    target_angle = np.arccos(np.clip(axes @ target.c2w[:3, 2], -1, 1)) / np.pi
    distance = target_distance + target_angle
    start_costs = dict(bank.counters)
    points = (
        _tiles(target, bank.target_geometry(target_id))
        if strategy == "coverage" and bank.meta["depth_verified"]
        else np.empty((0, 3))
    )
    cover = np.zeros((len(names), len(points)), bool)
    if points.size:
        for i, cam in enumerate(cams):
            cover[i] = _project(points, np.ones(len(points), bool), target, cam)[0]
    tile_count = np.zeros(len(points), np.uint32)
    remain = np.ones(len(names), bool)
    chosen = []
    reasons = []
    while remain.any():
        near = np.any(close[:, chosen], axis=1) if chosen else np.zeros(len(names), bool)
        diversity = np.min(pair[:, chosen], axis=1) if chosen else np.zeros(len(names))
        gain = (cover / (1 + tile_count)).mean(axis=1) if len(points) else np.zeros(len(names))
        if strategy == "nearest" or (strategy == "coverage" and not len(points)):
            utility = -distance
        elif strategy == "fixed":
            utility = diversity - 0.30 * near if chosen else -pair.mean(axis=1)
        else:
            utility = gain + 0.15 * diversity - 0.30 * near - 0.025 * distance / (1 + distance)
        i = int(np.argmax(np.where(remain, utility, -np.inf)))
        chosen.append(i)
        remain[i] = False
        tile_count += cover[i]
        reasons.append(
            {
                "source_id": names[i],
                "marginal_tile_coverage": float(gain[i]),
                "new_view_diversity": float(diversity[i]),
                "near_duplicate_of_prior": bool(near[i]),
                "target_distance_and_orientation": float(distance[i]),
                "projections_inside": int(cover[i].sum()),
                "selection_utility": float(utility[i]),
            }
        )
    return Plan(
        bank.sha256,
        target_id,
        strategy,
        tuple(names[i] for i in chosen),
        tuple(reasons),
        {
            "camera_candidates": len(names),
            "tile_grid": [12, 8],
            "valid_target_tiles": len(points),
            "camera_tile_projection_tests": len(names) * len(points),
            "rig_separation_scale_scene_units": scale,
            "near_duplicate_pairs": int(np.count_nonzero(np.triu(close, k=1))),
            "ungranted_ids": sorted(set(bank.ids) - set(names)),
            "seconds": perf_counter() - start,
            **{
                k: bank.counters[k] - v
                for k, v in start_costs.items()
                if k
                in (
                    "source_photos_decoded",
                    "source_renders_decoded",
                    "target_geometry_decoded",
                    "target_rgb_decoded",
                    "target_bytes_read",
                )
            },
        },
    )
