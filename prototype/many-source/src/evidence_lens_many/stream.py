"""A single ordered source scan with bounded per-pixel accumulators and auditable snapshots."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from evidence_lens.features import _local_variation
from evidence_lens.multisource import _project, camera_rays
from evidence_lens.scoring_io import alpha_valid, depth_valid, rgb_valid
from evidence_lens_bundle.common import require

from .schema import COLOUR_LIMIT, DEPTH_LIMIT, MIN_ALPHA, NUMERIC_NAMES, OPAQUE, OPTIONAL, SIZE


def _ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan),
        where=np.asarray(denominator) > 0,
    )


class _Moments:
    def __init__(self, shape):
        self.n = np.zeros(shape, np.uint32)
        self.total = np.zeros(shape, np.float64)
        self.square = np.zeros(shape, np.float64)
        self.minimum = np.full(shape, np.inf, np.float64)
        self.maximum = np.full(shape, -np.inf, np.float64)

    def update(self, value, use):
        good = use & np.isfinite(value)
        sample = np.where(good, value, 0.0).astype(np.float64)
        self.n += good
        self.total += sample
        self.square += sample * sample
        np.minimum(self.minimum, value, out=self.minimum, where=good)
        np.maximum(self.maximum, value, out=self.maximum, where=good)

    def values(self):
        mean = _ratio(self.total, self.n)
        sd = np.sqrt(np.maximum(0, _ratio(self.square, self.n) - mean * mean))
        return mean, sd, np.where(self.n, self.minimum, np.nan), np.where(self.n, self.maximum, np.nan)


@dataclass
class Step:
    bank_sha256: str
    target_id: str
    strategy: str
    queried_source_ids: tuple
    nominal_budgets: tuple
    library_size: int
    available_count: int
    missing_source_ids: tuple
    X: np.ndarray
    finite: np.ndarray
    invalid_geometry: np.ndarray
    incomplete: np.ndarray
    reason: np.ndarray
    in_frame_count: np.ndarray
    comparable_count: np.ndarray
    support_count: np.ndarray
    depth_conflict_count: np.ndarray
    colour_conflict_count: np.ndarray
    first_witness: np.ndarray
    cost: dict

    @property
    def actual_k(self):
        return len(self.queried_source_ids)

    @property
    def query_count(self):
        return np.broadcast_to(np.uint32(self.actual_k), self.finite.shape)


def _snapshot(
    bank, target_id, plan, ids, aliases, missing, target, geometry, local, counts, moments, witness, cost
):
    shape = SIZE[::-1]
    n = len(ids)
    available = n - len(missing)
    z, alpha, rgb = target["z"], target["alpha"], target["rgb"]
    ta = alpha_valid(alpha)
    tr = rgb_valid(rgb)
    tz = depth_valid(z)
    inside, comp, support, depth_conflict, colour_conflict = (
        counts[k] for k in ("inside", "comp", "support", "depth", "colour")
    )
    opacity, _, _, _ = moments["alpha"].values()
    relative, relative_std, _, relative_max = moments["depth"].values()
    angle, angle_std, _, angle_max = moments["angle"].values()
    photo, photo_std, photo_min, _ = moments["photo"].values()
    self_mae, _, _, self_max = moments["self"].values()
    features = [
        np.where(ta, alpha, np.nan),
        local[1],
        local[0],
        np.log1p(len(bank.ids)),
        np.log1p(n),
        n / len(bank.ids),
        available / n,
        inside / n,
        comp / n,
        _ratio(comp, inside),
        _ratio(support, comp),
        _ratio(depth_conflict, comp),
        _ratio(colour_conflict, support),
        opacity,
        relative,
        relative_max,
        relative_std,
        angle,
        angle_max,
        angle_std,
        photo,
        photo_min,
        photo_std,
        self_mae,
        self_max,
    ]
    numeric = [np.broadcast_to(a, shape) for a in features]
    x = np.stack(
        [*numeric, *(~np.isfinite(numeric[NUMERIC_NAMES.index(k)]) for k in OPTIONAL)], axis=-1
    ).astype(np.float32)
    reason = np.select(
        [
            np.full(shape, not bank.meta["depth_verified"]),
            np.full(shape, not cost["target_available"]),
            ~tr,
            ~ta,
            ~tz,
            alpha < MIN_ALPHA,
            np.full(shape, available == 0),
        ],
        [1, 2, 3, 4, 5, 6, 7],
        default=0,
    ).astype(np.uint8)
    return Step(
        bank.sha256,
        target_id,
        plan.strategy,
        ids,
        tuple(aliases),
        len(bank.ids),
        available,
        tuple(missing),
        x,
        reason == 0,
        ~geometry,
        np.full(shape, bool(missing)),
        reason,
        inside.copy(),
        comp.copy(),
        support.copy(),
        depth_conflict.copy(),
        colour_conflict.copy(),
        witness.copy(),
        dict(cost),
    )


def stream_features(bank, target_id, plan, *, budgets=(4, 8, 16, 32, "all")):
    require(
        bank.meta["reconstruction_provenance"] == "verified",
        "scoring requires a verified upstream map; source inspection is available",
    )
    require(
        bank.meta["depth_verified"],
        "complete scoring requires independently verified expected camera-Z; inspect the bank for missing materials",
    )
    require(
        plan.bank_sha256 == bank.sha256
        and plan.target_id == target_id
        and len(set(plan.ids)) == len(plan.ids)
        and set(plan.ids) == set(bank.ids) & bank.granted,
        "query plan belongs to different sources, target or frame",
    )
    groups = {}
    for budget in budgets:
        require(
            budget == "all" or type(budget) is int and budget > 0,
            "query budgets are positive integers or all",
        )
        k = min(budget, len(plan.ids)) if budget != "all" else len(plan.ids)
        if budget not in groups.setdefault(k, []):
            groups[k].append(budget)
    require(bool(groups), "at least one query budget required")
    counters = dict(bank.counters)
    started = perf_counter()
    frame = bank.target(target_id)
    shape = SIZE[::-1]
    target_available = frame is not None
    if frame is None:
        frame = {
            "rgb": np.full((*shape, 3), np.nan, np.float32),
            "z": np.full(shape, np.nan, np.float32),
            "alpha": np.full(shape, np.nan, np.float32),
        }
    z, alpha, rgb = frame["z"], frame["alpha"], frame["rgb"]
    valid_z = depth_valid(z)
    geometry = valid_z & alpha_valid(alpha) & (alpha >= MIN_ALPHA) & bank.meta["depth_verified"]
    local = _local_variation(rgb, z, valid_z)
    cam = bank.cameras[target_id]
    points = camera_rays(cam) * np.where(geometry, z, 0)[..., None]
    counts = {n: np.zeros(shape, np.uint32) for n in ("inside", "comp", "support", "depth", "colour")}
    moments = {n: _Moments(shape) for n in ("alpha", "depth", "angle", "photo", "self")}
    witness = np.full(shape, -1, np.int32)
    missing = []
    projected = compared = 0
    projection_time = photometry_time = 0.0
    for i, name in enumerate(plan.ids[: max(groups)], 1):
        tick = perf_counter()
        inside, v, u, zp, angle = _project(points, geometry, cam, bank.cameras[name])
        projection_time += perf_counter() - tick
        projected += 1
        counts["inside"] += inside
        pair = bank.source(name)
        if pair is None:
            missing.append(name)
        else:
            tick = perf_counter()
            photo, f, _ = pair
            source_a, source_z = f["alpha"][v, u], f["z"][v, u]
            comp = inside & alpha_valid(source_a) & (source_a >= MIN_ALPHA) & depth_valid(source_z)
            residual = np.zeros(shape, np.float32)
            residual[comp] = np.abs(zp[comp] - source_z[comp]) / np.maximum(
                np.maximum(zp[comp], source_z[comp]), 1e-6
            )
            opaque = comp & (alpha >= OPAQUE) & (source_a >= OPAQUE)
            support = opaque & (residual <= DEPTH_LIMIT)
            original = photo[v, u]
            with np.errstate(invalid="ignore"):
                colour = np.abs(original - rgb).mean(axis=-1)
                self_mae = np.abs(original - f["rgb"][v, u]).mean(axis=-1)
            counts["comp"] += comp
            counts["support"] += support
            counts["depth"] += opaque & ~support
            counts["colour"] += support & np.isfinite(colour) & (colour > COLOUR_LIMIT)
            witness[(witness < 0) & comp] = bank.source_index[name]
            for which, val, ok in (
                ("alpha", source_a, comp),
                ("depth", residual, comp),
                ("angle", angle, comp),
                ("photo", colour, support),
                ("self", self_mae, comp),
            ):
                moments[which].update(val, ok)
            photometry_time += perf_counter() - tick
            compared += 1
        if i in groups:
            cost = {k: bank.counters[k] - val for k, val in counters.items() if k != "cache_peak_bytes"}
            cost.update(
                cache_peak_bytes=bank.counters["cache_peak_bytes"],
                cache_capacity_bytes=bank.capacity,
                source_geometry_projections=projected,
                source_depth_arrays_compared=compared,
                source_photometric_comparisons=compared,
                projection_seconds=projection_time,
                photometric_seconds=photometry_time,
                target_available=target_available,
                in_frame_pixel_source_pairs=int(counts["inside"].sum(dtype=np.uint64)),
                comparable_pixel_source_pairs=int(counts["comp"].sum(dtype=np.uint64)),
                supported_photometric_pixel_source_pairs=int(moments["photo"].n.sum(dtype=np.uint64)),
            )
            output = _snapshot(
                bank,
                target_id,
                plan,
                plan.ids[:i],
                groups[i],
                missing,
                frame,
                geometry,
                local,
                counts,
                moments,
                witness,
                cost,
            )
            output.cost["elapsed_through_snapshot_seconds"] = perf_counter() - started
            paused = perf_counter()
            yield output
            started += perf_counter() - paused


def audit_subset(subset, full=None):
    """Model-conditioned scope states and actual registered source witnesses, without labels."""
    state = np.zeros(subset.finite.shape, np.uint8)
    witness = np.full(subset.finite.shape, -1, np.int32)
    if full is not None:
        require(
            (full.bank_sha256, full.target_id) == (subset.bank_sha256, subset.target_id)
            and set(subset.queried_source_ids) <= set(full.queried_source_ids),
            "full audit identity differs from selected sources or target",
        )
        if full.actual_k != full.library_size or full.available_count != full.library_size:
            state[:] = 3
        else:
            positive = full.comparable_count > 0
            require(
                not np.any((subset.comparable_count > 0) & ~positive), "full audit lost a selected witness"
            )
            gap = (subset.comparable_count == 0) & positive
            state[:] = np.where(positive, 5, 2)
            state[gap] = 1
            witness[gap] = full.first_witness[gap]
    else:
        state[subset.incomplete] = 3
    state[subset.invalid_geometry] = 4
    witness[state != 1] = -1
    return {
        "state": state,
        "selected_gap": state == 1,
        "gap_witness": witness,
        "full_complete": bool(
            full is not None and full.actual_k == full.available_count == full.library_size
        ),
    }
