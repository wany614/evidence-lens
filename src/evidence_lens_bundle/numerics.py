"""Label-blind numerical bridge for frozen and resident score arrays."""

import hashlib

import numpy as np

from .bindings import array_sha
from .common import require
from .scoring import REASON_NAMES, counts

METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
PERCENTS = (1, 2, 5, 10, 20, 40, 60, 80)
CHANNELS = ("risk", "valid", "state", "unknown_reason")


def compare_arrays(old, new, threshold, *, grid_shape=(256, 256)):
    a, b = old["risk"], new["risk"]
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape == grid_shape, "invalid score grid")
    require(all(old[k].shape == new[k].shape == grid_shape for k in CHANNELS[1:]), "invalid support grid")
    paired = old["valid"] & new["valid"]
    a64, b64 = a[paired].astype(np.float64), b[paired].astype(np.float64)
    signed = b64 - a64
    delta = np.abs(signed)
    changes = (a64 < threshold) != (b64 < threshold)
    crossed = np.minimum(np.abs(a64 - threshold), np.abs(b64 - threshold))[changes]
    diffs = {"risk": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))}
    diffs.update({k: int(np.count_nonzero(old[k] != new[k])) for k in CHANNELS[1:]})
    return {
        "bit_exact": not any(diffs.values()),
        "changed_pixels": diffs,
        "old_sha256": {k: array_sha(old[k]) for k in CHANNELS},
        "new_sha256": {k: array_sha(new[k]) for k in CHANNELS},
        "old_counts": counts(old),
        "new_counts": counts(new),
        "common_finite": len(delta),
        "max_abs_difference_common_finite": float(delta.max()) if len(delta) else None,
        "min_signed_new_minus_old": float(signed.min()) if len(signed) else None,
        "max_signed_new_minus_old": float(signed.max()) if len(signed) else None,
        "mean_signed_new_minus_old": float(signed.mean()) if len(signed) else None,
        "mean_abs_difference_common_finite": float(delta.mean()) if len(delta) else None,
        "threshold": threshold,
        "threshold_crossings_common_finite": int(changes.sum()),
        "old_min_abs_distance_to_threshold": float(
            np.min(np.abs(a[old["valid"]].astype(np.float64) - threshold))
        )
        if old["valid"].any()
        else None,
        "new_min_abs_distance_to_threshold": float(
            np.min(np.abs(b[new["valid"]].astype(np.float64) - threshold))
        )
        if new["valid"].any()
        else None,
        "crossing_min_distance_to_threshold": float(crossed.min()) if len(crossed) else None,
        "crossing_max_nearest_distance_to_threshold": float(crossed.max()) if len(crossed) else None,
    }


def matched_sets(old, new, obj, condition, view, l1_threshold, *, grid_shape=(256, 256)):
    """Reproduce the registered PCG64 cutoff-tie ordering without accessing an evaluation mask."""

    def common(group):
        require(
            all(group[m]["valid"].shape == group[m]["risk"].shape == grid_shape for m in METHODS),
            "invalid matched-coverage grid",
        )
        return np.logical_and.reduce([group[m]["valid"] for m in METHODS]).ravel()

    left, right = common(old), common(new)
    n = len(left)
    seed = int.from_bytes(
        hashlib.sha256(f"el-recovery-coverage-v1|{obj}|{condition}|{view}".encode()).digest()[:8], "big"
    )
    rank = np.empty(n, np.int64)
    rank[np.random.Generator(np.random.PCG64(seed)).permutation(n)] = np.arange(n)
    quotas = {f"{q}%_canvas": (2 * q * n + 99) // 200 for q in PERCENTS}
    for name, group, domain in (("old", old, left), ("new", new, right)):
        quotas[name + "_frozen_MS-L1"] = int(
            (group["MS-L1"]["risk"].ravel()[domain].astype(np.float64) < l1_threshold).sum()
        )
    result = {}
    for m in METHODS:

        def ranked(group, domain):
            ids = np.flatnonzero(domain)
            return ids[np.lexsort((rank[ids], group[m]["risk"].ravel()[ids]))]

        a, b = ranked(old, left), ranked(new, right)
        rows = []
        for name, k in quotas.items():
            a_ok, b_ok = k <= len(a), k <= len(b)
            added = removed = None
            if a_ok and b_ok:
                added = int(np.setdiff1d(b[:k], a[:k], assume_unique=True).size)
                removed = int(np.setdiff1d(a[:k], b[:k], assume_unique=True).size)
            rows.append(
                {
                    "anchor": name,
                    "requested_k": k,
                    "old_attainable": a_ok,
                    "new_attainable": b_ok,
                    "added_pixels": added,
                    "removed_pixels": removed,
                }
            )
        result[m] = rows
    return {
        "domain": "four-method finite intersection per implementation on the full native deployment canvas",
        "evaluation_mask_read": False,
        "old_common": int(left.sum()),
        "new_common": int(right.sum()),
        "changed_common_mask": int((left != right).sum()),
        "old_domain_sha256": array_sha(left.reshape(grid_shape)),
        "new_domain_sha256": array_sha(right.reshape(grid_shape)),
        "methods": result,
    }


def checked_arrays(arrays, threshold, *, legacy_reason=None, grid_shape=(256, 256)):
    from evidence_lens.l1 import risk_state

    if legacy_reason is not None:
        require(set(arrays) == {"risk", "valid", "state"}, "wrong historical score roles")
        arrays = {**arrays, "unknown_reason": legacy_reason}
    require(set(arrays) == set(CHANNELS), "unexpected score-array roles")
    for k, dt in zip(CHANNELS, (np.float32, np.bool_, np.uint8, np.uint8)):
        require(arrays[k].dtype == dt and arrays[k].shape == grid_shape, "invalid score channel: " + k)
    v, r, reason = arrays["valid"], arrays["risk"], arrays["unknown_reason"]
    require(
        np.array_equal(v, np.isfinite(r))
        and np.isnan(r[~v]).all()
        and np.all((r[v] >= 0) & (r[v] <= 1))
        and np.array_equal(arrays["state"], risk_state(r, threshold))
        and np.array_equal(reason == 0, v)
        and np.all(reason < len(REASON_NAMES)),
        "historical score/support/state/reason is inconsistent",
    )
    return arrays
