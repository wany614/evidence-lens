"""Five-method held-out metrics. All reductions use declared masks and whole score ties."""

from collections import Counter

import numpy as np

from .evaluation import _ap, _groups, _mask_hash, _regions
from .risk_model import REASONS

METHODS = {
    "constant": "constant",
    "1-alpha": "alpha_score",
    "L0": "l0",
    "L1-full": "l1_full",
    "L1-geometry": "l1_geometry",
}
REGIONS = (
    "all",
    "observed",
    "disoccluded",
    "outside_input_fov",
    "edge",
    "interior",
    "low_contribution",
    "sufficient_contribution",
    "unknown_contribution",
    "background_or_invalid",
)
OUTCOMES = {
    "evaluable_fraction": ("E", "V"),
    "invalid_fraction": ("invalid", "V"),
    "rgb_error_incidence": ("Y", "E"),
    "confirmed_error_fraction_of_V": ("Y", "V"),
    "rgb_mae": ("rgb_error_sum", "E"),
    "sensitivity_025": ("Y025", "E"),
    "sensitivity_100": ("Y100", "E"),
    "common_coverage": ("S", "V"),
    "common_prevalence": ("SY", "S"),
    "common_excluded_error_incidence": ("excluded_Y", "excluded_E"),
    "depth_evaluable_surface_fraction": ("Z", "surface"),
    "depth_mae_m": ("z_error_sum", "Z"),
    "depth_high_fraction": ("ZY", "Z"),
    "low_support_surface_fraction": ("low_support", "surface"),
}
SYSTEM = {
    "detection": ("HY", "Y"),
    "warning_precision": ("HY", "H"),
    "warning_coverage": ("H", "V"),
    "low_risk_error": ("LY", "L"),
    "low_coverage": ("L", "V"),
    "unknown_fraction": ("U", "E"),
    "unknown_error_incidence": ("UY", "U"),
    "scoreable_coverage": ("A", "V"),
}
QUANTILES = (0, 0.10, 0.25, 0.50, 0.75, 0.90, 1)


def ratio(numerator, denominator, missing):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "missing_reason": None if denominator else missing,
    }


def rates(counts, definitions):
    return {name: ratio(counts[n], counts[d], "empty_" + d) for name, (n, d) in definitions.items()}


def distribution(values):
    arr = np.asarray(values)
    if not np.isfinite(arr).all():
        raise ValueError("nonfinite evaluable depth error")
    return {
        "count": int(arr.size),
        "probabilities": list(QUANTILES),
        "values": np.quantile(arr, QUANTILES).tolist() if arr.size else None,
        "missing_reason": None if arr.size else "no_evaluable_depth",
    }


def count(mask):
    return int(np.count_nonzero(mask))


def rank(scores, y, region):
    g = _groups(scores[region], y[region])
    n, p = count(region), count(y & region)
    status = (
        "two_class"
        if 0 < p < n
        else ("empty_mask" if n == 0 else "all_positive" if n == p else "all_negative")
    )
    return {
        "ap": _ap(g[1], g[2]),
        "status": status,
        "n": n,
        "positive": p,
        "negative": n - p,
        "unique_score_count": len(g[0]),
    }, g


def ranking_ap(record):
    if record["status"] != "two_class":
        if record["status"] not in ("empty_mask", "all_positive", "all_negative"):
            raise ValueError("undeclared ranking status")
        return None
    ap = record["ap"]
    if not isinstance(ap, (int, float)) or isinstance(ap, bool) or not np.isfinite(ap) or not 0 <= ap <= 1:
        raise ValueError("two-class view must have finite AP in [0,1]")
    return ap


def curve_arrays(groups, v_count, threshold):
    score, sizes, errors = groups
    accepted = np.r_[np.int64(0), np.cumsum(sizes, dtype=np.int64)]
    accepted_errors = np.r_[np.int64(0), np.cumsum(errors, dtype=np.int64)]
    # At row zero no score group has been accepted. 'score' otherwise gives the highest included tie.
    return {
        "score": np.r_[np.float32(np.nan), score],
        "accepted": accepted,
        "errors": accepted_errors,
        "V": np.int64(v_count),
        "frozen_tau": np.float64(threshold),
        "frozen_index": np.int64(np.searchsorted(score.astype(np.float64), threshold, side="left")),
    }


def pooled_groups(groups):
    scores, sizes, errors = (np.concatenate([g[i] for g in groups]) for i in range(3))
    if not len(scores):
        return scores, sizes, errors
    order = np.argsort(scores, kind="stable")
    scores = scores[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    return scores[starts], np.add.reduceat(sizes[order], starts), np.add.reduceat(errors[order], starts)


def _unknown_reasons(method, labels, reasons, unsupported):
    e = labels["rgb_evaluable"]
    if method == "L0":
        return [(r, reasons == i) for i, r in REASONS.items()]
    if method == "1-alpha":
        return [("invalid_target_alpha", ~labels["alpha_valid"])]
    if method == "constant":
        return []
    if unsupported:
        return [("unsupported_l1", np.ones(e.shape, bool))]
    return [
        ("invalid_target_alpha", ~labels["alpha_valid"]),
        ("invalid_target_depth", labels["invalid_depth"]),
    ]


def explanatory_counts(unknown, y, reasons):
    remaining = unknown.copy()
    result = {}
    for reason, indicated in reasons:
        mask = remaining & indicated
        if mask.any():
            result[reason] = {"U": count(mask), "UY": count(mask & y)}
            remaining[mask] = False
    if remaining.any():
        result["unavailable_score"] = {"U": count(remaining), "UY": count(remaining & y)}
    return result


def view_statistics(labels, scores, reasons, thresholds, unsupported=False):
    """Return system numerators, paired AP, tie groups and actual operating states for one target."""
    if set(scores) != set(METHODS) or set(thresholds) != set(METHODS):
        raise ValueError("five distinct methods and thresholds required")
    v = labels["reference_valid"]
    e = labels["rgb_evaluable"]
    y = labels["rgb_high_error"] == 1
    if v.dtype != bool or e.dtype != bool or np.any(e & ~v):
        raise ValueError("invalid E/V")
    error = labels["rgb_error"]
    if (
        not np.array_equal(np.isfinite(error), e)
        or np.any(error[e] < 0)
        or not np.array_equal(labels["rgb_high_error"][e], (error[e] > np.float32(0.05)).astype(np.uint8))
        or np.any(labels["rgb_high_error"][~e] != 255)
    ):
        raise ValueError("continuous errors, Y and E disagree")
    z_eval, surface = labels["depth_evaluable"], labels["reference_surface"]
    if z_eval.dtype != bool or surface.dtype != bool or np.any(z_eval & ~surface):
        raise ValueError("depth eligibility requires a reference surface")
    for absolute, signed in (
        ("depth_error", "signed_depth_error"),
        ("relative_depth_error", "signed_relative_depth_error"),
    ):
        a, b = labels[absolute], labels[signed]
        if (
            not np.array_equal(np.isfinite(a), z_eval)
            or not np.array_equal(np.isfinite(b), z_eval)
            or not np.array_equal(a[z_eval], np.abs(b[z_eval]))
        ):
            raise ValueError("evaluable depth requires finite signed and matching absolute errors")
    available, low, common = {}, {}, e.copy()
    for method, score in scores.items():
        if score.shape != v.shape or score.dtype != np.float32 or np.isinf(score).any():
            raise ValueError(f"invalid {method} scores")
        valid = np.isfinite(score)
        if np.any((score[valid] < 0) | (score[valid] > 1)):
            raise ValueError(f"out-of-range {method} score")
        available[method] = e & valid
        low[method] = available[method] & (score.astype(np.float64) < thresholds[method])
        common &= valid
    result, all_groups = {}, {}
    for region, mask in _regions(labels):
        region = region.removeprefix("prediction_")
        rv, re, rs = v & mask, e & mask, common & mask
        surface, z = mask & labels["reference_surface"], mask & labels["depth_evaluable"]
        vc, ec, yc, sc, sy = count(rv), count(re), count(y & re), count(rs), count(y & rs)
        base = {
            "V": vc,
            "E": ec,
            "Y": yc,
            "invalid": vc - ec,
            "S": sc,
            "SY": sy,
            "excluded_E": ec - sc,
            "excluded_Y": yc - sy,
            "Y025": count(re & (error > np.float32(0.025))),
            "Y100": count(re & (error > np.float32(0.10))),
            "rgb_error_sum": float(error[re].sum(dtype=np.float64)),
            "surface": count(surface),
            "low_support": count(surface & labels["low_support"]),
            "Z": count(z),
            "ZY": count(z & (labels["depth_high_error"] == 1)),
            "z_error_sum": float(labels["depth_error"][z].sum(dtype=np.float64)),
        }
        methods = {}
        for method, score in scores.items():
            a, l = available[method] & mask, low[method] & mask
            h, u = a & ~l, re & ~a
            nc = {
                "A": count(a),
                "L": count(l),
                "H": count(h),
                "U": count(u),
                "HY": count(h & y),
                "LY": count(l & y),
                "UY": count(u & y),
            }
            nc["false_warnings"] = nc["H"] - nc["HY"]
            nc["missed_errors"] = yc - nc["HY"]
            if nc["L"] + nc["H"] + nc["U"] != ec or nc["HY"] + nc["LY"] + nc["UY"] != yc:
                raise ValueError("risk state counts violate E/Y partitions")
            own, groups = rank(score, y, a)
            shared = own.copy() if np.array_equal(a, rs) else rank(score, y, rs)[0]
            methods[method] = {
                "counts": nc,
                "system": rates(base | nc, SYSTEM),
                "common_ranking": shared,
                "own_ranking": own,
                "own_mask_sha256": _mask_hash(a),
                "unknown_reasons": explanatory_counts(
                    u, y, _unknown_reasons(method, labels, reasons, unsupported)
                ),
            }
            if region == "all":
                all_groups[method] = groups
        result[region] = {
            "counts": base,
            "outcomes": rates(base, OUTCOMES),
            "methods": methods,
            "S_mask_sha256": _mask_hash(rs),
            "V_mask_sha256": _mask_hash(rv),
            "E_mask_sha256": _mask_hash(re),
        }
    if set(result) != set(REGIONS):
        raise ValueError("missing frozen region")
    return result, all_groups


def _sum_counts(records):
    result = Counter()
    for row in records:
        result.update(row)
    return dict(result)


def pool_family(view_rows):
    """Pool system counts within family, average only two-class per-view AP."""
    output = {}
    for region in REGIONS:
        rows = {v: record["regions"][region] for v, record in view_rows.items()}
        base = _sum_counts(r["counts"] for r in rows.values())
        methods = {}
        for method in METHODS:
            samples = {v: r["methods"][method] for v, r in rows.items()}
            nc = _sum_counts(r["counts"] for r in samples.values())
            unknown = {}
            for r in samples.values():
                for reason, c in r["unknown_reasons"].items():
                    unknown[reason] = _sum_counts([unknown.get(reason, {}), c])
            rec = {"counts": nc, "system": rates(base | nc, SYSTEM), "unknown_reasons": unknown}
            for field in ("common_ranking", "own_ranking"):
                eligible = {v: ap for v, x in samples.items() if (ap := ranking_ap(x[field])) is not None}
                omitted = {v: x[field]["status"] for v, x in samples.items() if v not in eligible}
                rec[field] = {
                    "ap": float(np.mean(list(eligible.values()))) if eligible else None,
                    "eligible_view_count": len(eligible),
                    "eligible_views": list(eligible),
                    "excluded_views": omitted,
                    "missing_reason": None if eligible else "no_two_class_views",
                }
            methods[method] = rec
        if len({tuple(r["common_ranking"]["eligible_views"]) for r in methods.values()}) != 1:
            raise ValueError("methods used different common-mask views")
        output[region] = {"counts": base, "outcomes": rates(base, OUTCOMES), "methods": methods}
    return output
