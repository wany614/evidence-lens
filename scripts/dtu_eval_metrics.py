"""Real-photo RGB domains and auxiliary diagnostics. No training, exposure fit or threshold selection."""

import numpy as np

from evidence_lens.formal_inputs import require
from vision_evaluate import groups, ranking, compact_curve
from dtu_score import METHODS

TAU_RGB = np.float32(0.05)
RATES = {
    "validity": ("E", "V"),
    "rgb_error_fraction": ("Y", "E"),
    "confirmed_error_fraction_of_V": ("Y", "V"),
    "rgb_mae": ("rgb_mae_sum", "E"),
    "aux3_mae": ("aux3_mae_sum", "E"),
    "aux3_error_fraction": ("Y3", "E"),
    "own_prevalence": ("SY", "S"),
    "common_prevalence": ("CY", "C"),
    "own_scoreable_fraction": ("S", "E"),
    "common_scoreable_fraction": ("C", "E"),
    "common_coverage_V": ("C", "V"),
    "low_coverage": ("L", "V"),
    "conditional_low_coverage": ("L", "S"),
    "low_error": ("LY", "L"),
    "high_error": ("HY", "H"),
    "unknown": ("U", "E"),
    "unknown_error": ("UY", "U"),
    "detection": ("HY", "Y"),
    "missed": ("missed_Y", "Y"),
    "brier": ("brier_sum", "S"),
}


def good_rgb(img):
    return np.isfinite(img).all(2) & (img >= 0).all(2) & (img <= 1).all(2)


def labels(ref, pred, mask):
    require(
        ref.dtype == np.float32 and mask.dtype == bool and ref.ndim == 3 and ref.shape == (mask.shape + (3,)),
        "wrong reference shape",
    )
    v = mask & good_rgb(ref)
    e = np.zeros_like(v) if pred is None else v & good_rgb(pred)
    err = np.full(v.shape, np.nan, np.float32)
    aux = err.copy()
    y = np.full(v.shape, 255, np.uint8)
    if pred is None:
        return v, e, y, err, aux
    require(pred.dtype == np.float32 and pred.shape == ref.shape, "wrong target prediction shape")
    err[e] = np.abs(pred[e] - ref[e]).mean(1, dtype=np.float32)
    y[e] = (err[e] > TAU_RGB).astype(np.uint8)
    aux[e] = err[e]
    # The centre prediction remains fixed. Reference candidates must belong to the same valid domain.
    h, w = v.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            iy, fy = max(0, -dy), min(h, h - dy)
            ix, fx = max(0, -dx), min(w, w - dx)
            left = np.s_[iy:fy, ix:fx]
            right = np.s_[iy + dy : fy + dy, ix + dx : fx + dx]
            use = e[left] & v[right]
            if use.any():
                near = np.abs(pred[left][use] - ref[right][use]).mean(1, dtype=np.float32)
                tile = aux[left]
                tile[use] = np.minimum(tile[use], near)
    require(np.all(aux[e] <= err[e]), "3x3 min error exceeds centre error")
    return v, e, y, err, aux


def ratios(counts):
    return {
        k: {
            "numerator": counts[n],
            "denominator": counts[d],
            "value": counts[n] / counts[d] if counts[d] else None,
            "undefined_reason": None if counts[d] else "empty_" + d,
        }
        for k, (n, d) in RATES.items()
    }


def observation(v, e, y, err, aux, scores, thresholds):
    require(not (e & ~v).any() and np.all((y[e] == 0) | (y[e] == 1)), "RGB supervision domain differs")
    common = e & np.logical_and.reduce([np.isfinite(scores[m]) for m in METHODS])
    pos = y == 1
    n = lambda a: int(np.count_nonzero(a))
    shared = {
        "V": n(v),
        "E": n(e),
        "V_minus_E": n(v & ~e),
        "Y": n(pos & e),
        "C": n(common),
        "CY": n(common & pos),
        "Y3": n(e & (aux > TAU_RGB)),
        "rgb_mae_sum": float(err[e].sum(dtype=np.float64)),
        "aux3_mae_sum": float(aux[e].sum(dtype=np.float64)),
    }
    out = {}
    evidence = {}
    for m in METHODS:
        risk = scores[m]
        s = e & np.isfinite(risk)
        l = s & (risk.astype(np.float64) < thresholds[m])
        h = s & ~l
        u = e & ~s
        values = risk[s].astype(np.float64)
        yy = pos[s]
        bins = np.minimum((values * 10).astype(np.int64), 9)
        calibration = [
            {
                "bin": i,
                "count": n(bins == i),
                "positives": n(yy & (bins == i)),
                "risk_sum": float(values[bins == i].sum()),
            }
            for i in range(10)
        ]
        c = {
            **shared,
            "S": n(s),
            "SY": n(s & pos),
            "L": n(l),
            "LY": n(l & pos),
            "H": n(h),
            "HY": n(h & pos),
            "U": n(u),
            "UY": n(u & pos),
            "missed_Y": n((u | l) & pos),
            "brier_sum": float(((values - yy.astype(np.float64)) ** 2).sum()),
        }
        require(
            c["E"] == c["L"] + c["H"] + c["U"] and c["Y"] == c["LY"] + c["HY"] + c["UY"] and c["C"] <= c["S"],
            "confusion counts do not partition E and Y",
        )
        group = groups(risk[s], yy)
        out[m] = {
            "counts": c,
            "rates": ratios(c),
            "own": ranking(risk[s], yy),
            "common": ranking(risk[common], pos[common]),
            "bin_scores": calibration,
            "risk_coverage": compact_curve([group], c["V"], thresholds[m]),
        }
        evidence[m] = group
    return out, evidence


def aggregate_views(rows, raw_groups, threshold, planned_views=None):
    cs = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
    means = {}
    for group in ("own", "common"):
        vals = [r[group]["ap"] for r in rows if r[group]["ap"] is not None]
        means[group] = {
            "ap": float(np.mean(vals)) if vals else None,
            "defined_views": len(vals),
            "planned_views": len(rows) if planned_views is None else planned_views,
        }
    bins = []
    for i in range(10):
        c = {k: sum(r["bin_scores"][i][k] for r in rows) for k in ("count", "positives", "risk_sum")}
        bins.append({"bin": i, **c})
    ece = sum(abs(x["positives"] - x["risk_sum"]) for x in bins) / cs["S"] if cs["S"] else None
    curve = compact_curve(raw_groups, cs["V"], threshold)
    require(
        (curve["frozen_L"], curve["frozen_LY"]) == (cs["L"], cs["LY"]),
        "operating counts disagree with whole-tie curve",
    )
    return {
        "counts": cs,
        "rates": ratios(cs),
        "AP": means,
        "risk_coverage": curve,
        "bin_scores": bins,
        "ece10": ece,
    }
