"""Frozen B feature definitions and the public linear/rule inference contracts."""

import numpy as np

from evidence_lens_bundle.common import encoded, require, sha

BASELINE_ID = "sha256:6e8e837afa0818cdf8c100b494672b46c10bdbd22e522f388983543f0a5b0f23"
PROFILE = "analysis-378x252-v1"
SIZE = (378, 252)
MIN_ALPHA = 0.01
OPAQUE = 0.5
DEPTH_LIMIT = 0.10
COLOUR_LIMIT = 0.10

DEFINITIONS = {
    "target_alpha": "native accumulated target opacity",
    "target_local_depth_range": "3x3 positive target camera-Z range divided by centre Z, minimum three valid samples",
    "target_local_rgb_mae": "3x3 mean display-RGB channel MAE to centre, minimum three valid samples",
    "log_library_size": "log1p(unique accessible source identities, including declared unavailable members)",
    "log_query_size": "log1p(unique queried source identities, including selected unavailable members)",
    "query_library_fraction": "queried unique IDs / unique accessible IDs",
    "available_query_fraction": "queried IDs with both original-photo-derived RGB and self-render / queried IDs",
    "in_frame_query_fraction": "geometric in-frame count / queried IDs",
    "comparable_query_fraction": "comparable count / queried IDs",
    "comparable_in_frame_fraction": "comparable count / geometric in-frame count",
    "support_comparable_fraction": "depth-supported count / comparable count",
    "depth_conflict_comparable_fraction": "opaque depth-conflict count / comparable count",
    "colour_conflict_support_fraction": "supported sources with target/photo MAE above 0.10 / supported count",
    "source_alpha_mean": "mean source opacity among comparable sources",
    "relative_depth_mean": "mean absolute camera-Z difference / max(Z_projected, Z_source, 1e-6) among comparable sources",
    "relative_depth_max": "maximum relative camera-Z residual among comparable sources",
    "relative_depth_std": "population standard deviation of relative camera-Z residual among comparable sources",
    "angle_mean": "mean target/source observation-ray angle in radians among comparable sources",
    "angle_max": "maximum observation-ray angle in radians among comparable sources",
    "angle_std": "population standard deviation of observation-ray angle among comparable sources",
    "photo_mae_mean": "mean absolute target-to-source photo RGB residual among finite depth-supported colours",
    "photo_mae_min": "minimum target-to-source photo residual among finite depth-supported colours",
    "photo_mae_std": "population standard deviation of target-to-source photo residual among finite depth-supported colours",
    "self_mae_mean": "mean original-photo versus same-checkpoint self-render RGB residual among finite comparable colours",
    "self_mae_max": "maximum original-photo versus same-checkpoint self-render RGB residual among finite comparable colours",
}
NUMERIC_NAMES = tuple(DEFINITIONS)
OPTIONAL = (*NUMERIC_NAMES[:3], *NUMERIC_NAMES[9:])
FEATURE_NAMES = (*NUMERIC_NAMES, *("missing_" + name for name in OPTIONAL))
RULES = ("coverage", "alpha_missing", "residual")
REASONS = (
    "rank_available",
    "depth_unverified",
    "target_render_unavailable",
    "invalid_target_rgb",
    "invalid_target_alpha",
    "invalid_target_z",
    "target_alpha_below_minimum",
    "queried_material_unavailable",
)
STATES = (
    "selected_subset_only",
    "subset_comparability_gap",
    "audited_library_no_comparable_evidence",
    "library_incomplete_or_unavailable",
    "geometry_not_evaluable",
    "audited_comparable_evidence",
)


def feature_schema():
    return {
        "schema": "el-b-many-features-1",
        "baseline_id": BASELINE_ID,
        "analysis_profile": PROFILE,
        "numeric": list(NUMERIC_NAMES),
        "columns": list(FEATURE_NAMES),
        "definitions": DEFINITIONS,
        "missing_indicators_for": list(OPTIONAL),
        "missing_encoding": "NaN numeric value and 1 in corresponding flag; an empty count is zero",
        "accumulator": "float64 sums, second moments and extrema; uint32 counts; FP32 output",
        "projection": "OpenCV integer pixel-centres, floor(u+0.5), fixed scene coordinate frame",
        "comparable": "inside, positive target and source expected camera-Z, valid opacity at or above 0.01 at both pixels, and available source pair",
        "support": "comparable and both opacities >=0.5 and relative camera-Z residual <=0.10",
        "risk_domain": "finite target RGB and evaluable target geometry, with at least one available queried source pair; comparable_count may be zero",
        "rank": "uncalibrated ascending error risk, tau=null",
        "reasons": list(REASONS),
        "audit_states": list(STATES),
        "rules": {
            "coverage": "1-comparable_query_fraction",
            "alpha_missing": "0.5*(1-target_alpha) + 0.5*(1-support_count/query_count)",
            "residual": "mean supported photo MAE, 1.0 when no finite supported colour",
        },
        "normalization_scope": "counts and ratios use the denominators named in each feature; no historical model clipping",
    }


FEATURE_SCHEMA_SHA256 = sha(encoded(feature_schema()))


def score_lr(step, model):
    require(
        model.get("schema") == "el-b-cpu-lr-1"
        and model.get("baseline_id") == BASELINE_ID
        and model.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256
        and model.get("columns") == list(FEATURE_NAMES)
        and "tau" in model
        and model["tau"] is None,
        "different B-CPU-LR model, grid, schema or threshold",
    )
    arrays = [np.asarray(model[k], np.float64) for k in ("impute", "mean", "scale", "coef")]
    require(
        all(a.shape == (len(FEATURE_NAMES),) and np.isfinite(a).all() for a in arrays)
        and np.all(arrays[2] > 0)
        and np.isfinite(model["intercept"]),
        "invalid linear head parameters",
    )
    impute, mean, scale, coef = arrays
    risk = np.full(step.finite.shape, np.nan, np.float32)
    ids = np.flatnonzero(step.finite)
    x = step.X.reshape(-1, len(FEATURE_NAMES))
    for start in range(0, len(ids), 8192):
        group = ids[start : start + 8192]
        raw = x[group].astype(np.float64)
        z = (np.where(np.isfinite(raw), raw, impute) - mean) / scale
        logits = np.einsum("ij,j->i", z, coef, optimize=False) + model["intercept"]
        risk.ravel()[group] = np.exp(-np.logaddexp(0.0, -logits))
    return risk


def simple_ranks(step):
    coverage = 1.0 - step.X[..., NUMERIC_NAMES.index("comparable_query_fraction")]
    opacity = step.X[..., NUMERIC_NAMES.index("target_alpha")]
    missing = 0.5 * (1 - opacity) + 0.5 * (1 - step.support_count.astype(np.float32) / step.actual_k)
    colour = step.X[..., NUMERIC_NAMES.index("photo_mae_mean")]
    return {
        k: np.where(step.finite, val, np.nan).astype(np.float32)
        for k, val in zip(RULES, (coverage, missing, np.where(np.isfinite(colour), colour, 1.0)))
    }
