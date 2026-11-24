"""Apply an explicitly pinned operating point to the matching bundle alongside its original states."""

from pathlib import Path

import numpy as np

from .common import RELEASE_INDEX_SHA256, digest, exact_keys, json_bytes, profile, require, sha

SCOPE_FIELDS = frozenset(
    (
        "layout",
        "primary_source_id",
        "lighting_id",
        "resolution",
        "rgb_encoding",
        "depth_semantics",
        "alpha_semantics",
        "recipe_sha256",
        "stop_step",
        "renderer",
        "reconstructor",
        "release_index_sha256",
        "model_sha256",
        "model_index_sha256",
        "feature_schema_sha256",
        "preprocessing_sha256",
        "feature_columns",
        "source_ids",
        "source_camera_sha256",
        "render_camera_sha256",
    )
)
EMPIRICAL = ("empirical_calibration_only", "evaluated_on_holdout")


def pinned(path, identity, role):
    raw = Path(path).read_bytes()
    require(sha(raw) == digest(identity, role), role + " SHA-256 differs")
    return json_bytes(raw)


def declarations(bundle):
    return {
        "bundle_sha256": bundle.sha256,
        "original_case_sha256": bundle.meta["creation"]["original_case_sha256"],
        "checkpoint_sha256": bundle.producer["checkpoint_sha256"],
        "bundle_camera_sha256": bundle.meta["camera"]["sha256"],
        "source_rgb_sha256": {s: bundle.meta["sources"][s]["sha256"] for s in bundle.order},
        "render_sha256": {
            v: (item["file"]["sha256"] if item["status"] == "ready" else None)
            for v, item in bundle.producer["frames"].items()
        },
    }


def select(bundle, method, profile_path, profile_sha, scope_path, scope_sha):
    require(
        (profile_path is None) == (profile_sha is None),
        "supply both calibration profile and its independent SHA-256",
    )
    require(
        (scope_path is None) == (scope_sha is None),
        "supply both deployment scope proof and its independent SHA-256",
    )
    if profile_path is None:
        require(scope_path is None, "deployment scope proof needs a calibration profile")
        return None
    p = pinned(profile_path, profile_sha, "calibration profile")
    require(
        p.get("schema") == "el-b2-calibration-profile-1"
        and p.get("profile_version") == 1
        and p.get("method") == "MS-L1",
        "unsupported calibration profile",
    )
    condition = p.get("condition")
    status = p.get("status")
    tau = p.get("tau")
    s = p.get("scope")
    require(
        condition in ("M2", "M4") and status in (*EMPIRICAL, "no_feasible_operating_point"),
        "unsupported calibration outcome",
    )
    exact_keys(s, SCOPE_FIELDS, "calibration scope")
    require(
        p.get("comparison") == "score < tau"
        and p.get("epsilon") == 0.05
        and p.get("original_tau") == profile()["models"]["MS-L1"]["threshold"],
        "calibration label or original threshold differs",
    )
    require(
        p.get("target")
        == {
            "maximum_mean_low_error": 0.05,
            "minimum_mean_coverage": 0.20,
            "require_every_object_nonempty": True,
        },
        "unexpected calibration objective",
    )
    holdout = p.get("evaluated_on_holdout")
    hit = p.get("holdout_target_met")
    require(
        type(holdout) is bool
        and (
            (holdout and type(hit) is bool and status == "evaluated_on_holdout")
            or (not holdout and hit is None and status != "evaluated_on_holdout")
        ),
        "inconsistent holdout claim",
    )
    if status in EMPIRICAL:
        require(
            type(tau) in (int, float) and np.isfinite(tau) and 0 <= tau <= np.nextafter(1.0, np.inf),
            "invalid selected cutoff",
        )
    else:
        require(tau is None, "infeasible profile cannot supply a calibrated threshold")
    source_ids = ["s00", "s03"] if condition == "M2" else ["s00", "s01", "s02", "s03"]
    orig = profile()["models"]["MS-L1"]
    fixed = {
        "layout": "DTU-fixed",
        "primary_source_id": "s00",
        "lighting_id": 3,
        "resolution": [256, 256],
        "rgb_encoding": "display_srgb_0_1",
        "depth_semantics": "expected_camera_z_m",
        "alpha_semantics": "accumulated_opacity",
        "renderer": {"name": "gsplat", "version": "1.5.3"},
        "release_index_sha256": RELEASE_INDEX_SHA256,
        "model_sha256": orig["weights_sha256"],
        "model_index_sha256": orig["model_index_sha256"],
        "feature_schema_sha256": orig["feature_schema_sha256"],
        "preprocessing_sha256": orig["preprocessing_sha256"],
        "feature_columns": 39,
        "source_ids": source_ids,
    }
    require(
        all(s[k] == v for k, v in fixed.items()) and type(s["stop_step"]) is int and s["stop_step"] > 0,
        "profile is outside the supported fixed-release schema",
    )
    recon = s["reconstructor"]
    exact_keys(recon, ("name", "gaussian_cap", "optimizer_seed", "worker_tree_sha256"), "reconstructor")
    require(
        isinstance(recon["name"], str)
        and recon["name"]
        and type(recon["optimizer_seed"]) is int
        and type(recon["gaussian_cap"]) is int
        and recon["gaussian_cap"] > 0,
        "invalid declared reconstructor",
    )
    digest(recon["worker_tree_sha256"], "reconstructor code")
    for k in ("recipe_sha256", "source_camera_sha256", "render_camera_sha256"):
        digest(s[k], k)
    mismatches = []
    if getattr(bundle, "versioned_grid", False):
        mismatches.append("registered calibration requires the original 256 x 256 analysis and preprocessing")
    if method != p["method"]:
        mismatches.append("scoring method")
    if bundle.condition != condition or list(bundle.order) != source_ids or bundle.primary != "s00":
        mismatches.append("condition or source layout")
    if bundle.meta["creation"]["original_case_sha256"] is None:
        mismatches.append("external reconstruction asset")
    if scope_path is None:
        mismatches.append("independently pinned fixed-layout deployment evidence missing")
    else:
        proof = pinned(scope_path, scope_sha, "deployment scope")
        require(
            proof.get("schema") == "el-b2-deployment-scope-1" and proof.get("verified") is True,
            "invalid deployment scope proof",
        )
        match = declarations(bundle)
        require(
            proof.get("bundle") == match, "deployment scope proof belongs to a different bundle or render"
        )
        if proof.get("condition") != condition:
            mismatches.append("scope condition")
        actual = proof.get("scope")
        exact_keys(actual, SCOPE_FIELDS, "deployment scope")
        mismatches.extend(k for k in sorted(SCOPE_FIELDS) if actual[k] != s[k])
        origins = proof.get("origins")
        require(
            isinstance(origins, dict) and set(origins) >= {"source", "cohort", "freeze", "readback", "case"},
            "deployment scope lacks its original evidence",
        )
        for k, v in origins.items():
            require(isinstance(v, dict) and v.get("run_id") and v.get("member"), "missing origin for " + k)
            digest(v.get("manifest_sha256"), k + " manifest")
            digest(v.get("sha256"), k + " member")
    return {
        "status": "outside_calibration_scope" if mismatches else status,
        "profile_status": status,
        "applied": not mismatches and status in EMPIRICAL,
        "profile_sha256": profile_sha,
        "scope_sha256": scope_sha,
        "mismatches": mismatches,
        "condition": condition,
        "layout": s["layout"],
        "threshold": tau,
        "original_threshold": p["original_tau"],
        "comparison": "finite FP32 score cast to float64 < tau",
        "calibration_target": p["target"],
        "evaluated_on_holdout": holdout,
        "holdout_target_met": hit,
    }


def states(risk, tau):
    finite = np.isfinite(risk)
    out = np.full(risk.shape, 255, np.uint8)
    out[finite] = (risk[finite].astype(np.float64) >= tau).astype(np.uint8)
    return out


def state_counts(state):
    return {
        "canvas": int(state.size),
        "low": int(np.sum(state == 0)),
        "high": int(np.sum(state == 1)),
        "unknown": int(np.sum(state == 255)),
    }
